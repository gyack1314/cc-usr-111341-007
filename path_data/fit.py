"""稳健联合估计：延迟相加模型 + 超时单侧区间 + 非负稀疏。

延迟模型（相对健康基线，单位 ms）：
    y_k = sum_g F[k,g] * beta_g + u_source(k) + e_k

F[k,g] 是组 g 代表列在该样本完整有向路径上的实际经过次数（环路按次数累计）。
beta_g 是组 g 的增量：sum_{c in g} r_c * delta_c（r_c 为组权重；列完全共线的
组内 r_c=1，beta 即组内增量之和）。u 是相对健康源偏移的小幅漂移。

超时模型：
    success=false 的样本只知道 y_k > T_k。
    目标对成功样本用平方损失，对超时样本用单侧 hinge：
        max(T_k - (baseline_k + increment_k), 0)^2
    拟合低于下界才受罚；高于下界梯度为零。超时样本从不删除，也从不把 T_k
    当作精确观测。由超时单独支持出的增量只报下界。

稳健：成功残差用 Tukey 双权迭代降权，尺度用 MAD。

稀疏：组系数与源漂移均施加 L1；组系数非负（只建模“变慢”方向），源漂移
允许正负。源漂移也稀疏，使“所有源共同偏移”优先被共享组件解释，而单源
异常既可能属于该源独有的组件，也可能属于源漂移——这一不可区分性在报告中
单列，不强行二选一。

识别约定：故障窗的源漂移被中心化（各源相对均值为零），绝对水平由组件
承担；健康窗里源偏移与路径基线不可分时，由 baseline 模块显式声明。
"""
import math

from .linalg import pseudoinverse_diagonal, solve

DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def median(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def robust_scale(values):
    vals = sorted(abs(v) for v in values)
    if not vals:
        return None
    med = vals[len(vals) // 2]
    return max(1.4826 * med, 1e-9)


def tukey_weights(residuals, c=4.685, scale=None):
    s = scale or robust_scale(residuals)
    if not s:
        return [1.0] * len(residuals)
    return [(1.0 - (r / (c * s)) ** 2) ** 2 if abs(r / (c * s)) < 1.0 else 0.0
            for r in residuals]


def build_features(rows, grouping):
    groups = grouping['groups']
    sources = sorted({row['source'] for row in rows})
    labels = [{'kind': 'group', 'components': list(g['components']),
               'reference': g['reference'], 'weights': dict(g['weights'])}
              for g in groups]
    labels += [{'kind': 'source', 'source': s} for s in sources]
    feature = []
    for row in rows:
        vec = [float(row['counts'].get(g['reference'], 0.0)) for g in groups]
        vec += [1.0 if row['source'] == s else 0.0 for s in sources]
        feature.append(vec)
    return feature, labels, groups, sources


def assemble(design, targets, weights):
    n = len(design[0]) if design else 0
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, y, w in zip(design, targets, weights):
        if w <= 0:
            continue
        for i in range(n):
            if row[i] == 0:
                continue
            atb[i] += w * row[i] * y
            for j in range(n):
                ata[i][j] += w * row[i] * row[j]
    return ata, atb


def coordinate_descent(ata, atb, theta, penalties, n_group, component_upper=None,
                      max_sweeps=500, tol=1e-12):
    """循环坐标下降解带 L1 的加权二次问题。

    目标：0.5 x'Ax - b'x + Σ pen_i |x_i|，前 n_group 个分量非负且 <= 上界。
    坐标闭式解为软阈值。先用最大对角元逐步更新会因变量尺度差异收敛极慢，
    这里每轮按对角元归一化（等价于对每列做尺度变换），并对最后一步做精确
    的最小二乘精修，确保线性可识别方向上的解达到数值最优。
    """
    n = len(theta)
    theta = list(theta)
    if n == 0:
        return theta
    for _ in range(max_sweeps):
        change = 0.0
        for i in range(n):
            diag = ata[i][i]
            if diag <= 0:
                continue
            partial = atb[i] - sum(ata[i][j] * theta[j] for j in range(n) if j != i)
            v = partial / diag
            t = penalties[i] / diag
            if i < n_group:
                v = max(v - t, 0.0)
                if component_upper is not None:
                    v = min(v, max(component_upper, 0.0))
            else:
                v = math.copysign(max(abs(v) - t, 0.0), v) if v else 0.0
            change = max(change, abs(v - theta[i]))
            theta[i] = v
        if change < tol * max(1.0, max((abs(v) for v in theta), default=0.0)):
            break
    return _refine_support(ata, atb, theta, penalties, n_group, component_upper)


def _refine_support(ata, atb, theta, penalties, n_group, component_upper):
    """对当前活跃集（非零且未被 L1 截断）做一次无惩罚最小二乘精修。"""
    active = [i for i, v in enumerate(theta) if abs(v) > 1e-9]
    if not active:
        return theta
    n = len(theta)
    sub = [[ata[i][j] for j in active] for i in active]
    rhs = [atb[i] - sum(ata[i][j] * theta[j] for j in range(n)
                        if j not in active) for i in active]
    coef = solve(sub, rhs)
    if coef is None:
        return theta
    out = list(theta)
    for pos, i in enumerate(active):
        v = coef[pos]
        if i < n_group and v < 0:
            v = 0.0
        if i < n_group and component_upper is not None:
            v = min(v, max(component_upper, 0.0))
        out[i] = v
    return out


def fit_target(rows, components, baseline_predict, grouping, lam=1.0,
               max_iter=60, tol=1e-9, robust=True, use_source_drift=True,
               _skip_timeout_probe=False):
    """在目标窗样本上拟合组增量、整体水平与源漂移。

    参数化：
        r_k = F[k]·beta + g + u_source(k) + e_k
    其中 beta 为组增量（非负、L1），g 为所有样本共享的整体水平，
    u 为各源相对整体水平的漂移（L1），且强制 sum(u)=0 以消除平移冗余。

    为什么显式保留 g：单源异常与“所有组件一起变慢”在观测上完全等价。
    把整体水平单列为 g，可以使单源异常落到 u 上而不是被摊进所有组件；
    而 g 本身的可解释性（共同源偏移 vs 所有组件共同上调）作为不可区分
    解释在报告中单列，不冒充组件定位结果。
    """
    def base_value(row):
        info = baseline_predict(row)
        return None if info is None else info['value']
    usable = [row for row in rows if base_value(row) is not None]
    missing = [{'probe': row['probe'], 'path': row['path'], 'version': row['version'],
                'reason': baseline_predict(row)['reason']}
               for row in rows if base_value(row) is None]
    if not usable:
        return {'ok': False, 'reason': '目标窗没有可用基线的样本',
                'missing_baseline': missing}
    sources = sorted({row['source'] for row in usable})
    fit_sources = sources if (use_source_drift and len(sources) >= 2) else []
    feature, labels, groups, _ = build_features(
        [dict(row, source='<single>') for row in usable] if not fit_sources else usable,
        grouping)
    n_group = len(groups)
    if not fit_sources:
        # 去掉源列，仅保留组特征
        feature = [row[:n_group] for row in feature]
        labels = labels[:n_group]
    # 布局：[beta_1..beta_G, g, u_1..u_S]
    n = n_group + 1 + len(fit_sources)
    for i in range(len(feature)):
        feature[i] = feature[i][:n_group] + [1.0] + [
            1.0 if (fit_sources and usable[i]['source'] == s) else 0.0
            for s in fit_sources]
    layout = {
        'groups': list(range(n_group)),
        'global': n_group,
        'sources': list(range(n_group + 1, n)),
    }
    observed_increments = []
    for row in usable:
        base = base_value(row)
        observed_increments.append(
            max((row['latency'] if row['success'] else row['timeout']) - base, 0.0))
    component_upper = 2.0 * max(observed_increments or [1.0])
    theta = [0.0] * n
    success_weight = [1.0] * len(usable)
    sigma = None
    for _ in range(max_iter):
        targets, weights = [], []
        for i, row in enumerate(usable):
            base = base_value(row)
            pred = sum(feature[i][j] * theta[j] for j in range(n))
            if row['success']:
                targets.append(row['latency'] - base)
                weights.append(success_weight[i])
            else:
                floor = row['timeout'] - base
                if pred < floor:
                    targets.append(floor)
                    weights.append(1.0)
                else:
                    targets.append(0.0)
                    weights.append(0.0)
        ata, atb = assemble(feature, targets, weights)
        ridge = max([ata[i][i] for i in range(n)] or [1.0]) * 1e-9
        for i in range(n):
            ata[i][i] += ridge
        # beta 与 u 受 L1，g 不受惩罚（它是不可区分的整体水平，惩罚它会
        # 把水平错误地挤到别处）。
        pen = [lam] * n_group + [0.0] + [lam] * len(fit_sources)
        theta_new = coordinate_descent(ata, atb, theta, pen, n_group, component_upper)
        # 平移不变性：强制 sum(u)=0，剩余整体水平归入 g。
        if fit_sources:
            u_vals = theta_new[n_group + 1:]
            mean_u = sum(u_vals) / len(u_vals)
            theta_new[n_group] += mean_u
            theta_new[n_group + 1:] = [v - mean_u for v in u_vals]
        change = max((abs(theta_new[i] - theta[i]) for i in range(n)), default=0.0)
        theta = theta_new
        residuals = []
        for i, row in enumerate(usable):
            if row['success']:
                pred = sum(feature[i][j] * theta[j] for j in range(n))
                residuals.append(row['latency'] - base_value(row) - pred)
        sigma = robust_scale(residuals)
        if robust and len(residuals) >= 4 and sigma:
            # 只在残差分布足够“干净”时启用双权：若多数样本系统性变慢，
            # MAD 尺度会包含信号本身，双权会把真实信号当离群降权。
            median_resid = median(residuals)
            if abs(median_resid) > 0.5 * sigma:
                success_weight = [1.0] * len(usable)
            else:
                w = tukey_weights(residuals, scale=sigma)
                it = iter(w)
                for i, row in enumerate(usable):
                    success_weight[i] = next(it) if row['success'] else 1.0
        if change < tol * max(1.0, max((abs(v) for v in theta), default=0.0)):
            break

    fitted, residuals = {}, {}
    for i, row in enumerate(usable):
        increment = sum(feature[i][j] * theta[j] for j in range(n))
        base = base_value(row)
        fitted[row['probe']] = base + increment
        residuals[row['probe']] = (row['latency'] - base - increment) if row['success'] else None

    # 识别“只由超时下界驱动”的组：把目标窗的成功样本单独再拟合一次。
    # 若某组在完整拟合中显著、但在成功样本拟合中归零，它就没有点估计，
    # 只有下界；不能进入候选组件集合。
    success_only_beta = {}
    if any(not row['success'] for row in usable):
        success_rows = [row for row in usable if row['success']]
        if success_rows:
            refit = fit_target(success_rows, components, baseline_predict,
                               grouping, lam=lam, max_iter=25, robust=robust,
                               use_source_drift=use_source_drift,
                               _skip_timeout_probe=True)
            if refit.get('ok'):
                success_only_beta = refit.get('group_beta', {})
    success_res = [r for r in residuals.values() if r is not None]
    sigma_est = robust_scale(success_res) if success_res else None
    se = {}
    unidentifiable = []
    if sigma_est and n:
        ata, _ = assemble(
            feature,
            [row['latency'] - base_value(row) if row['success'] else 0.0
             for row in usable],
            [success_weight[i] if row['success'] else 0.0 for i in range(len(usable))])
        # 用伪逆对角线：不可识别的方向（零空间）对应的系数标准误无界，
        # 报告为 None，而不是用岭项造出一个虚假的小标准误。
        inv_diag, null_basis = pseudoinverse_diagonal(ata, tol=1e-9)
        for i in range(n):
            if i in null_basis or not math.isfinite(inv_diag[i]):
                se[('group', i)] = None
            else:
                se[('group', i)] = math.sqrt(max(sigma_est ** 2 * inv_diag[i], 0.0))
        for vec in null_basis:
            idx = [i for i, v in enumerate(vec) if abs(v) > 1e-6]
            unidentifiable.append({
                'components_or_terms': _term_names(idx, groups, n_group, fit_sources),
                'direction': {_term_name(i, groups, n_group, fit_sources): vec[i]
                              for i in idx},
            })
    else:
        for i in range(n):
            se[('group', i)] = None

    def group_se(idx):
        return se.get(('group', idx))

    delta = {}
    group_beta = {}
    for gi, group in enumerate(groups):
        beta = theta[gi]
        group_beta['|'.join(group['components'])] = beta
        members = group['components']
        wsum = sum(group['weights'][c] for c in members) or 1.0
        for c in members:
            # 组内等权分摊只是**约定**；组内各组件不可区分时，不能把
            # beta/|group| 当作每个组件的物理增量。individual 标志在渲染时
            # 依据组大小给出。
            delta[c] = beta * group['weights'][c] / wsum
    source_drift = {s: theta[n_group + 1 + j] for j, s in enumerate(fit_sources)}
    return {
        'ok': True,
        'lambda': lam,
        'labels': labels,
        'layout': layout,
        'group_beta': group_beta,
        'groups': [{'components': g['components'], 'beta': theta[gi],
                    'se': group_se(gi)} for gi, g in enumerate(groups)],
        'global_level': theta[n_group],
        'global_level_se': group_se(n_group),
        'sources': source_drift,
        'source_se': {s: group_se(n_group + 1 + j)
                      for j, s in enumerate(fit_sources)},
        'source_drift_fitted': bool(fit_sources),
        'delta_convention': delta,
        'fitted': fitted,
        'residuals': residuals,
        'sigma': sigma_est,
        'unidentifiable_directions': unidentifiable,
        'success_only_group_beta': success_only_beta,
        'usable_rows': len(usable),
        'missing_baseline': missing,
        'success_weight': {usable[i]['probe']: success_weight[i]
                           for i in range(len(usable))},
    }


def _term_name(i, groups, n_group, fit_sources):
    if i < n_group:
        return '|'.join(groups[i]['components'])
    if i == n_group:
        return '<global>'
    return 'source:' + fit_sources[i - n_group - 1]


def _term_names(idx, groups, n_group, fit_sources):
    return [_term_name(i, groups, n_group, fit_sources) for i in idx]


def active_groups(fit):
    out = []
    for g in fit.get('groups', []):
        se = g['se']
        threshold = max(2.0 * se, 1e-6) if se is not None else 1e-6
        if g['beta'] > threshold:
            out.append(g)
    return out


def active_sources(fit):
    out = {}
    for s, v in fit.get('sources', {}).items():
        se = fit.get('source_se', {}).get(s)
        threshold = max(2.0 * se, 1e-6) if se is not None else 1e-6
        if abs(v) > threshold:
            out[s] = v
    return out


def summary(fit, rows):
    residuals = [fit['residuals'].get(r['probe']) for r in rows
                 if fit['residuals'].get(r['probe']) is not None]
    violations = 0
    for row in rows:
        if row['success'] or row['probe'] not in fit.get('fitted', {}):
            continue
        tolerance = 1e-6 * max(1.0, row['timeout'])
        if fit['fitted'][row['probe']] < row['timeout'] - tolerance:
            violations += 1
    residuals_sorted = sorted(abs(r) for r in residuals)
    return {
        'lambda': fit['lambda'],
        'sigma': fit.get('sigma'),
        'active_groups': [g['components'] for g in active_groups(fit)],
        'active_sources': active_sources(fit),
        'timeout_lower_bound_violations': violations,
        'n_success_used': len(residuals),
        'sse_success': sum(r * r for r in residuals),
        'median_abs_residual': residuals_sorted[len(residuals_sorted) // 2]
        if residuals_sorted else None,
        'source_drift_fitted': fit.get('source_drift_fitted', False),
        'lower_bound_only_groups': lower_bound_only_groups(fit, rows),
    }


def lower_bound_only_groups(fit, rows):
    """某组在“只用成功样本”的拟合中归零、只在含超时样本时显著：
    它就是只由下界驱动的组，估计值只是下界。"""
    groups = fit.get('groups', [])
    success_beta = fit.get('success_only_group_beta') or {}
    out = []
    for g in groups:
        if g['beta'] <= 1e-6:
            continue
        key = '|'.join(g['components'])
        if key not in success_beta:
            continue  # 没有超时样本，或成功重拟合不可用
        if abs(success_beta[key]) > max(1.0, 0.1 * g['beta']):
            continue  # 成功样本本身支持该组，不是纯下界
        members = set(g['components'])
        hits = [row for row in rows
                if members & set(row['counts']) and row['probe'] in fit['fitted']]
        timeouts = [row['probe'] for row in hits if not row['success']]
        if not timeouts:
            continue
        out.append({
            'components': g['components'],
            'point_estimate_lower_bound': g['beta'],
            'success_only_estimate': success_beta.get(key),
            'timeout_probes': timeouts,
            'note': '该组只在含超时样本的拟合中显著，成功样本拟合下归零；'
                    '估计值是下界，不是点估计。',
        })
    return out
