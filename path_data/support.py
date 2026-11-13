"""候选支撑枚举与等价解释。

在主模型之外做小支撑交叉检查，**参数化与主模型一致**：
    r_k = F_S[k]·beta_S + g + u_source(k) + e_k
- S 是被允许的组件组支撑；beta_S 非负、受 L1 与分量上界；
- g（整体水平）与源漂移 u 是每个支撑都带的干扰项，不作稀疏惩罚，因为
  “共同偏移”是必须在每种支撑下都被公平允许的备选解释；
- 超时样本用单侧 hinge，不当作精确观测。

因此“空支撑 + 整体水平”是一个合法解释。只有当某个组件支撑显著优于它时，
枚举才提供组件层面的证据。等价解释指容差带内互不包含的不同支撑；全零
支撑与组件支撑同样落在容带内时，明确报告“无法区分组件与共同偏移”。

枚举预算按评估次数计，耗尽时剩余支撑登记为 unresolved，绝不静默丢弃。
"""
from itertools import combinations

from .fit import assemble, coordinate_descent

DEFAULT_MAX_EVALUATIONS = 400


def _active_floor(beta, noise_sigma=None):
    """实质非零阈值：主效应的 5%，且不低于 0.5 倍零模型噪声尺度。

    只用相对阈值会把 0.3ms 的噪声当独立解释；只用绝对阈值又会随单位变化。
    两者取大，阈值由数据自身的噪声尺度标定。
    """
    peak = max(beta, default=0.0)
    floor = max(1e-6, 0.05 * peak)
    if noise_sigma:
        floor = max(floor, 0.5 * noise_sigma)
    return floor


def _nuisance(rows, sources):
    """整体水平列 + 源漂移列（参考源略去以消除平移冗余）。"""
    ref = sources[0] if sources else None
    columns = []
    for row in rows:
        vec = [1.0] + [1.0 if row['source'] == s else 0.0
                       for s in sources if s != ref]
        columns.append(vec)
    return columns


def _fit_support(feature, nuisance, targets, weights, floors, floor_weights,
                 support, component_upper, max_rounds=20):
    """在支撑上做加权二次 + 单侧 hinge 的坐标下降（超时按当前拟合进入下一轮）。"""
    sub = [[row[g] for g in support] + nuisance[k]
           for k, row in enumerate(feature)]
    n_cols = len(sub[0]) if sub else 0
    theta = [0.0] * n_cols
    penalties = [0.0] * n_cols
    for _ in range(max_rounds):
        w = list(weights)
        changed = False
        for k, fw in floor_weights.items():
            pred = sum(sub[k][i] * theta[i] for i in range(n_cols))
            if pred < floors[k] - 1e-9:
                if w[k] == 0.0:
                    changed = True
                w[k] = 1.0
            else:
                if w[k] != 0.0:
                    changed = True
                w[k] = 0.0
        ata, atb = assemble(sub, targets, w)
        theta = coordinate_descent(ata, atb, theta, penalties, len(support),
                                   component_upper)
        if not floor_weights or not changed:
            break
    return theta


def _loss(feature, nuisance, targets, weights, floors, floor_weights, support, theta):
    pred = [sum(feature[k][g] * theta[i] for i, g in enumerate(support))
            + sum(nuisance[k][i] * theta[len(support) + i]
                  for i in range(len(nuisance[k])))
            for k in range(len(feature))]
    sse = sum(w * (pred[k] - targets[k]) ** 2 for k, w in enumerate(weights))
    hinge = 0.0
    for k, fw in floor_weights.items():
        gap = floors[k] - pred[k]
        if gap > 0:
            hinge += fw * gap * gap
    return sse + hinge, pred


def enumerate_supports(rows, baseline_predict, grouping, max_size=2,
                       max_evaluations=DEFAULT_MAX_EVALUATIONS,
                       tolerance_ratio=0.05):
    """枚举组支撑；返回按损失排序的评估、等价集合与未评估计数。"""
    groups = grouping['groups']
    feature, targets, weights, floors, floor_weights = [], [], [], [], {}
    usable = []
    for row in rows:
        info = baseline_predict(row)
        base = None if info is None else info['value']
        if base is None:
            continue
        idx = len(feature)
        feature.append([float(row['counts'].get(g['reference'], 0.0)) for g in groups])
        if row['success']:
            targets.append(row['latency'] - base)
            weights.append(1.0)
            floors.append(0.0)
        else:
            targets.append(0.0)
            weights.append(0.0)
            floors.append(row['timeout'] - base)
            floor_weights[idx] = 1.0
        usable.append(row)
    if not usable:
        return {'evaluations': [], 'equivalence_class': [], 'unresolved_count': 0,
                'evaluated_count': 0, 'budget_exhausted': False,
                'usable_probes': [], 'timeout_probes': [],
                'reason': '目标窗没有可用健康基线的样本，无法枚举支撑'}
    sources = sorted({row['source'] for row in usable})
    nuisance = _nuisance(usable, sources)
    observed = [max(floors[k] if k in floor_weights else targets[k], 0.0)
                for k in range(len(feature))]
    component_upper = 2.0 * max(observed or [1.0])
    # 零支撑模型的残差尺度：作为“实质非零”的噪声标定。
    theta0 = _fit_support(feature, nuisance, targets, weights, floors,
                          floor_weights, [], component_upper)
    loss0, pred0 = _loss(feature, nuisance, targets, weights, floors,
                         floor_weights, [], theta0)
    success_idx = [k for k in range(len(feature)) if k not in floor_weights]
    if success_idx:
        noise_sigma = (sum((pred0[k] - targets[k]) ** 2 for k in success_idx)
                       / len(success_idx)) ** 0.5
    else:
        noise_sigma = None
    n_groups = len(groups)
    all_supports = []
    for size in range(0, min(max_size, n_groups) + 1):
        all_supports.extend(combinations(range(n_groups), size))
    evaluations = []
    truncated = False
    for support in all_supports:
        if len(evaluations) >= max_evaluations:
            truncated = True
            break
        theta = _fit_support(feature, nuisance, targets, weights, floors,
                             floor_weights, list(support), component_upper)
        loss, _ = _loss(feature, nuisance, targets, weights, floors,
                        floor_weights, list(support), theta)
        beta = theta[:len(support)]
        floor = _active_floor(beta, noise_sigma)
        active_mask = [i for i, b in enumerate(beta) if b > floor]
        evaluations.append({
            'support_groups': [groups[support[i]]['components'] for i in active_mask],
            'requested_support': [groups[g]['components'] for g in support],
            'support_indices': list(support),
            'beta': [float(b) for b in beta],
            'global_level': float(theta[len(support)]),
            'loss': loss,
        })
    evaluations.sort(key=lambda e: (e['loss'], len(e['support_groups']),
                                    e['requested_support']))
    if not evaluations:
        return {'evaluations': [], 'equivalence_class': [], 'unresolved_count': 0,
                'evaluated_count': 0, 'budget_exhausted': False}
    active_evals = [e for e in evaluations if e['support_groups']]
    zero_eval = min((e for e in evaluations if not e['support_groups']),
                    key=lambda e: e['loss'], default=None)
    zero_loss = zero_eval['loss'] if zero_eval else float('inf')
    if active_evals:
        best_active = min(e['loss'] for e in active_evals)
        tolerance = best_active + tolerance_ratio * max(abs(best_active), 1e-12)
    else:
        best_active = None
        tolerance = None
    band = [e for e in active_evals if e['loss'] <= tolerance] if active_evals else []
    # 不做“子集即冗余”的剪枝：一个更大的支撑若新增组的系数实质非零，
    # 它代表不同解释（例如“两个组件同时故障” vs “一个组件故障 + 共同偏移”），
    # 必须保留为等价解释，不能因为更小支撑也在容差带内就丢掉。
    # 等价解释的标识同时包含“支撑组”和“整体水平是否实质非零”：
    # “组件 a 异常”与“整体水平上调 + 组件 b 异常”在 a+b=global 的共线
    # 设计下是同一个数据解释，但它们是不同的物理叙事，必须分别列出。
    global_scale = max(noise_sigma or 1.0, 1e-9)
    distinct, seen = [], set()
    for e in band:
        global_flag = abs(e['global_level']) > max(2.0 * global_scale, 1.0)
        key = (tuple(sorted(tuple(g) for g in e['support_groups'])), global_flag)
        if key in seen:
            continue
        seen.add(key)
        distinct.append({'support_groups': e['support_groups'],
                         'beta': e['beta'], 'loss': e['loss'],
                         'global_level': e['global_level'],
                         'uses_common_offset': global_flag})
    evidence = best_active is not None and best_active < zero_loss - 1e-12
    ordered = sorted(active_evals, key=lambda e: e['loss'])
    best_sets = frozenset(tuple(g) for g in ordered[0]['support_groups']) if ordered else frozenset()
    competitor = None
    for candidate in ordered[1:]:
        other = frozenset(tuple(g) for g in candidate['support_groups'])
        if not (other <= best_sets or best_sets <= other):
            competitor = candidate
            break
    second = competitor['loss'] if competitor is not None else None
    sparse_gap = (second - best_active) / max(abs(best_active), 1e-12) \
        if second is not None and best_active is not None else None
    return {
        'evaluations': evaluations,
        'equivalence_class': distinct,
        'best_loss': evaluations[0]['loss'],
        'best_active_loss': best_active,
        'zero_support_loss': zero_loss,
        'zero_support_global_level': zero_eval['global_level'] if zero_eval else None,
        'tolerance': tolerance,
        'evidence_against_null': evidence,
        'second_best_loss': second,
        'relative_loss_gap_to_second': sparse_gap,
        'evaluated_count': len(evaluations),
        'total_supports': len(all_supports),
        'unresolved_count': max(len(all_supports) - len(evaluations), 0),
        'budget_exhausted': truncated,
        'noise_sigma': noise_sigma,
        'component_upper': component_upper,
        'usable_probes': [row['probe'] for row in usable],
        'timeout_probes': [usable[k]['probe'] for k in sorted(floor_weights)],
        'note': ('未评估的支撑不代表已排除；预算耗尽时仍列在unresolved_count中。'
                 if truncated else '全部支撑均已评估。'),
    }
