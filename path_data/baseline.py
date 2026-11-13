"""仅用已确认健康窗口估计路径基线、组件组基线与探测源偏移。

约定与限制：
- 只有 confirmed_at 不晚于分析时点的健康窗才可见（时点可见性）；分析时点
  之后的健康确认不参与，避免把后来的结论泄漏进历史决策。
- 健康窗里发生路由切换时按 (path, version) 分别建基线，绝不把两个版本混在
  同一均值里。
- 延迟基线用加权最小二乘 + Tukey 稳健重加权；少量异常探测不会拖动基线。
- 探测源偏移只在健康窗估计。若某路径版本只由一个源探测，路径基线与该源
  偏移在数学上不可分；两者之和仍可预测，但单个数值只是约定解。
- 预测时要求“路径版本节点”和“源节点”在健康覆盖二部图的同一连通分量内：
  只有该分量内的“路径基线 + 源偏移”之和是不随约定平移变化的。跨分量或
  未覆盖源的组合不预测，登记为 missing，而不是静默外推。

基线从不使用故障窗数据，也不使用任何人工故障标签。
"""
from .linalg import rank, solve

BASELINE_NOTE = (
    '组件参考基线只在 confirmed_at 不晚于分析时点的健康窗估计；故障窗数据'
    '不参与基线，源偏移也不在故障窗重估，以免把异常吸收成源偏移。'
)


def visible_healthy_windows(tables, analysis_window):
    out = set()
    for row in tables['healthy']:
        if row['confirmed_at'] <= analysis_window:
            out.add(row['window'])
    return out


def trimmed_mean(values, trim=0.0):
    vals = sorted(values)
    if not vals:
        return None
    if trim <= 0 or len(vals) < 3:
        return sum(vals) / len(vals)
    k = int(len(vals) * trim)
    core = vals[k:len(vals) - k] or vals
    return sum(core) / len(core)


def median(values):
    vals = sorted(values)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def robust_scale(values):
    vals = sorted(abs(v) for v in values)
    if not vals:
        return None
    return max(1.4826 * vals[len(vals) // 2], 1e-9)


def _tukey(values, scale, c=4.685):
    return [(1.0 - (v / (c * scale)) ** 2) ** 2 if abs(v / (c * scale)) < 1.0 else 0.0
            for v in values]


def _weighted_lstsq(design, target, weights, ridge=1e-9):
    n = len(design[0]) if design else 0
    if not n:
        return None
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, y, w in zip(design, target, weights):
        if w <= 0:
            continue
        for i in range(n):
            if row[i] == 0:
                continue
            atb[i] += w * row[i] * y
            for j in range(n):
                ata[i][j] += w * row[i] * row[j]
    scale = max([ata[i][i] for i in range(n)] or [1.0])
    for i in range(n):
        ata[i][i] += ridge * max(scale, 1.0)
    return solve(ata, atb)


def _irls(design, target, iterations=8):
    weights = [1.0] * len(target)
    coef = _weighted_lstsq(design, target, weights)
    if coef is None:
        return None, weights
    for _ in range(iterations):
        residuals = [sum(r[i] * coef[i] for i in range(len(coef))) - y
                     for r, y in zip(design, target)]
        scale = robust_scale(residuals)
        if scale is None:
            break
        new_weights = _tukey(residuals, scale)
        coef = _weighted_lstsq(design, target, new_weights)
        if coef is None:
            break
        weights = new_weights
    return coef, weights


def _path_source_model(healthy):
    keys = []
    seen = set()
    for row in healthy:
        key = (row['path'], row['version'])
        if key not in seen:
            seen.add(key)
            keys.append(key)
    sources = sorted({row['source'] for row in healthy})
    ref = sources[0] if sources else None
    ncols = len(keys) + max(len(sources) - 1, 0)
    design, target = [], []
    for row in healthy:
        vec = [0.0] * ncols
        vec[keys.index((row['path'], row['version']))] = 1.0
        if row['source'] != ref:
            vec[len(keys) + sources.index(row['source']) - 1] = 1.0
        design.append(vec)
        target.append(float(row['latency']))
    full_rank = bool(design) and rank(design) == ncols
    coef, weights = _irls(design, target)
    if coef is None:
        return None
    path_baselines = {keys[i]: coef[i] for i in range(len(keys))}
    offsets = {ref: 0.0}
    for i, s in enumerate(sources):
        if s != ref:
            offsets[s] = coef[len(keys) + i - 1]
    return {
        'path_baselines': path_baselines,
        'source_offsets': offsets,
        'full_rank': full_rank,
        'columns': ncols,
        'keys': keys,
        'sources': sources,
        'weights': weights,
    }


def _fallback_model(healthy):
    per_path = {}
    for row in healthy:
        per_path.setdefault((row['path'], row['version']), []).append(row['latency'])
    path = {k: trimmed_mean(v, 0.1) for k, v in per_path.items()}
    residuals = {}
    for row in healthy:
        residuals.setdefault(row['source'], []).append(
            row['latency'] - path[(row['path'], row['version'])])
    offsets = {s: median(v) for s, v in residuals.items()}
    total = sum(len(v) for v in residuals.values()) or 1
    shift = sum(offsets[s] * len(residuals[s]) for s in offsets) / total
    offsets = {s: v - shift for s, v in offsets.items()}
    return {
        'path_baselines': path,
        'source_offsets': offsets,
        'full_rank': False,
        'columns': len(path) + max(len(offsets) - 1, 0),
        'keys': sorted(path),
        'sources': sorted(offsets),
    }


def _group_model(healthy, grouping):
    groups = grouping['groups']
    sources = sorted({row['source'] for row in healthy})
    ref = sources[0] if sources else None
    ncols = len(groups) + max(len(sources) - 1, 0)
    design, target = [], []
    for row in healthy:
        vec = [float(row['counts'].get(g['reference'], 0)) for g in groups]
        vec += [0.0] * max(len(sources) - 1, 0)
        if row['source'] != ref:
            vec[len(groups) + sources.index(row['source']) - 1] = 1.0
        design.append(vec)
        target.append(float(row['latency']))
    full_rank = bool(design) and rank(design) == ncols
    coef, _ = _irls(design, target)
    if coef is None:
        return {'usable': False, 'reason': '健康窗组基线无解', 'theta': {}, 'offsets': {}}
    theta = {g['components'][0]: coef[i] for i, g in enumerate(groups)}
    offsets = {ref: 0.0}
    for i, s in enumerate(sources):
        if s != ref:
            offsets[s] = coef[len(groups) + i - 1]
    reason = '' if full_rank else (
        '健康窗的组特征与源偏移列线性相关，组基线不唯一；'
        '仅用于健康窗未覆盖路径的投影，且已标注为模型投影值')
    return {
        'usable': True,
        'full_rank': full_rank,
        'theta': theta,
        'offsets': offsets,
        'rank': rank(design) if design else 0,
        'columns': ncols,
        'reason': reason,
    }


def coverage_graph(healthy):
    """健康覆盖二部图：节点为 path@vN 与 source:s，边为观测到的 (路径版本, 源)。"""
    edges = {}
    pairs = set()
    for row in healthy:
        pv = f'{row["path"]}@v{row["version"]}'
        src = f'source:{row["source"]}'
        edges.setdefault(pv, set()).add(src)
        edges.setdefault(src, set()).add(pv)
        pairs.add((row['path'], row['version'], row['source']))
    comp_of = {}
    comps = []
    for node in sorted(edges):
        if node in comp_of:
            continue
        stack = [node]
        comp_id = len(comps)
        members = []
        comp_of[node] = comp_id
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nxt in edges.get(cur, ()):
                if nxt not in comp_of:
                    comp_of[nxt] = comp_id
                    stack.append(nxt)
        comps.append(sorted(members))
    return {'pairs': pairs, 'component_of': comp_of, 'components': comps}


def estimate(rows, components, healthy_set, grouping):
    healthy = [row for row in rows if row['window'] in healthy_set and row['success']]
    healthy_timeouts = [row for row in rows if row['window'] in healthy_set
                        and not row['success']]
    notes = []
    if not healthy:
        return {
            'method': 'unavailable',
            'notes': ['没有可见的健康成功样本，无法建立参考基线'],
            'path_baselines': {}, 'source_offsets': {},
            'group_baseline': {'usable': False}, 'healthy_success_samples': 0,
            'healthy_timeouts': len(healthy_timeouts),
            'coverage': coverage_graph([]),
        }
    model = _path_source_model(healthy)
    if model is None:
        model = _fallback_model(healthy)
        method = 'fallback_path_center_convention'
        notes.append('健康窗样本不足以联合求解路径基线与源偏移，已改用路径稳健均值'
                     '加居中源偏移的约定；源偏移数值不可作为独立结论。')
    else:
        method = 'robust_joint_path_and_source_ls'
        if not model['full_rank']:
            notes.append('健康窗的路径-源联合设计不满秩：路径基线与源偏移只能确定'
                         '其和，以下数值是按参考源置零的约定解，仅用于预测，'
                         '不作为源偏移结论。')
    coverage = coverage_graph(healthy)
    group = _group_model(healthy, grouping)
    if group.get('reason'):
        notes.append(group['reason'])
    if healthy_timeouts:
        notes.append(f'健康窗含{len(healthy_timeouts)}个超时探测，未用于基线'
                     '（超时只给下界，不能用于估计均值）。')
    return {
        'method': method,
        'notes': notes,
        'path_baselines': model['path_baselines'],
        'source_offsets': model['source_offsets'],
        'group_baseline': group,
        'healthy_success_samples': len(healthy),
        'healthy_timeouts': len(healthy_timeouts),
        'path_source_design_full_rank': model['full_rank'],
        'path_source_design_columns': model['columns'],
        'healthy_sources': sorted({row['source'] for row in healthy}),
        'coverage': coverage,
    }


def make_predictor(baseline):
    """返回 row → {value, provenance, reason} 的预测器。

    provenance:
      direct              该 (路径版本, 源) 在健康窗直接观测过
      path_plus_offset    路径版本与源都在同一健康连通分量，偏移可估
      group_projected     路径版本未在健康窗出现，用组件组基线投影（较弱）
      unavailable         无法给出可信基线
    """
    path_baselines = baseline.get('path_baselines', {})
    offsets = baseline.get('source_offsets', {})
    group = baseline.get('group_baseline', {}) or {}
    coverage = baseline.get('coverage', {}) or {}
    pairs = coverage.get('pairs', set())
    comp_of = coverage.get('component_of', {})

    def predict(row):
        key = (row['path'], row['version'])
        source = row['source']
        pv_node = f'{row["path"]}@v{row["version"]}'
        src_node = f'source:{source}'
        if key in path_baselines:
            if source not in offsets:
                return {'value': None, 'provenance': 'unavailable',
                        'reason': f'源{source}在健康窗没有样本，源偏移不可估'}
            c_pv = comp_of.get(pv_node)
            c_src = comp_of.get(src_node)
            if c_pv is None or c_src is None or c_pv != c_src:
                return {'value': None, 'provenance': 'unavailable',
                        'reason': '路径版本与源不在健康覆盖的同一连通分量，'
                                  '绝对水平不可比'}
            if (row['path'], row['version'], source) in pairs:
                provenance = 'direct'
            else:
                provenance = 'path_plus_offset'
            return {'value': path_baselines[key] + offsets[source],
                    'provenance': provenance, 'reason': ''}
        if group.get('usable') and group.get('full_rank'):
            theta = group.get('theta', {})
            g_offsets = group.get('offsets', {})
            if any(c not in theta for c in row['counts']):
                return {'value': None, 'provenance': 'unavailable',
                        'reason': '路径含健康窗从未观测到的组件，组件组基线无法投影'}
            if source not in g_offsets:
                return {'value': None, 'provenance': 'unavailable',
                        'reason': f'源{source}在健康窗没有样本，源偏移不可估'}
            value = sum(theta[c] * n for c, n in row['counts'].items())
            return {'value': value + g_offsets[source],
                    'provenance': 'group_projected',
                    'reason': '该路径版本未在健康窗出现，按组件组基线投影；'
                              '若组基线不唯一则只是约定值'}
        return {'value': None, 'provenance': 'unavailable',
                'reason': '该路径版本没有健康基线，且组件组基线不可用'}

    return predict


def classify_missing(rows, predictor):
    """按原因汇总无法预测的目标窗样本。"""
    out = {}
    for row in rows:
        info = predictor(row)
        if info['value'] is not None:
            continue
        key = info['reason']
        entry = out.setdefault(key, {'count': 0, 'probes': []})
        entry['count'] += 1
        entry['probes'].append(row['probe'])
    return out
