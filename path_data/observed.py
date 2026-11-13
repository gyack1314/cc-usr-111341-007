"""同路径版本的探测对照：最直接的“可观察路径残差”。

同一 (path, version, source) 在健康窗与故障窗的延迟差，是无需任何模型假设
的 raw 可观察量。它是模型残差之外的交叉检查：若模型给出“组件 x 异常 20ms”
但同路径同源在故障窗只慢了 3ms，这两个结论不能同时成立。

输出逐 (path, version, source) 的：
- 健康窗样本数、稳健位置（截尾均值）、范围
- 故障窗成功样本数、稳健位置、范围
- 成功延迟差；若故障窗含超时，报告“至少超过门槛 T”的下界说明，
  不把超时折算成差分。
"""
from .baseline import trimmed_mean


def _median(values):
    vals = sorted(values)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def path_residuals(rows, healthy_set):
    healthy = {}
    for row in rows:
        if row['window'] in healthy_set and row['success']:
            healthy.setdefault((row['path'], row['version'], row['source']), []).append(row['latency'])
    target = {}
    timeouts = {}
    for row in rows:
        if row['window'] in healthy_set:
            continue
        key = (row['path'], row['version'], row['source'])
        if row['success']:
            target.setdefault(key, []).append(row['latency'])
        else:
            timeouts.setdefault(key, []).append(row['timeout'])
    out = []
    keys = sorted(set(healthy) | set(target) | set(timeouts), key=lambda k: (k[0], k[1], k[2]))
    for key in keys:
        h = healthy.get(key, [])
        t = target.get(key, [])
        to = timeouts.get(key, [])
        hloc = trimmed_mean(h, 0.1) if h else None
        tloc = trimmed_mean(t, 0.1) if t else None
        entry = {
            'path': key[0],
            'version': key[1],
            'source': key[2],
            'healthy_n': len(h),
            'healthy_location': hloc,
            'healthy_min': min(h) if h else None,
            'healthy_max': max(h) if h else None,
            'target_success_n': len(t),
            'target_success_location': tloc if t else None,
            'target_timeout_n': len(to),
            'target_timeout_floor_min': min(to) if to else None,
            'observed_difference': (tloc - hloc) if (t and h and hloc is not None) else None,
            'status': 'ok',
            'notes': [],
        }
        if not h:
            entry['status'] = 'no_healthy_comparison'
            entry['notes'].append('该路径版本在该源没有健康成功样本，不能给出无模型差分')
        if to:
            entry['notes'].append(
                f'故障窗有{len(to)}个超时；只知道延迟超过门槛{min(to)}ms，'
                '这是下界，不参与精确差分')
            if not t:
                entry['status'] = 'timeout_only_lower_bound'
        if h and t and hloc is not None:
            entry['observed_difference'] = tloc - hloc
        out.append(entry)
    return out


def window_level(rows, healthy_set):
    """按窗的路径残差，用于展示故障发生于哪些窗、是否有路由切换。"""
    out = {}
    for row in rows:
        if row['window'] in healthy_set:
            continue
        entry = out.setdefault(row['window'], {'success': 0, 'timeout': 0,
                                               'paths': {}, 'sources': set()})
        entry['paths'].setdefault(f'{row["path"]}@v{row["version"]}', []).append(
            row['latency'] if row['success'] else f'>T{row["timeout"]}')
        entry['sources'].add(row['source'])
        entry['success' if row['success'] else 'timeout'] += 1
    for entry in out.values():
        entry['sources'] = sorted(entry['sources'])
    return out
