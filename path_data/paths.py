"""路径版本解析与按样本的有向路径展开。

每个探测样本必须对应一条完整的、带版本的路径：
- 探测表 schema=1 不含 version 列，因此版本由 paths 表内该路径 id 的版本唯一性
  决定；若同一路径 id 登记了多个版本，调用方必须提供 probe_versions
  （{probe_id: version}）或 version_schedule（{path_id: [(首个可见窗, version)]}）。
  分析绝不自行猜测版本。
- paths.components 按实际经过顺序给出，允许重复；设计矩阵按实际经过次数
  累计，环路重复经过不会被折叠成 0/1。
"""
from .io import DataError


def paths_by_version(paths):
    out = {}
    for row in paths:
        key = (row['id'], row['version'])
        if key in out:
            raise DataError(f'路径版本重复登记: {key}')
        comps = row['components']
        if not isinstance(comps, list) or not comps or not all(isinstance(c, str) and c for c in comps):
            raise DataError(f'路径{key}的components必须是非空字符串列表')
        out[key] = list(comps)
    return out


def _schedule_version(schedule, path, window):
    entry = schedule.get(path)
    if entry is None:
        return None
    if isinstance(entry, dict):
        if window in entry:
            return entry[window]
        earlier = [w for w in entry if w <= window]
        return entry[max(earlier)] if earlier else None
    fits = [(since, ver) for since, ver in entry if since <= window]
    return max(fits)[1] if fits else None


def resolve_version(probe, versions, probe_versions=None, version_schedule=None):
    if probe_versions and probe['id'] in probe_versions:
        version = probe_versions[probe['id']]
    elif len(versions) == 1:
        version = next(iter(versions))
    else:
        version = _schedule_version(version_schedule or {}, probe['path'], probe['window'])
        if version is None:
            raise DataError(
                f'探测{probe["id"]}无法确定路径版本：{probe["path"]}有多个版本'
                f'{sorted(versions)}，需提供probe_versions或窗口→版本计划')
    if version not in versions:
        raise DataError(f'探测{probe["id"]}引用了不存在的路径版本'
                        f'({probe["path"]}, {version})')
    return version


def expand(components_per_path):
    counts = {}
    for comp in components_per_path:
        counts[comp] = counts.get(comp, 0) + 1
    return counts


def prepare_rows(tables, probe_versions=None, version_schedule=None,
                 analysis_window=None):
    """把资料表展开为统一的分析行。不读取任何人工故障标签。"""
    by_version = paths_by_version(tables['paths'])
    versions_per_path = {}
    for path_id, version in by_version:
        versions_per_path.setdefault(path_id, set()).add(version)
    rows = []
    for probe in tables['probes']:
        if analysis_window is not None and probe['window'] > analysis_window:
            continue
        versions = versions_per_path.get(probe['path'])
        if not versions:
            raise DataError(f'探测{probe["id"]}引用了不存在的路径{probe["path"]}')
        version = resolve_version(probe, versions, probe_versions, version_schedule)
        comps = by_version[(probe['path'], version)]
        if probe['success']:
            if probe['latency_ms'] is None:
                raise DataError(f'探测{probe["id"]}成功但缺少latency_ms')
            if probe['timeout_ms'] is not None and probe['latency_ms'] > probe['timeout_ms']:
                raise DataError(f'探测{probe["id"]}延迟高于其超时门槛却标记成功')
        else:
            if probe['latency_ms'] is not None:
                raise DataError(f'探测{probe["id"]}失败却带有精确latency_ms')
            if probe['timeout_ms'] is None:
                raise DataError(f'探测{probe["id"]}失败却缺少超时门槛')
        rows.append({
            'probe': probe['id'],
            'source': probe.get('source') or '<unknown>',
            'window': probe['window'],
            'path': probe['path'],
            'version': version,
            'components': comps,
            'counts': expand(comps),
            'success': bool(probe['success']),
            'latency': probe['latency_ms'],
            'timeout': probe['timeout_ms'],
        })
    return rows


def route_switches(rows):
    seen = {}
    for row in rows:
        seen.setdefault(row['path'], set()).add(row['version'])
    return {p: sorted(v) for p, v in seen.items() if len(v) > 1}
