"""下一条最有区分价值的额外探测建议。

建议只在给定候选路径与候选组件等价类上做计算，不发送任何网络请求。
价值度量用“可区分性”的可观察替代量：一条新探测若使某个未分辨方向的
设计列被单独覆盖，那么它就能把等价类拆开。对每个候选（路径版本、源），
计算“加入后新增的可分辨方向数 + 对候选增量估计方差的降低”。
"""
from .linalg import nullspace, rank


def recommend(rows, components, grouping, candidate_paths, candidate_sources,
              ambiguity, max_suggestions=3):
    """在给定候选路径上，按对可识别性的增益排序返回建议。

    对每个候选（路径版本, 源）加入一行新探测，重算原始组件设计矩阵的秩与
    零空间，以及承载“等价支撑”的组特征矩阵的秩。增益最大者优先；若所有
    候选都无增益，明确说明“给定候选路径无法拆开该不可区分方向”，而不是
    给出虚假的优先建议。
    """
    component_index = {c: i for i, c in enumerate(components)}
    base_component = [[float(row['counts'].get(c, 0.0)) for c in components]
                      for row in rows]
    comp_rank = rank(base_component) if base_component and base_component[0] else 0
    comp_null = nullspace(base_component) if base_component and base_component[0] else []
    base_feature = _feature(rows, grouping)
    base_rank = rank(base_feature) if base_feature and base_feature[0] else 0
    base_null = nullspace(base_feature) if base_feature and base_feature[0] else []
    seen = {(row['path'], row['version'], row['source']) for row in rows}
    scored = []
    for path_id, version, comps in candidate_paths:
        counts = _counts(comps)
        if any(c not in component_index for c in counts):
            raise ValueError(f'候选路径{path_id}@v{version}含未登记组件')
        component_row = [float(counts.get(c, 0.0)) for c in components]
        feature_row = [float(counts.get(g['reference'], 0.0))
                       for g in grouping['groups']]
        component_rank_gain = rank(base_component + [component_row]) - comp_rank
        component_null_after = nullspace(base_component + [component_row])
        component_reduction = max(len(comp_null) - len(component_null_after), 0)
        group_rank_gain = rank(base_feature + [feature_row]) - base_rank
        group_null_after = nullspace(base_feature + [feature_row])
        group_reduction = max(len(base_null) - len(group_null_after), 0)
        for source in candidate_sources or ['<unknown>']:
            scored.append({
                'path': path_id,
                'version': version,
                'components': comps,
                'source': source,
                'component_rank_gain': component_rank_gain,
                'component_null_space_reduction': component_reduction,
                'group_rank_gain': group_rank_gain,
                'group_null_space_reduction': group_reduction,
                'new_path_source_pair': (path_id, version, source) not in seen,
            })
    if not scored:
        return {'suggestions': [], 'note': '没有提供可用的候选路径，未生成建议'}
    scored.sort(key=lambda item: (-item['component_rank_gain'],
                                  -item['component_null_space_reduction'],
                                  -item['group_rank_gain'],
                                  -int(item['new_path_source_pair']),
                                  item['path'], item['version'], item['source']))
    best_gain = scored[0]['component_rank_gain']
    # 若所有给定候选都无增益，仍然给出按“已有观测次数最少”排序的候选，
    # 但明确标注它们不能拆开当前方向；不能把“无增益”包装成建议。
    if best_gain == 0 and all(item['component_null_space_reduction'] == 0
                              for item in scored):
        by_novelty = sorted(scored, key=lambda item: (
            not item['new_path_source_pair'], item['path'], item['version'],
            item['source']))
        scored = by_novelty
    out = []
    for item in scored[:max_suggestions]:
        reason = []
        if item['component_rank_gain'] > 0:
            reason.append(f'可使组件设计矩阵秩增加{item["component_rank_gain"]}，'
                          '拆开当前不可识别的组件方向')
        if item['component_null_space_reduction'] > 0:
            reason.append(f'可消除{item["component_null_space_reduction"]}个'
                          '不可观察的组件方向')
        if item['group_rank_gain'] > 0:
            reason.append(f'组特征矩阵秩增加{item["group_rank_gain"]}')
        if item['new_path_source_pair']:
            reason.append('该路径版本-源组合当前未观测过')
        if not reason:
            reason.append('对当前不可识别方向没有额外约束')
        out.append({
            'action': 'add_probe_on_given_path',
            'path': item['path'],
            'version': item['version'],
            'components': item['components'],
            'source': item['source'],
            'expected_component_rank_gain': item['component_rank_gain'],
            'expected_component_null_space_reduction':
                item['component_null_space_reduction'],
            'expected_group_rank_gain': item['group_rank_gain'],
            'expected_group_null_space_reduction':
                item['group_null_space_reduction'],
            'reason': '；'.join(reason),
            'is_best_available': item['component_rank_gain'] == best_gain,
        })
    no_gain = best_gain == 0 and all(
        item['component_null_space_reduction'] == 0 for item in scored)
    return {
        'suggestions': out,
        'given_candidates_can_resolve': not no_gain,
        'basis': '只针对给定候选路径与源，基于组件/组设计矩阵的秩与零空间增益；'
                 '不发送网络请求，也不新增未给定的路径。',
        'limitation': ('给定的候选路径无法拆开当前不可区分方向；需要一条覆盖'
                       '共现阶段之外组合的路径。' if no_gain else ''),
        'unresolved': ambiguity,
    }


def _counts(comps):
    out = {}
    for c in comps:
        out[c] = out.get(c, 0) + 1
    return out


def _feature(rows, grouping):
    return [[float(row['counts'].get(g['reference'], 0.0)) for g in grouping['groups']]
            for row in rows]
