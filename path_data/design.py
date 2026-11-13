"""组件-路径设计矩阵、可识别性诊断与不可区分等价类。

设计矩阵 A 的行是一次探测样本，列是一个组件；A[k, c] 是组件 c 在该样本
实际经过的完整有向路径上的经过次数（环路按次数累计）。定位只能依赖 A 的
列空间：任何落在 A 的零空间里的改动都不改变任何可观察路径，因此无法由
数据区分。

诊断分四层：
1. 组件设计矩阵的秩与零空间方向。
2. 成比例列：两个组件在所有路径上按固定比例出现（最常见的是完全相同），
   它们的个体增量不可分辨，只能给出带已知权重的组合量。
3. 组特征的剩余秩亏：组之间仍可能线性相关，此时稀疏解不唯一，必须由
   支撑枚举把尚未排除的解释列出来。
4. 与源偏移的联合可识别性：若一条路径版本在健康窗只由一个源探测，
   路径基线与源偏移不可分。这是“探测源共同偏移”能影响多条路径的机制，
   必须在建模前说清楚，而不是当作噪声。
"""
from .linalg import mat_vec, nullspace, rank

COLUMN_TOL = 1e-9


def design_matrix(rows, components):
    index = {c: i for i, c in enumerate(components)}
    mat = []
    for row in rows:
        vec = [0.0] * len(components)
        for comp, count in row['counts'].items():
            if comp not in index:
                raise KeyError(f'组件{comp}未在components表中登记')
            vec[index[comp]] += float(count)
        mat.append(vec)
    return mat


def _columns(mat):
    if not mat:
        return []
    return [[row[j] for row in mat] for j in range(len(mat[0]))]


def _ratio(col, ref):
    ratios = [a / b for a, b in zip(col, ref) if b != 0]
    if not ratios:
        return 0.0
    ratios.sort()
    return ratios[len(ratios) // 2]


def group_columns(mat, components, tol=COLUMN_TOL):
    """把成比例（同方向）的列合并为可识别的组特征。

    代表列取组内总经过次数最小的成员列，组权重 r_j = col_j / col_ref。
    组特征系数 beta 的含义是“按代表列跑一次所对应的组增量”：
        sum_{j in group} r_j * x_j = beta。
    列完全相同时 r_j = 1，beta 是组内各组件增量之和。
    """
    cols = _columns(mat)
    zero = []
    groups = []
    used = [False] * len(cols)
    for j, col in enumerate(cols):
        if used[j]:
            continue
        peak = max((abs(v) for v in col), default=0.0)
        if peak <= tol:
            zero.append(components[j])
            used[j] = True
            continue
        direction = [v / peak for v in col]
        members = [j]
        used[j] = True
        for k in range(j + 1, len(cols)):
            if used[k]:
                continue
            other = cols[k]
            other_peak = max((abs(v) for v in other), default=0.0)
            if other_peak <= tol:
                continue
            if max(abs(a - b) for a, b in zip(direction, [v / other_peak for v in other])) <= 1e-6:
                members.append(k)
                used[k] = True
        ref = min(members, key=lambda m: sum(cols[m]))
        weights = {components[m]: _ratio(cols[m], cols[ref]) for m in members}
        groups.append({
            'components': [components[m] for m in members],
            'weights': weights,
            'reference': components[ref],
            'direction': list(cols[ref]),
        })
    feature = [[groups[g]['direction'][i] for g in range(len(groups))]
               for i in range(len(mat))]
    return {'groups': groups, 'feature': feature, 'unobserved_components': zero,
            'proportional_column_groups': [
                {'components': g['components'], 'weights': g['weights'],
                 'reference': g['reference']}
                for g in groups if len(g['components']) > 1]}


def joint_identifiability(rows, components):
    """路径版本×源 的联合设计：检查路径基线/源偏移是否能分开。

    模型 latency = path_version_effect + source_offset。若每个路径版本只被
    一个源覆盖，则该路径版本效应与源效应完全混杂；用连通分量诊断：
    把 (路径版本) 与 (源) 作为二部图节点，一条样本连一条边。每个连通分量
    内只能识别相对差异（分量置零约定），跨分量的绝对水平不可比。
    """
    edges = {}
    for row in rows:
        pv = f'{row["path"]}@v{row["version"]}'
        src = f'source:{row["source"]}'
        edges.setdefault(pv, set()).add(src)
        edges.setdefault(src, set()).add(pv)
    nodes = set(edges)
    seen = set()
    components_of_graph = []
    for node in sorted(nodes):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in edges.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components_of_graph.append(sorted(comp))
    columns = len(nodes) - len(components_of_graph)
    # 绝对延迟水平只有在所有路径版本与源连成同一分量时才可比；多分量时
    # 每个分量内部只确定“路径基线+源偏移”的和，分量之间缺一个整体平移。
    comparable = len(components_of_graph) <= 1
    confounded = []
    for comp in components_of_graph:
        sources = [n for n in comp if n.startswith('source:')]
        paths = [n for n in comp if '@v' in n]
        if len(sources) == 1 and len(paths) >= 1:
            confounded.append({'path_versions': paths, 'source': sources[0][len('source:'):],
                               'note': '该分量只有一个源；路径基线与源偏移只能确定其和'})
    return {
        'bipartite_components': components_of_graph,
        'n_components': len(components_of_graph),
        'residual_degrees_of_freedom': columns,
        'all_absolute_levels_comparable': comparable,
        'source_confounded_path_versions': confounded,
        'note': ('源偏移只在健康窗估计；故障窗的源漂移被弱正则约束，'
                 '以免把多路径共同异常误记成源偏移。'),
    }


def identifiability(rows, components):
    mat = design_matrix(rows, components)
    n = len(components)
    r = rank(mat) if mat and mat[0] else 0
    ns = nullspace(mat) if mat and mat[0] else []
    grouped = group_columns(mat, components)
    group_mat = grouped['feature']
    group_rank = rank(group_mat) if group_mat and group_mat[0] else 0
    group_null = nullspace(group_mat) if group_mat and group_mat[0] else []

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = [0.0] * n
            diff[i], diff[j] = 1.0, -1.0
            if max((abs(v) for v in mat_vec(mat, diff)), default=0.0) <= 1e-6:
                pairs.append([components[i], components[j]])

    directions = []
    for vec in ns:
        idx = [i for i, v in enumerate(vec) if abs(v) > 1e-9]
        directions.append({
            'components': [components[i] for i in idx],
            'direction': {components[i]: vec[i] for i in idx},
            'meaning': '沿该方向改变组件增量不改变任何已观测路径的延迟',
        })
    group_null_dirs = []
    for vec in group_null:
        idx = [i for i, v in enumerate(vec) if abs(v) > 1e-9]
        group_null_dirs.append({
            'groups': [grouped['groups'][i]['components'] for i in idx],
            'direction': {grouped['groups'][i]['components'][0]: vec[i] for i in idx},
        })

    return {
        'n_components': n,
        'n_observation_rows': len(rows),
        'rank': r,
        'full_rank': r == n,
        'null_space_dimension': len(ns),
        'null_space_directions': directions,
        'identical_column_pairs': pairs,
        'indistinguishable_groups': [g['components'] for g in grouped['groups']
                                     if len(g['components']) > 1],
        'proportional_column_groups': grouped['proportional_column_groups'],
        'unobserved_components': grouped['unobserved_components'],
        'groups': [{'components': g['components'], 'weights': g['weights'],
                    'reference': g['reference'], 'proportional': len(g['components']) > 1}
                   for g in grouped['groups']],
        'group_rank': group_rank,
        'group_null_space_dimension': len(group_null),
        'group_null_space_directions': group_null_dirs,
        'design_matrix': mat,
        'group_feature': group_mat,
        'grouping': grouped,
        'joint_path_source': joint_identifiability(rows, components),
    }
