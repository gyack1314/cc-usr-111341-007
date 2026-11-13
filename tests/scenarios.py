"""合成场景资料：用于验证定位结论能区分“一个组件”和“一组不可区分组件”。

这里生成的是分析层的测试夹具，不写入 data/ 下的登记资料包，也不发送网络
请求。每个场景返回 (components, paths, probes, healthy) 四张表，字段与
docs/data-contract.md 一致。故障标签只由测试断言使用，分析代码从不读取。
"""
import random


def make_tables(component_ids, path_rows, probe_rows, healthy_rows):
    return {
        'components': [{'id': c} for c in component_ids],
        'paths': path_rows,
        'probes': probe_rows,
        'healthy': healthy_rows,
    }


def _probe(pid, source, window, path, success, latency, timeout):
    return {'id': pid, 'source': source, 'window': window, 'path': path,
            'success': success, 'latency_ms': latency, 'timeout_ms': timeout}


def common_upstream(seed=7, target_delta=40, windows=6, noise=1.0):
    """共同上游 c 被多条路径共享：应能定位到 c，且各路径残差同向。

    c 是每条路径的前缀组件；另有 a、b、d 分别只出现在部分路径上。
    健康窗 1-3，故障窗 4-6。
    """
    rng = random.Random(seed)
    components = ['c', 'a', 'b', 'd']
    paths = [
        {'id': 'p1', 'version': 1, 'components': ['c', 'a']},
        {'id': 'p2', 'version': 1, 'components': ['c', 'b']},
        {'id': 'p3', 'version': 1, 'components': ['c', 'a', 'd']},
        # p4 绕过共同上游 c：否则 c 列与整体水平列完全共线，c 与共同源偏移
        # 不可区分（这正是可识别性诊断要指出的情形）。
        {'id': 'p4', 'version': 1, 'components': ['b', 'd']},
        # p5 只经过 c：使 c 可被单独约束。
        {'id': 'p5', 'version': 1, 'components': ['c']},
    ]
    # 健康基线：c=10, a=5, b=8, d=3；源偏移 a_src=+2, b_src=0。
    base = {'c': 10, 'a': 5, 'b': 8, 'd': 3}
    source_offset = {'src1': 2, 'src2': 0}
    probes = []
    n = 0
    for window in range(1, windows + 1):
        for path in paths:
            source = 'src1' if path['id'] in ('p1', 'p3', 'p5') else 'src2'
            for _repeat in range(3):
                n += 1
                healthy = window <= 3
                delta_c = target_delta if not healthy else 0
                value = sum(base[x] for x in path['components']) + source_offset[source] + delta_c
                value += rng.gauss(0, noise)
                # 故障窗里 p4 不走 c 的故障（用于检验“共同上游”而非全路径）
                if not healthy and path['id'] == 'p4':
                    value -= delta_c
                value = max(value, 1.0)
                probes.append(_probe(f's{n}', source, window, path['id'], True,
                                     round(value, 3), 200))
    healthy_rows = [{'id': f'h{w}', 'window': w, 'confirmed_at': 3}
                    for w in (1, 2, 3)]
    return make_tables(components, paths, probes, healthy_rows), {'c': 40.0}


def identical_path_components(seed=11, windows=6, noise=1.0):
    """组件 x 与 y 在每条路径上总是一起出现：只能报“x+y 的组合量”。"""
    rng = random.Random(seed)
    components = ['x', 'y', 'z']
    paths = [
        {'id': 'p1', 'version': 1, 'components': ['x', 'y']},
        {'id': 'p2', 'version': 1, 'components': ['x', 'y', 'z']},
        {'id': 'p3', 'version': 1, 'components': ['z']},
    ]
    base = {'x': 6, 'y': 6, 'z': 9}
    source_offset = {'s1': 0, 's2': 1}
    probes = []
    n = 0
    for window in range(1, windows + 1):
        for path in paths:
            for source in ('s1', 's2'):
                for _ in range(3):
                    n += 1
                    value = sum(base[x] for x in path['components']) + source_offset[source]
                    if window >= 4 and path['id'] in ('p1', 'p2'):
                        value += 25  # 组合增量
                    value += rng.gauss(0, noise)
                    probes.append(_probe(f's{n}', source, window, path['id'], True,
                                         round(max(value, 1), 3), 200))
    healthy_rows = [{'id': f'h{w}', 'window': w, 'confirmed_at': 3}
                    for w in (1, 2, 3)]
    return make_tables(components, paths, probes, healthy_rows), {'x': None, 'y': None}


def source_offset(seed=13, windows=8, noise=1.0):
    """探测源 src1 自身共同变慢，多路径同时升高：不应归因到某个组件。"""
    rng = random.Random(seed)
    components = ['c', 'a', 'b']
    paths = [
        {'id': 'p1', 'version': 1, 'components': ['c', 'a']},
        {'id': 'p2', 'version': 1, 'components': ['c', 'b']},
        {'id': 'p3', 'version': 1, 'components': ['a', 'b']},
    ]
    base = {'c': 10, 'a': 5, 'b': 7}
    probes = []
    n = 0
    for window in range(1, windows + 1):
        for path in paths:
            for source in ('src1', 'src2'):
                for _ in range(3):
                    n += 1
                    value = sum(base[x] for x in path['components']) + rng.gauss(0, noise)
                    if source == 'src1' and window >= 5:
                        value += 30
                    probes.append(_probe(f's{n}', source, window, path['id'], True,
                                         round(max(value, 1), 3), 200))
    healthy_rows = [{'id': f'h{w}', 'window': w, 'confirmed_at': 4}
                    for w in (1, 2, 3, 4)]
    return make_tables(components, paths, probes, healthy_rows), {'src1': 30.0}


def timeout_and_routing(seed=17, windows=7, noise=1.0, new_version_in_healthy=True):
    """故障窗有超时和窗内路由切换：超时只作下界，路径版本按样本使用。"""
    rng = random.Random(seed)
    components = ['c', 'a', 'b', 'd']
    paths = [
        {'id': 'p1', 'version': 1, 'components': ['c', 'a']},
        {'id': 'p1', 'version': 2, 'components': ['c', 'b']},   # 窗内切换
        {'id': 'p2', 'version': 1, 'components': ['c', 'd']},
        # p3 不含 c：使 c 列不与 a/b/d 的组合同向，共同上游才可单独识别。
        {'id': 'p3', 'version': 1, 'components': ['a', 'b']},
    ]
    base = {'c': 10, 'a': 5, 'b': 9, 'd': 4}
    probes = []
    probe_versions = {}
    n = 0
    for window in range(1, windows + 1):
        for path_id in ('p1', 'p2', 'p3'):
            for _ in range(4):
                n += 1
                # p1 在窗内切换版本；new_version_in_healthy=False 时新版本
                # 只在故障窗出现，用于验证“无健康基线不得静默插补”。
                switched = window >= 5 or (new_version_in_healthy and window == 3)
                if path_id == 'p1' and switched:
                    version = 2 if (n % 2 == 0) else 1
                else:
                    version = 1
                probe_versions[f's{n}'] = version
                comps = next(p['components'] for p in paths
                             if p['id'] == path_id and p['version'] == version)
                value = sum(base[x] for x in comps) + rng.gauss(0, noise)
                # 共用上游 c 在窗 5 变慢；只有经过 c 的路径（p1/p2）变慢，
                # p3（不含 c）保持健康——这是“共同上游”与“全体偏移”的关键
                # 区分证据。窗 6 起 p1 的部分探测超过门槛变成超时。
                if window >= 5 and 'c' in comps:
                    value += 35
                if window >= 6 and path_id == 'p2':
                    value += 200
                    success = value <= 100
                    probes.append(_probe(f's{n}', 's1', window, path_id, success,
                                         round(value, 3) if success else None, 100))
                else:
                    probes.append(_probe(f's{n}', 's1', window, path_id, True,
                                         round(max(value, 1), 3), 200))
    healthy_rows = [{'id': f'h{w}', 'window': w, 'confirmed_at': 4}
                    for w in (1, 2, 3, 4)]
    return (make_tables(components, paths, probes, healthy_rows),
            {'c': 35.0, 'probe_versions': probe_versions})


def loop_path(seed=19, windows=5, noise=0.5):
    """环路重复经过同一组件：设计矩阵按实际经过次数计数。"""
    rng = random.Random(seed)
    components = ['gw', 'svc']
    paths = [
        {'id': 'p1', 'version': 1, 'components': ['gw', 'svc', 'gw']},
        {'id': 'p2', 'version': 1, 'components': ['svc']},
    ]
    base = {'gw': 4, 'svc': 9}
    probes = []
    n = 0
    for window in range(1, windows + 1):
        for path in paths:
            for _ in range(4):
                n += 1
                value = sum(base[x] for x in path['components']) + rng.gauss(0, noise)
                if window >= 4 and path['id'] == 'p1':
                    value += 12
                probes.append(_probe(f's{n}', 's1', window, path['id'], True,
                                     round(max(value, 1), 3), 200))
    healthy_rows = [{'id': f'h{w}', 'window': w, 'confirmed_at': 3}
                    for w in (1, 2, 3)]
    return make_tables(components, paths, probes, healthy_rows), {'gw': 6.0}
