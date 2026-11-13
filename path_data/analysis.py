"""把诊断、基线、拟合、枚举与建议串成一次定位报告。"""
from .baseline import (
    estimate, make_predictor, classify_missing, visible_healthy_windows,
    BASELINE_NOTE,
)
from .design import identifiability
from .fit import DEFAULT_LAMBDAS, fit_target, summary as fit_summary
from .io import DataError
from .observed import path_residuals, window_level
from .paths import prepare_rows, route_switches
from .recommend import recommend
from .replay import replay
from .support import enumerate_supports


DEFAULT_OPTIONS = {
    'lambda': None,
    'lambdas': DEFAULT_LAMBDAS,
    'max_support_size': 2,
    'max_support_evaluations': 400,
    'support_tolerance_ratio': 0.05,
    'replay': True,
    'candidate_paths': None,
    'candidate_sources': None,
    'max_suggestions': 3,
    'probe_versions': None,
    'version_schedule': None,
    'analysis_window': None,
}


def analyse(tables, options=None):
    """入口：资料表 + 选项 → 可证据化的候选组件报告。

    报告结构（全部为纯数据，可 JSON 序列化）：
      identifiability  可识别性诊断（必须先看）
      baseline         健康窗参考基线及其可识别性约定
      fit              稳健联合估计
      candidates       候选组件集合 / 等价解释
      observed         可观察路径残差
      replay           时间外回放与选参
      recommendations  下一条最有区分价值的探测建议
      warnings         限制与依赖
    """
    opts = dict(DEFAULT_OPTIONS)
    opts.update(options or {})
    components = [row['id'] for row in tables['components']]
    if len(set(components)) != len(components):
        raise DataError('components存在重复身份')

    analysis_window = opts['analysis_window']
    if analysis_window is None:
        analysis_window = max((p['window'] for p in tables['probes']), default=0)
    rows = prepare_rows(tables, opts['probe_versions'], opts['version_schedule'])
    visible = [r for r in rows if r['window'] <= analysis_window]
    healthy_set = visible_healthy_windows(tables, analysis_window)
    target_rows = [r for r in visible if r['window'] not in healthy_set]

    diag = identifiability(visible, components)
    grouping = diag['grouping']

    base = estimate(visible, components, healthy_set, grouping)
    predict = make_predictor(base)
    missing = classify_missing(target_rows, predict)

    lam = opts['lambda']
    selection = None
    if lam is None:
        if opts['replay']:
            selection = replay(tables, components, lambdas=opts['lambdas'],
                               probe_versions=opts['probe_versions'],
                               version_schedule=opts['version_schedule'])
            lam = selection.get('selected_lambda')
        if lam is None:
            lam = 1.0
            if selection is not None:
                selection['fallback_lambda'] = lam
                selection['fallback_reason'] = '回放不可评分，采用预注册的保守默认λ'

    fit = fit_target(target_rows, components, predict, grouping, lam=lam)
    fit_view = {'lambda': lam}
    if fit.get('ok'):
        fit_view.update({
            'groups': fit['groups'],
            'sources': fit['sources'],
            'source_se': fit['source_se'],
            'sigma': fit['sigma'],
            'usable_rows': fit['usable_rows'],
            'delta_convention': fit['delta_convention'],
            'missing_baseline': fit.get('missing_baseline', []),
            'fitted': fit.get('fitted'),
            'residuals': fit.get('residuals'),
            'group_beta': fit.get('group_beta'),
            'global_level': fit.get('global_level'),
            'global_level_se': fit.get('global_level_se'),
            'summary': fit_summary(fit, target_rows),
        })
    else:
        fit_view.update({'ok': False, 'reason': fit.get('reason'),
                         'missing_baseline': fit.get('missing_baseline', missing)})

    support = enumerate_supports(
        target_rows, predict, grouping,
        max_size=opts['max_support_size'],
        max_evaluations=opts['max_support_evaluations'],
        tolerance_ratio=opts['support_tolerance_ratio'],
    )

    candidates, equivalence = _candidates(fit, support, diag, grouping,
                                           target_rows)
    observed = path_residuals(visible, healthy_set)
    warnings = _warnings(diag, base, fit, support, opts, target_rows, missing)

    recs = {'suggestions': [], 'note': '未提供候选路径，未生成探测建议'}
    candidate_paths = opts['candidate_paths'] or []
    if candidate_paths:
        ambiguity = {
            'indistinguishable_groups': diag['indistinguishable_groups'],
            'null_space_directions': diag['null_space_directions'],
        }
        recs = recommend(visible, components, grouping, candidate_paths,
                         opts['candidate_sources'] or sorted({r['source'] for r in visible}),
                         ambiguity, opts['max_suggestions'])
        recs['path_source_identifiability'] = diag['joint_path_source']

    replay_view = selection
    if replay_view is None and opts['replay']:
        replay_view = replay(tables, components, lambdas=opts['lambdas'],
                             probe_versions=opts['probe_versions'],
                             version_schedule=opts['version_schedule'])

    return {
        'analysis_window': analysis_window,
        'healthy_windows_visible': sorted(healthy_set),
        'window_count': len({r['window'] for r in visible}),
        'route_switches': route_switches(visible),
        'identifiability': _strip_internal(diag),
        'baseline': _baseline_view(base),
        'fit': fit_view,
        'candidates': candidates,
        'equivalence': equivalence,
        'observed_path_residuals': observed,
        'window_level': window_level(visible, healthy_set),
        'support_enumeration': _support_view(support),
        'replay': replay_view,
        'recommendations': recs,
        'warnings': warnings,
        'notes': [BASELINE_NOTE],
    }


def _baseline_view(base):
    return {
        'method': base.get('method'),
        'notes': base.get('notes', []),
        'path_baselines': {f'{k[0]}@v{k[1]}': v
                           for k, v in base.get('path_baselines', {}).items()},
        'source_offsets': base.get('source_offsets', {}),
        'healthy_success_samples': base.get('healthy_success_samples'),
        'healthy_timeouts_ignored': base.get('healthy_timeouts'),
        'path_source_design_full_rank': base.get('path_source_design_full_rank'),
        'group_baseline_usable': bool(base.get('group_baseline', {}).get('usable')
                                      and base.get('group_baseline', {}).get('full_rank')),
        'group_baseline_reason': base.get('group_baseline', {}).get('reason', ''),
        'coverage_components': base.get('coverage', {}).get('components', []),
    }


def _support_view(support):
    return {
        'best_loss': support.get('best_loss'),
        'best_active_loss': support.get('best_active_loss'),
        'second_best_loss': support.get('second_best_loss'),
        'relative_loss_gap_to_second': support.get('relative_loss_gap_to_second'),
        'zero_support_loss': support.get('zero_support_loss'),
        'evidence_against_null': support.get('evidence_against_null'),
        'tolerance': support.get('tolerance'),
        'equivalence_class': support.get('equivalence_class', []),
        'evaluations': [
            {'support_groups': e['support_groups'], 'beta': e['beta'], 'loss': e['loss']}
            for e in support.get('evaluations', [])
        ],
        'noise_sigma': support.get('noise_sigma'),
        'component_upper': support.get('component_upper'),
        'evaluated_count': support.get('evaluated_count'),
        'total_supports': support.get('total_supports'),
        'unresolved_count': support.get('unresolved_count'),
        'budget_exhausted': support.get('budget_exhausted'),
        'note': support.get('note'),
        'timeout_probes': support.get('timeout_probes', []),
    }


def _strip_internal(diag):
    return {k: v for k, v in diag.items() if k not in ('design_matrix', 'group_feature', 'grouping')}


def lower_bound_groups(fit, rows):
    if not fit.get('ok'):
        return []
    from .fit import lower_bound_only_groups
    return lower_bound_only_groups(fit, rows)


def _candidates(fit, support, diag, grouping, target_rows):
    """把主拟合与支撑枚举合成候选集合与等价解释。

    判据：
    - 组增量需同时超过 2 倍近似标准误和预注册的最小效应（1ms）；
    - 只在含超时样本的拟合中显著、在成功样本拟合中归零的组只报下界；
    - 若整体水平 g 显著但没有任何组显著，报告“无法在组件与共同偏移之间
      区分”，不把整体水平摊成组件增量；
    - 存在完全共现组时，只能报组组合量，个体组件标为不可识别；
    - 支撑枚举给出多个在容差带内同等好的解释时，标为等价解释。
    """
    if not fit.get('ok'):
        return {'component_set': [], 'status': 'insufficient_baseline',
                'interpretation': '没有可用健康基线，无法定位',
                'why': fit.get('reason')}, []
    lower_bound_only = lower_bound_groups(fit, target_rows)
    lb_groups = {tuple(item['components']) for item in lower_bound_only}
    active = []
    for g in fit['groups']:
        beta = g['beta']
        se = g['se']
        if tuple(g['components']) in lb_groups:
            continue  # 只由超时支持：作为下界单列，不算已定位
        threshold = max(2.0 * se, 1.0, 1e-6) if se is not None else max(1.0, 1e-6)
        if beta > threshold:
            active.append(g)
    members = sorted({c for g in active for c in g['components']})
    indistinguishable = [g['components'] for g in grouping['groups']
                         if len(g['components']) > 1]
    eq = support.get('equivalence_class', [])
    distinct = _distinct_supports(eq)
    has_evidence = support.get('evidence_against_null', False)
    equivalent = _distinct_alternatives(distinct)
    # 不可区分时，候选集合取所有等价解释涉及的组件之并：
    # “只能定位到这一组组件”比只报主拟合的活跃组件更诚实。
    candidate_set = members
    if equivalent:
        candidate_set = sorted({c for item in distinct
                                for g in item['support_groups'] for c in g})
    # 完全共现组：成员在物理上不可分，候选集合必须以组为单位表述。
    if indistinguishable and members:
        grouped = set()
        for g in grouping['groups']:
            if set(g['components']) & set(members):
                grouped |= set(g['components'])
        candidate_set = sorted(grouped)
    global_level = fit.get('global_level')
    global_se = fit.get('global_level_se')
    success_residuals = [r for r in (fit.get('residuals') or {}).values()
                         if r is not None]
    global_active = (global_level is not None and global_se is not None
                     and global_level > max(2.0 * global_se, 1.0))
    if not success_residuals and support.get('timeout_probes'):
        # 目标窗只有超时：任何增量都只是下界，不能称“定位到”或“共同偏移”。
        status = 'lower_bound_only'
        interpretation = ('目标窗只有超时样本：只能给出延迟下界，'
                          '不能把门槛当作点估计，也不足以区分组件与共同偏移')
    elif not members and global_active and not _distinct_alternatives(distinct):
        status = 'common_offset_only'
        interpretation = ('只能定位到共同偏移：整体延迟升高但没有任何组件'
                          '单独可识别；共同源偏移与“所有组件同步上调”不可区分')
    elif not has_evidence and not members:
        status = 'no_evidence'
        interpretation = '现有资料不足以区分故障组件与无故障'
    elif len(members) == 1 and not indistinguishable \
            and diag['null_space_dimension'] == 0 \
            and not equivalent:
        status = 'single_component'
        interpretation = '定位到一个组件'
    else:
        status = 'indistinguishable_set'
        interpretation = '只能定位到一组不可区分组件（或等价解释）'
    timeout_only = []
    if support.get('timeout_probes') and not success_residuals and members:
        timeout_only = [g['components'] for g in active]
    candidate = {
        'component_set': candidate_set,
        'component_estimates': {
            c: _component_estimate(fit, grouping, c) for c in members
        },
        'status': status,
        'interpretation': interpretation,
        'indistinguishable_groups': indistinguishable,
        'common_offset': {
            'global_level': global_level,
            'global_level_se': global_se,
            'significant': global_active,
            'source_drift': {s: v for s, v in fit.get('sources', {}).items()
                             if abs(v) > 1e-6},
            'interpretation': ('整体水平把“多路径共同变慢”单列；它与共享组件'
                               '增量在只有路径总量观测时可能不可区分。'),
        },
        'equivalent_supports': distinct,
        'equivalent_explanation': equivalent,
        'sparse_prior_dependence': _sparse_dependence(support),
        'selected_by_sparse_prior': bool(support.get('evaluations')),
        'supported_only_by_timeout_lower_bound': timeout_only,
        'lower_bound_only': lower_bound_only,
        'dependence': (
            '候选集合来自 L1 非负稀疏拟合，并使用“组件贡献不超过观测最大增量'
            '两倍”的分量上界；若不同支撑在容差带内同等好，则把最小支撑当作'
            '一种解释而非唯一解释，结论依赖稀疏先验与容差带。由超时单独支持'
            '出的增量只是下界。'
        ),
    }
    return candidate, {'distinct_supports': distinct, 'equivalent': equivalent}


def _distinct_alternatives(distinct):
    """是否存在两个在物理叙事上不同的解释。

    不同 = 支撑组互不包含，或其中一个把整体水平当作共同偏移来解释。
    后一种情形同样不可由这批资料区分：例如“组件 a 异常”与
    “整体水平上调 + 组件 b 异常”在共线设计下给出相同拟合。
    """
    sets = []
    for item in distinct:
        groups = frozenset(tuple(g) for g in item['support_groups'])
        if groups:
            sets.append((groups, bool(item.get('uses_common_offset'))))
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            gi, oi = sets[i]
            gj, oj = sets[j]
            if not (gi <= gj or gj <= gi) or oi != oj:
                return True
    return False


def _sparse_dependence(support):
    gap = support.get('relative_loss_gap_to_second')
    return {
        'relative_loss_gap_to_second': gap,
        'level': ('none' if gap is None else
                  'high' if gap < 0.05 else
                  'moderate' if gap < 0.25 else 'low'),
        'note': '容差带内的其他支撑与最佳解释的损失差很小，说明最小支撑的'
                '选择主要依赖稀疏先验；损失差大才说明数据本身支持该支撑。',
    }


def _component_estimate(fit, grouping, component):
    for g in grouping['groups']:
        if component in g['components']:
            beta = next(item for item in fit['groups']
                        if item['components'] == g['components'])
            wsum = sum(g['weights'][c] for c in g['components']) or 1.0
            return {
                'group': g['components'],
                'group_beta': beta['beta'],
                'group_se': beta['se'],
                # 组内分摊权重只是约定；共现组内不可区分时只有组和可报。
                'convention_weight_in_group': g['weights'][component] / wsum,
                'identifiable_individually': len(g['components']) == 1,
                'value_is_group_share_only': len(g['components']) > 1,
            }
    return None


def _distinct_supports(eq):
    out = []
    seen = set()
    for item in eq:
        key = tuple(sorted(tuple(g) for g in item['support_groups']))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _warnings(diag, base, fit, support, opts, target_rows, missing):
    warnings = []
    if diag['null_space_dimension'] > 0:
        warnings.append(
            f'设计矩阵不满秩（零空间维数{diag["null_space_dimension"]}）：'
            '存在不改变任何已观测路径的组件增量方向，不能把它们分别报成确定增量。')
    if diag['indistinguishable_groups']:
        groups = ['、'.join(g) for g in diag['indistinguishable_groups']]
        warnings.append('存在完全共现组件组：' + '；'.join(groups) +
                        '。只能给出组内组合量，个体增量不可识别。')
    if not diag['joint_path_source']['all_absolute_levels_comparable']:
        warnings.append('健康窗的路径版本与探测源没有连成同一张二部图：'
                        '跨分量的绝对延迟水平不可直接比较。')
    if base.get('notes'):
        warnings.extend(base['notes'])
    if missing:
        detail = '；'.join(f'{reason}({item["count"]}条)' for reason, item in sorted(missing.items()))
        warnings.append('以下目标窗样本缺少可用健康基线，未进入模型：' + detail)
    if support.get('budget_exhausted'):
        warnings.append(f'支撑枚举预算耗尽：{support.get("unresolved_count")}个支撑未评估，'
                        '这些解释尚未排除，不能声称已唯一确定。')
    if fit.get('ok') and fit.get('sigma') is None:
        warnings.append('目标窗没有成功样本，尺度不可估；超时只提供下界信息。')
    timeouts = [r for r in target_rows if not r['success']]
    if timeouts:
        warnings.append(
            f'目标窗有{len(timeouts)}个超时样本：只按“延迟>T”的下界进入似然，'
            '不把T当作精确观测，也不删除样本。由超时支持出的增量只是下界，'
            '不能当作点估计。')
    if fit.get('ok') and fit.get('sources'):
        drift = {s: v for s, v in fit['sources'].items() if abs(v) > 1e-6}
        if drift:
            warnings.append('目标窗估计到源漂移' +
                            '，'.join(f'{s}:{v:.2f}' for s, v in drift.items()) +
                            '；源漂移是共同偏移，不应归因到单条路径的组件。')
    warnings.append('延迟相加模型假设组件增量线性可加；排队非线性造成的残差'
                    '只作为路径残差报告，不强行归到某个物理组件。')
    return warnings
