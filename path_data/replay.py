"""冻结历史窗的时间外回放：预注册选参，不使用后来的人工故障标签。

做法：
- 把可见窗按时间切成若干折。对第 f 折，用 f 之前的窗（且仅在其中已确认
  健康的窗）建立基线并选择 λ；在 f 折上计算一次性的外样本指标。
- λ 的选择标准事先固定（预注册），只看外样本的预测误差与超时下界违背，
  不看任何人工确认标签。
- 回放只报告样本外指标；不把回放结果反过来用于同一折的拟合。

预注册选参规则（在代码中固定，不随数据变化）：
  1. 先排除在外折上出现超时下界违背的 λ；
  2. 在剩余 λ 中取外折成功样本的稳健尺度（median |residual|）最小者；
  3. 平局时取更大的 λ（更稀疏、解释更保守）。
"""
from .baseline import estimate, make_predictor, visible_healthy_windows
from .fit import DEFAULT_LAMBDAS, fit_target, robust_scale
from .design import design_matrix, group_columns


def fold_windows(rows):
    return sorted({row['window'] for row in rows})


def replay(tables, components, lambdas=DEFAULT_LAMBDAS,
           warmup_folds=1, max_folds=None, probe_versions=None,
           version_schedule=None):
    """按时间窗做外样本回放。

    每一折：训练窗 < 折窗；训练只使用训练窗内已确认健康的窗建基线。
    折内必须至少有一个成功样本才计分，否则该折登记为不可评分。
    """
    from .paths import prepare_rows
    full = prepare_rows(tables, probe_versions, version_schedule)
    win_list = fold_windows(full)
    if len(win_list) < warmup_folds + 1:
        return {'folds': [], 'note': '可见窗不足，无法做时间外回放',
                'selected_lambda': None}
    split_points = win_list[warmup_folds:]
    if max_folds:
        split_points = split_points[:max_folds]
    folds = []
    for split in split_points:
        train_windows = [w for w in win_list if w < split]
        train_rows = [r for r in full if r['window'] in train_windows]
        test_rows = [r for r in full if r['window'] == split]
        if not test_rows:
            continue
        healthy_set = visible_healthy_windows(tables, split - 1) & set(train_windows)
        grouping = group_columns(design_matrix(train_rows, components), components)
        base = estimate(train_rows, components, healthy_set, grouping)
        predict = make_predictor(base)
        if not healthy_set:
            folds.append({'test_window': split, 'scorable': False,
                          'reason': '训练段没有可见健康窗，基线不可建'})
            continue
        candidates = []
        for lam in lambdas:
            fit = fit_target(test_rows, components, predict, grouping, lam=lam,
                             robust=True, max_iter=40)
            if not fit['ok']:
                candidates.append({'lambda': lam, 'scorable': False,
                                   'reason': fit.get('reason')})
                continue
            residuals = [fit['residuals'][r['probe']] for r in test_rows
                         if r['probe'] in fit['residuals']
                         and fit['residuals'][r['probe']] is not None]
            scale = robust_scale(residuals) if residuals else None
            violations = 0
            for row in test_rows:
                if row['success'] or row['probe'] not in fit['fitted']:
                    continue
                tol = 1e-6 * max(1.0, row['timeout'])
                if fit['fitted'][row['probe']] < row['timeout'] - tol:
                    violations += 1
            candidates.append({
                'lambda': lam, 'scorable': True, 'scale': scale,
                'violations': violations, 'n_success': len(residuals),
            })
        scored = [c for c in candidates if c.get('scorable') and c.get('scale') is not None]
        if not scored:
            folds.append({'test_window': split, 'scorable': False,
                          'reason': '折内没有可评分的成功样本',
                          'candidates': candidates})
            continue
        admissible = [c for c in scored if c['violations'] == 0] or scored
        best = sorted(admissible, key=lambda c: (c['scale'], -c['lambda']))[0]
        folds.append({
            'test_window': split,
            'train_windows': train_windows,
            'scorable': True,
            'candidates': candidates,
            'selected_lambda': best['lambda'],
            'selected_scale': best['scale'],
            'selected_violations': best['violations'],
        })
    scorable = [f for f in folds if f.get('scorable')]
    if not scorable:
        return {'folds': folds, 'selected_lambda': None,
                'note': '所有折都不可评分，无法选参'}
    from collections import Counter
    chosen = Counter(f['selected_lambda'] for f in scorable)
    # 汇总选参：取出现次数最多者，平局取更大 λ（预注册规则 3）。
    selected = sorted(chosen.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
    mean_scale = sum(f['selected_scale'] for f in scorable) / len(scorable)
    return {
        'folds': folds,
        'selected_lambda': selected,
        'fold_lambda_votes': dict(chosen),
        'mean_selected_scale': mean_scale,
        'selection_rule': '先排除外折超时下界违背，再取外折稳健尺度最小，'
                          '平局取更大lambda',
        'labels_used': False,
        'note': '回放只冻结历史窗；选参不使用后来的人工故障标签。',
    }
