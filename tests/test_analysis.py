"""定位分析的行为测试：可识别性、超时、路由切换、源偏移与回放。"""
import json
import unittest

from path_data.analysis import analyse
from path_data.io import DataError
from path_data.paths import prepare_rows
from path_data.replay import replay
from tests import scenarios


def run(tables, **options):
    opts = {'replay': False, 'lambda': 0.5}
    meta = options.pop('meta', {}) or {}
    if meta.get('probe_versions'):
        opts['probe_versions'] = meta['probe_versions']
    opts.update(options)
    return analyse(tables, opts)


class CommonUpstreamTests(unittest.TestCase):
    def test_locates_single_shared_component(self):
        tables, _ = scenarios.common_upstream()
        report = run(tables)
        self.assertEqual(report['candidates']['status'], 'single_component')
        self.assertEqual(report['candidates']['component_set'], ['c'])
        self.assertEqual(report['candidates']['interpretation'], '定位到一个组件')
        # 组件增量应接近真实增量 40ms
        estimate = report['candidates']['component_estimates']['c']
        self.assertAlmostEqual(estimate['group_beta'], 40, delta=3)
        # 整体水平不应被当作组件
        self.assertFalse(report['candidates']['common_offset']['significant'])

    def test_path_without_shared_component_stays_healthy(self):
        tables, _ = scenarios.common_upstream()
        report = run(tables)
        residuals = {(r['path'], r['version']): r for r in report['observed_path_residuals']}
        self.assertLess(abs(residuals[('p4', 1)]['observed_difference']), 3)
        self.assertGreater(residuals[('p1', 1)]['observed_difference'], 30)


class IdenticalPathTests(unittest.TestCase):
    def test_reports_group_not_individual_components(self):
        tables, _ = scenarios.identical_path_components()
        report = run(tables)
        candidates = report['candidates']
        self.assertEqual(candidates['status'], 'indistinguishable_set')
        self.assertEqual(candidates['indistinguishable_groups'], [['x', 'y']])
        self.assertEqual(sorted(candidates['component_set']), ['x', 'y'])
        for c in ('x', 'y'):
            self.assertFalse(
                candidates['component_estimates'][c]['identifiable_individually'])
        diag = report['identifiability']
        self.assertIn(['x', 'y'], diag['identical_column_pairs'])
        self.assertGreaterEqual(diag['null_space_dimension'], 1)


class SourceOffsetTests(unittest.TestCase):
    def test_does_not_blame_a_component_for_source_drift(self):
        tables, _ = scenarios.source_offset()
        report = run(tables)
        candidates = report['candidates']
        self.assertEqual(candidates['status'], 'common_offset_only')
        self.assertEqual(candidates['component_set'], [])
        drift = candidates['common_offset']['source_drift']
        self.assertGreater(abs(drift.get('src1', 0)), 10)
        # 不应出现任何组件异常
        for group in report['fit']['groups']:
            self.assertLess(group['beta'], 2)

    def test_recommends_source_side_is_not_needed_for_timeout(self):
        tables, _ = scenarios.source_offset()
        report = run(tables)
        self.assertTrue(any('源漂移' in w for w in report['warnings']))


class TimeoutRoutingTests(unittest.TestCase):
    def test_timeout_is_lower_bound_not_exact(self):
        tables, meta = scenarios.timeout_and_routing()
        report = run(tables, meta=meta)
        self.assertTrue(any('超时' in w for w in report['warnings']))
        # 超时样本要被保留，不能被删除
        timeout_probes = report['support_enumeration']['timeout_probes']
        self.assertTrue(timeout_probes)
        target_timeouts = {p['id'] for p in tables['probes']
                           if not p['success'] and p['window'] <= 7}
        self.assertEqual(set(timeout_probes), target_timeouts)
        # 下界违背不应发生
        self.assertEqual(
            report['fit']['summary']['timeout_lower_bound_violations'], 0)

    def test_routes_switch_inside_window(self):
        tables, meta = scenarios.timeout_and_routing()
        report = run(tables, meta=meta)
        self.assertEqual(report['route_switches'], {'p1': [1, 2]})
        rows = prepare_rows(tables, meta['probe_versions'], analysis_window=6)
        p1 = [r for r in rows if r['path'] == 'p1']
        self.assertEqual({r['version'] for r in p1}, {1, 2})

    def test_route_without_healthy_baseline_not_silently_dropped(self):
        tables, meta = scenarios.timeout_and_routing(new_version_in_healthy=False)
        report = run(tables, meta=meta, analysis_window=6)
        missing = report['fit'].get('missing_baseline') or []
        self.assertTrue(missing, '窗内新版本没有健康基线，必须登记为缺失而不是静默插补')
        for item in missing:
            self.assertIn('reason', item)
        candidates = report['candidates']
        # 缺基线的样本没有被硬塞一个插补值：它们出现在 missing_baseline 中，
        # 报告同时说明这会让结论变弱，而不是静默给出单组件结论。
        self.assertTrue(any('缺少可用健康基线' in w for w in report['warnings']))


class LoopTests(unittest.TestCase):
    def test_repeated_component_counted_by_visit(self):
        tables, _ = scenarios.loop_path()
        rows = prepare_rows(tables)
        p1 = next(r for r in rows if r['path'] == 'p1')
        self.assertEqual(p1['counts']['gw'], 2)
        report = run(tables)
        self.assertEqual(report['candidates']['component_set'], ['gw'])
        estimate = report['candidates']['component_estimates']['gw']
        # 12ms 总增量由 2 次经过分摊 → 每次约 6ms
        self.assertAlmostEqual(estimate['group_beta'], 6, delta=1.5)


class IdentifiabilityTests(unittest.TestCase):
    def test_unknown_path_version_rejected(self):
        tables, _ = scenarios.timeout_and_routing()
        with self.assertRaisesRegex(DataError, '版本'):
            prepare_rows(tables)  # 未提供版本映射

    def test_bad_success_latency_rejected(self):
        tables, _ = scenarios.common_upstream()
        tables = json.loads(json.dumps(tables))
        tables['probes'][0]['timeout_ms'] = 1
        with self.assertRaisesRegex(DataError, '超时'):
            prepare_rows(tables)

    def test_visible_healthy_windows_respect_time(self):
        tables, _ = scenarios.common_upstream()
        # 健康确认可见窗是 3；分析时点 2 时该确认还不可见。
        report = analyse(tables, {'replay': False, 'lambda': 0.5,
                                  'analysis_window': 2})
        self.assertEqual(report['healthy_windows_visible'], [])
        self.assertEqual(report['candidates']['status'], 'insufficient_baseline')
        report3 = analyse(tables, {'replay': False, 'lambda': 0.5,
                                   'analysis_window': 3})
        self.assertEqual(report3['healthy_windows_visible'], [1, 2, 3])


class ReplayTests(unittest.TestCase):
    def test_replay_never_uses_labels(self):
        tables, _ = scenarios.common_upstream()
        result = replay(tables, ['c', 'a', 'b', 'd'])
        self.assertFalse(result['labels_used'])
        self.assertIn('selection_rule', result)
        self.assertIn('folds', result)

    def test_selected_lambda_feeds_analysis(self):
        tables, _ = scenarios.common_upstream()
        report = analyse(tables, {'replay': True})
        self.assertIsNotNone(report['replay'])
        self.assertEqual(report['fit']['lambda'], report['replay']['selected_lambda'])


class RecommendationTests(unittest.TestCase):
    def test_recommendation_only_on_given_paths(self):
        tables, _ = scenarios.identical_path_components()
        report = run(tables, candidate_paths=[('p1', 1, ['x', 'y'])],
                     candidate_sources=['s1', 's2'])
        recs = report['recommendations']
        self.assertEqual({s['action'] for s in recs['suggestions']},
                         {'add_probe_on_given_path'})
        self.assertFalse(recs['given_candidates_can_resolve'])
        self.assertTrue(recs['limitation'])

    def test_recommendation_prefers_discriminating_path(self):
        tables, _ = scenarios.identical_path_components()
        # 追加一条只经过 x 的候选路径应能拆开 x/y。
        paths = json.loads(json.dumps(tables))
        report = run(paths, candidate_paths=[('p1', 1, ['x', 'y']),
                                             ('new', 1, ['x'])],
                     candidate_sources=['s1'])
        top = report['recommendations']['suggestions'][0]
        self.assertEqual(top['path'], 'new')
        self.assertGreater(top['expected_component_rank_gain'], 0)


class BudgetAndSparseTests(unittest.TestCase):
    def _two_fault_tables(self):
        import random
        rng = random.Random(3)
        components = ['a', 'b', 'c', 'd']
        paths = [
            {'id': 'p1', 'version': 1, 'components': ['a', 'c']},
            {'id': 'p2', 'version': 1, 'components': ['b', 'd']},
            {'id': 'p3', 'version': 1, 'components': ['a']},
            {'id': 'p4', 'version': 1, 'components': ['b']},
        ]
        baseline = {'a': 5, 'b': 7, 'c': 6, 'd': 4}
        probes, n = [], 0
        for window in range(1, 8):
            for path in paths:
                if window == 4 and path['id'] == 'p2':
                    continue  # 缺路径样本
                for _ in range(3):
                    n += 1
                    value = sum(baseline[x] for x in path['components']) + rng.gauss(0, 1)
                    if window >= 5:
                        if 'a' in path['components']:
                            value += 20
                        if 'b' in path['components']:
                            value += 30
                    probes.append(scenarios._probe(
                        f's{n}', 's1', window, path['id'], True,
                        round(max(value, 1), 3), 300))
        healthy = [{'id': f'h{w}', 'window': w, 'confirmed_at': 4}
                   for w in (1, 2, 3, 4)]
        return scenarios.make_tables(components, paths, probes, healthy)

    def test_two_sparse_faults_and_missing_path_are_handled(self):
        tables = self._two_fault_tables()
        report = run(tables, lam=0.0)
        self.assertEqual(report['candidates']['status'], 'indistinguishable_set')
        self.assertIn('a', report['candidates']['component_set'])
        self.assertIn('b', report['candidates']['component_set'])
        # 缺路径样本不应让分析崩溃；observability 仍逐路径给出
        observed = {r['path'] for r in report['observed_path_residuals']}
        self.assertTrue({'p1', 'p2', 'p3', 'p4'} <= observed)

    def test_budget_exhaustion_keeps_unresolved(self):
        tables = self._two_fault_tables()
        report = run(tables, max_support_evaluations=2,
                     max_support_size=2)
        support = report['support_enumeration']
        self.assertTrue(support['budget_exhausted'])
        self.assertGreater(support['unresolved_count'], 0)
        self.assertTrue(any('预算耗尽' in w for w in report['warnings']))

    def test_equivalent_interpretations_are_listed(self):
        tables = self._two_fault_tables()
        report = run(tables, lam=0.0)
        supports = report['candidates']['equivalent_supports']
        self.assertGreaterEqual(len(supports), 2)
        labels = {tuple(sorted(tuple(g) for g in item['support_groups']))
                  for item in supports}
        self.assertGreaterEqual(len(labels), 2)


class ReportShapeTests(unittest.TestCase):
    def test_report_is_json_serializable(self):
        tables, _ = scenarios.common_upstream()
        report = run(tables)
        json.dumps(report, ensure_ascii=False)

    def test_linear_additivity_assumption_is_stated(self):
        tables, _ = scenarios.common_upstream()
        report = run(tables)
        self.assertTrue(any('线性可加' in w for w in report['warnings']))
        self.assertTrue(any('排队' in w for w in report['warnings']))

    def test_sparse_prior_dependence_is_labelled(self):
        tables, _ = scenarios.identical_path_components()
        report = run(tables)
        dependence = report['candidates']['sparse_prior_dependence']
        self.assertIn('level', dependence)
        self.assertIn('note', dependence)


if __name__ == '__main__':
    unittest.main()
