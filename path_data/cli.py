"""命令行入口：资料完整性检查 + 定位分析。"""
import argparse
import json

from .analysis import analyse
from .domain import inspect_tables
from .io import DataError, load


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='共享网络路径体验劣化的定位分析（只读本地资料，不发送网络请求）')
    parser.add_argument('manifest')
    parser.add_argument('--analysis-window', type=int, default=None,
                        help='只使用不晚于该窗的资料（默认取所有探测窗的最大值）')
    parser.add_argument('--lambda', dest='lam', type=float, default=None,
                        help='固定稀疏强度；默认由冻结历史窗回放预注册选择')
    parser.add_argument('--max-support-size', type=int, default=2)
    parser.add_argument('--max-support-evaluations', type=int, default=400)
    parser.add_argument('--support-tolerance-ratio', type=float, default=0.05)
    parser.add_argument('--candidate-path', action='append', nargs=2,
                        metavar=('PATH_ID', 'VERSION'),
                        help='用于生成额外探测建议的候选路径版本，可重复')
    parser.add_argument('--candidate-source', action='append', default=None)
    parser.add_argument('--max-suggestions', type=int, default=3)
    parser.add_argument('--no-replay', action='store_true')
    parser.add_argument('--probe-version', action='append', nargs=2,
                        metavar=('PROBE_ID', 'VERSION'),
                        help='探测行无法唯一确定路径版本时的手工映射，可重复')
    parser.add_argument('--schema-only', action='store_true',
                        help='只做资料完整性检查，不运行分析')
    args = parser.parse_args(argv)
    try:
        tables = load(args.manifest)
        inspection = inspect_tables(tables)
        if args.schema_only:
            print(json.dumps(inspection, ensure_ascii=False, indent=2))
            return
        candidate_paths = None
        if args.candidate_path:
            by_version = {}
            for row in tables['paths']:
                by_version[(row['id'], row['version'])] = row['components']
            candidate_paths = []
            for path_id, version in args.candidate_path:
                version = _coerce(version)
                key = (path_id, version)
                if key not in by_version:
                    raise DataError(f'候选路径版本不存在: {key}')
                candidate_paths.append((path_id, version, by_version[key]))
        options = {
            'analysis_window': args.analysis_window,
            'lambda': args.lam,
            'max_support_size': args.max_support_size,
            'max_support_evaluations': args.max_support_evaluations,
            'support_tolerance_ratio': args.support_tolerance_ratio,
            'replay': not args.no_replay,
            'candidate_paths': candidate_paths,
            'candidate_sources': args.candidate_source,
            'max_suggestions': args.max_suggestions,
            'probe_versions': {p: _coerce(v) for p, v in (args.probe_version or [])},
        }
        report = analyse(tables, options)
        report['inspection'] = inspection
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    except (DataError, ValueError, KeyError) as exc:
        parser.exit(2, f'{exc}\n')


def _coerce(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    return str(value)


if __name__ == '__main__':
    main()
