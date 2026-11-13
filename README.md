# 共享网络路径的体验劣化定位与可识别诊断

本项目保存业务分析所需的本地资料，并在此基础上提供**可证据化**的候选组件
定位：按探测样本的完整有向路径、健康基线与联合稳健估计，给出候选组件集合、
不可区分等价解释、可观察路径残差以及下一条最有区分价值的额外探测建议。
分析只读本地资料，不发送网络请求，也不使用人工故障标签选参。

先读可识别性诊断，再读候选结论——两者缺一不可。方法与假设见
[docs/analysis.md](docs/analysis.md)。

## 资料

资料表包含 components、paths、probes、healthy。字段单位和关系见
[docs/data-contract.md](docs/data-contract.md)。data/normal.json 登记常规
资料，data/boundary.json 登记特殊业务情形，data/broken-digest.json 用于
检查资料损坏的拒绝行为。JSONL 文件按清单内 SHA256 和字节长度绑定。

## 运行

运行环境为 Python 3.11 及以上，仅使用标准库，无需额外服务。

```sh
python -m unittest discover -s tests -v
python -m compileall -q path_data tests
python -m path_data.cli data/normal.json
python -m path_data.cli data/boundary.json
```

只做资料完整性检查、不运行分析：

```sh
python -m path_data.cli data/normal.json --schema-only
```

指定分析时点、固定 λ、或生成额外探测建议：

```sh
python -m path_data.cli data/normal.json --analysis-window 1 --no-replay --lambda 1.0
python -m path_data.cli data/normal.json --candidate-path p1 1 \
  --candidate-source a --max-suggestions 3
```

容器内使用同样的检查入口：

```sh
docker build -t path-data .
docker run --rm path-data
```

## 包结构

- `io.py`：按清单摘要读取资料。
- `domain.py`：表集合与引用检查。
- `paths.py`：按样本解析路径版本、展开完整有向路径（环路按经过次数）。
- `design.py`：组件-路径设计矩阵、可识别性诊断、等价类。
- `baseline.py`：仅用可见健康窗估计路径基线、组基线与源偏移。
- `fit.py`：延迟相加 + 超时单侧下界的稳健稀疏估计。
- `support.py`：小支撑枚举与等价解释。
- `replay.py`：冻结历史窗的时间外回放与预注册选参。
- `recommend.py`：给定候选路径上的额外探测建议。
- `observed.py`：无模型的可观察路径残差。
- `analysis.py` / `cli.py`：报告入口。
