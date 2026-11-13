# 共享网络路径的体验劣化定位分析

本模块在既有资料读取与身份检查之上，提供**可证据化**的候选组件定位。
所有结论只来自给定的探测样本与路径版本；不发送网络请求，不读取人工故障
标签，也不把相关性热图当作根因。

## 1. 输入与时间语义

- 探测样本：起终点（source / path）、时间窗、路径版本、成功或超时、成功
  时的总时延、超时门槛 `timeout_ms`。
- 路径版本：`paths.components` 是按实际经过顺序排列的完整有向路径，允许
  重复；环路中同一组件按实际经过次数进入设计矩阵（不是 0/1）。
- 同一窗内可能路由切换：每个样本使用**它自己**的路径版本。若探测资料无法
  唯一确定版本（同一 path id 登记了多个版本），必须显式提供
  `probe_versions` 或窗口→版本计划，分析不会猜。
- 健康基线只在 `confirmed_at` 不晚于分析时点的健康窗估计。分析时点之后的
  健康确认不可见，避免把后来的结论泄漏进历史决策。

## 2. 可识别性诊断（先看这一节）

模型只能约束设计矩阵 A 的列空间，其中 A[k, c] 是组件 c 在样本 k 的完整
路径上的经过次数。报告给出：

- `rank`、`full_rank`、`null_space_dimension`：设计矩阵的秩与零空间。
- `null_space_directions`：不改变任何已观测路径的组件增量组合。
- `identical_column_pairs` / `indistinguishable_groups`：在所有路径上完全
  共现的组件。它们的个体增量不可分辨，只能给出带已知权重的组合量。
  报告**不会**给共现组内每个组件各报一个确定增量。
- `joint_path_source`：路径版本与探测源的二部覆盖图。若某路径版本只被一个
  源覆盖，路径基线与源偏移只能确定其和；跨连通分量的绝对水平不可比。
- `unobserved_components`：从未出现在任何已观测路径上的组件。

## 3. 模型假设

### 3.1 延迟相加模型

相对健康基线，单位 ms：

```
r_k = y_k - baseline(path_k, version_k, source_k)
    = Σ_g F[k,g] · β_g + g + u_source(k) + e_k
```

- `F[k,g]`：组 g 的代表路径在样本 k 上的经过次数（环路累加）。
- `β_g ≥ 0`：组增量（只建模“变慢”方向）。
- `g`：所有样本共享的整体水平。
- `u_source`：各源相对整体水平的漂移，强制 `Σu = 0`。

假设：组件增量线性可加。排队非线性、重试、连接复用等造成的残差只作为
**可观察路径残差**报告，不强行归到某个物理组件。报告在 `warnings` 中明确
写出这一点。

### 3.2 超时模型

`success=false` 的样本只知道 `y_k > T_k`。估计目标对成功样本用平方损失、
对超时样本用单侧 hinge：

```
max(T_k - (baseline_k + increment_k), 0)²
```

- 超时样本**不删除**（删除会让故障窗均值变小），
- **不把 T_k 当作精确观测**（否则会把拟合值钉在下界），
- 只出现在超时样本上的组，其估计值在 `lower_bound_only` 中单列为下界，
  不作为点估计，也不进入候选组件集合。

### 3.3 基线与源偏移

- 路径基线按 `(path, version)` 稳健估计（IRLS + Tukey 权），健康窗内的
  路由切换不会把两个版本混进同一均值。
- 源偏移只在健康窗估计，故障窗不再重估；故障窗的源漂移 `u` 被 L1 约束，
  且中心化为零，避免把共同异常吸收成源偏移。
- 投影响应区分 `direct` / `path_plus_offset` / `group_projected` /
  `unavailable`。没有可信基线的样本登记为 `missing_baseline` 并说明原因，
  **绝不静默插补**。

### 3.4 稀疏与稳健

- 组系数与源漂移施加 L1；λ 由冻结历史窗的时间外回放预注册选择，
  不使用人工故障标签。
- 分量上界：组件在单个观测单元上的贡献不超过“观测最大增量 × 2”（预注册
  放松倍数）。它只排除数值上荒谬的不可识别方向，不是物理因果约束。
- 成功残差用 Tukey 双权降权，但设有门控：若多数样本系统性变慢，保持均权，
  避免把真实故障当离群洗掉。
- 近似标准误来自加权正规矩阵的伪逆对角线；不可识别的方向报告为 `None`，
  不用岭项造出虚假的小标准误。

## 4. 等价解释与稀疏先验依赖

`support_enumeration` 在**与主模型相同的参数化**下枚举 |S| ≤ 2 的组支撑：
每个支撑都带整体水平与源漂移作为干扰项，超时用单侧 hinge。于是
“空支撑 + 整体水平”是合法解释。

- `evidence_against_null`：组件支撑是否显著优于空支撑。
- `equivalence_class`：容差带内互不包含的不同支撑。
- `relative_loss_gap_to_second`：最佳解释与真正不同的次优解释的损失差。
  差距小说明最小支撑的选择主要依赖稀疏先验，报告在
  `candidates.sparse_prior_dependence` 中标注为 `high`。
- 枚举预算耗尽时，`unresolved_count` 保留尚未排除的支撑数；
  **未评估 ≠ 已排除**。

候选状态：

| status | 含义 |
|---|---|
| `single_component` | 定位到一个组件 |
| `indistinguishable_set` | 只能定位到一组不可区分组件或等价解释 |
| `common_offset_only` | 只能定位到共同偏移（源漂移 / 全体同步上调） |
| `no_evidence` | 不足以区分故障与无故障 |
| `lower_bound_only` | 目标窗只有超时，只能给下界 |
| `insufficient_baseline` | 没有可用健康基线 |

## 5. 时间外回放

`replay` 按时间窗切折：每折只用更早的窗建基线并在该折上评分。选参规则
预注册：

1. 先排除在外折出现超时下界违背的 λ；
2. 在剩余 λ 中取外折成功样本的稳健尺度（median absolute residual）最小；
3. 平局取更大的 λ（更稀疏、更保守）。

回放只报告样本外指标，`labels_used=false`。

## 6. 可观察路径残差

`observed_path_residuals` 逐 `(path, version, source)` 给出健康窗与故障窗
的稳健位置差。这是**无模型**的交叉检查：若模型报“组件 c 异常 40ms”，
但同一路径同源几乎没变，这两个结论不能同时成立。含超时的组合只报
`>T` 下界，不折算成精确差分。

## 7. 额外探测建议

`recommendations` 只针对调用方给定的候选路径（`--candidate-path`）与源，
计算“加入后组件设计矩阵的秩增益 / 零空间缩减”。建议**不发送网络请求**，
也不新增未给定的路径。若给定候选无法拆开当前不可区分方向，明确返回
`given_candidates_can_resolve=false` 与 `limitation`。

## 8. 使用

```sh
python -m path_data.cli data/normal.json
python -m path_data.cli data/boundary.json \
  --candidate-path p1 1 --candidate-source a
python -m path_data.cli data/normal.json --analysis-window 1 --no-replay --lambda 1.0
```

在 Python 中：

```python
from path_data import analyse
from path_data.io import load
report = analyse(load('data/normal.json'), {'lambda': 1.0})
```

## 9. 报告字段速查

- `identifiability`：秩、零空间、共现组、路径版本-源覆盖图。
- `baseline`：基线方法、逐路径版本基线、源偏移、投影可用性与原因。
- `fit`：组增量、整体水平、源漂移、近似标准误、不可识别方向、
  成功样本重拟合结果、`lower_bound_only`。
- `candidates`：`status`、候选组件集合、每组件估计、共同偏移、
  等价解释、稀疏先验依赖等级。
- `observed_path_residuals`：逐 `(path, version, source)` 的无模型差分。
- `support_enumeration`：支撑评估、容差带、未评估计数、噪声标定。
- `replay`：每折候选 λ、选中的 λ、预注册规则、`labels_used`。
- `recommendations`：给定候选路径上的秩增益与零空间缩减。
- `warnings`：所有不可识别性、超时、缺基线、模型假设与预算限制。

