"""共享网络路径体验劣化的定位分析：路径矩阵、可识别性、稳健估计与回放。

模块划分：
- paths：按探测样本的时间窗选择路径版本，展开完整有向路径（含环路重复经过）。
- design：组件-路径设计矩阵、可识别性诊断与等价类。
- baseline：仅用已确认健康窗口估计组件参考基线与探测源偏移。
- fit：延迟相加与超时单侧下界的联合稳健估计（L1 稀疏）。
- replay：冻结历史窗的时间外回放与预注册选参。
- recommend：依据候选路径残差给出额外探测建议。
- analysis：把以上串成一次定位报告的入口。
"""

from .analysis import analyse
from .io import DataError

__all__ = ['analyse', 'DataError']
