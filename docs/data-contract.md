# 业务资料约定

paths.components为按实际经过顺序列出的组件ID，允许重复；path引用完整路径版本。success=false时latency_ms=null，仅有timeout_ms下界。window为固定观察窗序号，confirmed_at为健康确认可见窗。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- components：id。
- paths：id、version、components。
- probes：id、source、window、path、success、latency_ms、timeout_ms。
- healthy：id、window、confirmed_at。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。
