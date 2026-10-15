# 三数据集九Ranker分层三动作论文级任务计划

## 冻结原则

- 新实验目录独立，不修改现有 canonical、ENA24 8-Ranker、Legacy control
  或 Legacy paper evaluation。
- 先通过兼容性和复现门禁，再运行新矩阵。
- 三数据集算法规则一致；模型参数和阈值按数据集冻结。
- audit/final 不参与模型、阈值、lambda、fallback 或方法选择。
- final 失败原样保留。

## 阶段0：确定第九Ranker语义

- [x] 审计ENA24、COCO、Oregon历史32D模型与当前canonical检测器/FPN。
- [x] 确认ENA24完全兼容。
- [x] 确认COCO存在896D YOLO26s与当前448D YOLO26n硬不兼容。
- [x] 确认Oregon存在旧location模型与当前event-disjoint模型哈希不一致。
- [x] 用户冻结“Controlled-32D主矩阵 + Legacy桥接附录”。

验证：方法名称、检测器成本、是否进入共享阈值约束均无歧义。

## 阶段1：协议与失败测试

- [x] 写最终 `configs/protocol.json`。
- [x] 写矩阵规模、Ranker集合和Random排除测试。
- [x] 写模型/FPN维数与SHA兼容性失败测试。
- [x] 写动作极值恒等式和Top-K选择集合测试。
- [x] 写Fixed-byte重校lambda合同测试。

验证：测试首先因为实现缺失而失败。

## 阶段2：最小通用引擎

- [x] 实现数据集source-root适配器。
- [x] 复用现有8-Ranker score和三动作评测函数。
- [x] 实现获批的第九Ranker适配器。
- [x] 实现数据集隔离输出、输入hash和completion marker。

验证：纯函数测试、编译、三个dataset preflight全部通过。

## 阶段3：复现门禁

- [ ] COCO完整复现当前8-Ranker SharedAction。
- [ ] ENA24完整复现当前8-Ranker SharedAction。
- [ ] Oregon完整复现当前8-Ranker SharedAction。
- [ ] ENA24 Current-Controlled-32D复现已完成32D对照结果。
- [ ] 对COCO/Oregon记录获批第九Ranker的兼容类别和来源hash。

验证：核心指标最大差值不超过 `1e-12`；任何差异停止后续运行。

## 阶段3b：Legacy桥接附录

- [ ] ENA24精确复用已完成Legacy-Frozen-32D结果并核对哈希。
- [ ] COCO记录896D YOLO26s与当前448D YOLO26n不兼容；若运行历史输入桥接，单列检测器成本。
- [ ] Oregon记录旧checkpoint/split/taxonomy与当前协议不兼容；不把迁移结果加入主表。

验证：Legacy不进入共享阈值、跨数据集平均排名或32D-vs49D主结论。

## 阶段4：Fixed-call风险优先切片

- [ ] 先在calibration评估8个非随机Ranker × 10%/25%/40%。
- [ ] 选择每数据集共享ROI阈值并冻结per-Ranker消融阈值。
- [ ] 在final运行三个主预算并保留失败。
- [ ] 扩展至5%/15%/50%/65%/80%。
- [ ] 运行Random 50 seeds。

验证：每数据集Fixed-call为216个聚合主结果，Random raw为1,200行。

## 阶段5：Online与Fixed-byte

- [ ] calibration冻结每Ranker/target Online frame threshold。
- [ ] calibration选择共享和per-Ranker Online ROI阈值。
- [ ] 每policy/candidate/Ranker/budget重校Fixed-byte lambda。
- [ ] 在final运行Online与Fixed-byte完整矩阵。

验证：每数据集主矩阵共384个唯一工作点；Fixed-call/Online策略间
入选图片ID一致。

## 阶段6：并行运行与Bootstrap

- [ ] 三个数据集按独立输出目录并行运行core。
- [ ] core全部校验后并行运行bootstrap。
- [ ] 每数据集24个比较点，每点200次。
- [ ] checkpoint支持安全续跑，selection/protocol hash不一致时拒绝复用。

验证：三数据集合计72个唯一bootstrap比较；无丢失、无重复。

## 阶段7：验证、审查与报告

- [ ] 独立复算行数、唯一键、动作比例、AP差、KB差和安全通过数。
- [ ] 核对全部冻结输入运行前后SHA-256。
- [ ] 运行完整单元测试、脚本编译和代码审查。
- [ ] 生成三数据集完整中文表、跨数据集平均排名和负结果说明。
- [ ] 明确COCO/ENA24 diagnostic与Oregon post-hoc final限制。

验证：只有全部门禁通过才写总 `COMPLETED`。

## 并行边界

- 可以并行：三个数据集core、三个数据集bootstrap、只读数据审计。
- 必须串行：协议冻结、通用代码合同、每数据集SharedAction复现门禁。
- 不允许：三个数据集各自调整AP容忍、预算、动作定义或结果回退。

## 主要风险

| 风险 | 影响 | 控制 |
|---|---|---|
| 历史Legacy检测器与current不同 | Ranker对比不公平 | Controlled-32D主矩阵或明确transfer附录 |
| Legacy加入共享约束导致Infinity | 主策略0 ROI | 原样报告；per-Ranker阈值只作消融 |
| COCO bootstrap耗时长 | 运行数小时 | 独立checkpoint、数据集并行、相同动作对复用 |
| Oregon final已访问 | 不能称新隐藏测试 | 标记post-hoc exploratory，禁止final选择 |
| Fixed-byte动作改变入选集合 | 不能与Fixed-call直接归因 | 每policy重新校lambda并单独报告 |
