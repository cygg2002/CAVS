# 三数据集成本感知自动选择与风险门禁计划

## 阶段0：协议冻结

- [x] 新建独立实验目录。
- [x] 冻结8个可部署Ranker、3个动作策略和10个论文工作点。
- [x] 冻结参考方法、候选排序、fallback和final禁止事项。
- [x] 将Random限定为参考，不进入自动选择。

验证：规格中不存在final挑选、失败后换第二候选或修改既有实验的路径。

## 阶段1：只读审计与失败测试

- [x] 核验三数据集20260723 core完成状态、行数和输入哈希。
- [x] 核验COCO/Oregon audit资产；确认ENA24无兼容独立audit。
- [x] 写候选网格、参考选择、模式目标和tie-break失败测试。
- [x] 写风险门禁、fallback、final隔离和系统测量声明失败测试。

验证：新测试先因实现缺失而失败。

## 阶段2：ENA24严格划分

- [x] 对400张calibration原图计算16×16灰度dHash。
- [x] Hamming≤6进行union-find近重复分组。
- [x] 以整组为单位按稀有类别和样本量平衡selector/gate。
- [x] 写出图像哈希、component、类别分布与划分清单。

验证：component不跨分区；只能声明near-duplicate-group-disjoint。

## 阶段3：最小自动选择器

- [x] 实现从calibration curve提取3个冻结策略。
- [x] ENA24在selector半重新校ROI阈值、Online阈值和Fixed-byte lambda。
- [x] 实现每数据集、模式、预算的参考与 provisional 选择。
- [x] 输出10个工作点及可复算的候选表。
- [x] 验证选择过程没有读取final结果或final selections。

验证：纯函数测试通过；每数据集24候选×10工作点完整。

## 阶段4：风险门禁

- [x] 重建COCO/Oregon audit上的参考与provisional动作。
- [x] 在ENA24 gate半重建参考与provisional动作。
- [x] 运行200次配对bootstrap并保存checkpoint。
- [x] CI下界小于-0.001时只回退reference，不尝试第二候选。
- [x] Fixed-call/Online同时要求KB差CI上界≤0。

验证：每数据集10个门禁结果完整、无重复，动作相同点精确为0。

## 阶段5：冻结final评测

- [x] 在风险门禁完成并写出冻结选择哈希后才打开final结果。
- [x] 从20260723不可变结果中提取冻结方法对应行；ENA24按严格200/200划分重新评测240个final候选。
- [x] 核验对应final action map哈希。
- [x] 计算同24候选final oracle和regret，仅作诊断。

验证：每数据集恰好10行；冻结选择不因oracle改变。

## 阶段6：跨数据集消融

- [ ] 只用三数据集calibration选择每工作点一个全局配置。
- [ ] 在各数据集门禁并冻结。
- [ ] 比较dataset-adaptive、global-one-config、reference和final oracle。
- [ ] 统计AP、KB、E2E、fallback率、安全违反率和oracle regret。

验证：明确回答单一学习器/动作能否跨数据集稳定。

## 阶段7：真实系统测量

- [x] 审计本地Mac与远程GPU服务器是否能组成真实edge/cloud路径。
- [x] 核验模型、数据、网络控制、时间同步和能耗接口的可用性与缺口。
- [ ] 若条件齐全，独立运行组件与端到端测量。
- [x] 若条件不齐全，只输出阻塞项和可执行测量计划，不生成伪实测值。

验证：任何“measured E2E”都有真实host、模型、样本量、网络状态和原始日志。

## 阶段8：审查与报告

- [x] 运行完整测试和脚本编译。
- [x] 独立复算行数、唯一键、哈希、门禁和fallback。
- [x] 检查源实验运行前后哈希一致。
- [x] 生成中文结果和论文可用主表。
- [x] 运行代码质量与数据可信度审查。

验证：只有全部必需门禁通过才写总 `COMPLETED`。

## 并行边界

- 可并行：三数据集preflight、三数据集risk-gate、系统测量条件审计。
- 必须串行：协议冻结 → provisional选择 → 风险门禁 → final打开。
- 严禁并行越过：未写冻结选择哈希前读取final结果。

## 风险与控制

| 风险 | 控制 |
|---|---|
| ENA24缺少event/camera元数据 | dHash近重复组隔离；event/camera disjoint标记unknown |
| ENA24严格拆分导致阈值变化 | selector半重校全部阈值/lambda并重新评final |
| audit调度与final调度不一致 | 冻结Online阈值和Fixed-byte lambda，测试动作重建 |
| 自动选择偏向低KB但低AP | 先相对最佳Shared reference施加AP容忍 |
| gate失败后继续挑候选造成多重试探 | 固定只测一个provisional，失败直接reference |
| final oracle造成后验挑选 | oracle在冻结文件之后计算，字段标记diagnostic_only |
| 单机公式冒充真实E2E | 系统测量preflight失败即阻止measured声明 |
