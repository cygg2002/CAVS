# V2 中文初稿计划

## 阶段 1：固定论点和证据边界

- [x] 确定题目、主问题和 CAVS 方法名。
- [x] 固定 COCO/Oregon、Fixed-byte 主协议与 Original/Q90 主动作空间。
- [x] 固定 post-hoc、诊断与待完成结果的措辞边界。

## 阶段 2：撰写方法前半部分

- [x] 标题、摘要、关键词。
- [x] 引言：Top-K 局限、真实字节异质性、核心观察与贡献。
- [x] 相关工作：选择性卸载、任务感知传输、预算决策。
- [x] 问题定义：动作、成本、预算与 Top-K 特例。
- [x] CAVS：检测集合编码、contextual reward、动作价值学习和调度。

## 阶段 3：撰写实验和结果

- [x] 数据集、数据角色、检测器与载荷。
- [x] 外部基线、内部消融、统计协议。
- [x] COCO 已核验结果与机制消融。
- [x] Oregon canonical 结果与 Cloud-V3 三种子点结果主表；bootstrap 区间继续占位。
- [x] ROI、AFPD/Occupancy 与 teacher ceiling 的边界分析。
- [x] RK3588 真机实现与网络实验占位章节。

## 阶段 4：讨论和附录

- [x] 理论与经验结果的对应关系。
- [x] 泛化、失败模式、有效性威胁和伦理/复现声明。
- [x] 结论。
- [x] Top-K 特例证明、实验完整表与真机测量清单附录。
- [x] 参考文献。

## 完成检查

- [x] 论文各标准章节均有完整文字，不以提纲代替正文。
- [x] 未完成结果全部明确留空。
- [x] 现有数字有来源且协议不混用。
- [x] 创新表述为组合创新，不把既有通用思想包装成首创。
- [x] 生成集中占位符清单，便于后续补实验。

## 2026-08-14 V2.1 结果同步

- [x] 将 COCO 主方法固定为独立训练的 CAVS-OQ-Core，避免与 Raw ActionValue 和 OQR-OQMask 混名。
- [x] 更新 COCO 三种子外部基线、同骨干 Joint/TwoStage 和 CostInValue 点结果。
- [x] 更新 Current/Historical × Reversible/MAUR 严格 2×2 点结果与同选择融合区间。
- [x] 写入 Oregon Cloud-V3 独立 OQ、内部基线、ORIC/MORIC、Confidence、Random 的 Fixed-byte/Fixed-call 点结果。
- [ ] 回填正在运行的 COCO/Oregon bootstrap 与 2×2 v2 置信区间。
- [ ] 回填 RK3588 真机、多带宽和新注册确认集结果。
