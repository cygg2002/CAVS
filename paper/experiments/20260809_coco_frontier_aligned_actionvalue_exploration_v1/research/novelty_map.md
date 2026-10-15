# Action-Frontier Path Distillation：文献边界与创新审计

检索日期：2026-08-09。本文档只记录本轮定向检索能支持的判断；“未检索到”等于有限检索结论，不等于不存在证明。

## 已确认事实

- ORIC/MORIC 所在工作将端侧检测结果映射为逐图上传收益，用于资源受限的检测卸载；其公开摘要把问题表述为根据本地检测估计单图 reward，再决定哪些图片上传。[Optimizing Edge Offloading Decisions for Object Detection](https://arxiv.org/abs/2410.18919)
- DCSB 使用 difficult-case discriminator 在“本地处理/上传完整图片”间做图像级决策，并非多动作、按字节容量联合调度。[Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator](https://arxiv.org/abs/2108.12858)
- SPO/SPO+ 已明确提出让预测训练直接感知下游优化结构。因此，“给 ActionValue 加一个优化相关 loss”本身不是创新。[Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005)
- 决策学习也已有 pointwise、pairwise、listwise 的 feasible-solution 排序视角。因此，“跨候选排序”本身也不是创新。[Decision-Focused Learning: Through the Lens of Learning to Rank](https://arxiv.org/abs/2112.03609)
- 近年的方法已经把 Lagrangian decomposition、shadow price、可微 knapsack 用于受约束决策训练或资源分配；单纯引入 `reward - lambda * cost` 或可微背包同样不能构成独立创新。[Scaling Decision-Focused Learning to Large Problems with Lagrangian Decomposition](https://arxiv.org/abs/2606.08797)、[Differentiable Knapsack and Top-k Operators via Dynamic Programming](https://arxiv.org/abs/2601.21775)、[The Shadow Price of Reasoning](https://arxiv.org/abs/2606.03092)
- SmartChoices 证明“context + 可选动作 + 反馈”的 learned decision abstraction 已有通用系统实现，所以不能把普通多动作 contextual policy 当作新贡献。[SmartChoices: Augmenting Software with Learned Implementations](https://arxiv.org/abs/2304.13033)

## 本轮未检索到的同构方法

在上述相邻方向以及针对 `parametric solution path distillation`、`upper-envelope distillation`、`shadow-price path action selection`、`multi-action edge-cloud offloading` 的定向查询中，尚未检索到同时满足以下四点的方法：

1. 学习器仍输出每个图像—动作的静态、成本无关 value；
2. 训练 teacher 是该图像所有动作线 `reward_a - lambda * bytes_a` 的完整参数化上包络；
3. 在所有成对动作交点的前后显式监督动作切换，而不是只在一个预算或一个 lambda 上优化；
4. Original、Q90、ROI 与 Edge 在同一 envelope 中竞争，已支配 ROI 自动退出路径。

因此，本轮方法可以暂称 **Action-Frontier Path Distillation (AFPD)**。严谨表述应是“相对本轮检索语料具有结构差异”，不能声称绝对首创。

## 推测与待证伪机制

- 推测：当前 ActionValue 的主要错误来自不同成本动作的值尺度没有保留正确交点，而不只是检测集合编码器容量不足。
- 推测：在每个动作交点前后蒸馏完整路径，会减少固定字节调度中的 Edge→Q90 误升级和 Original/Q90 错配。
- 推测：ROI 的有效样本虽然稀疏，但若它在某段 shadow-price 区间位于上包络，路径监督比独立 ROI safe 分类更容易保留它；被其他动作支配的 ROI 不应得到额外正偏置。

这些都不是结论，只有严格 OOF 结果可以支持或否定。

## 主观研究判断

- 先固定现有集合编码器，只改变监督目标，是当前最有判别力的最小实验。
- 如果 AFPD 只提升 10% 而在 25%/40% 退化，应优先检查交点/动作尺度，不应继续堆 Transformer。
- 如果 OQR Teacher 明显高于 OQ、但 Learned OQR 仍不提升，下一轮应研究“集合值/Top-k 候选动作”或反馈式 ROI，而不是再做一个静态 ROI 二分类门。
- 如果 OQ、OQR 两条路径均未通过预注册 gate，应停止从同一批静态 Edge 框继续压榨新 Ranker，并转向引入时序或云端反馈的新信息。
