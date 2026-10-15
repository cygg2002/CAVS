# CAVS 学习器架构与 COCO/Oregon 主结果概览

## 方法：Contextual Action-Value Scheduler（CAVS）

CAVS 面向固定上行字节容量下的端云目标检测。端侧检测器先为每张图像产生 Top-64 检测集合；每个检测框由类别、位置/尺度、置信度及图像级统计共同表征。学习器对每个图像-动作对直接预测其**上下文 AP50-95 价值**：

\[
V_{i,a}=f_\theta(\mathcal{D}_i,a),\qquad
\max_x\sum_{i,a}x_{i,a}V_{i,a}\quad
\text{s.t.}\quad\sum_{i,a}x_{i,a}b_{i,a}\leq B,\quad\sum_a x_{i,a}\leq1.
\]

其中 \(\mathcal{D}_i\) 是端侧检测集合，\(a\in\{\text{Edge},\text{Original},\text{Q90},\text{ROI}\}\)，\(b_{i,a}\) 是真实 payload 字节。三种子模型的分数集成后，由 exact fixed-byte scheduler 在**图像是否上传**与**上传动作**之间联合决策。

这与两阶段方法不同：`SharedAction` 先给图片一个标量分数、决定上传成员，再选择 Original/Q90/ROI；CAVS 则让每个 `(图片, 动作)` 直接竞争。因此，Q90 的低字节成本、Original 的高保真价值以及 ROI 的局部价值都在同一个容量约束中比较。

### O / OQ / OQR 与 MAUR

动作空间缩写只描述可用的**云端动作**；`Edge`（不上云，保留端侧结果）始终存在。

| 变体 | 可用动作 | 消融问题 |
|---|---|---|
| `CAVS-O` | Edge、Original | 只允许全分辨率整图上传时，CAVS 的基础能力如何？ |
| `CAVS-OQ` | Edge、Original、Q90 | Q90 的压缩动作能否改善精度—字节折中？ |
| `CAVS-OQR` | Edge、Original、Q90、Current@640 ROI | 在 OQ 之外加入 ROI 后，是否还有可学习的额外收益？ |

`MAUR` 是 **Merge-After-Upload-Refinement**：ROI 云端框先映射回原图坐标，再按 IoU 匹配与端侧检测进行上传后的细化合并；ROI 外的端侧检测保持保留，最后统一 NMS。因而 MAUR 只在选择 ROI 时执行，Original 和 Q90 直接使用云端整图检测结果。

## Fixed-byte 主结果

指标为 Official AP50-95；容量为所有图像均上传 Original 时总字节的 10% / 25% / 40%。同一数据集、同一预算列内的方法使用相同评估器和近乎相同的实际 KB/图；Random 为 50 个冻结随机选择的均值。

### COCO val2017（5,000 张）

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR + MAUR（我们，Joint-OQR）** | **0.480277** | **0.532900** | **0.560178** |
| CAVS-OQ（动作空间消融，Joint-OQ） | 0.478224 | 0.528884 | 0.559286 |
| ORIC | 0.449588 | 0.495165 | 0.526971 |
| MORIC | 0.452755 | 0.496918 | 0.528256 |
| Confidence | 0.447376 | 0.489725 | 0.520410 |
| Random | 0.433699 | 0.468162 | 0.499313 |

`CAVS-OQR+MAUR` 相对最强外部基线 MORIC 的增益为 **+0.027522 / +0.035982 / +0.031922 AP**。相对 CAVS-OQ，ROI+MAUR 带来 **+0.002053 / +0.004016 / +0.000892 AP**；说明在 COCO 上，ROI 是小而稳定的补充动作，主要收益仍来自 Original/Q90 的联合字节调度。

### Oregon Cloud-V3 final（4,962 张）

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR + MAUR（我们）** | **0.506028** | 0.517405 | 0.516663 |
| **CAVS-OQ** | 0.499366 | **0.517611** | **0.519335** |
| ORIC | 0.483153 | 0.502061 | 0.509599 |
| MORIC | 0.479826 | 0.493286 | 0.497783 |
| Confidence | 0.489643 | 0.502499 | 0.510449 |
| Random | 0.471087 | 0.484147 | 0.495973 |

在 Oregon，CAVS-OQ 相对每个预算中最强的 ORIC/MORIC 分别提高 **+0.016213 / +0.015550 / +0.009736 AP**，相对包含 Confidence 的最强比较器提高 **+0.009722 / +0.015112 / +0.008886 AP**。OQR 在 10% 更高，但三个预算的最终选择均为 0 个 ROI；且在 25%/40% 略低于独立训练的 OQ。因此，Oregon 目前支持 CAVS 的 O/Q 联合调度泛化，不支持将 ROI 描述为已稳定泛化的收益来源。

`CAVS-O` 并非独立训练的 Original-only 学习器：统一 `O/OQ/OQR` 消融使用**同一三种子 OQR checkpoint 的动作硬掩码**，三行只改变允许动作，正是用于隔离 Original、Q90 与 ROI 的动作空间贡献。上表的 CAVS-O 数值来自该统一实验的 final ensemble 归档；不得用历史 `Original-only` 的 DualHead、ORIC 或 MORIC 行替代。由于 COCO/Oregon 主候选的 OQR 行采用后续的 MAUR 配置、OQ 行来自独立 OQ 训练，CAVS-O 应作为动作空间的硬掩码对照解读，而不应与这两行混同为同一次独立重训的三组比较。

## 结论与边界

1. **核心学习器应定义为 CAVS 的 action-conditioned joint scheduler，而不是传统 DualHead 或单一 ROI 规则。** 它在两个数据集、三个 Fixed-byte 工作点均超过 ORIC、MORIC、Confidence 与 Random。
2. **COCO 的最佳部署实例是 CAVS-OQR+MAUR；跨数据集最稳健的核心是 CAVS-OQ。** ROI 应作为可选扩展和动作空间消融报告，不能据 Oregon 结果宣称普适收益。
3. CAVS 的优势限定在真实字节容量调度：它不是只靠更强的静态图片 Top-K 排序，而是通过图像成员、Original/Q90/ROI 动作和 payload 成本的联合优化获得收益。

以上均为冻结协议下的点估计。COCO val2017 与 Oregon final 均已被历史探索使用，应在论文中标记为 post-hoc 论文级复用；最终确认性结论仍需互斥终评与完成的 paired/event-cluster bootstrap 支持。
