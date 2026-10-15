# 面向异构像素载荷的成本感知端云协同目标检测：上下文图像—动作价值学习与严格字节调度

Cost-Aware Edge–Cloud Object Detection with Contextual Image–Action Value Learning under Byte Budgets

> **稿件状态（2026-08-14）：V2.2 中文完整初稿，不可直接投稿。** 本版完成了针对 ORIC/MORIC、DCSB、OffloaD、任务感知传输、预算化多专家选择及 2025–2026 最近邻工作的定位复核。COCO val2017 与 Oregon final 均已被上游探索访问，本文结果属于事后论文级复用，不称为新的隐藏终评。COCO 严格 OQ 主模型、同骨干消融和 Oregon Cloud-V3 的三种子点估计已经写入；最新 paired/bootstrap 统计仍在运行。Orange Pi 5 Ultra/RK3588 真机测量、新注册确认集及尚未完成的置信区间继续以 `【待补：...】` 标记，不用点估计冒充统计确认。

[作者姓名]

[作者单位]

[通讯作者及电子邮箱]

## 摘要

> 资源受限的端云协同目标检测不仅需要判断是否调用云端，还需要决定传输何种图像表示。现有图像级排序忽略了不同表示的检测收益和编码大小均会随图像变化。本文提出上下文动作价值调度器（Contextual Action-Value Scheduler，CAVS），根据端侧检测结果估计每个图像—动作对的上下文边际效用，并在严格字节预算下联合选择图像与传输载荷。CAVS 将任务价值与通信成本解耦：价值估计器不使用载荷大小，容量调度器仅在决策阶段依据实际编码字节处理通信约束。我们在 COCO 和 Oregon 野生动物数据集上，以全量原图传输开销的 10%、25% 和 40% 作为通信预算进行评估。CAVS 在两个数据集上的 AP50–95 分别达到 0.478–0.559 和 0.499–0.519。在相同输入特征、奖励标签和字节预算下，图像—动作联合调度在 COCO 上较“先排序图像、再选择载荷”的两阶段设计提高 0.028–0.036 AP50–95；当预算仅按云端调用数量定义时，两者差异消失。结果表明，动作条件价值估计与显式字节核算能够刻画单一图像排序无法表达的精度—通信权衡。

Resource-constrained edge–cloud object detection requires deciding not only whether to invoke cloud inference but also which image representation to transmit. Existing image-level ranking ignores that the accuracy benefit and encoded size of each representation vary across images. We propose the Contextual Action-Value Scheduler (CAVS), which estimates the contextual marginal utility of each image–action pair from edge detection outputs and jointly selects images and payloads under a strict byte budget. CAVS decouples task value from communication cost: payload size is excluded from the value estimator and is used only by the capacity scheduler. We evaluate CAVS on COCO and the Oregon wildlife dataset at 10%, 25%, and 40% of the full-image transmission budget. CAVS achieves AP50–95 ranges of 0.478–0.559 on COCO and 0.499–0.519 on Oregon. Under identical input features, reward labels, and byte budgets, joint image–action scheduling improves AP50–95 by 0.028–0.036 over a two-stage image-ranking and payload-selection design on COCO; the difference vanishes when the budget is expressed only as a fixed number of cloud calls. These results show that action-conditioned value estimation with explicit byte accounting captures accuracy–communication trade-offs that cannot be represented by image ranking alone.





端云协同目标检测需要在有限上行资源下决定哪些图像交给云端，以及对每张图像传输何种像素载荷。选择性卸载中的一条主流路线先为每张图像预测一个静态优先级，再在固定上传率下选择 Top-K 图像；区域采样与任务感知编码则通常另行优化传输表示。前一种范式在仅有一种等成本云动作时成立，但面对 Original、不同压缩质量和 ROI 等异构动作时，同一图像的检测收益和实际字节成本均随动作变化，图像级排序与载荷选择因而不可分离。本文提出上下文动作价值调度器 CAVS（Contextual Action-Value Scheduler）。CAVS 将决策单元从“图像”提升为“图像—动作”，利用端侧检测集合的置换不变表示，为每个候选载荷预测与数据集级 AP50–95 对齐的上下文边际价值；价值网络不直接接收载荷字节，通信成本由外部严格容量调度器显式处理。调度器在每幅图像至多选择一个动作的约束下，联合决定是否上传以及使用 Original 或 Q90，并保证总请求字节不超过给定容量。本文进一步证明，传统 Top-K 是单一等成本云动作下的特殊情形，而异构动作通常不存在一个与预算无关的图像标量排序能够覆盖全部最优选择。

在统一缓存和评估器上的 COCO val2017 事后完整评测中，独立 OQ 训练的 CAVS 在 10%、25% 和 40% 固定字节容量下分别获得 0.478224、0.528884 和 0.559286 mAP50–95，对三种子同协议最强 ORIC/MORIC 外部基线的点估计提升为 0.025469、0.031966 和 0.031030。相同骨干、奖励和训练种子下，联合图像—动作调度相对“先排图像、再选动作”的两阶段方案提高 0.027850、0.036146 和 0.034582 AP，表明提升主要来自决策结构而非额外模型容量。Oregon Cloud-V3 上，全部依赖云模型输出的学习器重新训练后，CAVS-OQ 在相同容量下取得 0.499366、0.517611 和 0.519335 AP，比最强 ORIC/MORIC 点估计高 0.016213、0.015550 和 0.009736；相对包含 Confidence 的最强比较器仍高 0.009722、0.015112 和 0.008886。COCO 最新配对统计、Oregon event/image-cluster 统计为【待补：正在运行的冻结 bootstrap】，Orange Pi 5 Ultra/RK3588 的端侧时延、能耗、热稳定性和多带宽端到端结果为【待补：真实设备测量】。现有点结果支持图像—动作联合价值机制，但新的独立注册终评仍是确认泛化优势的必要条件。

**关键词：** 端云协同；目标检测；选择性卸载；动作价值学习；通信成本；容量约束调度

## Abstract

Edge–cloud collaborative object detection must decide both which images should be escalated to the cloud and which pixel payload should be transmitted under a limited uplink budget. One major line of selective-offloading methods assigns a static priority score to each image and chooses the Top-K images under an upload-ratio budget, while region sampling and task-aware encoding usually optimize the transmitted representation separately. The image-ranking formulation is appropriate when there is only one equal-cost cloud action. With heterogeneous payloads such as original images, JPEG-compressed images, and RoIs, however, both the detection benefit and the actual number of transmitted bytes become action dependent, making image ranking and payload selection inseparable. We present CAVS, a Contextual Action-Value Scheduler that treats an image–action pair, rather than an image alone, as the basic decision unit. CAVS encodes the edge detector output as a permutation-invariant detection set and predicts an AP50–95-aligned contextual marginal value for each candidate payload. Payload bytes are deliberately excluded from the value predictor and handled explicitly by a strict-capacity scheduler. Under a multiple-choice constraint, the scheduler jointly determines whether an image should be uploaded and which payload action should be used, while guaranteeing that the total request size does not exceed the byte budget. We further show that conventional Top-K routing is a special case with one equal-cost cloud action, whereas a budget-independent scalar image ranking is generally insufficient for heterogeneous actions.

In a post-hoc full evaluation on COCO val2017 using shared caches and evaluators, the independently trained OQ instance of CAVS achieves AP50–95 scores of 0.478224, 0.528884, and 0.559286 at the 10%, 25%, and 40% fixed-byte capacities. These point estimates exceed the strongest protocol-matched three-seed ORIC/MORIC baseline by 0.025469, 0.031966, and 0.031030 AP, respectively. Under the same backbone, rewards, and training seeds, joint image–action scheduling improves over a two-stage image-ranking-then-action policy by 0.027850, 0.036146, and 0.034582 AP. After retraining all cloud-dependent learners for Oregon Cloud-V3, CAVS-OQ obtains 0.499366, 0.517611, and 0.519335 AP, outperforming the strongest ORIC/MORIC point estimate by 0.016213, 0.015550, and 0.009736 AP. The frozen paired and event/image-cluster bootstrap analyses are **[in progress]**, and the latency, energy, thermal, and end-to-end measurements on an Orange Pi 5 Ultra/RK3588 are **[to be completed]**. The current point estimates support the image–action joint-value mechanism, while a newly registered independent test set remains necessary for confirmatory claims.

**Keywords:** edge–cloud collaboration; object detection; selective offloading; action-value learning; communication cost; capacity-constrained scheduling

## 1 引言

智能相机、野外监测节点和移动机器人通常需要持续执行目标检测。轻量端侧检测器能够在低功耗设备上快速给出结果，但其模型容量、输入分辨率和算子支持有限，对小目标、遮挡目标和长尾类别的表现往往弱于云端大模型。将所有图像上传至云端可以提高检测能力，却会持续占用上行带宽，并引入编码、排队、传输、云推理和返回结果的额外时延。因此，端云协同检测并不是简单地在“端侧”与“云端”之间二选一，而是一个受通信与计算资源约束的逐样本决策问题。

选择性卸载研究通常把该问题表述为“哪些图像值得上传”。Adaptive Feeding、DCSB、OffloaD 和 ORIC/MORIC 分别使用难度、检测失败风险或上下文检测收益估计上传价值 [10–13,20]。其中 ORIC/MORIC 的重要观察是，目标检测的 mAP 由跨图像、跨类别的匹配结果共同决定，单图置信度或单图正确率并不等价于一张图像对最终 mAP 的边际贡献 [13]。DCSB 的带宽自适应扩展还引入区域采样和随网络状态切换的判别器 [12]，因此不能把所有既有工作概括为固定阈值的整图 Top-K。更准确的剩余缺口是：现有检测卸载方法通常分别优化图片是否上传、区域/压缩表示或运行时调度，较少把多个像素载荷的**动作条件检测价值**与**逐样本真实字节**放入同一个决策问题。

真实通信系统还必须回答“上传什么”。一张图像可以保留端侧结果，也可以上传 Original、JPEG Q90，或在适用场景中上传一个或多个 ROI。不同动作不仅具有不同平均成本，实际字节还随图像内容显著变化。以固定上传率表示成本会忽略两个事实：第一，压缩载荷可能让相同字节容量容纳更多图像；第二，同一图像上的 Original 和 Q90 可能具有不同的云端收益次序。因此，如果先独立选择图像、再套用一个载荷策略，前一阶段的静态图像分数必须同时承担“是否调用云端”和“调用哪个动作”两个不同问题，容易产生动作价值尺度失配。

本文从这一结构性矛盾出发，提出如下研究问题：**在云动作的检测收益和实际字节均随图像变化时，能否直接学习每个图像—动作的任务价值，并在严格通信容量下联合决定图片成员关系与载荷动作？** 我们给出的回答是 CAVS，即上下文动作价值调度器。CAVS 不先固定一个上传集合，而是为每个候选 ((i,a)) 预测任务对齐价值，并让 Original、Q90 与不上传动作在同一容量约束中竞争。其价值模型只描述检测效用，真实 payload 字节留给外部调度器处理，从结构上避免把“便宜”错误学习成“检测价值高”。

CAVS 的设计由三个经验事实共同驱动。第一，前期 DualHead 与 SharedAction 的两阶段方案在固定字节预算下明显弱于图像—动作联合估值，说明主要瓶颈不是简单缺少一个成本头，而是决策粒度不正确。第二，将成本或保守下置信界直接混入价值分数会使学习器过度偏好不上传，造成严重容量闲置。第三，ROI 的可恢复收益随数据域、crop profile、云模型和融合方式显著变化。区域采样、任务感知 ROI 和不确定区域拼接已有充分先例 [12,14–16,34,35,39,40]；本文不把 ROI 本身包装为创新，而把它作为检验 CAVS 能否安全扩展到第三种异构载荷的候选动作。

从优化角度看，传统 Top-K 并非错误，而是 CAVS 问题的一个退化特例：当每张图像只有一个云动作且该动作成本相同，容量约束等价于固定调用数，选择最高价值的 K 张图像即为最优。只要动作成本或动作相对收益发生异质化，一个预算无关的图像排序通常无法同时服务所有容量。CAVS 因而不是在 Top-K 前增加一个更复杂的 Ranker，而是把预测对象从图像值升级为动作条件价值，并把真实成本留给容量求解。

本文的主要贡献如下。

1. **从图像排序到图像—动作决策的问题推广。** 将端云检测中的选择性卸载建模为图像—动作多选容量问题，明确区分任务价值、载荷成本和预算语义；证明 Top-K 是单一等成本云动作下的特殊情况，并给出异构动作下静态图像排序不足的反例。

2. **AP 对齐、价值—成本解耦的联合调度方法。** 提出 CAVS，把 ORIC 的 contextual image reward 思想推广到 contextual image-action reward；使用端侧检测集合为 Original/Q90/可选 ROI 学习动作条件的 AP50–95 边际价值，并用逐样本真实字节在外部容量求解器中联合确定上传成员与载荷。本文不把集合编码、上下文奖励、JPEG、ROI 或拉格朗日优化中的任一组件单独声称为首次提出。

3. **机制与系统边界并重的评测。** 在 COCO 和 Oregon 两个成本结构差异明显的检测域中，以 Fixed-byte 为主协议，对比 Random、Confidence、ORIC/MORIC、同骨干两阶段方案及 O/OQ/OQR、Profile×Fusion、成本输入等消融；同时报告 Fixed-call 边界和动作分布，以识别收益究竟来自图像排序还是成本—动作耦合。两个域的冻结三种子点估计均支持核心机制，最新 bootstrap、新注册确认集和 RK3588 真机结果仍为【待补】，不计作已完成贡献。

本文其余部分组织如下。第 2 节回顾选择性卸载、任务感知传输和预算决策研究；第 3 节定义系统与优化问题；第 4 节介绍 CAVS；第 5 节说明 RK3588 原型设计；第 6 节给出实验协议；第 7 节报告现有结果和待补主实验；第 8 节讨论理论含义、适用边界与论文主张；第 9 节总结全文。

## 2 相关工作

### 2.1 端云协同推理与选择性卸载

Neurosurgeon 根据移动端、网络和云端的性能模型选择神经网络切分位置 [1]，后续 split computing、边缘智能与早退研究系统分析了中间特征传输、多出口推理和端云划分的精度—时延权衡 [2–4]。面向视频分析的综述与早期系统还归纳了移动端模型选择、时延感知执行和云边资源协同的设计空间 [5–7]。这类工作主要决定“网络在哪里执行”。本文采用两个可独立工作的冻结检测器：端侧结果始终可用，调度器选择是否请求云端复核，因此研究对象是检测级选择性升级，而不是层级切分。

视频分析系统还利用时序冗余、帧过滤和配置 profile 减少推理或传输。NoScope、Reducto 和 Chameleon 分别从专用过滤器、运动/内容变化和动态配置角度降低视频分析成本 [8,9,30]。OneAdapt 进一步利用在线反馈自适应配置 [33]。这些系统通常依赖连续视频中的时间相关性。本文当前关注单帧、无跟踪状态的调度，因而将时序信息视为后续增强方向，而不是把视频过滤结果直接与单帧 CAVS 混合比较。

目标检测中的选择性卸载更接近本文问题。Adaptive Feeding 在快速与高精度检测器之间学习选择 [10]；DCSB 使用难例判别器识别需要云端处理的图像，其带宽自适应版本加入 regional sampling、discriminator zoo 和视频采样率更新 [11,12]；DCCI 进一步把 hard-case discriminator、设备模型压缩以及运行时负载/网络调度组合起来 [38]。OffloaD 将 introspective、golden 和 latency score 融合为检测失败指标，并用滑动窗口调整端云执行频率 [20]；EdgeBoost 说明经校准的置信度在资源受限推理中仍是强基线 [21]。级联理论也给出了置信度延期何时充分的条件 [24]。ORIC/MORIC 则使用 contextual reward 估计一张图像对整体检测 mAP 的边际作用 [13]。本文直接继承其“评价应与数据集级检测指标对齐”的思想，但决策变量不同：ORIC/MORIC 主要预测图像上传价值，CAVS 预测每个图像—载荷动作的价值，并在真实字节容量内联合选择。

最近的 Budget-Adaptive Routing 进一步指出，常见的 weak-conditioned router 始终在路由前运行弱检测器，高上传预算时可能为最终仍交给强模型的样本重复付出端侧前向成本；其方案按预算在 weak-conditioned 与 weak-skipping estimator 间切换 [41]。CAVS 当前属于 weak-conditioned 设计：Edge 输出既是回退结果也是在线价值输入。本文优化的是上传载荷和云动作，而不是所有预算下的最小端侧计算；高预算 weak-skipping 因此被列为真机边界对照，而非被当前离线 AP/KB 表掩盖。

### 2.2 ROI、压缩与任务感知表示

通信成本也可以通过改变传输表示降低。RegionFilter 根据区域内容过滤视频 [14]；EdgeDuet 结合低分辨率检测和高分辨率 tile，缓解小目标场景中的带宽压力 [15]；AccMPEG 根据像素区域对分析任务的影响分配编码质量 [16]。TileClipper 和 RegenHance 等近期系统也表明，空间 tile、区域选择和任务相关增强可以在视频分析中形成新的精度—成本前沿 [34,35]。另一条路线传输中间特征并进行压缩或语义编码 [22,23,31]。

本文不提出新的编解码器，而是把标准 Original、JPEG Q90 及可选 ROI 视为离散动作。这样做有两个目的：一是保持检测器和像素编码链路冻结，使学习器收益不与新编解码器训练混杂；二是用逐图真实请求字节而不是平均压缩率约束调度。2025 年的 Selective Cloud Offloading 已使用 conformal prediction 识别不确定检测区域，并将多个 ROI 二维拼接为一次云请求，以共同降低传输、推理和 API-call 成本 [39]；VaBUS 也利用背景建模和自适应编码只上传高置信 ROI [40]。因此，本文不把“选择性区域上传”或“区域拼接”本身列为贡献。CAVS 的区别是让 ROI 与 Original、Q90、Edge 一起接受 AP50–95 上下文动作价值监督，并在逐图实际字节容量中竞争；ROI 的 proposal 覆盖、裁剪上下文、云模型训练域和融合负收益均显式进入动作奖励与消融。

### 2.3 学习延期、预测后优化与预算调度

Learning to defer 将模型与专家之间的分配表述为条件风险最小化 [17–19]，置信度校准研究则提醒神经网络分数不一定对应真实风险 [25]。FrugalML 在预算约束下选择不同预测服务 [32]。这些研究共同说明，路由器不仅需要预测任务收益，还必须显式面对动作成本和专家互补性。

CAVS 还与 predict-then-optimize 和 decision-focused learning 相关。Smart Predict-then-Optimize（SPO）强调预测误差应由下游优化决策衡量 [36]。FrugalML 已在预算约束下联合利用不同预测服务的样本相关强弱与调用价格 [32]，所以“预算下选择异构专家”也不是本文首次提出。CAVS 的任务特异性来自目标检测的非可加 AP、像素载荷及端云融合。当前实现并未对离散求解器进行端到端反向传播：它先学习 AP 对齐的动作价值，再用真实成本执行容量调度。因此，本文将方法描述为“任务对齐的预测后优化”，而不宣称提出了新的通用 decision-focused 优化理论。

### 2.4 与 ORIC/MORIC 的结构差异

ORIC/MORIC 是本文最重要的同类外部基线。两者利用端侧检测结构和 contextual reward 改善图像级卸载排序，在固定调用预算下具有很强竞争力 [13]。CAVS 与其共享两个基本认识：端侧置信度并非云端收益的充分统计量；训练奖励应尽量接近最终检测指标。区别集中在以下三点。

第一，ORIC/MORIC 的基本对象是一张图像，CAVS 的基本对象是图像—动作对。第二，CAVS 将价值与字节成本分离，预测器不因一个动作更便宜就自动赋予更高任务价值；预算求解器在推理阶段使用逐样本实际字节。第三，CAVS 的主评测是 Fixed-byte，而 Fixed-call 仅用于隔离图像排序能力。最新点估计中，CAVS 在 COCO Fixed-call 下仅小幅高于外部方法，在 Oregon 则低于最强比较器；更关键的是，同骨干 Joint 与 TwoStage 在 COCO Fixed-call 下完全相同，而 Fixed-byte 差距达到 0.028–0.036 AP。这说明 CAVS 的主要增益来自异构动作与字节容量的联合决策，而不是普遍占优的图像排序。

**表 1 代表性研究与本文问题设置的结构比较。** “支持”表示方法的核心协议显式处理该维度，不表示实现能力的绝对有无。

| 方法 | 主要决策单元 | 多像素载荷动作 | 逐图真实字节约束 | 与检测 mAP 对齐的奖励 | 本文中的角色 |
|---|---|---:|---:|---:|---|
| Adaptive Feeding [10] | 图像/检测器 | 否 | 否 | 否 | 双模型选择先驱 |
| DCSB [11,12] | 图像/区域采样 | 是 | 带宽自适应 | 否 | 难例路由与区域采样基线类别 |
| ORIC/MORIC [13] | 图像 | 否 | 以上传比例为主 | 是 | 最强同协议外部排序基线 |
| OffloaD [20] | 图像/任务 | 否 | 时延与调度成本 | 间接 | 失败风险调度 |
| Selective Cloud Offloading [39] | 检测框/拼接 ROI | 是 | API call/处理成本 | 否 | 最近邻 ROI 选择与打包方法 |
| RegionFilter/EdgeDuet [14,15] | 区域/tile | 是 | 是 | 任务相关 | 空间载荷方法 |
| AccMPEG [16] | 编码区域 | 是 | 是 | 是 | 任务感知编码 |
| FrugalML [32] | 样本/服务 | 是 | 调用费用 | 分类效用 | 预算化专家选择 |
| CAVS（本文） | 图像—载荷动作 | 是 | 是 | contextual AP50–95 | 联合动作价值与容量调度 |

本文不声称上述通用组件——集合编码、上下文奖励、JPEG、ROI、区域拼接、多专家预算选择、拉格朗日乘子或背包式选择——本身首次出现。创新主张限于它们围绕检测级异构像素载荷形成的整体决策方式：**动作条件的 AP 对齐价值、价值—成本解耦以及严格字节容量下的图像—动作联合调度。** 据本文检索到的相邻工作，尚未发现同时显式实现这三个维度的检测卸载方法；这一表述是有限范围的文献判断，不写成无法证实的“绝对首次”。

## 3 系统模型与问题定义

### 3.1 端云检测流程

给定图像序列或评测批次 ​\(\mathcal D=\{I_i\}_{i=1}^{N}\)，端侧检测器 \(D_e\) 首先对所有图像推理，得到检测集合

\[
E_i=D_e(I_i)=\{(b_{ij},y_{ij},p_{ij})\}_{j=1}^{n_i},
\]

其中 \(b_{ij}\)、\(y_{ij}\) 和 \(p_{ij}\) 分别表示框坐标、预测类别和置信度。端侧输出既是未上传时的最终结果，也是 CAVS 唯一必需的在线语义输入。系统为每张图像提供动作集合

\[
\mathcal A_i=\{e,o,q\},
\]

其中 \(e\) 表示保留 Edge 结果，\(o\) 表示上传 Original，\(q\) 表示上传 JPEG Q90。可选 ROI 扩展记为 \(r\)，仅在端侧 proposal、坐标映射和云端融合均可用时加入 \(\mathcal A_i\)。

对任意云动作 \(a\ne e\)，编码器产生载荷 \(P_i^a\)，其实际请求字节为 \(c_i^a=|P_i^a|\)。云端检测器 \(D_c\) 输出 \(C_i^a=D_c(P_i^a)\)，经坐标恢复和必要的端云融合得到最终结果 \(F_i^a\)。Edge 动作满足 \(F_i^e=E_i\) 且 \(c_i^e=0\)。当网络失败、任务超时或云服务不可用时，系统可直接回退到 \(F_i^e\)，但本文当前离线精度表不把故障注入混入调度收益。

**表 2 主要符号。**

| 符号 | 含义 |
|---|---|
| \(i\)、\(a\) | 图像索引与候选动作 |
| \(E_i\)、\(F_i^a\) | 端侧检测结果与执行动作 \(a\) 后的结果 |
| \(z_i\)、\(u_i^a\) | 图像/检测集合特征与动作特征 |
| \(v_i^a\)、\(\hat v_i^a\) | 真实代理价值与学习器预测价值 |
| \(c_i^a\) | 图像 \(i\) 执行动作 \(a\) 的实际请求字节 |
| \(x_i^a\) | 是否为图像 \(i\) 选择动作 \(a\) 的二元变量 |
| \(B\) | 当前批次或时间窗口的上行字节容量 |
| \(\lambda\) | 容量约束的影子价格/拉格朗日乘子 |

### 3.2 图像—动作联合容量问题

若 \(\hat v_i^a\) 表示动作 \(a\) 相对 Edge 的预测任务价值，则 CAVS 的部署决策写为

\[
\begin{aligned}
\max_{x}\quad & \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a\hat v_i^a,\\
\text{s.t.}\quad & \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a c_i^a\le B,\\
& \sum_{a\in\mathcal A_i}x_i^a=1,\quad \forall i,\\
& x_i^a\in\{0,1\}.
\end{aligned}
\tag{1}
\]

由于 Edge 动作成本为零，等式约束不会强迫上传，而只是保证每幅图像最终有一个输出。式（1）具有多选背包结构，但 \(\hat v_i^a\) 只是对非可加 mAP 的学习代理。本文不宣称最大化式（1）等价于直接最大化数据集级 mAP；所有最终结论仍由统一 COCO-style evaluator 对完整预测集合计算。

### 3.3 三种预算语义

本文区分 Fixed-call 与 Fixed-byte，并把在线窗口实现单独报告。

**Fixed-call。** 给定调用比例 \(\rho\)，严格选择 \(K=\lfloor\rho N\rfloor\) 幅图像请求云端。它消除了“便宜动作容纳更多图像”的作用，适合评估纯图像成员排序，但不能代表真实字节相同。

**Fixed-byte。** 以 Always-Original 的总请求字节为参考，将容量定义为

\[
B_\rho=\rho\sum_{i=1}^{N}c_i^o,\qquad \rho\in\{0.10,0.25,0.40\}.
\tag{2}
\]

Original、Q90 和 Edge 在同一 \(B_\rho\) 内竞争。由于 Q90 更小，实际调用比例可以高于名义容量比例，这是 Fixed-byte 的机制而不是预算漂移。

**流式窗口预算。** 真机部署时，批量容量需要转换为长度为 \(W\) 的时间窗口或令牌桶。例如上行速率为 \(R_{up}\) bit/s 时，可用载荷预算可写为

\[
B_W=\frac{R_{up}W}{8}-B_{protocol}-B_{reserve},
\tag{3}
\]

其中后两项分别覆盖协议开销与安全余量。窗口大小、任务过期和队列规则会影响真实时延，相关结果为【待补：RK3588 流式实验】。

### 3.4 Top-K 的特例与静态排序的不足

**命题 1（Top-K 特例）。** 假设每幅图像只有 Edge 与一个云动作 \(g\)，且所有图像的云动作成本均为常数 \(c>0\)。则式（1）在容量 \(B\) 下等价于选择 \(K=\lfloor B/c\rfloor\) 个 \(\hat v_i^g\) 最大的图像。

*证明概要。* 任意包含不超过 \(K\) 个云动作的集合均可行。若可行集合包含价值较低的 \(j\) 而未包含价值较高的 \(i\)，交换两者不改变成本并提高或保持目标。重复交换即得到 Top-K。完整证明见附录 A。

**命题 2（预算无关图像排序通常不足）。** 当至少两种云动作具有异构成本与相对价值时，通常不存在一个与预算无关的图像标量 \(s_i\)，使得对所有容量都可以仅取 \(s_i\) 的前缀并在前缀内选择动作而得到式（1）的最优解。

一个两图反例即可说明。设图像 1 的候选为 Original \((c,v)=(10,10)\)，图像 2 的候选为 Q90 \((6,7)\)，且两图 Edge 均为 \((0,0)\)。预算 \(B=6\) 时只能选择图像 2；预算 \(B=10\) 时选择图像 1；如果再给图像 1 增加 Q90 候选 \((5,4)\)，不同预算还会改变同一图像内部的动作次序。任何固定图像前缀都无法同时表达这些容量相关选择。该反例不说明所有两阶段方法必然失败，而是说明它们需要额外的预算与动作条件才能覆盖一般情形。

## 4 上下文动作价值调度器 CAVS

### 4.1 方法概览

CAVS 包含三个相互分离的模块：检测集合编码器、动作条件价值学习器和严格容量调度器。端侧检测集合被编码为图像语义表示；每个候选动作与该表示交互并产生 \(\hat v_i^a\)；调度器读取 \(\hat v_i^a\) 和真实 \(c_i^a\)，在容量内选择动作。图 1 应在最终排版中显示这一数据流。

**【待补图 1：CAVS 总体架构。左侧为 RK3588 端侧检测与集合编码，中间为动作价值预测，右侧为 Fixed-byte 联合调度、编码上传和云端检测。应明确字节只进入调度器，不直接进入 Value head。】**

这一分工是有意设计的。若 payload 大小直接进入价值头并以单位字节收益监督，学习器可能把“低成本”当成“高任务价值”，导致 Fixed-call 下的排序含义发生变化；若把保守风险下界直接代替价值，负偏差会使 Edge 的零值占优并造成预算闲置。CAVS 因此学习“如果执行该动作，它对检测任务有多大价值”，再由调度器决定该价值是否值得付出对应字节。

### 4.2 端侧检测集合编码

对每幅图像最多保留置信度最高的 \(M=64\) 个端侧检测。第 \(j\) 个框的数值特征为

\[
r_{ij}=[x_c,y_c,w,h,wh,p]_{ij},
\tag{4}
\]

其中坐标与尺寸按图像宽高归一化，\(wh\) 为面积，\(p\) 为置信度。类别 \(y_{ij}\) 由 8 维可学习嵌入表示。数值特征和类别嵌入经共享 MLP 得到框表示 \(h_{ij}\)，随后执行平均池化与最大池化：

\[
h_i^{set}=\left[\frac{1}{n_i}\sum_j h_{ij}\;\middle\|\;\max_j h_{ij}\right].
\tag{5}
\]

该编码对检测框输入次序置换不变，同时保留总体分布与显著框信息。图像级辅助特征包括宽高比、框数量、置信度最大值与均值、检测框总面积，以及低成本亮度和边缘密度统计。没有检测框时使用显式空集掩码与零池化向量，而不是伪造检测框。

当前实现采用轻量 MLP 而非更深的 Transformer：框隐藏维度为 32，动作隐藏维度为 32，联合隐藏维度为 64，dropout 为 0.05。前期关系编码诊断只在低预算产生有限增益、在高预算没有稳定改善，因此主稿优先保留可解释且适合 RK3588 CPU 执行的集合表示。

### 4.3 动作表示与价值—成本解耦

动作特征 \(u_i^a\) 描述动作类型、编码分辨率、云调用次数、区域数量和几何属性。训练代码还可记录 payload 比率和资源统计，但在 CAVS Value head 中，直接成本字段被置零：

\[
\tilde u_i^a=\operatorname{StripCost}(u_i^a).
\tag{6}
\]

动作类型通过可学习嵌入表示。图像集合表示与动作表示拼接后得到

\[
\hat v_i^a=f_\theta(h_i^{set},z_i^{scene},\tilde u_i^a).
\tag{7}
\]

Edge 被定义为零收益参考，即 \(\hat v_i^e=0\)。Original 与 Q90 使用同一网络参数，从而使网络能够学习“同一图像上不同动作的相对优势”，而不是为每个动作训练互不校准的独立 Ranker。

### 4.4 从 ORIC 图像奖励到 AP50–95 上下文动作奖励

目标检测 mAP 不是逐图可加指标。ORIC 已通过在代表性上下文中替换一张图像的弱/强检测结果，构造了比单图奖励更接近数据集级 mAP 的监督信号 [13]。本文不重新声称这一原则，而把替换变量从“是否使用强检测器”推广为“对图像 \(i\) 执行动作 \(a\)”。为得到可训练的样本级信号，本文采用 fold-local 上下文边际奖励。对训练目标图像 \(i\)，从同一训练折内构造不含 \(i\) 的上下文集合 \(\mathcal C_i\)，其大小当前设为 32。令 \(\operatorname{AP}_{50:95}(\mathcal P)\) 表示对预测集合 \(\mathcal P\) 计算的 COCO-style mAP50–95，则动作奖励为

\[
v_i^a=(|\mathcal C_i|+1)\left[
\operatorname{AP}_{50:95}(F_{\mathcal C_i}^{e}\cup F_i^a)-
\operatorname{AP}_{50:95}(F_{\mathcal C_i}^{e}\cup F_i^e)
\right].
\tag{8}
\]

缩放项把小上下文中的边际变化恢复到近似单样本量级。上下文只用于离线生成监督标签；部署时不访问 GT，也不需要重新计算 AP。严格 OOF 实验中，上下文、奖励统计与模型训练均限制在对应训练折内，以防 fold 外标签进入训练。

式（8）仍是对全数据集 mAP 的局部代理，可能受上下文大小、类别组成和随机采样影响；更接近最终指标的奖励也可能更难从 Edge 输出估计。本文因此把三个问题分开报告：Teacher/Oracle 奖励能否形成更优选择上界；学习器能恢复多少 Teacher 收益；容量调度后的完整预测集合能否提高 Official AP。投稿版还应补充 context size 与采样种子的稳定性【待补：\(|\mathcal C|\in\{8,16,32,64\}\) 或等成本子集】，而不能只凭奖励相关性宣称最终性能。

### 4.5 训练目标

价值网络同时学习数值回归和同图动作次序。令标准化奖励为 \(\bar v_i^a\)，回归项采用 Smooth L1：

\[
\mathcal L_{reg}=\frac{1}{|\Omega|}\sum_{(i,a)\in\Omega}
\operatorname{SmoothL1}(\hat v_i^a,\bar v_i^a).
\tag{9}
\]

对同一图像的动作对 \((a,b)\)，若 \(v_i^a>v_i^b\)，使用按真实奖励差加权的 logistic 排序项：

\[
\mathcal L_{act}=\frac{1}{|\Pi|}\sum_{(i,a,b)\in\Pi}
w_i^{ab}\log\left(1+\exp[-(\hat v_i^a-\hat v_i^b)]\right),
\tag{10}
\]

其中 \(w_i^{ab}\propto |v_i^a-v_i^b|\)。总损失为

\[
\mathcal L=0.5\mathcal L_{reg}+0.5\mathcal L_{act}.
\tag{11}
\]

训练使用 AdamW，初始学习率 \(10^{-3}\)、权重衰减 \(10^{-4}\)，最多 60 epoch，early-stopping patience 为 8。论文级实验使用 3 个预先固定种子并集成预测；任何超参数变化都应在新 calibration/final 协议前冻结。

### 4.6 容量可行的联合调度

对给定影子价格 \(\lambda\ge 0\)，每幅图像独立选择

\[
a_i(\lambda)=\arg\max_{a\in\mathcal A_i}\left(\hat v_i^a-\lambda c_i^a\right).
\tag{12}
\]

随着 \(\lambda\) 增大，昂贵动作受到更强惩罚。实现通过固定 80 次二分搜索寻找一个总成本不超过 \(B\) 的 \(\lambda\)。由于动作是离散的，拉格朗日解可能留下部分容量；随后从当前动作出发枚举所有可行升级，按正边际价值与增量字节的比值排序，逐次加入不超预算的升级，直至没有正收益可行升级。最后强制断言

\[
\sum_i c_i^{a_i}\le B.
\tag{13}
\]

该过程可稳定产生容量可行解，并在现有规模上快速运行，但它不是对一般多选背包的精确最优性证明。论文中统一称其为“严格容量调度器”或“容量可行拉格朗日调度器”，不称为“全局精确求解器”。

**算法 1 CAVS Fixed-byte 调度。**

```text
输入：动作价值 {v_hat[i,a]}、真实字节 {c[i,a]}、容量 B
输出：每图动作 a[i]

1. 将 Edge 动作设为 value=0、cost=0
2. 二分搜索 lambda：a[i] = argmax_a(v_hat[i,a] - lambda*c[i,a])
3. 保留总成本不超过 B 的最小惩罚可行解
4. 枚举从当前动作到其他动作的正价值升级
5. 按 边际价值/增量字节 从高到低执行仍然可行的升级
6. 断言 sum_i c[i,a[i]] <= B，返回 a
```

### 4.7 ROI 与多质量动作扩展

式（1）允许加入 ROI、Q70/Q80/Q90 或多个分辨率动作，无需改变问题形式。但动作空间扩展不等于自动提高性能。ROI 需要额外特征描述 proposal 覆盖、裁剪边界、扩框 profile、缩放后目标尺寸和融合风险；多质量 JPEG 则需要为每个质量生成冻结缓存，并核对云模型是否处于相同输入域。

本文的主实例采用 OQ（Original/Q90），原因不是 ROI 没有任何价值，而是现有证据显示其安全收益稀疏且跨协议不稳定。AFPD 等后续探索尝试学习完整 \(reward-\lambda\cdot bytes\) 动作前沿，在部分预算取得增益，但尚未稳定超过独立 CAVS-OQ-Core。第 7.5 节把这些结果作为动作扩展边界报告。

### 4.8 在线复杂度与回退

端侧额外开销由至多 64 个框的两层集合编码和小型 MLP 构成，远小于一次检测器推理。调度器的批量复杂度近似为 \(O(TN|\mathcal A|+U\log U)\)，其中 \(T=80\) 为二分次数，\(U\) 为候选升级数。在线窗口中可通过预计算动作字节、减少二分次数或温启动 \(\lambda\) 进一步降低开销。

任何云动作超时或失败都回退到端侧结果；这一工程回退保证“有输出”，但不保证回退后的统计 AP 仍满足某个形式化风险界。队列溢出、网络中断和 deadline miss 的实验为【待补：真机故障与压力测试】。

## 5 Orange Pi 5 Ultra / RK3588 原型设计

### 5.1 硬件与软件边界

计划中的端侧平台为 Orange Pi 5 Ultra，其 SoC 为 Rockchip RK3588，集成 CPU、GPU 与 NPU [37]。端侧检测器将导出为 RKNN 并在 NPU 上运行；CAVS 集合编码器与调度器优先在 CPU 上执行，以减少自定义 NPU 图编译复杂度。云端运行冻结的高容量检测器。当前稿件没有使用厂商峰值 TOPS 估算真实时延或能效，最终仅报告端到端实测。

必须特别注意，FP32/PyTorch 端侧模型转换为 RKNN INT8/FP16 后，检测框、置信度和类别分布可能变化。由于 CAVS 的输入正是端侧检测集合，真机论文实验不能直接复用桌面模型生成的学习器：应先冻结 RKNN 模型与量化配置，重新生成 fit/calibration/final 的 Edge 特征和动作奖励，再重新训练 CAVS、Confidence、ORIC/MORIC 等所有依赖端侧输出的学习器。

### 5.2 端到端执行链路

真机链路按以下组件计时：图像读取与预处理、端侧 NPU 推理、后处理、CAVS 评分、窗口调度、JPEG 编码、请求排队、上行传输、云端解码与推理、结果返回和结果整理。除逐组件时延外，还应报告从图像进入到最终检测可用的端到端时延。

**【待补图 2：RK3588—网络—云端测试床。标明相机/数据回放、NPU、CPU 调度、编码线程、有界队列、网络整形器、云 GPU 和回退路径。】**

### 5.3 真机测量协议

为避免只报告均值，最终实验至少包含：

- 时延：各组件及端到端 P50/P95/P99；
- 吞吐：持续 FPS、窗口内处理数量和 deadline miss rate；
- 通信：payload bytes、HTTP/TCP/TLS 后 wire bytes、上传成功率；
- 资源：CPU/NPU 利用率、内存峰值、温度和频率降级；
- 能耗：设备空闲功率、Edge-only、编码上传和完整协同的平均功率与 J/frame；
- 网络：1/5/10/20 Mbps，上行 RTT 与抖动至少两档；有线网络为主，Wi-Fi 作为补充；
- 稳定性：不少于【待补：持续运行时长】的长时间测试与故障注入。

**表 3 RK3588 真机系统结果占位。**

| 方法 | 上行带宽 | AP50–95 | Payload KB/图 | Wire KB/图 | E2E P50/P95/P99 | FPS | J/frame | 温度峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edge-only | — | 【待补】 | 0 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Always-Q90 | 10 Mbps | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ORIC/MORIC | 10 Mbps | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| CAVS | 10 Mbps | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Cloud-first / weak-skipping 上界 | 10 Mbps | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

真机实验完成前，本文只声称 CAVS 的模型规模适合端侧执行，不声称系统已经达到实时或能耗最优。

## 6 实验设计

### 6.1 研究问题

实验围绕七个问题展开。

- **RQ1：** 在相同 Fixed-byte 容量下，CAVS 是否优于 Random、Confidence 和 ORIC/MORIC？
- **RQ2：** 该优势来自更强的图像排序，还是来自图像—动作联合调度？
- **RQ3：** 价值—成本解耦、跨动作校准和 ROI 扩展分别有什么作用？
- **RQ4：** COCO 上的结论能否迁移到采用不同图像分布、类别构成和 payload 尺度的 Oregon？
- **RQ5：** 在 RK3588 和真实网络下，精度—字节优势能否转化为端到端时延与能耗收益？
- **RQ6：** Edge 与 Cloud 的逐图互补性、Teacher ceiling 与 learned reward 之间各有多大差距，性能损失发生在奖励定义、价值估计还是容量决策？
- **RQ7：** 在高上传预算下，CAVS 始终运行 Edge 的 weak-conditioned 流程相对 Cloud-first/weak-skipping 上界付出多少额外时延与能耗？

RQ1–RQ3 已有 COCO 事后完整点结果与同骨干消融；RQ4 已有 Oregon Cloud-V3 从头重训后的点结果。RQ6 已有部分 Oracle/学习器诊断但尚未按最终主模型统一汇总。两者最新冻结 bootstrap 尚未全部完成，RQ5 与 RQ7 尚待真机实验。

### 6.2 数据集与数据角色

**COCO。** 使用 COCO 2017 [28]。学习器的 canonical fit 集为 train2017 中冻结的 9,678 张图像，calibration 为与 fit 互斥的 571 张图像，完整 val2017 共 5,000 张作为当前评测集。由于 val2017 已被多轮上游实验访问，本稿把其结果明确标记为 `post-hoc full evaluation`，不称为新的 hidden final。

**Oregon。** Oregon 野生动物数据使用相机/事件感知的冻结协议：fit 9,923 张、calibration 4,961 张、final 4,962 张。历史 canonical V1 检测器和 Cloud-V2 诊断已经访问过 final，因此相应表格同样是探索性证据。论文点结果采用新的 Cloud-V3 epoch-28 checkpoint；由于云端模型改变会改变 Original/Q90/ROI 的真实动作奖励，所有内部学习器与 ORIC/MORIC 均已从 fit 重新训练，没有复用 V1/V2 学习器。

**表 4 数据角色与当前证据状态。**

| 数据集 | Fit | Calibration | 当前完整评测 | 划分原则 | 当前证据状态 |
|---|---:|---:|---:|---|---|
| COCO | 9,678 | 571 | val2017 5,000 | canonical 冻结清单 | 事后完整复用 |
| Oregon | 9,923 | 4,961 | final 4,962 | 相机/事件感知冻结清单 | Cloud-V3 已重训并评出点结果；final 为事后复用 |
| 新确认集 | 【待补】 | 【待补】 | 【待补】 | 与既有数据及调参历史互斥 | 尚未建立 |

ENA24 在 V1 中用于研究 ROI 和跨域差异，但本稿主线暂时聚焦 COCO 与 Oregon：前者是通用目标检测基准，后者代表 payload 大、背景稳定但事件相关性强的野外场景。删去 ENA24 不应被表述为方法只适用于两个域；它是为了让主实验围绕“异构字节动作联合调度”而不是多套不一致 ROI 协议展开。投稿前可把 ENA24 结果放入补充材料，或在新协议统一后恢复。

### 6.3 检测器、动作与缓存

端侧和云端检测器均在学习器训练前冻结，当前模型属于 YOLO26 系列 [29]。COCO 使用【待补：端侧 YOLO26 型号、权重 SHA、输入尺寸】和云端 checkpoint SHA-256 `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`（【待补：正式型号与输入尺寸】）。Oregon 使用 Cloud-V3 epoch-28 `best.pt`，SHA-256 为 `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`；其训练验证日志为 P=0.86874、R=0.67210、AP50=0.74694、AP50–95=0.54282。Oregon 端侧模型及其 RK3588 量化版本为【待补：型号、量化类型与 SHA】。所有 Original/Q90/ROI 请求均由对应云 checkpoint 重新推理并生成缓存。

主动作集合为 Edge、Original、Q90。Q90 使用标准 JPEG quality=90；应同时记录编码前尺寸、payload 字节和含协议开销的 wire 字节。ROI 只在扩展实验中使用，其 crop profile、推理尺度、坐标回映射、融合算法和不可用回退规则必须随缓存一同冻结。

基础上下界包括 Edge-only、Cloud-only Original 和 Cloud-only Q90。COCO 当前 detector-only 数字为【待补：用与表 6 完全相同 evaluator 重算】。Oregon Cloud-V3 final 上，Edge-only、Cloud-only Original 和 Cloud-only Q90 的 Official AP 分别为 0.454815、0.520966 和 0.509918；后两者的 request KB/图分别为 1595.49 和 97.98。其 AP50/AP75 分别为 0.717773/0.581400（Original）和 0.705003/0.568464（Q90），Edge-only 为 0.639788/0.504071。历史检测器数字不混入 Cloud-V3 主表。

### 6.4 对比方法

主对比包含以下层次。

1. **Random：** 在相同预算和动作政策下随机选择；完整结果应报告预先固定的多种子均值、区间或经验带。
2. **Confidence：** 根据端侧检测置信度排序，是最低复杂度的强启发式基线。
3. **ORIC/MORIC：** 使用作者论文定义和同协议训练的外部学习基线 [13]；不得用 MORIC-style 代理替代正式复现。
4. **Current-BudgetSet、SetValue 与 DualHead：** 本项目中从成本混合分数到价值/成本辅助分解的两阶段内部基线。
5. **CAVS-OQ-Core：** 独立在 Edge/Original/Q90 动作上训练三个种子并进行联合估值与 Fixed-byte 调度；作为不依赖 ROI 的核心方法与动作空间消融，不与历史 Raw ActionValue 或由 OQR checkpoint 屏蔽 ROI 得到的 OQ-mask 结果混名。
6. **CAVS-OQR：** 在同一联合调度框架中加入冻结 Profile 与融合语义的 ROI 动作。若 Current@640+MAUR 在 COCO 与 Oregon 的冻结统计都通过，它作为主部署实例，CAVS-OQ-Core 作为动作消融；否则 OQR 只作为数据域相关扩展。AFPD-OQR 与 Occupancy Gate 保留为负消融或探索边界。
7. **DCSB/Selective Cloud Offloading 关系对照：** 若无法在完全相同缓存上忠实复现，应至少在 Fixed-call 补充材料中报告规则化 DCSB-style 难例路由，并用结构表说明 conformal ROI packing 与 CAVS-OQR 的决策目标和成本口径差异；不能把二者省略后声称 ROI 选择优于所有既有方法。

在 Fixed-byte 中，各方法必须使用相同逐图 payload、相同容量和相同 evaluator。对于图像级 Ranker，分别报告 Original-only 和冻结 SharedAction-NoROI，或在方法定义允许时使用其原生动作策略；主表不得按评测集结果为每个方法事后挑选未预注册策略而不披露。

### 6.5 训练与冻结协议

学习器训练采用 5-fold OOF 和 3 个预先固定随机种子。fold 内生成 contextual reward，模型、标准化统计和 early-stopping 均不访问 held-out fold。完成开发诊断后，在全量 fit 上训练三种子模型；calibration 只评测一次，用于验证预注册 gate 或冻结必要阈值；final/val 不用于模型、epoch、超参数、动作门限或比较器选择。

当前 COCO val2017 和 Oregon final 均不满足新的 confirmatory final 条件。因此本稿采用双层报告：现有完整表用于展示机制和点估计；新的互斥终评集用于最终确认，结果全部留为【待补：新注册 final】。Cloud-V3 点结果生成后不再据此调整主方法、动作政策或超参数。

### 6.6 指标与统计检验

主精度指标为 COCO-style mAP50–95（Official AP），同时报告 AP50 和 AP75。通信指标为 request KB/image；真机额外报告 wire KB/image。Fixed-byte 主预算为 10%、25% 和 40% Always-Original 容量。Fixed-call 使用相同三档调用比例。

主方法与每个冻结比较器进行 1,000 次图像级 paired bootstrap [27]，报告均值差、95% 区间和 \(\Pr(\Delta AP>0)\)。实验纪律参考“先冻结、后检验”的风险校准思想 [26]，但本文的经验 bootstrap 门槛不等价于形式化有限样本风险保证。Oregon 相机图像存在事件簇相关性，最终还应增加 event/camera-cluster bootstrap；仅使用图像级 bootstrap 会低估相关样本带来的不确定性。

预期确认门槛为：三个 Fixed-byte 预算中至少两个相对最强外部基线提高不低于 0.001 AP，其余预算不低于 -0.001；容量利用率不低于 98%；三个种子的方向基本一致；Fixed-call 允许不占优，但必须如实报告。该门槛应在访问新 final 前写入协议，而不是根据结果修改。

### 6.7 实现与复现

所有数据角色清单、检测器 SHA、payload 缓存 SHA、类别映射、阈值、随机种子和环境版本均写入 machine-readable protocol。主结果脚本应具备以下断言：样本 identity 一一对应；不同数据角色按 identity 而非裸 image_id 检查互斥；所有分数有限；每图至多一个动作；总字节不超容量；final 不进入训练或阈值选择；同一比较表共享 evaluator 与缓存。

训练硬件与耗时为【待补：CPU/GPU 型号、并行度、每模型训练耗时】；推理软件版本为【待补：Python、PyTorch、Ultralytics/RKNN Toolkit 版本】。代码与冻结协议计划在录用后公开。

## 7 结果与分析

### 7.1 COCO Fixed-byte 主结果

表 5 汇总严格补全协议下的 COCO val2017 点结果。CAVS-OQ-Core、ORIC 和 MORIC 均以三个种子训练并以 ensemble 分数评测；Random 为 50 个预固定种子的 AP 均值。所有方法共享云模型、动作缓存、容量和 evaluator。主表只列 OQ 可比政策；历史九 Ranker 与 Original-only/SharedAction-NoROI 完整矩阵移入补充材料。

**表 5 COCO Fixed-byte 结果（Official AP / request KB per image，post-hoc val2017）。**

| 方法 | 动作政策 | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| **CAVS-OQ-Core（本文）** | Joint-OQ | **0.478224 / 15.91** | **0.528884 / 39.78** | **0.559286 / 63.65** |
| CAVS-OQ-Core-TwoStage | 先排图像、后选动作 | 0.450374 / 15.91 | 0.492738 / 39.78 | 0.524704 / 63.65 |
| CostInValue-OQ | Joint-OQ，Value 含 cost | 0.476131 / 15.91 | 0.528447 / 39.78 | 0.559668 / 63.65 |
| ORIC-AP5095 | SharedAction-OQ | 0.449588 / 15.91 | 0.495165 / 39.78 | 0.526971 / 63.65 |
| MORIC-AP5095 | SharedAction-OQ | 0.452755 / 15.91 | 0.496918 / 39.78 | 0.528256 / 63.65 |
| Confidence | SharedAction-OQ | 0.447376 / 15.91 | 0.489725 / 39.78 | 0.520410 / 63.65 |
| Random（50 seeds） | SharedAction-OQ | 0.433699 / 15.91 | 0.468162 / 39.78 | 0.499313 / 63.65 |

CAVS 相对每个预算中最强 ORIC/MORIC 的点估计提升为 0.025469、0.031966 和 0.031030 AP；相对 Confidence 提高 0.030848、0.039159 和 0.038875；相对 Random 50-seed 均值提高 0.044525、0.060721 和 0.059973。三个工作点的字节几乎相同且容量利用率接近 100%，因此这些差异不是通过超预算获得。

同骨干 TwoStage 对照使机制归因更直接：Joint-OQ 分别提高 0.027850、0.036146 和 0.034582 AP，而网络表征、reward、训练种子和字节求解器均保持一致。该结果支持“图像成员关系与动作选择不可分离”的核心命题。它仍不能支持在新数据上的确认性 SOTA，因为 val2017 已被历史实验使用；CAVS 对冻结外部方法、Confidence、Random 与 TwoStage 的 1,000 次 paired bootstrap 为【待补：20260813 冻结统计完成后填入】。

### 7.2 CAVS 如何使用字节容量

**表 6 CAVS 在 COCO Fixed-byte 下的动作组成。**

| 容量 | AP | AP50 | AP75 | KB/图 | Original | Q90 | 实际上传率 | 上传图像数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.478224 | 0.653405 | 0.521689 | 15.91 | 0.76% | 23.78% | 24.54% | 1,227 |
| 25% | 0.528884 | 0.708478 | 0.578464 | 39.78 | 1.62% | 52.22% | 53.84% | 2,692 |
| 40% | 0.559286 | 0.738728 | 0.612046 | 63.65 | 6.26% | 72.94% | 79.20% | 3,960 |

名义 10% 容量实际处理了 24.54% 图像，40% 容量处理了 79.20% 图像，因为大部分被选动作是 Q90。随着容量增加，Original 比例从 0.76% 增至 6.26%，说明调度器并非固定使用最便宜动作，而是在高预算下为少数图像支付更高像素保真成本。这正是 Fixed-byte 与 Fixed-call 不可互换的原因。

### 7.3 Fixed-call 边界：CAVS 不是普遍最强 Ranker

**表 7 COCO Fixed-call 辅助结果（Official AP / KB per image）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-OQ-Core | **0.447997 / 16.07** | **0.489166 / 39.13** | **0.519119 / 62.49** |
| CAVS-OQ-Core-TwoStage | 0.447997 / 16.07 | 0.489166 / 39.13 | 0.519119 / 62.49 |
| ORIC-AP5095 | 0.446574 / 16.16 | 0.482896 / 38.50 | 0.513605 / 58.81 |
| MORIC-AP5095 | 0.445960 / 16.79 | 0.485589 / 39.24 | 0.515464 / 59.57 |
| Confidence | 0.432446 / 14.23 | 0.474322 / 37.37 | 0.505710 / 59.72 |
| Random（50 seeds） | 0.424777 / 12.29 | 0.448945 / 30.74 | 0.473705 / 49.37 |

在严格固定调用数时，CAVS 的点估计仅比最强外部方法高 0.001423、0.003577 和 0.003654 AP，明显小于 Fixed-byte 下的 0.025–0.032 AP 优势；统计显著性为【待补：最新 paired bootstrap】。更关键的是，CAVS-OQ-Core 与同骨干 TwoStage 在 Fixed-call 下完全相同：两者均选择固定数量的 Original，联合容量调度不再产生额外自由度。因而更准确的结论不是“CAVS 只是全面更强的 Ranker”，而是其主要优势来自图像成员关系、O/Q 动作与真实字节容量的联合适配。

### 7.4 价值—成本分离与跨动作校准

早期 Current-BudgetSet 将成本倾向混入排序分数；SetValue 和 DualHead 尝试拆分价值与成本，但仍先选图像、再由 SharedAction 决定载荷。早期开发诊断中，ActionValue 联合调度相对 DualHead+SharedAction 的 Fixed-byte AP 在 OOF 提高 0.029580、0.037365 和 0.037183，在冻结 calibration 提高 0.024484、0.040407 和 0.035591。最新完整 COCO 同骨干对照进一步得到 +0.027850、+0.036146 和 +0.034582，并排除了不同网络表征与训练种子这一混杂因素。

CostInValue-OQ 与去除成本输入的 CAVS-OQ-Core 共享联合调度结构。CostInValue 在 10%、25% 和 40% 的差值分别为 -0.002093、-0.000437 和 +0.000382 AP：成本从 Value 中去除在低预算明确更有利，在高预算基本持平。该结果支持“价值预测与容量成本分工”，但不支持“任何预算下加入成本都会退化”的绝对命题。

历史诊断还对 Original 与 Q90 分别进行了 fold-local isotonic calibration。开发 calibration 上该操作曾改善部分预算，但在对应的完整 val2017 run 中比未校准 ActionValue 低 0.001053、0.000416 和 0.000451 AP。因此，跨动作尺度误差确实存在，但单调后校准不是稳定升级；当前主实例保持独立训练、未做事后单调校准的 CAVS-OQ-Core。

风险/LCB 变体则产生相反极端：大多数云动作的保守值低于 Edge=0，调度器几乎不上传并导致 AP 崩溃。该诊断说明风险估计若直接替代任务价值，必须同时约束最低容量利用率或仅作为独立 accept/reject 机制；本文不把失败的 LCB head 纳入主方法。

### 7.5 ROI、AFPD 与 Occupancy Gate

ROI 诊断说明“ROI 一定有害”和“ROI 一定更省且更准”都不成立。为避免把 Profile、融合规则和学习器选择混在一起，我们补做了严格的 Profile×Fusion 2×2：Current@640 与 Historical-cost@640 两种扩框分别使用 Reversible 和 MAUR，各自基于真实融合结果重新生成 contextual reward，并以同一 5-fold、3-seed、全量 fit 和 Fixed-byte 调度器训练。

**表 8 COCO Profile×Fusion 2×2 严格全重训点结果（Fixed-byte AP / KB，post-hoc）。**

| ROI Profile | 融合 | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| Current@640 | Reversible | 0.479071 / 15.91 | 0.531817 / 39.78 | 0.559932 / 63.65 |
| **Current@640** | **MAUR** | **0.480277 / 15.91** | **0.532900 / 39.78** | 0.560178 / 63.65 |
| Historical-cost@640 | Reversible | 0.477784 / 15.91 | 0.531402 / 39.78 | 0.560025 / 63.65 |
| Historical-cost@640 | MAUR | 0.478155 / 15.91 | 0.531983 / 39.78 | **0.560701 / 63.65** |

该表目前是点估计，严格 15 组因果 bootstrap 为【待补：2×2 v2 完成后填入】，因此不能按评测集逐预算挑选最优 Profile×Fusion 组成新的主方法。作为更窄的执行规则消融，在冻结完全相同的 CAVS-OQR selections 后仅把 Reversible 换为 MAUR，10%、25% 和 40% 的 AP 分别提高 0.000919、0.000713 和 0.000167；95% CI 分别为 [0.000555, 0.001274]、[0.000350, 0.001068] 和 [-0.000065, 0.000396]。这证明 MAUR 在低/中预算的同选择执行上更强，但 40% 尚未确认，也不能替代按 MAUR reward 全重训的 2×2 结果。

AFPD（Action-Frontier Path Distillation）尝试监督每幅图像完整的 \(reward-\lambda bytes\) 动作上包络，Occupancy Gate 则进一步预测 ROI 的可占用前沿区域。此前独立重训中，AFPD-OQR 相对 Current-OQR 在 10% 为明确负差（均值 -0.004497，95% CI [-0.008982,-0.000263]），25% 与 40% 区间跨零；相对 Current-OQ 只有 40% 为明确正差（均值 +0.002936，95% CI [+0.001474,+0.004534]）。OccupancyFree/Aware 对 AFPD 的六个区间均跨零。由此，AFPD 有研究新意且在高预算可能利用更多 ROI，但目前不能取代 CAVS-OQ-Core；Occupancy Gate 也没有稳定增强证据。

### 7.6 Oregon 历史 canonical 诊断

在旧 canonical Oregon 检测器上，ActionValue-OQ-Joint 已显示与 COCO 相同的 Fixed-byte 趋势。表 9 仅使用 fit OOF 与 frozen calibration，不访问该 Track 的 final。

**表 9 Oregon canonical V1 Fixed-byte 诊断（AP / KB）。**

| 证据层 | 方法 | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| 5-fold OOF | Current-BudgetSet | 0.490038 / 148.02 | 0.499167 / 370.04 | 0.506361 / 592.06 |
| 5-fold OOF | SetValue | 0.491507 / 148.01 | 0.499988 / 370.01 | 0.505976 / 592.05 |
| 5-fold OOF | DualHead | 0.491459 / 148.01 | 0.500979 / 370.01 | 0.508267 / 592.07 |
| 5-fold OOF | ActionValue-OQ-Joint | **0.511769 / 148.02** | **0.511184 / 370.04** | **0.514295 / 592.07** |
| Frozen calibration | Current-BudgetSet | 0.468057 / 154.45 | 0.474473 / 386.02 | 0.478787 / 617.83 |
| Frozen calibration | SetValue | 0.462707 / 154.46 | 0.475274 / 386.13 | 0.479161 / 617.78 |
| Frozen calibration | DualHead | 0.462685 / 154.46 | 0.476772 / 386.14 | 0.478955 / 617.83 |
| Frozen calibration | ActionValue-OQ-Joint | **0.471968 / 154.45** | **0.479579 / 386.14** | **0.480083 / 617.83** |

在 calibration 上，ActionValue 相对 DualHead 提高 0.009283、0.002807 和 0.001128 AP。然而 Fixed-call 下 ActionValue 分别只有 0.433287、0.454844 和 0.464568，不及不同预算下的 Current-BudgetSet 或 SetValue。这与 COCO 的机制一致：优势来自真实字节约束下的联合动作调度，而非纯 Top-K 排序。

### 7.7 Oregon Cloud-V3 论文主结果

Cloud-V3 改变了云端错误分布和动作 reward，因此表 9 不能作为新模型的主结果。表 10 中所有学习器均使用 Cloud-V3 缓存从头训练；CAVS、ORIC 与 MORIC 使用三个种子 ensemble，Random 为 50-seed 均值。该 final 已被历史工作访问，因此下表是事后论文级点结果，不是新的注册终评。

**表 10 Oregon Cloud-V3 Fixed-byte 主结果（Official AP / KB）。**

| 方法 | 动作政策 | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| **CAVS-OQ-Core（本文）** | Joint-OQ | **0.499366 / 159.54** | **0.517611 / 398.86** | **0.519335 / 638.20** |
| DualHead | SharedAction-NoROI | 0.472613 / 159.55 | 0.497240 / 398.87 | 0.509971 / 638.19 |
| SetValue | SharedAction-NoROI | 0.472841 / 159.55 | 0.490655 / 398.87 | 0.509153 / 638.20 |
| Current-BudgetSet | SharedAction-NoROI | 0.467299 / 159.55 | 0.487911 / 398.87 | 0.507091 / 638.19 |
| ORIC-AP5095 | SharedAction-NoROI | 0.483153 / 159.55 | 0.502061 / 398.87 | 0.509599 / 638.20 |
| MORIC-AP5095 | SharedAction-NoROI | 0.479826 / 159.52 | 0.493286 / 398.87 | 0.497783 / 638.18 |
| Confidence | SharedAction-NoROI | 0.489643 / 159.55 | 0.502499 / 398.87 | 0.510449 / 638.19 |
| Random（50 seeds） | SharedAction-NoROI | 0.471087 / 159.51 | 0.484147 / 398.72 | 0.495973 / 638.03 |

**表 11 Oregon Cloud-V3 Fixed-call 辅助结果。**

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
|---|---:|---:|---:|
| CAVS-OQ-Core | 0.472458 / 140.97 | 0.491012 / 340.42 | 0.499795 / 526.05 |
| ORIC-AP5095 | **0.476378 / 127.94** | **0.492551 / 357.16** | 0.506444 / 563.48 |
| MORIC-AP5095 | 0.468654 / 145.85 | 0.483139 / 351.44 | 0.503429 / 550.87 |
| Confidence | 0.466649 / 80.38 | 0.486806 / 282.77 | **0.511141 / 529.49** |
| Random（50 seeds） | 0.459091 / 129.59 | 0.466056 / 323.69 | 0.474096 / 517.14 |

CAVS-OQ-Core 相对最强 ORIC/MORIC 的 Fixed-byte 点估计提高 0.016213、0.015550 和 0.009736 AP；相对每预算包含 Confidence 在内的最强比较器仍提高 0.009722、0.015112 和 0.008886。与 COCO 相反，Oregon Fixed-call 下 CAVS 在三个预算均不是最佳，说明大 payload 域中的收益同样依赖 O/Q 成本适配而非纯成员排序。Cloud-V3 的 image/event-cluster bootstrap、类别与目标尺寸分解为【待补：冻结统计完成后填入】。

统一 OQR 扩展还给出一个需要单独解释的点结果：同一 OQR checkpoint 屏蔽 ROI 后为 0.509129、0.518860 和 0.520862 AP，而开放 Current@640+Reversible 后为 0.509129、0.518854 和 0.520825，实际只选择 0/1/1 个 ROI。该结果不能命名为独立 OQ 训练，也不证明 ROI 有效；它说明在 Cloud-V3 Oregon 上，当前静态学习器几乎完全依赖 O/Q 动作。论文主表因此保留语义更干净的独立 CAVS-OQ-Core。

### 7.8 Oregon Adaptive ROI 的上界与部署差距

Cloud-V2 诊断曾扫描 4 个 ROI profile 与 160/320/640 三种云推理尺度。GT-aware 12 候选 teacher 在 40% ROI 上从 0.518897 AP、1470.61 KB/图提高到 0.567346 AP、938.41 KB/图，即 AP +0.048449 且字节 -532.20。该结果证明 Profile×Scale 动作空间具有互补性，但每张图的动作由 GT 选择，不能作为部署性能。

后续学习器采用破坏式 MAUR 替换时得到 0.517851 AP、1392.42 KB/图（ROI 7.66%），采用可逆融合时得到 0.518771 AP、1462.36 KB/图（ROI 0.86%）。可逆融合把 AP 损失从 0.001046 降至 0.000126，却几乎不再使用 ROI。这说明 Oracle 动作空间上限与静态可部署选择之间仍有巨大差距；当前主稿不把 Cloud-resolved Profile×Scale 写成已完成贡献。

### 7.9 真机与网络结果

**表 12 多带宽端到端结果。**

| 数据集 | 方法 | 1 Mbps AP / P95 | 5 Mbps AP / P95 | 10 Mbps AP / P95 | 20 Mbps AP / P95 | J/frame |
|---|---|---:|---:|---:|---:|---:|
| COCO | Edge-only | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| COCO | MORIC | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| COCO | CAVS | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Oregon | Edge-only | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Oregon | MORIC | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Oregon | CAVS | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

真机结果应回答两个不同问题：在相同通信预算下 CAVS 是否提高 AP；在相同 AP 或相同 deadline 下 CAVS 是否降低 wire bytes、P95 时延或能耗。只测策略 MLP 的微秒/毫秒开销不足以支持系统论文主张。

## 8 讨论

### 8.1 当前证据究竟证明了什么

为避免把机制推断写成事实，本节将结论分为三层。

**已验证事实。** 在已消费的 COCO val2017 Fixed-byte 表中，独立 CAVS-OQ-Core 的三个点估计均高于同协议三种子 ORIC/MORIC、Confidence、Random 与 TwoStage；所有方法使用几乎相同 KB/图。Joint 相对同骨干 TwoStage 提高 0.027850–0.036146 AP，而 Fixed-call 两者完全相同。Oregon Cloud-V3 从头重训后，CAVS-OQ-Core 在 Fixed-byte 三点均高于 ORIC/MORIC 与 Confidence，但 Fixed-call 三点均不是最佳。AFPD 与 Occupancy 未稳定替代 CAVS；Oregon OQR 也几乎不选择 ROI。

**由对照支持的机制推断。** Fixed-byte 提升远大于 Fixed-call，且 CAVS 大量选择 Q90、少量选择 Original，因此最合理的收益解释是“图像成员关系与动作成本联合适配”，而不是单纯提高困难图像排序。同骨干 Joint/TwoStage 完整评测比早期 DualHead 代理消融更强地支持这一解释。CostInValue 在低预算退化而高预算近似持平，进一步说明任务价值与字节成本应分工处理，但该设计并非所有预算下都严格占优。

**尚待验证的研究判断。** Oregon Cloud-V3 点结果已表明该机制在大 payload 域仍有价值，但其统计稳定性仍需 event/image-cluster bootstrap。字节优势能否在真机上转化为上行时延、能耗和 deadline 收益仍未知；只有完成冻结统计、RK3588 实测与新确认集后，才能形成确认性跨域结论。

### 8.2 为什么论文主线应是“成本感知端云协同”

“成本感知的端云协同目标检测”比“更强上传 Ranker”更准确。首先，CAVS 的 Fixed-byte 优势显著大于 Fixed-call，且在 Oregon Fixed-call 下不是最强；其主要优势在动作成本异质且预算以真实字节定义时出现。其次，调度器处理的是通信资源约束，而非单一分类分数。第三，未来 RK3588 原型还会同时面对编码、网络、云推理和 deadline 成本。

但“成本感知”不能只停留在标题。最终稿至少要满足三项证据：使用 actual payload/wire bytes，而非仅上传率；在多带宽下报告端到端时延；说明通信成本与云计算成本如何进入约束。若只保留离线 Fixed-byte AP 表而没有真机实验，更稳妥的题目是“基于上下文动作价值调度的预算约束端云目标检测”，而不是强调完整系统成本。

### 8.3 CAVS 的创新边界

CAVS 的独立创新不应表述为“首次使用双头网络”“首次使用 contextual reward”“首次上传 ROI”或“首次用背包优化卸载”。ORIC/MORIC 已定义 contextual image reward [13]，DCSB、Selective Cloud Offloading 与多种视频系统已处理区域传输 [12,14–16,34,35,39,40]，FrugalML 已研究预算化多服务选择 [32]。论文可主张的组合创新是：

1. 在检测级选择性卸载中，把图像—像素载荷动作作为统一价值预测对象；
2. 用 AP50–95 上下文边际奖励监督动作价值，同时从 Value head 去除直接字节输入；
3. 让 Edge、Original 与 Q90 在逐图真实字节容量下联合竞争，而不是先 Top-K 再选择载荷；
4. 用 Fixed-call 反事实对照明确证明优势来自成本—动作耦合，而非隐藏协议收益。

这一主张与 ORIC/MORIC 有清晰继承关系而非割裂关系：ORIC/MORIC 解决“图像价值如何更接近检测 mAP”，CAVS 进一步解决“一个图像存在多个成本和收益不同的云动作时，如何把价值预测用于容量决策”。论文应在引言、奖励定义和相关工作中主动承认这一继承，反而更能突出问题扩展的必要性。当前检索没有发现与上述完整组合同构的方法，但投稿措辞仍应使用“据我们检索到的相邻工作”，而不是无法验证的绝对首创。

AFPD 的 frontier-path 监督可能形成第二项创新，但目前数据不支持把它并列为主方法。若新注册实验只在 40% 高预算改善，应把 AFPD 定位为高预算 ROI 扩展；若三预算稳定超过 CAVS，再考虑将其提升为主方法。Occupancy Gate 当前没有统计依据继续发展。

### 8.4 跨数据集泛化与方法选择

COCO 与 Oregon 的价值不在于简单增加数据集数量，而在于提供不同成本结构。COCO 图像较小、类别多，Q90 使 10%/25%/40% 字节容量分别覆盖 24.54%/53.84%/79.20% 图像；Oregon Original 平均 1595.49 KB，而 Q90 仅 97.98 KB，动作成本差异更大，同时背景和事件相关性更强。CAVS 在两个域的 Fixed-byte 点估计都超过外部方法，而在 Oregon Fixed-call 下不占优，这种“跨域相同机制、不同纯排序表现”比简单声称全面 SOTA 更符合理论预期。

一个跨域共享 checkpoint 不是当前目标；每个云模型改变后重新训练轻量调度器是方法假设的一部分。因此，现有结果证明的是方法机制在两个独立域上的可复现性，不是模型参数的零样本跨域泛化。真正的泛化主张仍需要 domain-held-out、新注册集或跨云模型免重训实验，目前尚不具备。

### 8.5 有效性威胁

**评测集复用。** COCO val2017 和 Oregon 历史 final 已被多轮实验访问，存在研究者自由度和隐性选择偏差。当前数字只能称为事后完整评测。新注册 final 或外部 benchmark 是投稿前最高优先级。

**非可加奖励。** contextual reward 只是对数据集级 mAP 的局部近似。上下文大小、类别构成和匹配波动会影响标签；式（1）对代理价值可加，不代表 Official AP 可加。完整 evaluator 和 paired bootstrap 不能省略。

**外部基线复现。** ORIC/MORIC 必须在相同 detector outputs、fit split、动作缓存和预算上重新训练。若实现使用不同 top-k boxes、AP50/AP50–95 奖励或网络容量，应分别报告并做作者代码核对。

**模型改变。** Cloud-V3 不能复用 Cloud-V1/V2 学习器。RKNN 量化后的 Edge 输出也不能默认等价于 PyTorch 输出。忽略这一点会使学习器输入分布与 reward 同时错位。

**Oregon 相关性。** 图像级 bootstrap 可能把同一相机/事件的相关帧当成独立样本；主结果应增加 cluster bootstrap，并报告跨事件而非仅跨图像泛化。

**容量口径。** request payload bytes 不等于 wire bytes，也不包含下行、握手和重传。Fixed-byte 离线表证明的是 payload 容量优势；系统成本主张必须使用真实网络测量。

**求解器最优性。** 当前拉格朗日加贪心修复保证容量可行，但不保证一般离散多选背包全局最优。可在小规模子集上与动态规划/MILP 对比最优差距，结果为【待补：scheduler optimality gap】。

**动作空间。** OQ 不覆盖低质量高分辨率全图、多 ROI packing、特征传输和时序反馈。CAVS 可以扩展动作集合，但学习器能否可靠区分这些动作需要重新验证。

**ROI 新颖性与公平性。** 区域采样、任务感知 ROI、共形不确定区域选择和多 ROI packing 已有先例 [12,14–16,34,35,39,40]。若 CAVS-OQR 成为主结果，外部基线必须共享同一 ROI 可用性与 payload，或明确区分“同动作调度公平比较”和“完整系统比较”；不能把新增动作空间带来的上界全部归因于学习器。

**端侧前向成本。** CAVS 当前必须先运行 Edge 检测器，再根据其输出选择云动作。Budget-Adaptive Routing 表明高上传率时 weak-skipping 可能更省计算 [41]。因此本文不声称 CAVS 在所有预算下最小化端侧时延；真机实验应报告 Edge 前向重复成本并给出 Cloud-first/weak-skipping 边界。

**真机可重复性。** RK3588 驱动、RKNN Toolkit、量化配置、散热器和电源都会影响性能。最终需提供镜像、模型哈希、设备功耗测量方法和重复次数。

### 8.6 后续实验优先级

为了避免继续无边界地尝试 Ranker，后续只按以下顺序推进。

1. **完成当前冻结统计。** 不再重训主学习器，只完成 COCO 20260813 paired bootstrap、COCO 2×2 v2 因果 bootstrap，以及 Oregon image/event-cluster bootstrap，并把区间写回对应表格。
2. **完成 RK3588 同模型闭环。** 冻结 RKNN Edge 后重新生成学习器输入，不以桌面 cache 代替真机输出。
3. **建立新确认集。** 冻结 CAVS、外部比较器和成功门槛后再访问，用于确认当前 post-hoc 优势。
4. **只补复现性缺口。** 补充调度器最优差距、单模型均值±标准差、软件环境和数据许可；Joint/TwoStage、CostInValue、OQ/OQR 和融合消融已具备点结果，不再发展新变体。
5. **可选新动作。** 仅在前三项完成后再评估 Q70/Q80/Q90/Original 或窗口化在线策略；这些属于下一篇/下一协议，不回写当前方法选择。

## 9 结论

本文研究真实字节约束下的端云协同目标检测。核心观点是：当一张图像具有多个成本和检测收益不同的云载荷时，正确决策单元不是图像，而是图像—动作。基于这一观点，CAVS 使用端侧检测集合预测 Original/Q90 的 AP 对齐上下文价值，并由容量可行调度器在逐图实际字节下联合选择上传成员与动作。传统 Top-K 是单一等成本云动作下的特例。

当前 COCO val2017 事后完整点结果显示，独立 CAVS-OQ-Core 在 10%、25% 和 40% Fixed-byte 容量下取得 0.478224、0.528884 和 0.559286 AP，比同协议最强 ORIC/MORIC 高 0.025469、0.031966 和 0.031030。相同骨干下 Joint 相对 TwoStage 提高 0.027850、0.036146 和 0.034582 AP，而 Fixed-call 两者完全相同，直接支持收益来自图像—动作与字节容量的联合调度。Oregon Cloud-V3 从头重训的 CAVS-OQ-Core 取得 0.499366、0.517611 和 0.519335 AP，比最强 ORIC/MORIC 点估计高 0.016213、0.015550 和 0.009736；但在 Fixed-call 下不是最佳。最新 bootstrap、RK3588 真机性能和新注册终评仍为【待补】。

因此，当前最稳妥的论文结论不是“CAVS 已在所有场景全面超越现有方法”，而是：**动作条件的任务价值与显式字节约束能够揭示图像级 Top-K 路由忽略的精度—通信机会；这一机制在 COCO 与 Oregon 的事后 Fixed-byte 点结果中均成立，并由同骨干结构消融支持，但仍需冻结统计、新确认集和真机证据完成确认。**

## Statements and Declarations

### Funding

【待补：基金名称与项目编号；若无基金，使用期刊要求的无资助声明。】

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

【待补：按 CRediT taxonomy 填写 Conceptualization、Methodology、Software、Validation、Investigation、Writing、Supervision 等。】

### Data availability

COCO 可通过官方渠道获得 [28]。Oregon 数据的正式名称、许可、下载方式及敏感位置元数据处理为【待补：数据提供方授权与正式引文】。论文生成的 split manifests、缓存哈希、冻结协议和聚合结果将在许可允许的范围内公开。

### Code availability

【待补：匿名代码仓库、归档 DOI、环境镜像与公开时间。】复现包计划包含数据角色清单、检测器与 payload 哈希、reward 构建脚本、模型训练、容量调度、统一 evaluator、bootstrap 和 RK3588 部署代码。

### Ethics approval and consent

COCO 为公开研究数据。Oregon 野生动物相机数据不用于人物身份识别；涉及人物画面、私人地点或受保护物种位置时，将按数据许可进行匿名化和访问控制。【待补：机构伦理审批或“不适用”正式措辞。】

## 附录 A 理论补充

### A.1 Top-K 特例的完整证明

设每幅图像只有 Edge 与同成本云动作 \(g\)，且 \(c_i^g=c>0\)。忽略非正价值动作后，式（1）等价于选择集合 \(S\subseteq\{1,\ldots,N\}\)：

\[
\max_S\sum_{i\in S}\hat v_i^g,
\quad |S|c\le B.
\]

令 \(K=\min(N,\lfloor B/c\rfloor)\)。若一个最优集合 \(S\) 的大小小于 \(K\)，且存在未选择的正价值样本，则加入它会提高目标且仍可行，矛盾。因此最优集合含最多 K 个正价值样本。若 \(S\) 包含 \(j\) 而不包含 \(i\)，且 \(\hat v_i^g>\hat v_j^g\)，用 \(i\) 置换 \(j\) 不改变成本并严格提高目标，矛盾。故选择前 K 个正价值样本最优。若正价值样本少于 K，则剩余图像保留 Edge。

### A.2 异构动作下前缀排序的反例

考虑两个图像与三个预算。图像 1 有动作 \((c,v)=(10,10)\) 和 \((5,4)\)，图像 2 有动作 \((6,7)\)，两者均可选 Edge \((0,0)\)。预算 5 的最优解为图像 1 的便宜动作；预算 6 的最优解为图像 2；预算 10 的最优解为图像 1 的昂贵动作；预算 11 的最优解为图像 1 便宜动作与图像 2。由此，不同容量不仅改变图像成员关系，还改变同图动作。若一个静态图像排序以图像 1 为首，则预算 6 的最优前缀不成立；若以图像 2 为首，则预算 5 和 10 的最优前缀不成立。只有把动作与预算纳入选择，才能表示全部解。

### A.3 拉格朗日动作前沿

式（1）的拉格朗日松弛为

\[
L(x,\lambda)=\sum_{i,a}x_i^a(\hat v_i^a-\lambda c_i^a)+\lambda B.
\]

对固定 \(\lambda\)，约束 \(\sum_a x_i^a=1\) 使每图可独立取最大净值动作，即式（12）。不同 \(\lambda\) 扫描的是离散动作集合的 supported value-cost frontier。由于整数性和非凸离散点，某些容量最优解可能不是任何单一 \(\lambda\) 的直接解；贪心修复用于提高容量利用率，但不消除一般最优性间隙。

## 附录 B 实现与超参数

**表 B1 CAVS 当前冻结超参数。**

| 项目 | 值 |
|---|---:|
| 最大检测框数 | 64 |
| 类别嵌入维度 | 8 |
| Box hidden | 32 |
| Action hidden | 32 |
| Joint hidden | 64 |
| Dropout | 0.05 |
| Context size | 32 |
| 优化器 | AdamW |
| 初始学习率 | 1e-3 |
| Weight decay | 1e-4 |
| 最大 epoch | 60 |
| Early-stop patience | 8 |
| Reward / action-rank loss 权重 | 0.5 / 0.5 |
| OOF folds | 5 |
| 训练 seeds | 3 |
| 拉格朗日二分次数 | 80 |

【待补：三个固定随机种子、batch size、reward 标准化细节、tie-break 规则、payload 单位和完整模型参数量。】

## 附录 C 投稿前结果替换规则

1. 不直接覆盖 post-hoc 表；新 confirmatory 表单独编号并注明协议版本。
2. Cloud 模型或 Edge RKNN 发生变化时，必须重新生成动作缓存并重训所有依赖输出的学习器。
3. 任何在 calibration 或 final 后提出的新变体都进入下一协议，不回写当前冻结结果。
4. 主表优先报告预注册比较器；事后“每预算取最高方法”只作为探索性分析。
5. 所有 `【待补：...】` 在投稿前必须被数值、正式声明或“删除该主张”三者之一替换。

## 参考文献

[1] Kang, Y., Hauswald, J., Gao, C., et al. Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. *ASPLOS*, 615–629 (2017). https://doi.org/10.1145/3037697.3037698

[2] Matsubara, Y., Levorato, M., Restuccia, F. Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges. *ACM Computing Surveys* 55(5), 1–30 (2023). https://doi.org/10.1145/3527155

[3] Zhou, Z., Chen, X., Li, E., et al. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. *Proceedings of the IEEE* 107(8), 1738–1762 (2019). https://doi.org/10.1109/JPROC.2019.2918951

[4] Teerapittayanon, S., McDanel, B., Kung, H. T. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *ICPR*, 2464–2469 (2016). https://doi.org/10.1109/ICPR.2016.7900006

[5] Xu, R., Razavi, S., Zheng, R. Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques. *IEEE Communications Surveys & Tutorials* 25(4), 2951–2982 (2023). https://doi.org/10.1109/COMST.2023.3323091

[6] Ran, X., Chen, H., Zhu, X., Liu, Z., Chen, J. DeepDecision: A Mobile Deep Learning Framework for Edge Video Analytics. *IEEE INFOCOM*, 1421–1429 (2018). https://doi.org/10.1109/INFOCOM.2018.8485905

[7] Yi, S., Hao, Z., Zhang, Q., et al. LAVEA: Latency-Aware Video Analytics on Edge Computing Platform. *ACM/IEEE SEC*, 1–13 (2017). https://doi.org/10.1145/3132211.3134459

[8] Li, Y., Padmanabhan, A., Zhao, P., et al. Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. *ACM SIGCOMM*, 359–376 (2020). https://doi.org/10.1145/3387514.3405874

[9] Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M. NoScope: Optimizing Neural Network Queries over Video at Scale. *PVLDB* 10(11), 1586–1597 (2017). https://doi.org/10.14778/3137628.3137664

[10] Zhou, H.-Y., Gao, B.-B., Wu, J. Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors. *ICCV*, 3505–3513 (2017). https://doi.org/10.1109/ICCV.2017.379

[11] Cao, Z., Li, Z., Chen, Y., Pan, H., Hu, Y., Liu, J. Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. *IEEE ICDCS*, 259–270 (2023). https://doi.org/10.1109/ICDCS57875.2023.00062

[12] Cao, Z., Cheng, Y., Zhou, Z., et al. Edge-Cloud Collaborated Object Detection via Bandwidth Adaptive Difficult-Case Discriminator. *IEEE Transactions on Mobile Computing* 24(2), 1181–1196 (2025). https://doi.org/10.1109/TMC.2024.3474743

[13] Qiu, J., Wang, R., Hu, B., Guérin, R., Lu, C. Optimizing Edge Offloading Decisions for Object Detection. *ACM/IEEE Symposium on Edge Computing* (2024). https://doi.org/10.1109/SEC62691.2024.00021

[14] Chen, T., Bu, Y., Zeng, Y., Xie, L., Lu, S. RegionFilter: Region-Aware Video Filtering Mechanism on Resource-Constrained Edge Nodes. *Computer Networks* 251, 110624 (2024). https://doi.org/10.1016/j.comnet.2024.110624

[15] Yang, Z., Wang, X., Wu, J., et al. EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision. *IEEE/ACM Transactions on Networking* 31(4), 1765–1778 (2023). https://doi.org/10.1109/TNET.2022.3223412

[16] Du, K., Zhang, Q., Arapin, A., et al. AccMPEG: Optimizing Video Encoding for Accurate Video Analytics. *Proceedings of Machine Learning and Systems* 4 (2022). https://proceedings.mlsys.org/paper_files/paper/2022/hash/853f7b3615411c82a2ae439ab8c4c96e-Abstract.html

[17] Mozannar, H., Sontag, D. Consistent Estimators for Learning to Defer to an Expert. *ICML*, PMLR 119, 7076–7087 (2020). https://proceedings.mlr.press/v119/mozannar20b.html

[18] Verma, R., Nalisnick, E. Calibrated Learning to Defer with One-vs-All Classifiers. *ICML*, PMLR 162 (2022). https://proceedings.mlr.press/v162/verma22c.html

[19] Mao, A., Mohri, M., Zhong, Y. Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees for Learning to Defer. *ICML*, PMLR 267, 43035–43066 (2025). https://proceedings.mlr.press/v267/mao25c.html

[20] Gomez, C., Patkhullaev, D., Hernandez, A. C. OffloaD: Detection Failure-Based Scheduler for Offloading Object Detection. *ACM Transactions on Embedded Computing Systems* 23(5), 1–26 (2024). https://doi.org/10.1145/3677321

[21] Said, N., Landsiedel, O. EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading. *Computer Networks* 269, 111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437

[22] Choi, H., Bajic, I. V. Deep Feature Compression for Collaborative Object Detection. *IEEE ICIP*, 3743–3747 (2018). https://arxiv.org/abs/1802.03931

[23] Elgamal, T., Shi, S., Gupta, V., Jana, R., Nahrstedt, K. SiEVE: Semantically Encoded Video Analytics on Edge and Cloud. *ICDCS* (2020). https://doi.org/10.1109/ICDCS47774.2020.00182

[24] Jitkrittum, W., Gupta, N., Menon, A. K., Narasimhan, H., Rawat, A. S., Kumar, S. When Does Confidence-Based Cascade Deferral Suffice? *NeurIPS* 36, 9891–9906 (2023).

[25] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. *ICML*, PMLR 70, 1321–1330 (2017). https://proceedings.mlr.press/v70/guo17a.html

[26] Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., Lei, L. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *The Annals of Applied Statistics* 19(2) (2025). https://doi.org/10.1214/24-AOAS1998

[27] Efron, B., Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman & Hall/CRC (1993).

[28] Lin, T.-Y., Maire, M., Belongie, S., et al. Microsoft COCO: Common Objects in Context. *ECCV*, 740–755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

[29] Jocher, G., Qiu, J., Liu, M., et al. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748

[30] Jiang, J., Ananthanarayanan, G., Bodik, P., Sen, S., Stoica, I. Chameleon: Scalable Adaptation of Video Analytics. *ACM SIGCOMM*, 253–266 (2018). https://doi.org/10.1145/3230543.3230574

[31] Matsubara, Y., Callegaro, D., Baidya, S., Levorato, M., Singh, S. Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks. *ICPR* (2021). https://arxiv.org/abs/2007.15818

[32] Chen, L., Zaharia, M., Zou, J. FrugalML: How to Use ML Prediction APIs More Accurately and Cheaply. *NeurIPS* 33 (2020). https://arxiv.org/abs/2006.07512

[33] Du, K., Liu, Y., Hao, Y., et al. OneAdapt: Fast Configuration Adaptation for Video Analytics Applications via Backpropagation. arXiv:2310.02422 (2023). https://arxiv.org/abs/2310.02422

[34] Chaudhary, S., Taneja, A., Singh, A., et al. TileClipper: Lightweight Selection of Regions of Interest from Videos for Traffic Surveillance. *USENIX ATC*, 967–984 (2024). https://www.usenix.org/conference/atc24/presentation/chaudhary

[35] Wang, W., Mi, L., Cen, S., et al. Region-based Content Enhancement for Efficient Video Analytics at the Edge. *USENIX NSDI*, 613–633 (2025). https://www.usenix.org/conference/nsdi25/presentation/wang-weijun

[36] Elmachtoub, A. N., Grigas, P. Smart “Predict, then Optimize”. *Management Science* 68(1), 9–26 (2022). https://doi.org/10.1287/mnsc.2020.3922

[37] Orange Pi. Orange Pi 5 Ultra Hardware Specifications. https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-Ultra.html (accessed 2026-08-11).

[38] Hu, Y., Li, Z., Chen, Y., et al. Content-Aware Adaptive Device–Cloud Collaborative Inference for Object Detection. *IEEE Internet of Things Journal* 10(21), 19087–19101 (2023). https://doi.org/10.1109/JIOT.2023.3279579

[39] Dehghani, D., Chen, Y., Yu, X., Koudas, N. Selective Cloud Offloading for Accurate and Efficient Object Detection. *IEEE ICDM*, 1145–1154 (2025). https://doi.org/10.1109/ICDM65498.2025.00123

[40] Wang, H., Li, Q., Sun, H., et al. VaBUS: Edge-Cloud Real-Time Video Analytics via Background Understanding and Subtraction. *IEEE Journal on Selected Areas in Communications* 41(1), 90–106 (2023). https://doi.org/10.1109/JSAC.2022.3221995

[41] Geng, W., Mohan, N., Ott, J. Budget-Adaptive Routing: Skipping the Weak When the Strong Answers Anyway. *ACM SIGCOMM NAIC Workshop* (2026). https://doi.org/10.1145/3789240.3828740
