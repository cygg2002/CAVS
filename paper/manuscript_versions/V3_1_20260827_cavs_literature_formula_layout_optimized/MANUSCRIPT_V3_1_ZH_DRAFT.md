# CAVS：严格字节预算下基于上下文图像—动作价值的端云协同目标检测

**CAVS: Contextual Image–Action Value Scheduling for Edge–Cloud Object Detection under Hard Byte Budgets**

[作者姓名 1]¹，[作者姓名 2]²，[通讯作者]¹*  
¹[单位、城市、邮编、国家]  
²[单位、城市、邮编、国家]  
*[通讯作者邮箱]

> **稿件状态（2026-08-27）：V3.1 文献、公式与版式优化稿。** 本稿已根据 ORIC/MORIC、Budget-Adaptive Routing、TURBO 及任务感知传输工作收紧创新边界，并按实际训练与在线控制代码校正符号。已核验的 COCO、Oregon、VOC 点结果和 RK3588 先导结果已填入；尚未完成的统一 bootstrap、正式零前视在线矩阵、能耗与多网络条件结果均以 `【待补：…】` 明示。三个公开 final/test 均曾被开发过程访问，因此当前统计属于事后完整评测，不表述为新的隐藏终评或确认性 SOTA。该说明在正式投稿时删除，并由实验协议声明替代。

## 摘要

端云协同目标检测不仅要判断一幅图像是否值得调用云端，还要决定上传原图、压缩图或感兴趣区域等何种像素载荷。现有检测级选择性卸载通常学习图像分数并控制上传率；系统级配置优化虽可选择压缩或云模型，却不直接估计每幅图像在不同像素载荷下对数据集级检测 AP 的边际贡献。本文提出上下文动作价值调度器 CAVS（Contextual Action-Value Scheduler），将决策单元由“图像”推广为“图像—动作”。CAVS 使用端侧检测集合的置换不变表示，学习与 AP50–95 对齐的动作条件边际价值；载荷字节不进入价值网络，而由外部调度器在严格容量约束下显式处理。我们给出两种互补实现：全局 Fixed-byte 调度刻画批量容量上限，零前视因果控制器则在每帧到达后立即选择并执行动作。

我们在 COCO、Oregon 野生动物和 PASCAL VOC 三个数据集上，以全量原图字节的 10%、25% 和 40% 为预算进行评测。完整 CAVS-OQR 在 COCO 上取得 0.480277、0.532900 和 0.560178 AP50–95，在 Oregon 上取得 0.506028、0.517405 和 0.516663，在跨检测器架构的 VOC 配置上取得 0.531486、0.579678 和 0.612721。相较同协议 MORIC，COCO 与 VOC 的点估计分别提高 0.028048–0.035986 和 0.020509–0.027070 AP；Oregon 相较每档最强的 ORIC/MORIC/Confidence 比较器提高 0.006214–0.016385 AP。同骨干消融表明，联合图像—动作估值较“先排序图像、再选择载荷”在 COCO 上提高 0.027850–0.036146 AP。真实 Orange Pi 5 Ultra/RK3588 与云 GPU 的零前视先导实验中，CAVS-OQR 在三档预算取得 0.454842、0.505302 和 0.529707 AP，并比在线 ORIC 高 0.015829–0.040814 AP；但其相对部署对齐离线上限仍有 0.024253–0.030482 AP 的差距。结果说明，动作条件价值与逐样本字节调度能够改善异构载荷下的精度—通信权衡，同时也揭示全局容量优化转化为严格因果在线策略时的实际代价。统一配对统计与正式系统矩阵仍为【待补：完成后替换摘要中的点估计措辞】。

**关键词：** 端云协同；目标检测；选择性卸载；动作价值学习；严格字节预算；因果在线调度；RK3588

## Abstract

Edge–cloud object detection must decide not only whether an image should invoke a cloud detector, but also which pixel payload—such as the original image, a compressed image, or a region of interest—should be transmitted. Detection-oriented selective offloading typically learns an image score and controls an invocation ratio, whereas system-level configuration optimizers do not directly estimate each image’s AP contribution under alternative payloads. We present CAVS, a Contextual Action-Value Scheduler that promotes the decision unit from an image to an image–action pair. CAVS encodes edge detections as a permutation-invariant set and learns action-conditioned marginal values aligned with AP50–95. Encoded bytes are excluded from the value network and handled explicitly by a scheduler under a hard capacity constraint. A global fixed-byte scheduler characterizes the batch capacity frontier, while a zero-lookahead causal controller selects and immediately executes an action upon each frame arrival.

We evaluate CAVS on COCO, the Oregon wildlife dataset, and PASCAL VOC at 10%, 25%, and 40% of the Always-Original byte budget. The full CAVS-OQR configuration achieves AP50–95 scores of 0.480277/0.532900/0.560178 on COCO, 0.506028/0.517405/0.516663 on Oregon, and 0.531486/0.579678/0.612721 on the cross-architecture VOC setting. Against protocol-matched MORIC, the COCO and VOC point-estimate gains are 0.028048–0.035986 and 0.020509–0.027070 AP, respectively; on Oregon, CAVS exceeds the strongest budget-wise ORIC/MORIC/Confidence comparator by 0.006214–0.016385 AP. A controlled same-backbone ablation shows that joint image–action valuation improves over image-ranking-then-payload-selection by 0.027850–0.036146 AP on COCO. In a pilot zero-lookahead Orange Pi 5 Ultra/RK3588–GPU deployment, CAVS-OQR obtains 0.454842/0.505302/0.529707 AP and exceeds online ORIC by 0.015829–0.040814 AP, while remaining 0.024253–0.030482 AP below its deployment-aligned offline frontier. These results support action-conditioned valuation and per-sample byte accounting for heterogeneous-payload offloading, while exposing the practical cost of translating a global capacity solution into a strictly causal online policy. The unified paired statistics and final system matrix are **[to be completed]**.

**Keywords:** edge–cloud collaboration; object detection; selective offloading; action-value learning; hard byte budget; causal online scheduling; RK3588

## 1 引言

轻量端侧检测器使智能相机、野外监测节点和移动机器人能够在本地持续分析图像，但受模型容量、输入分辨率、功耗和端侧算子支持限制，其检测质量通常弱于云端大模型。将全部图像上传云端能够提高准确率，却会持续占用上行带宽，并引入编码、排队、传输、云推理和结果回传开销。端云协同检测因此不是简单的端/云二选一，而是在通信资源约束下为每个输入选择执行路径的问题 [1–7,35]。

已有选择性卸载研究主要回答“哪些图像值得上传”。DCSB 使用难例判别器识别端侧容易漏检的图像 [8,9]；OffloaD 结合检测失败与时延分数调节卸载 [10]；ORIC/MORIC 则指出，目标检测的 mAP 是跨图像和跨类别聚合的非可加指标，单图置信度不等于该图像对最终 mAP 的贡献，并据此学习上下文图像奖励 [11]。Budget-Adaptive Routing 进一步研究上传预算变化时 weak-conditioned 与 weak-skipping 路由位置的切换 [26]。这些工作显著推进了检测级路由，但其主要决策仍是单个云动作是否执行。

真实系统还必须回答“上传什么”。同一图像可以保留端侧结果，也可以上传 Original、JPEG Q90，或上传由端侧框构造的 ROI。不同载荷的检测收益和编码字节均随图像变化。例如，Q90 可能以小幅精度代价容纳更多请求；ROI 可能放大小目标，也可能丢失上下文或放大端侧定位误差。在这种异构动作空间中，先给图像一个与动作无关的分数，再在入选图像中分配载荷，会把“是否上传”和“上传哪种载荷”这两个耦合问题人为拆开。

本文研究如下问题：**当载荷收益和成本均依赖图像时，能否直接学习每个图像—动作的检测价值，并在真实字节容量下联合决定上传成员与载荷类型？** 为此，我们提出 CAVS。CAVS 将端侧检测集合编码为紧凑语义表示，对 Original、Q90 和可选 ROI 分别预测相对 Edge 的上下文 AP 边际价值，再由容量调度器读取实际编码字节完成联合选择。价值网络不直接使用载荷大小，从而保持“检测价值”与“通信价格”的语义分离。

传统 Top-K 并非错误，而是该问题的一个特殊情形：若每张图像只有一个等成本云动作，则严格字节预算等价于固定调用数，选取最高价值的 $K$ 幅图像即可。只要存在多个成本不同且相对收益随图像变化的动作，一个预算无关的图像排序通常无法表示全部最优选择。CAVS 的核心不是把 Ranker 做得更深，而是改变预测与优化的基本单元。

离线全局调度与真实在线部署还具有不同信息结构。全局调度先获得一个评测批次中全部图像的价值和成本，再求解总容量；它适合衡量策略的批量容量上限，却能看到未来。零前视在线策略必须在当前帧到达后立即决定，不能等待若干帧或查看后续图像，并且要满足前缀预算安全。本文明确分开两种口径：前者用于算法机制比较，后者用于 RK3588 硬件在环验证。

本文的贡献归纳为三点。

1. **检测级图像—动作问题定义。** 我们将 Edge、Original、Q90 和可选 ROI 统一为带逐图实测字节的多选容量问题，区分 Fixed-call、全局 Fixed-byte 与因果在线预算，并给出图像 Top-K 成为正确特例的条件。

2. **AP 对齐且价值—成本解耦的 CAVS。** 我们把 ORIC/MORIC 的 contextual image reward 推广为 contextual image–action reward，使用端侧检测集合学习每幅图像在不同像素载荷下的检测价值，并在调度层以真实编码字节联合决定成员与载荷。本文不把 JPEG、ROI、集合池化、上下文奖励、ILP/背包或拉格朗日价格单独宣称为首创；创新主张限定为面向检测 AP 的逐图异构载荷价值建模及其与逐样本字节调度的衔接。

3. **从离线容量上限到零前视真机执行的分层证据。** 我们给出容量可行的批量调度器和逐帧因果控制器，并在 COCO、Oregon、VOC 三个数据集以及 SSDLite→Faster R-CNN 的跨架构配置上检验方法；RK3588–GPU 先导系统进一步量化离线—在线差距。跨架构测试沿用 ORIC/MORIC 的稳健性评测思想，本文新增证据在于多载荷与严格字节协议，而非首次比较不同检测器架构。

本文余下部分安排如下：第 2 节回顾相关工作；第 3 节定义系统与问题；第 4 节介绍 CAVS；第 5 节描述全局与因果在线实现；第 6 节给出实验方法；第 7 节报告结果；第 8 节讨论适用边界；第 9 节总结有效性威胁；第 10 节给出结论。

## 2 相关工作

### 2.1 端云协同与选择性卸载

Neurosurgeon、split computing 和边缘智能研究主要决定神经网络在哪一层或哪一设备执行 [1–4]。NoScope、Reducto、Chameleon 和 OneAdapt 等视频分析系统利用专用过滤器、时间冗余或周期配置降低计算与传输成本 [5–7,12]。这类系统通常依赖视频时序或网络分割，而本文假设端侧检测结果始终可独立使用，研究检测完成后的样本级升级。

DCSB 从多目标和小目标更易漏检的观察出发训练 difficult-case discriminator，其带宽自适应扩展还包含 regional sampling 与判别器切换 [8,9]；DCCI 进一步结合难例判断、设备模型压缩与运行时网络调度 [36]。OffloaD 联合检测失败风险和时延指标，并在真实平台及受损网络中报告准确率—时延 [10]。EdgeBoost 和级联理论说明，经校准的置信度仍是必须认真比较的强基线 [27,37]。ORIC/MORIC 用上下文替换定义一幅图像对整体 mAP 的边际贡献，并以端侧 Top proposals 训练轻量 MLP；其真实测试台采用 Jetson Xavier NX、RTX 3090 和 1-GbE [11]。本文继承 AP 对齐的奖励思想，但把二元 Edge/Cloud 选择推广为动作条件的载荷选择，并以实际字节而非仅上传图像比例作为主要约束。

### 2.2 任务感知传输与区域处理

任务感知编码和区域传输通过空间不均匀质量减少无关像素。AccMPEG 使用准确率梯度指导宏块质量 [13]；EdgeDuet、TileClipper、RegenHance、RegionFilter 和 VaBUS 分别利用切片、ROI、区域增强或背景模型降低视频分析传输 [14–18]。Selective Cloud Offloading 使用不确定性区域和 packing 改善云端目标检测 [19]。Hawlader 等在真实硬件与网络仿真中比较多档 JPEG/H.265，表明压缩质量会同时改变检测精度与端到端时延，过强压缩甚至可能使云端路径劣于本地 [39]。这些研究说明“并非所有像素等价”，也说明 ROI 与多质量编码本身已有充分先例。

CAVS 与上述工作的差别是优化粒度：它不固定使用一种区域/编码规则，而是把多种已存在的像素载荷作为竞争动作，由统一的检测价值和字节预算决定是否使用。因而，ROI 在本文中是可选动作而非创新本体；若其收益不稳定，调度器应能够少选或不选，而不是为了证明模块有效而强制使用。

### 2.3 学习延期、预测后优化与资源约束控制

Learning to defer 和多专家路由研究学习何时交给更强专家 [20–22]；FrugalML 研究预算化 API 选择 [23]；predict-then-optimize 则强调预测误差应根据下游决策影响评估 [24]。这些工作为动作条件价值提供一般背景，但目标检测 AP 的非可加性、像素载荷成本和端云融合使本文问题不能直接退化为单样本分类损失。

实时边缘分类中的 token-bucket 路由明确区分平均速率、短时 burst 和因果逐样本决策 [25]。路由分数的后校准可以改善阈值含义，但不能保证下游非可加 AP 最优 [28]。Budget-Adaptive Routing 进一步指出，weak-conditioned router 在高上传预算仍需为最终上传样本支付弱模型前向开销，并以 0.153-GFLOP 的 weak-skipping estimator 在高预算降低最多约 19.1 ms 延迟 [26]。CAVS 当前依赖端侧检测框生成语义特征与 ROI，属于 weak-conditioned 路线；本文将其高预算计算税列为系统边界，不声称在所有预算下端侧计算最优。

TURBO 从系统级角度联合选择云模型、压缩配置和跨服务带宽，并以 utility curve 与 ILP 在实时网络条件下满足 SLO [38]。它说明“多配置＋资源优化”本身并非本文独有。TURBO 的决策单位是服务/模型配置，其效用来自离线 profile；CAVS 的决策单位是当前图像—像素载荷，监督信号是 contextual AP，成本是每幅图像实际编码字节。两者分别位于系统配置层和检测样本层，可形成互补而非互斥的控制层。

### 2.4 与最接近方法的关系

表 1 从任务信号、配置空间、成本口径和决策粒度比较邻近方法。ORIC/MORIC 是最直接的检测路由基线；TURBO 是最接近的多配置资源优化系统；AccMPEG 与区域方法则是载荷侧参照。

**表 1 与相邻工作的结构比较。**

| 方法 | 任务价值信号 | 动作/配置空间 | 主要成本口径 | 决策粒度 | 部署语义 |
|---|---|---|---|---|---|
| DCSB / bandwidth-adaptive DCSB [8,9] | 难例判别 | Edge/Cloud，区域扩展 | 上传率/带宽 | 图像或区域 | 在线，真实平台 |
| ORIC/MORIC [11] | contextual mAP reward | Edge/Original | offload ratio | 图像 | 冻结阈值逐图执行 |
| Budget-Adaptive Routing [26] | contextual AP target | weak-conditioned / weak-skipping | offload ratio＋计算 | 图像与路由位置 | 预算自适应逐图路由 |
| AccMPEG [13] | 任务准确率代理 | 宏块质量 | 带宽/时延 | 空间块 | 在线编码 |
| TURBO [38] | profiled control utility | 云模型＋压缩配置 | 带宽＋SLO | 服务—配置 | 动态运行时，实车验证 |
| Selective Cloud Offloading [19] | 不确定性/检测质量 | ROI packing | 传输/API 成本 | 区域 | 可在线 |
| **CAVS（本文）** | **AP50–95 contextual reward** | **Edge/O/Q/ROI** | **每图编码字节** | **图像—载荷** | **全局上限＋零前视 HIL** |

据本文检索到的相邻工作，尚未见在**单图检测路由**中同时学习 AP 对齐的图像—像素载荷价值，并以每图实测编码字节联合选择样本成员与载荷的方法。该范围明确排除 TURBO 一类服务级多配置优化，也不把多质量压缩、ROI、ILP/背包或真实硬件单独视为首创；表述限定于已检索文献，不使用无法证实的“绝对首次”。

## 3 系统模型与问题定义

### 3.1 端云检测与动作空间

给定输入序列或批次 $\mathcal D=\{I_i\}_{i=1}^{N}$，端侧检测器 $D_e$ 对当前图像产生检测集合

\[
E_i=D_e(I_i)=\{(b_{ij},y_{ij},p_{ij})\}_{j=1}^{n_i},
\tag{1}
\]

其中 $b_{ij}$、$y_{ij}$ 和 $p_{ij}$ 分别为框、类别和置信度。$E_i$ 同时是未上传时的最终预测，也是路由器的语义输入。候选动作集合为

\[
\mathcal A_i=\{e,o,q,r\},
\tag{2}
\]

其中 $e$ 表示保留 Edge，$o$ 表示上传原始 JPEG，$q$ 表示上传 quality=90 的 JPEG，$r$ 表示上传由端侧框构造并映射回原图的 ROI。O、OQ 和 OQR 分别表示只开放 {Original}、{Original, Q90} 和 {Original, Q90, ROI} 的云动作空间。

对云动作 $a\ne e$，编码器产生载荷 $P_i^a$，其 payload 成本为 $c_i^a=|P_i^a|$。云检测器 $D_c$ 输出 $C_i^a$，经坐标映射与融合得到 $F_i^a$。Edge 满足 $F_i^e=E_i$、$c_i^e=0$。云请求失败或超时时系统回退到 $F_i^e$。payload bytes 与 HTTP wire bytes 分开记录，避免把 Base64 或请求头开销隐藏在预算定义之外。

![图 1 CAVS 系统与信息流。](figures/fig_cavs_system_overview.svg)

**图 1 CAVS 系统与信息流。** 价值头仅接收端侧检测语义与去成本动作描述；实际编码字节只进入调度器。虚线区分离线监督生成、全局调度和零前视部署。

### 3.2 全局严格字节问题

令 $\hat v_i^a$ 为动作 $a$ 相对 Edge 的预测任务价值，$x_i^a\in\{0,1\}$ 为动作选择变量。全局 Fixed-byte 决策写为

\[
\begin{aligned}
\max_x\quad & \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a\hat v_i^a,\\
\text{s.t.}\quad & \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a c_i^a\le B,\\
& \sum_{a\in\mathcal A_i}x_i^a=1,\quad \forall i,\\
& x_i^a\in\{0,1\}.
\end{aligned}
\tag{3}
\]

该问题具有多选背包结构。由于检测 AP 不可逐图相加，式（3）的目标是可学习代理而非对最终 AP 的等价分解；最终质量始终由完整预测集合的 COCO-style evaluator 计算。

本文以 Always-Original 的总 payload 为参考，定义

\[
B_\rho=\rho\sum_{i=1}^{N} c_i^o,\qquad \rho\in\{0.10,0.25,0.40\}.
\tag{4}
\]

因此 10% 表示原图总字节的 10%，并不表示只能上传 10% 图像。Q90 或 ROI 较小时，实际云调用率可以显著高于名义预算比例。

### 3.3 Fixed-call 与 Top-K 特例

Fixed-call 固定云调用数 $K=\lfloor\rho N\rfloor$，用于隔离纯图像排序能力。若每图只有一个成本相同的云动作，则式（3）等价于选择价值最大的 $K$ 幅图像；这正是 Top-K。若动作成本或同图动作价值次序不同，则静态图像排序一般不足。考虑图像 1 的两个动作 $(c,v)=(10,10)$、$(5,4)$，图像 2 的动作 $(6,7)$：预算 5、6、10 与 11 分别要求不同的图像成员和动作组合，不存在一个预算无关前缀同时覆盖四个最优解。

### 3.4 零前视因果在线问题

在线时，令 $\mathcal F_t$ 表示作出第 $t$ 个决策时可观测的信息。因果策略要求 $a_t$ 对 $\mathcal F_t$ 可测：

\[
a_t=\pi(E_t,s_t;\theta),\qquad
\mathcal F_t=\sigma\!\left(I_{1:t},E_{1:t},a_{1:t-1},c_{1:t}^{\cdot},s_t\right),
\quad a_t\text{ is }\mathcal F_t\text{-measurable},
\tag{5}
\]

其中 $c_{1:t}^{\cdot}$ 表示截至当前帧已经构造的候选成本，$s_t$ 是累计字节与影子价格等控制状态；$\mathcal F_t$ 不含 $I_{t+1:N}$。本文对“实时在线”的操作性定义为：`buffer_frames=0`、`zero_lookahead=true`、`global_plan=false`；当前帧决定后立即编码和上传，提交结果后才读取下一帧。顺序读取 COCO 文件不等于摄像头采集，但只要决策不访问未来帧，算法仍是因果的。

在线控制还要求前缀安全。对前 $t$ 帧，令累计参考原图字节为 $C_t^o=\sum_{\tau\le t}c_\tau^o$、累计已花费字节为 $S_t=\sum_{\tau\le t}c_\tau^{a_\tau}$，则要求

\[
S_t\le \left\lfloor\rho C_t^o\right\rfloor+\beta_t,
\qquad \forall t\in\{1,\ldots,N\},
\tag{6}
\]

其中 $\beta_t$ 是运行前声明且与未来输入无关的 burst envelope。本文主零缓冲协议使用严格前缀模式 $\beta_t\equiv0$；paced-burst 仅作为控制器消融并单独报告。该约束与板端控制器中基于整数分子/分母的可行动作屏蔽一致，也比只检查第 $N$ 帧总预算更接近真实链路。

## 4 CAVS：上下文图像—动作价值学习

### 4.1 方法概览

CAVS 由检测集合编码器、动作条件价值网络和成本外置调度器组成。端侧检测框经共享编码与池化形成 $h_i$；动作描述 $u_i^a$ 经动作编码器形成嵌入；联合网络输出 $\hat v_i^a$。部署阶段另行编码候选载荷获得 $c_i^a$，再交给全局或在线调度器。该分工保证一个动作“便宜”不会在训练标签中被误当成“检测价值高”。

### 4.2 检测集合编码

每幅图像保留置信度最高的 $M=64$ 个端侧检测。框特征包含归一化中心、宽高、面积与置信度：

\[
r_{ij}=[\bar x_c,\bar y_c,\bar w,\bar h,\bar w\bar h,p]_{ij}.
\tag{7}
\]

类别使用 8 维可学习嵌入。共享 MLP 将每个检测映射为 $h_{ij}$，再连接均值池化和最大池化：

\[
h_i^{set}=
\begin{cases}
\left[\dfrac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}\;\middle\|\;
\max_{1\le j\le n_i}h_{ij}\right], & n_i>0,\\[4pt]
\mathbf 0, & n_i=0.
\end{cases}
\tag{8}
\]

辅助场景特征包括图像宽高比、框数量、置信度统计、检测框覆盖面积以及低成本亮度/边缘密度。空检测集使用显式掩码和零池化。当前网络采用 class embedding 8、box hidden 32、action hidden 32、joint hidden 64 和 dropout 0.05，以控制端侧额外计算。

### 4.3 动作表示与成本解耦

动作描述包含动作 ID、目标输入尺度、ROI 数量和归一化几何统计。虽然数据管线记录 payload ratio 等成本信息，主 CAVS 的 Value head 使用去成本动作表示。网络在训练折标准化空间输出 $\hat{\bar v}_i^a$，再用该模型的训练折统计量反变换为调度价值：

\[
\tilde u_i^a=\operatorname{StripCost}(u_i^a),\qquad
\hat{\bar v}_i^a=f_\theta(h_i^{set},z_i^{scene},\tilde u_i^a),\qquad
\hat v_i^a=\sigma_{\mathrm{tr}}\hat{\bar v}_i^a+\mu_{\mathrm{tr}}.
\tag{9}
\]

Edge 在反变换后强制设 $\hat v_i^e=0$。Original、Q90 和 ROI 共享参数，使模型可以学习同图动作相对次序。三个 full-fit 模型分别反标准化后再对 $\hat v_i^a$ 求均值，避免把不同种子的标准化坐标直接混合。真实 $c_i^a$ 只在式（3）、式（13）或在线控制器中出现。

### 4.4 AP50–95 上下文动作奖励

ORIC/MORIC 已说明 mAP 对齐奖励需要在图像上下文中定义 [11]。本文把替换变量由“弱/强检测”扩展到动作 $a$。对 fit 中的目标图像 $i$，从同一训练折构造不含 $i$ 的上下文索引集 $\mathcal C_i$，当前大小为 32，并令 $F_{\mathcal C_i}^{e}=\bigcup_{j\in\mathcal C_i}F_j^e$。若 $\operatorname{AP}(\cdot)$ 表示 COCO-style AP50–95，则

\[
v_i^a=(|\mathcal C_i|+1)\left[
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^a)-
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^e)
\right].
\tag{10}
\]

上下文与标注只用于离线生成监督；部署不访问 GT。5-fold OOF 中，reward 上下文、标准化统计和模型训练均限制在训练折内。式（10）仍是对完整 AP 的局部代理，因此我们以最终完整 evaluator 而非 reward 回归误差作为主指标。

### 4.5 训练目标

价值网络同时学习奖励回归与同图动作排序。标准化奖励为 $\bar v_i^a=(v_i^a-\mu_{\mathrm{tr}})/\sigma_{\mathrm{tr}}$；对可用动作集合 $\Omega$，采用 Smooth L1：

\[
\mathcal L_{reg}=\frac{1}{|\Omega|}\sum_{(i,a)\in\Omega}
\operatorname{SmoothL1}(\hat{\bar v}_i^a,\bar v_i^a).
\tag{11}
\]

若 $\bar v_i^a>\bar v_i^b$，动作排序项为

\[
\mathcal L_{rank}=
\frac{\sum_{(i,a,b)\in\Pi}w_i^{ab}
\log\!\left(1+e^{-\operatorname{sgn}(\bar v_i^a-\bar v_i^b)
(\hat{\bar v}_i^a-\hat{\bar v}_i^b)}\right)}
{\sum_{(i,a,b)\in\Pi}w_i^{ab}},
\quad w_i^{ab}=|\bar v_i^a-\bar v_i^b|.
\tag{12}
\]

式（12）按有效 pair 的权重和归一化，与实现一致。总损失为 $0.5\mathcal L_{reg}+0.5\mathcal L_{rank}$。训练使用 AdamW，batch size 192、初始学习率 $10^{-3}$、权重衰减 $10^{-4}$、最多 50 epoch、early-stopping patience 8，并将梯度范数裁剪为 5。每个正式配置使用 5-fold OOF 评估开发稳定性，并以三个固定种子在完整 fit 上重训；最终 value 为三个反标准化模型预测的均值。

### 4.6 ROI 动作与 MAUR 融合

Current@640 ROI 从端侧检测框出发，经类别无关合并与面积相关扩框产生裁剪区域，送入云检测器的 640 输入。云端框映射回原图后，MAUR（匹配感知联合替换）保留 crop 外 Edge 框，并用 crop 内云预测替换相应端侧结果；实现必须裁剪到合法图像边界并过滤非有限或零面积框。ROI 几何、云推理和融合共同决定 $F_i^r$，因此 OQR 学习器必须使用真实 ROI+MAUR reward 从头训练，不能把 OQ checkpoint 临时开放 ROI 后命名为完整 OQR。

ROI 并非被强制使用。若其预测净价值不及 Original、Q90 或 Edge，调度器可以在某数据集或预算中选择零个 ROI。这个设计对跨域稳定性很重要，也使 O/OQ/OQR 成为动作空间而非网络容量消融。

## 5 调度与系统实现

### 5.1 全局容量可行调度

给定影子价格 $\lambda$，每幅图像独立选择

\[
a_i(\lambda)=\arg\max_{a\in\mathcal A_i}
\left(\hat v_i^a-\lambda c_i^a\right).
\tag{13}
\]

实现以 80 次二分寻找不超 $B$ 的最小惩罚可行解，再枚举从当前动作到其他动作的正价值升级，按边际价值/增量字节执行仍可行的升级，最后强制断言总成本不超过容量。该算法在实验规模上获得接近 100% 的容量利用率，但它是容量可行的拉格朗日—修复求解器，不宣称对一般整数多选背包精确最优。

**算法 1 全局 Fixed-byte 调度。**

```text
输入：所有图像的动作价值 v_hat[i,a]、真实字节 c[i,a]、容量 B
1  将 Edge 设为 value=0, cost=0
2  二分 lambda，并令 a[i] = argmax_a(v_hat[i,a]-lambda*c[i,a])
3  保留总成本不超过 B 的最小惩罚解
4  枚举正价值、正增量字节的动作升级
5  按边际价值/增量字节依次加入仍可行的升级
6  断言 sum_i c[i,a[i]] <= B，返回全部动作
```

### 5.2 零前视在线控制

在线控制器不先评分完整数据集。当前帧 $t$ 到达后，系统运行 RKNN Edge，实时构造候选载荷和价值，并依据当前价格 $\lambda_t$ 选择

\[
a_t=\arg\max_a\left(\hat v_t^a-\lambda_t c_t^a\right),
\tag{14}
\]

同时由 token/credit 状态屏蔽会导致前缀超预算的动作。动作立即执行，云端响应融合完成后才读取下一帧。控制价格可采用从 calibration 冻结的常数，也可按累计消耗误差进行受限 primal–dual 更新：

\[
\lambda_{t+1}=\left[\lambda_t+\eta\left(c_t^{a_t}-\rho c_t^o\right)\right]_+.
\tag{15}
\]

现有诊断表明，自适应价格相对冻结在线策略只恢复 0.000000、0.000085 和 0.001168 AP，不能消除 0.024–0.030 AP 的离线—在线差距。因此主稿把它作为控制器消融，而不包装为已经解决在线损失的新方法。

### 5.3 RK3588–云硬件链路

板端为 Orange Pi 5 Ultra / RK3588 [34]，端侧检测器以 RKNN-FP16 执行。每帧流程为：读取图像 → RKNN 推理 → CAVS/基线决策 → 必要时编码 Original/Q90/ROI → HTTP 上传云 GPU → 云检测 → 坐标恢复与 MAUR → 提交预测。运行合同固定序列、检测器哈希、策略哈希、预算控制文件和服务协议哈希，防止跨 run 复用旧候选或响应。

正式传输协议应使用二进制或 multipart；若仍使用 JSON Base64，则论文同时报告 JPEG payload 与真实 HTTP body/wire bytes。云端在返回前裁剪框到合法边界并过滤 $x_1\ge x_2$、$y_1\ge y_2$ 或非有限分数；客户端继续严格验证，失败时整帧回退 Edge。延迟分别记录 Edge、策略、编码、上传 RTT、云推理、融合与 E2E 的 mean/P50/P95/P99。

### 5.4 复杂度与高预算边界

Value 网络至多编码 64 个框。对当前顺序执行实现，令 $p_a(\rho)$ 为预算 $\rho$ 下动作 $a$ 的经验频率，则平均单帧时延可分解为

\[
\bar T_{\mathrm{CAVS}}(\rho)=
\bar T_e+\bar T_r+
\sum_{a\in\mathcal A\setminus\{e\}}p_a(\rho)
\left(\bar T_{\mathrm{enc}}^a+\bar T_{\mathrm{wire}}^a+
\bar T_c^a+\bar T_{\mathrm{fuse}}^a\right).
\tag{16}
\]

该式是均值分解，不替代 P95/P99 或流水线吞吐测量。模型参数量、MAC/FLOPs、RK3588 上 Value CPU mean/P95、峰值内存以及式（16）各项的实测值为【待补】。全局调度复杂度近似为 $O(TN|\mathcal A|+U\log U)$，在线单帧动作选择为 $O(|\mathcal A|)$。CAVS 在决定上传前始终运行弱检测器，因此高预算下存在 weak-forward tax；与 cloud-first/weak-skipping 的公平延迟拐点为【待补：同设备、同云端、同网络下的高预算系统对照】。

## 6 实验方法

### 6.1 研究问题

实验围绕六个问题展开：

- **RQ1：** 在相同 Fixed-byte 容量下，CAVS 是否优于 ORIC/MORIC、Confidence 与 Random？
- **RQ2：** 收益是否来自图像—动作联合决策，而非更大的网络或隐藏字节差异？
- **RQ3：** Q90 和 ROI 分别贡献什么，动作扩展是否跨域稳定？
- **RQ4：** 方法能否跨数据域和检测器架构复现？
- **RQ5：** 全局容量上限转化为零前视在线控制后损失多少？
- **RQ6：** 在 RK3588 与真实网络中，准确率、payload、wire bytes、时延、吞吐和能耗如何权衡？

### 6.2 数据集与角色隔离

**COCO。** 使用 COCO [30] train2017 的 canonical fit=9,678、calibration=571 和 val2017=5,000。Edge/Cloud 为冻结的 YOLO26n/YOLO26x [33]，统一推理配置和 evaluator。val2017 已被开发历史访问，本文称其为 post-hoc full evaluation。投稿归档还需给出 fit/calibration 的确定性抽样规则、完整 image-ID 列表及 SHA-256【待补】。

**Oregon。** 使用相机陷阱野生动物数据，45 类；fit=9,923、calibration=4,961、final=4,962，按相机事件进行不交叉划分。云端采用 Cloud-V3 epoch-28（SHA-256 `d3890a57...4642d9`），其训练日志为 precision 0.86874、recall 0.67210、AP50 0.74694、AP50–95 0.54282。所有依赖云输出的学习器均在 Cloud-V3 缓存上重训。final 已被开发历史访问；统计主单位应为 event cluster，image bootstrap 仅作敏感性分析。投稿前还需披露端侧检测器的准确架构与哈希、事件构造时间阈值、各 split 的相机/事件数、事件大小分布和类别频次【待补】。

**VOC。** 使用 PASCAL VOC [31] 的 VOC2007+2012 trainval 构造 fit=15,629、calibration=922，并以 VOC2007 test=4,952 评测。端侧为 Torchvision/PyTorch [32] 的 `ssdlite320_mobilenet_v3_large` COCO_V1，云端为 `fasterrcnn_resnet50_fpn_v2` COCO_V1；二者使用官方权重且不训练检测器。ORIC 已用 SSD-MobileNetV3/Faster R-CNN 等检测器对检查其奖励估计的跨架构稳健性 [11]；本文沿用这一评测思想，但考察 AP50–95、多像素载荷与逐图字节调度。VOC 的作用是验证方法流程的跨架构适用性，不是声称首次比较异构检测器或使用专门 VOC 权重刷新检测器 SOTA。test 同样是公开且已访问的 post-hoc benchmark。COCO→VOC 的 20 类映射、被忽略类别、框/分数后处理、输入归一化和权重版本/哈希为【待补：归档表】。

**表 2 三数据集协议。**

| 数据集 | 类别 | fit / calibration / final | Edge → Cloud | 主要验证角色 |
|---|---:|---:|---|---|
| COCO | 80 | 9,678 / 571 / 5,000 | YOLO26n → YOLO26x | 核心机制与消融 |
| Oregon | 45 | 9,923 / 4,961 / 4,962 | 冻结端模型 → Cloud-V3 | 领域化、事件相关性 |
| VOC | 20 | 15,629 / 922 / 4,952 | SSDLite-MNV3 → Faster R-CNN-R50-FPN-V2 | 跨架构迁移 |

### 6.3 检测、编码与评价协议

主质量指标为 COCO-style AP50–95，同时报告 AP50 与 AP75。Original 直接使用冻结原始 JPEG；Q90 使用统一 JPEG quality=90、`optimize=True`、`subsampling=2`；ROI 使用 Current profile 与 MAUR。所有方法共享相同动作预测缓存、字节表、容量分母和 evaluator。Fixed-byte 容量按式（4）定义；实际 payload 必须不超 $B_\rho$，并报告容量利用率。系统实验另报 HTTP wire bytes，不能用 payload 预算掩盖 Base64 膨胀。由于一次 ROI 动作可能包含多个 crop，正式成本表还必须分开报告“云动作图像数、HTTP 请求数、crop 数、云 GPU 总毫秒与返回字节”，不能把调用图片数等同于云计算量【待补】。

### 6.4 比较方法

- **Edge-only / Always-Original / Always-Q90：** 端点与上界参考。
- **Random-50：** 50 个预固定随机种子分别形成 selection，主表报告 AP 均值与跨种子标准差；paired bootstrap 必须在每个重采样内评价 50 套 selection 后再求均值。
- **Confidence：** 由端侧置信度构造图像优先级，并使用同一 SharedAction 或在线预算控制。
- **DCSB：** 难例分类器重实现；在 VOC 表作为附加基线。
- **ORIC/MORIC：** 按论文定义重实现图像级 contextual reward/估值，并与统一动作执行器衔接。本文不声称作者官方 checkpoint 的逐位复现。
- **TwoStage：** 与 CAVS 使用同一检测集合骨干、训练数据和种子，先产生图像标量，再用 SharedAction 分配载荷，用于隔离联合图像—动作结构。
- **CAVS-O / OQ / OQR：** 分别独立生成真实 reward 并从头训练，不通过屏蔽完整模型动作冒充独立消融。

### 6.5 训练、冻结与统计

每个学习器使用 5-fold OOF × 3 seeds，并在完整 fit 上训练 3 个模型形成 ensemble。calibration 仅用于冻结超参数、动作控制与在线价格；final/test 不进入训练、阈值或动作选择。检测器或推理协议改变时，必须重建候选与 reward 并重训相关学习器。除 ensemble 主结果外，投稿前还应报告三个单模型 AP 的均值±SD、ensemble 增益、各动作 Spearman/Kendall、top-value recall 与 value–cost frontier regret【待补】。

主比较采用 1,000 次 paired image bootstrap [29]；Oregon 主分析采用 1,000 次 event-cluster bootstrap。每档报告 AP 差值均值、百分位 95% CI 和双侧方向比例。只有区间排除 0 时使用“统计支持”或“显著”；否则仅称点估计更高。当前完成状态如下：COCO Profile×Fusion 15 组 bootstrap 已完成；COCO 最新主比较、Oregon 主比较和 VOC 主比较为【待补】。

### 6.6 硬件在环协议

批处理 HIL 先在板端生成整批价值再做全局调度，真实执行 RKNN 和 HTTP，但不称为实时在线。因果在线 HIL 要求 `zero_lookahead=true`、`global_plan=false`，逐帧立即上传。正式矩阵为 CAVS、ORIC、MORIC、Confidence、Random × 10%/25%/40%，必须在同一 RK3588、云 GPU、网络路径和传输协议下串行执行，以免争抢 NPU/GPU 污染延迟。每组验收条件为 5,000/5,000、云失败 0、非法框 0、前缀预算违规 0 和运行合同哈希完整。

## 7 结果与分析

### 7.1 三数据集 Fixed-byte 主结果

图 2 和表 3 汇总当前已核验的三数据集点结果。为保持一个预先定义的完整系统配置，主行统一报告 CAVS-OQR；OQ 作为动作空间消融单独分析。不同数据集不共享检测器 checkpoint 或 CAVS checkpoint，验证的是方法流程的可复现性，而非参数的零样本迁移。

![图 2 三数据集 Fixed-byte 结果。](figures/fig_fixed_byte_three_datasets.svg)

**图 2 三数据集 Fixed-byte AP50–95 点结果。** CAVS 使用 OQR 完整动作空间；各比较方法共享所在数据集的检测缓存、容量分母和 evaluator。误差条将在统一 bootstrap 完成后加入【待补】。

**表 3 三数据集 Fixed-byte AP50–95 主结果（事后完整点估计）。**

| 数据集 / 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **COCO — CAVS-OQR** | **0.480277** | **0.532900** | **0.560178** |
| COCO — MORIC-AP5095 | 0.452230 | 0.496914 | 0.528170 |
| COCO — ORIC-AP5095 | 0.449174 | 0.494970 | 0.527069 |
| COCO — Confidence | 0.447029 | 0.489546 | 0.520311 |
| COCO — Random-50 | 0.433586 | 0.468908 | 0.500443 |
| **Oregon — CAVS-OQR** | **0.506028** | **0.517405** | **0.516663** |
| Oregon — MORIC-AP5095 | 0.479826 | 0.493286 | 0.506171 |
| Oregon — ORIC-AP5095 | 0.483153 | 0.502061 | 0.509599 |
| Oregon — Confidence | 0.489643 | 0.502499 | 0.510449 |
| Oregon — Random-50 | 0.471087 | 0.484147 | 0.495973 |
| **VOC — CAVS-OQR** | **0.531486** | **0.579678** | **0.612721** |
| VOC — MORIC-AP5095 | 0.504416 | 0.554956 | 0.592212 |
| VOC — ORIC-AP5095 | 0.502039 | 0.553636 | 0.589734 |
| VOC — Confidence | 0.495439 | 0.532334 | 0.569091 |
| VOC — Random-50 | 0.482501 | 0.503851 | 0.532850 |

COCO 上，CAVS-OQR 相对 MORIC 的点估计提升为 0.028048、0.035986 和 0.032009 AP；相对 Confidence 提高 0.033248、0.043355 和 0.039867。VOC 上，相对 MORIC 的提升为 0.027070、0.024722 和 0.020509 AP。Oregon 的最强比较器随预算变化：三档均为 Confidence，CAVS-OQR 分别高 0.016385、0.014906 和 0.006214 AP。由于 COCO/VOC 主 paired bootstrap 与 Oregon event-cluster bootstrap 尚未闭环，上述只能称为一致的正点估计，不能写成三数据集统计显著。

三组结果提供互补证据。COCO 与 VOC 中，预算增加带来单调 AP 上升；Oregon 在 25% 后饱和并轻微回落，说明更高通信容量不保证静态价值代理能继续改善非可加 AP。CAVS 的优化目标是预测价值和的容量代理，Edge 动作又始终可用，但有限样本学习误差仍可能让额外动作替换少量有用的端侧预测。因此，部署端若要求单调性能，需要在 calibration 上增加保守接受或嵌套策略约束，而不能从式（3）直接推导真实 AP 单调性。

### 7.2 COCO：核心机制和容量使用

COCO CAVS-OQR 的 AP50/AP75 分别为 0.656669/0.521907、0.712500/0.582650 和 0.738890/0.612353。表 4 给出动作分布。三档容量利用率均超过 99.997%，说明优势不是通过隐藏超预算或大量闲置产生。

**表 4 COCO CAVS-OQR 动作分布。**

| 预算 | Edge | Original | Q90 | ROI | 云动作数 | 容量利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 3,591 | 29 | 1,076 | 304 | 1,409 | 99.9976% |
| 25% | 2,049 | 63 | 2,468 | 420 | 2,951 | 99.9985% |
| 40% | 848 | 209 | 3,651 | 292 | 4,152 | 99.9996% |

名义 10% 容量已经允许 28.18% 图像调用云端，40% 时达到 83.04%，这是压缩和 ROI 动作的正常结果，而非预算口径变化。低预算中，Q90 与 ROI 用更少字节扩大云覆盖；高预算中 Original 数量增加，说明价值/成本权衡会随影子价格变化。

### 7.3 联合图像—动作估值的因果消融

同骨干 TwoStage 消融与 CAVS-OQ 使用相同的 top-64 集合编码、optimizer、fold、训练种子、O/Q reward 和 Fixed-byte 求解器；唯一关键变化是 TwoStage 先输出图像标量，再由 SharedAction 选择载荷，而 CAVS 直接预测每个图像—动作价值。

**表 5 COCO 同骨干结构消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| TwoStage：图像排序 → SharedAction | 0.450374 | 0.492738 | 0.524704 |
| CAVS-OQ：Joint image–action | **0.478224** | **0.528884** | **0.559286** |
| 差值 | **+0.027850** | **+0.036146** | **+0.034582** |

这组对照比“换不同 Ranker”更能定位核心贡献：网络骨干、输入与训练预算未增加，提升来自价值预测和容量决策粒度的改变。早期 OOF/calibration 诊断也给出同方向的 +0.029580/+0.037365/+0.037183 与 +0.024484/+0.040407/+0.035591，但正文主张以表 5 的完整评测为准。最新三种子结构消融的统一 paired CI 为【待补】。

### 7.4 O/OQ/OQR 动作空间消融

VOC 提供最干净的三档动作空间结果，因为 O、OQ、OQR 均按各自 reward 独立执行 5-fold×3 seeds 和 full-fit×3 seeds。端点 AP 为 Edge 0.473322、Original 0.657511、Q90 0.653684、ROI+MAUR 0.499819。

![图 3 VOC 动作空间消融。](figures/fig_voc_action_space_ablation.svg)

**图 3 VOC 动作空间消融。** （a）O、OQ 与 OQR 的绝对 AP50–95；（b）新增 Q90 与新增 ROI 的边际点增益。ROI 在 40% 预算为负，说明动作可用不等于动作必然有益；误差条待统一 paired bootstrap 完成后加入。

**表 6 VOC 跨架构动作空间消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-O | 0.520249 | 0.557280 | 0.589427 |
| CAVS-OQ | 0.530787 | 0.579142 | **0.614054** |
| CAVS-OQR | **0.531486** | **0.579678** | 0.612721 |
| OQ − O | +0.010538 | +0.021862 | +0.024627 |
| OQR − OQ | +0.000699 | +0.000537 | −0.001333 |

Q90 在三档均有较大的正点估计，且收益随容量增加而增大；ROI 只在 10%/25% 提供不足 0.001 AP 的小幅正差，40% 反而下降 0.001333。动作数也支持该解释：OQR 在 10%/25%/40% 只选择 77/49/3 个 ROI，而 Q90 为 796/1,816/2,796。由此，跨架构增益主要来自 OQ 异构成本，而不是 ROI。

Oregon 的现象更极端。独立 CAVS-OQ 为 0.499366/0.517611/0.519335；CAVS-OQR 为 0.506028/0.517405/0.516663，但三档最终均没有选择 ROI。二者差异来自独立训练后形成了不同的 O/Q selection，而非 ROI 执行收益。因此，OQR 在 Oregon 低预算更高不能用来宣称 ROI 有效；中高预算 OQ 反而更稳。统一框架的正确主张是“ROI 可以参与竞争”，不是“ROI 在所有域都应提高精度”。

### 7.5 COCO Profile × Fusion 2×2

为分离 ROI 几何与融合规则，我们在同一 COCO 划分、云模型、fold、种子和调度器下，对 Current@640/Historical-cost@640 与 Reversible/MAUR 四组各自重建真实 reward 并全量重训。

**表 7 COCO Profile × Fusion 严格全重训（AP50–95）。**

| ROI profile | Fusion | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| Current@640 | Reversible | 0.479071 | 0.531817 | 0.559932 |
| Current@640 | MAUR | **0.480277** | **0.532900** | 0.560178 |
| Historical-cost@640 | Reversible | 0.477784 | 0.531402 | 0.560025 |
| Historical-cost@640 | MAUR | 0.478155 | 0.531983 | **0.560701** |

15 个效应、每个 1,000 次 bootstrap 已完成。Current 下 MAUR−Reversible 的 AP 差值均值和 95% CI 为：10% +0.001162 [−0.001916,+0.003460]，25% +0.001097 [−0.000061,+0.002212]，40% +0.000216 [−0.000607,+0.000946]。三档区间均跨 0，故不能声称 MAUR 全面显著优于 Reversible。点结果支持 Current+MAUR 作为预先固定的主配置，但不支持按每档事后挑选四格最大值。

### 7.6 Oregon：领域化与预算饱和

Oregon Original 平均 1,595.49 KB，而 Q90 平均 97.98 KB，动作成本比 COCO 更不均衡。Edge、Always-Original 和 Always-Q90 的 AP 分别为 0.454815、0.520966 和 0.509918。表 3 中，CAVS-OQR 在 10% 已达到 0.506028，距离 Always-Original 仅 0.014938；25% 后性能基本饱和。独立 CAVS-OQ 的动作组成见表 8。

**表 8 Oregon Cloud-V3 CAVS-OQ 动作组成。**

| 预算 | Original | Q90 | 云动作数 | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|
| 10% | 1,002 | 2,199 | 3,201 | 0.700449 | 0.559539 |
| 25% | 2,171 | 1,927 | 4,098 | 0.716757 | 0.575467 |
| 40% | 2,891 | 1,567 | 4,458 | 0.716114 | 0.577230 |

预算增加时 Original 替换 Q90，但 AP50 在 25% 后轻微下降，进一步说明模型输出域、融合误差和 AP 非可加性会破坏“更多字节必然更准”的直觉。Oregon 的统计相关单位是相机事件而非独立图片；在 event-cluster bootstrap 完成前，只报告点估计，不把 0.006–0.016 AP 优势写成显著。

### 7.7 VOC：跨检测器架构适用性

VOC 的 Edge/Cloud 分别来自 one-stage SSDLite 与 two-stage Faster R-CNN，且均为官方 COCO 权重。CAVS-OQR 相对 MORIC 的点估计提高 0.027070、0.024722、0.020509 AP，相对 Confidence 提高 0.036047、0.047344、0.043630 AP。该结果削弱了“CAVS 的收益只来自 YOLO 特定置信度或 head 表征”的解释，但由于 ORIC 已报告跨检测器对稳健性 [11]，本文不把“跨架构评测”本身列为独立算法创新。

不过，这不是 checkpoint 的零样本迁移：动作 reward 和轻量学习器在新的端/云检测器输出上重新训练。本文据此主张**跨检测器架构的方法适用性**，而不是参数级跨架构泛化。VOC 1,000 次 paired bootstrap、Random-50 nested bootstrap 与单模型均值±SD 为【待补】。

### 7.8 RK3588 批处理硬件在环

表 9 的两阶段全局 HIL 使用真实 RKNN-FP16 和真实 HTTP 云执行，但先完成全量评分再调度，因而是硬件在环批处理而非因果在线。

**表 9 COCO RK3588 两阶段全局 HIL（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR** | **0.461068** | **0.503569** | **0.530318** |
| ORIC | 0.439772 | 0.470365 | 0.489860 |
| MORIC | 0.430747 | 0.461580 | 0.486673 |
| Confidence | 0.427594 | 0.461527 | 0.486238 |

CAVS 相对 ORIC 提高 0.021296、0.033204 和 0.040458 AP，说明在 RKNN 检测域中仍保持比较优势；但绝对值低于论文 PyTorch 缓存域。该差异来自端侧模型后端、单图预处理、由端侧框诱导的 ROI 几何和重新训练后的 selection，而不是 HTTP 本身：此前 Mac PyTorch 单图协议中，对应缓存与真实 HTTP 的 25% AP 仅相差 0.000061。

### 7.9 零前视因果在线先导结果

![图 4 COCO 离线到在线的精度差距。](figures/fig_coco_offline_to_online_gap.svg)

**图 4 COCO 从缓存协议到零前视部署的精度阶梯。** （a）论文缓存、部署对齐离线、RK3588 全局 HIL 与因果在线先导；（b）部署对齐离线到因果在线的 AP50–95 差值。在线值仍带云回退、退化框过滤和 Base64 协议 caveat，不能作为最终注册系统结果。

**表 10 COCO 因果在线 HIL 先导点结果（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR** | **0.454842** | **0.505302** | **0.529707** |
| ORIC | 0.439013 | 0.468441 | 0.488893 |
| CAVS − ORIC | +0.015829 | +0.036861 | +0.040814 |

该运行满足零前视与逐帧立即执行，但仍是先导证据：各档存在 1–5 次云响应回退，评测临时过滤了极少量退化零面积框，JSON Base64 使 wire bytes 约为 JPEG payload 的 1.34 倍，并且 MORIC/Confidence/Random 的完整同协议 15 组矩阵尚未完成。因此，表 10 可用于分析在线机制，不能替代最终系统主表。

部署对齐离线 CAVS 为 0.479095、0.531125、0.560189，在线差距为 0.024253、0.025823、0.030482 AP。这个差值不是单一“网络传输精度损失”：HTTP 不改变图像内容时几乎不影响 AP；主要结构差别是全局调度可以比较完整 5,000 张图像的价值—成本前沿，而在线控制器必须在未知未来下用冻结价格与前缀预算逐帧决定。此外，在线 CAVS 的四动作价值与 ROI 几何均依赖 RKNN 输出，错误会同时影响价值、动作和融合；ORIC 仅做 Edge/Original 阈值式决策，因而相对自身离线参考的损失更小。

现有 primal–dual 更新只在 25%/40% 恢复 0.000085/0.001168 AP，说明在线差距不能仅靠微调价格消除。更可能的改进方向包括 calibration 上的 distributionally robust price、基于历史而非未来的分位数估计、可证明 regret 的多动作 online knapsack、限幅 credit 和场景漂移检测；任何新控制器必须在新的冻结序列上评测，不能根据这 5,000 张 final 事后选择。

### 7.10 在线系统时延与字节口径

当前最完整的 25% Mac 云先导 run 共处理 5,000 帧，AP 为 0.505289，容量利用率 99.02%，前缀预算违规为 0。总耗时 1,328.54 s（3.76 FPS），平均 E2E 253.91 ms，P95 为 462.37 ms；Edge-only 帧平均 112.23 ms，上传帧平均 346.33 ms，上传 RTT P50/P95/P99 为 178.34/391.22/901.07 ms。动作分布为 Edge 1,974、Q90 2,430、ROI 530、Original 66。

JPEG payload 为 201,680,614 B，对应 25% 容量 203,676,291 B 的 99.02%；实际 HTTP body 为 270,294,246 B，即 payload 的约 1.34 倍。若论文预算定义为 JPEG payload，该 run 满足 25%；若定义为链路实际字节，则其 wire 占原始总字节约 33.18%。正式系统必须改为二进制/multipart 或显式将 wire bytes 纳入控制器。

**表 11 正式零前视系统矩阵（投稿前替换）。**

| 方法 | 预算 | AP50–95 | payload / wire MB | E2E mean / P95 ms | FPS | J/frame | deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAVS | 10% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| CAVS | 25% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| CAVS | 40% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ORIC | 10/25/40% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| MORIC | 10/25/40% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Confidence | 10/25/40% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Random | 10/25/40% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

**【待补图 5：正式在线系统结果。】** 左图绘制五种方法的 AP50–95–wire-byte Pareto 曲线；中图用堆叠柱分解 Edge、CAVS、编码、RTT、云推理与融合的 mean/P95；右图报告 1/5/10/20 Mbps 或实测多网络条件下的 deadline miss。数据完成前不根据先导 run 绘制终稿图。

**【待补图 6：定性动作案例。】** 每个数据集各选择一组由 calibration 规则预定义的成功/失败案例，依次展示 Edge 框、CAVS 价值、动作字节、所选载荷和最终云/融合框；必须同时包含 ROI 有益、ROI 无益、Q90 优于 Original 性价比以及在线控制误选四类，避免只展示正例。

## 8 讨论

### 8.1 结果支持了什么

第一，三个数据集的点估计均支持 CAVS 在 Fixed-byte 下优于图像级 ORIC/MORIC、Confidence 和 Random。第二，同骨干 Joint/TwoStage 的 0.028–0.036 AP 差值直接支持“图像成员与动作选择需要联合”的核心机制。第三，VOC 的 SSDLite→Faster R-CNN 结果说明该机制不依赖单一 YOLO 组合。第四，真实 RK3588 批处理和在线先导中，CAVS 对 ORIC 的比较优势仍存在。

这些事实不等于四个更强的主张：当前不能称三数据集确认性 SOTA；不能称 ROI 普遍有效；不能称离线结果可以被真实在线逐位复现；也不能称通信预算已经等价于真实 wire 成本。主 paired/bootstrap、完整统一在线矩阵和新注册确认集分别对应这些证据缺口。

### 8.2 为什么主线是字节成本感知端云协同，而非更强 Ranker

CAVS 的核心优势出现在动作成本异构且容量按字节定义时。Fixed-call 移除“同一字节容纳更多 Q90/ROI”的自由度后，联合结构优势会显著缩小；Oregon 的 Fixed-call 诊断甚至不由 CAVS 领先。因此最准确的叙事是：CAVS 学习动作条件检测价值，并由资源层根据实际编码字节决定如何购买这些价值。当前证据支持的是通信载荷成本，而非对能耗、货币成本或云 GPU 成本的统一优化；在这些量完成前，标题和结论均避免使用过宽的“通用成本最优”。它也不是普遍更强的单一难度排序器。

### 8.3 ROI 是可选能力而非普适贡献

COCO 中 ROI 被实际使用并带来小幅点收益；VOC 中收益小且高预算为负；Oregon 最新 OQR 完全不执行 ROI。三者共同说明 ROI 效果取决于端侧框质量、目标尺度、扩框、云输入和融合。论文将 OQR 作为完整系统配置，同时把 OQ 作为关键消融，以呈现真实边界而非只展示有利数据集。

### 8.4 离线—在线差距能否降低

理论上，该差距并非不可降低，但不可能在无分布假设下保证与能看未来的全局解相同。可行路线包括：（1）在独立 calibration 序列上估计多动作价值/字节分布并冻结价格；（2）用 online primal–dual、bandit knapsack 或 drift-aware quantile 更新适应非平稳数据；（3）允许有限但预先声明的 token burst，而不缓存未来图像；（4）学习直接面向 online regret 或 prefix-constrained reward 的控制器；（5）减少动作价值对不稳定 ROI 几何的敏感性。本文的自适应价格诊断收益很小，说明未来工作必须改进状态与训练目标，而不是只增大 CPU 并行或反复扫描 $\lambda$。

### 8.5 与 ORIC/MORIC 的公平比较

ORIC/MORIC 的理论评测以缓存弱/强检测结果计算 mAP，真实系统则逐图用冻结阈值决定是否上传；主要预算是 offload ratio。CAVS 的离线主问题更难：一个图片有多个不同字节动作，成员与动作在总容量内联合求解。在线比较时，我们将 ORIC/MORIC 接入同一 RK3588、云端、序列和前缀预算控制，但这属于统一系统适配，并非作者官方运行。正文应同时报告其原论文协议差异与我们的统一重实现，避免把 Fixed-byte 优势错误归因于算法名称。

### 8.6 可部署性与计算税

CAVS 的在线决策不等待未来帧，符合因果性；但每帧仍先运行 RKNN Edge。在云预算接近 100% 时，直接调用云端可能比 Edge→router→cloud 更低时延。正式论文应报告 CAVS 与 cloud-first/weak-skipping 的预算拐点，并将 CAVS 定位于通信稀缺且 Edge 结果本身有使用价值的区域，而非所有网络条件下的唯一最优架构。

## 9 有效性威胁与局限

**评测集复用。** COCO val2017、Oregon final 和 VOC2007 test 均在长期开发中被访问。角色隔离防止直接训练泄漏，但无法消除研究者层面的事后选择。当前结果是探索性/事后论文级证据；确认性主张需要新的注册数据或时间后移测试集。

**统计完整性。** COCO Profile×Fusion bootstrap 已完成，但最新三数据集主比较尚未全部完成。Random 具有图片采样和随机 selection 两层不确定性，必须使用 nested Random-50 bootstrap；Oregon 必须按事件聚类。投稿前不得用普通单 Random seed 的 CI 替代主表统计。

**重实现偏差。** ORIC/MORIC、DCSB 与 Confidence 在统一代码库中重训和适配；这保证共享缓存和预算，却不等于作者官方实现逐位复现。应公开 reward、特征、fold 和超参数，并在可能时补作者代码交叉验证。

**预算外部有效性。** 离线主表约束 JPEG payload，不含 TCP/IP、TLS、HTTP header 或 Base64。真实系统必须同时报告 wire bytes、云动作图像数、请求/crop 数、云 GPU 时间、返回字节和编码开销；否则“字节成本感知”只覆盖通信载荷的一部分。

**设备与网络范围。** 当前真实设备只有一类 RK3588，云端和网络路径有限。MPS 与远程 GPU 的先导 run 不能混成统一延迟表。最终需要同一云后端、二进制协议、多带宽/RTT/抖动条件、功率计能耗、温度与降频记录。

**序列与流式语义。** 顺序播放静态 COCO 图像满足零前视，但不包含真实视频的时间相关性、突发事件和概念漂移。本文当前结论适用于独立或弱相关帧；视频窗口、跟踪与场景状态属于后续扩展。

**代理目标。** Contextual reward 与预测价值和仍不是完整 AP 的精确可加分解。容量增加可能造成非单调 AP，且 $\lambda$ 调度只给出离散前沿的 supported solutions 加贪心修复。Teacher ceiling、regret、TIDE/类别/尺寸分解和 exact solver 小规模审计为【待补】。

## 10 结论

本文提出 CAVS，将端云目标检测从图像级二元上传推广为异构像素载荷下的图像—动作联合价值调度。CAVS 以端侧检测集合预测 AP50–95 对齐的动作条件价值，将真实字节成本留给容量控制，并同时支持批量严格 Fixed-byte 与零前视逐帧执行。COCO、Oregon 和跨架构 VOC 的事后完整点结果一致支持该机制；同骨干消融进一步表明，主要收益来自联合图像—动作决策而非单纯增大 Ranker。RK3588 先导部署保持了相对 ORIC 的优势，也揭示 0.024–0.030 AP 的离线—在线转换代价。

这项工作的稳健结论不是“ROI 或压缩在所有场景都有效”，而是：当候选载荷具有不同且样本相关的任务价值与字节成本时，图像级 Top-K 不足，价值预测与资源调度需要在动作层面衔接。完成统一 bootstrap、正式在线矩阵、wire/能耗测量和独立确认集后，CAVS 将形成从算法机制到真实端云系统的完整证据链。

## Statements and Declarations

### Funding

【待补：基金名称与项目编号；如无，写 “The authors received no specific funding for this work.”】

### Competing interests

The authors declare no competing interests.

### Author contributions

【待补：按 CRediT taxonomy 填写 Conceptualization、Methodology、Software、Validation、Investigation、Writing、Supervision。】

### Data availability

COCO 与 PASCAL VOC 按其公开许可获取。Oregon 数据的公开范围、申请方式和敏感位置处理为【待补：数据许可与访问声明】。

### Code availability

【待补：匿名代码仓库、冻结配置、模型哈希、评测脚本与复现实验清单。】

### Ethics approval

COCO 与 VOC 为公开数据。Oregon 野生动物图像不用于人物身份识别；涉及人员、私人地点或受保护物种坐标时按数据许可匿名化与访问控制。【待补：机构伦理审批编号或“不适用”正式声明。】

## 附录 A 理论补充

### A.1 Top-K 特例证明

设每幅图像只有 Edge 与一个同成本云动作 $g$，且 $c_i^g=c>0$。式（3）等价于选择集合 $S$：

\[
\max_S\sum_{i\in S}\hat v_i^g,
\qquad |S|c\le B.
\]

令 $K=\min(N,\lfloor B/c\rfloor)$。若可行集合包含价值较低的 $j$ 而漏掉价值更高的 $i$，交换两者不改变成本并提高目标，故最优解为前 $K$ 个正价值样本。若正价值少于 $K$，剩余样本保留 Edge。

### A.2 拉格朗日动作前沿

式（3）的松弛为

\[
L(x,\lambda)=\sum_{i,a}x_i^a(\hat v_i^a-\lambda c_i^a)+\lambda B.
\]

固定 $\lambda$ 后，每图可独立取式（13）的最大净值动作。扫描 $\lambda$ 得到离散动作集合的 supported value–cost frontier；整数非凸点可能不是任何单一 $\lambda$ 的直接解，因此容量修复提高利用率但不消除一般最优性间隙。

## 附录 B 冻结超参数与复现检查

**表 B1 CAVS 当前冻结超参数。**

| 项目 | 值 |
|---|---:|
| 最大检测框数 | 64 |
| 类别嵌入维度 | 8 |
| Box / Action / Joint hidden | 32 / 32 / 64 |
| Dropout | 0.05 |
| Context size | 32 |
| 优化器 | AdamW |
| 初始学习率 / weight decay | 1e-3 / 1e-4 |
| 最大 epoch / patience | 50 / 8 |
| Regression / rank loss 权重 | 0.5 / 0.5 |
| OOF folds / full-fit seeds | 5 / 3 |
| $\lambda$ 二分次数 | 80 |
| Batch size / gradient clip | 192 / 5 |

投稿归档必须包含：（1）数据角色清单与图像 ID 哈希；（2）Edge/Cloud/RKNN 模型 SHA-256；（3）Original/Q90/ROI 编码与推理合同；（4）reward 与 OOF 完成标记；（5）三个 full-fit checkpoint；（6）冻结 selection/control；（7）bootstrap seeds；（8）在线 frames.jsonl、run contract、服务日志和系统汇总；（9）所有表格从机器可读结果自动生成的脚本。

## 附录 C 投稿前占位替换规则

1. 每个 `【待补】` 必须由实测结果、正式声明或删除相应主张三者之一解决。
2. 不用新 bootstrap 覆盖旧点表；点估计、差值和 CI 必须来自同一冻结 selection。
3. 改变检测器、RKNN、预处理、编码、融合或预算分母时，建立新协议并重训所有依赖输出的学习器。
4. 不按 final 每档挑选最佳 OQ/OQR 或在线控制器组成虚构统一方法。
5. 摘要只保留最终主表中统计与系统验收均通过的数字。

## 参考文献

[1] Kang Y, Hauswald J, Gao C, et al. Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. *ASPLOS*. 2017:615–629. https://doi.org/10.1145/3037697.3037698

[2] Matsubara Y, Levorato M, Restuccia F. Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges. *ACM Computing Surveys*. 2023;55(5):1–30. https://doi.org/10.1145/3527155

[3] Zhou Z, Chen X, Li E, et al. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. *Proceedings of the IEEE*. 2019;107(8):1738–1762. https://doi.org/10.1109/JPROC.2019.2918951

[4] Teerapittayanon S, McDanel B, Kung HT. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *ICPR*. 2016:2464–2469. https://doi.org/10.1109/ICPR.2016.7900006

[5] Kang D, Emmons J, Abuzaid F, Bailis P, Zaharia M. NoScope: Optimizing Neural Network Queries over Video at Scale. *PVLDB*. 2017;10(11):1586–1597. https://doi.org/10.14778/3137628.3137664

[6] Li Y, Padmanabhan A, Zhao P, et al. Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. *ACM SIGCOMM*. 2020:359–376. https://doi.org/10.1145/3387514.3405874

[7] Jiang J, Ananthanarayanan G, Bodik P, Sen S, Stoica I. Chameleon: Scalable Adaptation of Video Analytics. *ACM SIGCOMM*. 2018:253–266. https://doi.org/10.1145/3230543.3230574

[8] Cao Z, Li Z, Chen Y, Pan H, Hu Y, Liu J. Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. *IEEE ICDCS*. 2023:259–270. https://doi.org/10.1109/ICDCS57875.2023.00062

[9] Cao Z, Cheng Y, Zhou Z, et al. Edge-Cloud Collaborated Object Detection via Bandwidth Adaptive Difficult-Case Discriminator. *IEEE Transactions on Mobile Computing*. 2025;24(2):1181–1196. https://doi.org/10.1109/TMC.2024.3474743

[10] Gomez C, Patkhullaev D, Hernandez AC. OffloaD: Detection Failure-Based Scheduler for Offloading Object Detection. *ACM Transactions on Embedded Computing Systems*. 2024;23(5):1–26. https://doi.org/10.1145/3677321

[11] Qiu J, Wang R, Hu B, Guérin R, Lu C. Optimizing Edge Offloading Decisions for Object Detection. *ACM/IEEE Symposium on Edge Computing*. 2024. https://doi.org/10.1109/SEC62691.2024.00021

[12] Du K, Liu Y, Hao Y, et al. OneAdapt: Fast Configuration Adaptation for Video Analytics Applications via Backpropagation. arXiv:2310.02422. 2023. https://arxiv.org/abs/2310.02422

[13] Du K, Zhang Q, Arapin A, et al. AccMPEG: Optimizing Video Encoding for Accurate Video Analytics. *Proceedings of Machine Learning and Systems*. 2022;4. https://arxiv.org/abs/2204.12534

[14] Yang Z, Wang X, Wu J, et al. EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision. *IEEE/ACM Transactions on Networking*. 2023;31(4):1765–1778. https://doi.org/10.1109/TNET.2022.3223412

[15] Chaudhary S, Taneja A, Singh A, et al. TileClipper: Lightweight Selection of Regions of Interest from Videos for Traffic Surveillance. *USENIX ATC*. 2024:967–984. https://www.usenix.org/conference/atc24/presentation/chaudhary

[16] Wang W, Mi L, Cen S, et al. Region-based Content Enhancement for Efficient Video Analytics at the Edge. *USENIX NSDI*. 2025:613–633. https://www.usenix.org/conference/nsdi25/presentation/wang-weijun

[17] Chen T, Bu Y, Zeng Y, Xie L, Lu S. RegionFilter: Region-Aware Video Filtering Mechanism on Resource-Constrained Edge Nodes. *Computer Networks*. 2024;251:110624. https://doi.org/10.1016/j.comnet.2024.110624

[18] Wang H, Li Q, Sun H, et al. VaBUS: Edge-Cloud Real-Time Video Analytics via Background Understanding and Subtraction. *IEEE Journal on Selected Areas in Communications*. 2023;41(1):90–106. https://doi.org/10.1109/JSAC.2022.3221995

[19] Dehghani D, Chen Y, Yu X, Koudas N. Selective Cloud Offloading for Accurate and Efficient Object Detection. *IEEE ICDM*. 2025:1145–1154. https://doi.org/10.1109/ICDM65498.2025.00123

[20] Mozannar H, Sontag D. Consistent Estimators for Learning to Defer to an Expert. *ICML*. 2020;PMLR 119:7076–7087. https://proceedings.mlr.press/v119/mozannar20b.html

[21] Verma R, Nalisnick E. Calibrated Learning to Defer with One-vs-All Classifiers. *ICML*. 2022;PMLR 162. https://proceedings.mlr.press/v162/verma22c.html

[22] Mao A, Mohri M, Zhong Y. Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees for Learning to Defer. *ICML*. 2025;PMLR 267:43035–43066. https://proceedings.mlr.press/v267/mao25c.html

[23] Chen L, Zaharia M, Zou J. FrugalML: How to Use ML Prediction APIs More Accurately and Cheaply. *NeurIPS*. 2020;33. https://arxiv.org/abs/2006.07512

[24] Elmachtoub AN, Grigas P. Smart “Predict, then Optimize”. *Management Science*. 2022;68(1):9–26. https://doi.org/10.1287/mnsc.2020.3922

[25] Chakrabarti A, Guérin R, Lu C, Liu J. Real-Time Edge Classification: Optimal Offloading under Token Bucket Constraints. *ACM/IEEE Symposium on Edge Computing*. 2021:41–54. https://doi.org/10.1145/3453142.3492329

[26] Geng W, Mohan N, Ott J. Budget-Adaptive Routing: Skipping the Weak When the Strong Answers Anyway. *ACM SIGCOMM NAIC Workshop*. 2026. https://doi.org/10.1145/3789240.3828740

[27] Jitkrittum W, Gupta N, Menon AK, Narasimhan H, Rawat AS, Kumar S. When Does Confidence-Based Cascade Deferral Suffice? *NeurIPS*. 2023;36:9891–9906.

[28] Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. *ICML*. 2017;PMLR 70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html

[29] Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall/CRC; 1993.

[30] Lin TY, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. *ECCV*. 2014:740–755. https://doi.org/10.1007/978-3-319-10602-1_48

[31] Everingham M, Eslami SMA, Van Gool L, Williams CKI, Winn J, Zisserman A. The PASCAL Visual Object Classes Challenge: A Retrospective. *International Journal of Computer Vision*. 2015;111:98–136. https://doi.org/10.1007/s11263-014-0733-5

[32] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*. 2019;32.

[33] Jocher G, Qiu J, Liu M, et al. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv:2606.03748. 2026. https://doi.org/10.48550/arXiv.2606.03748

[34] Orange Pi. Orange Pi 5 Ultra Hardware Specifications. Accessed 2026-08-27. https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-Ultra.html

[35] Xu R, Razavi S, Zheng R. Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques. *IEEE Communications Surveys & Tutorials*. 2023;25(4):2951–2982. https://doi.org/10.1109/COMST.2023.3323091

[36] Hu Y, Li Z, Chen Y, et al. Content-Aware Adaptive Device–Cloud Collaborative Inference for Object Detection. *IEEE Internet of Things Journal*. 2023;10(21):19087–19101. https://doi.org/10.1109/JIOT.2023.3279579

[37] Said N, Landsiedel O. EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading. *Computer Networks*. 2025;269:111437. https://doi.org/10.1016/j.comnet.2025.111437

[38] Schafhalter P, Krentsel A, Wei H, et al. TURBO: Utility-Aware Bandwidth Allocation for Cloud-Augmented Autonomous Control. *OASIcs New Ideas in Networked Systems*. 2026;139:18:1–18:34. https://doi.org/10.4230/OASIcs.NINeS.2026.18

[39] Hawlader F, Robinet F, Frank R. Leveraging the Edge and Cloud for V2X-Based Real-Time Object Detection in Autonomous Driving. *Computer Communications*. 2024;213:372–381. https://doi.org/10.1016/j.comcom.2023.11.025
