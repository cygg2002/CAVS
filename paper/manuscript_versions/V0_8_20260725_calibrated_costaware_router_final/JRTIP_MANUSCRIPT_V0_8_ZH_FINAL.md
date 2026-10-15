# 面向端云协同目标检测的校准式成本感知路由与自适应载荷选择

Calibrated Cost-Aware Routing and Adaptive Payload Selection for Edge-Cloud Object Detection

[作者确认：作者姓名]

[作者确认：作者单位；通讯作者姓名与电子邮箱]

## 摘要

资源受限视觉终端通常只能运行轻量目标检测器，而将全部图像上传云端会持续占用上行带宽，并引入编码、传输、排队和云端推理开销。现有端云检测方法多集中于识别难例，或者单独优化压缩与区域传输，较少在统一协议下联合决定上传哪些样本、上传何种载荷以及如何施加预算。本文提出一种校准式成本感知端云目标检测框架，将在线决策解耦为样本排序、Original/Q90/ROI 载荷选择和 Fixed-call/Online/Fixed-byte 预算调度，并在离线阶段仅使用 fit、selector 与 gate 数据选择和冻结配置。对于 Fixed-call 和 Online，控制器在 Official AP 容忍约束下最小化实际请求字节；对于 Fixed-byte，控制器在容量约束内最大化 Official AP。临时候选在独立 gate 上接受 200 次配对 bootstrap 经验风险门禁，失败时直接回退 SharedAction 参考配置，final 不参与选择、调参或修复。

本文在 COCO、ENA24 和 Oregon 三个数据域、8 个可部署 Ranker、3 个动作策略和 10 个预算工作点上评测该流程，共形成 30 个冻结结果。门禁通过 13/30 个工作点，但其中 6 个在 gate 上与参考动作完全相同，真正比较不同动作并通过的只有 7 个；其余 17 个回退。COCO 的 10 个候选全部回退；ENA24 在低预算 Fixed-call/Online 上获得 0.74-1.73 KB/image 的小幅节省，final AP 变化为 -0.000387 至 +0.000097；Oregon 的非平凡通过点满足 0.001 容忍量，但 Fixed-byte 不产生额外字节节省。三个数据集相对同一候选集合 final oracle 的平均 AP regret 分别为 0.003815、0.008836 和 0.003981。结果表明，不存在跨数据域和预算语义统一最优的 Ranker 或 ROI 策略；成本感知控制器能够阻止部分校准后验乐观配置进入 final，但当前门禁也存在明显保守性。本文因此将贡献定位为可审计的解耦决策架构与校准流程，而不是某个跨域通用学习器或形式化安全保证。

**关键词：** 端云协同；目标检测；选择性卸载；成本感知路由；载荷选择；风险门禁

## 1 引言

低功耗智能摄像头被广泛用于野外监测、安防巡检和无人值守感知。轻量检测器能够在本地提供基本输出，但其模型容量、输入分辨率和硬件算子受限，对小目标、遮挡和域外场景的识别能力通常弱于云端高容量检测器。全量上云可以利用更强模型，却要求持续传输图像，并使响应时间受到网络往返、服务排队和云端计算的共同影响。因此，端云协同检测的核心不是简单选择端侧或云端，而是在有限资源下决定哪些图像值得上传以及上传何种表示。

现有研究已经证明选择性卸载具有价值。Adaptive Feeding 使用 SVM 在快速与高精度检测器之间选择 [10]；DCSB 及其带宽自适应版本通过难例判别器决定是否上云 [11,12]；OffloaD 根据检测失败风险调度任务 [20]；ORIC/MORIC 进一步指出，目标检测 mAP 是跨图像、跨类别聚合的指标，上传收益不应简化为单图分类正确性 [13]。这些方法主要关注“上传哪些图像”，且多数使用整图载荷。

通信成本还取决于“上传什么”。Original 保留完整图像信息但字节量最高；Q90 JPEG 可以显著减少请求量，却可能损害细粒度纹理；ROI 进一步压缩空间范围，但依赖端侧候选框，并可能因漏检和上下文缺失而降低云端检测质量。RegionFilter、EdgeDuet 和 AccMPEG 分别从区域过滤、小目标 tile 和任务感知编码角度说明，空间内容与编码质量会改变精度和传输成本 [14-16]。因此，只优化上传图像数量或固定采用 ROI，都不能充分描述端云系统的成本收益。

样本排序、载荷选择和预算调度之间还存在耦合。Fixed-call 固定上传样本数，适合比较排序能力；Online 通过冻结阈值逐样本接纳任务，但实际上传率会随数据分布漂移；Fixed-byte 限制总请求字节，此时更小的载荷还可能使预算容纳更多样本。同一策略在 Fixed-call 中可能因裁剪损失而不利，却在 Fixed-byte 中因释放容量而有效。若不区分预算语义，容易把上传率漂移或载荷压缩造成的收益错误归因于样本学习器。

本文据此提出校准式成本感知路由与自适应载荷选择框架。系统在线只执行一个冻结的 Ranker、动作策略和预算调度器；离线控制器则在 selector 数据上求解配置级精度成本问题，并在 gate 数据上检查候选相对参考配置的 AP 与请求字节差异。该设计不把某个排序学习器视为跨域通用解，而把端侧置信度、类别收益先验和检测收益学习器作为可替换信号。论文的创新重点是可审计的配置选择、风险门禁与预算语义解耦。

本文的主要贡献如下。

1. 提出一种解耦样本路由、载荷动作和预算调度的端云目标检测架构，在不重新训练端侧和云端检测器的条件下统一支持 Original、Q90 与 ROI。

2. 提出一个校准式成本感知控制器。该控制器针对调用量和字节预算采用不同优化目标，在 AP 容忍约束下选择低成本配置，并通过独立配对 bootstrap 门禁冻结候选或回退参考策略。

3. 在三个不同数据域上使用统一检测缓存、Official AP 和实际载荷字节，系统比较 8 个可部署 Ranker、3 个载荷策略以及 Fixed-call、Online、Fixed-byte 三种预算语义，并完整保留门禁失败和回退结果。

4. 通过跨域冻结结果揭示单一 Ranker 与 ROI 策略的适用边界，并量化校准选择相对 final oracle 的 regret 和保守门禁的机会成本。

本文其余部分组织如下。第 2 节回顾选择性卸载、载荷优化和成本敏感路由；第 3 节定义问题并分析 Confidence 排序的充分条件；第 4 节介绍系统架构与控制器；第 5 节说明实验协议；第 6 节报告冻结结果；第 7 节讨论系统验证与局限；第 8 节总结全文。

## 2 相关工作

### 2.1 选择性卸载与检测级路由

Neurosurgeon 依据设备、网络和服务器代价选择网络切分位置 [1]，split computing 与早退研究进一步分析中间特征传输和多出口推理的成本 [2-4]。视频分析系统还通过模型选择、边缘调度和帧过滤减少不必要的推理或传输 [5-9]。本文不切分单个检测器，也不依赖视频冗余，而是让端侧检测器先产生可直接使用的结果，再选择部分图像交给独立云端检测器复核。

Adaptive Feeding 使用 SVM 判断图像应由快速还是高精度检测器处理 [10]。DCSB 使用检测语义特征识别难例，其后续版本增加带宽自适应和区域采样 [11,12]。ORIC/MORIC 通过 contextual reward 估计一张图对整体检测 mAP 的边际贡献 [13]。OffloaD 以检测失败为卸载信号 [20]。EdgeBoost 则在分类任务中校准端侧置信度或 margin 后上传低置信样本 [21]。这些工作说明 Confidence 是必要强基线，也说明端侧不确定性与云端相对收益并不总是等价。

### 2.2 ROI、压缩与任务感知传输

RegionFilter 使用区域信息减少资源受限节点的视频传输 [14]；EdgeDuet 将低分辨率端侧检测与高分辨率 tile 处理结合，面向小目标场景降低带宽 [15]；AccMPEG 根据图像区域对下游分析精度的影响分配编码质量 [16]。特征压缩和语义编码则通过传输中间表示降低通信 [22,23]。与这些方法不同，本文不改变检测器或训练新的编码器，而是在统一缓存上比较 Original、Q90 和 ROI，并把实际请求字节直接纳入配置选择。

### 2.3 成本敏感路由与风险校准

Learning to defer 将模型与下游专家之间的分配建模为条件风险最小化问题 [17]，后续工作讨论了校准式 defer 和多专家路由 [18,19]。现代神经网络置信度可能与真实风险不一致 [25]，因此仅使用一个未校准阈值并不能保证部署成本或精度。Learn then Test 等研究提供了在独立数据上筛选候选并检查风险的通用思路 [26]。

本文研究具有非加性 Official AP 的目标检测，同时决定样本与像素载荷，并区分调用量与实际字节预算。最终配置必须在 final 前冻结。本文不宣称 200 次 pointwise bootstrap 门禁提供形式化或同时安全保证，而将其视为防止 selector 点估计过拟合的经验风险控制步骤。

## 3 问题定义与成本感知原理

### 3.1 端云检测与动作空间

给定图像集合 \(\mathcal D=\{(I_i,Y_i)\}_{i=1}^{N}\)，端侧检测器 \(D_e\) 输出检测集合 \(E_i=D_e(I_i)\)。未上传时，系统直接输出 \(E_i\)。若图像被上传，载荷动作属于

\[
\mathcal A=\{\mathrm{Original},\mathrm{Q90},\mathrm{ROI}\}.
\]

云端检测器 \(D_c\) 对载荷执行复核，并经尺度恢复、坐标回映射和框融合得到 \(F_i(a)\)。令 \(c_i(a)\) 表示动作 \(a\) 的实际请求字节，边缘保留动作记为 \(a=\mathrm{Edge}\)，其请求成本为零。

### 3.2 样本价值与载荷成本

在可加代理效用下，动作 \(a\) 对图像 \(i\) 的净价值可写为

\[
V_i(a;\lambda)=\mathbb E[\Delta U_i(a)\mid z_i]-\lambda c_i(a),
\]

其中 \(\Delta U_i(a)\) 表示云端动作相对端侧输出的检测效用增量，\(z_i\) 是在线可观测的端侧检测、图像统计和载荷属性，\(\lambda\) 是预算对应的成本乘子。理想动作满足

\[
a_i^*=\arg\max_{a\in\{\mathrm{Edge}\}\cup\mathcal A}V_i(a;\lambda).
\]

该表达式说明，端侧不确定性、云端修复收益和单位字节收益是三个不同量。已有级联决策理论同样指出，置信度延期只在特定风险结构下充分 [24]。为明确本文采用成本感知决策的原因，给出如下命题。

**命题 1（Confidence 排序的充分条件）。** 设 \(h_i\) 为端侧不确定性，且样本效用可加。若最优云动作相对端侧的期望收益可写为关于 \(h_i\) 的单调不减函数 \(g(h_i)\)，所有样本的最优载荷成本均为同一常数 \(c\)，则在 Fixed-call 预算下按 \(h_i\) 从高到低选择前 \(K\) 个样本是最优的。反之，只要存在两个样本 \(i,j\) 满足 \(h_i>h_j\)，但 \(g_i-\lambda c_i<g_j-\lambda c_j\)，就存在一个使该样本对跨越选择边界的预算 \(K\)，此时 Confidence 次序不是最优的。

证明可由交换论证得到：在前述充分条件下，任意包含较低 \(h_j\) 而未包含较高 \(h_i\) 的可行集合，都可用 \(i\) 替换 \(j\) 且不降低总效用；反例条件则直接使该替换降低净效用。实际检测中，类别、目标尺寸、拥挤程度、端云错误互补性以及 Original/Q90/ROI 字节均会改变 \(g_i\) 或 \(c_i\)，因此不能预设 Confidence 为充分统计量。

Official AP 不是逐图可加指标，因而本文不把单图代理收益直接称为 AP 增量。主控制器使用统一评估器在 selector 集上计算配置级 Official AP，从而避免把未验证的逐图代理写成最终优化目标。

### 3.3 配置与预算模式

样本 Ranker \(r\) 根据 \(z_i\) 产生分数 \(s_i^{(r)}\)，载荷策略 \(\pi\) 为已接纳样本选择动作。完整配置定义为

\[
\theta=(r,\pi,m,B,\tau,\lambda),
\]

其中 \(m\) 是预算模式，\(B\) 是预算，\(\tau\) 是动作门限，\(\lambda\) 是 Fixed-byte 调度乘子。Fixed-call 严格接纳得分最高的 \(K=\lfloor BN\rfloor\) 张图像：

\[
\mathcal U_{call}=\operatorname{TopK}(\{s_i^{(r)}\}_{i=1}^{N},K).
\]

Online 使用 selector 数据上冻结的阈值 \(t_{r,B}\)：

\[
\mathcal U_{online}=\{i:s_i^{(r)}\ge t_{r,B}\}.
\]

Fixed-byte 约束所有接纳载荷的总请求字节：

\[
\sum_{i\in\mathcal U}c_i(a_i)\le C_B.
\]

三种模式分别衡量严格调用预算、阈值部署漂移和总字节容量，不在同一上传率假设下混合比较。

## 4 校准式成本感知端云协同方法

### 4.1 总体架构

系统包含在线执行和离线校准两个阶段。在线阶段依次执行端侧检测、样本评分、预算接纳、载荷选择、云端复核与结果融合。离线阶段利用冻结检测缓存和独立数据角色完成候选校准、风险门禁和配置冻结。未接纳图像立即使用端侧结果；接纳图像按冻结动作发送云端。离线控制器不修改端侧或云端检测器权重，各候选共享相同预测、载荷和评估器缓存。

**[图 1：校准式成本感知端云协同架构。在线阶段只执行一个冻结配置；离线阶段完成候选选择、风险门禁和一次性 final 评测。]**

### 4.2 样本卸载信号

框架不绑定单一 Ranker。公共特征由 17D 端侧检测与图像统计量，以及仅在 fit 集拟合的 32D FPN-PCA 表示组成。Confidence 使用空检优先和最低部署框置信度的补值；ClassGainPrior 估计类别条件云端收益；M2E 及其 residual 变体学习端云收益代理；SemanticRank 使用公共语义特征进行成对排序；MORIC-style 和 GEC-style 分别提供 contextual-reward 相关与组合基线；Current-Controlled-32D 用于检验受控语义维度结构。

自动配置只允许选择 8 个确定性 Ranker：Confidence、ClassGainPrior、M2E、Confidence-Residual M2E、SemanticRank、MORIC-style、GEC-style 和 Current-Controlled-32D。Random 使用 50 个种子评估随机波动，不参与部署选择。Legacy-Frozen-32D 依赖历史冻结模型，只作为兼容性实验，不进入本次严格三数据集控制器。

### 4.3 自适应载荷策略

三种载荷策略共享同一 Original/Q90/ROI 缓存。

**SharedAction。** 使用 fit-only 的共享动作头选择 Original、Q90 或 ROI，作为每个工作点的参考策略。动作头不读取 Ranker 标识或排序分位，确保样本排序与载荷选择可独立比较。

**CostEligible-AllROI。** 当 ROI 可用且请求字节小于 SharedAction 基础动作时切换为 ROI，否则保留基础动作。该方法是简单成本门禁，用于检查复杂动作模型是否具有必要性。

**GBDT-Hierarchical。** 先得到 SharedAction 基础动作，再使用 fit-only GBDT 估计 ROI 是否在降低字节的同时保持可接受质量。训练正例要求 ROI 的逐图质量代理不低于基础动作减 0.02，且 ROI 请求字节更少。输入包括 49D 公共特征、Q90/Original 字节比、ROI/Original 字节比、ROI 面积比及载荷可用性，共 54D。模型采用平衡样本权重、150 次提升迭代、0.05 学习率、最大深度 3、最小叶节点样本数 20 和 \(L_2=1.0\)，随机种子固定为 20260722。阈值按数据集和预算模式在 selector 上校准，并在全部非随机 Ranker 间共享。

若 GBDT-Hierarchical 与 CostEligible-AllROI 产生相同动作，二者按等价结果报告，不以模型复杂度作为贡献。

### 4.4 配置级成本优化

令 \(Q(\theta;\mathcal S)\) 表示配置 \(\theta\) 在集合 \(\mathcal S\) 上的 Official mAP50-95，令

\[
C(\theta;\mathcal S)=\frac{1}{|\mathcal S|}\sum_{i\in\mathcal S}\frac{c_i(a_i)}{1024}
\]

表示按全部输入图像均摊的请求 KB/image。每个数据集、预算模式和工作点先在 SharedAction 配置中选择 selector AP 最高者作为参考 \(\theta_{ref}\)。精度容忍量固定为 \(\epsilon=0.001\)。

对于 Fixed-call 和 Online，控制器求解

\[
\min_{\theta\in\Theta}C(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.
\]

对于 Fixed-byte，总请求容量已由预算限制，控制器求解

\[
\max_{\theta\in\Theta}Q(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad C(\theta;\mathcal D^{sel})\le C_B,\;Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.
\]

Fixed-call 和 Online 在最低成本并列时依次选择 AP 更高、策略更简单和预注册 Ranker 顺序更靠前的配置；Fixed-byte 在最高 AP 并列时依次选择请求字节更低、策略更简单和预注册 Ranker 顺序更靠前的配置。策略简单性顺序冻结为 SharedAction、GBDT-Hierarchical、CostEligible-AllROI。

### 4.5 独立风险门禁与冻结回退

临时候选 \(\theta_{prov}\) 与参考配置在 gate 集上进行 200 次配对图像级 bootstrap [27]。第 \(b\) 次重复同时对两种配置使用相同图像索引，计算

\[
\Delta Q^{(b)}=Q(\theta_{prov};\mathcal D_b^{gate})-Q(\theta_{ref};\mathcal D_b^{gate}),
\]

\[
\Delta C^{(b)}=C(\theta_{prov};\mathcal D_b^{gate})-C(\theta_{ref};\mathcal D_b^{gate}).
\]

本文取 200 个重采样差值的 2.5% 和 97.5% 分位数作为逐工作点经验区间。Fixed-call 和 Online 要求 AP 差区间下界不低于 \(-\epsilon\)，且请求 KB 差区间上界不高于 0；Fixed-byte 只执行 AP 门禁。冻结配置为

\[
\theta_{frozen}=\begin{cases}\theta_{prov},&\text{门禁通过},\\\theta_{ref},&\text{门禁失败}.\end{cases}
\]

门禁失败后不尝试第二候选。配置、阈值和来源哈希写入冻结文件后，final 只评估一次；final oracle 仅用于计算诊断 regret，不改变冻结配置。该门禁是 pointwise 经验筛选，不构成 30 个工作点上的同时统计保证。

**算法 1：成本感知配置校准与风险门禁。** 输入 selector 集、gate 集、候选集合、预算模式和容忍量；首先在 SharedAction 候选中确定参考，删除 selector AP 低于参考减容忍量的候选；Fixed-call/Online 最小化请求 KB，Fixed-byte 最大化 Official AP；随后在 gate 上执行配对 bootstrap，候选通过则冻结，否则直接冻结参考；写入选择、区间、回退状态和来源哈希后，final 只评估冻结配置，oracle 仅作诊断。

### 4.6 在线开销

配置搜索只在离线执行。在线系统只运行一个冻结 Ranker、一个载荷策略和一次预算判断。Confidence 与线性 Ranker 的开销相对端侧检测较小；Current-Controlled-32D 按冻结协议计入 0.08 ms/image 和 32 个排序权重。GBDT 动作头开销单独记录，不混入尚未实测的端到端时延。

## 5 实验设置

### 5.1 数据集与数据角色

本文使用 COCO [28]、ENA24 和 Oregon Critters。COCO 提供 80 类通用场景对照。ENA24 是项目使用的 22 类目标存在型野生动物检测集，当前划分不含空图，平均约 1.1 个标注框/图，因此不能据此评价空帧过滤。Oregon Critters 是项目使用的 45 类野生动物相机图像集，原图分辨率和字节量较高，用于检验类别长尾与背景域变化下的成本收益。ENA24 与 Oregon 的公开来源和许可证尚待作者确认，本文不虚构数据集引文。

每个数据集按用途划分为拟合集 fit、选择集 selector、门禁集 gate 和最终评测集 final。fit 用于训练 Ranker 与动作模型，selector 用于配置选择，gate 只用于门禁，final 不参与拟合、选择、阈值校准或回退。

| 数据集 | fit | selector | gate/audit | final | 评测边界 |
|---|---:|---:|---:|---:|---|
| COCO | 9,678 | 571 | 630 | 5,000 | val2017 固定 diagnostic |
| ENA24 | 1,400 | 200 | 200 | 1,758 | selector/gate 近重复组隔离 |
| Oregon | 9,923 | 4,961 | 9,923 | 4,962 | gate 独立；final 为 post-hoc exploratory reuse |

ENA24 的 400 张 calibration 图像先计算 16x16 灰度 dHash，将汉明距离不超过 6 的样本连接为近重复分量，再以固定种子将完整分量分配至 selector 和 gate。共形成 385 个连通组，组间零交叉。由于缺少可靠事件、相机和位置元数据，该划分只能称为 near-duplicate-group-disjoint。Oregon 采用冻结的 random-image 主协议；相机背景可能跨 detector、selector 和 final 角色出现，这一边界在第 7 节讨论。

### 5.2 检测器与载荷

统一矩阵使用冻结 YOLO26n 级端侧检测器与 YOLO26x 级云端检测器。YOLO26 为 2026 年预印本，本文不把检测器结构作为创新 [29]。三个数据集分别沿用其 canonical 权重、预测缓存、类别映射和数据划分，不跨数据集共享归一化参数或动作阈值。

Original、640-Q90 和 ROI 载荷均预先生成并记录实际字节。Q90 将长边缩放至 640 像素，使用 JPEG quality 90 和 4:2:0 色度抽样。ROI 由端侧置信度不低于 0.25 的候选框生成，按候选面积动态扩框，裁剪总面积上限为原图的 65%，并以 Q90 编码；每张图最多触发一次云调用，云端框经坐标回映射后以 IoU 0.5 的 MAUR 规则与端侧结果融合。ROI 不可用时按冻结策略回退至 Q90 或 Original。所有候选共享相同检测缓存、载荷缓存和 Official AP 评估器，不根据 final 修改 ROI 或 Q90 动作。

### 5.3 对比方法与工作点

系统参考包括 Edge-only、Cloud-only Original、Always Q90、Always ROI fallback 和 Shared ContextM2E 全上传。样本排序基线包括 Random 以及前述 8 个确定性 Ranker。载荷策略为 SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical。每个工作点有 24 个可部署候选，Random 不进入自动选择。

注册工作点为 Fixed-call 10%/25%/40%，Online nominal 10%/25%/40%，以及 Fixed-byte Original 字节预算 5%/10%/20%/40%。每个数据集共 10 个工作点、240 个 selector 候选，三个数据集共 720 个候选。主方法是 selector 选择并经 gate 门禁后的冻结配置；主要对照为最佳 SharedAction reference、未经过 gate 的 provisional 候选和 diagnostic final oracle。原计划的“三数据集共用一个配置”消融截至证据冻结日尚未执行，本文不将其列为完成实验。

### 5.4 评测指标

检测指标以 conf=0.001 的 COCO-style Official mAP50-95 为主，并保留 mAP50 和 mAP75。成本指标包括实际请求 KB/image、响应 KB/image、上传率、Original/Q90/ROI 比例、均摊 GFLOPs、Ranker 参数量和 Ranker 时延。Online 同时报告 nominal 与实际上传率；Fixed-byte 同时报告实际字节和接纳样本比例。

E2E 字段由端侧推理、编码、请求字节、给定带宽和云端推理的冻结模型估算得到。本文表格只写作 Estimated E2E@10Mbps，不将其称为真实链路测量，也不据此声称实时性能。

### 5.5 系统参考

| 数据集 | 方法 | Official mAP50-95 | 请求 KB/image |
|---|---|---:|---:|
| COCO | Edge-only | 0.407904 | 0.00 |
| COCO | Cloud-only Original | 0.574243 | 159.12 |
| COCO | Always Q90 | 0.569870 | 84.08 |
| COCO | Always ROI fallback | 0.559398 | 71.16 |
| ENA24 | Edge-only | 0.712660 | 0.00 |
| ENA24 | Cloud-only Original | 0.741774 | 433.29 |
| ENA24 | Always Q90 | 0.736860 | 94.34 |
| ENA24 | Always ROI fallback | 0.737294 | 89.91 |
| Oregon | Edge-only | 0.454815 | 0.00 |
| Oregon | Cloud-only Original | 0.499095 | 1595.49 |
| Oregon | Always Q90 | 0.471120 | 97.98 |
| Oregon | Always ROI fallback | 0.459940 | 96.03 |

三数据集的端云精度差和图像字节分布显著不同。Oregon 的 Original 请求成本远高于另两个数据集，而 Q90 损失也更明显，因此成本必须基于实际载荷计算，不能仅比较上传比例。COCO 的绝对 AP 使用 20260723 冻结 final 预测缓存口径，不与更早的 COCO val diagnostic 表合并。

## 6 结果与分析

### 6.1 门禁与冻结结果总览

30 个工作点中，门禁通过 13 个、回退 17 个。通过点中 6 个在 gate 上与参考动作逐图完全相同，只有 7 个真正比较了不同动作。冻结后共有 22 个 SharedAction 和 8 个 GBDT-Hierarchical；CostEligible-AllROI 未被最终冻结。表 2 将“相同动作通过”与“非同动作通过”分开，避免把平凡零差值误写为成本优化成功。

| 数据集 | 工作点 | 非同动作通过 | 相同动作通过 | 回退 | 冻结 GBDT/Shared | 平均 AP regret | 最大 AP regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | 10 | 0 | 0 | 10 | 0 / 10 | 0.003815 | 0.011312 |
| ENA24 | 10 | 3 | 5 | 2 | 4 / 6 | 0.008836 | 0.013187 |
| Oregon | 10 | 4 | 1 | 5 | 4 / 6 | 0.003981 | 0.017063 |
| 合计 | 30 | 7 | 6 | 17 | 8 / 22 | — | — |

相对各工作点 SharedAction reference，COCO 因全部回退而 AP 和请求 KB 差均为 0。ENA24 的 10 点平均 AP 差为 -0.000057，最差为 -0.000387；仅 4 个非平凡 final 动作变化合计节省 4.95 KB/image，均摊到 10 点为 0.50 KB/image。Oregon 的 10 点平均 AP 差为 -0.000239，最差为 -0.000833；唯一明确的 Fixed-call 成本变化为节省 0.29 KB/image，Fixed-byte 通过点受相同容量上限约束，不代表额外通信节省。

**[图 2：三数据集冻结配置的 Official AP 与请求 KB/image 工作点。各面板独立缩放；颜色区分 Fixed-call、Online 和 Fixed-byte。]**

**[图 3：三数据集风险门禁结果。通过点进一步区分非同动作比较与 gate 上动作完全相同的平凡通过。]**

### 6.2 COCO 冻结结果

COCO 的 10 个 provisional 候选全部未通过 630 张 audit 门禁，因此冻结结果全部使用各工作点最佳 SharedAction reference。Fixed-call/Online 的 GBDT 候选在 selector 上只损失约 0.00007-0.00025 AP 并节省 1.22-2.09 KB/image，但 audit AP 差区间下界达到 -0.00327 至 -0.00418，低于 -0.001 门槛。

| 模式 | 预算 | 冻结 Ranker + 策略 | Official AP | KB/image | Estimated E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | SemanticRank + SharedAction | 0.443825 | 16.29 | 38.38 ms | 0.00% | 回退 |
| Fixed-call | 25% | SemanticRank + SharedAction | 0.483577 | 39.95 | 67.92 ms | 0.00% | 回退 |
| Fixed-call | 40% | Controlled-32D + SharedAction | 0.514654 | 60.93 | 95.34 ms | 0.08% | 回退 |
| Online | 10% | SemanticRank + SharedAction | 0.456962 | 23.96 | 47.80 ms | 0.00% | 回退 |
| Online | 25% | SemanticRank + SharedAction | 0.501559 | 54.08 | 85.77 ms | 0.04% | 回退 |
| Online | 40% | Controlled-32D + SharedAction | 0.521382 | 67.41 | 103.57 ms | 0.08% | 回退 |
| Fixed-byte | 5% | SemanticRank + SharedAction | 0.429258 | 7.95 | 34.33 ms | 1.70% | 回退 |
| Fixed-byte | 10% | Confidence + SharedAction | 0.441859 | 15.90 | 48.36 ms | 2.68% | 回退 |
| Fixed-byte | 20% | MORIC-style + SharedAction | 0.467554 | 31.82 | 73.28 ms | 3.66% | 回退 |
| Fixed-byte | 40% | Controlled-32D + SharedAction | 0.520311 | 63.65 | 112.97 ms | 3.72% | 回退 |

门禁也产生了保守的 false negative。COCO Fixed-byte 被拒绝的 GBDT 候选在 final 上相对 reference 实际提高 0.00475-0.00994 AP，但协议禁止根据 final 恢复候选。COCO 的平均 final-oracle regret 为 0.003815，最大为 0.011312。这说明门禁确实阻止了 audit 上证据不足的配置，同时也牺牲了部分 final 上的潜在收益。

### 6.3 ENA24 冻结结果

ENA24 使用 200 张 selector 和 200 张近重复组隔离 gate。8 个工作点通过，其中 5 个在 gate 上动作相同。GBDT-Hierarchical 在 10%/25% Fixed-call 与 Online 被保留，40% 工作点回退；4 个 Fixed-byte 工作点的 provisional 本身就是 SharedAction。

| 模式 | 预算 | 冻结 Ranker + 策略 | Official AP | KB/image | Estimated E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | Confidence + GBDT | 0.710024 | 40.22 | 57.65 ms | 0.17% | 通过 |
| Fixed-call | 25% | ClassGainPrior + GBDT | 0.726491 | 91.75 | 110.11 ms | 0.51% | 通过 |
| Fixed-call | 40% | SemanticRank + SharedAction | 0.727665 | 163.79 | 178.90 ms | 0.00% | 回退 |
| Online | 10% | Confidence + GBDT | 0.713217 | 60.76 | 77.89 ms | 0.23% | 通过 |
| Online | 25% | ClassGainPrior + GBDT | 0.726475 | 96.57 | 114.86 ms | 0.57% | 通过 |
| Online | 40% | SemanticRank + SharedAction | 0.727164 | 161.09 | 176.10 ms | 0.00% | 回退 |
| Fixed-byte | 5% | Confidence + SharedAction | 0.716519 | 21.66 | 41.33 ms | 0.00% | 相同动作 |
| Fixed-byte | 10% | ClassGainPrior + SharedAction | 0.727101 | 43.31 | 75.65 ms | 0.00% | 相同动作 |
| Fixed-byte | 20% | M2E + SharedAction | 0.735639 | 86.65 | 126.95 ms | 0.00% | 相同动作 |
| Fixed-byte | 40% | Confidence + SharedAction | 0.738626 | 173.31 | 210.28 ms | 0.00% | 相同动作 |

4 个 GBDT final 工作点相对 reference 的 AP/KB 变化依次为：Fixed-call 10% 的 +0.000097/-0.74 KB，Fixed-call 25% 的 -0.000373/-1.54 KB，Online 10% 的 +0.000090/-0.94 KB，Online 25% 的 -0.000387/-1.73 KB。精度变化均在预设 0.001 容忍量内，但 ROI 比例仅 0.17%-0.57%，通信节省也较小。

Fixed-call 10% 在 gate 上与 reference 动作完全相同，因精确零差值通过，但在 final 上出现少量 ROI。因此该点只能表述为“gate 未观察到动作差异且 final 未明显退化”，不能表述为 gate 已验证 ROI 安全。ENA24 平均 final-oracle regret 为 0.008836，三个数据集中最高，主要反映 200/200 小样本 selector-gate 划分的配置排序和统计功效不足。

### 6.4 Oregon 冻结结果

Oregon 有 5 个门禁通过点，其中 1 个为相同动作。唯一明确的 Fixed-call 成本优化出现在 10%；25%/40% Fixed-call 和全部 Online 候选因 audit AP 区间不足而回退。

| 模式 | 预算 | 冻结 Ranker + 策略 | Official AP | KB/image | Estimated E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | GEC-style + GBDT | 0.465108 | 95.63 | 102.05 ms | 0.04% | 通过 |
| Fixed-call | 25% | Confidence + SharedAction | 0.468745 | 163.11 | 167.31 ms | 0.00% | 回退 |
| Fixed-call | 40% | Confidence + SharedAction | 0.484184 | 396.85 | 365.46 ms | 0.00% | 回退 |
| Online | 10% | ClassGainPrior + SharedAction | 0.451582 | 249.02 | 229.00 ms | 0.00% | 回退 |
| Online | 25% | Confidence + SharedAction | 0.469655 | 181.17 | 182.63 ms | 0.00% | 回退 |
| Online | 40% | Confidence + SharedAction | 0.485449 | 437.01 | 399.25 ms | 0.00% | 回退 |
| Fixed-byte | 5% | Confidence + GBDT | 0.472367 | 79.77 | 114.19 ms | 0.06% | 通过 |
| Fixed-byte | 10% | Confidence + GBDT | 0.475635 | 159.55 | 183.53 ms | 0.06% | 通过 |
| Fixed-byte | 20% | Confidence + GBDT | 0.484405 | 319.10 | 316.31 ms | 0.06% | 通过 |
| Fixed-byte | 40% | Confidence + SharedAction | 0.492746 | 638.20 | 579.60 ms | 0.00% | 相同动作 |

Fixed-call 10% 相对 reference 的 final AP 为 -0.000097、请求量为 -0.29 KB/image，收益量级很小。Fixed-byte 5%/10%/20% 相对 reference 的 final AP 分别下降 0.000681、0.000833 和 0.000780，仍处于 0.001 容忍内；由于其请求量由容量上限约束，这三点不能解释为额外字节节省。Oregon 平均 final-oracle regret 为 0.003981，最大值 0.017063 出现在 Online 10%，说明 selector 选择的 ClassGainPrior 未迁移为 final 最优 GEC-style。

### 6.5 Ranker 的跨域差异

冻结配置直接否定了“单一 Ranker 跨域统一最优”的强主张。COCO 的 10 点中 SemanticRank 占 5 点、Controlled-32D 占 3 点；ENA24 主要选择 Confidence 4 点、ClassGainPrior 3 点；Oregon 则有 8 点冻结为 Confidence，另有 GEC-style 和 ClassGainPrior。该差异不是动作策略单独造成的，因为 reference 本身在 SharedAction 内按 selector AP 选择 Ranker。

因此，SemanticRank、M2E、ClassGainPrior 和 Confidence 在本文中应被视为可插拔候选，而不是论文的唯一算法贡献。当前证据支持“根据数据域和预算工作点进行校准选择”，不支持“某个高维语义学习器普遍优于简单置信度”。同时，final oracle regret 说明当前 selector 规模和门禁仍未把这一思想完全转化为低 regret 控制器。

### 6.6 载荷动作与预算语义

**[图 4：冻结配置的 Edge、Original、Q90 和 ROI 动作组成。比例按全部输入图像计算，并标注实际上传率与 ROI 比例。]**

图 4 显示，冻结结果中的 ROI 不是主导载荷。ENA24 和 Oregon 的 ROI 比例均低于 0.6%；COCO 的 Fixed-call/Online 低于 0.1%，仅 Fixed-byte SharedAction 达到 1.70%-3.72%。因此不能声称 GBDT-Hierarchical 或 Adaptive ROI 在三个数据集上普遍降低成本。当前更稳定的动作仍是 Original/Q90 混合，ROI 只在少数样本上触发。

预算语义改变了结果解释。Fixed-call 中更小载荷会直接体现为 KB/image 下降；Online 还必须同时观察实际上传率漂移；Fixed-byte 中字节容量近似固定，动作压缩的作用是改变可接纳样本和 AP，而不是在同一容量下继续声称节省字节。Oregon Fixed-byte 的三个通过点正说明“AP 容忍内保留”不等于“通信成本进一步下降”。

### 6.7 门禁、诊断消融与机会成本

风险门禁是本文最重要的受控组成分析。若去掉 gate，selector 会直接部署 provisional 候选；COCO 的 6 个 Fixed-call/Online 候选虽在 selector 上满足 AP 容忍和成本目标，却在 audit 上出现 -0.00327 至 -0.00418 的 AP 区间下界。门禁使这些配置回退，避免把 selector 后验乐观当作 final 结论。

门禁同时带来 false-negative 成本。COCO Fixed-byte 与 ENA24 40% 的部分被拒候选在 final 上反而表现更好，但预注册协议仍保留回退。13 个通过点中只有 7 个进行了非同动作比较，说明仅报告“43.3% 通过率”会夸大有效优化。final oracle 平均 regret 为 0.003815-0.008836，最大可达 0.017063，也不支持“控制器接近 oracle”的表述。

本轮完成的诊断只能回答 reference、provisional、gate 回退和 final oracle 的作用。尚未执行的全数据集统一配置、不同 bootstrap 次数、聚类 bootstrap 和新的隐藏 final 属于后续实验，不作为本文已完成消融。当前结果更适合把控制器描述为可审计且保守的第一版方法，而不是具有形式安全保证的最终解。

## 7 系统验证与局限

### 7.1 原型可实现性

现有 T41 原型已经实现 JPEG/HTTP、有界异步队列和轻量线性决策器。历史组件实验显示，选择器计算相对端侧检测和 JPEG 编码较小，主要系统开销来自端侧推理、编码、网络和云服务。该原型说明在线执行路径可以实现，但其检测器、图像编码和网络条件并非本文冻结三数据集控制器的完整同协议部署。

本文没有把历史组件延迟直接相加为当前 E2E，也没有把公式估算冒充异机实测。投稿系统方向前仍需在明确的端侧设备与服务器上测量编码、上行传输、排队、云端推理、下行响应、吞吐和能耗，并覆盖 1/5/10/20 Mbps、RTT 与抖动条件。

### 7.2 有效性威胁

第一，COCO val2017 和 ENA24 final 是固定 diagnostic；Oregon final 已被既有 canonical 实验访问，本轮属于 post-hoc exploratory reuse。严格确认性结论仍需要新的隐藏测试或预注册外部数据。

第二，ENA24 只实现近重复组隔离，缺少事件、相机和位置元数据。Oregon 采用 random-image 主协议，同一相机背景可能跨数据角色出现；其图像级 bootstrap 也忽略事件簇相关性。因此 Oregon 结果不能解释为跨相机或跨地点泛化。

第三，200 次 bootstrap 只产生逐工作点经验区间，未进行 30 个工作点的同时误差控制。相同动作快捷通过也不能替代在 gate 上观察到 ROI 差异后的安全验证。

第四，排序学习器使用可学习代理目标，而 Official AP 不是逐图可加指标。final oracle 是同一 24 候选集合内的后验诊断，不是不可实现的全局 AP oracle，也不得用于修复主结果。

第五，ROI 依赖端侧候选框，无法修复完全漏检；当前动作空间也未覆盖 tile packing、任务感知编码和特征压缩。冻结结果中 ROI 比例很低，尚不能证明 ROI 具有跨域普适价值。

第六，Estimated E2E 不是实测。新的成本感知控制器尚未完成冻结异机链路的端到端时延、吞吐和能耗测量，因此本文不声称真实实时性能优势。

第七，端侧和云端使用同系列检测器，错误可能高度相关。更异构或专用的云端检测器可能改变可路由收益及最佳载荷策略。

## 8 结论

本文提出一种面向端云目标检测的校准式成本感知路由与自适应载荷选择框架。该框架将样本 Ranker、Original/Q90/ROI 载荷动作和 Fixed-call/Online/Fixed-byte 预算调度解耦，并在统一检测缓存上通过配置级 Official AP 与实际请求字节进行选择。对于调用量和在线模式，控制器在 AP 容忍约束下最小化请求成本；对于字节预算，控制器在容量约束内最大化检测精度。独立配对 bootstrap 门禁用于冻结候选或回退参考配置，final 不参与选择或修复。

三数据集 30 个冻结工作点中，13 个通过门禁、17 个回退；剔除 6 个 gate 动作相同点后，非平凡通过只有 7 个。ENA24 获得 0.74-1.73 KB/image 的小幅成本下降，Oregon 的保留变化较小，COCO 全部回退。不同数据集最终依赖不同 Ranker，GBDT-Hierarchical 也不是统一有效的 ROI 策略。本文的核心结论不是某个学习器跨域最优，而是样本价值、载荷成本和预算语义必须被解耦并在 final 前校准冻结。当前门禁提升了实验可审计性，但其 regret 和 false negative 表明，扩大独立 gate、使用聚类感知不确定性估计和完成真实系统测量仍是必要工作。

## Statements and Declarations

### Funding

[作者确认：基金名称、项目编号；若无基金，请写明。]

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

[作者确认：按 CRediT 角色填写 Conceptualization、Methodology、Software、Validation、Writing 等贡献。]

### Data availability

COCO 可从官方渠道获取。ENA24 与 Oregon 的公开性、许可证和下载地址须由作者确认。本文生成的角色清单、冻结配置、来源哈希和匿名化汇总应随论文或补充材料发布，但不能在未确认数据许可前公开敏感图像或位置元数据。

### Code availability

[作者确认：匿名代码仓库、归档 DOI 和公开时间。] 复现材料应包含协议文件、缓存清单、选择器、风险门禁、汇总脚本和环境信息。

### Ethics and consent

本文不进行身份识别。涉及人物、私有相机或敏感野生动物位置的数据，在公开前应满足相应知情同意、匿名化、机构伦理和数据许可要求。

## 参考文献

[1] Kang, Y., Hauswald, J., Gao, C., et al.: Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. ASPLOS, 615-629 (2017). https://doi.org/10.1145/3037697.3037698

[2] Matsubara, Y., Levorato, M., Restuccia, F.: Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges. ACM Computing Surveys 55(5), 1-30 (2023). https://doi.org/10.1145/3527155

[3] Zhou, Z., Chen, X., Li, E., et al.: Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. Proceedings of the IEEE 107(8), 1738-1762 (2019). https://doi.org/10.1109/JPROC.2019.2918951

[4] Teerapittayanon, S., McDanel, B., Kung, H.T.: BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. ICPR, 2464-2469 (2016). https://doi.org/10.1109/ICPR.2016.7900006

[5] Xu, R., Razavi, S., Zheng, R.: Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques. IEEE Communications Surveys & Tutorials 25(4), 2951-2982 (2023). https://doi.org/10.1109/COMST.2023.3323091

[6] Ran, X., Chen, H., Zhu, X., Liu, Z., Chen, J.: DeepDecision: A Mobile Deep Learning Framework for Edge Video Analytics. IEEE INFOCOM, 1421-1429 (2018). https://doi.org/10.1109/INFOCOM.2018.8485905

[7] Yi, S., Hao, Z., Zhang, Q., et al.: LAVEA: Latency-Aware Video Analytics on Edge Computing Platform. ACM/IEEE SEC, 1-13 (2017). https://doi.org/10.1145/3132211.3134459

[8] Li, Y., Padmanabhan, A., Zhao, P., et al.: Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. ACM SIGCOMM, 359-376 (2020). https://doi.org/10.1145/3387514.3405874

[9] Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M.: NoScope: Optimizing Neural Network Queries over Video at Scale. PVLDB 10(11), 1586-1597 (2017). https://doi.org/10.14778/3137628.3137664

[10] Zhou, H.-Y., Gao, B.-B., Wu, J.: Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors. ICCV, 3505-3513 (2017). https://doi.org/10.1109/ICCV.2017.379

[11] Cao, Z., Cheng, Y., Zhou, Z., et al.: Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. ICDCS (2023); preprint: https://arxiv.org/abs/2108.12858

[12] Cao, Z., Cheng, Y., Zhou, Z., et al.: Edge-Cloud Collaborated Object Detection via Bandwidth Adaptive Difficult-Case Discriminator. IEEE Transactions on Mobile Computing 24(2), 1181-1196 (2025). https://doi.org/10.1109/TMC.2024.3474743

[13] Qiu, J., Wang, R., Hu, B., Guerin, R., Lu, C.: Optimizing Edge Offloading Decisions for Object Detection. ACM/IEEE Symposium on Edge Computing (2024). https://doi.org/10.1109/SEC62691.2024.00021

[14] Chen, T., Bu, Y., Zeng, Y., Xie, L., Lu, S.: RegionFilter: Region-Aware Video Filtering Mechanism on Resource-Constrained Edge Nodes. Computer Networks 251, 110624 (2024). https://doi.org/10.1016/j.comnet.2024.110624

[15] Yang, Z., Wang, X., Wu, J., et al.: EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision. IEEE/ACM Transactions on Networking 31(4), 1765-1778 (2023). https://doi.org/10.1109/TNET.2022.3223412

[16] Du, K., Pervaiz, A., Yuan, X., et al.: AccMPEG: Optimizing Video Encoding for Accurate Video Analytics. Proceedings of Machine Learning and Systems 4 (2022). https://proceedings.mlsys.org/paper_files/paper/2022/hash/853f7b3615411c82a2ae439ab8c4c96e-Abstract.html

[17] Mozannar, H., Sontag, D.: Consistent Estimators for Learning to Defer to an Expert. ICML, PMLR 119, 7076-7087 (2020). https://proceedings.mlr.press/v119/mozannar20b.html

[18] Verma, R., Nalisnick, E.: Calibrated Learning to Defer with One-vs-All Classifiers. ICML, PMLR 162 (2022). https://proceedings.mlr.press/v162/verma22c.html

[19] Mao, A., Mohri, M., Zhong, Y.: Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees for Learning to Defer. ICML, PMLR 267, 43035-43066 (2025). https://proceedings.mlr.press/v267/mao25c.html

[20] Gomez, C., Patkhullaev, D., Hernandez, A.C.: OffloaD: Detection Failure-Based Scheduler for Offloading Object Detection. ACM Transactions on Embedded Computing Systems 23(5), 1-26 (2024). https://doi.org/10.1145/3677321

[21] Said, N., Landsiedel, O.: EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading. Computer Networks 269, 111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437

[22] Choi, H., Bajic, I.V.: Deep Feature Compression for Collaborative Object Detection. ICIP (2018). https://arxiv.org/abs/1802.03931

[23] Elgamal, T., Shi, S., Gupta, V., Jana, R., Nahrstedt, K.: SiEVE: Semantically Encoded Video Analytics on Edge and Cloud. ICDCS (2020). https://doi.org/10.1109/ICDCS47774.2020.00182

[24] Jitkrittum, W., Gupta, N., Menon, A.K., Narasimhan, H., Rawat, A.S., Kumar, S.: When Does Confidence-Based Cascade Deferral Suffice? NeurIPS (2023). https://arxiv.org/abs/2307.02764

[25] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of Modern Neural Networks. ICML, PMLR 70, 1321-1330 (2017). https://proceedings.mlr.press/v70/guo17a.html

[26] Angelopoulos, A.N., Bates, S., Candes, E.J., Jordan, M.I., Lei, L.: Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. The Annals of Applied Statistics 19(2) (2025). https://doi.org/10.1214/24-AOAS1998

[27] Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman & Hall/CRC (1993).

[28] Lin, T.-Y., Maire, M., Belongie, S., et al.: Microsoft COCO: Common Objects in Context. ECCV, 740-755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

[29] Jocher, G., Qiu, J., Liu, M., et al.: Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748
