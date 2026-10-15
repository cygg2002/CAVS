# 端云协同目标检测的风险受控配置采纳：可审计校准协议与收益边界分析

Risk-Controlled Configuration Adoption for Edge-Cloud Object Detection: An Auditable Calibration Protocol with Benefit-Boundary Analysis

[作者确认：作者姓名]

[作者确认：作者单位；通讯作者姓名与电子邮箱]

## 摘要

资源受限视觉终端通常只能运行轻量目标检测器，而持续上传图像会占用上行带宽并引入编码、传输与云端推理开销。已有选择性卸载方法多以点估计在校准数据上挑选策略，选择过程缺乏统计风险控制，跨域部署时难以审计。本文研究一个可直接回答的问题：在无法保证学习型样本排序跨域有效的前提下，系统如何以可审计的方式采纳成本更低的卸载配置？为此提出风险受控的配置采纳协议：在线决策解耦为样本排序、载荷动作（Original/Q90/ROI）与预算调度（Fixed-call/Online/Fixed-byte），离线控制器仅使用拟合集、选择集与门禁集完成配置级成本-精度校准，临时候选须通过独立门禁集上 200 次配对 bootstrap 的经验风险检验，否则回退参考配置，终评集只评一次。在 COCO、ENA24 与 Oregon 三个数据域共 30 个冻结工作点上的探索性评测显示：门禁通过 13 个（非同动作比较 7 个）、回退 17 个；终评精度变化全部处于 -0.83×10⁻³ 至 +0.10×10⁻³ 之间，未出现一次容忍违反；ENA24 低预算获得 1.5%-1.8% 的请求字节节省。协议同时给出收益边界：冻结选择与终评 oracle 诊断一致表明不存在跨域统一最优的排序器或载荷策略；学习型排序收益仅在 COCO 一致超出 50 种子随机波动带；门禁保守性主要源于统计功效不足，误拒机会成本被完整量化。本文的贡献是可审计的配置采纳协议及其揭示的收益边界，而非某个跨域通用学习器。

**关键词：** 端云协同；目标检测；选择性卸载；配置校准；风险门禁；可审计评测

## 1 引言

低功耗智能摄像头被广泛用于野外监测、安防巡检和无人值守感知。轻量检测器能够在本地提供基本输出，但其模型容量、输入分辨率和硬件算子受限，对小目标、遮挡和域外场景的识别能力通常弱于云端高容量检测器。全量上云可以利用更强模型，却要求持续传输图像，并使响应时间受到网络往返、服务排队和云端计算的共同影响。因此，端云协同检测的核心不是简单选择端侧或云端，而是在有限资源下决定哪些图像值得上传以及上传何种表示。

现有研究已经证明选择性卸载具有价值。Adaptive Feeding 使用 SVM 在快速与高精度检测器之间选择 [10]；DCSB 及其带宽自适应版本通过难例判别器决定是否上云 [11,12]；OffloaD 根据检测失败风险调度任务 [20]；ORIC/MORIC 进一步指出，目标检测 mAP 是跨图像、跨类别聚合的指标，上传收益不应简化为单图分类正确性 [13]。这些方法主要关注"上传哪些图像"，且多数使用整图载荷。

通信成本还取决于"上传什么"。Original 保留完整图像信息但字节量最高；Q90 JPEG 可以显著减少请求量，却可能损害细粒度纹理；ROI 进一步压缩空间范围，但依赖端侧候选框，并可能因漏检和上下文缺失而降低云端检测质量。RegionFilter、EdgeDuet 和 AccMPEG 分别从区域过滤、小目标 tile 和任务感知编码角度说明，空间内容与编码质量会改变精度和传输成本 [14-16]。因此，只优化上传图像数量或固定采用 ROI，都不能充分描述端云系统的成本收益。

样本排序、载荷选择和预算调度之间还存在耦合。Fixed-call（固定调用比例）固定上传样本数，适合比较排序能力；Online（在线阈值）通过冻结阈值逐样本接纳任务，但实际上传率会随数据分布漂移；Fixed-byte（固定字节容量）限制总请求字节，此时更小的载荷还可能使预算容纳更多样本。同一策略在 Fixed-call 中可能因裁剪损失而不利，却在 Fixed-byte 中因释放容量而有效。若不区分预算语义，容易把上传率漂移或载荷压缩造成的收益错误归因于样本学习器。

更重要的是，配置选择本身存在后验乐观风险：在同一份校准数据上既做候选排序又做阈值调校，选出的"最优"配置往往在部署后无法复现校准阶段的收益。已有工作普遍以点估计选择策略或配置，选择过程与评估数据不分离，也没有失败后的回退语义。本文据此研究一个可直接回答的问题：**在无法保证学习型样本排序跨域有效的前提下，系统如何以可审计的方式采纳成本更低的卸载配置？**两条动机观察（数据见 6.5 节）使该问题的前提成立：其一，终评 oracle 诊断显示三个数据域的最优排序器族互不相同，预设任何单一学习器都无法跨域成立；其二，50 种子随机波动带显示样本排序的可得收益本身是域依赖的，在两个野生动物域与随机波动同量级。因此问题的重心不是训练更强的排序器，而是在给定候选集合上构造可信的选择、验证与回退机制。

本文提出的风险受控配置采纳协议如下：系统在线只执行一个冻结的排序器、载荷策略和预算调度器；离线控制器在选择集上求解配置级精度-成本问题，并在独立门禁集上以配对 bootstrap 检验候选相对参考配置的精度与字节差异，检验失败即回退参考配置。该设计不把某个排序学习器视为跨域通用解，而把端侧置信度、类别收益先验和检测收益学习器作为可替换信号。

本文的主要贡献如下。

1. 提出面向端云检测卸载的风险受控配置采纳协议：拟合、选择、门禁与终评四层数据角色分离，配置级校准在精度容忍约束下最小化实际请求字节，独立门禁集上 200 次配对 bootstrap 检验决定冻结候选或回退参考配置——若无该门禁，COCO 的 6 个候选将以 -4.2×10⁻³ 至 -3.3×10⁻³ 的门禁集精度区间下界进入部署。该协议可视为风险控制校准思想 [26] 在端云检测字节预算配置采纳上的首次落地。

2. 将预算语义显式化为 Fixed-call、Online 与 Fixed-byte 三种，并以实际请求字节为成本口径统一评测 Original、Q90 与 ROI 载荷动作，使上传率漂移与载荷压缩的收益不再被错误归因于样本学习器；全部评测在不重新训练端侧和云端检测器的条件下完成。

3. 给出收益边界的系统刻画：在三个数据域、每域 240 个候选共 720 个配置上，冻结选择与终评 oracle 诊断一致显示无跨域统一最优排序器；学习型排序收益仅在 COCO 一致超出 50 种子随机波动带；门禁功效分析将"保守性"主要归因于统计功效不足，误拒的机会成本被完整量化。

4. 完整保留 17 个回退与全部负结果，并随论文发布冻结协议、30 个工作点配置总表与复现脚本。需要说明，本文贡献不在排序器或载荷动作本身（二者均有先例，见表 1），而在配置级"校准-门禁-回退"的可审计采纳协议及其揭示的收益边界。

本文余下部分组织如下。第 2 节回顾相关工作并界定与已发表方法的可比性边界；第 3 节定义问题并给出置信度排序的充分与失效条件；第 4 节介绍系统架构与控制器；第 5 节说明实验协议；第 6 节报告冻结结果与诊断分析；第 7 节讨论系统验证与局限；第 8 节总结全文。

## 2 相关工作

### 2.1 选择性卸载与检测级路由

Neurosurgeon 依据设备、网络和服务器代价选择网络切分位置 [1]，split computing 与早退研究进一步分析中间特征传输和多出口推理的成本 [2-4]。视频分析系统还通过模型选择、边缘调度和帧过滤减少不必要的推理或传输 [5-9]；Chameleon 进一步表明配置选择本身（帧率、分辨率、模型）是视频分析的核心杠杆，并依赖周期性 profile 重新选点 [30]。本文不切分单个检测器，也不依赖视频冗余，而是让端侧检测器先产生可直接使用的结果，再选择部分图像交给独立云端检测器复核；与配置 profile 类工作的关键差异是选择过程本身受统计门禁约束（见 2.4 节）。

Adaptive Feeding 使用 SVM 判断图像应由快速还是高精度检测器处理 [10]。DCSB 使用检测语义特征识别难例，其后续版本增加带宽自适应和区域采样 [11,12]。ORIC/MORIC 通过上下文收益（contextual reward）估计一张图对整体检测 mAP 的边际贡献 [13]。OffloaD 以检测失败为卸载信号 [20]。EdgeBoost 则在分类任务中校准端侧置信度或 margin 后上传低置信样本 [21]。这些工作说明置信度是必要的强基线，也说明端侧不确定性与云端相对收益并不总是等价。

### 2.2 ROI、压缩与任务感知传输

RegionFilter 使用区域信息减少资源受限节点的视频传输 [14]；EdgeDuet 将低分辨率端侧检测与高分辨率 tile 处理结合，面向小目标场景降低带宽 [15]；AccMPEG 根据图像区域对下游分析精度的影响分配编码质量 [16]。特征压缩和语义编码则通过传输中间表示降低通信 [22,23]。与这些方法不同，本文不改变检测器或训练新的编码器，而是在统一缓存上比较 Original、Q90 和 ROI，并把实际请求字节直接纳入配置选择。

### 2.3 成本敏感路由与风险校准

Learning to defer 将模型与下游专家之间的分配建模为条件风险最小化问题 [17]，后续工作讨论了校准式 defer 和多专家路由 [18,19]。FrugalML 在预算约束下学习多个付费推理服务之间的调用策略，其收益被明确归因于服务间的互补强弱 [32]。现代神经网络置信度可能与真实风险不一致 [25]，因此仅使用一个未校准阈值并不能保证部署成本或精度。Learn then Test（LTT）提供了在独立数据上筛选候选并控制风险的通用框架 [26]。

本文研究具有非逐图可加官方评测精度的目标检测，同时决定样本与像素载荷，并区分调用量与实际字节预算。本文的门禁可视为 LTT 式风险控制校准在端云检测字节预算配置采纳上的落地：对临时候选执行"精度差不低于 -ε"的检验，失败即回退安全默认配置；与 [26] 的差异在于本文使用 200 次配对 bootstrap 的经验区间而非有限样本 p 值，因此不宣称形式化或同时性安全保证，而将其视为防止选择集点估计过拟合的经验风险控制步骤。最终配置必须在终评前冻结。

### 2.4 与已发表方法的可比性边界

本文未与文献 [10-16,20,21] 进行同协议的直接数值对比：这些方法使用不同的检测器、数据集与成本口径，且多数没有公开可复用的统一预测与载荷缓存，直接移植数值会混淆方法差异与协议差异。表 1 列出各方法与本文协议的关键差异，以及本文候选集合中对应的内部基线。其中 MORIC-style 仅复用 [13] 的上下文收益信号构造思想，GEC-style 为本文自建的组合特征基线，二者均不是对外部方法的忠实复现；在统一缓存协议下复现代表性外部方法（尤其与本文场景最接近的 DCSB 难例判别器 [11]）是后续工作。

**表 1** 代表性方法与本文协议的差异。"统计门禁/回退"指配置或策略被采纳前是否经过独立数据上的统计风险检验、失败时是否有预注册的回退语义；本文所有数值比较均在自建候选集合内部进行，外部方法的数值不可直接移植

| 方法 | 决策粒度与机制 | 载荷形式 | 预算语义 | 统计门禁/回退 | 与本文协议的主要差异 | 本文对应内部基线 |
|---|---|---|---|---|---|---|
| Adaptive Feeding [10] | 整图、双检测器选择 | 整图 | 无显式预算 | 无 | 在两个检测器间切换，不做上云复核与字节核算 | Confidence |
| DCSB [11,12] | 整图难例判别 | 整图（后续版含区域采样） | 上传率 | 无 | 固定阈值/上传率；检测器与数据不同 | Confidence、M2E |
| ORIC/MORIC [13] | 上传收益估计 | 整图 | 调用量 | 无 | 收益口径与训练目标不同；本文仅复用其思想 | MORIC-style |
| OffloaD [20] | 检测失败风险调度 | 整图 | 队列/时延 | 无 | 以失败风险为信号，面向调度而非字节预算 | Confidence-Residual M2E |
| EdgeBoost [21] | 校准置信度阈值 | 整图 | 上传率 | 无 | 分类任务；无检测级收益与载荷选择 | Confidence（Online 模式） |
| RegionFilter/EdgeDuet/AccMPEG [14-16] | 区域/tile/编码质量 | 区域或重编码流 | 带宽 | 无 | 需修改编码或传输栈；本文保持冻结检测器与标准 JPEG | ROI 载荷动作 |
| NoScope/Chameleon [9,30] | 配置 profile 点估计选点 | 视频流 | GPU 时间/帧率 | 无 | 选择与评估数据不分离；无失败回退 | 本文离线控制器（对照） |
| FrugalML [32] | 预算约束下的服务调用策略 | 整图 | 调用费用 | 无 | 分类任务；点估计策略学习 | ClassGainPrior |
| Learn then Test [26] | 风险控制校准（通用框架） | — | — | 有限样本 p 值 | 无端云检测与字节预算落地 | 本文门禁的理论母本 |
| 本文 | 配置级校准 + 门禁 + 回退 | 整图/Q90/ROI | 调用量与实际字节 | 200 次配对 bootstrap + 回退 | — | — |

## 3 问题定义与成本感知动机

本文使用以下数据角色术语（英文对照见附录 B）：拟合集（fit）用于训练排序器与动作模型；选择集（selector）用于配置选择与阈值校准；门禁集（gate，部分实验记录中亦称 audit，二者同义）只用于风险检验；终评集（final）不参与拟合、选择、阈值校准或回退，仅在配置冻结后评测一次。

### 3.1 端云检测与动作空间

给定图像集合 \(\mathcal D=\{(I_i,Y_i)\}_{i=1}^{N}\)，端侧检测器 \(D_e\) 输出检测集合 \(E_i=D_e(I_i)\)。未上传时，系统直接输出 \(E_i\)。若图像被上传，载荷动作属于

\[
\mathcal A=\{\mathrm{Original},\mathrm{Q90},\mathrm{ROI}\}.
\]

云端检测器 \(D_c\) 对载荷执行复核，并经尺度恢复、坐标回映射和框融合得到 \(F_i(a)\)。令 \(c_i(a)\) 表示动作 \(a\) 的实际请求字节，边缘保留动作记为 \(a=\mathrm{Edge}\)，其请求成本为零。

### 3.2 样本价值、载荷成本与置信度排序的边界

在可加代理效用下，动作 \(a\) 对图像 \(i\) 的净价值可写为

\[
V_i(a;\lambda)=\mathbb E[\Delta U_i(a)\mid z_i]-\lambda c_i(a),
\]

其中 \(\Delta U_i(a)\) 表示云端动作相对端侧输出的检测效用增量，\(z_i\) 是在线可观测的端侧检测、图像统计和载荷属性，\(\lambda\ge 0\) 是把字节成本折算进净价值的乘子。理想动作满足

\[
a_i^*=\arg\max_{a\in\{\mathrm{Edge}\}\cup\mathcal A}V_i(a;\lambda).
\]

该表达式说明，端侧不确定性、云端修复收益和单位字节收益是三个不同量。已有级联决策理论同样指出，置信度延期只在特定风险结构下充分 [24]。为说明本文采用成本感知决策而非预设置信度排序的原因，给出如下观察。

**观察 1（置信度排序的充分条件与失效条件）。** 考虑 Fixed-call 预算 \(|\mathcal U|=K\) 与可加代理目标 \(\sum_{i\in\mathcal U} v_i\)，其中 \(v_i=g_i-\lambda c_i\)，\(g_i\) 为图像 \(i\) 最优云动作相对端侧的期望效用增量，\(c_i\) 为对应载荷字节。设端侧不确定性分数 \(h_i\) 互异。若 (i) \(g_i=g(h_i)\) 为 \(h_i\) 的单调不减函数，且 (ii) 所有样本的最优载荷成本相同（\(c_i\equiv c\)），则按 \(h_i\) 从高到低选择前 \(K\) 个样本对任意 \(\lambda\ge 0\) 最大化该目标。反之，若存在样本对满足 \(h_i>h_j\) 而 \(v_i<v_j\)，则存在使该样本对跨越选择边界的预算 \(K\)，使置信度排序严格次优。完整陈述与交换论证证明见附录 A。

观察 1 只在逐图可加的代理效用层面成立，作用是说明不能预设置信度为充分统计量：实际检测中，类别、目标尺寸、拥挤程度、端云错误互补性以及 Original/Q90/ROI 的字节差异都会使 \(g_i\) 与 \(c_i\) 异质，从而破坏条件 (i) 或 (ii)。同时必须强调，官方评测精度（COCO-style mAP50-95，下称 Official AP）不是逐图可加指标，因此本文不把单图代理收益当作精度增量：第 4.4 节的控制器不优化任何逐图代理，而是在配置级用统一评估器直接计算选择集上的 Official AP 与实际请求字节。

### 3.3 配置与预算模式

样本排序器（Ranker）\(r\) 根据 \(z_i\) 产生分数 \(s_i^{(r)}\)，载荷策略 \(\pi\) 为已接纳样本选择动作。完整配置定义为

\[
\theta=(r,\pi,m,B,\tau,\lambda),
\]

其中 \(m\) 是预算模式，\(B\) 是预算，\(\tau\) 是动作门限，\(\lambda\) 是 Fixed-byte 调度乘子。Fixed-call 严格接纳得分最高的 \(K=\lfloor BN\rfloor\) 张图像：

\[
\mathcal U_{call}=\operatorname{TopK}(\{s_i^{(r)}\}_{i=1}^{N},K).
\]

Online 使用选择集上冻结的阈值 \(t_{r,B}\)：

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

系统包含在线执行和离线校准两个阶段，如图 1 所示。在线阶段依次执行端侧检测、样本评分、预算接纳、载荷选择、云端复核与结果融合；未接纳图像立即使用端侧结果，接纳图像按冻结动作发送云端。离线阶段利用冻结检测缓存和独立数据角色完成候选校准、风险门禁和配置冻结。离线控制器不修改端侧或云端检测器权重，各候选共享相同预测、载荷和评估器缓存。

**[图 1：可审计成本感知端云协同架构。在线阶段只执行一个冻结配置；离线阶段完成候选选择、风险门禁和一次性终评。]**

### 4.2 样本卸载信号

框架不绑定单一排序器。公共特征由 17 维端侧检测与图像统计量，以及仅在拟合集上拟合的 32 维 FPN-PCA 表示组成。Confidence 排序器的分数定义为 1 减去部署检测框（端侧置信度不低于 0.25 的检测框）中的最低置信度；无部署框的图像赋予固定最高分，使空检图像最先上传。ClassGainPrior 估计类别条件云端收益；M2E 及其 residual 变体学习端云收益代理；SemanticRank 使用公共语义特征进行成对排序；MORIC-style 提供上下文收益相关基线，GEC-style 为本文自建的组合特征基线（M2E、Confidence 与 ClassGainPrior 分数经拟合集分位归一后的固定加权组合；见 2.4 节可比性声明）；Current-Controlled-32D（下文缩写 Controlled-32D）用于检验受控语义维度结构。

自动配置只允许选择 8 个确定性排序器：Confidence、ClassGainPrior、M2E、Confidence-Residual M2E、SemanticRank、MORIC-style、GEC-style 和 Controlled-32D。Random 使用 50 个种子评估随机波动（其波动带在 6.5 节作为诊断参照报告），不参与部署选择。Legacy-Frozen-32D 依赖历史冻结模型，只作为兼容性实验，不进入本次严格三数据集控制器。

### 4.3 自适应载荷策略

三种载荷策略共享同一 Original/Q90/ROI 缓存。

**SharedAction（共享动作头）。** 使用仅在拟合集上训练的共享动作头选择 Original、Q90 或 ROI，作为每个工作点的参考策略。动作头不读取排序器标识或排序分位，确保样本排序与载荷选择可独立比较。

**CostEligible-AllROI（成本可行 ROI 门控）。** 当 ROI 可用且请求字节小于 SharedAction 基础动作时切换为 ROI，否则保留基础动作。该方法是简单成本门禁，用于检查复杂动作模型是否具有必要性。

**GBDT-Hierarchical（分层 GBDT 动作头）。** 先得到 SharedAction 基础动作，再使用仅在拟合集上训练的 GBDT 估计 ROI 是否在降低字节的同时保持可接受质量。训练正例要求 ROI 的逐图质量代理不低于基础动作减 0.02（该阈值是逐图代理尺度上的训练标签容差，与 4.4 节配置级精度容忍量 \(\epsilon=0.001\) 属于两个不同尺度，二者无对应关系），且 ROI 请求字节更少。输入包括 49 维公共特征，Q90/Original 字节比、ROI/Original 字节比与 ROI 面积比（各 1 维），以及 Q90 与 ROI 载荷可用性（各 1 维），共 54 维。模型采用平衡样本权重、150 次提升迭代、0.05 学习率、最大深度 3、最小叶节点样本数 20 和 \(L_2=1.0\)；随机种子与全部超参数在冻结协议文件中记录，未做调优。动作阈值按数据集和预算模式在选择集上校准，并在全部非随机排序器间共享。

若 GBDT-Hierarchical 与 CostEligible-AllROI 产生相同动作，二者按等价结果报告，不以模型复杂度作为贡献。

### 4.4 配置级成本优化

令 \(Q(\theta;\mathcal S)\) 表示配置 \(\theta\) 在集合 \(\mathcal S\) 上的 Official AP，令

\[
C(\theta;\mathcal S)=\frac{1}{|\mathcal S|}\sum_{i\in\mathcal S}\frac{c_i(a_i)}{1024}
\]

表示按全部输入图像均摊的请求 KB/图。每个数据集、预算模式和工作点先在 SharedAction 配置中选择选择集 Official AP 最高者作为参考配置 \(\theta_{ref}\)。精度容忍量为预注册的 \(\epsilon=0.001\)，至多为三域端云精度差（0.029-0.166）的 3.4%；其与各门禁集统计功效的匹配关系在 6.7 节定量分析。

对于 Fixed-call 和 Online，控制器求解

\[
\min_{\theta\in\Theta}C(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.
\]

对于 Fixed-byte，总请求容量已由预算限制，控制器求解

\[
\max_{\theta\in\Theta}Q(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad C(\theta;\mathcal D^{sel})\le C_B,\;Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.
\]

Fixed-call 和 Online 在最低成本并列时依次选择精度更高、策略更简单和预注册排序器顺序更靠前的配置；Fixed-byte 在最高精度并列时依次选择请求字节更低、策略更简单和预注册排序器顺序更靠前的配置。策略简单性顺序冻结为 SharedAction、GBDT-Hierarchical、CostEligible-AllROI。

### 4.5 独立风险门禁与冻结回退

临时候选配置 \(\theta_{prov}\)（provisional）与参考配置在门禁集上进行 200 次配对图像级 bootstrap [27]。第 \(b\) 次重复同时对两种配置使用相同图像索引，计算

\[
\Delta Q^{(b)}=Q(\theta_{prov};\mathcal D_b^{gate})-Q(\theta_{ref};\mathcal D_b^{gate}),
\]

\[
\Delta C^{(b)}=C(\theta_{prov};\mathcal D_b^{gate})-C(\theta_{ref};\mathcal D_b^{gate}).
\]

本文取 200 个重采样差值的 2.5% 和 97.5% 分位数作为逐工作点经验区间。Fixed-call 和 Online 要求精度差区间下界不低于 \(-\epsilon\)，且请求 KB 差区间上界不高于 0；Fixed-byte 只执行精度门禁。冻结配置为

\[
\theta_{frozen}=\begin{cases}\theta_{prov},&\text{门禁通过},\\\theta_{ref},&\text{门禁失败}.\end{cases}
\]

门禁失败后不尝试第二候选。配置、阈值和来源哈希写入冻结文件后，终评集只评估一次；终评 oracle 仅用于计算诊断性差距（regret），不改变冻结配置。在形式上，该门禁是风险控制校准 [26] 的经验版特例：检验对象为配置级精度差，失败动作为回退安全默认配置；但它是逐点经验筛选，不构成 30 个工作点上的同时性统计保证，其统计功效与门禁集规模和动作差异范围的关系在 6.7 节定量分析。

**算法 1：成本感知配置校准与风险门禁。** 输入选择集、门禁集、候选集合、预算模式和容忍量；首先在 SharedAction 候选中确定参考配置，删除选择集精度低于参考减容忍量的候选；Fixed-call/Online 最小化请求 KB，Fixed-byte 最大化 Official AP；随后在门禁集上执行配对 bootstrap，候选通过则冻结，否则直接冻结参考配置；写入选择、区间、回退状态和来源哈希后，终评集只评估冻结配置，oracle 仅作诊断。

### 4.6 在线开销

配置搜索只在离线执行。在线系统只运行一个冻结排序器、一个载荷策略和一次预算判断。Confidence 与线性排序器的开销相对端侧检测较小；Controlled-32D 按冻结协议计入 0.08 ms/图的评分时延与 32 个排序权重的参数量。GBDT 动作头开销单独记录，不混入第 5.4 节的估算端到端时延。

## 5 实验设置

### 5.1 数据集与数据角色

本文使用 COCO [28]、ENA24 和 Oregon Critters。COCO 提供 80 类通用场景对照。ENA24 是 22 类目标存在型野生动物检测集，当前划分不含空图，可核验子集的平均标注约 1.1 框/图，因此不能据此评价空帧过滤能力。Oregon Critters 是 45 类野生动物相机图像集，原图分辨率和字节量较高，用于检验类别长尾与背景域变化下的成本收益。两个野生动物数据集的正式出处与许可证信息见 Data availability 声明。

每个数据集按用途划分为拟合集、选择集、门禁集和终评集，划分规模见表 2。

**表 2** 数据角色划分与评测边界

| 数据集 | 拟合集 | 选择集 | 门禁集 | 终评集 | 评测边界 |
|---|---:|---:|---:|---:|---|
| COCO | 9,678 | 571 | 630 | 5,000 | val2017 固定诊断口径 |
| ENA24 | 1,400 | 200 | 200 | 1,758 | 选择/门禁近重复组隔离 |
| Oregon | 9,923 | 4,961 | 9,923 | 4,962 | 门禁独立；终评为事后探索性复用 |

ENA24 的选择集与门禁集来自同一个 400 张校准池：先对每张图计算 16×16 灰度 dHash，将汉明距离不超过 6 的样本连接为近重复分量，再以固定种子将完整分量分配至两侧，共形成 385 个连通组，组间零交叉。由于缺少可靠的事件、相机和位置元数据，该划分只能称为近重复组隔离（near-duplicate-group disjoint），不能声称事件或相机隔离。Oregon 采用冻结的随机图像级主协议，相机背景可能跨数据角色出现；其终评集曾被上游实验访问，本轮属于事后探索性复用（post-hoc exploratory reuse）。这两条边界在 7.2 节作为有效性威胁讨论。

### 5.2 检测器与载荷

统一评测使用冻结的 YOLO26n 级端侧检测器与 YOLO26x 级云端检测器；YOLO26 为 2026 年预印本，本文不把检测器结构作为创新 [29]。三个数据集分别沿用各自已冻结的权重、预测缓存、类别映射和数据划分，不跨数据集共享归一化参数或动作阈值。COCO 的绝对精度采用本文冻结的终评预测缓存口径，不与更早的 COCO val 诊断表合并。

Original、640-Q90 和 ROI 载荷均预先生成并记录实际字节。Q90 将长边缩放至 640 像素，使用 JPEG quality 90 和 4:2:0 色度抽样。ROI 由端侧置信度不低于 0.25 的候选框生成，按候选面积动态扩框，裁剪总面积上限为原图的 65%，并以 Q90 编码；每张图最多触发一次云调用，云端框经坐标回映射后以 IoU 0.5 的融合规则（MAUR，Merge-After-Upload-Refinement，即上传复核后按 IoU 匹配合并端云检测框）与端侧结果融合。ROI 生成参数在拟合阶段固定，未做调优。ROI 不可用时按冻结策略回退至 Q90 或 Original。所有候选共享相同检测缓存、载荷缓存和评估器，不根据终评结果修改任何动作。

### 5.3 对比方法与工作点

系统参考包括 Edge-only、Cloud-only Original、Always Q90、Always ROI fallback 和共享动作头全上传（全部图像上传、载荷动作由 4.3 节的共享动作头选择，实验记录中称 Shared ContextM2E）。样本排序基线包括 Random 以及 4.2 节的 8 个确定性排序器；其中 MORIC-style 与 GEC-style 为外部方法思想的同协议简化对照（复现口径见 2.4 节与 4.2 节，不代表对原方法的忠实复现），使全部排序器对比在同一缓存、同一评估器与同一预算语义下完成。载荷策略为 SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical。每个工作点有 24 个可部署候选，Random 不进入自动选择。

注册工作点为 Fixed-call 10%/25%/40%，Online 名义 10%/25%/40%，以及 Fixed-byte Original 字节预算 5%/10%/20%/40%。每个数据集共 10 个工作点、240 个选择集候选，三个数据集共 720 个候选。主方法是选择集选择并经门禁检验后的冻结配置；主要对照为各工作点最佳 SharedAction 参考配置、未经门禁的临时候选和诊断性终评 oracle。原计划的"三数据集共用一个配置"消融截至证据冻结日尚未执行，本文不将其列为完成实验，相关问题只能部分地由 6.5 节的诊断分析回答。

### 5.4 评测指标

检测指标以 conf=0.001 的 COCO-style Official mAP50-95 为主，并保留 mAP50 和 mAP75。成本指标包括实际请求 KB/图、响应 KB/图、上传率、Original/Q90/ROI 比例、均摊 GFLOPs、排序器参数量和排序器时延。Online 同时报告名义与实际上传率；Fixed-byte 同时报告实际字节和接纳样本比例。

端到端（E2E）时延字段由端侧推理、编码、请求字节、给定带宽和云端推理的冻结时延模型估算得到。本文所有 E2E 数值均为估算值（表中记作"估算 E2E@10Mbps"），不是真实链路测量，本文不据此声称实时性能；真实设备测量作为局限在 7.1 节讨论。

### 5.5 系统参考

表 3 给出三数据集的系统参考。三个数据域的端云精度差和图像字节分布显著不同：Oregon 的 Original 请求成本远高于另两个数据集，而 Q90 损失也更明显。这说明成本必须基于实际载荷字节计算，不能仅比较上传比例。

**表 3** 系统参考（Official AP 与请求 KB/图）

| 数据集 | 方法 | Official AP | 请求 KB/图 |
|---|---|---:|---:|
| COCO | Edge-only | 0.4079 | 0.00 |
| COCO | Cloud-only Original | 0.5742 | 159.12 |
| COCO | Always Q90 | 0.5699 | 84.08 |
| COCO | Always ROI fallback | 0.5594 | 71.16 |
| COCO | 共享动作头全上传 | 0.5732 | 134.83 |
| ENA24 | Edge-only | 0.7127 | 0.00 |
| ENA24 | Cloud-only Original | 0.7418 | 433.29 |
| ENA24 | Always Q90 | 0.7369 | 94.34 |
| ENA24 | Always ROI fallback | 0.7373 | 89.91 |
| ENA24 | 共享动作头全上传 | 0.7459 | 343.68 |
| Oregon | Edge-only | 0.4548 | 0.00 |
| Oregon | Cloud-only Original | 0.4991 | 1595.49 |
| Oregon | Always Q90 | 0.4711 | 97.98 |
| Oregon | Always ROI fallback | 0.4599 | 96.03 |
| Oregon | 共享动作头全上传 | 0.4796 | 1037.00 |

## 6 结果与分析

除特别说明外，本节所有终评结果均为探索性口径（见 5.1 节与 7.2 节的数据角色边界）；严格确认性结论需要新的隐藏测试集。精度差值以 10⁻³ 为单位报告，与容忍量 \(\epsilon=0.001\) 的尺度一致；主表精度保留 4 位小数。

先给出总括：30 个冻结工作点的终评精度变化全部处于 -0.83×10⁻³ 至 +0.10×10⁻³ 之间，没有任何一点越过 -1×10⁻³ 的预注册容忍下界。本节其余部分回答三个问题：这一"零容忍违反"记录付出了多少回退代价（6.1-6.4）、换来了哪些节省（6.1、6.3）、以及样本排序与载荷策略的收益边界在哪里（6.5-6.7）。

### 6.1 门禁与冻结结果总览

30 个工作点中，门禁通过 13 个、回退 17 个。通过点中 6 个在门禁集上与参考动作逐图完全相同（其中 5 个的临时候选与参考本身即为同一配置），真正比较了不同动作并通过的只有 7 个。冻结后共有 22 个 SharedAction 和 8 个 GBDT-Hierarchical；CostEligible-AllROI 未被最终冻结。表 4 将"同动作通过"与"非同动作通过"分开统计，避免把平凡零差值误报为成本优化成功；图 2 给出全部冻结工作点的精度-成本位置，图 3 按数据集展示门禁通过与回退的构成。

**表 4** 门禁与冻结结果总览。平均/最大 regret 为冻结配置相对同候选集合终评 oracle 的 Official AP 差距（诊断性指标）

| 数据集 | 工作点 | 非同动作通过 | 同动作通过 | 回退 | 冻结 GBDT/Shared | 平均 regret | 最大 regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | 10 | 0 | 0 | 10 | 0 / 10 | 0.0038 | 0.0113 |
| ENA24 | 10 | 3 | 5 | 2 | 4 / 6 | 0.0088 | 0.0132 |
| Oregon | 10 | 4 | 1 | 5 | 4 / 6 | 0.0040 | 0.0171 |
| 合计 | 30 | 7 | 6 | 17 | 8 / 22 | — | — |

相对各工作点的 SharedAction 参考配置，COCO 因全部回退而精度和请求 KB 差均为零。ENA24 的 10 点平均终评精度差为 -0.06×10⁻³，最差为 -0.39×10⁻³；4 个在终评上出现动作变化的通过点合计节省 4.96 KB/图（相对各自参考请求量为 1.5%-1.8%），均摊到 10 个工作点为 0.50 KB/图。Oregon 的 10 点平均精度差为 -0.24×10⁻³，最差为 -0.83×10⁻³；唯一明确的 Fixed-call 成本变化为节省 0.29 KB/图，Fixed-byte 通过点受相同容量上限约束，不代表额外通信节省。

**[图 2：三数据集冻结配置的 Official AP 与请求 KB/图工作点。各面板独立缩放；同一预算模式的工作点按预算升序连线，形状区分模式。]**

**[图 3：三数据集风险门禁结果。通过点进一步区分非同动作比较与门禁集上动作完全相同的平凡通过。]**

### 6.2 COCO 冻结结果

COCO 的 10 个临时候选全部未通过基于 630 张独立门禁集图像的检验，因此冻结结果全部为各工作点最佳 SharedAction 参考配置（表 5）。Fixed-call/Online 的 GBDT 候选在选择集上只损失约 0.07×10⁻³ 至 0.25×10⁻³ 的精度并节省 1.22-2.09 KB/图，但门禁集精度差区间下界达到 -4.2×10⁻³ 至 -3.3×10⁻³，低于 -1×10⁻³ 门槛。

**表 5** COCO 冻结终评结果（10 个工作点全部回退至参考配置）。E2E 为估算值（见 5.4 节）

| 模式 | 预算 | 冻结排序器 + 策略 | Official AP | KB/图 | 估算E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | SemanticRank + SharedAction | 0.4438 | 16.29 | 38 ms | 0.00% | 回退 |
| Fixed-call | 25% | SemanticRank + SharedAction | 0.4836 | 39.95 | 68 ms | 0.00% | 回退 |
| Fixed-call | 40% | Controlled-32D + SharedAction | 0.5147 | 60.93 | 95 ms | 0.08% | 回退 |
| Online | 10% | SemanticRank + SharedAction | 0.4570 | 23.96 | 48 ms | 0.00% | 回退 |
| Online | 25% | SemanticRank + SharedAction | 0.5016 | 54.08 | 86 ms | 0.04% | 回退 |
| Online | 40% | Controlled-32D + SharedAction | 0.5214 | 67.41 | 104 ms | 0.08% | 回退 |
| Fixed-byte | 5% | SemanticRank + SharedAction | 0.4293 | 7.95 | 34 ms | 1.70% | 回退 |
| Fixed-byte | 10% | Confidence + SharedAction | 0.4419 | 15.90 | 48 ms | 2.68% | 回退 |
| Fixed-byte | 20% | MORIC-style + SharedAction | 0.4676 | 31.82 | 73 ms | 3.66% | 回退 |
| Fixed-byte | 40% | Controlled-32D + SharedAction | 0.5203 | 63.65 | 113 ms | 3.72% | 回退 |

门禁也产生了保守的误拒（false negative）。COCO Fixed-byte 被拒绝的 GBDT 候选在终评上相对参考配置实际提高 4.8×10⁻³ 至 9.9×10⁻³ 的精度，但协议禁止根据终评恢复候选。COCO 的平均终评 oracle regret 为 0.0038，最大为 0.0113。这说明门禁确实阻止了门禁集上证据不足的配置，同时也牺牲了部分终评上的潜在收益；其构成在 6.7 节进一步分析。

### 6.3 ENA24 冻结结果

ENA24 使用 200 张选择集和 200 张近重复组隔离的门禁集，是三个数据域中统计功效最弱的设置（见 6.7 节），其通过点应理解为低功效条件下的探索性结果。8 个工作点通过，其中 5 个在门禁集上动作相同；GBDT-Hierarchical 在 10%/25% 的 Fixed-call 与 Online 被保留，40% 工作点回退（表 6）。

**表 6** ENA24 冻结终评结果。门禁列：通过 = 非同动作比较通过；通过（同动作）= 门禁集上动作逐图相同的平凡通过；回退 = 检验失败回退参考配置。E2E 为估算值

| 模式 | 预算 | 冻结排序器 + 策略 | Official AP | KB/图 | 估算E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | Confidence + GBDT | 0.7100 | 40.22 | 58 ms | 0.17% | 通过（同动作）† |
| Fixed-call | 25% | ClassGainPrior + GBDT | 0.7265 | 91.75 | 110 ms | 0.51% | 通过 |
| Fixed-call | 40% | SemanticRank + SharedAction | 0.7277 | 163.79 | 179 ms | 0.00% | 回退 |
| Online | 10% | Confidence + GBDT | 0.7132 | 60.76 | 78 ms | 0.23% | 通过 |
| Online | 25% | ClassGainPrior + GBDT | 0.7265 | 96.57 | 115 ms | 0.57% | 通过 |
| Online | 40% | SemanticRank + SharedAction | 0.7272 | 161.09 | 176 ms | 0.00% | 回退 |
| Fixed-byte | 5% | Confidence + SharedAction | 0.7165 | 21.66 | 41 ms | 0.00% | 通过（同动作） |
| Fixed-byte | 10% | ClassGainPrior + SharedAction | 0.7271 | 43.31 | 76 ms | 0.00% | 通过（同动作） |
| Fixed-byte | 20% | M2E + SharedAction | 0.7356 | 86.65 | 127 ms | 0.00% | 通过（同动作） |
| Fixed-byte | 40% | Confidence + SharedAction | 0.7386 | 173.31 | 210 ms | 0.00% | 通过（同动作） |

†该点冻结配置（Confidence + GBDT）与参考配置不同，但二者在门禁集上动作逐图相同，以精确零差值通过；终评上出现 0.17% 的 ROI 动作。因此该点只能表述为"门禁集未观察到动作差异且终评未明显退化"，不能表述为门禁已验证 ROI 安全。4 个 Fixed-byte 同动作点的临时候选与参考为同一配置，门禁差值恒为零。

4 个在终评上出现动作变化的通过点相对参考配置的精度/KB 变化依次为：Fixed-call 10% 的 +0.10×10⁻³ / -0.74 KB（-1.8%），Fixed-call 25% 的 -0.37×10⁻³ / -1.54 KB（-1.7%），Online 10% 的 +0.09×10⁻³ / -0.94 KB（-1.5%），Online 25% 的 -0.39×10⁻³ / -1.73 KB（-1.8%）。精度变化均未越过预设的 -1×10⁻³ 容忍下界（正向变化不受约束），但 ROI 比例仅 0.17%-0.57%，通信节省的绝对量与相对量都较小。节省集中在紧约束的低预算区间，与选择性卸载仅在资源受限工作区间产生净收益的既有观察一致 [21,31]。

ENA24 的平均终评 oracle regret 为 0.0088，三个数据集中最高，主要反映 200/200 的小样本选择-门禁划分在配置排序与统计功效两方面都不足（定量分析见 6.7 节）。

### 6.4 Oregon 冻结结果

Oregon 有 5 个门禁通过点，其中 1 个为同动作通过。唯一明确的 Fixed-call 成本优化出现在 10%；25%/40% 的 Fixed-call 和全部 Online 候选因门禁集精度区间不足而回退（表 7）。

**表 7** Oregon 冻结终评结果。门禁列含义同表 6。E2E 为估算值

| 模式 | 预算 | 冻结排序器 + 策略 | Official AP | KB/图 | 估算E2E@10Mbps | ROI | 门禁 |
|---|---:|---|---:|---:|---:|---:|---|
| Fixed-call | 10% | GEC-style + GBDT | 0.4651 | 95.63 | 102 ms | 0.04% | 通过 |
| Fixed-call | 25% | Confidence + SharedAction | 0.4687 | 163.11 | 167 ms | 0.00% | 回退 |
| Fixed-call | 40% | Confidence + SharedAction | 0.4842 | 396.85 | 365 ms | 0.00% | 回退 |
| Online | 10% | ClassGainPrior + SharedAction | 0.4516 | 249.02 | 229 ms | 0.00% | 回退 |
| Online | 25% | Confidence + SharedAction | 0.4697 | 181.17 | 183 ms | 0.00% | 回退 |
| Online | 40% | Confidence + SharedAction | 0.4854 | 437.01 | 399 ms | 0.00% | 回退 |
| Fixed-byte | 5% | Confidence + GBDT | 0.4724 | 79.77 | 114 ms | 0.06% | 通过 |
| Fixed-byte | 10% | Confidence + GBDT | 0.4756 | 159.55 | 184 ms | 0.06% | 通过 |
| Fixed-byte | 20% | Confidence + GBDT | 0.4844 | 319.10 | 316 ms | 0.06% | 通过 |
| Fixed-byte | 40% | Confidence + SharedAction | 0.4927 | 638.20 | 580 ms | 0.00% | 通过（同动作） |

Fixed-call 10% 相对参考配置的终评精度差为 -0.10×10⁻³、请求量差为 -0.29 KB/图，收益量级很小。Fixed-byte 5%/10%/20% 的通过点相对参考配置的终评精度分别下降 0.68×10⁻³、0.83×10⁻³ 和 0.78×10⁻³，仍处于容忍量内；由于其请求量由容量上限约束，这三点不能解释为额外字节节省。Oregon 的平均终评 oracle regret 为 0.0040，最大值 0.0171 出现在 Online 10%，说明选择集选择的 ClassGainPrior 未迁移为终评最优的 GEC-style。

### 6.5 排序器的跨域差异：冻结选择、oracle 分布与随机波动带

**发现 1（无跨域统一最优排序器）。** 冻结配置呈现明显的跨域差异：COCO 的 10 点中 SemanticRank 占 5 点、Controlled-32D 占 3 点；ENA24 主要选择 Confidence（4 点）与 ClassGainPrior（3 点）；Oregon 有 8 点冻结为 Confidence，另有 GEC-style 和 ClassGainPrior 各 1 点。该差异不是载荷策略单独造成的，因为参考配置本身即在 SharedAction 内按选择集精度选择排序器。

由于冻结选择可能受选择集噪声影响（Oregon Online 10% 的 0.0171 regret 正说明选择集最优未迁移到终评），本文用终评 oracle 的最优配置分布做交叉诊断：COCO 的 10 个工作点上，oracle 最优排序器为 SemanticRank（5 点）与 Controlled-32D（5 点）；ENA24 为 ClassGainPrior（5 点）、Controlled-32D（3 点）、M2E 与 MORIC-style（各 1 点）；Oregon 为 GEC-style（4 点）、Confidence（3 点）与 MORIC-style（3 点）。冻结选择与 oracle 分布在方向上一致：三个数据域各自偏好不同的排序器族，oracle 层面同样不存在跨域统一最优者。oracle 最优载荷策略同样分化——ENA24 有 9/10 点为 CostEligible-AllROI，说明该域可得的成本收益主要来自简单成本可行性门禁而非 GBDT 排序；COCO 为 5 点 GBDT 与 5 点 SharedAction；Oregon 以 SharedAction 为主（7 点）。以配置一致性口径审计选择质量：30 个工作点中 6 个冻结配置恰为同候选集合的终评 oracle 最优（regret 为零），7 个 regret 不超过 \(\epsilon\)，14 个不超过 5×10⁻³——当前控制器远非低 regret 自动配置器，这一差距在 6.7 节主要归因于门禁功效与选择集规模。需要强调，oracle 基于终评单次评测，仍受评测噪声影响，只能作为诊断性证据；"单一配置跨数据集部署的量化代价"这一确认性问题须由尚未执行的统一配置消融回答（见 5.3 节）。

**发现 2（样本排序收益的随机波动带边界）。** Random 波动带给出另一条边界。以 SharedAction 载荷、50 个随机种子在 Fixed-call 三个预算上评测（数值来自与主矩阵同源的冻结结果）：COCO 的冻结配置在 10%/25%/40% 分别高出随机带上界 15.2×10⁻³、31.0×10⁻³ 与 35.7×10⁻³（带分别为 [0.4220,0.4287]、[0.4451,0.4526]、[0.4704,0.4790]）；ENA24 的三个冻结点全部落在随机带内（0.7100 低于带均值 0.7158；0.7265 与带上界 0.7271 相当；0.7277 在 [0.7165,0.7332] 内）；Oregon 的 10%/25% 在带内、40% 高出带上界 3.9×10⁻³。随机带为 SharedAction 口径，而 ENA24 低预算冻结点为 GBDT 策略，但动作层对终评精度的影响不超过 0.4×10⁻³，不改变带级结论。

因此，本文的证据支持"根据数据域和预算工作点进行校准选择"，且给出了更细的边界：学习型样本排序的收益只在 COCO 一致地超出随机波动；在两个野生动物域，样本排序收益与随机波动同量级，端云精度差主要由上传比例本身兑现。SemanticRank、M2E、ClassGainPrior 和 Confidence 应被视为可插拔候选，而非论文的算法贡献；在当前选择集规模与门禁设置下，校准选择尚未构成低 regret 的控制器。

### 6.6 载荷动作与预算语义

**[图 4：冻结配置的 Edge、Original、Q90 和 ROI 动作组成。比例按全部输入图像计算，并标注实际上传率与 ROI 比例。]**

图 4 显示，冻结结果中的 ROI 不是主导载荷。ENA24 和 Oregon 的 ROI 比例均低于 0.6%；COCO 的 Fixed-call/Online 低于 0.1%，仅 Fixed-byte 的 SharedAction 参考达到 1.70%-3.72%。因此不能声称 GBDT-Hierarchical 或 CostEligible-AllROI 这类 ROI 倾向策略在三个数据集上普遍降低成本。当前更稳定的动作仍是 Original/Q90 混合，ROI 只在少数样本上触发。

预算语义改变了结果解释。Fixed-call 中更小载荷会直接体现为 KB/图下降；Online 还必须同时观察实际上传率漂移；Fixed-byte 中字节容量近似固定，动作压缩的作用是改变可接纳样本和精度，而不是在同一容量下继续声称节省字节。Oregon Fixed-byte 的三个通过点正说明"精度容忍内保留"不等于"通信成本进一步下降"。

### 6.7 门禁功效、拦截与误拒的机会成本

风险门禁是本文最重要的受控组成分析，其效果需要从拦截与误拒两个方向同时评估，并且必须先回答一个统计问题：在 200 次配对 bootstrap 与 \(\epsilon=0.001\) 的设定下，各门禁集实际能检出多大的精度差？

**发现 3（保守性主要为统计功效不足所致）。** 表 8 按数据集与动作差异范围汇总非同动作工作点的精度差区间半宽。候选通过精度门禁要求区间下界不低于 \(-\epsilon\)，即区间中点不低于半宽减 \(\epsilon\)。当候选与参考的动作差异集中于极少数图像时（如 Oregon Fixed-call 10% 与 Fixed-byte 5%/10%/20%），半宽仅 0.04×10⁻³ 至 0.06×10⁻³，门禁具备充分分辨率；而当动作差异广泛时，半宽达到 \(\epsilon\) 的 1.6 至 30 倍——此时任何真实但中等幅度的动作收益（如 ±0.5×10⁻³）在统计上都无法与退化区分，门禁结构上只能放行动作差异极小或收益异常显著的候选。因此 17 个回退中至少 COCO Fixed-byte 与 ENA24/Oregon 40% 的部分回退应解读为"证据不足"而非"候选有害"；"保守性"部分是设计内生的，而非纯粹的域现象。三个门禁集规模相差约 50 倍（200/630/9,923），跨数据集的通过率因此不可直接比较。

**表 8** 门禁功效汇总：非同动作工作点的精度差区间半宽（200 次配对 bootstrap，2.5%/97.5% 分位）

| 数据集 | 工作点组 | 点数 | 半宽范围（×10⁻³） | 半宽/\(\epsilon\) |
|---|---|---:|---|---|
| COCO | Fixed-call 与 Online | 6 | 1.6-2.1 | 1.6-2.1 |
| COCO | Fixed-byte | 4 | 7.8-11.3 | 7.8-11.3 |
| ENA24 | 非同动作 5 点 | 5 | 0.5-29.5 | 0.5-29.5 |
| Oregon | 动作差异广泛（Fixed-call 25%/40% 与 Online） | 5 | 11.3 | 约 11 |
| Oregon | 动作差异极少（Fixed-call 10% 与 Fixed-byte 5%/10%/20%） | 4 | 0.04-0.06 | <0.1 |

在拦截方向，门禁的价值可以直接量化：若去掉门禁，选择集将直接部署临时候选；COCO 的 6 个 Fixed-call/Online 候选虽在选择集上满足精度容忍和成本目标（仅损失 0.07×10⁻³ 至 0.25×10⁻³、节省 1.22-2.09 KB/图），却在门禁集上出现 -4.2×10⁻³ 至 -3.3×10⁻³ 的精度区间下界。门禁使这些配置回退，避免把选择集后验乐观当作终评结论。

在误拒方向，机会成本同样应如实报告。COCO Fixed-byte 被拒的 GBDT 候选在终评上相对参考提高 4.8×10⁻³ 至 9.9×10⁻³；ENA24 40% 被拒的 ClassGainPrior + CostEligible-AllROI 候选在终评上同时提高精度并大幅降低请求量（Fixed-call 40%：+8.2×10⁻³ 且 -44.77 KB/图，相对参考 -27%；Online 40%：+10.5×10⁻³ 且 -28.08 KB/图，相对参考 -17%），但 200 张门禁集上的区间过宽（半宽约 29×10⁻³），按预注册协议仍保留回退。这些误拒与表 8 的功效分析一致：损失最大的机会恰好出现在功效最弱的工作点组。13 个通过点中只有 7 个进行了非同动作比较，仅报告"43.3% 通过率"会夸大有效优化；终评 oracle 平均 regret 为 0.0038-0.0088、最大 0.0171，也不支持"控制器接近 oracle"的表述。

本轮完成的诊断只能回答参考配置、临时候选、门禁回退和终评 oracle 的作用。尚未执行的全数据集统一配置消融、不同 bootstrap 次数、聚类 bootstrap 和新的隐藏终评集属于后续工作，不作为本文已完成消融。当前结果将控制器定位为可审计且保守的第一版部署安全机制：它可靠地阻止了后验乐观配置上线，但以当前门禁集规模，其误拒代价在低功效工作点上可观。

## 7 系统验证与局限

### 7.1 原型可实现性与真实系统测量缺口

本文实现的原型系统包含 JPEG/HTTP 传输、有界异步队列和轻量线性决策器。历史组件实验显示，选择器计算相对端侧检测和 JPEG 编码较小，主要系统开销来自端侧推理、编码、网络和云服务。该原型说明在线执行路径可以实现，但其检测器、图像编码和网络条件并非本文冻结三数据集控制器的完整同协议部署。

本文没有把历史组件延迟直接相加为当前端到端时延，也没有把公式估算标注为实测。面向真实部署的结论仍需在明确的端侧设备与服务器上测量编码、上行传输、排队、云端推理、下行响应、吞吐和能耗，并覆盖 1/5/10/20 Mbps、不同往返时延与抖动条件；这是本文与实时部署主张之间尚未跨越的证据缺口。

### 7.2 有效性威胁

第一，终评集的确认性效力有限。COCO val2017 与 ENA24 终评为固定诊断口径；Oregon 终评已被上游实验访问，本轮属于事后探索性复用。上游设计选择（排序器结构、动作模型超参、特征工程）对终评的隐性适配无法完全排除，因此第 6 节全部结果为探索性口径，严格确认性结论需要新的隐藏测试集或预注册外部数据。

第二，数据划分边界受元数据限制。ENA24 只能实现近重复组隔离，缺少事件、相机和位置元数据；Oregon 采用随机图像级主协议，同一相机背景可能跨数据角色出现，其图像级 bootstrap 也忽略事件簇相关性。因此两个野生动物域的结果不能解释为跨相机或跨地点泛化。

第三，门禁的统计限制。200 次 bootstrap 只产生逐工作点经验区间，未做 30 个工作点的同时性误差控制；6.7 节显示区间半宽在动作差异广泛的工作点达到 \(\epsilon\) 的 1.6-30 倍，门禁的保守性部分是功效不足的直接后果；三个门禁集规模相差约 50 倍，跨数据集通过率不可比。同动作快捷通过也不能替代在门禁集上观察到 ROI 差异后的安全验证。

第四，排序学习器使用可加代理目标训练，而 Official AP 不是逐图可加指标；观察 1 只提供代理层面的动机。终评 oracle 是同一 24 候选集合内的事后诊断，不是不可实现的全局最优，也不得用于修复主结果。

第五，ROI 依赖端侧候选框，无法修复完全漏检；当前动作空间未覆盖 tile packing、任务感知编码和特征压缩。冻结结果中 ROI 比例低于 4%，不能证明 ROI 具有跨域普适价值。

第六，全部端到端时延为冻结时延模型的估算值，未经真实链路验证；本文不声称实时性能优势。

第七，端侧和云端使用同系列检测器，错误可能高度相关；更异构或专用的云端检测器可能改变可路由收益及最佳载荷策略。

第八，6.5 节的随机波动带表明，在两个野生动物域，样本排序收益与随机波动同量级；结合 ENA24 选择集/门禁集各仅 200 张的规模，该域的通过点只能作为低功效条件下的探索性证据。

## 8 结论

本文围绕"在无法保证学习型排序跨域有效的前提下，如何以可审计的方式采纳成本更低的卸载配置"这一问题，提出了面向端云目标检测的风险受控配置采纳协议：样本排序、Original/Q90/ROI 载荷动作与 Fixed-call/Online/Fixed-byte 预算调度相互解耦，离线控制器在统一检测缓存上以配置级 Official AP 与实际请求字节完成选择，独立配对 bootstrap 门禁决定冻结候选或回退参考配置，终评集不参与选择或修复。30 个冻结工作点、13 通过/17 回退、终评精度变化全部处于 -0.83×10⁻³ 至 +0.10×10⁻³ 之间（零容忍违反）、ENA24 低预算节省 1.5%-1.8% 请求字节——这组数字概括了该协议的行为：保守、可审计、从不越过预注册容忍。

三数据集 30 个冻结工作点的探索性结果给出三条有证据支撑的结论：其一，冻结门禁能够可靠拦截选择集后验乐观配置——COCO 全部 6 个此类候选被回退，代价是可量化的误拒机会成本；其二，解耦架构使预算语义造成的错误归因可以被识别——Fixed-byte 通过点不构成额外字节节省这一区分正是解耦报告的直接产物；其三，冻结选择与终评 oracle 诊断一致显示不存在跨域统一最优的排序器或载荷策略，且学习型样本排序的收益只在 COCO 一致超出随机波动带。相应地，两项主张不在本文证据范围内：解耦架构相对端到端联合方法的优势未经对比实验验证；单一配置跨数据集部署的量化代价有待尚未执行的统一配置消融回答。

当前控制器应理解为可审计且保守的第一版部署安全机制，而非低 regret 的自动配置器。扩大独立门禁集并按 6.7 节功效分析匹配容忍量、引入事件/相机聚类感知的不确定性估计、执行统一配置消融，以及在真实端云链路上完成端到端测量，是把该框架推向部署结论的必要后续工作。

## Statements and Declarations

### Funding

[作者确认：基金名称、项目编号；若无基金，请写明 "The authors declare that no funds, grants, or other support were received during the preparation of this manuscript."]

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

[作者确认：按 CRediT 角色填写 Conceptualization、Methodology、Software、Validation、Writing 等贡献。]

### Data availability

COCO 可从官方渠道公开获取 [28]。ENA24 与 Oregon Critters 为野生动物相机图像数据集，本文使用其冻结划分与预测缓存；二者的正式出处、许可证与访问方式由作者在投稿前确认并补充。[作者确认：ENA24 与 Oregon Critters 的正式引文、许可证类型与下载地址；若数据不可完全公开，说明合理获取途径。] 本文生成的数据角色清单、冻结配置、来源哈希和汇总结果将随论文或补充材料发布；在数据许可确认前，不公开原始图像或位置敏感元数据。

### Code availability

[作者确认：匿名代码仓库地址、归档 DOI 和公开时间。] 复现材料包含协议文件、缓存清单、配置选择器、风险门禁、汇总脚本和环境信息。

### Ethics and consent

本文不进行个体身份识别。涉及人物影像、私有相机部署或敏感野生动物位置的数据，在公开前须满足相应知情同意、匿名化、机构伦理和数据许可要求。[作者确认：按数据提供方与机构要求补充正式伦理声明。]

## 附录 A 观察 1 的形式化陈述与证明

**陈述。** 设图像集合 \(\{1,\dots,N\}\)，每图净价值 \(v_i=g_i-\lambda c_i\)，其中 \(g_i\ge 0\) 为最优云动作相对端侧的期望效用增量，\(c_i>0\) 为对应载荷字节，\(\lambda\ge 0\) 为字节折价。Fixed-call 预算下的可行解为满足 \(|\mathcal U|=K\) 的子集，目标为 \(\max_{\mathcal U}\sum_{i\in\mathcal U}v_i\)。设端侧不确定性分数 \(h_1,\dots,h_N\) 互异（若有并列，以任意预先固定的规则打破），置信度排序解 \(\mathcal U_{conf}(K)\) 为 \(h\) 值最大的 \(K\) 个样本。

**A.1（充分性）。** 若 (i) 存在单调不减函数 \(g(\cdot)\) 使 \(g_i=g(h_i)\)，且 (ii) \(c_i\equiv c\) 为常数，则对任意 \(K\) 与任意 \(\lambda\ge 0\)，\(\mathcal U_{conf}(K)\) 是最优解。

*证明。* 在条件 (ii) 下，任何可行解的成本项都等于 \(\lambda cK\)，与选取无关，故目标等价于最大化 \(\sum_{i\in\mathcal U}g(h_i)\)。设 \(\mathcal U\) 为任意可行解且 \(\mathcal U\ne\mathcal U_{conf}(K)\)，则存在 \(i\in\mathcal U_{conf}(K)\setminus\mathcal U\) 与 \(j\in\mathcal U\setminus\mathcal U_{conf}(K)\)。由 \(\mathcal U_{conf}(K)\) 的定义与 \(h\) 值互异，\(h_i>h_j\)，于是由单调性 \(g(h_i)\ge g(h_j)\)。将 \(j\) 换为 \(i\) 得到可行解 \(\mathcal U'=\mathcal U\cup\{i\}\setminus\{j\}\)，目标变化 \(g(h_i)-g(h_j)\ge 0\)。有限次交换后可将 \(\mathcal U\) 变换为 \(\mathcal U_{conf}(K)\) 且目标不降，故 \(\mathcal U_{conf}(K)\) 最优。∎

**A.2（失效条件）。** 若存在样本对 \(i,j\) 满足 \(h_i>h_j\) 且 \(v_i<v_j\)，则存在预算 \(K\) 使 \(\mathcal U_{conf}(K)\) 严格次优。

*证明。* 取 \(K=|\{l:h_l\ge h_i\}|\)。由 \(h\) 值互异，\(\mathcal U_{conf}(K)\) 恰好包含 \(h\) 值不低于 \(h_i\) 的全部样本，故 \(i\in\mathcal U_{conf}(K)\) 而 \(j\notin\mathcal U_{conf}(K)\)。构造 \(\mathcal U'=\mathcal U_{conf}(K)\cup\{j\}\setminus\{i\}\)，其目标比 \(\mathcal U_{conf}(K)\) 高 \(v_j-v_i>0\)，故 \(\mathcal U_{conf}(K)\) 非最优。∎

**说明。** 当 \(\lambda=0\) 且成本同质时，A.1 退化为经典的"按收益单调信号取前 K"；当成本异质（例如 Original/Q90/ROI 字节差异显著）或收益非单调于 \(h\)（例如类别与尺寸效应）时，A.2 的前提在实际检测数据中容易成立。该观察只约束逐图可加的代理目标；配置级 Official AP 的非可加性是第 4.4 节采用配置级评估器的直接原因。

## 附录 B 术语对照表

**表 B1** 中英术语对照

| 中文术语 | 英文/记号 | 说明 |
|---|---|---|
| 拟合集 | fit | 训练排序器与动作模型 |
| 选择集 | selector | 配置选择与阈值校准 |
| 门禁集 | gate（实验记录中亦称 audit） | 仅用于风险检验 |
| 终评集 | final | 冻结后一次性评测 |
| 排序器 | Ranker | 样本上传优先级评分器 |
| 载荷动作 | payload action | Edge 保留 / Original / Q90 / ROI |
| 载荷策略 | payload policy | 为已接纳样本选择载荷动作的策略 |
| 共享动作头 | SharedAction | 参考载荷策略 |
| 成本可行 ROI 门控 | CostEligible-AllROI | 简单成本门禁策略 |
| 分层 GBDT 动作头 | GBDT-Hierarchical | 学习型载荷策略 |
| 固定调用比例 | Fixed-call | 上传样本数固定的预算模式 |
| 在线阈值 | Online | 冻结阈值逐样本接纳的预算模式 |
| 固定字节容量 | Fixed-byte | 总请求字节受限的预算模式 |
| 参考配置 | reference | 各工作点选择集精度最高的 SharedAction 配置 |
| 临时候选配置 | provisional | 控制器在选择集上选出的待检验配置 |
| 冻结配置 | frozen | 通过门禁或回退后固定的部署配置 |
| 误拒 | false negative | 门禁拒绝了终评上实际更优的候选 |
| oracle 差距 | oracle regret | 冻结配置与同候选集合终评最优的精度差（诊断） |
| 近重复组隔离 | near-duplicate-group disjoint | 近重复图像成组划分、组不跨侧 |
| 事后探索性复用 | post-hoc exploratory reuse | 终评集曾被上游实验访问后的再次使用 |
| 官方评测精度 | Official AP | conf=0.001 的 COCO-style mAP50-95 |

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

[11] Cao, Z., Li, Z., Chen, Y., Pan, H., Hu, Y., Liu, J.: Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. IEEE ICDCS, 259-270 (2023). https://doi.org/10.1109/ICDCS57875.2023.00062

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

[22] Choi, H., Bajic, I.V.: Deep Feature Compression for Collaborative Object Detection. IEEE ICIP, 3743-3747 (2018). Preprint: https://arxiv.org/abs/1802.03931

[23] Elgamal, T., Shi, S., Gupta, V., Jana, R., Nahrstedt, K.: SiEVE: Semantically Encoded Video Analytics on Edge and Cloud. ICDCS (2020). https://doi.org/10.1109/ICDCS47774.2020.00182

[24] Jitkrittum, W., Gupta, N., Menon, A.K., Narasimhan, H., Rawat, A.S., Kumar, S.: When Does Confidence-Based Cascade Deferral Suffice? Advances in Neural Information Processing Systems 36, 9891-9906 (2023).

[25] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of Modern Neural Networks. ICML, PMLR 70, 1321-1330 (2017). https://proceedings.mlr.press/v70/guo17a.html

[26] Angelopoulos, A.N., Bates, S., Candes, E.J., Jordan, M.I., Lei, L.: Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. The Annals of Applied Statistics 19(2) (2025). https://doi.org/10.1214/24-AOAS1998

[27] Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman & Hall/CRC (1993).

[28] Lin, T.-Y., Maire, M., Belongie, S., et al.: Microsoft COCO: Common Objects in Context. ECCV, 740-755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

[29] Jocher, G., Qiu, J., Liu, M., et al.: Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748

[30] Jiang, J., Ananthanarayanan, G., Bodik, P., Sen, S., Stoica, I.: Chameleon: Scalable Adaptation of Video Analytics. ACM SIGCOMM, 253-266 (2018). https://doi.org/10.1145/3230543.3230574

[31] Matsubara, Y., Callegaro, D., Baidya, S., Levorato, M., Singh, S.: Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks. ICPR (2021). Preprint: https://arxiv.org/abs/2007.15818

[32] Chen, L., Zaharia, M., Zou, J.: FrugalML: How to Use ML Prediction APIs More Accurately and Cheaply. Advances in Neural Information Processing Systems 33 (2020). Preprint: https://arxiv.org/abs/2006.07512
