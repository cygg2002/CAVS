# CAVS：面向成本感知端云协同目标检测的上下文图像—动作价值调度

**CAVS: Contextual Image–Action Value Scheduling for Cost-Aware Edge–Cloud Object Detection**

[作者姓名]¹，[作者姓名]²，[通讯作者]¹*

¹[单位、城市、邮编、国家]

²[单位、城市、邮编、国家]

*[通讯作者邮箱]

## 摘要

端云协同目标检测常将资源分配简化为“是否上传图像”，但真实系统还需在原始图像、压缩 JPEG 和感兴趣区域（region of interest，ROI）等异构载荷之间选择。本文提出上下文图像—动作价值调度（Contextual Image–Action Value Scheduling，CAVS），将决策单元由图像扩展为图像—载荷对。CAVS 以端侧检测集合的置换不变表示预测动作条件价值，训练奖励与 IoU 阈值 0.50–0.95 上平均的 COCO-style AP（AP50–95）对齐；实际编码字节不进入价值网络，而由全信息离线调度或零前视在线控制处理。在冻结的事后评测协议下，以 Always-Original JPEG payload 的 10%、25% 和 40% 为预算，CAVS 在 COCO、Oregon 野生动物和 PASCAL VOC 上的九个点估计均优于图像级路由。相较协议匹配的上下文图像级奖励基线，COCO 和 VOC 分别提高 0.0280–0.0360 和 0.0205–0.0271 AP；COCO 同价值输出的受控结构消融显示，联合图像—动作容量调度较“两阶段每图动作投影—容量选择”提高 0.0228–0.0356 AP。在 RK3588 端云零前视测试床上，CAVS 在三档预算下相较各档最强外部基线分别提高 0.0163、0.0369 和 0.0372 AP，所有前缀 payload-byte 约束均满足，平均端到端时延为 162.7–238.5 ms。结果表明，动作条件价值学习与逐样本字节计量能够改善精度—通信权衡，同时揭示了全信息调度与因果在线执行之间的部署差距。

**关键词：** 端云协同；目标检测；选择性卸载；资源分配；边缘计算

## Abstract

Selective edge–cloud object detection is often reduced to deciding whether an image should be offloaded. A deployable system must additionally choose among heterogeneous payloads, such as the original image, compressed JPEG, and a region of interest. We propose Contextual Image–Action Value Scheduling (CAVS), which treats each image–payload pair as a decision unit. A permutation-invariant encoder maps edge detections to action-conditioned values trained with a reward aligned to COCO-style average precision over intersection-over-union thresholds from 0.50 to 0.95 (AP50–95). Explicit encoded-byte costs are excluded from the value network and handled by either a full-information feasible scheduler or a zero-lookahead online controller. Under frozen post-hoc evaluation protocols and budgets of 10%, 25%, and 40% of the Always-Original JPEG payload, CAVS yields the highest point estimate in all nine dataset–budget settings on COCO, Oregon wildlife, and PASCAL VOC. It improves over the protocol-matched context-reward image-level baseline by 0.0280–0.0360 AP on COCO and 0.0205–0.0271 AP on VOC. A controlled COCO ablation with identical action-value outputs indicates a 0.0228–0.0356 AP gain for joint image–action budget scheduling over two-stage per-image action projection followed by budgeted selection. On an RK3588 edge–cloud zero-lookahead testbed, CAVS exceeds the strongest evaluated external baseline by 0.0163, 0.0369, and 0.0372 AP while satisfying every prefix payload-byte constraint; measured mean end-to-end latency is 162.7–238.5 ms. These results support retaining image–action alternatives until constrained scheduling while quantifying the remaining gap between full-information scheduling and causal deployment.

**Keywords:** edge–cloud collaboration; object detection; selective offloading; resource allocation; edge computing

## 1 引言

轻量检测器使智能相机、野外监测节点和移动机器人能够在本地持续处理图像，但受模型容量、功耗和端侧算子支持限制，其检测质量通常低于云端大模型。全部上传通常能够提高精度，却持续消耗上行带宽，并引入编码、传输、云推理和结果回传开销。因此，端云协同检测需要在任务质量与通信资源之间进行选择 [1–7,35]。本文所称“成本感知”专指编码后 JPEG payload 的通信字节成本；链路层字节、下行结果、能耗、货币成本和云计算量另行测量或列为研究边界。

现有方法从不同角度判断是否调用强检测器。Adaptive Feeding（AF）原本面向快/慢检测器级联，根据图像难度在二者之间选择 [40]；Difficult-Case based Small-Big model（DCSB）通过三阈值规则识别难例 [8,9]；OffloaD 依据检测失败风险进行选择性卸载 [10]。Object detection Reward per Image with Context evaluation（ORIC）进一步使用代表性上下文评估单幅图像对整体检测指标的贡献，并以其累积分布映射得到 modified ORIC values（MORIC）[11]。这些工作为二元快/慢检测或端/云选择提供了重要基础，但其原生控制方式并不都等价于本文的严格逐图字节预算。

真实链路还必须决定“上传什么”。同一输入可以保留端侧结果，也可以直接上传数据集原始 JPEG bitstream（Original）、上传以质量参数 90 重新编码的 JPEG（Q90），或上传由端侧检测框生成的感兴趣区域（region of interest，ROI）。在相同 payload-byte budget 下，Q90 与 ROI 能覆盖更多输入，但可能分别损失图像细节与场景上下文。由于每个载荷的检测收益和实际编码字节都随图像变化，先用动作无关分数排序图像、再为入选图像分配载荷，会割裂相互耦合的成员选择与载荷选择。

为此，本文提出上下文图像—动作价值调度（Contextual Image–Action Value Scheduling，CAVS）。CAVS 将端侧检测集合编码为紧凑语义表示，为每个候选载荷预测相对端侧执行的上下文 AP 边际价值；容量调度器再依据逐图实测字节选择执行成员与载荷。显式字节成本不进入价值网络，因此同一组价值预测可由不同预算下的资源控制器复用。这里的价值预测与资源调度是两个互补模块：前者比较任务效用，后者保证字节可行性。

离线与在线部署具有不同信息结构。全信息离线方案观察评测批次中全部候选价值和成本，并通过拉格朗日—修复算法产生容量可行参考；该启发式不保证一般多选背包的精确最优。零前视在线方案则在图像 $i$ 到达后、观察图像 $i+1$ 之前完成决策与执行，并保证任意输入前缀均不超过 payload 预算。本文分别报告二者，避免把真实硬件传输与因果在线决策混为一谈。

本文的主要贡献如下：

1. 将端云目标检测建模为带逐图实测 payload 字节的图像—动作选择问题，统一表示 Edge、Original、Q90 和可选 ROI，并分别给出全信息离线容量约束与零前视在线前缀约束。

2. 提出 AP 对齐、显式价值—成本解耦的 CAVS，把 ORIC 的上下文图像级奖励推广到图像—动作价值；价值网络预测载荷效用，容量调度器据此联合选择执行成员与载荷。

3. 在 COCO、Oregon 与跨检测器架构的 VOC 配置上进行冻结协议评测，并在 RK3588 端云测试床上验证真实逐图在线执行。COCO 同价值输出的受控结构消融支持将多个图像—动作候选保留到容量调度阶段；在线三档预算相较各档最强外部基线分别提高 0.0163、0.0369 和 0.0372 AP，同时满足全部前缀 payload-byte 约束。

## 2 相关工作

### 2.1 检测级选择性卸载

Neurosurgeon 和 split computing 研究神经网络在端、边、云之间的切分 [1–4]；NoScope、Reducto 和 Chameleon 等视频系统则利用专用过滤器或时间冗余降低处理成本 [5–7]。本文假设端侧检测结果始终可独立使用，关注检测完成后的样本级升级，而非网络切分或视频帧过滤。

与本文更接近的是快/慢检测器级联与选择性卸载。AF 以候选框统计训练线性分类器，在快速检测器与高精度检测器之间选择；它本身并非为端云卸载提出，但可适配为二元端/云比较器 [40]。DCSB 使用置信度、检测数量和最小目标面积比例三项阈值识别难例 [8,9]；device–cloud collaborative inference（DCCI）进一步结合内容与设备配置 [36]。OffloaD、EdgeBoost 与 V2X 端云检测分别从失败风险、置信度增强和车辆网络时延出发进行选择性卸载 [10,37,39]。

ORIC 将目标图像与代表性上下文集合共同评估，以近似该图像对整体 mAP 的边际贡献；MORIC 则对 ORIC 值进行累积分布映射，并通过加权回归强调高价值样本 [11]。原工作在多组检测器对和 Jetson–GPU 测试床上验证二元卸载。CAVS 扩展其基于上下文、与 mAP 对齐的奖励，将图像级 Edge/Cloud 选择推广为图像—动作估值，使 Original、Q90 和 ROI 能在统一字节约束下竞争。

### 2.2 任务感知载荷与系统配置

任务感知编码和区域传输通过减少无关像素来节省通信。AccMPEG 依据任务梯度调整宏块质量 [13]；EdgeDuet、TileClipper、RegenHance、RegionFilter 和 VaBUS 使用切片、区域增强或背景建模 [14–18]；Selective Cloud Offloading 通过不确定性区域和 packing 改善云端检测 [19]。这些工作表明压缩与 ROI 具有潜在收益，也揭示了区域裁剪对上下文丢失和端侧定位误差的敏感性。

OneAdapt 通过在线梯度快速调整视频分析配置 [12]；TURBO 在服务级联合选择云模型、压缩配置和带宽，以满足实时系统的效用与时限约束 [38]。这些方法的决策对象是时间窗口或服务—配置；CAVS 则为当前图像的候选像素载荷预测检测价值，并以该图像的实际编码字节计费。

### 2.3 决策学习与在线资源控制

Learning to defer、多专家路由、FrugalML 和 predict-then-optimize 分别研究专家选择、预算化模型调用和面向下游目标的预测 [20–24]。置信度级联研究界定了标量置信度足够的条件 [27]，校准研究则表明神经网络原始分数并不天然等价于可靠概率 [28]。实时 token-bucket 路由进一步区分平均速率、短时突发和因果逐样本决策 [25]。这些思想为 CAVS 提供一般决策框架，但检测 AP 的非可加性以及载荷成本的逐图变化，使本问题不能直接化为单样本分类损失。

Budget-Adaptive Routing 将依赖轻量模型输出的路由称为 weak-conditioned routing，并指出其在高卸载率下仍承担强制执行轻量检测器的开销 [26]。CAVS 属于这一范式，因为端侧检测框既是价值特征，也是 ROI 的几何来源。因此，本文显式报告 mandatory weak-detector inference overhead。综合现有研究，CAVS 的区别不在于首次使用 JPEG、ROI 或容量优化，而在于以 AP 对齐的图像—动作价值统一比较这些载荷，并由逐样本字节约束联合选择成员与动作。

## 3 系统模型与问题定义

图 1 给出 CAVS 的端到端信息流。端侧检测结果一方面进入价值模型，预测相对 Edge 的云动作价值；另一方面驱动 Original、Q90 与 ROI 候选载荷的构造和实际字节测量。动作价值与逐图字节成本共同进入预算约束决策，并分别由全信息调度器或零前视在线控制器选择执行动作。离线训练使用标注生成奖励，部署阶段不访问标注。

![图 1 CAVS 系统与信息流。](figures/fig_cavs_system_overview.svg)

**图 1 CAVS 系统与信息流。** 端侧检测同时驱动云动作价值预测与候选 payload 构造；候选载荷在调度前完成编码并得到实际字节成本。全信息调度器使用完整批次信息产生容量可行参考，零前视控制器则仅使用当前与历史信息，在读取下一输入前完成决策和执行。

### 3.1 端云检测与动作空间

给定输入序列或批次 $\mathcal D=\{I_i\}_{i=1}^{N}$，端侧检测器 $D_e$ 对当前图像产生检测集合

\[
E_i=D_e(I_i)=\{(b_{ij},y_{ij},p_{ij})\}_{j=1}^{n_i},
\tag{1}
\]

其中 $b_{ij}$、$y_{ij}$ 和 $p_{ij}$ 分别为框、类别和置信度。$E_i$ 同时是未上传时的最终预测，也是路由器的语义输入。配置级候选动作集合及每图可行动作集定义为

\[
\mathcal A^{\mathrm O}=\{e,o\},\qquad
\mathcal A^{\mathrm{OQ}}=\{e,o,q\},\qquad
\mathcal A^{\mathrm{OQR}}=\{e,o,q,r\},\qquad
\mathcal A_i\subseteq\mathcal A^{\Gamma},
\tag{2}
\]

其中 $\Gamma\in\{\mathrm O,\mathrm{OQ},\mathrm{OQR}\}$ 表示冻结的动作空间配置，$e$ 表示保留端侧结果，$o$、$q$ 和 $r$ 分别表示 Original、Q90 和 ROI。配置中出现的 $e$、$o$ 和 $q$ 始终可用；仅当端侧框能够构造满足第 4.4 节 ROI profile 的有效 crop 时，才有 $r\in\mathcal A_i$，否则从该图像的可行动作集中移除 $r$。下文在配置 $\Gamma$ 固定时将 $\mathcal A_i^\Gamma$ 简写为 $\mathcal A_i$。ROI 动作不是多个独立 crop：系统把合格端侧框扩张后合并为一个外接裁剪框，并将该 crop 编码成一个 JPEG payload。因此，每幅图像在每种配置下仍只选择一个动作。

对每个可行云动作 $a\in\mathcal A_i\setminus\{e\}$，候选构造模块在调度前生成载荷 $P_i^a$，并以编码后 JPEG 长度 $c_i^a=|P_i^a|$ 作为主预算成本；调度器只上传最终选中动作对应的载荷。Original 与 Q90 的云输出直接作为整图检测结果，ROI 输出则经 crop 坐标恢复与 MAUR 融合得到 $F_i^r$；端侧动作满足 $F_i^e=E_i$、$c_i^e=0$。若请求失败，已发送的 payload 仍计入累计成本，检测结果回退为 $F_i^e$；正式在线主矩阵未出现云请求失败。本文从问题定义起即只以 JPEG payload 为优化预算，并将 HTTP 上行、下行检测结果和协议开销作为独立系统指标报告。

### 3.2 全信息 Fixed-byte 问题

令 $\hat v_i^a$ 为动作 $a$ 相对 Edge 的预测任务价值，其中 $\hat v_i^e=0$，价值模型只预测可行云动作；令 $x_i^a\in\{0,1\}$ 为动作选择变量。全信息 Fixed-byte 决策写为

\[
\begin{gathered}
\max_x\quad \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a\hat v_i^a,\\
\text{s.t.}\quad \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a c_i^a\le B,\\
\sum_{a\in\mathcal A_i}x_i^a=1,\quad \forall i,\\
x_i^a\in\{0,1\}.
\end{gathered}
\tag{3}
\]

式（3）是多选容量问题。由于检测 AP 不能分解为独立单图贡献，$\sum_{i,a}x_i^a\hat v_i^a$ 是与 AP 对齐的可学习代理目标，而不是对 AP 的直接可加优化；最终质量始终由完整预测集合上的 evaluator 计算。

本文以 Always-Original 的总 payload 为参考，定义

\[
B_\rho=\rho\sum_{i=1}^{N} c_i^o,\qquad \rho\in\{0.10,0.25,0.40\}.
\tag{4}
\]

因此，10% 指 Always-Original JPEG payload 总量的 10%，而非上传 10% 的图像。若每图只有一个等成本云动作，式（3）退化为选择至多 $K$ 个正价值样本；正价值不足 $K$ 时其余图像保留 Edge（附录 A.1）。一旦候选动作成本不同，成员与动作通常需要联合求解。

### 3.3 零前视因果在线问题

全信息调度可观察批次中全部图像，在线策略则只能使用当前及历史信息。本文的零前视协议要求：图像 $i$ 的动作必须在读取图像 $i+1$ 前确定并执行；系统完成可能的上传、云推理、融合与结果提交后，才读取下一幅图像。实现不使用未来图像、标注或预先计算的全序列候选表。

在线控制还要求前缀安全。对前 $t$ 个顺序输入，令累计参考原图字节为 $C_t^o=\sum_{\tau\le t}c_\tau^o$、累计已花费 payload 为 $S_t=\sum_{\tau\le t}c_\tau^{a_\tau}$，则主协议要求

\[
S_t\le \left\lfloor\rho C_t^o\right\rfloor,
\qquad \forall t\in\{1,\ldots,N\},
\tag{5}
\]

式（5）不允许借用未来字节，因此比仅约束序列末端总量更严格。在线结果对输入顺序敏感；正式 COCO trace 按升序 image ID 固定，顺序清单与校验哈希随复现材料发布。

## 4 CAVS：上下文图像—动作价值学习

### 4.1 检测集合与动作表示

CAVS 由检测集合编码器、动作条件价值网络和外置成本调度器组成。每幅图像至多保留置信度最高的 $M=64$ 个端侧检测，记实际数量为 $n_i$；框特征包括归一化中心坐标、宽、高、面积和置信度，类别使用可学习嵌入。共享多层感知机（multilayer perceptron，MLP）编码各检测后，通过均值池化和最大池化获得置换不变表示：

\[
h_i^{set}=
\begin{cases}
\left[\dfrac{1}{n_i}\sum_j h_{ij}\;\middle\|\;\max_j h_{ij}\right], & n_i>0,\\[3pt]
\mathbf 0, & n_i=0.
\end{cases}
\tag{6}
\]

除逐框集合表示外，CAVS 还使用由端侧检测输出计算的图像级全局统计 $z_i$，包括对数宽高比、按 64 归一并截断的有效检测数量、最大与平均置信度，以及截断到 $[0,1]$ 的逐框面积率之和；无有效检测时，数量、置信度和面积统计置为 0。该向量不包含原始像素统计或语义场景类别。

动作条件输入由可学习的动作类型嵌入 $\operatorname{Emb}(a)$ 和动作描述 $u_i^a$ 组成。$u_i^a$ 包含归一化云输入尺度、云调用数量和区域数量；ROI 额外包含合并 crop 的面积率及来源指示。$\operatorname{StripCost}(u_i^a)$ 将 payload 比例、编码与响应字节以及资源成本等显式成本坐标置为 0；兼容保留的预算输入也固定为 0。网络先在训练折目标的标准化空间预测动作价值，再反标准化为调度器使用的价值：

\[
\hat y_i^a=f_\theta\!\left(h_i^{set},z_i,\operatorname{Emb}(a),\operatorname{StripCost}(u_i^a)\right),
\qquad
\hat v_i^a=\sigma_{tr}\hat y_i^a+\mu_{tr}.
\tag{7}
\]

其中 $\mu_{tr}$ 和 $\sigma_{tr}$ 分别为训练折原始云动作奖励的均值和标准差。部署时端侧参考价值设为 $\hat v_i^e=0$，不同云动作共享网络参数。实际编码字节 $c_i^a$ 不进入价值网络，只在调度阶段参与预算约束；因此，同一价值模型可与不同字节预算配合使用。

### 4.2 上下文图像—动作奖励

ORIC 在代表性图像上下文中定义与 mAP 对齐的单图奖励；MORIC 则是对 ORIC 值施加经验累积分布映射后得到的图像级路由分数 [11]。CAVS 保留 ORIC 式上下文 AP 差分，但将被替换对象从二元弱/强检测推广为动作 $a$，并不对动作训练目标施加 MORIC 的 CDF 映射。对训练图像 $i$，从同一训练折构造不含 $i$ 的上下文 $\mathcal C_i$，对云动作定义原始上下文奖励

\[
r_i^a=(|\mathcal C_i|+1)\left[
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^a)-
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^e)
\right],\quad a\in\mathcal A_i,\;a\ne e,
\qquad r_i^e=0.
\tag{8}
\]

正式协议取 $|\mathcal C_i|=32$。上下文从目标图像所在训练折内、排除目标图像后无放回抽取；抽样由 fold、目标 image ID 和固定 context seed 的 SHA-256 哈希确定。每个目标图像的全部动作共享同一上下文，不同训练 seed 也复用同一冻结上下文表，从而避免动作差值混入 context sampling noise。上下文与标注只用于离线监督生成；部署仅使用端侧预测。该奖励仍是完整数据集 AP 的局部代理，因此模型选择和最终比较均以完整 evaluator 为准，而非奖励回归误差。

### 4.3 回归—排序联合训练

令训练折原始云动作奖励的均值和标准差分别为 $\mu_{tr}$ 与 $\sigma_{tr}$，并定义标准化训练目标 $y_i^a=(r_i^a-\mu_{tr})/\sigma_{tr}$。式（7）中的 $\hat y_i^a$ 是网络预测的标准化价值，$\hat v_i^a$ 是反标准化后交给调度器的预测动作价值。CAVS 同时最小化 Smooth L1 回归误差和同图动作排序误差：

\[
\mathcal L=\alpha\frac{1}{|\Omega|}\sum_{(i,a)\in\Omega}
\operatorname{SmoothL1}(\hat y_i^a,y_i^a)
+(1-\alpha)\frac{\sum_{(i,a,b)\in\Pi}w_i^{ab}
\log\!\left(1+e^{-s_i^{ab}(\hat y_i^a-\hat y_i^b)}\right)}
{\sum_{(i,a,b)\in\Pi}w_i^{ab}},
\tag{9}
\]

其中 $s_i^{ab}=\operatorname{sgn}(y_i^a-y_i^b)$、$w_i^{ab}=|y_i^a-y_i^b|$，正式配置取 $\alpha=0.5$。因此，第 4 节的概念链依次为原始奖励 $r_i^a$、标准化目标 $y_i^a$、标准化预测 $\hat y_i^a$ 和调度价值 $\hat v_i^a$。五折 out-of-fold（OOF）预测用于开发阶段的稳定性与机制诊断；最终学习器不集成 OOF 模型，而是以三个预先固定的种子在完整 fit 上独立重训，再平均反标准化后的价值。训练与统计协议见第 6.3 节。

### 4.4 ROI profile 与融合规则

主配置 Current@640 首先保留端侧置信度不低于 0.25 的框。按单框面积占原图比例 $s$ 扩张：以框中心为中心，$s<1\%$ 时将宽和高分别扩张至原来的 1.8 倍，$1\%\le s<10\%$ 时分别扩张至 1.6 倍，$s\ge10\%$ 时分别扩张至 1.4 倍；扩张框裁剪到图像边界后合并为一个外接矩形。若不存在满足置信度阈值的端侧框，或合并区域超过原图面积的 65%，则该图像的 ROI 动作不可用，并从第 3.1 节定义的每图可行动作集 $\mathcal A_i$ 中移除；否则 crop 以 Q90 编码并送入 640 输入的云检测器。本文将这一确定性几何规则称为 Current@640 profile。

本文定义 masked-area update-and-replacement（MAUR）融合：先将云端 crop 预测映射回原图；删除中心点落在 crop 内的端侧框，并删除与云框同类且交并比（intersection over union，IoU）不低于 0.5 的重复端侧框；随后合并其余端侧框与云框，并执行类内 IoU 阈值 0.7 的非极大值抑制（non-maximum suppression，NMS）。框中心恰落在裁剪边界时按区内处理。Current@640 只产生一个合并 crop，因此不存在多个 ROI 之间的融合顺序问题。

2×2 消融还使用两个对照。Historical-cost@640 在 $s<2\%$、$2\%\le s<4\%$、$4\%\le s<8\%$ 和 $s\ge8\%$ 时，分别将框的宽和高扩张至原来的 4.0、3.0、2.5 和 1.4 倍。Reversible fusion 以同类 IoU 不低于 0.3 贪心匹配端/云框；匹配时云置信度不低于端侧置信度才替换，未匹配端侧框保留，未匹配云框仅在置信度不低于 0.1 时加入，最后执行类内 NMS 0.7。

ROI 是可选动作而非强制模块：若其预测净值低于 Original、Q90 或 Edge，调度器可以完全不选 ROI。O、OQ 和 OQR 是三种独立训练的动作空间配置；各配置均重新生成 reward 并重训，因此 OQ−O 与 OQR−OQ 反映“增加动作并重新学习后的配置差异”，而不是只改变执行动作的严格单变量因果效应。

## 5 调度与系统实现

### 5.1 全局容量可行调度

给定影子价格 $\lambda$，每幅图像独立选择

\[
a_i(\lambda)=\arg\max_{a\in\mathcal A_i}
\left(\hat v_i^a-\lambda c_i^a\right).
\tag{10}
\]

实现以 $\lambda_L=0$ 和 $\lambda_U=1+\max_{i,a:c_i^a>0}[\max(0,\hat v_i^a)/c_i^a]$ 为搜索区间，执行 80 次二分并保留容量可行一侧。每图净值并列时依次选择成本更低的动作；若成本仍相同，则按预先固定的动作顺序确定。随后枚举所有正价值增量升级，并按“边际价值/增量字节、增量价值、较小增量成本、预先固定的动作顺序、图像索引”的确定性顺序尝试加入，直至候选耗尽；不可行升级被跳过。该拉格朗日—修复过程保证最终总 payload 不超过 $B$，但不保证一般多选背包的精确最优，因此全文仅将其称为全信息容量可行参考（full-information feasible reference）。

### 5.2 零前视在线控制

在线控制器不预先评分完整序列。当前输入到达后，系统构造候选载荷、测量实际字节并预测动作价值。记经动作可用性与前缀预算共同筛选后的集合为

\[
\mathcal A_t^{\mathrm{feas}}=
\left\{a\in\mathcal A_t:\;S_{t-1}+c_t^a\le
\left\lfloor\rho C_t^o\right\rfloor\right\},
\]

其中 $\mathcal A_t$ 已按第 3.1 和 4.4 节排除本来不可用的 ROI 动作，Edge 因成本为 0 始终可行。控制器随后依据当前价格选择

\[
a_t=\arg\max_{a\in\mathcal A_t^{\mathrm{feas}}}
\left(\hat v_t^a-\lambda_t\frac{c_t^a}{m_o}\right),
\qquad
\lambda_{t+1}=\Pi_{[0,5]}\!\left[
\lambda_t+\eta\frac{c_t^{a_t}-\rho c_t^o}{m_o}
\right],
\tag{11}
\]

其中 $m_o$ 为 calibration 上 Original payload 的中位数，$\Pi$ 表示投影。初始价格候选为 $\{0,0.01,\ldots,2.00,2.5,3,4,5\}$。

步长候选为 $\{0,0.00025,0.0005,0.001,0.0015,0.002,0.003,0.005,0.01,0.02\}$。需要强调的是，前缀预算的硬可行性由 $\mathcal A_t^{\mathrm{feas}}$ 的动作屏蔽直接保证；价格更新仅调节后续动作的价值—成本偏好，用于改善容量跟踪。系统在 calibration 上选择预测动作价值总和最大的前缀可行轨迹，并依次以更低累计字节、更小初始价格和更小更新步长处理并列。CAVS-OQ 在 10%/25%/40% 的冻结 $(\lambda_1,\eta)$ 为 $(0.47,0)$、$(0.30,0.001)$ 和 $(0.17,0.0015)$；CAVS-OQR 对应为 $(0.47,0)$、$(0.30,0.0015)$ 和 $(0.17,0.002)$。静态价格消融令 $\eta=0$。动作执行和云端融合完成后才读取下一幅输入。

### 5.3 RK3588–云硬件链路

板端使用 Orange Pi 5 Ultra/RK3588 [34,45]，操作系统为 Orange Pi 1.0.0（Ubuntu 22.04.4 LTS），kernel 为 6.1.43-rockchip-rk3588，Python 为 3.10.12。端侧模型是输入 640 的 YOLO26n RKNN-FP16 导出版本；正式日志记录 RKNN Toolkit Lite 2.3.2、runtime 2.3.0 和 driver 0.9.6。云端为共享 RTX 4090 上的 YOLO26x PyTorch checkpoint，服务环境为 Python 3.11.5、PyTorch 2.1.2、CUDA 11.8、cuDNN 8.7 和 Ultralytics 8.4.10。每幅输入依次完成图像读取、端侧检测、Original/Q90/ROI 候选载荷构建及实际字节测量、动作价值预测与预算决策、被选载荷上传、云端检测、坐标恢复和结果融合；未选载荷不上传。

系统使用紧凑二进制 HTTP 请求体承载定长头、规范化 metadata 和原始 JPEG bytes，不使用 Base64。模型 checkpoint、输入顺序、协议版本、策略及控制参数在实验前冻结并随运行记录，以保证不同方法共享同一执行环境。服务端和客户端均过滤非有限、越界或零面积框。系统分别记录算法预算使用的 JPEG payload、实际 HTTP 上下行字节、各阶段时延和完整端到端（end-to-end，E2E）时延。

### 5.4 复杂度与高预算边界

集合编码至多处理 64 个框；全信息调度复杂度近似为 $O(TN|\mathcal A|+U\log U)$，其中 $T$ 为价格搜索次数、$U$ 为候选升级数；在线动作选择为 $O(|\mathcal A|)$。由于 CAVS 在决定上传前始终运行端侧检测器，高预算下仍存在 mandatory weak-detector inference overhead。本文在结果部分报告完整阶段时延，但不以共享 GPU 测量替代独占模型基准，也不对未测量的能耗和峰值内存作推断。

## 6 实验方法

### 6.1 数据集与检测器

**COCO。** 使用 COCO 2017 [30]。从 train2017 构造互斥的价值模型训练集和控制器校准集，并在 val2017 的 5,000 幅图像上评测。端侧与云端检测器分别为输入 640 的 YOLO26n 和 YOLO26x [33]。

**Oregon。** Oregon Critters 是作者持有的 45 类相机陷阱野生动物数据集，共 99,231 幅图像。数据按相机事件划分，检测器训练、价值模型训练、控制器校准和最终评测所用事件互不重叠。端侧与云端分别采用输入 640 的 YOLO26n 和 YOLO26x，并仅使用相应训练与验证数据确定模型参数。

**PASCAL VOC。** 使用 VOC2007+2012 trainval 构造价值模型训练集和校准集，并在 VOC2007 test 的 4,952 幅图像上评测 [31]。端侧采用 SSDLite320-MobileNetV3，云端采用 Faster R-CNN-ResNet50-FPN-V2；二者使用官方 COCO 预训练权重，不在 VOC 上微调 [41–44]。评测时将输出映射到 VOC 的 20 类标签空间，并按官方语义忽略 `difficult` 目标。

三个数据集均严格分离训练、校准与评测数据；评测集不参与模型、超参数或控制参数选择。具体划分清单和模型摘要随复现材料提供。

### 6.2 指标、编码与比较方法

主指标为 IoU 0.50 至 0.95（间隔 0.05）上的 AP50–95。COCO 使用 `pycocotools` 的 COCOeval；VOC 使用相同 IoU 网格和 101 点插值，并按第 6.1 节忽略 `difficult` 目标。Original 复用原始 JPEG 位流，Q90 采用 Pillow 的 `quality=90`、`optimize=True` 和 `subsampling=2`，ROI 使用第 4.4 节定义的 Current@640 与 MAUR。检测输出统一采用 0.001 的置信度阈值并至多保留每图 300 个框，YOLO 的 NMS IoU 阈值为 0.7。所有方法共享检测输出、候选载荷、Always-Original 容量分母和评测器。系统实验另报告云覆盖率、JPEG payload、HTTP 上下行字节、端到端时延和吞吐。通信量采用十进制单位，即 $1\ \mathrm{kB}=1{,}000$ B、$1\ \mathrm{MB}=1{,}000{,}000$ B。

比较方法包括 Edge-only、Always-Original、Random、Confidence、DCSB [8]、AF [40] 和 ORIC/MORIC [11]。ORIC/MORIC 按原方法构造图像级分数，并在共同的 Edge/Original 固定字节控制下评测。DCSB 与 AF 保留各自的图像评分规则，但使用本文冻结端侧检测器的输出；为匹配统一字节预算，其适配版本按图像分数选择 Original，原生阈值工作点则单独报告。因此这些结果属于协议对齐的重实现或适配，而不是官方实现的逐位复现。

TwoStage 受控消融直接复用 CAVS-OQ 的动作价值，不另行训练评分模型。它先在每幅图像内完成动作投影：

\[
\tilde a_i=\arg\max_{a\in\{o,q\}}\hat v_i^a,
\qquad
s_i=\max_{a\in\{o,q\}}\hat v_i^a,
\tag{12}
\]

投影后的 $(\tilde a_i,s_i)$ 进入与 CAVS-OQ 相同的固定字节调度器。因此该消融仅改变动作选择的时机：TwoStage 在容量分配前固定载荷，CAVS-OQ 则让 Original 与 Q90 持续竞争到容量求解阶段。零前视比较中的图像级方法均使用式（11）的前缀可行性屏蔽和相同控制器，需要尺度对齐的分数仅利用校准集完成映射；不同方法之间只改变当前图像的评分方式。

### 6.3 训练与统计

各 CAVS 配置均使用五折折外预测进行开发阶段的稳定性与机制诊断，最终在完整训练集上以三个预先固定的随机种子独立训练，并平均反标准化后的动作价值。折外模型不参与最终集成，评测结果也不用于选择训练种子。正式学习器最多接收 64 个检测框，context size 为 32；回归与排序损失权重均为 0.5，采用 AdamW（初始学习率 $10^{-3}$、weight decay $10^{-4}$）、batch size 192，最多训练 50 个 epoch，并以 patience 8 早停。训练种子不依据评测结果筛选；三个单模型的均值与样本标准差在第 7.1 节报告，仅用于描述训练稳定性。

离线 Random 使用 50 个预先固定的随机种子，在线 Random 使用 5 个，分别报告均值及种子间离散程度。主结果为冻结评测集上的点估计。仅 Profile×Fusion 消融使用 1,000 次配对图像 bootstrap [29]，报告方法差值的百分位 95% 置信区间；该区间反映评测图像重采样的不确定性，不包含重新训练造成的模型不确定性。未计算区间的比较均表述为点估计。

### 6.4 端云测试床协议

COCO 测试床同时评测全信息离线参考和零前视在线部署。前者预先生成完整输入序列的候选价值与载荷，再执行第 5.1 节的容量可行调度；后者按照第 3.3 节逐图观测、决策和执行，决策时不读取后续输入。离线—在线比较采用同一部署对齐价值模型、RKNN 端侧输出、云端检测器和动作定义，使差异主要来自完整序列可见性、前缀容量约束与实时执行路径，而不混入模型版本变化。

所有在线方法使用相同的 Orange Pi、RTX 4090 云端、模型参数、二进制协议和固定输入顺序，并分别独立完成 5,000 幅图像的顺序处理；当前动作完成后才读取下一幅图像。网络敏感性实验仅控制往返时延和可用带宽，同时保持无丢包、无云端失败和无截止时间回退，且调度器不读取网络状态。因此该实验衡量传输条件对时延与吞吐的影响，而不评价网络状态自适应调度。

## 7 结果与分析

### 7.1 三数据集 Fixed-byte 主结果

图 2 汇总三组端—云检测器在冻结协议下的点估计结果。正文以预先固定的 CAVS-OQR 为主方法；每个数据集均重新生成动作奖励并训练路由器，因此这里检验的是算法流程的跨数据集适用性，而不是 checkpoint 的零样本迁移。

![图 2 三数据集 Fixed-byte 结果。](figures/fig_fixed_byte_three_datasets_s7_revised.svg)

**图 2 三数据集 Fixed-byte AP50–95。** 粗蓝实线及实心圆表示 CAVS-OQR 三模型价值集成后的最终 AP；其余折线分别表示所有已评测外部比较器。各方法同时使用独立颜色、线型、标记形状和右端直连标签，以辨认数值接近或局部重叠的折线。DCSB 与 AF 均为 Original-only 的预算匹配适配；所有方法共享所在数据集的检测输出、容量分母和 evaluator。

CAVS-OQR 在本文所评测的九个数据集—预算组合中均取得最高 AP 点估计。相对每档表现最佳的所评测外部比较器，COCO 的增益为 +0.0280/+0.0360/+0.0320 AP，Oregon 为 +0.0164/+0.0149/+0.0062 AP，VOC 为 +0.0271/+0.0247/+0.0205 AP。COCO 与 VOC 的点结果随预算单调提高；Oregon 从 25% 的 0.517405 降至 40% 的 0.516663，下降 0.000742 AP，说明优化可加价值代理并不保证非可加 AP 随预算单调提高。

补齐的 VOC Adapted DCSB/AF 结果中，Adapted AF 在三档均较强，CAVS-OQR 相对其提高 0.028775/0.037102/0.038995 AP；纳入原有 MORIC 后，MORIC 仍是三档最强外部比较器，因此上述 VOC 总体领先值保持不变。VOC 的 DCSB-Native 和 AF-Native 分别取得 0.568991 和 0.597841 AP，但消耗 45.36% 和 51.48% 的 Always-Original 字节，故不与 10%/25%/40% 固定字节点直接比较。

三个独立训练单模型的 AP 样本 SD 在 COCO 为 0.000274–0.001553，在 VOC 为 0.000914–0.001957，在 Oregon 为 0.005350–0.006335；集成结果相对三个单模型均值的差异范围为 −0.002734 至 +0.003217 AP。特别地，Oregon 40% 档相对最佳比较器的点差 0.006214 AP 与该档跨训练 SD 0.006277 处于同一量级，因此这一路径只支持点估计排序，不能据此声称统计显著。正文未完成逐项 paired interval 的比较均遵循同一表述限制。

**表 1 三数据集 Fixed-byte AP50–95 完整点结果。** CAVS-OQR 为三个独立模型反标准化价值平均后的集成结果，Random-50 为 50 个预先固定随机种子的均值。DCSB 与 AF 为 Original-only 的预算匹配适配。

| 数据集 / 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **COCO — CAVS-OQR** | **0.480277** | **0.532900** | **0.560178** |
| COCO — MORIC | 0.452230 | 0.496914 | 0.528170 |
| COCO — ORIC | 0.449174 | 0.494970 | 0.527069 |
| COCO — Confidence | 0.447029 | 0.489546 | 0.520311 |
| COCO — Adapted DCSB | 0.442595 | 0.478828 | 0.510588 |
| COCO — Adapted AF | 0.443563 | 0.477221 | 0.510021 |
| COCO — Random-50 | 0.433586 | 0.468908 | 0.500443 |
| **Oregon — CAVS-OQR** | **0.506028** | **0.517405** | **0.516663** |
| Oregon — MORIC | 0.479826 | 0.493286 | 0.506171 |
| Oregon — ORIC | 0.483153 | 0.502061 | 0.509599 |
| Oregon — Confidence | 0.489643 | 0.502499 | 0.510449 |
| Oregon — Adapted DCSB | 0.468816 | 0.485255 | 0.487301 |
| Oregon — Adapted AF | 0.465760 | 0.484632 | 0.494235 |
| Oregon — Random-50 | 0.471087 | 0.484147 | 0.495973 |
| **VOC — CAVS-OQR** | **0.531486** | **0.579678** | **0.612721** |
| VOC — MORIC | 0.504416 | 0.554956 | 0.592212 |
| VOC — ORIC | 0.502039 | 0.553636 | 0.589734 |
| VOC — Confidence | 0.495439 | 0.532334 | 0.569091 |
| VOC — Adapted DCSB | 0.494798 | 0.528534 | 0.553232 |
| VOC — Adapted AF | 0.502711 | 0.542576 | 0.573726 |
| VOC — Random-50 | 0.482501 | 0.503851 | 0.532850 |

### 7.2 COCO：核心机制和容量使用

表 2 显示 CAVS 不把字节预算退化为调用率预算：压缩与 ROI 使 10% 字节容量覆盖 28.18% 的图像，40% 时覆盖 83.04%，且三档容量利用率均超过 99.997%。

**表 2 COCO CAVS-OQR 调度组成与容量使用。**

| 预算 | Edge | Original | Q90 | ROI | 云覆盖率 | 容量利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 3,591 | 29 | 1,076 | 304 | 28.18% | 99.9976% |
| 25% | 2,049 | 63 | 2,468 | 420 | 59.02% | 99.9985% |
| 40% | 848 | 209 | 3,651 | 292 | 83.04% | 99.9996% |

低预算主要利用 Q90 和 ROI 扩大云覆盖，高预算则增加 Original；这一动作迁移与预算增加时影子价格下降的机制预期一致，但表 2 本身不是对该机制的独立因果证明。

### 7.3 联合图像—动作容量调度的受控结构消融

如表 3 所示，TwoStage 与 CAVS-OQ 使用完全相同的 top-64 集合编码、优化器、fold、训练种子、O/Q contextual reward、动作价值输出和 Fixed-byte 求解器。TwoStage 先按式（12）把每图的两个动作价值投影为一个固定载荷及其原始价值，再进行容量选择；CAVS-OQ 则让 Original 与 Q90 候选一直保留到容量调度阶段。两者均不对价值做经验分位变换，受控差异仅为动作投影发生在容量选择之前还是之中。该比较支持联合图像—动作容量调度机制，但不外推为一般条件下的严格因果证明。

**表 3 COCO 受控同骨干结构消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| TwoStage：每图动作投影 → 容量选择 | 0.455455 | 0.493310 | 0.525178 |
| CAVS-OQ：联合图像—动作容量调度 | **0.478224** | **0.528884** | **0.559286** |
| 差值 | **+0.022769** | **+0.035573** | **+0.034107** |

在分数、骨干、输入、训练设置和求解器保持一致的条件下，CAVS-OQ 的点结果比 TwoStage 高 0.0228–0.0356 AP。按相同训练 seed 配对后，三个单模型在三档预算上的差值也均为正，差值的跨种子 mean±SD 分别为 0.023169±0.001068、0.035475±0.002310 和 0.034729±0.001115。结果支持“将动作候选保留到容量调度阶段优于先做每图动作投影”的机制；该结论限于 COCO 的受控配置，不外推为三个数据集均完成了相同结构消融，跨种子 SD 也不替代图像级置信区间。当前实验尚未单独比较仅回归、仅排序与回归—排序联合损失，因而表 3 的收益不能归因于式（9）中的某一个训练项。

### 7.4 O/OQ/OQR 动作空间消融

图 3 与表 4 给出 VOC 的三档动作空间比较：O、OQ、OQR 均按各自 reward 生成五折 OOF 诊断，并分别以三个 full-fit seeds 重训。作为单动作参照，全量执行 Edge、Original、Q90 和 ROI+MAUR 时的 AP 分别为 0.473322、0.657511、0.653684 和 0.499819。

![图 3 VOC 动作空间消融。](figures/fig_voc_action_space_ablation_s7_revised.svg)

**图 3 VOC 动作空间消融与 OQR 动作使用。** （a）O、OQ 与 OQR 独立重训配置的绝对 AP50–95；（b）OQR 在 4,952 幅测试图像中选择 Q90 与 ROI 的数量，纵轴采用对数刻度并标注原始计数。该图同时显示性能变化和新增动作的实际使用程度；精确配置差值列于表 4。

**表 4 VOC 跨架构动作空间消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-O | 0.520249 | 0.557280 | 0.589427 |
| CAVS-OQ | 0.530787 | 0.579142 | **0.614054** |
| CAVS-OQR | **0.531486** | **0.579678** | 0.612721 |
| OQ − O | +0.010538 | +0.021862 | +0.024627 |
| OQR − OQ | +0.000699 | +0.000537 | −0.001333 |

OQ 配置相对 O 配置高 0.0105–0.0246 AP，占观察到的动作空间收益的大部分；OQR 相对 OQ 的配置差值仅为 +0.0007、+0.0005 和 −0.0013 AP。图 3(b) 显示 OQR 仅选择 77/49/3 个 ROI，而 Q90 动作为 796/1,816/2,796 个；随着预算增加，Q90 使用量上升而 ROI 接近消失。由于各配置独立生成 reward 并重训，这些差值不能解释为仅执行 Q90 或 ROI 的纯因果增益。

Oregon 的 OQR 三档均未执行 ROI；OQ 与 OQR 的差异来自独立训练形成的 O/Q 动作分配，而非 ROI 执行收益。因此本文只主张 ROI 可参与成本竞争，不主张它跨数据集普遍增益。

### 7.5 COCO Profile × Fusion 2×2

为评估 ROI 几何规则与融合方式对最终性能的影响，我们在同一 COCO 划分、云模型、fold、种子和调度器下，对 Current@640/Historical-cost@640 与 Reversible/MAUR 四组各自重建真实 reward 并全量重训。

**表 5 COCO Profile × Fusion 独立全量重训（AP50–95）。** Current@640+MAUR 为预先固定的主配置；表中不按预算事后突出单列最大值。

| ROI profile | Fusion | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| Current@640 | Reversible | 0.479071 | 0.531817 | 0.559932 |
| Current@640（预设主配置） | MAUR | 0.480277 | 0.532900 | 0.560178 |
| Historical-cost@640 | Reversible | 0.477784 | 0.531402 | 0.560025 |
| Historical-cost@640 | MAUR | 0.478155 | 0.531983 | 0.560701 |

Current+MAUR 是预先固定的主配置，但四格差异很小。Current 下 MAUR−Reversible 的 95% CI 在 10%、25% 和 40% 均跨 0（分别为 [−0.001916,+0.003460]、[−0.000061,+0.002212] 和 [−0.000607,+0.000946]）。区间由 1,000 次 paired image bootstrap 得到。因此，结果不支持“MAUR 全面显著更优”，也不允许按预算事后挑选四格最大值。

### 7.6 部署对齐缓存参考与零前视在线

![图 4 COCO 离线到在线的精度差距。](figures/fig_coco_offline_to_online_gap.svg)

**图 4 COCO 部署对齐缓存参考与零前视在线结果。** （a）同一 RKNN 输出域、动作定义和 CAVS-OQR 价值模型下的全信息缓存参考与零前视在线结果；（b）观察到的 cache−online AP50–95 差值。差值同时包含完整序列可见性、前缀容量控制以及缓存与实时执行路径的变化，不能解释为单一的“在线调度损失”。

全信息缓存参考为 0.4791/0.5311/0.5602，零前视在线为 0.4548/0.5053/0.5309，观察到的差值为 0.0243/0.0258/0.0293 AP。两者已控制 RKNN 输出域、动作语义与价值模型一致，从而排除了直接将 PyTorch 域学习器迁移到板端这一明显混杂；但现有实验没有进一步把序列信息、前缀控制和实时执行边界逐项拆开。三档在线运行均处理 5,000 幅固定顺序图像，云失败、非法框和最大前缀超额均为 0，提示请求失败不是该差距的直接来源。

### 7.7 零前视在线准确率

表 6 汇总同一零前视协议、同一硬件链路和同一 prefix-byte 计量下的在线准确率。

**表 6 COCO 零前视在线端云测试床结果（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR** | **0.454831** | **0.505348** | **0.530900** |
| CAVS-OQ（无 ROI） | 0.451892 | 0.500310 | 0.527717 |
| ORIC | 0.438553 | 0.468420 | 0.492202 |
| MORIC | 0.430520 | 0.463411 | 0.485750 |
| Confidence | 0.427864 | 0.461410 | 0.482862 |
| Adapted DCSB（Original-only） | 0.432997 | 0.464606 | 0.490678 |
| Adapted AF（Original-only） | 0.433734 | 0.465847 | 0.493665 |
| Random-5 | 0.421070±0.002418 | 0.444743±0.003250 | 0.465573±0.002642 |

CAVS-OQR 在三档均取得所评测方法中的最高在线点估计。每档所评测外部方法中表现最佳者分别为 10%/25% 下的 ORIC 和 40% 下的 Adapted AF，对应增益为 +0.016278、+0.036928 和 +0.037235 AP，即 0.0163–0.0372 AP。CAVS-OQR 相对独立训练的 CAVS-OQ 高 +0.002939/+0.005038/+0.003183 AP；这些是配置点差，不是带置信区间的纯 ROI 因果效应。除 Random-5 外，每个方法—预算单元为一次固定顺序的 5,000 图像完整运行；Random-5 报告五个预先固定种子的均值±样本 SD。所有运行的云失败、非法框和最大前缀超额均为 0。DCSB 与 AF 的原生阈值分别消耗 76.36% 和 78.17% 的 Always-Original payload，不属于 10%/25%/40% 固定预算点，故不混入表 6。

### 7.8 在线时延与通信成本

图 5 综合展示在线准确率、CAVS-OQR 的 E2E 分布统计和逐输入通信量。

![图 5 正式在线系统精度、E2E 与通信成本。](figures/fig_online_system_results_s7_revised.svg)

**图 5 正式在线系统结果。** （a）八种策略在三档预算下的 AP50–95，其中 DCSB 与 AF 为 Original-only 预算匹配适配；（b）CAVS-OQR 每档 5,000 个输入的 E2E mean、P50、P95 和 P99；（c）每输入图像的 JPEG payload、协议额外上行和检测结果下行。图（a）和图（b）的每条折线均在右端直接标注，图（c）直接标注堆叠组成；图（b）的分位数描述逐输入时延分布，不是重复运行之间的置信区间。

**表 7 CAVS-OQR 零前视在线系统结果。** E2E 从端侧开始处理当前输入计时，覆盖图像读取与解码、端侧推理、调度决策、按需编码与上传、云端推理、结果回传及端侧融合，至最终检测结果生成。等效 FPS 按 $1000/\mathrm{E2E\ mean}$ 换算，用于将平均响应时间表示为等效串行处理速率。

| 预算 | AP50–95 | 云覆盖率 | Payload/图 | HTTP 双向/图 | E2E mean/P50/P95/P99 | 等效 FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.454831 | 30.16% | 16.256 kB | 20.732 kB | 162.71/120.79/305.61/477.87 ms | 6.15 |
| 25% | 0.505348 | 60.80% | 40.591 kB | 49.291 kB | 199.01/217.98/306.07/408.19 ms | 5.02 |
| 40% | 0.530900 | 83.88% | 65.049 kB | 75.756 kB | 238.47/246.58/327.94/450.09 ms | 4.19 |

随着预算从 10% 提高到 40%，精度增加 0.0761 AP，而 E2E mean 从 162.71 增至 238.47 ms，体现了精度—时延权衡。10% 档有 69.84% 的输入保留 Edge，故 P50 仅为 120.79 ms；25% 与 40% 档的云覆盖率超过 50%，P50 相应升至 217.98 和 246.58 ms。P99 为 477.87/408.19/450.09 ms，说明均值不能代表尾部时延。

逐阶段均值分解进一步显示，JPEG 解码、RKNN 预处理/推理和端侧 NMS 构成的必经端侧阶段为 93.65/98.21/101.96 ms；候选构建与控制器为 16.63/17.21/17.43 ms；以全部 5,000 个输入为分母、未上传输入记 0 后，Cloud HTTP 的均值贡献为 42.56/72.15/108.82 ms；ROI 融合为 4.01/4.60/2.80 ms，其余清洗与执行开销为 5.87/6.84/7.47 ms。

三档 payload 利用率均不低于 99.64%。二进制请求头与 metadata 的额外上行为 225–229 B/上传输入，占上行 payload 不超过 0.43%；每输入检测结果下行为 4.407–10.518 kB。被选择图像上的 Q90 payload 约为对应原图的 52%–54%，ROI 为 22%–24%。作为参考端点，Edge-only 为 0.405264 AP、96.14 ms 和 10.40 等效 FPS，Always-Original 为 0.544817 AP、251.07 ms 和 3.98 等效 FPS。上述时延来自共享 RTX 4090 的端云系统观测，不应解释为独占 GPU 的纯模型基准。

### 7.9 控制器容量跟踪与代表性链路压力测试

图 6 展示四种代表性策略在 25% 预算下的链路压力测试，表 8 比较自适应与静态价格的容量跟踪。

![图 6 受控网络下的延迟与吞吐。](figures/fig_network_sensitivity.svg)

**图 6 25% 预算下的代表性链路压力测试。** 每个方法—链路单元均为一次固定顺序的 5,000 图像完整运行。原生链路的平均 ping 约为 5–7 ms；另两组条件分别同时设置为 20 ms/20 Mbps 和 50 ms/5 Mbps。在无丢包、云失败和 deadline fallback 且动作不读取网络状态时，链路条件不改变动作与 AP，只改变 E2E 和吞吐。由于 RTT 与带宽被同时改变，该实验比较的是两个代表性压力条件，不能分离二者的独立效应，也不提供运行间置信区间。

CAVS-OQR 在原生、20 ms/20 Mbps 和 50 ms/5 Mbps 下的 E2E mean 分别为 199.01、260.78 和 350.02 ms，P95 为 306.07、427.05 和 620.03 ms，P99 为 408.19、517.86 和 694.54 ms，吞吐为 4.73、3.67 和 2.77 FPS。在最受限条件下，mean 与 P95 相对原生链路分别增加 151.01 和 313.96 ms，而 AP 均为 0.505348。准确率不变是实验设计的直接结果，不表示 CAVS 对一般弱网天然鲁棒；若出现丢包、请求失败或 deadline-triggered fallback，动作完成率与准确率均可能变化。

**表 8 自适应价格相对静态价格（$\eta=0$）的容量跟踪与 AP 变化。** $\Delta$AP 为自适应减静态；两种控制器的最大前缀超额均为 0 B。

| 方法 | 预算 | 静态利用率 | 自适应利用率 | 最大前缀超额 | $\Delta$AP |
|---|---:|---:|---:|---:|---:|
| CAVS-OQ | 10% | 99.739% | 99.739% | 0 B | +0.000000 |
| CAVS-OQ | 25% | 99.114% | 99.412% | 0 B | −0.000339 |
| CAVS-OQ | 40% | 99.233% | 99.788% | 0 B | −0.000114 |
| CAVS-OQR | 10% | 99.767% | 99.767% | 0 B | +0.000000 |
| CAVS-OQR | 25% | 99.020% | 99.645% | 0 B | +0.000070 |
| CAVS-OQR | 40% | 97.665% | 99.804% | 0 B | +0.001153 |

表 8 直接支持自适应价格主要改善容量跟踪、而非普遍提高精度：OQR 40% 的容量利用率由 97.665% 提高至 99.804%，但各档 AP 变化均不超过 0.001153。

## 8 讨论与局限

### 8.1 证据范围

三数据集点估计结果、COCO 同价值输出的受控结构消融和 RK3588 端云实测共同支持一个一致但有限的结论：在本文受控配置下，当候选动作的价值与字节成本均随图像变化时，将多个图像—动作候选保留到容量求解阶段，可以优于先为每图固定动作再执行容量选择。VOC 的 SSDLite→Faster R-CNN 结果提供了超出 YOLO–YOLO 检测器对的支持性证据，但一组额外检测器对不足以证明跨架构的普遍适用性。本文的贡献是通信 payload 约束下的动作条件价值调度，而不是普遍更强的难度 Ranker，也不是对能耗、货币成本或云 GPU 占用的统一优化。

ROI 体现了同一框架中的真实边界：它在 COCO 在线系统中提供小幅增益，在 VOC 高预算下为负，在 Oregon 中未被执行。因而 OQR 是完整系统配置，OQ 是关键消融；结果不支持“ROI 跨域普遍有效”，也不支持依据数据集或预算事后选择不同动作空间并将其解释为统一固定方法。

### 8.2 离线—在线差距与部署权衡

部署域缓存参考与零前视控制之间观察到 0.024–0.029 AP 差距。在线控制器一般不能复现依赖未来输入的全信息 schedule；图 4 已控制 RKNN 输出域和部署对齐策略一致，但缓存物化、完整序列可见性、前缀控制与实时执行边界仍同时变化，因此不能把全部下降命名为纯粹的“在线调度损失”。未来可在独立时间序列上学习状态相关价格或直接优化 prefix-constrained reward，并用同策略、同候选的受控 replay 进一步分离信息结构与执行路径。本文的价格消融主要改善容量利用率而非 AP，说明仅通过在线价格调整尚不足以消除观察到的离线—在线性能差距。

CAVS 还承担 mandatory weak-detector inference overhead：每个输入在决策前均执行 RKNN。其平均 E2E 随预算由 162.71 增至 238.47 ms，在 50 ms/5 Mbps 下 25% 档达到 350.02 ms。这些结果表明，CAVS 更适用于通信预算受限且端侧预测本身具有可用价值的场景；在高预算、严格 deadline 或端侧推理昂贵时，应进一步研究跳过轻量检测器或按 deadline 回退的策略。本文没有实现这两类机制。

### 8.3 比较、公平性与外部有效性

ORIC/MORIC、DCSB 和 AF 均在统一检测输出、预算和硬件下重实现或适配，并非对原作者代码与实验协议的逐项复现。特别地，ORIC/MORIC 原问题以二元上传和 offloading ratio 为主，DCSB/AF 的原生阈值也不对应多动作 Fixed-byte；本文因此在第 6.2 节提前披露改动，并将原生工作点与预算匹配适配分开报告。CAVS-O/OQ/OQR 与 TwoStage 用于内部机制分析，但 O/OQ/OQR 的独立重训意味着相邻配置差值不是纯执行动作效应。

COCO val2017、Oregon evaluation 和 VOC2007 test 的结果均按冻结协议解释为事后评测，而非预注册确认集。正文只对完成 paired image bootstrap 的比较报告区间，其余差异按点估计解释；Random 同时报告多个预固定种子。进一步的确认性结论需要在未参与开发的时间后移数据或预先注册的独立测试集上验证。

主预算定义是 JPEG payload。二进制 HTTP 请求头与 metadata 的额外上行字节相对 JPEG payload 低于 0.43%，但下行检测框、链路层包头、能耗和云计算量不属于优化目标。真实系统实验仅覆盖 COCO、RK3588、共享 RTX 4090 和按 image ID 顺序播放的静态图像 trace；Oregon 与 VOC 提供算法层的跨域/跨架构评测，但未在板端重复整套系统矩阵。在线结果还依赖这一固定输入顺序，受控网络不包含丢包，静态图像序列也不刻画视频相关性、突发事件或概念漂移。以上限制约束了对其他设备、无线网络、能效和连续视频的外推。

## 9 结论

本文提出 CAVS，将端云目标检测从图像级二元选择扩展为面向异构候选动作的图像—动作价值调度。CAVS 以端侧检测集合预测与 AP50–95 对齐的动作条件价值，并将实际编码字节作为独立成本交由容量调度器处理，同时支持容量可行的全信息 Fixed-byte 调度和零前视逐图执行。三数据集结果提供了跨数据域和检测器配置的支持性证据；COCO 同价值输出的受控结构消融表明，将 Original 与 Q90 候选保留到容量调度阶段是观察到收益的重要来源。RK3588 端云测试床进一步验证了前缀 payload 预算可行性、真实通信成本，以及相对本文所评测重实现与适配基线的在线 AP 点估计优势。

结果也界定了适用范围：VOC 中 OQ 相对 O 的配置差异明显大于 OQR 相对 OQ 的进一步变化，但 Q90 仍只是单一压缩质量动作；ROI 的效果依赖数据域与控制条件；零前视控制一般不能复现依赖未来输入的全信息 schedule。在无丢包、无云失败、无 deadline fallback 且动作固定的网络压力实验中，链路恶化只改变时延和吞吐，这一结论不能外推到会触发回退的弱网。总体而言，这些结果支持在端云协同目标检测中，将动作条件价值预测与逐样本字节约束资源分配进行显式结合。

## Statements and Declarations

### Funding

【投稿前必须由作者确认】填写基金名称与项目编号；如无资助，改为期刊接受的正式无资助声明。

### Competing interests

【投稿前必须由作者确认】填写利益冲突声明；如无相关利益，使用期刊接受的正式无利益冲突声明。

### Author contributions

【投稿前必须由作者确认】按 CRediT taxonomy 填写作者贡献。

### Data availability

COCO 与 PASCAL VOC 按其公开许可获取。Oregon Critters 为作者持有的相机陷阱数据；【投稿前必须由作者确认】填写数据许可、可访问范围、申请流程及敏感位置信息处理方式。

### Code availability

【投稿前必须由作者确认】填写匿名代码仓库或归档 DOI。

### Ethics approval

【投稿前必须由作者确认】根据 Oregon 数据采集与许可填写“不适用”及依据，或填写机构伦理审批编号。

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

固定 $\lambda$ 后，每图可独立取式（10）的最大净值动作。扫描 $\lambda$ 得到离散动作集合的 supported value–cost frontier；整数非凸点可能不是任何单一 $\lambda$ 的直接解，因此容量修复提高利用率但不消除一般最优性间隙。

## 附录 B 冻结超参数与复现检查

**表 B1 CAVS 冻结模型、训练与控制参数。**

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
| 五折划分 / 独立完整训练次数 | 5 / 3 |
| 数据划分与训练随机种子 | 评测前固定；具体值随复现配置发布 |
| Context 抽样随机性 | 按第 4.2 节的确定性哈希规则固定 |
| 全信息 $\lambda$ 区间 / 二分次数 | 第 5.1 节数据依赖上界 / 80 |
| 在线 $\lambda$ 候选 / 投影区间 | 0:0.01:2，2.5，3，4，5 / [0,5] |
| 在线 $\eta$ 候选 | 0，0.00025，0.0005，0.001，0.0015，0.002，0.003，0.005，0.01，0.02 |
| OQ 在线 $(\lambda_1,\eta)$，10/25/40% | (0.47,0) / (0.30,0.001) / (0.17,0.0015) |
| OQR 在线 $(\lambda_1,\eta)$，10/25/40% | (0.47,0) / (0.30,0.0015) / (0.17,0.002) |
| Batch size / gradient clip | 192 / 5 |

**表 B2 CAVS-OQR 三个独立训练模型与集成模型的 AP50–95。** “均值±SD”表示三次独立训练结果的均值与样本标准差，不是图像样本标准差或显著性区间。

| 数据集 | 预算 | 三训练 seed 均值±SD | Ensemble | Ensemble−均值 |
|---|---:|---:|---:|---:|
| COCO | 10% | 0.479239±0.001553 | 0.480277 | +0.001038 |
| COCO | 25% | 0.532415±0.000482 | 0.532900 | +0.000485 |
| COCO | 40% | 0.560046±0.000274 | 0.560178 | +0.000132 |
| Oregon | 10% | 0.508762±0.006335 | 0.506028 | −0.002734 |
| Oregon | 25% | 0.514188±0.005350 | 0.517405 | +0.003217 |
| Oregon | 40% | 0.514986±0.006277 | 0.516663 | +0.001676 |
| VOC | 10% | 0.529939±0.001413 | 0.531486 | +0.001547 |
| VOC | 25% | 0.577284±0.000914 | 0.579678 | +0.002395 |
| VOC | 40% | 0.612286±0.001957 | 0.612721 | +0.000435 |

Oregon 10% 的 ensemble 低于三个单模型均值，说明集成不是保证单调提高 AP 的机制；本文固定使用预先定义的三模型平均分数，而不按 evaluation 结果选择单 seed 或 ensemble。

**表 B3 DCSB 与 AF 的离线原生阈值工作点。**

| 数据集 | 方法 | AP50–95 | 云调用率 | Payload/Always-Original | 上传图像 |
|---|---|---:|---:|---:|---:|
| COCO | AF-Native | 0.560217 | 74.020% | 74.241% | 3,701 |
| COCO | DCSB-Native | 0.562011 | 76.140% | 76.108% | 3,807 |
| Oregon | AF-Native | 0.458955 | 2.882% | 3.495% | 143 |
| Oregon | DCSB-Native | 0.475223 | 11.205% | 11.865% | 556 |
| VOC | AF-Native | 0.597841 | 49.758% | 51.478% | 2,464 |
| VOC | DCSB-Native | 0.568991 | 44.527% | 45.358% | 2,205 |

表 B3 保留各方法在 calibration 上冻结后得到的自然阈值工作点，不把它们解释为 10%/25%/40% 预算结果。AF 在 Oregon 的原生阈值过于保守，AP 仅略高于 Edge-only 0.454815；VOC 的两个原生阈值则分别消耗 45.358% 和 51.478% 的 Always-Original 字节。这些结果共同说明，原生阈值跨数据域并不自动对应相同通信成本。

**表 B4 Adapted DCSB 与 Adapted AF 的零前视 RK3588 端云系统结果。**

| 方法 | 工作点 | AP50–95 | 上传输入 | Payload MB | 字节率 | E2E mean/P95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adapted DCSB | 10% | 0.432997 | 831 | 81.192 | 9.966% | 119.20 / 228.26 | 7.69 |
| Adapted DCSB | 25% | 0.464606 | 1,686 | 203.357 | 24.961% | 141.08 / 255.66 | 6.56 |
| Adapted DCSB | 40% | 0.490678 | 2,424 | 325.622 | 39.968% | 163.88 / 276.82 | 5.70 |
| Adapted AF | 10% | 0.433734 | 810 | 81.396 | 9.991% | 117.98 / 230.74 | 7.77 |
| Adapted AF | 25% | 0.465847 | 1,636 | 203.479 | 24.976% | 143.78 / 267.49 | 6.45 |
| Adapted AF | 40% | 0.493665 | 2,375 | 325.725 | 39.981% | 162.05 / 273.61 | 5.76 |
| DCSB-Native | 原生 | 0.534972 | 3,831 | 622.075 | 76.356% | 200.91 / 289.67 | 4.69 |
| AF-Native | 原生 | 0.533136 | 3,890 | 636.873 | 78.172% | 205.06 / 289.01 | 4.60 |

表 B4 的字节率以 814.705 MB Always-Original payload 为分母。8 组均为 5,000/5,000、云失败 0、非法框 0；六个预算匹配组容量利用率为 99.658%–99.952%，且前缀预算违规为 0。两个 Native 行是自然阈值工作点，没有被赋予 10%/25%/40% 前缀预算，因而不报告该项违规统计。E2E 与 FPS 来自共享 RTX 4090 条件下的完整串行运行，不替代独占 GPU 微基准。

**表 B5 三数据集 Fixed-byte AP50–95 完整点结果。** DCSB 与 AF 为 Original-only 预算匹配适配。

| 数据集 / 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **COCO — CAVS-OQR** | **0.480277** | **0.532900** | **0.560178** |
| COCO — MORIC | 0.452230 | 0.496914 | 0.528170 |
| COCO — ORIC | 0.449174 | 0.494970 | 0.527069 |
| COCO — Confidence | 0.447029 | 0.489546 | 0.520311 |
| COCO — Adapted DCSB | 0.442595 | 0.478828 | 0.510588 |
| COCO — Adapted AF | 0.443563 | 0.477221 | 0.510021 |
| COCO — Random-50 | 0.433586 | 0.468908 | 0.500443 |
| **Oregon — CAVS-OQR** | **0.506028** | **0.517405** | **0.516663** |
| Oregon — MORIC | 0.479826 | 0.493286 | 0.506171 |
| Oregon — ORIC | 0.483153 | 0.502061 | 0.509599 |
| Oregon — Confidence | 0.489643 | 0.502499 | 0.510449 |
| Oregon — Adapted DCSB | 0.468816 | 0.485255 | 0.487301 |
| Oregon — Adapted AF | 0.465760 | 0.484632 | 0.494235 |
| Oregon — Random-50 | 0.471087 | 0.484147 | 0.495973 |
| **VOC — CAVS-OQR** | **0.531486** | **0.579678** | **0.612721** |
| VOC — MORIC | 0.504416 | 0.554956 | 0.592212 |
| VOC — ORIC | 0.502039 | 0.553636 | 0.589734 |
| VOC — Confidence | 0.495439 | 0.532334 | 0.569091 |
| VOC — Adapted DCSB | 0.494798 | 0.528534 | 0.553232 |
| VOC — Adapted AF | 0.502711 | 0.542576 | 0.573726 |
| VOC — Random-50 | 0.482501 | 0.503851 | 0.532850 |

**表 B6 CAVS-OQR E2E 逐阶段 mean/P95（全部输入图像，ms）。** 未上传输入的 Cloud HTTP round-trip 和非 ROI 输入的融合记为 0；各阶段 P95 来自不同输入，不能相加。

| 阶段 | 10% | 25% | 40% |
|---|---:|---:|---:|
| JPEG 解码 | 6.85 / 17.38 | 7.41 / 17.63 | 7.78 / 17.60 |
| Edge 预处理 | 0.54 / 0.99 | 0.55 / 0.99 | 0.58 / 1.03 |
| RKNN-FP16 | 65.47 / 78.55 | 67.47 / 83.31 | 68.31 / 84.68 |
| Edge 解码与 NMS | 20.79 / 45.58 | 22.78 / 50.31 | 25.29 / 55.80 |
| 候选构建、编码与策略评分 | 15.17 / 23.63 | 15.54 / 24.92 | 15.81 / 25.42 |
| 控制器与决策余项 | 1.46 / 2.45 | 1.67 / 2.66 | 1.62 / 2.63 |
| Cloud HTTP round-trip | 42.56 / 148.53 | 72.15 / 131.93 | 108.82 / 161.28 |
| ROI + MAUR | 4.01 / 26.27 | 4.60 / 28.94 | 2.80 / 7.81 |
| 结果清洗与未单列执行余项 | 5.87 / 14.34 | 6.84 / 15.72 | 7.47 / 15.76 |
| **总 E2E** | **162.71 / 305.61** | **199.01 / 306.07** | **238.47 / 327.94** |

**表 B7 CAVS-OQR 逐图通信与动作 payload。** 动作均值以被该动作选中的图像为条件，跨动作均值不可直接作为压缩率比较；kB 与 MB 均采用第 6.2 节定义的十进制单位。

| 指标 | 10% | 25% | 40% |
|---|---:|---:|---:|
| Payload/输入图 | 16.256 kB | 40.591 kB | 65.049 kB |
| HTTP 上行/输入图 | 16.325 kB | 40.729 kB | 65.238 kB |
| 下行/输入图 | 4.407 kB | 8.563 kB | 10.518 kB |
| 双向/输入图 | 20.732 kB | 49.291 kB | 75.756 kB |
| Payload/上传图 | 53.900 kB | 66.761 kB | 77.550 kB |
| Original/被选图 | 45.232 kB | 55.202 kB | 80.936 kB |
| Q90/被选图 | 63.286 kB | 73.187 kB | 80.415 kB |
| Q90/对应原图 | 52.07% | 53.73% | 53.23% |
| ROI/被选图 | 31.870 kB | 38.868 kB | 44.041 kB |
| ROI/对应原图 | 21.75% | 23.67% | 24.20% |

**表 B8 COCO 零前视系统的软件与模型环境。**

| 组件 | 冻结配置 |
|---|---|
| 板端硬件 | Orange Pi 5 Ultra，RK3588，NPU core 0 |
| 板端系统 | Orange Pi 1.0.0 / Ubuntu 22.04.4 LTS，kernel 6.1.43-rockchip-rk3588，Python 3.10.12 |
| 端侧检测器 | YOLO26n，640，RKNN-FP16，SHA-256 `0915a992…de12d` |
| RKNN 栈 | Toolkit Lite 2.3.2，runtime 2.3.0，driver 0.9.6 |
| 板端 JPEG 编码 | Pillow 12.3.0，libjpeg-turbo 3.1.4.1（JPEG ABI 6.2） |
| 云端硬件 | 共享 NVIDIA RTX 4090 |
| 云端软件 | Python 3.11.5，PyTorch 2.1.2，CUDA 11.8，cuDNN 8.7，Ultralytics 8.4.10 |
| 云端检测器 | YOLO26x，640，SHA-256 `9fdd44a3…03e92` |
| 在线 trace | COCO val2017，按 image ID 升序，5,000 个顺序图像输入 |

### 补充复现清单

复现材料应包含以下可审计对象：

1. 三个数据集的角色清单、image/event ID 及其 SHA-256；COCO 在线顺序按 image ID 升序固定，sequence SHA-256 为 `aa36e57fb97291562937947c9cebe221c330a0f8da89d5fc865cd8eee66d058b`。

2. COCO 端侧 PyTorch、RKNN-FP16 和云端 checkpoint。在线 RKNN SHA-256 为 `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`，云模型为 `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`。

3. Oregon 端/云 checkpoint；对应 SHA-256 分别为 `b975f9f8035ddeb7c3f66e0ffa62d4030d40ec6c8c9d75c08ba517312f0a2c4b` 与 `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`。

4. Original/Q90/ROI 编码参数、Current/Historical profile、MAUR/Reversible 融合实现，以及各角色的动作预测与 payload 长度清单。

5. 五折划分、冻结 context 表、三个完整训练模型、标准化统计、在线控制参数、配对 bootstrap 随机种子和 Random 随机种子列表。

6. 每个在线输入的动作、预测价值、payload/HTTP 字节、前缀容量、端/云输出和阶段时延，以及云服务的软件栈与模型哈希。主系统的软件环境见第 5.3 节；检测阈值与动作协议见第 6.2 节。

## 参考文献

[1] Kang Y, Hauswald J, Gao C, et al (2017) Neurosurgeon: collaborative intelligence between the cloud and mobile edge. In: Proceedings of ASPLOS, pp 615–629. https://doi.org/10.1145/3037697.3037698

[2] Matsubara Y, Levorato M, Restuccia F (2023) Split computing and early exiting for deep learning applications: survey and research challenges. ACM Comput Surv 55(5):1–30. https://doi.org/10.1145/3527155

[3] Zhou Z, Chen X, Li E, et al (2019) Edge intelligence: paving the last mile of artificial intelligence with edge computing. Proc IEEE 107(8):1738–1762. https://doi.org/10.1109/JPROC.2019.2918951

[4] Teerapittayanon S, McDanel B, Kung HT (2016) BranchyNet: fast inference via early exiting from deep neural networks. In: Proceedings of ICPR, pp 2464–2469. https://doi.org/10.1109/ICPR.2016.7900006

[5] Kang D, Emmons J, Abuzaid F, Bailis P, Zaharia M (2017) NoScope: optimizing neural network queries over video at scale. Proc VLDB Endow 10(11):1586–1597. https://doi.org/10.14778/3137628.3137664

[6] Li Y, Padmanabhan A, Zhao P, et al (2020) Reducto: on-camera filtering for resource-efficient real-time video analytics. In: Proceedings of ACM SIGCOMM, pp 359–376. https://doi.org/10.1145/3387514.3405874

[7] Jiang J, Ananthanarayanan G, Bodik P, Sen S, Stoica I (2018) Chameleon: scalable adaptation of video analytics. In: Proceedings of ACM SIGCOMM, pp 253–266. https://doi.org/10.1145/3230543.3230574

[8] Cao Z, Li Z, Chen Y, Pan H, Hu Y, Liu J (2023) Edge-cloud collaborated object detection via difficult-case discriminator. In: Proceedings of IEEE ICDCS, pp 259–270. https://doi.org/10.1109/ICDCS57875.2023.00062

[9] Cao Z, Cheng Y, Zhou Z, et al (2025) Edge-cloud collaborated object detection via bandwidth adaptive difficult-case discriminator. IEEE Trans Mob Comput 24(2):1181–1196. https://doi.org/10.1109/TMC.2024.3474743

[10] Gomez C, Patkhullaev D, Hernandez AC (2024) OffloaD: detection failure-based scheduler for offloading object detection. ACM Trans Embed Comput Syst 23(5):1–26. https://doi.org/10.1145/3677321

[11] Qiu J, Wang R, Hu B, Guérin R, Lu C (2024) Optimizing edge offloading decisions for object detection. In: Proceedings of the ACM/IEEE Symposium on Edge Computing. https://doi.org/10.1109/SEC62691.2024.00021

[12] Du K, Liu Y, Hao Y, Zhang Q, Wang H, Huang Y, Ananthanarayanan G, Jiang J (2023) OneAdapt: fast adaptation for deep learning applications via backpropagation. In: Proceedings of the ACM Symposium on Cloud Computing, pp 158–176. https://doi.org/10.1145/3620678.3624653

[13] Du K, Zhang Q, Arapin A, et al (2022) AccMPEG: optimizing video encoding for accurate video analytics. Proc Mach Learn Syst 4. https://arxiv.org/abs/2204.12534

[14] Yang Z, Wang X, Wu J, et al (2023) EdgeDuet: tiling small object detection for edge assisted autonomous mobile vision. IEEE/ACM Trans Netw 31(4):1765–1778. https://doi.org/10.1109/TNET.2022.3223412

[15] Chaudhary S, Taneja A, Singh A, et al (2024) TileClipper: lightweight selection of regions of interest from videos for traffic surveillance. In: Proceedings of USENIX ATC, pp 967–984. https://www.usenix.org/conference/atc24/presentation/chaudhary

[16] Wang W, Mi L, Cen S, et al (2025) Region-based content enhancement for efficient video analytics at the edge. In: Proceedings of USENIX NSDI, pp 613–633. https://www.usenix.org/conference/nsdi25/presentation/wang-weijun

[17] Chen T, Bu Y, Zeng Y, Xie L, Lu S (2024) RegionFilter: region-aware video filtering mechanism on resource-constrained edge nodes. Comput Netw 251:110624. https://doi.org/10.1016/j.comnet.2024.110624

[18] Wang H, Li Q, Sun H, et al (2023) VaBUS: edge-cloud real-time video analytics via background understanding and subtraction. IEEE J Sel Areas Commun 41(1):90–106. https://doi.org/10.1109/JSAC.2022.3221995

[19] Dehghani D, Chen Y, Yu X, Koudas N (2025) Selective cloud offloading for accurate and efficient object detection. In: Proceedings of IEEE ICDM, pp 1145–1154. https://doi.org/10.1109/ICDM65498.2025.00123

[20] Mozannar H, Sontag D (2020) Consistent estimators for learning to defer to an expert. In: Proceedings of ICML, PMLR 119:7076–7087. https://proceedings.mlr.press/v119/mozannar20b.html

[21] Verma R, Nalisnick E (2022) Calibrated learning to defer with one-vs-all classifiers. In: Proceedings of ICML, PMLR 162. https://proceedings.mlr.press/v162/verma22c.html

[22] Mao A, Mohri M, Zhong Y (2025) Mastering multiple-expert routing: realizable H-consistency and strong guarantees for learning to defer. In: Proceedings of ICML, PMLR 267:43035–43066. https://proceedings.mlr.press/v267/mao25c.html

[23] Chen L, Zaharia M, Zou J (2020) FrugalML: how to use ML prediction APIs more accurately and cheaply. Adv Neural Inf Process Syst 33. https://arxiv.org/abs/2006.07512

[24] Elmachtoub AN, Grigas P (2022) Smart “predict, then optimize”. Manag Sci 68(1):9–26. https://doi.org/10.1287/mnsc.2020.3922

[25] Chakrabarti A, Guérin R, Lu C, Liu J (2021) Real-time edge classification: optimal offloading under token bucket constraints. In: Proceedings of the ACM/IEEE Symposium on Edge Computing, pp 41–54. https://doi.org/10.1145/3453142.3492329

[26] Geng W, Mohan N, Ott J (2026) Budget-adaptive routing: skipping the weak when the strong answers anyway. In: Proceedings of the ACM SIGCOMM NAIC Workshop. https://doi.org/10.1145/3789240.3828740

[27] Jitkrittum W, Gupta N, Menon AK, Narasimhan H, Rawat AS, Kumar S (2023) When does confidence-based cascade deferral suffice? Adv Neural Inf Process Syst 36:9891–9906

[28] Guo C, Pleiss G, Sun Y, Weinberger KQ (2017) On calibration of modern neural networks. In: Proceedings of ICML, PMLR 70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html

[29] Efron B, Tibshirani RJ (1993) An introduction to the bootstrap. Chapman & Hall/CRC, Boca Raton

[30] Lin TY, Maire M, Belongie S, et al (2014) Microsoft COCO: common objects in context. In: Proceedings of ECCV, pp 740–755. https://doi.org/10.1007/978-3-319-10602-1_48

[31] Everingham M, Eslami SMA, Van Gool L, Williams CKI, Winn J, Zisserman A (2015) The PASCAL visual object classes challenge: a retrospective. Int J Comput Vis 111:98–136. https://doi.org/10.1007/s11263-014-0733-5

[32] Paszke A, Gross S, Massa F, et al (2019) PyTorch: an imperative style, high-performance deep learning library. Adv Neural Inf Process Syst 32

[33] Ultralytics (2026) Ultralytics YOLO26 documentation. https://docs.ultralytics.com/models/yolo26. Accessed 30 Aug 2026

[34] Orange Pi (2026) Orange Pi 5 Ultra hardware specifications. https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-Ultra.html. Accessed 27 Aug 2026

[35] Xu R, Razavi S, Zheng R (2023) Edge video analytics: a survey on applications, systems and enabling techniques. IEEE Commun Surv Tutor 25(4):2951–2982. https://doi.org/10.1109/COMST.2023.3323091

[36] Hu Y, Li Z, Chen Y, et al (2023) Content-aware adaptive device-cloud collaborative inference for object detection. IEEE Internet Things J 10(21):19087–19101. https://doi.org/10.1109/JIOT.2023.3279579

[37] Said N, Landsiedel O (2025) EdgeBoost: confidence boosting for resource constrained inference via selective offloading. Comput Netw 269:111437. https://doi.org/10.1016/j.comnet.2025.111437

[38] Schafhalter P, Krentsel A, Wei H, et al (2026) TURBO: utility-aware bandwidth allocation for cloud-augmented autonomous control. OASIcs New Ideas Netw Syst 139:18:1–18:34. https://doi.org/10.4230/OASIcs.NINeS.2026.18

[39] Hawlader F, Robinet F, Frank R (2024) Leveraging the edge and cloud for V2X-based real-time object detection in autonomous driving. Comput Commun 213:372–381. https://doi.org/10.1016/j.comcom.2023.11.025

[40] Zhou HY, Gao BB, Wu J (2017) Adaptive feeding: achieving fast and accurate detections by adaptively combining object detectors. In: Proceedings of ICCV, pp 3505–3513. https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Adaptive_Feeding_Achieving_ICCV_2017_paper.html

[41] PyTorch Contributors (2025) SSDLite320 MobileNetV3 Large—Torchvision 0.21 model documentation. https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html. Accessed 30 Aug 2026

[42] PyTorch Contributors (2025) Faster R-CNN ResNet-50 FPN V2—Torchvision 0.21 model documentation. https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html. Accessed 30 Aug 2026

[43] Howard A, Sandler M, Chu G, et al (2019) Searching for MobileNetV3. In: Proceedings of ICCV, pp 1314–1324. https://doi.org/10.1109/ICCV.2019.00140

[44] Ren S, He K, Girshick R, Sun J (2017) Faster R-CNN: towards real-time object detection with region proposal networks. IEEE Trans Pattern Anal Mach Intell 39(6):1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031

[45] Rockchip (2025) RKNN Toolkit2 and RKNN Toolkit Lite2. https://github.com/airockchip/rknn-toolkit2. Accessed 30 Aug 2026
