# CAVS：面向成本感知端云协同目标检测的上下文图像—动作价值调度

**CAVS: Contextual Image–Action Value Scheduling for Cost-Aware Edge–Cloud Object Detection**

[作者姓名]¹，[作者姓名]²，[通讯作者]¹*

¹[单位、城市、邮编、国家]

²[单位、城市、邮编、国家]

*[通讯作者邮箱]

## 摘要

端云协同目标检测常将资源分配简化为“是否上传图像”，但真实系统还需在原始图像、压缩 JPEG 和感兴趣区域（region of interest，ROI）等异构载荷之间选择。本文提出上下文图像—动作价值调度（Contextual Image–Action Value Scheduling，CAVS），将决策单元由图像扩展为图像—载荷对。CAVS 以端侧检测集合的置换不变表示预测动作条件价值，训练奖励与 IoU 阈值 0.50–0.95 上平均的 COCO-style AP（AP50–95）对齐；实际编码字节不进入价值网络，而由全信息离线调度或零前视在线控制处理。在冻结的事后评测协议下，以 Always-Original JPEG payload 的 10%、25% 和 40% 为预算，CAVS 在 COCO、Oregon 野生动物和 PASCAL VOC 上的九个点估计均优于图像级路由。相较协议匹配的 MORIC，COCO 和 VOC 分别提高 0.0280–0.0360 和 0.0205–0.0271 AP；COCO 受控同骨干消融显示，联合图像—动作估值较“两阶段图像排序—载荷分配”提高 0.0279–0.0361 AP。在 RK3588–GPU 零前视端云测试床上，CAVS 在三档预算下相较各档最强外部基线分别提高 0.0163、0.0369 和 0.0372 AP，所有前缀 payload-byte 约束均满足，平均端到端时延为 162.7–238.5 ms。结果表明，动作条件价值学习与逐样本字节计量能够改善精度—通信权衡，同时揭示了全信息调度与因果在线执行之间的部署差距。

**关键词：** 端云协同；目标检测；选择性卸载；资源分配；边缘计算

## Abstract

Selective edge–cloud object detection is often reduced to deciding whether an image should be offloaded. A deployable system must additionally choose among heterogeneous payloads, such as the original image, compressed JPEG, and a region of interest. We propose Contextual Image–Action Value Scheduling (CAVS), which treats each image–payload pair as a decision unit. A permutation-invariant encoder maps edge detections to action-conditioned values trained with a reward aligned to COCO-style average precision over intersection-over-union thresholds from 0.50 to 0.95 (AP50–95). Explicit encoded-byte costs are excluded from the value network and handled by either a full-information feasible scheduler or a zero-lookahead online controller. Under frozen post-hoc evaluation protocols and budgets of 10%, 25%, and 40% of the Always-Original JPEG payload, CAVS yields the highest point estimate in all nine dataset–budget settings on COCO, Oregon wildlife, and PASCAL VOC. It improves over the protocol-matched MORIC baseline by 0.0280–0.0360 AP on COCO and 0.0205–0.0271 AP on VOC. A controlled same-backbone COCO ablation indicates a 0.0279–0.0361 AP gain over two-stage image ranking and payload assignment. On an RK3588–GPU zero-lookahead testbed, CAVS exceeds the strongest evaluated external baseline by 0.0163, 0.0369, and 0.0372 AP while satisfying every prefix payload-byte constraint; measured mean end-to-end latency is 162.7–238.5 ms. These results support joint image–action valuation while quantifying the remaining gap between full-information scheduling and causal deployment.

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

3. 在 COCO、Oregon 与跨检测器架构的 VOC 配置上进行冻结协议评测，并在 RK3588–GPU 测试床上验证真实逐图在线执行。COCO 受控同骨干消融支持联合图像—动作估值的作用；在线三档预算相较各档最强外部基线分别提高 0.0163、0.0369 和 0.0372 AP，同时满足全部前缀 payload-byte 约束。

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

图 1 给出 CAVS 的端到端信息流。端侧检测既提供未上传时的结果，也为价值模型提供语义输入；真实编码字节只进入调度器。离线训练使用标注生成奖励，部署阶段不访问标注。

![图 1 CAVS 系统与信息流。](figures/fig_cavs_system_overview.svg)

**图 1 CAVS 系统与信息流。** 全信息离线调度产生容量可行参考；零前视控制器在不读取未来输入的条件下逐图决策并立即执行。

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

其中 $e$ 表示保留端侧结果，$o$、$q$ 和 $r$ 分别表示 Original、Q90 和 ROI。三种动作配置严格定义为 $\mathrm O=\{e,o\}$、$\mathrm{OQ}=\{e,o,q\}$ 和 $\mathrm{OQR}=\{e,o,q,r\}$。ROI 动作不是多个独立 crop：系统把合格端侧框扩张后合并为一个外接裁剪框，并将该 crop 编码成一个 JPEG payload。因此，每幅图像在每种配置下仍只选择一个动作。

对云动作 $a\ne e$，图像编码模块产生载荷 $P_i^a$，其主预算成本为编码后 JPEG 长度 $c_i^a=|P_i^a|$。云检测经坐标恢复与动作特定融合得到 $F_i^a$；端侧动作满足 $F_i^e=E_i$、$c_i^e=0$。若请求失败，已发送的 payload 仍计入累计成本，检测结果回退为 $F_i^e$；正式在线主矩阵未出现云请求失败。本文从问题定义起即只以 JPEG payload 为优化预算，并将 HTTP 上行、下行检测结果和协议开销作为独立系统指标报告。

### 3.2 全信息 Fixed-byte 问题

令 $\hat v_i^a$ 为动作 $a$ 相对 Edge 的预测任务价值，$x_i^a\in\{0,1\}$ 为动作选择变量。全信息 Fixed-byte 决策写为

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

式（5）不允许借用未来字节，因此比仅约束序列末端总量更严格。在线结果对输入顺序敏感；正式 COCO trace 按升序 image ID 固定，具体清单和哈希见附录 B。

## 4 CAVS：上下文图像—动作价值学习

### 4.1 检测集合与动作表示

CAVS 由检测集合编码器、动作条件价值网络和外置成本调度器组成。每幅图像保留置信度最高的 $M=64$ 个检测；框特征包括归一化中心、宽、高、面积和置信度，类别使用可学习嵌入。共享多层感知机（multilayer perceptron，MLP）编码每个检测后，以均值池化和最大池化获得置换不变表示：

\[
h_i^{set}=
\begin{cases}
\left[\dfrac{1}{n_i}\sum_j h_{ij}\;\middle\|\;\max_j h_{ij}\right], & n_i>0,\\[3pt]
\mathbf 0, & n_i=0.
\end{cases}
\tag{6}
\]

场景特征 $z_i$ 包含图像宽高比、检测框数量、最大/平均置信度、框总覆盖率、亮度均值/标准差和边缘密度。动作描述 $u_i^a$ 包含动作类型以及 payload 无关的几何信息；对 ROI，几何量为归一化 crop 宽、高、面积和宽高比。输入构造阶段显式置零原始 payload 对数、动作编码字节和 payload ratio 等成本坐标。网络先在训练折标准化空间预测价值，再反标准化：

\[
\hat{\bar v}_i^a=f_\theta(h_i^{set},z_i,\operatorname{StripCost}(u_i^a)),
\qquad
\hat v_i^a=\sigma_{tr}\hat{\bar v}_i^a+\mu_{tr}.
\tag{7}
\]

端侧动作的价值固定为 0，三个云动作共享参数。显式编码字节 $c_i^a$ 不进入价值网络，只在调度阶段使用；这一设计减少了价值预测对直接成本字段的依赖，但不否认目标尺度、ROI 几何等语义特征可能与 payload 大小相关。

### 4.2 上下文图像—动作奖励

ORIC/MORIC 在图像上下文中定义 mAP 对齐奖励 [11]。CAVS 将被替换对象从二元弱/强检测推广为动作 $a$。对训练图像 $i$，从同一训练折构造不含 $i$ 的上下文 $\mathcal C_i$，并定义

\[
v_i^a=(|\mathcal C_i|+1)\left[
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^a)-
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^e)
\right].
\tag{8}
\]

正式协议取 $|\mathcal C_i|=32$。上下文从目标图像所在训练折内、排除目标图像后无放回抽取；抽样由 fold、目标 image ID 和固定 context seed 的 SHA-256 哈希确定。每个目标图像的全部动作共享同一上下文，不同训练 seed 也复用同一冻结上下文表，从而避免动作差值混入 context sampling noise。上下文与标注只用于离线监督生成；部署仅使用端侧预测。该奖励仍是完整数据集 AP 的局部代理，因此模型选择和最终比较均以完整 evaluator 为准，而非奖励回归误差。

### 4.3 回归—排序联合训练

令 $\bar v_i^a$ 为训练折内标准化奖励。CAVS 同时最小化 Smooth L1 回归误差和同图动作排序误差：

\[
\mathcal L=\alpha\frac{1}{|\Omega|}\sum_{(i,a)\in\Omega}
\operatorname{SmoothL1}(\hat{\bar v}_i^a,\bar v_i^a)
+(1-\alpha)\frac{\sum_{(i,a,b)\in\Pi}w_i^{ab}
\log\!\left(1+e^{-s_i^{ab}(\hat{\bar v}_i^a-\hat{\bar v}_i^b)}\right)}
{\sum_{(i,a,b)\in\Pi}w_i^{ab}},
\tag{9}
\]

其中 $s_i^{ab}=\operatorname{sgn}(\bar v_i^a-\bar v_i^b)$、$w_i^{ab}=|\bar v_i^a-\bar v_i^b|$，正式配置取 $\alpha=0.5$。五折 out-of-fold（OOF）预测用于开发阶段的稳定性与机制诊断；最终学习器不集成 OOF 模型，而是以三个预先固定的种子在完整 fit 上独立重训，再平均反标准化后的价值。完整超参数和种子列于附录 B。

### 4.4 ROI profile 与融合规则

主配置 Current@640 首先保留端侧置信度不低于 0.25 的框。按单框面积占原图比例 $s$ 扩张：$s<1\%$ 时使用 1.8 倍边长，$1\%\le s<10\%$ 时使用 1.6 倍，$s\ge10\%$ 时使用 1.4 倍；扩张框裁剪到图像边界后合并为一个外接矩形。若合并区域超过原图面积的 65%，该图像的 ROI 动作不可用；否则 crop 以 Q90 编码并送入 640 输入的云检测器。本文将这一确定性几何规则称为 Current@640 profile。

本文定义 masked-area update-and-replacement（MAUR）融合：先将云端 crop 预测映射回原图；删除中心点落在 crop 内的端侧框，并删除与云框同类且交并比（intersection over union，IoU）不低于 0.5 的重复端侧框；随后合并其余端侧框与云框，并执行类内 IoU 阈值 0.7 的非极大值抑制（non-maximum suppression，NMS）。框中心恰落在裁剪边界时按区内处理。Current@640 只产生一个合并 crop，因此不存在多个 ROI 之间的融合顺序问题。

2×2 消融还使用两个对照。Historical-cost@640 在 $s<2\%$、$2\%\le s<4\%$、$4\%\le s<8\%$ 和 $s\ge8\%$ 时分别采用 4.0、3.0、2.5 和 1.4 倍扩张。Reversible fusion 以同类 IoU 不低于 0.3 贪心匹配端/云框；匹配时云置信度不低于端侧置信度才替换，未匹配端侧框保留，未匹配云框仅在置信度不低于 0.1 时加入，最后执行类内 NMS 0.7。

ROI 是可选动作而非强制模块：若其预测净值低于 Original、Q90 或 Edge，调度器可以完全不选 ROI。O、OQ 和 OQR 是三种独立训练的动作空间配置；各配置均重新生成 reward 并重训，因此 OQ−O 与 OQR−OQ 反映“增加动作并重新学习后的配置差异”，而不是只改变执行动作的严格单变量因果效应。

## 5 调度与系统实现

### 5.1 全局容量可行调度

给定影子价格 $\lambda$，每幅图像独立选择

\[
a_i(\lambda)=\arg\max_{a\in\mathcal A_i}
\left(\hat v_i^a-\lambda c_i^a\right).
\tag{10}
\]

实现以 $\lambda_L=0$ 和 $\lambda_U=1+\max_{i,a:c_i^a>0}[\max(0,\hat v_i^a)/c_i^a]$ 为搜索区间，执行 80 次二分并保留容量可行一侧。每图净值并列时依次选择成本更低、动作 ID 字典序更小的动作。随后枚举所有正价值增量升级，并按“边际价值/增量字节、增量价值、较小增量成本、动作 ID、图像索引”的固定顺序尝试加入，直至候选耗尽；不可行升级被跳过。该拉格朗日—修复过程保证最终总 payload 不超过 $B$，但不保证一般多选背包的精确最优，因此全文仅将其称为 full-information feasible reference。

### 5.2 零前视在线控制

在线控制器不预先评分完整序列。当前输入到达后，系统构造候选载荷和价值，先屏蔽会违反式（5）的动作，再依据当前价格选择

\[
a_t=\arg\max_{a\in\mathcal A_t^{feasible}}
\left(\hat v_t^a-\lambda_t\frac{c_t^a}{m_o}\right),
\qquad
\lambda_{t+1}=\Pi_{[0,5]}\!\left[
\lambda_t+\eta\frac{c_t^{a_t}-\rho c_t^o}{m_o}
\right],
\tag{11}
\]

其中 $m_o$ 为 calibration 上 Original payload 的中位数，$\Pi$ 表示投影。初始价格在 $[0,5]$、步长为 0.01 的网格上选择；对每个候选价格，再从预设步长集合中选择使 calibration 预测动作价值总和最大的前缀可行轨迹，并以更低累计字节和更小价格作为并列规则。CAVS-OQR 在 10%/25%/40% 的冻结 $(\lambda_1,\eta)$ 分别为 $(0.47,0)$、$(0.30,0.0015)$ 和 $(0.17,0.002)$；静态价格消融令 $\eta=0$。动作执行和云端融合完成后才读取下一幅输入。

### 5.3 RK3588–云硬件链路

板端使用 Orange Pi 5 Ultra/RK3588 [34,45]，操作系统为 Orange Pi 1.0.0（Ubuntu 22.04.4 LTS），kernel 为 6.1.43-rockchip-rk3588，Python 为 3.10.12。端侧模型是输入 640 的 YOLO26n RKNN-FP16 导出版本；正式日志记录 RKNN Toolkit Lite 2.3.2、runtime 2.3.0 和 driver 0.9.6。云端为共享 RTX 4090 上的 YOLO26x PyTorch checkpoint，服务环境为 Python 3.11.5、PyTorch 2.1.2、CUDA 11.8、cuDNN 8.7 和 Ultralytics 8.4.10。每幅输入依次完成读取、端侧检测、动作决策、可选编码与上传、云检测、坐标恢复和融合。

系统使用紧凑二进制 HTTP 请求体承载定长头、规范化 metadata 和原始 JPEG bytes，不使用 Base64。模型 checkpoint、输入顺序、协议版本、策略及控制参数均以 SHA-256 固定，以保证不同方法共享同一执行环境。服务端和客户端均过滤非有限、越界或零面积框。系统分别记录算法预算使用的 JPEG payload、实际 HTTP 上下行字节、各阶段时延和完整端到端（end-to-end，E2E）时延。

### 5.4 复杂度与高预算边界

集合编码至多处理 64 个框；全信息调度复杂度近似为 $O(TN|\mathcal A|+U\log U)$，其中 $T$ 为价格搜索次数、$U$ 为候选升级数；在线动作选择为 $O(|\mathcal A|)$。由于 CAVS 在决定上传前始终运行端侧检测器，高预算下仍存在 mandatory weak-detector inference overhead。本文在结果部分报告完整阶段时延，但不以共享 GPU 测量替代独占模型基准，也不对未测量的能耗和峰值内存作推断。

## 6 实验方法

### 6.1 数据集与检测器

**COCO。** COCO 2017 [30] 的一个预先物化、含 12,000 幅 train2017 图像的路由器候选池按 image ID 的 SHA-256 哈希确定性划分（seed=2026）：哈希分位小于 0.80 的 9,678 幅图像作为 fit，0.80–0.85 的 571 幅图像作为 calibration；其余 development roles 不用于本文主结果。评测使用 val2017 的 5,000 幅图像。fit、calibration 和 evaluation 的 image-ID 清单及 SHA-256 摘要随复现材料发布。端、云模型分别为冻结的 YOLO26n 和 YOLO26x [33]，checkpoint SHA-256 分别为 `9b09…4fef` 与 `9fdd…3e92`。

**Oregon。** Oregon Critters 是作者持有的 45 类相机陷阱野生动物集合，共 99,231 幅图像；其许可、访问方式和敏感位置信息处理仍需由作者在投稿前填写，本文不将其表述为公开数据集。事件按标准化相机 ID 构造：相邻图像间隔不超过 5 min、单事件持续不超过 30 min 且最多包含 50 幅图像；划分在相机事件层面互斥。本文使用 selector fit=9,923、calibration=4,961 和 evaluation=4,962。端模型为 YOLO26n（输入 640，冻结 epoch-46 checkpoint，SHA-256 `b975f9f8…2c4b`）；云模型为 YOLO26x（输入 640，冻结 epoch-28 checkpoint，SHA-256 `d3890a57…2d9`），其 detector-validation AP50–95/AP50 为 0.54282/0.74694。该云模型来自作者的域内微调链；完整训练 recipe 与 checkpoint 需随代码归档发布，内部名称“Cloud-V3”不作为可复现模型定义。

**PASCAL VOC。** VOC [31] 由 VOC2007+2012 trainval 构造 fit=15,629 和 calibration=922，使用 VOC2007 test 的 4,952 幅图像评测。端模型为 Torchvision 0.21.0 的 `ssdlite320_mobilenet_v3_large`，权重枚举为 `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`；云模型为 `fasterrcnn_resnet50_fpn_v2`，权重枚举为 `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` [41–44]。两种检测器均采用官方固定权重，不在 VOC 上训练，从而提供一组 YOLO 以外的跨架构证据。

三个 evaluation split 均在方法开发过程中被访问，因此本文把结果限定为冻结协议下的事后评测，而非未见的确认性测试。fit 用于学习器训练，calibration 用于冻结控制器及预先声明的超参数；evaluation 图像不进入相应冻结运行的学习器拟合或阈值选择，但曾被先前探索访问。全文统一称 evaluation split，避免暗示其为从未访问的确认性测试集。

### 6.2 指标、编码与比较方法

主指标为 COCO-style AP50–95。Original 直接复用数据集原始 JPEG bitstream；Q90 以 Pillow 固定 `quality=90`、`optimize=True` 和 `subsampling=2`；ROI 使用第 4.4 节定义的 Current@640 与 MAUR。检测导出使用置信度阈值 0.001、IoU 0.7 和每图最多 300 个框。所有方法共享相应数据集的检测输出、字节表、Always-Original 容量分母和 evaluator。在线系统另报告上传输入数、JPEG payload、HTTP 上下行、E2E 时延和串行 pipeline 吞吐。

比较器包括 Edge-only、Always-Original、Random、Confidence、DCSB [8]、AF [40] 以及 ORIC/MORIC [11]。ORIC 重实现保留同折 context reward、MLP 估计器和加权均方误差；MORIC 保留 $\mathrm{MORIC}_i=\mathrm{CDF}(\mathrm{ORIC}_i)$ 的归一化。为适配统一 Fixed-byte 问题，二者在比较表中执行 Edge/Original 二元动作并使用本文共同的容量控制，因此属于 protocol-matched reimplementations，而非官方逐位复现。

DCSB 保留原论文的置信度、目标数和最小目标面积比例三阈值规则。AF 使用 Top-K 类别直方图、候选框统计和线性支持向量机，但以本文冻结端侧检测器的 proposals 替代原文 proposal 生成器。二者的原生 calibration 阈值工作点单独报告；Fixed-byte 行使用其难例 margin 排序并执行 Original，全文统一标为 Adapted DCSB 与 Adapted AF。TwoStage 与 CAVS 共用 top-64 检测集合骨干、数据划分和训练超参数，但先预测图像标量，再由 SharedAction 规则分配 O/Q 载荷，用于受控比较两阶段与联合图像—动作建模。CAVS-O、OQ 和 OQR 均按各自动作空间独立生成奖励并训练。

### 6.3 训练与统计

每个 CAVS 配置生成五折 OOF 预测用于开发诊断，并以三个预先固定的种子在完整 fit 上重训后平均价值分数。COCO 的 full-fit seeds 为 20260809/20260810/20260811，Oregon 为 20260810/20260811/20260812，VOC 为 20260825/20260826/20260827；fold seeds 与其余超参数见附录 B。单模型结果报告三个训练 seed 间的 mean±sample SD，作为描述性稳定性指标而非置信区间。

离线 Random 报告 50 个预先指定随机种子的 AP 均值；在线 Random 报告五个预先指定种子的均值和 seed 间 SD。主表采用冻结 evaluation split 上的点估计。表 5 的 Profile×Fusion 效应使用 1,000 次 paired image bootstrap [29]：每次按 image ID 有放回重采样并在同一重采样集上评价配对方法，报告差值的百分位 95% 区间。只有区间排除 0 时使用显著性措辞；无区间的配置差异明确标为点估计。

### 6.4 端云测试床协议

COCO 系统实验包含两种信息结构。全信息 batch execution 先获得完整序列候选，再执行第 5.1 节的容量可行调度，用作真实 RKNN/HTTP 执行下的离线参考；zero-lookahead online execution 严格按第 3.3 节逐图决策并立即上传。正式在线比较覆盖 CAVS-OQR、CAVS-OQ、ORIC、MORIC、Confidence、五个 Random 种子、Adapted DCSB 和 Adapted AF 的 10%/25%/40% 工作点，并另测 Edge-only、Always-Original、静态价格消融和受控网络条件。

全部在线方法使用同一 Orange Pi、同一共享 RTX 4090、相同模型 checkpoint、二进制协议和按 image ID 升序固定的 COCO trace，且串行运行以避免 NPU/GPU 并发污染时延。所有正文主结果均完成 5,000 个顺序图像输入，云失败、非法框和前缀 payload 超额均为 0。网络敏感性实验在无丢包、无云失败、无 deadline fallback 且动作策略不读取网络状态的条件下施加 RTT/带宽限制；因此它检验 transport stress，而不是 network-state-adaptive routing。

## 7 结果与分析

### 7.1 三数据集 Fixed-byte 主结果

图 2 汇总三组端—云检测器上的冻结点结果。正文以预先固定的 CAVS-OQR 为主方法；每个数据集均重新生成动作奖励并训练路由器，因此这里检验的是算法流程的跨数据集适用性，而不是 checkpoint 的零样本迁移。

![图 2 三数据集 Fixed-byte 结果。](figures/fig_fixed_byte_three_datasets.svg)

**图 2 三数据集 Fixed-byte AP50–95。** 图中绘制覆盖全部三数据集的 CAVS-OQR、MORIC、Confidence 和 Random；所有方法共享所在数据集的检测输出、容量分母和 evaluator。完整数值及其余比较器见表 B5。

**表 1 CAVS-OQR 与每个数据集最强外部比较器的 AP50–95。** 单元格内数值依次对应 10%/25%/40% 预算；$\Delta$AP 为 CAVS-OQR 减比较器。

| 数据集 / 比较器 | CAVS-OQR | 比较器 AP | $\Delta$AP |
|---|---|---|---|
| COCO / MORIC | **0.4803 / 0.5329 / 0.5602** | 0.4522 / 0.4969 / 0.5282 | **+0.0280 / +0.0360 / +0.0320** |
| Oregon / Confidence | **0.5060 / 0.5174 / 0.5167** | 0.4896 / 0.5025 / 0.5104 | **+0.0164 / +0.0149 / +0.0062** |
| VOC / MORIC | **0.5315 / 0.5797 / 0.6127** | 0.5044 / 0.5550 / 0.5922 | **+0.0271 / +0.0247 / +0.0205** |

CAVS 在九个数据集—预算组合中均取得最高点估计，相对各数据集最强比较器提高 0.0062–0.0360 AP。COCO 与 VOC 的点结果随预算单调提高；Oregon 从 25% 的 0.517405 降至 40% 的 0.516663，下降 0.000742 AP，表明该冻结价值代理未在更高预算下继续改善非可加指标。DCSB 与 AF 的原生阈值不对应固定字节容量，故正文不把其自然工作点混入表 1；预算匹配适配和原生工作点分别列于表 B5 与表 B3。由于这些主结果没有逐项 paired interval，本文只使用“最高点估计”，不使用“显著优于”。

### 7.2 COCO：核心机制和容量使用

表 2 显示 CAVS 不把字节预算退化为调用率预算：压缩与 ROI 使 10% 字节容量覆盖 28.18% 的图像，40% 时覆盖 83.04%，且三档容量利用率均超过 99.997%。

**表 2 COCO CAVS-OQR 调度组成与容量使用。**

| 预算 | Edge | Original | Q90 | ROI | 云覆盖率 | 容量利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 3,591 | 29 | 1,076 | 304 | 28.18% | 99.9976% |
| 25% | 2,049 | 63 | 2,468 | 420 | 59.02% | 99.9985% |
| 40% | 848 | 209 | 3,651 | 292 | 83.04% | 99.9996% |

低预算主要利用 Q90 和 ROI 扩大云覆盖，高预算则增加 Original；这一动作迁移与预算增加时影子价格下降的机制预期一致，但表 2 本身不是对该机制的独立因果证明。

### 7.3 联合图像—动作估值的受控结构消融

TwoStage 与 CAVS-OQ 使用相同的 top-64 集合编码、优化器、fold、训练种子、O/Q reward 和 Fixed-byte 求解器。主要受控差异是输出目标与载荷建模：TwoStage 先输出图像标量，再由 SharedAction 规则选择载荷；CAVS 直接预测每个图像—动作价值。由于两者的预测目标与输出结构不同，该比较支持联合估值机制，但不被表述为只改变一个标量的严格因果实验。

**表 3 COCO 受控同骨干结构消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| TwoStage：图像排序 → SharedAction | 0.450374 | 0.492738 | 0.524704 |
| CAVS-OQ：Joint image–action | **0.478224** | **0.528884** | **0.559286** |
| 差值 | **+0.027850** | **+0.036146** | **+0.034582** |

在骨干、输入和训练预算保持一致的条件下，CAVS-OQ 的点结果比 TwoStage 高 0.0279–0.0361 AP，支持“联合图像—动作估值优于先排序后分配”的核心机制。该结论限于 COCO 的受控配置，不外推为三个数据集均完成了相同结构消融。

### 7.4 O/OQ/OQR 动作空间消融

VOC 提供完整的三档动作空间比较：O、OQ、OQR 均按各自 reward 生成五折 OOF 诊断，并分别以三个 full-fit seeds 重训。端点 AP 为 Edge 0.473322、Original 0.657511、Q90 0.653684、ROI+MAUR 0.499819。

![图 3 VOC 动作空间消融。](figures/fig_voc_action_space_ablation.svg)

**图 3 VOC 动作空间消融。** （a）O、OQ 与 OQR 配置的绝对 AP50–95；（b）独立重训配置之间的点估计差值。OQR−OQ 在 40% 为负，说明增加候选动作并重新训练不保证改善最终 AP。

**表 4 VOC 跨架构动作空间消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-O | 0.520249 | 0.557280 | 0.589427 |
| CAVS-OQ | 0.530787 | 0.579142 | **0.614054** |
| CAVS-OQR | **0.531486** | **0.579678** | 0.612721 |
| OQ − O | +0.010538 | +0.021862 | +0.024627 |
| OQR − OQ | +0.000699 | +0.000537 | −0.001333 |

OQ 配置相对 O 配置高 0.0105–0.0246 AP，占观察到的动作空间收益的大部分；OQR 相对 OQ 的配置差值仅为 +0.0007、+0.0005 和 −0.0013 AP。相应地，OQR 仅选择 77/49/3 个 ROI，而 Q90 动作为 796/1,816/2,796 个。由于各配置独立生成 reward 并重训，这些差值不能解释为仅执行 Q90 或 ROI 的纯因果增益。

Oregon 的 OQR 三档均未执行 ROI；OQ 与 OQR 的差异来自独立训练形成的 O/Q 动作分配，而非 ROI 执行收益。因此本文只主张 ROI 可参与成本竞争，不主张它跨数据集普遍增益。

### 7.5 COCO Profile × Fusion 2×2

为分离 ROI 几何与融合规则，我们在同一 COCO 划分、云模型、fold、种子和调度器下，对 Current@640/Historical-cost@640 与 Reversible/MAUR 四组各自重建真实 reward 并全量重训。

**表 5 COCO Profile × Fusion 独立全量重训（AP50–95）。**

| ROI profile | Fusion | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| Current@640 | Reversible | 0.479071 | 0.531817 | 0.559932 |
| Current@640 | MAUR | **0.480277** | **0.532900** | 0.560178 |
| Historical-cost@640 | Reversible | 0.477784 | 0.531402 | 0.560025 |
| Historical-cost@640 | MAUR | 0.478155 | 0.531983 | **0.560701** |

Current+MAUR 是预先固定的主配置，但四格差异很小。Current 下 MAUR−Reversible 的 95% CI 在 10%、25% 和 40% 均跨 0（分别为 [−0.001916,+0.003460]、[−0.000061,+0.002212] 和 [−0.000607,+0.000946]）。区间由 1,000 次 paired image bootstrap 得到。因此，结果不支持“MAUR 全面显著更优”，也不允许按预算事后挑选四格最大值。

### 7.6 从离线调度到因果在线

![图 4 COCO 离线到在线的精度差距。](figures/fig_coco_offline_to_online_gap.svg)

**图 4 COCO 从缓存协议到零前视部署的描述性精度阶梯。** （a）论文缓存、部署域缓存、RK3588 全信息 batch execution 与正式因果在线结果；（b）部署域缓存到正式在线的 AP50–95 差值。不同阶梯不完全同协议，因此该图用于展示部署差距，不作为单因素因果分解。

RK3588 全信息 batch execution 先观察完整序列再调度，CAVS-OQR 为 0.4611/0.5036/0.5303 AP；它是真实 RKNN 与 HTTP 执行下的离线参考，而不是在线性能声明。部署域缓存参考为 0.4791/0.5311/0.5602，正式在线相对其低 0.0243–0.0293 AP。该差距共同包含 RKNN 检测输出域、全信息到零前视前缀控制的信息结构、ROI 成员变化以及离线/在线软件栈差异。本文没有在同一正式硬件协议内独立随机化这些因素，因而不把差值归因于单一“网络传输损失”；三档正式在线运行的云失败、非法框和前缀超额均为 0，提示请求失败本身不是主要解释。

### 7.7 零前视在线准确率

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

CAVS-OQR 在三档均取得最高在线点估计。每档最强外部基线分别为 10%/25% 下的 ORIC 和 40% 下的 Adapted AF，对应增益为 +0.016278、+0.036928 和 +0.037235 AP，即 0.0163–0.0372 AP。CAVS-OQR 相对独立训练的 CAVS-OQ 高 +0.002939/+0.005038/+0.003183 AP；这些是配置点差，不是带置信区间的纯 ROI 因果效应。所有正文在线方法均完成 5,000 个顺序输入，且云失败、非法框和最大前缀超额均为 0。DCSB 与 AF 的原生阈值分别消耗 76.36% 和 78.17% 的 Always-Original payload，故只在表 B4 单列，不与固定预算结果混写。

### 7.8 在线时延与通信成本

![图 5 正式在线系统精度、E2E 与通信成本。](figures/fig_online_system_results.svg)

**图 5 正式在线系统结果。** （a）八种策略的 AP50–95；（b）CAVS-OQR 在全部输入图像上的 E2E mean 逐阶段堆叠及总 P95；（c）每输入图像的 JPEG payload、协议额外上行和检测结果下行。图（a）中的 DCSB 与 AF 为 Original-only 预算匹配适配；阶段 P95 来自不同输入，不能相加。

**表 7 CAVS-OQR 零前视在线系统结果。** E2E 和吞吐为整条串行 pipeline 的测量值。

| 预算 | AP50–95 | 云覆盖率 | Payload/图 | HTTP 双向/图 | E2E mean/P95 | FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.454831 | 30.16% | 16.256 kB | 20.732 kB | 162.71/305.61 ms | 5.76 |
| 25% | 0.505348 | 60.80% | 40.591 kB | 49.291 kB | 199.01/306.07 ms | 4.73 |
| 40% | 0.530900 | 83.88% | 65.049 kB | 75.756 kB | 238.47/327.94 ms | 3.99 |

随着预算提高，精度增加 0.0761 AP，而平均 E2E 从 162.71 增至 238.47 ms，体现了精度—时延权衡。端侧 JPEG 解码、RKNN 预处理/推理和 NMS 合计约 94–102 ms。Cloud HTTP round-trip latency 包含请求传输、服务器校验/JPEG 解码、云推理和响应返回；表 7 的均值以全部 5,000 个输入为分母，未上传输入在该阶段记为 0，因此其均值贡献由 42.56 增至 108.82 ms。三档 payload 利用率均不低于 99.64%。二进制请求头与 metadata 的额外**上行**字节为 225–229 B/上传输入，占上行 payload 不超过 0.43%；这不包括每输入 4.407–10.518 kB 的下行检测结果。Q90 payload 约为其对应原图的 52%–54%，ROI 约为 22%–24%；完整阶段和动作成本列于表 B6–B7。作为参考端点，Edge-only 为 0.405264 AP、96.14 ms 和 9.39 FPS，Always-Original 为 0.544817 AP、251.07 ms 和 3.80 FPS。上述时延来自共享 RTX 4090 的端云系统观测，不应解释为独占 GPU 的纯模型基准。

### 7.9 控制器与网络敏感性

![图 6 受控网络下的延迟与吞吐。](figures/fig_network_sensitivity.svg)

**图 6 25% 预算下的 transport sensitivity。** 在无丢包、云失败和 deadline fallback 且动作不读取网络状态时，受控 RTT/带宽不改变动作与 AP，只改变 E2E 和吞吐；原生 ping 是 Orange Pi 到实验云服务器的往返测量，均值约为 5–7 ms。

在 50 ms/5 Mbps 下，CAVS-OQR 的平均 E2E 相对原生网络增加 151.01 ms，P95 增加 313.96 ms，吞吐由 4.73 降至 2.77 FPS；AP 仍为 0.505348。准确率不变是实验设计的直接结果，而不是 CAVS 对弱网天然鲁棒：动作固定、传输无丢包、云请求无失败且没有 deadline-triggered fallback。若这些条件改变，准确率也可能随链路状态变化。

**表 8 自适应价格相对静态 $\lambda$ 的容量跟踪与 AP 变化。** $\Delta$AP 为自适应减静态；两种控制器的最大前缀超额均为 0 B。

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

三数据集点结果、COCO 受控同骨干消融和 RK3588–GPU 实测共同支持一个一致但有限的结论：当候选载荷的价值与字节成本均随图像变化时，联合估计图像—动作价值可以优于先做图像排序再分配载荷。VOC 的 SSDLite→Faster R-CNN 结果提供了超出 YOLO–YOLO 检测器对的证据，但一组额外检测器对不足以证明跨架构的普遍适用性。本文的贡献是通信 payload 约束下的动作条件价值调度，而不是普遍更强的难度 Ranker，也不是对能耗、货币成本或云 GPU 占用的统一优化。

ROI 体现了同一框架中的真实边界：它在 COCO 在线系统中提供小幅增益，在 VOC 高预算下为负，在 Oregon 中未被执行。因而 OQR 是完整系统配置，OQ 是关键消融；结果不支持“ROI 跨域普遍有效”，也不支持按数据集或预算事后挑选不同动作空间组成虚构的统一方法。

### 8.2 离线—在线差距与部署权衡

部署域缓存参考与零前视控制之间观察到 0.024–0.029 AP 差距。在线控制器一般不能复现依赖未来输入的全信息 schedule；同时，图 4 的差值还包含 RKNN 输出域和软件栈差异，因此不能把全部下降命名为“在线调度损失”。未来可在独立时间序列上学习状态相关价格或直接优化 prefix-constrained reward，并用同硬件、同候选的受控 replay 分离检测域与信息结构。本文的价格消融主要改善容量利用率而非 AP，说明重复扫描单一静态价格不足以消除该差距。

CAVS 还承担 mandatory weak-detector inference overhead：每个输入在决策前均执行 RKNN。其平均 E2E 随预算由 162.71 增至 238.47 ms，在 50 ms/5 Mbps 下 25% 档达到 350.02 ms。因此，CAVS 适合通信受限且端侧预测可直接复用的场景；在高预算、严格 deadline 或端侧推理昂贵时，应进一步研究跳过轻量检测器或按 deadline 回退的策略。本文没有实现这两类机制。

### 8.3 比较、公平性与外部有效性

ORIC/MORIC、DCSB 和 AF 均在统一检测输出、预算和硬件下重实现或适配，并非作者代码的逐位官方复现。特别地，ORIC/MORIC 原问题以二元上传和 offloading ratio 为主，DCSB/AF 的原生阈值也不对应多动作 Fixed-byte；本文因此在第 6.2 节提前披露改动，并将原生工作点与预算匹配适配分开报告。CAVS-O/OQ/OQR 与 TwoStage 用于内部机制分析，但 O/OQ/OQR 的独立重训意味着相邻配置差值不是纯执行动作效应。

COCO val2017、Oregon evaluation 和 VOC2007 test 在开发过程中被访问，因此结果应理解为冻结协议下的事后完整评测，而非未见确认集。正文只对完成 paired image bootstrap 的比较报告区间，其余差异按点估计解释；Random 同时报告多个预固定种子。进一步的确认性主张需要时间后移数据或新注册测试集。

主预算定义是 JPEG payload。二进制 HTTP 的额外上行已实测低于 0.43%，但下行检测框、链路层包头、能耗和云计算量不属于优化目标。真实系统实验仅覆盖 COCO、RK3588、共享 RTX 4090 和按 image ID 顺序播放的静态图像 trace；Oregon 与 VOC 提供算法层的跨域/跨架构评测，但未在板端重复整套系统矩阵。在线结果还依赖这一固定输入顺序，受控网络不包含丢包，静态图像序列也不刻画视频相关性、突发事件或概念漂移。以上限制约束了对其他设备、无线网络、能效和连续视频的外推。

## 9 结论

本文提出 CAVS，将端云目标检测从图像级二元选择扩展为异构像素载荷下的联合图像—动作价值调度。CAVS 以端侧检测集合预测与 AP50–95 对齐的动作条件价值，将显式编码字节交由资源控制，并同时支持容量可行的全信息 Fixed-byte 调度和零前视逐图执行。三数据集点结果提供跨域证据；COCO 受控同骨干消融表明，联合图像—动作估值是观察到收益的重要来源。RK3588–GPU 测试床进一步验证了前缀 payload 预算可行性、真实通信成本，以及相对本文所评测重实现/适配基线的在线点估计优势。

结果也界定了适用范围：VOC 中 OQ 配置占观察到的动作空间收益的大部分，但 Q90 只是单一压缩质量动作；ROI 的效果依赖数据域与控制条件；零前视控制一般不能复现依赖未来输入的全信息 schedule。在无丢包、无云失败、无 deadline fallback 且动作固定的网络压力实验中，链路恶化只改变时延和吞吐，这一结论不能外推到会触发回退的弱网。总体而言，当载荷价值与传输字节均为样本相关时，在动作层连接任务价值预测与资源分配是一条有证据支持的端云路由设计路径。

## Statements and Declarations

### Funding

[由作者填写基金名称与项目编号；如无专项资助，填写相应声明。]

### Competing interests

[由作者填写。]

### Author contributions

[由作者按 CRediT taxonomy 填写。]

### Data availability

COCO 与 PASCAL VOC 按其公开许可获取。Oregon Critters 为作者持有的相机陷阱数据；[由作者在投稿前填写数据许可、可访问范围、申请流程及敏感位置信息处理方式。]

### Code availability

[由作者填写匿名代码仓库或归档 DOI。]

### Ethics approval

[由作者根据 Oregon 数据采集与许可确认“不适用”或填写机构伦理审批编号。]

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
| OOF folds / full-fit seeds 数 | 5 / 3 |
| COCO fold seed / full-fit seeds | 20260806 / 20260809, 20260810, 20260811 |
| Oregon fold seed / full-fit seeds | 20260810 / 20260810, 20260811, 20260812 |
| VOC fold seed / full-fit seeds | 20260825 / 20260825, 20260826, 20260827 |
| Context 抽样 seed | 各数据集 fold seed |
| 全信息 $\lambda$ 区间 / 二分次数 | 第 5.1 节数据依赖上界 / 80 |
| 在线 $\lambda$ 网格 / 投影区间 | 0:0.01:5 / [0,5] |
| OQR 在线 $(\lambda_1,\eta)$，10/25/40% | (0.47,0) / (0.30,0.0015) / (0.17,0.002) |
| Batch size / gradient clip | 192 / 5 |

**表 B2 CAVS-OQR 三个 full-fit 单模型与 ensemble 的 AP50–95。** “均值±SD”表示三个训练 seed 间的 mean±sample SD，不是图像样本标准差或显著性区间。

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

表 B3 保留各方法在 calibration 上冻结后得到的自然阈值工作点，不把它们解释为 10%/25%/40% 预算结果。AF 在 Oregon 的原生阈值过于保守，AP 仅略高于 Edge-only 0.454815；这也说明原生阈值跨数据域并不自动对应相同通信成本。VOC 未在协议冻结前完成可审计的 DCSB/AF 原生与预算匹配实现，故明确记为 not evaluated，不以静默缺行替代结果。

**表 B4 Adapted DCSB 与 Adapted AF 的零前视 RK3588–GPU 系统结果。**

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

表 B4 的字节率以 814.705 MB Always-Original payload 为分母。六个预算匹配组容量利用率为 99.658%–99.952%；所有 8 组均为 5,000/5,000、云失败 0、非法框 0 和前缀预算违规 0。E2E 与 FPS 来自共享 RTX 4090 条件下的完整串行运行，不替代独占 GPU 微基准。

**表 B5 三数据集 Fixed-byte AP50–95 完整点结果。** DCSB 与 AF 为 Original-only 预算匹配适配；VOC 的 DCSB/AF 未评测。

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
| 清洗、提交与日志余项 | 5.87 / 14.34 | 6.84 / 15.72 | 7.47 / 15.76 |
| **总 E2E** | **162.71 / 305.61** | **199.01 / 306.07** | **238.47 / 327.94** |

**表 B7 CAVS-OQR 逐图通信与动作 payload。** 动作均值以被该动作选中的图像为条件，跨动作均值不可直接作为压缩率比较。

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

5. 五折 membership、冻结 context 表、三个 full-fit checkpoints、标准化统计、在线控制参数、paired bootstrap seed（20260812）和 Random seed 列表。

6. 每个在线输入的动作、预测价值、payload/HTTP 字节、前缀容量、端/云输出和阶段时延，以及云服务的软件栈与模型哈希。主系统的软件环境见第 5.3 节；检测阈值与动作协议见第 6.2 节。

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

[40] Zhou HY, Gao BB, Wu J. Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors. *IEEE International Conference on Computer Vision (ICCV)*. 2017:3505–3513. https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Adaptive_Feeding_Achieving_ICCV_2017_paper.html

[41] PyTorch Contributors. SSDLite320 MobileNetV3 Large—Torchvision 0.21 model documentation. 2025. https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html. Accessed 30 Aug 2026.

[42] PyTorch Contributors. Faster R-CNN ResNet-50 FPN V2—Torchvision 0.21 model documentation. 2025. https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html. Accessed 30 Aug 2026.

[43] Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2019:1314–1324. https://doi.org/10.1109/ICCV.2019.00140

[44] Ren S, He K, Girshick R, Sun J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2017;39(6):1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031

[45] Rockchip. RKNN Toolkit2 and RKNN Toolkit Lite2. 2025. https://github.com/airockchip/rknn-toolkit2. Accessed 30 Aug 2026.
