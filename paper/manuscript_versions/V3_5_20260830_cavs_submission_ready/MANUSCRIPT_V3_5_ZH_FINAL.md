# CAVS：面向成本感知端云协同目标检测的上下文图像—动作价值调度

**CAVS: Contextual Image–Action Value Scheduling for Cost-Aware Edge–Cloud Object Detection**

[作者姓名]¹，[作者姓名]²，[通讯作者]¹*

¹[单位、城市、邮编、国家]

²[单位、城市、邮编、国家]

*[通讯作者邮箱]

## 摘要

端云协同目标检测通常只判断图像是否上传，但多质量编码与感兴趣区域使系统还必须决定上传何种像素载荷。本文提出上下文图像—动作价值调度器 CAVS，将决策单元由“图像”推广为“图像—载荷”。CAVS 以端侧检测集合的置换不变表示预测与 AP50–95 对齐的动作条件边际价值，并将实际编码字节留给全局严格容量调度器或零前视因果控制器处理。在全量原图字节的 10%、25% 和 40% 预算下，COCO、Oregon 野生动物和 PASCAL VOC 的实验均显示 CAVS 优于图像级路由：相较同协议 MORIC，COCO 和 VOC 分别提高 0.0280–0.0360 与 0.0205–0.0271 AP；Oregon 相较每档最强比较器提高 0.0062–0.0164 AP。同骨干消融进一步表明，图像—动作联合估值较“先排序图像、再分配载荷”提高 0.0279–0.0361 AP。54 组 Orange Pi 5 Ultra/RK3588–RTX 零前视运行中，CAVS 比最强在线外部基线提高 0.0163–0.0387 AP，所有前缀字节约束均得到满足，平均端到端时延为 162.7–238.5 ms。结果表明，动作条件价值学习与逐样本字节计量能够改善精度—通信前沿，同时也揭示了全局容量上限转化为因果在线策略时的可测部署损失。

**关键词：** 端云计算；目标检测；选择性卸载；资源分配；嵌入式系统

## Abstract

Selective edge–cloud object detection typically decides whether to offload an image, but deployable systems must also choose which pixel payload to transmit. We present CAVS, a Contextual Image–Action Value Scheduler that treats each image–payload pair as a decision unit. A permutation-invariant encoder maps edge detections to action-conditioned marginal values aligned with average precision over intersection-over-union thresholds from 0.50 to 0.95 (AP). Actual encoded bytes remain outside the value model and are handled by either a global hard-capacity scheduler or a zero-lookahead causal controller. At 10%, 25%, and 40% of the Always-Original byte volume, experiments on Common Objects in Context (COCO), Oregon wildlife, and PASCAL Visual Object Classes (VOC) show consistent gains over image-level routing. CAVS improves over the protocol-matched contextual-reward baseline by 0.0280–0.0360 AP on COCO and 0.0205–0.0271 AP on VOC, and exceeds the strongest comparator on Oregon by 0.0062–0.0164 AP. A controlled same-backbone ablation attributes 0.0279–0.0361 AP to joint image–action valuation rather than image ranking followed by payload assignment. In 54 zero-lookahead Orange Pi 5 Ultra/RK3588–RTX runs, CAVS surpasses the strongest online external baseline by 0.0163–0.0387 AP while satisfying every prefix byte constraint; mean end-to-end latency is 162.7–238.5 ms. These results demonstrate that action-conditioned value learning and per-sample byte accounting improve the accuracy–communication frontier, while exposing a measurable gap between global scheduling and causal deployment.

**Keywords:** edge computing; object detection; selective offloading; resource allocation; embedded systems

## 1 引言

轻量检测器使智能相机、野外监测节点和移动机器人能够在本地持续处理图像，但受模型容量、功耗和端侧算子支持限制，其检测质量通常低于云端大模型。全部上传可获得较高精度，却持续消耗上行带宽，并引入编码、传输、云推理和结果回传开销。因此，端云协同检测的核心是在有限通信资源下选择哪些输入需要云端增强 [1–7,35]。

Adaptive Feeding、DCSB 和 OffloaD 分别以易难分类、难例判别和失败风险选择强检测器 [8–10,40]；ORIC/MORIC 进一步指出，目标检测的平均精度是跨图像、跨类别聚合的非可加指标，单图置信度并不等价于该图像对最终指标的贡献 [11]。这些方法主要解决“是否上传”，并通常以调用比例约束一个 Edge/Cloud 二元动作。

真实链路还必须决定“上传什么”。同一图像可保留端侧结果，也可上传原始 JPEG、质量为 90 的 JPEG，或由端侧检测框构造的感兴趣区域。不同载荷的收益和字节数均随图像变化：压缩图能够容纳更多请求但可能损失细节；区域上传能够放大目标但可能丢失上下文。在这种异构动作空间中，先用动作无关分数排序图像、再为入选图像分配载荷，会割裂相互耦合的成员选择与载荷选择。

为此，本文提出上下文图像—动作价值调度器 CAVS。CAVS 将端侧检测集合编码为紧凑语义表示，并为每个候选载荷预测相对端侧执行的上下文 AP 边际价值。价值网络不读取实际载荷大小；调度器再依据逐图实测字节，在给定容量内联合选择图像与载荷。该设计将“检测价值”与“通信价格”分离，使同一学习器能够适配不同字节预算。

离线全局调度与在线部署具有不同信息结构。前者观察一个批次的全部价值和成本，用于衡量严格总容量下的算法上限；后者在每帧到达后立即决策，不访问未来输入，并需保证任意前缀均不超预算。本文同时报告两种口径，避免把真实硬件传输误写为因果在线决策。

本文的主要贡献如下：

1. 提出带逐图实测字节的图像—动作建模，将 Edge、Original、Q90 和可选 ROI 统一为多选容量问题，并明确 Fixed-call、全局 Fixed-byte 与零前视在线预算的关系。

2. 提出 AP 对齐且价值—成本解耦的 CAVS，把上下文图像奖励推广为上下文图像—动作奖励，以统一模型联合决定上传成员和像素载荷。

3. 在 COCO、Oregon 和跨检测器架构的 VOC 配置上验证方法，并完成 54 组 RK3588–RTX 零前视运行。离线比较中 CAVS 相较同协议上下文奖励基线最多提高 0.0360 AP；在线比较中相较最强外部基线最多提高 0.0387 AP，同时满足全部前缀字节约束。

## 2 相关工作

### 2.1 检测级选择性卸载

Neurosurgeon 和 split computing 研究神经网络在端、边、云之间的切分 [1–4]；NoScope、Reducto 和 Chameleon 等视频系统则利用专用过滤器或时间冗余降低处理成本 [5–7]。本文假设端侧检测结果始终可独立使用，关注检测完成后的样本级升级，而非网络切分或视频帧过滤。

与本文更接近的是快/慢检测器级联。Adaptive Feeding 以候选框统计训练线性分类器，在两个检测器之间选择 [40]；DCSB 和 DCCI 学习难例并结合区域采样或设备配置 [8,9,36]；OffloaD、EdgeBoost 与 V2X 端云检测分别从失败风险、置信度增强和车辆网络时延出发进行选择性卸载 [10,37,39]。ORIC/MORIC 用上下文替换估计单幅图像对整体 mAP 的边际贡献，在多个检测器对和真实 Jetson–GPU 测试台上优于置信度、随机与难例基线 [11]。CAVS 继承其 AP 对齐思想，但把 Edge/Original 二元选择推广到成本不同的多像素载荷。

### 2.2 任务感知载荷与系统配置

任务感知编码和区域传输通过减少无关像素来节省通信。AccMPEG 依据任务梯度调整宏块质量 [13]；EdgeDuet、TileClipper、RegenHance、RegionFilter 和 VaBUS 使用切片、区域增强或背景建模 [14–18]；Selective Cloud Offloading 通过不确定性区域和 packing 改善云端检测 [19]。这些工作已充分说明压缩与 ROI 的价值，也表明区域裁剪可能受上下文丢失和端侧定位误差影响。

OneAdapt 通过在线梯度快速调整视频分析配置 [12]；TURBO 在服务级联合选择云模型、压缩配置和带宽，以满足实时系统的效用与时限约束 [38]。这些方法的决策对象是时间窗口或服务—配置；CAVS 则为当前图像的候选像素载荷预测检测价值，并以该图像的实际编码字节计费。

### 2.3 决策学习与在线资源控制

Learning to defer、多专家路由、FrugalML 和 predict-then-optimize 分别研究专家选择、预算化模型调用和面向下游目标的预测 [20–24]。置信度级联研究界定了标量置信度足够的条件 [27]，校准研究则表明神经网络原始分数并不天然等价于可靠概率 [28]。实时 token-bucket 路由进一步区分平均速率、短时突发和因果逐样本决策 [25]。这些思想为 CAVS 提供一般决策框架，但检测 AP 的非可加性以及载荷成本的逐图变化，使本问题不能直接化为单样本分类损失。

Budget-Adaptive Routing 指出，依赖端侧模型输出的路由在高上传率仍需支付端侧前向开销 [26]；CAVS 同样属于 weak-conditioned 路线，因为检测框既是价值特征，也是 ROI 几何来源。因此，本文把高预算计算税作为系统边界。综合现有研究，CAVS 的区别不在于首次使用 JPEG、ROI 或容量优化，而在于以 AP 对齐的图像—动作价值统一比较这些载荷，并由逐样本字节约束联合选择成员与动作。

## 3 系统模型与问题定义

图 1 给出 CAVS 的端到端信息流。端侧检测既提供未上传时的结果，也为价值模型提供语义输入；真实编码字节只进入调度器。离线训练使用标注生成奖励，部署阶段不访问标注。

![图 1 CAVS 系统与信息流。](figures/fig_cavs_system_overview.svg)

**图 1 CAVS 系统与信息流。** 全局调度用于衡量批量严格容量上限；零前视控制器逐帧决策并立即执行。

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

其中 $e$ 表示保留端侧结果，$o$、$q$ 和 $r$ 分别表示上传原始 JPEG、质量为 90 的 JPEG 和由端侧框构造的 ROI。O、OQ 和 OQR 分别表示开放一个、两个和三个云动作的配置。

对云动作 $a\ne e$，编码器产生载荷 $P_i^a$，成本为 $c_i^a=|P_i^a|$。云检测经坐标恢复与融合得到 $F_i^a$；端侧动作满足 $F_i^e=E_i$、$c_i^e=0$。请求失败时回退到 $F_i^e$。JPEG payload 与 HTTP wire bytes 分开记录，避免协议开销混入算法预算。

### 3.2 全局严格字节问题

令 $\hat v_i^a$ 为动作 $a$ 相对 Edge 的预测任务价值，$x_i^a\in\{0,1\}$ 为动作选择变量。全局 Fixed-byte 决策写为

\[
\begin{gathered}
\max_x\quad \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a\hat v_i^a,\\
\text{s.t.}\quad \sum_{i=1}^{N}\sum_{a\in\mathcal A_i}x_i^a c_i^a\le B,\\
\sum_{a\in\mathcal A_i}x_i^a=1,\quad \forall i,\\
x_i^a\in\{0,1\}.
\end{gathered}
\tag{3}
\]

式（3）是多选容量问题。检测 AP 不可逐图相加，因此其目标是可学习代理；最终质量仍由完整预测集合计算。

本文以 Always-Original 的总 payload 为参考，定义

\[
B_\rho=\rho\sum_{i=1}^{N} c_i^o,\qquad \rho\in\{0.10,0.25,0.40\}.
\tag{4}
\]

因此，10% 指原图总字节的 10%，而非上传 10% 的图像。若每图只有一个等成本云动作，式（3）退化为选择价值最高的 $K$ 幅图像，即传统 Top-K；一旦候选动作成本不同，成员与动作通常需要联合求解。

### 3.3 零前视因果在线问题

全局调度可观察批次中全部图像，在线策略则只能使用当前及历史信息。本文将零前视定义为 `buffer_frames=0`、`zero_lookahead=true` 和 `global_plan=false`：当前帧决定后立即编码与上传，提交结果后才读取下一帧。

在线控制还要求前缀安全。对前 $t$ 帧，令累计参考原图字节为 $C_t^o=\sum_{\tau\le t}c_\tau^o$、累计已花费字节为 $S_t=\sum_{\tau\le t}c_\tau^{a_\tau}$，则要求

\[
S_t\le \left\lfloor\rho C_t^o\right\rfloor+\beta_t,
\qquad \forall t\in\{1,\ldots,N\},
\tag{5}
\]

其中 $C_t^o=\sum_{\tau\le t}c_\tau^o$ 是累计参考原图字节，$S_t=\sum_{\tau\le t}c_\tau^{a_\tau}$ 是累计开销，$\beta_t$ 是预先声明的突发额度。主协议取 $\beta_t=0$，从而保证每个前缀都不超预算；paced-burst 只作为消融报告。

## 4 CAVS：上下文图像—动作价值学习

### 4.1 检测集合与动作表示

CAVS 由检测集合编码器、动作条件价值网络和外置成本调度器组成。每幅图像保留置信度最高的 $M=64$ 个检测；框特征包括归一化中心、宽高、面积和置信度，类别使用可学习嵌入。共享 MLP 编码每个检测后，以均值池化和最大池化获得置换不变表示：

\[
h_i^{set}=
\begin{cases}
\left[\dfrac{1}{n_i}\sum_j h_{ij}\;\middle\|\;\max_j h_{ij}\right], & n_i>0,\\[3pt]
\mathbf 0, & n_i=0.
\end{cases}
\tag{6}
\]

场景特征 $z_i$ 包含宽高比、框数量、置信度统计、框覆盖率和低成本图像统计。动作描述 $u_i^a$ 包含动作 ID、目标尺度和 ROI 几何，但删除 payload ratio 等成本字段。网络先在训练折标准化空间预测价值，再反标准化：

\[
\hat{\bar v}_i^a=f_\theta(h_i^{set},z_i,\operatorname{StripCost}(u_i^a)),
\qquad
\hat v_i^a=\sigma_{tr}\hat{\bar v}_i^a+\mu_{tr}.
\tag{7}
\]

端侧动作的价值固定为 0，三个云动作共享参数。实际成本 $c_i^a$ 不进入价值网络，只在调度阶段使用，从而避免把“便宜”误学为“有检测价值”。

### 4.2 上下文图像—动作奖励

ORIC/MORIC 在图像上下文中定义 mAP 对齐奖励 [11]。CAVS 将被替换对象从二元弱/强检测推广为动作 $a$。对训练图像 $i$，从同一训练折构造不含 $i$ 的上下文 $\mathcal C_i$，并定义

\[
v_i^a=(|\mathcal C_i|+1)\left[
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^a)-
\operatorname{AP}(F_{\mathcal C_i}^{e}\cup F_i^e)
\right].
\tag{8}
\]

上下文与标注只用于离线监督生成；部署仅使用端侧预测。该奖励仍是完整数据集 AP 的局部代理，因此模型选择和最终比较均以完整 evaluator 为准，而非奖励回归误差。

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

其中 $s_i^{ab}=\operatorname{sgn}(\bar v_i^a-\bar v_i^b)$、$w_i^{ab}=|\bar v_i^a-\bar v_i^b|$，正式配置取 $\alpha=0.5$。5-fold OOF 用于检查开发稳定性，随后以三个固定种子在完整 fit 上重训并平均反标准化后的价值。完整超参数列于附录 B。

### 4.4 ROI 与 MAUR 融合

ROI 动作从端侧检测框出发，经合并和面积相关扩框生成裁剪区域，再送入 640 输入的云检测器。MAUR 将云端框映射回原图，保留裁剪区外的端侧框，并替换裁剪区内的端侧结果。几何、云推理和融合共同决定 $F_i^r$，因此 OQR 必须用真实 ROI+MAUR 奖励从头训练。

ROI 是可选动作而非强制模块：若其价值低于 Original、Q90 或 Edge，调度器可以完全不选 ROI。O、OQ 和 OQR 因而构成动作空间消融，而不是网络容量消融。

## 5 调度与系统实现

### 5.1 全局容量可行调度

给定影子价格 $\lambda$，每幅图像独立选择

\[
a_i(\lambda)=\arg\max_{a\in\mathcal A_i}
\left(\hat v_i^a-\lambda c_i^a\right).
\tag{10}
\]

实现通过二分搜索 $\lambda$ 获得不超容量的解，再按边际价值/增量字节进行可行升级，最后断言总成本不超过 $B$。该拉格朗日—修复过程保证容量可行，但不宣称对一般多选背包精确最优。

### 5.2 零前视在线控制

在线控制器不预先评分完整序列。当前帧到达后，系统构造候选载荷和价值，屏蔽会违反式（5）的动作，再依据当前价格选择

\[
a_t=\arg\max_{a\in\mathcal A_t^{feasible}}
\left(\hat v_t^a-\lambda_t c_t^a\right),
\qquad
\lambda_{t+1}=\left[\lambda_t+\eta(c_t^{a_t}-\rho c_t^o)\right]_+.
\tag{11}
\]

主策略使用 calibration 冻结的价格；式（11）的受限 primal–dual 更新作为消融。动作执行和云端融合完成后才读取下一帧。

### 5.3 RK3588–云硬件链路

板端使用 Orange Pi 5 Ultra/RK3588 [34] 和 RKNN-FP16 检测器。每帧依次完成读取、端侧检测、路由决策、可选编码与上传、云检测、坐标恢复和融合。二进制 `CAVSHIL2` 协议替代 Base64 JSON；运行合同固定输入序列、模型、策略、控制文件和服务协议哈希，防止跨 run 复用旧候选或响应。系统分别记录 JPEG payload、HTTP 上下行、各阶段时延和完整 E2E，并在云端和客户端两侧过滤非法框。

### 5.4 复杂度与高预算边界

集合编码至多处理 64 个框；全局调度复杂度近似为 $O(TN|\mathcal A|+U\log U)$，其中 $T$ 为价格搜索次数、$U$ 为候选升级数；在线动作选择为 $O(|\mathcal A|)$。由于 CAVS 在决定上传前始终运行端侧检测器，高预算下仍存在 weak-forward tax。本文在结果部分报告完整阶段时延，但不以共享 GPU 测量替代独占模型基准，也不对未测量的能耗和峰值内存作推断。

## 6 实验方法

### 6.1 数据集与检测器

COCO [30] 使用 train2017 构造 fit=9,678 和 calibration=571，以 val2017 的 5,000 幅图像评测；端、云模型分别为冻结的 YOLO26n 和 YOLO26x [33]。Oregon 为 45 类相机陷阱野生动物数据，按相机事件划分 fit=9,923、calibration=4,961 和 final=4,962；云端使用 Cloud-V3 epoch-28，所有依赖云输出的学习器均重新训练。VOC [31] 由 VOC2007+2012 trainval 构造 fit=15,629 和 calibration=922，以 VOC2007 test 的 4,952 幅图像评测；端、云模型分别为 Torchvision [32] 的 SSDLite-MobileNetV3 与 Faster R-CNN-ResNet50-FPN-V2 官方权重，不训练检测器。表 1 汇总三组协议。

**表 1 数据集、划分与检测器配置。**

| 数据集 | 类别 | fit / calibration / final | Edge → Cloud | 主要验证角色 |
|---|---:|---:|---|---|
| COCO | 80 | 9,678 / 571 / 5,000 | YOLO26n → YOLO26x | 核心机制与消融 |
| Oregon | 45 | 9,923 / 4,961 / 4,962 | 冻结端模型 → Cloud-V3 | 跨域与事件相关性 |
| VOC | 20 | 15,629 / 922 / 4,952 | SSDLite-MNV3 → Faster R-CNN-R50-FPN-V2 | 跨架构验证 |

三个公开评测集均曾在开发阶段访问，因此本文将结果表述为冻结协议下的事后完整评测，而非新的隐藏测试。fit 用于学习器训练，calibration 仅用于冻结超参数和控制器，评测图像不参与训练或阈值选择。

### 6.2 指标、编码与比较方法

主指标为 COCO-style AP50–95，并报告 AP50 和 AP75。Original 使用原始 JPEG；Q90 固定 `quality=90`、`optimize=True` 和 `subsampling=2`；ROI 使用 Current profile 与 MAUR。所有方法共享检测缓存、字节表、容量分母和 evaluator。在线系统另报告上传帧、JPEG payload、HTTP 上下行、E2E 时延和吞吐。

比较器包括 Edge-only、Always-Original、Random-50、Confidence、DCSB [8]、Adaptive Feeding [40] 以及 ORIC/MORIC [11]。ORIC/MORIC 按论文奖励定义重实现；DCSB 使用原始三阈值难例规则；Adaptive Feeding 使用 Top-K 类别直方图、候选框特征和线性 SVM，但以冻结端侧 proposals 替代原文 Tiny-YOLO proposals。DCSB 与 Adaptive Feeding 的原生阈值工作点单独报告；Fixed-byte 表中的二者仅以各自 margin 排序并执行 Original，是统一成本下的适配结果，而非作者论文的官方工作点。TwoStage 与 CAVS 共用检测集合骨干但先排序图像、再分配载荷，用于隔离联合图像—动作建模；CAVS-O、OQ 和 OQR 均按各自动作空间重新生成奖励并训练。

### 6.3 训练与统计

每个 CAVS 配置进行 5-fold OOF，并以三个固定种子在完整 fit 上重训后平均预测。训练超参数见表 B1，单模型与 ensemble 结果见表 B2。Random-50 报告 50 套预固定 selection 的均值；不把单个随机种子冒充总体均值。主表采用冻结 final 的点估计；完成的因果比较另报告 1,000 次 paired bootstrap [29] 百分位 95% 区间。只有区间排除 0 时使用统计显著措辞，其余差异均按点估计解释。

### 6.4 硬件在环协议

批处理 HIL 先获得完整序列的价值后全局调度，仅作为真实执行的离线上限。因果在线 HIL 严格采用零缓冲、零前视和逐帧立即上传。正式矩阵包括 CAVS-OQR、CAVS-OQ、ORIC、MORIC、Confidence、五个 Random 种子、DCSB、Adaptive Feeding、端点、控制器消融和三种网络条件，共 54 组、270,000 帧。各组在同一 RK3588、RTX 4090、模型哈希和二进制协议下串行运行，并以 5,000/5,000 帧、云失败 0、非法框 0、前缀超额 0 作为验收条件。

## 7 结果与分析

### 7.1 三数据集 Fixed-byte 主结果

图 2 汇总三组端—云检测器上的冻结点结果。正文以预先固定的 CAVS-OQR 为主方法；每个数据集均重新生成动作奖励并训练路由器，因此这里检验的是算法流程的跨数据集适用性，而不是 checkpoint 的零样本迁移。

![图 2 三数据集 Fixed-byte 结果。](figures/fig_fixed_byte_three_datasets.svg)

**图 2 三数据集 Fixed-byte AP50–95。** 图中绘制覆盖全部三数据集的 CAVS-OQR、MORIC、Confidence 和 Random-50；所有方法共享所在数据集的检测缓存、容量分母和 evaluator。完整数值及其余比较器见表 B5。

**表 2 CAVS-OQR 与每个数据集最强外部比较器的 AP50–95。** 单元格内数值依次对应 10%/25%/40% 预算。

| 数据集 / 比较器 | CAVS-OQR | 比较器 AP | $\Delta$AP |
|---|---|---|---|
| COCO / MORIC | **0.4803 / 0.5329 / 0.5602** | 0.4522 / 0.4969 / 0.5282 | **+0.0280 / +0.0360 / +0.0320** |
| Oregon / Confidence | **0.5060 / 0.5174 / 0.5167** | 0.4896 / 0.5025 / 0.5104 | **+0.0164 / +0.0149 / +0.0062** |
| VOC / MORIC | **0.5315 / 0.5797 / 0.6127** | 0.5044 / 0.5550 / 0.5922 | **+0.0271 / +0.0247 / +0.0205** |

CAVS 在九个数据集—预算组合中均取得最高点估计，相对各数据集最强比较器提高 0.0062–0.0360 AP。COCO 与 VOC 的精度随预算单调提高；Oregon 在 25% 后饱和，说明增加字节并不保证有限样本价值代理下的非可加 AP 单调改善。DCSB 与 Adaptive Feeding 的原生阈值并不对应固定字节容量，故正文不把其自然工作点混入表 2；预算匹配适配和原生工作点分别列于表 B5 与表 B3。

### 7.2 COCO：核心机制和容量使用

表 3 显示 CAVS 不把字节预算退化为调用率预算：压缩与 ROI 使 10% 字节容量覆盖 28.18% 的图像，40% 时覆盖 83.04%，且三档容量利用率均超过 99.997%。

**表 3 COCO CAVS-OQR 调度组成与容量使用。**

| 预算 | Edge | Original | Q90 | ROI | 云覆盖率 | 容量利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 3,591 | 29 | 1,076 | 304 | 28.18% | 99.9976% |
| 25% | 2,049 | 63 | 2,468 | 420 | 59.02% | 99.9985% |
| 40% | 848 | 209 | 3,651 | 292 | 83.04% | 99.9996% |

低预算主要利用 Q90 和 ROI 扩大云覆盖，高预算则增加 Original，符合影子价格随容量降低后对高成本动作的偏好变化。

### 7.3 联合图像—动作估值的因果消融

同骨干 TwoStage 消融与 CAVS-OQ 使用相同的 top-64 集合编码、optimizer、fold、训练种子、O/Q reward 和 Fixed-byte 求解器；唯一关键变化是 TwoStage 先输出图像标量，再由 SharedAction 选择载荷，而 CAVS 直接预测每个图像—动作价值。

**表 4 COCO 同骨干结构消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| TwoStage：图像排序 → SharedAction | 0.450374 | 0.492738 | 0.524704 |
| CAVS-OQ：Joint image–action | **0.478224** | **0.528884** | **0.559286** |
| 差值 | **+0.027850** | **+0.036146** | **+0.034582** |

在骨干、输入和训练预算不变时，联合图像—动作估值带来 0.0279–0.0361 AP 提升，直接支持本文的核心机制，而不是仅说明某个 Ranker 更强。

### 7.4 O/OQ/OQR 动作空间消融

VOC 提供最干净的三档动作空间结果，因为 O、OQ、OQR 均按各自 reward 独立执行 5-fold×3 seeds 和 full-fit×3 seeds。端点 AP 为 Edge 0.473322、Original 0.657511、Q90 0.653684、ROI+MAUR 0.499819。

![图 3 VOC 动作空间消融。](figures/fig_voc_action_space_ablation.svg)

**图 3 VOC 动作空间消融。** （a）O、OQ 与 OQR 的绝对 AP50–95；（b）新增 Q90 与新增 ROI 的边际点增益。ROI 在 40% 预算为负，说明动作可用不等于动作必然有益。

**表 5 VOC 跨架构动作空间消融（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-O | 0.520249 | 0.557280 | 0.589427 |
| CAVS-OQ | 0.530787 | 0.579142 | **0.614054** |
| CAVS-OQR | **0.531486** | **0.579678** | 0.612721 |
| OQ − O | +0.010538 | +0.021862 | +0.024627 |
| OQR − OQ | +0.000699 | +0.000537 | −0.001333 |

Q90 在三档提高 0.0105–0.0246 AP，是跨架构收益的主要来源；ROI 的边际变化仅为 +0.0007、+0.0005 和 −0.0013 AP。相应地，OQR 仅选择 77/49/3 个 ROI，而 Q90 动作为 796/1,816/2,796 个。

Oregon 的 OQR 三档均未执行 ROI；OQ 与 OQR 的差异来自独立训练产生的 O/Q selection，而非 ROI 执行收益。因此本文只主张 ROI 可参与成本竞争，不主张它跨数据集普遍增益。

### 7.5 COCO Profile × Fusion 2×2

为分离 ROI 几何与融合规则，我们在同一 COCO 划分、云模型、fold、种子和调度器下，对 Current@640/Historical-cost@640 与 Reversible/MAUR 四组各自重建真实 reward 并全量重训。

**表 6 COCO Profile × Fusion 严格全重训（AP50–95）。**

| ROI profile | Fusion | 10% | 25% | 40% |
|---|---|---:|---:|---:|
| Current@640 | Reversible | 0.479071 | 0.531817 | 0.559932 |
| Current@640 | MAUR | **0.480277** | **0.532900** | 0.560178 |
| Historical-cost@640 | Reversible | 0.477784 | 0.531402 | 0.560025 |
| Historical-cost@640 | MAUR | 0.478155 | 0.531983 | **0.560701** |

Current+MAUR 是预先固定的主配置，但四格差异很小。Current 下 MAUR−Reversible 的 95% CI 在 10%、25% 和 40% 均跨 0（分别为 [−0.001916,+0.003460]、[−0.000061,+0.002212] 和 [−0.000607,+0.000946]），所以结果不支持“MAUR 全面显著更优”，也不允许按预算事后挑选四格最大值。

### 7.6 从离线调度到因果在线

![图 4 COCO 离线到在线的精度差距。](figures/fig_coco_offline_to_online_gap.svg)

**图 4 COCO 从缓存协议到零前视部署的精度阶梯。** （a）论文缓存、部署对齐离线、RK3588 全局 HIL 与正式因果在线结果；（b）部署对齐离线到正式在线的 AP50–95 差值。正式在线使用二进制协议，三档均为云失败 0、非法框 0 和前缀预算违规 0。

RK3588 批处理 HIL 先观察完整序列再调度，CAVS-OQR 为 0.4611/0.5036/0.5303 AP，相对 ORIC 提高 0.0213–0.0405 AP；它用于隔离真实 RKNN 与云执行，而不是在线性能声明。部署对齐离线 CAVS-OQR 为 0.4791/0.5311/0.5602，正式在线相对其下降 0.0243–0.0293 AP。此前 Mac 单图协议中，缓存与真实 HTTP 的 25% AP 仅差 0.000061，因此图 4 的主要差距来自 RKNN 检测域与“全局可比较完整序列”到“未知未来下满足前缀预算”的决策变化，而不是网络传输本身。

### 7.7 零前视在线准确率

**表 7 COCO 正式因果在线 HIL 结果（AP50–95）。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **CAVS-OQR** | **0.454831** | **0.505348** | **0.530900** |
| CAVS-OQ（无 ROI） | 0.451892 | 0.500310 | 0.527717 |
| ORIC | 0.438553 | 0.468420 | 0.492202 |
| MORIC | 0.430520 | 0.463411 | 0.485750 |
| Confidence | 0.427864 | 0.461410 | 0.482862 |
| DCSB（Original-only，预算匹配） | 0.432997 | 0.464606 | 0.490678 |
| Adaptive Feeding（Original-only，预算匹配） | 0.433734 | 0.465847 | 0.493665 |
| Random-5 | 0.421070±0.002418 | 0.444743±0.003250 | 0.465573±0.002642 |

CAVS-OQR 在三档均优于所有在线比较器，相对最强外部基线 ORIC 提高 0.0163–0.0387 AP，相对 Adaptive Feeding 提高 0.0211–0.0395 AP。与 CAVS-OQ 相比，ROI 动作带来 0.0029–0.0050 AP 正点增益；这与 VOC/Oregon 的离线边界不同，表明 ROI 收益依赖检测域与控制条件。正式矩阵中的每组均完成 5,000 帧，且云失败、非法框和最大前缀超额均为 0。DCSB 与 Adaptive Feeding 的原生阈值分别消耗 76.36% 和 78.17% 的 Always-Original 字节，故只在表 B4 单列，不与固定预算结果混写。

### 7.8 在线时延与通信成本

![图 5 正式在线系统精度、E2E 与通信成本。](figures/fig_online_system_results.svg)

**图 5 正式在线系统结果。** （a）八种策略的 AP50–95；（b）CAVS-OQR 全输入帧 E2E mean 的逐阶段堆叠及总 P95；（c）每输入图的 JPEG payload、协议额外上行和检测结果下行。图（a）中的 DCSB 与 Adaptive Feeding 为 Original-only 预算匹配适配；阶段 P95 来自不同帧，不能相加。

**表 8 CAVS-OQR 正式在线系统结果。**

| 预算 | AP50–95 | 云覆盖率 | Payload/图 | HTTP 双向/图 | E2E mean/P95 | FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.454831 | 30.16% | 16.256 kB | 20.732 kB | 162.71/305.61 ms | 5.76 |
| 25% | 0.505348 | 60.80% | 40.591 kB | 49.291 kB | 199.01/306.07 ms | 4.73 |
| 40% | 0.530900 | 83.88% | 65.049 kB | 75.756 kB | 238.47/327.94 ms | 3.99 |

随着预算提高，精度增加 0.0761 AP，而平均 E2E 从 162.71 增至 238.47 ms，体现了明确的精度—时延权衡。端侧 JPEG 解码、RKNN 预处理/推理和 NMS 合计约 94–102 ms；Cloud RTT 对全部输入帧的均值贡献由 42.56 增至 108.82 ms，是预算相关时延增长的主因。三档 payload 利用率均不低于 99.64%；二进制请求头与 metadata 仅增加 225–229 B/上传帧，上行协议开销不超过 0.43%。Q90 的逐图 payload 约为原图的 52%–54%，ROI 约为 22%–24%；完整阶段和动作成本列于表 B6–B7。作为端点，Edge-only 为 0.405264 AP、96.14 ms 和 9.39 FPS，Always-Original 为 0.544817 AP、251.07 ms 和 3.80 FPS。上述时延来自共享 RTX 4090 的端云系统观测，不应解释为独占 GPU 的纯模型基准。

### 7.9 控制器与网络敏感性

![图 6 受控网络下的延迟与吞吐。](figures/fig_network_sensitivity.svg)

**图 6 25% 预算下的网络敏感性。** 在云失败为 0 时，受控 RTT/带宽不改变动作与 AP，只改变 E2E 和吞吐；原生网络平均 ping 约为 5–7 ms。

在 50 ms/5 Mbps 下，CAVS-OQR 的平均 E2E 相对原生网络增加 151.01 ms，P95 增加 313.96 ms，吞吐由 4.73 降至 2.77 FPS；AP 仍保持 0.505348。这里未使用 deadline 或丢包驱动的回退，因而网络只影响系统效率；若加入 deadline-aware fallback，准确率也会随网络条件变化。

**表 9 自适应价格相对静态 $\lambda$ 的在线 AP50–95 差值。**

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-OQ | +0.000000 | −0.000339 | −0.000114 |
| CAVS-OQR | +0.000000 | +0.000070 | +0.001153 |

自适应价格的主要作用是容量跟踪，而非普遍提高精度：OQR 40% 的容量利用率由 97.665% 提高至 99.804%，但各档 AP 变化仅为千分位量级。

## 8 讨论与局限

### 8.1 证据范围

三数据集九个预算组合的点估计、同骨干 TwoStage 消融和 54 组 RK3588–RTX 在线运行共同支持一个一致结论：当候选载荷的价值与字节成本均随图像变化时，联合估计图像—动作价值比先做图像排序再分配载荷更有效。VOC 的 SSDLite→Faster R-CNN 组合进一步表明该机制不依赖单一 YOLO 检测器。这里的贡献是通信载荷约束下的动作条件价值调度，而不是普遍更强的难度 Ranker，也不是对能耗、货币成本或云 GPU 占用的统一优化。

ROI 体现了同一框架中的真实边界：它在 COCO 在线系统中提供小幅增益，在 VOC 高预算下为负，在 Oregon 中未被执行。因而 OQR 是完整系统配置，OQ 是关键消融；结果不支持“ROI 跨域普遍有效”，也不支持按数据集或预算事后挑选不同动作空间组成虚构的统一方法。

### 8.2 离线—在线差距与部署权衡

能观察完整序列的全局调度与零前视控制之间存在 0.024–0.029 AP 差距。无分布假设时，因果策略不可能保证复现未来可见的全局解；可行改进包括在独立序列上冻结多动作价格、采用 drift-aware primal–dual 或 bandit-knapsack 更新、允许预先声明的 token burst，以及直接优化 prefix-constrained reward。本文的价格消融主要改善容量跟踪而非 AP，说明进一步提升需要更好的状态与训练目标，而不是重复扫描 $\lambda$。

CAVS 还承担 weak-forward tax：每帧在决策前均执行 RKNN。其平均 E2E 随预算由 162.71 增至 238.47 ms，在 50 ms/5 Mbps 下 25% 档达到 350.02 ms。因此 CAVS 适合通信受限且端侧预测本身可复用的场景；高预算、严格 deadline 或端侧推理昂贵时，cloud-first、weak-skipping 或 deadline-aware fallback 可能更合适。

### 8.3 比较、公平性与外部有效性

ORIC/MORIC、DCSB 和 Adaptive Feeding 均在统一缓存、预算和硬件下重实现或适配，并非作者代码的逐位官方复现。特别地，ORIC/MORIC 原问题以二元上传和 offload ratio 为主，DCSB/Adaptive Feeding 的原生阈值也不对应多动作 Fixed-byte；本文因此将原生工作点与预算匹配适配分开报告，并用 CAVS-O/OQ/OQR 和 TwoStage 完成内部归因。

COCO val2017、Oregon final 和 VOC2007 test 在开发过程中被访问，因此结果应理解为冻结协议下的事后完整评测，而非未见确认集。正文只对已完成的因果比较报告置信区间，其余差异按点估计解释；Random 同时报告多个预固定种子，避免以单一随机 selection 代表总体。进一步的确认性主张需要时间后移数据或新注册测试集。

主预算定义是 JPEG payload。二进制 HTTP 的额外上行已实测低于 0.43%，但下行检测框、链路层包头、能耗和云计算量不属于优化目标。真实系统只覆盖 RK3588、共享 RTX 4090 和顺序播放的静态 COCO 图像；受控网络不包含丢包，静态序列也不刻画视频相关性、突发事件或概念漂移。以上限制约束了对其他设备、无线网络、能效和连续视频的外推。

## 9 结论

本文提出 CAVS，将端云目标检测从图像级二元上传扩展为异构像素载荷下的联合图像—动作价值调度。CAVS 以端侧检测集合预测 AP50–95 对齐的动作条件价值，将实际编码字节留给资源控制，并同时支持严格 Fixed-byte 批调度和零前视逐帧执行。三数据集结果和同骨干消融表明，收益主要来自联合决策粒度；正式 RK3588–RTX 系统则验证了前缀预算可行性、真实通信成本和在线精度优势。

结果也界定了适用范围：多质量 Q90 是稳定的主要增益来源，ROI 的作用依赖数据域；因果在线不能无代价复现未来可见的全局解；网络恶化主要增加时延而非在无失败时改变准确率。总体而言，当载荷价值与传输成本均为样本相关时，端云路由应在动作层连接任务价值预测与资源分配，而不应把图像排序、编码选择和预算控制割裂处理。

## Statements and Declarations

### Funding

（由作者填写基金名称与项目编号；如无资助，填写无专项资助声明。）

### Competing interests

（由作者确认后填写。）

### Author contributions

（由作者按 CRediT taxonomy 填写。）

### Data availability

COCO 与 PASCAL VOC 按其公开许可获取。Oregon 数据许可、访问方式及敏感位置信息处理说明由作者填写。

### Code availability

（由作者填写匿名代码仓库或归档 DOI；归档应包含冻结配置、模型哈希、评测脚本和复现实验清单。）

### Ethics approval

（由作者根据 Oregon 数据许可确认“不适用”或填写机构伦理审批编号。）

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

**表 B2 CAVS-OQR 三个 full-fit 单模型与 ensemble 的 AP50–95。**

| 数据集 | 预算 | 单模型均值±样本 SD | Ensemble | Ensemble−均值 |
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

Oregon 10% 的 ensemble 低于三个单模型均值，说明集成不是保证单调提高 AP 的机制；本文固定使用预先定义的三模型平均分数，而不按 final 选择单 seed 或 ensemble。

**表 B3 DCSB 与 Adaptive Feeding 的离线原生阈值工作点。**

| 数据集 | 方法 | AP50–95 | 云调用率 | Payload/Always-Original | 上传图像 |
|---|---|---:|---:|---:|---:|
| COCO | Adaptive Feeding-Native | 0.560217 | 74.020% | 74.241% | 3,701 |
| COCO | DCSB-Native | 0.562011 | 76.140% | 76.108% | 3,807 |
| Oregon | Adaptive Feeding-Native | 0.458955 | 2.882% | 3.495% | 143 |
| Oregon | DCSB-Native | 0.475223 | 11.205% | 11.865% | 556 |

表 B3 保留各方法在 calibration 上冻结后得到的自然阈值工作点，不把它们解释为 10%/25%/40% 预算结果。Adaptive Feeding 在 Oregon 的原生阈值过于保守，AP 仅略高于 Edge-only 0.454815；这也说明原生阈值跨数据域并不自动对应相同通信成本。

**表 B4 DCSB 与 Adaptive Feeding 的正式零前视 RK3588–RTX 系统结果。**

| 方法 | 工作点 | AP50–95 | 上传帧 | Payload MB | 字节率 | E2E mean/P95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| DCSB | 10% | 0.432997 | 831 | 81.192 | 9.966% | 119.20 / 228.26 | 7.69 |
| DCSB | 25% | 0.464606 | 1,686 | 203.357 | 24.961% | 141.08 / 255.66 | 6.56 |
| DCSB | 40% | 0.490678 | 2,424 | 325.622 | 39.968% | 163.88 / 276.82 | 5.70 |
| Adaptive Feeding | 10% | 0.433734 | 810 | 81.396 | 9.991% | 117.98 / 230.74 | 7.77 |
| Adaptive Feeding | 25% | 0.465847 | 1,636 | 203.479 | 24.976% | 143.78 / 267.49 | 6.45 |
| Adaptive Feeding | 40% | 0.493665 | 2,375 | 325.725 | 39.981% | 162.05 / 273.61 | 5.76 |
| DCSB | 原生 | 0.534972 | 3,831 | 622.075 | 76.356% | 200.91 / 289.67 | 4.69 |
| Adaptive Feeding | 原生 | 0.533136 | 3,890 | 636.873 | 78.172% | 205.06 / 289.01 | 4.60 |

表 B4 的字节率以 814.705 MB Always-Original payload 为分母。六个预算匹配组容量利用率为 99.658%–99.952%；所有 8 组均为 5,000/5,000、云失败 0、非法框 0 和前缀预算违规 0。E2E 与 FPS 来自共享 RTX 4090 条件下的完整串行运行，不替代独占 GPU 微基准。

**表 B5 三数据集 Fixed-byte AP50–95 完整点结果。** DCSB 与 Adaptive Feeding 为 Original-only 预算匹配适配。

| 数据集 / 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| **COCO — CAVS-OQR** | **0.480277** | **0.532900** | **0.560178** |
| COCO — MORIC | 0.452230 | 0.496914 | 0.528170 |
| COCO — ORIC | 0.449174 | 0.494970 | 0.527069 |
| COCO — Confidence | 0.447029 | 0.489546 | 0.520311 |
| COCO — DCSB | 0.442595 | 0.478828 | 0.510588 |
| COCO — Adaptive Feeding | 0.443563 | 0.477221 | 0.510021 |
| COCO — Random-50 | 0.433586 | 0.468908 | 0.500443 |
| **Oregon — CAVS-OQR** | **0.506028** | **0.517405** | **0.516663** |
| Oregon — MORIC | 0.479826 | 0.493286 | 0.506171 |
| Oregon — ORIC | 0.483153 | 0.502061 | 0.509599 |
| Oregon — Confidence | 0.489643 | 0.502499 | 0.510449 |
| Oregon — DCSB | 0.468816 | 0.485255 | 0.487301 |
| Oregon — Adaptive Feeding | 0.465760 | 0.484632 | 0.494235 |
| Oregon — Random-50 | 0.471087 | 0.484147 | 0.495973 |
| **VOC — CAVS-OQR** | **0.531486** | **0.579678** | **0.612721** |
| VOC — MORIC | 0.504416 | 0.554956 | 0.592212 |
| VOC — ORIC | 0.502039 | 0.553636 | 0.589734 |
| VOC — Confidence | 0.495439 | 0.532334 | 0.569091 |
| VOC — Random-50 | 0.482501 | 0.503851 | 0.532850 |

**表 B6 CAVS-OQR E2E 逐阶段 mean/P95（全部输入帧，ms）。** Edge 帧的 Cloud RTT 和非 ROI 帧的融合记为 0；各阶段 P95 来自不同帧，不能相加。

| 阶段 | 10% | 25% | 40% |
|---|---:|---:|---:|
| JPEG 解码 | 6.85 / 17.38 | 7.41 / 17.63 | 7.78 / 17.60 |
| Edge 预处理 | 0.54 / 0.99 | 0.55 / 0.99 | 0.58 / 1.03 |
| RKNN-FP16 | 65.47 / 78.55 | 67.47 / 83.31 | 68.31 / 84.68 |
| Edge 解码与 NMS | 20.79 / 45.58 | 22.78 / 50.31 | 25.29 / 55.80 |
| 候选构建、编码与策略评分 | 15.17 / 23.63 | 15.54 / 24.92 | 15.81 / 25.42 |
| 控制器与决策余项 | 1.46 / 2.45 | 1.67 / 2.66 | 1.62 / 2.63 |
| Cloud HTTP roundtrip | 42.56 / 148.53 | 72.15 / 131.93 | 108.82 / 161.28 |
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

完整复现需要固定数据角色与图像 ID 哈希、Edge/Cloud/RKNN 模型 SHA-256、编码与推理合同、reward/OOF 标记、三个 full-fit checkpoint、selection/control、bootstrap seeds，以及在线 frames、run contract、服务日志和系统汇总。表格应由机器可读结果自动生成。

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
