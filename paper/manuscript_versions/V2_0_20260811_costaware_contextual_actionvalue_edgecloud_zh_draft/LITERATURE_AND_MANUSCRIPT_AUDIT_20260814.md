# CAVS 相关文献与稿件定位审查（2026-08-14）

## 1. 结论先行

现有初稿的核心方向成立，但必须收紧创新主张。最稳妥的定位不是“比 ORIC/MORIC 更复杂的 Ranker”，也不是“提出 Adaptive ROI”，而是：

> **CAVS 将 ORIC/MORIC 的图像级、单云动作、等调用成本选择，推广为 AP 对齐的图像—动作联合价值学习，并在逐图真实字节容量下联合选择成员与载荷。**

这一区别具有可证明的结构性：单一等成本动作退化为 Top-K；Original、Q90、ROI 等动作具有不同且逐图变化的收益与字节时，静态图像前缀通常不足。COCO 的同骨干 TwoStage→Joint 消融正好支撑这一主张。

ROI 可以成为 CAVS-OQR 的主结果动作，但不能成为论文的首要新颖性来源。DCSB 的带宽自适应版本已经包含 regional sampling；2025 年 ICDM 的 Selective Cloud Offloading 进一步使用 conformal uncertainty 选择 ROI，并把多个不确定区域拼成一个云请求。因此，本文必须把 ROI 描述为由 CAVS 调度的候选载荷，而不是“首次提出选择性 ROI 上传”。

## 2. 代表性论文怎样组织论证

### 2.1 ORIC/MORIC

ORIC/MORIC 的写法值得借鉴，但不能照搬其创新表述。其论文结构非常清楚：

1. 先指出既有方法的两个具体缺口：单图指标忽略 mAP 的数据集上下文；分类阈值难以适应不同上传率。
2. 在方法前先给出形式化问题与实际不可观测性。
3. 明确拆开“理想奖励是什么”和“部署时如何估计奖励”。
4. 用 Oracle 验证奖励定义，再评估可部署估计器，避免把 Teacher 上界和学习器结果混为一谈。
5. 在引言中用小表格比较 prior work 是否 mAP-aligned、是否有 context、是否预算自适应。
6. 除主 AP 曲线外，报告上下文大小、检测错误类型、不同数据集/检测器对、真实测试台和各阶段时延。

对 CAVS 最重要的启示是：正文必须像 ORIC 一样主动写明“更合理的奖励未必更可学习”，并把 reward definition、value estimation 和 constrained scheduling 三个层次分开评估。

### 2.2 DCSB、DCCI 与 OffloaD

DCSB 用难例判别器完成小—大模型路由；其带宽自适应扩展加入 regional sampling、discriminator zoo 和视频采样率适配。DCCI 进一步同时考虑 hard-case discriminator、设备资源压缩和运行时网络/负载调度。OffloaD 则把 introspective、golden 与 latency score 放进滑动窗口调度，并在真实 Jetson/ROS2 测试台和退化网络下评测。

这些论文共同形成系统审稿人的预期：不能只给 AP/KB 离线表，还要报告调度器开销、编码、传输、云推理、回退和网络退化行为。Orange Pi 5 Ultra 实验因此不是装饰，而是支持“端云协同系统”定位的必要证据。

### 2.3 AccMPEG、RegionFilter、TileClipper 与 RegenHance

这些任务感知传输/视频分析工作不是 CAVS 的直接算法基线，但它们展示了系统论文的完整证据链：明确三个不可同时满足的目标，给出端侧低开销机制，构建真实原型，并同时报告任务精度、吞吐/时延、带宽和端侧计算。RegionFilter 还依赖云端反馈生成区域；RegenHance 使用 profile-based planner；这提醒本文不能把“Profile”或“任务相关 ROI”本身写成新概念。

CAVS 与它们的区别应写为：不训练新编解码器或增强器，而是在冻结像素载荷和检测器条件下，学习不同载荷的 AP 边际价值，并执行严格总字节调度。

### 2.4 Selective Cloud Offloading（ICDM 2025）

这是必须新增到正文的最近邻工作。它使用 conformal prediction 识别不确定检测区域，只上传高不确定 ROI，并通过二维矩形拼接减少云 API 调用和处理成本。其主要评测使用 precision、recall 和 API call 数。

CAVS 的区别不是“也上传 ROI”，而是：

- 决策监督是数据集级 AP50–95 的上下文边际价值，而不是检测框置信度/共形集合大小；
- Original、Q90、ROI 和 Edge 在同一图像—动作价值空间竞争；
- 约束是逐样本实际字节容量，而不仅是 API call 数；
- ROI 的 Profile 与融合方式作为动作语义进入重新生成的奖励，而不是固定不确定框打包；
- 对 ROI 负收益、云模型偶尔弱于端模型和融合破坏进行显式消融。

### 2.5 FrugalML 与预算化多专家选择

FrugalML 已经提出在预算约束下利用不同预测服务的样本相关强弱和成本。因此，CAVS 不能声称首次做“预算下的多动作/多专家选择”。本文的新颖性必须限定在目标检测的非可加 AP、像素载荷动作、逐图真实字节和端云检测融合这一组合。

### 2.6 Budget-Adaptive Routing（2026）

这项最新工作指出：常见路由器放在弱模型之后，高上传预算时仍为最终会交给强模型的样本支付弱模型前向成本；它用预算条件在 weak-conditioned 与 weak-skipping router 之间切换。它不直接解决 CAVS 的多载荷字节调度，但暴露了本文的一个真实边界：CAVS 当前始终先运行 Edge 检测器。

因此真机章节应增加两个对照：Always-run Edge 的 CAVS；高预算下跳过 Edge 的 Cloud-first/weak-skipping 上界。若暂时不实现，也必须在讨论中承认 CAVS 优化上传和云动作，而不保证所有预算下最小化端侧计算。

## 3. 论文应该形成的三层主张

### 主张 A：问题层

已有选择性卸载大多学习图像标量并按调用率选择；真实系统有多种逐图异构载荷，因此图片成员关系和载荷动作不可分离。Top-K 特例与反例负责理论支撑。

### 主张 B：方法层

CAVS 把 ORIC 的 contextual image reward 推广为 contextual image-action reward；集合编码器只预测任务价值，逐图字节仅由外部容量调度器使用。核心不是“双头”或“ROI gate”，而是动作条件价值与成本解耦的联合调度。

### 主张 C：证据层

COCO 证明一般场景下 Original/Q90 的字节适配与联合调度；Oregon Cloud-V3 检验域偏移和专用检测器条件；CAVS-OQR 验证 ROI 是否能在同一框架中作为可选动作带来额外 Pareto 收益。真机实验验证模型收益能否转化为端到端收益。

## 4. 现有稿件必须修改的地方

### 已在本轮修订

- 把 contextual reward 明确写成对 ORIC 思想的动作条件推广，不再暗示上下文化奖励本身由本文首次提出。
- 新增 DCCI、ICDM 2025 Selective Cloud Offloading、VaBUS 和 Budget-Adaptive Routing。
- 修正 DCSB 对 regional sampling 和带宽自适应的覆盖，避免把 prior work 简化成固定 Top-K。
- 把贡献压缩成三个可验证主张；未完成的 RK3588 实验不再算已完成贡献。
- 在实验问题中增加 detector complementarity、reward learnability、真实系统成本与 high-budget weak-skipping 边界。
- 在创新边界中明确：ROI、JPEG、集合编码、上下文奖励、拉格朗日调度和多专家预算选择均非单独首创。

### 结果出来后再替换

- 若 Oregon Cloud-V3 的 Current@640+MAUR 也稳定通过，主实例写为 CAVS-OQR，CAVS-OQ 作为动作空间消融。
- 若 Oregon OQR 不稳定，则主方法仍写 CAVS，COCO 采用 OQR、Oregon 采用 OQ，不声称统一 ROI 策略跨域最优。
- 所有“显著超过”必须等 paired/event-cluster bootstrap 完成；点估计只能写“高于”。

## 5. 投稿前最低完整证据

1. 主表：COCO 与 Oregon 的 Fixed-byte 10/25/40%，包含 Edge、Cloud、Random、Confidence、ORIC/MORIC、TwoStage、CAVS-O、CAVS-OQ、CAVS-OQR。
2. 统计：paired image bootstrap；Oregon 额外 event/camera cluster bootstrap；Random 使用 50-seed 统计对象一致的嵌套比较。
3. 核心因果消融：同骨干 TwoStage vs Joint；Value 包含成本 vs 价值—成本分离；O/OQ/OQR；Current/Historical × Reversible/MAUR。
4. 可学习性：Oracle reward、learned score Spearman、Teacher top-K overlap、安全/有害动作比例；清楚区分 Teacher ceiling 与部署结果。
5. 互补性：Edge 优于 Cloud 的图片比例、六类检测错误或至少 TIDE 分解、按目标尺寸/类别的收益。
6. 系统：RK3588 上 Edge、CAVS、JPEG/ROI 编码开销；网络传输；云推理；总时延、P95/P99、能耗、温度、容量利用率和掉线回退。
7. 鲁棒性：至少三个带宽档、带宽抖动/丢包或队列压力；高上传预算下 weak-skipping/Cloud-first 边界。
8. 复现：冻结 split、模型哈希、动作缓存、三种子、OOF/context 构造、外部实现适配说明。

## 6. 建议的论文题目与叙事

推荐题目：

> **面向异构像素载荷的成本感知端云协同目标检测：上下文图像—动作价值学习与严格字节调度**

英文：

> **Cost-Aware Edge–Cloud Object Detection with Contextual Image–Action Value Learning under Byte Budgets**

标题不突出 ROI，因为相邻工作已经覆盖 ROI 选择、区域采样和拼接；也不只写“选择性卸载”，因为这会把 CAVS 淹没在 DCSB、ORIC、OffloaD 和 2025 ICDM 工作中。最有辨识度的关键词是 **image–action value** 与 **byte budgets**。

建议全文采用如下叙事顺序：

1. 现有工作解决了“哪张图需要更强模型”；
2. 真实端云链路还必须决定“以什么载荷处理这张图”；
3. 异构载荷使 image ranking 与 action selection 耦合；
4. CAVS 学习 image–action contextual value，并让真实成本进入容量求解而不是价值网络；
5. COCO/Oregon 证明相对 ORIC/MORIC 的优势来自联合调度；
6. ROI 是框架可容纳的一类动作，收益受 Profile/融合/域影响；
7. RK3588 实验回答离线 AP/KB 优势能否成为真实 AP–latency–energy Pareto 优势。

## 7. 审查边界

本次审查区分三类事实：论文明确报告的设计/结果；基于论文之间结构差异得出的推断；对本文写作和实验的建议。没有发现与“上下文 image-action AP value + value/cost decoupling + actual-byte joint scheduling”完整同构的已发表方法，但这不等于可以宣称绝对首创。最稳妥的措辞是“据我们检索到的相邻工作，现有检测卸载方法尚未同时显式处理这三个维度”。
