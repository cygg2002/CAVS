# CAFRU：面向低功耗智能摄像头的成本感知帧级-感兴趣区域级选择性卸载与实时端云目标检测

作者姓名待补

作者单位待补

通讯作者：姓名与电子邮箱待补

## 摘要

低功耗智能摄像头受算力、内存和功耗约束，通常只能部署轻量目标检测
模型；将全部图像发送到云端虽然可使用计算规模更高的检测器，却会持续
占用上行带宽，并使结果受到网络往返时间和服务队列的影响。为此，本文
提出成本感知帧级-感兴趣区域级上传器（Cost-Aware Frame-ROI
Uploader，CAFRU），构建由端侧 YOLO26s、选择性上传器、有界异步队列
和云端 YOLO26x 复核器组成的实时端云目标检测系统。感兴趣区域
（region of interest，ROI）表示围绕端侧候选框扩展得到的图像区域。
CAFRU 使用四个线性预测头估计整帧和 ROI 复核获得正收益的概率及期望
收益，并通过“期望检测收益减去上传代价”选择整帧上传、ROI 上传或本地
输出。在 COCO val2017 的统一部署阈值口径下，JPEG 质量因子为 80 时，
系统将交并比阈值 0.50–0.95 上的平均精度均值（mAP50–95）从端侧单独
检测的 0.5350 提升至 0.5765，上传图像比例为 42.46%，均摊计算量为
79.7 GFLOPs/image；相比全量云端 YOLO26x@512 的 134.1
GFLOPs/image，计算量降低 40.6%。在 10 Mbps 上行带宽和 50 ms
网络往返时延下，系统平均端到端延迟为 83.6 ms，低于全量云端的
128.3 ms；全量云端的 mAP50–95 为 0.5996。组件级 T41 摄像头原型完成了
JPEG/HTTP 传输和有界异步调度，其近似特征线性决策器平均耗时
0.676 ms。结果表明，端云目标检测需要联合控制复核收益、上传粒度
和结果新鲜度，而不能只比较单个检测模型的精度或推理时间。

**关键词：** 端云协同；目标检测；选择性卸载；实时视频分析；智能摄像头；资源感知推理

## 1 引言

低功耗智能摄像头被广泛用于野外监测、安防巡检和无人值守感知。与将
连续视频全部发送到数据中心相比，本地检测能够减少原始图像传输，并在
网络不可用时保持基本响应。然而，资源受限视觉端的算力、内存、功耗和
算子支持限制了可部署模型的规模。实时检测器及其轻量变体已经形成较好
的精度-延迟折中 [1-5]，但同一模型系列内部仍存在明显的容量差异：本文
统一部署输出评估中，YOLO26s 的 COCO mAP50–95 为 0.5350，而
YOLO26x@512 在 JPEG 质量因子 80（Q80）条件下为 0.5996。边缘推理综述也表明，量化、
内存访问、编译器和
硬件算子支持会进一步限制端侧模型选择 [6,7]。

一种直接方案是保留端侧实时检测，并在云端部署容量更高的复核模型。
若所有帧均上云，系统成本将由图像编码、上行带宽、往返时延和云端排队
共同决定，网络抖动还会使已经返回的检测结果失去时效。协同推理研究已
探索模型切分、早退和任务卸载 [8-10]；视频分析系统则进一步研究了执行
位置、帧过滤、带宽自适应和服务器反馈 [11-18]。但目标检测的选择性
复核并不等价于图像分类中的低置信上传：一帧包含数量不定的目标框，
系统既要判断整帧是否值得重检，也要决定局部候选是否足以解决错误，还
需要在连续视频中处理迟到结果。

现有代表性方法分别覆盖了这一问题的不同侧面。Reducto 根据内容与任务
精度关系过滤摄像头帧 [15]，RegionFilter 利用服务器反馈进行区域感知
编码 [19]，EdgeBoost 通过置信度对资源受限分类任务进行选择性卸载
[20]。这些工作证明了选择性处理的价值，但尚未同时解决本文关注的
检测级收益监督、整帧/ROI 粒度选择、端云结果融合以及应用时限
（deadline）约束。
此外，神经网络置信度并不天然等于真实正确概率 [21]，仅靠固定低置信
规则可能上传无收益样本，也可能漏掉端侧空检或高置信错误。因此，上传器
应直接学习“云端复核相对当前端侧输出能否改善检测效用”，而不是只学习
“端侧是否不自信”。

本文提出 CAFRU，并将上传选择与实时传输调度统一到一套可部署的端云
系统中。主要贡献如下：

1. 提出检测效用驱动的双粒度上传方法。利用端侧图像/检测统计构建帧级
   与 ROI 级特征，以云端复核前后的目标级效用差生成监督，避免把置信度
   直接当作错误标签。

2. 设计由两个概率头和两个收益头组成的轻量上传器，并将其输出与显式
   上传数据量代价结合，形成可调的精度、上传量和均摊计算工作点。四个头
   均为线性模型，适合导出到低功耗设备。

3. 将选择性上传与有界异步队列、保留最新/拒绝新任务
   （keep-latest/drop-new）策略、结果年龄检查、基于 IoU 的框替换和
   类别感知 NMS 组成实时闭环，使系统优化目标从单次推理速度扩展到连续
   视频中的结果新鲜度。

4. 在完整 COCO train2017/val2017 和 T41 摄像头原型上分别评估算法
   收益与工程成本，并报告普通 ROI 规则、成本项、云端分辨率、JPEG
   质量、队列以及硬件运行时的作用边界。

## 2 相关工作

### 2.1 实时检测与资源受限部署

YOLO 将目标检测统一为单阶段回归问题 [1]，后续工作持续改进多尺度
表示、训练策略和端到端推理 [2-5]。YOLOv10 和 RT-DETR 分别从
无需非极大值抑制的 YOLO 与端到端 Transformer 检测方向优化实时
精度-延迟折中
[3,4]。本文采用 YOLO26s/YOLO26x 作为同系列的异构检测器；YOLO26
目前以 2026 年预印本形式公开 [5]，本文不将其模型结构作为创新点。

端侧部署不仅受 FLOPs 影响，还受内存、量化误差、算子支持、编译优化
和数据搬运限制 [6]。边缘智能研究因此主张在设备、边缘服务器和云之间
联合配置推理任务 [7]。本文与模型压缩工作互补：端侧检测器可替换为
其他轻量模型，云端复核器也可替换为更高容量检测器；CAFRU 关注的是
两者之间的任务选择和实时传输。

### 2.2 协同推理与选择性卸载

Neurosurgeon 依据设备和网络代价选择神经网络切分位置 [8]，
BranchyNet 通过中间出口提前终止简单样本 [9]，分割计算
（split computing）综述
总结了模型切分和早退的通信、能耗与精度问题 [10]。本文不传输中间
特征，而是在端侧完成可用的目标检测，再选择原始整帧或局部图像供云端
复核。这样可避免要求端云运行同一网络切分，也便于云端独立升级模型。

DeepDecision 在移动视频分析中联合选择执行位置和深度神经网络
（deep neural network，DNN）配置 [11]，
LAVEA 研究边缘视频任务的时延感知服务 [12]。EdgeBoost 证明置信度
驱动的选择性卸载能够改善资源受限分类 [20]，但分类样本只有单个标签，
目标检测还必须处理多框定位、类别一致性和局部坐标回映射。CAFRU 用
目标级真阳性/假阳性效用差训练上传器，并显式区分整帧与 ROI 复核。

### 2.3 实时视频选择与结果新鲜度

边缘视频分析需要同时控制计算、带宽和时延 [13]。AWStream 通过离线
剖析（profiling）学习精度-带宽关系并适配广域网
（wide-area network，WAN）[14]；Reducto 在摄像头端过滤
冗余帧 [15]；NoScope 使用专用级联和差分检测降低视频查询成本 [16]。
Glimpse 指出服务器反馈超过帧间隔后会降低连续识别质量，并使用触发帧
和本地缓存处理迟到提示 [17]。移动增强现实（augmented reality，AR）
目标检测也表明，云端推理需要与本地处理和时延控制协同设计 [18]。

RegionFilter 从空间冗余角度为 ROI 与非 ROI 区域选择不同编码配置
[19]。其区域由低层特征和服务器历史反馈生成，而本文 ROI 来自当前
端侧检测框的自适应扩展，并由预测检测收益决定是否上传。近期近边缘
目标检测综述和时序云边检测工作进一步说明，执行位置与时序信息仍是
活跃研究方向 [22,23]。本文关注从 COCO 效用监督到低功耗摄像头有界
队列的完整实现路径，同时报告 ROI 分支在当前实现中的有限增益。

## 3 方法

### 3.1 问题定义与系统架构

给定视频帧 \(I_t\)，端侧检测器 \(D_e\) 输出
\(B_e^t=\{b_i\}_{i=1}^{N_t}\)，其中每个检测
\(b_i=(x_1,y_1,x_2,y_2,c_i,p_i)\) 包含原图坐标框、类别和置信度。
云端复核器记为 \(D_c\)。上传器需要从动作集合
\(\mathcal A_t=\{\mathrm{skip},\mathrm{frame},\mathrm{roi}\}\)
中选择动作，在提高检测质量的同时限制上传与计算成本。

系统在线路径由五部分组成：摄像头采集与 YOLO26s 检测；CAFRU
上传决策；JPEG 编码与有界队列；YOLO26x 云端复核；结果年龄检查与
检测框融合。未上传帧立即输出 \(B_e^t\)。整帧或 ROI 任务进入独立
上传线程，主采集/检测线程不等待云端返回。云端结果携带帧标识和时间戳，
只有在结果年龄不超过应用时限（deadline）时才进入融合缓存。

**[图 1 插图位置：端云协同在线闭环架构。按
`FIGURE_DRAWING_SPEC_ZH.md` 绘制。]**

图 1 中，解码帧同时指向端侧检测和帧级像素统计，是因为图像复杂度
需要当前帧像素，而检测数量、覆盖率和置信度来自 \(B_e^t\)。ROI 特征
不等于整张图特征，也不等于仅上传端侧检测框坐标；系统先以端侧
框为中心自适应扩展图像区域，再上传该区域的 JPEG 像素和坐标偏移。
若不上云，端侧框直接进入输出；若云端结果迟到，则丢弃该异步结果而不
回写已经输出的历史帧。

### 3.2 低开销特征与 ROI 构造

CAFRU 不增加卷积特征提取网络。帧级向量 \(x_f\in\mathbb R^{15}\)
由四组统计组成：图像边缘密度；检测数量与覆盖率；最大、均值、最小和
标准差置信度；空检、稀疏、低覆盖、平均/最大框面积、贴边框比例、
低置信框比例和小目标比例。除边缘密度外，其余特征均可由端侧框线性
扫描获得。

对每个端侧框构造 \(x_{r,i}\in\mathbb R^{15}\)，包括置信度及其补值、
相对置信度、面积与对数面积、宽高比、边界距离、小目标/贴边指示、中心
位置、覆盖占比，以及图像复杂度、归一化检测数和框覆盖率三个共享帧级
上下文特征。候选 ROI 由端侧框 \(b_i\) 向四周
扩展后裁剪：

\[
R_i=\operatorname{clip}(\operatorname{expand}(b_i,\gamma_i),I_t),
\]

其中扩展倍率 \(\gamma_i\) 根据框面积和边界距离调整。小目标使用更大
扩展倍率；贴边目标经图像边界裁剪后形成不对称上下文。上传内容为
\((\operatorname{JPEG}(R_i),t,\Delta x_i,\Delta y_i)\)，云端框通过
偏移 \((\Delta x_i,\Delta y_i)\) 映射回原图坐标。

### 3.3 检测收益监督与四头上传器

训练阶段使用 COCO 人工标注 \(Y\) 计算端侧与复核后检测的目标级效用。
预测框与标注按类别一致且交并比（intersection over union，IoU）不低于
0.5 的条件进行一对一匹配。记真阳性和假阳性数量分别为 \(TP\) 和
\(FP\)，定义

\[
U(B,Y)=TP(B,Y)-\alpha FP(B,Y),
\]

其中本文取 \(\alpha=0.25\)。整帧复核收益为

\[
g_f=\frac{U(B_c,Y)-U(B_e,Y)}{\max(1,|Y|)}.
\]

ROI 复核先将云端 ROI 结果回映射并与未受影响的端侧框融合为 \(B_{fr}\)，
再计算

\[
g_{r,i}=\frac{U(B_{fr},Y)-U(B_e,Y)}{\max(1,|Y|)}.
\]

归一化用于避免拥挤图像仅因目标数量多而产生更大训练目标。二值标签为
\(z=\mathbb 1[g>0]\)。因此，空检帧、高置信误检或定位偏差只要云端
确实改善目标级效用，都可以成为正样本；低置信但云端无改进的样本不会
被强制标为正类。

上传器包含四个冻结的线性头：

\[
p_f=\sigma(w_f^\top \tilde x_f+b_f),\quad
\hat g_f=v_f^\top \tilde x_f+a_f,
\]

\[
p_{r,i}=\sigma(w_r^\top \tilde x_{r,i}+b_r),\quad
\hat g_{r,i}=v_r^\top \tilde x_{r,i}+a_r.
\]

概率头采用类别平衡逻辑回归，收益头采用 Huber 损失线性回归；所有特征
仅使用训练划分均值和标准差标准化。训练集内部按固定随机种子划分 90%
训练和 10% 验证，并加入小幅特征噪声与 L2 正则。四头合计只包含数十
个权重，导出后只需标准化、点积和 sigmoid。

**[图 2 插图位置：CAFRU 离线训练与在线决策。按
`FIGURE_DRAWING_SPEC_ZH.md` 绘制。]**

图 2 将离线训练和在线推理分为两个阶段。在线阶段只加载冻结参数，
不会根据当前视频继续训练。后续若场景分布发生变化，需在独立标注数据上
重新校准或训练，再发布新参数，这属于离线模型维护而非实时推理闭环。

### 3.4 成本感知决策与实时调度

对整帧和第 \(i\) 个 ROI，定义

\[
S_f=p_f\max(0,\hat g_f)-\lambda_b C_f,\qquad
S_{r,i}=p_{r,i}\max(0,\hat g_{r,i})-\lambda_b C_{r,i},
\]

其中 \(C=k/100\)，\(k\) 为估计 JPEG 大小（KB），\(\lambda_b\) 控制
带宽代价。若 \(p_f\ge \tau_f\)、\(S_f>0\) 且队列可接收任务，则上传
整帧；否则从满足 \(p_{r,i}\ge\tau_r\) 且 \(S_{r,i}>0\) 的候选中选择
最高分的至多 \(K\) 个 ROI；若均不满足，则不上云。提高
\(\tau_f,\tau_r\) 会减少上传，但通常牺牲检测收益。

当前学习头输入只包含图像与检测统计；上行带宽、网络往返时延（RTT）和
队列长度不是已训练的模型特征。独立运行时层在任务接纳时通过 drop-new
或 keep-latest 控制队列容量，并在结果返回后丢弃年龄超过时限的结果。
上行带宽和 RTT 会影响实际队列占用与结果年龄，但不直接进入学习头。
这样将“该图像是否可能受益”和“返回结果是否仍然及时”分离，避免网络
状态变化迫使重新训练学习头。

整帧上传时，当前实现以云端框替换该帧端侧框；不上云时保留端侧框。
ROI 上传时，系统在扩展区域内按 IoU 匹配替换目标框，并允许满足置信
阈值的未匹配云端框补充端侧漏检，最终执行类别感知非极大值抑制
（non-maximum suppression，NMS）。该门控机制可以利用端侧与云端
检测器的互补错误，但不保证协同输出必然超过全量云端；本文只在一致
评估口径下解释实测结果。

### 3.5 复杂度

帧级统计为 \(O(N_t)\)，ROI 特征和评分为 \(O(N_t)\)，四个线性头的
维度固定为 15。决策器参数量和 FLOPs 相对 YOLO26s 可忽略。图像编码、
网络传输和云端检测仅发生在被接纳任务上，均摊系统计算量定义为

\[
F_{\mathrm{sys}}=F_e+
\frac{1}{M}\sum_{t=1}^{M}
\left(a_f^t F_c(s_f^t)+\sum_i a_{r,i}^t F_c(s_{r,i}^t)\right),
\]

其中 \(F_e\) 为端侧检测计算量，\(F_c(s)\) 为云端检测器在输入尺寸
\(s\) 下的计算量。当前实现将整帧和 ROI 都缩放到设定的云端输入尺寸，
因此计算量按实际云端请求次数和输入尺寸统计，而不按 ROI 原图面积直接
缩放。

## 4 实验设置

### 4.1 数据集、模型与训练划分

标准精度实验采用 COCO [24]。上传器在 train2017 的 118,287 张图像
上生成监督，端侧和云端模型分别为官方预训练 YOLO26s 与 YOLO26x。
轻量头训练仅使用 train2017 内部的 90/10 划分；所有最终检测指标在
独立的 val2017 5,000 张图像上计算，因此验证图像不参与收益标签生成
或线性头训练。COCO 主结果使用 80 类标注；端侧、云端和协同输出均在
相同部署置信阈值下复评估，以避免把常规低阈值检测指标与部署输出指标
直接比较。

主评估中端侧输入尺寸为 640，端侧和云端部署置信阈值均为 0.25。云端分别评估
512 和 640 输入；平衡配置默认使用 512，以降低均摊计算。JPEG 部署
实验模拟质量因子 50–90（Q50–Q90）的编解码。完整命令、阈值数值和
数据文件映射见
Online Resource 1。
COCO 云端推理计时运行在单张 NVIDIA RTX 4090 上；由于延迟依赖具体
硬件，本文同时报告 GFLOPs/image 作为跨设备计算量参考。

真实设备采用 T41 摄像头系统级芯片（system on chip，SoC）。端侧模型经
INT8 神经网络加速器（neural-network accelerator，NNA）工具链部署，
摄像头输出 NV12，选择帧经 JPEG 编码后通过 HTTP multipart 发送至
YOLO26x 服务。T41 实验用于测量采集、端侧推理、编码、HTTP、云端
推理、队列和策略开销；不与工作站上的 COCO 推理时间直接比较。

### 4.2 对比方法

- **Edge-only（仅端侧）：** 所有图像仅使用 YOLO26s。

- **Cloud-only（仅云端）：** 所有图像均使用 YOLO26x。

- **Rule selective（规则选择）：** 由低置信、小目标、贴边、空检和周期关键帧规则
  决定上传。

- **ROI-rule（ROI 规则）：** 仅上传固定规则选出的扩展 ROI，用于检验普通
  ROI+云复核是否足够。

- **CAFRU：** 使用本文四头收益模型。Accuracy、Balanced、Efficient
  和 Bandwidth 表示从低到高阈值形成的四类部署工作点，而不是不同网络。

### 4.3 指标

检测指标包括精确率（Precision）、召回率（Recall）、IoU=0.50 时的
平均精度（AP50），以及 IoU 从 0.50 到 0.95、步长为 0.05 的平均精度
均值（mAP50–95）。上传图像比是至少
产生一次上传的图像数除以总图像数；KB/image 是总 JPEG 上传字节数除以
全部输入图像数，而不是除以上传次数，且日志中 1 KB 按 1024 bytes
计算；GFLOPs/image 是端侧与选择性云端计算的每图均摊十亿次理论浮点
运算量。

模型化端到端（end-to-end，E2E）延迟由端侧推理、编解码、上传传输、
网络往返时延、云端推理和队列等待组成。连续视频另外统计第 95 百分位
（P95）延迟、候选丢弃率和过期结果比例。真实 T41 的队列隔离实验中，
调度循环每秒帧数（frames per second，FPS）表示处理帧数除以墙钟时间；
“及时结果帧”表示处理当前帧
时存在年龄不超过时限 \(D\) 的云端缓存结果；有效帧率
（effective FPS@D）为及时结果帧数除以墙钟时间。一次云端返回可在时限
内服务多个后续帧，因此及时结果帧数可以大于云端返回次数。这些是系统
实时性指标，不是 COCO 检测指标。

### 4.4 有效性边界

T41 的 100 帧室内图像另使用 SpeciesNet 自动生成 `person` 框，仅作为
伪参考标注进行初步一致性检查。该数据没有完成人工逐框核验，因此不能
称为人工真值标注，也不能用于声称真实场景 mAP 已被严格证明。

板端线性决策器使用导出的四头参数，但当前 C++ 原型对图像复杂度、
检测数量归一化和相对置信度采用了近似实现，尚未完成与 Python 特征的
逐项数值等价测试。因此，板端策略耗时和该次运行上传率是有效的系统
测量，但不作为 COCO Balanced 工作点在 T41 上的精确复现。

## 5 结果与分析

### 5.1 COCO 主结果

**表 1 COCO val2017 主结果。所有检测结果均以置信阈值 0.25 的统一**
**部署输出口径复评估；本表不包含 JPEG 编解码，网络时延见表 3。**

| 方法 | Precision | Recall | mAP50 | mAP50–95 | 上传图像比 | KB/image | GFLOPs/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edge-only YOLO26s | 0.7147 | 0.5621 | 0.6672 | 0.5350 | 0.0000 | 0.00 | 22.8 |
| ROI-rule strict | 0.7300 | 0.5628 | 0.6681 | 0.5351 | 0.3824 | 1.50 | 74.1 |
| Rule selective | 0.7318 | 0.5832 | 0.6817 | 0.5486 | 0.6406 | 22.94 | 168.1 |
| CAFRU-Accuracy | 0.7835 | 0.6202 | 0.7251 | 0.5918 | 0.5656 | 34.90 | 98.7 |
| CAFRU-Balanced | 0.7758 | 0.6092 | 0.7163 | 0.5829 | 0.4248 | 26.55 | 79.8 |
| CAFRU-Efficient | 0.7671 | 0.5919 | 0.7032 | 0.5704 | 0.2442 | 15.87 | 55.6 |
| CAFRU-Bandwidth | 0.7519 | 0.5748 | 0.6861 | 0.5518 | 0.0920 | 6.28 | 35.2 |

在相同部署输出口径下，CAFRU-Balanced 相比 Edge-only 提升
0.0479 mAP50–95，并将上传图像比例限制在 42.48%；精度优先的
CAFRU-Accuracy 提升 0.0568。带宽受限时，CAFRU-Bandwidth 只上传
9.2% 图像，仍比 Edge-only 高 0.0168 mAP50–95。全量云端结果在
JPEG 部署条件下与 CAFRU 进行同口径比较，见表 3。

**[图 3：统一部署输出口径下，COCO mAP50–95、KB/image 与**
**GFLOPs/image 的折中；气泡面积表示均摊计算量。]**

表 1 和图 3 展示的是端侧输出、规则输出和 CAFRU 输出之间的同口径
比较。本文不把常规低阈值检测基线与置信阈值 0.25 的系统输出直接相减，
也不据此声称协同结果超过全量云端。表 3 的同阈值 Q80 实验表明，
CAFRU 的作用是在云端精度、上传量和端到端时延之间取得折中。

### 5.2 核心消融

**表 2 CAFRU 核心消融。不同分组只改变表中所示因素。**

| 分组 | 设置 | mAP50-95 | 上传图像比 | KB/image | GFLOPs/image | 主要结论 |
|---|---|---:|---:|---:|---:|---|
| 成本 | 无上传量代价 | 0.5856 | 0.5184 | 31.44 | 92.3 | 精度略高但上传更多 |
| 成本 | CAFRU-Balanced | 0.5830 | 0.4246 | 26.55 | 79.7 | 代价项抑制低价值上传 |
| 粒度 | ROI-rule strict | 0.5351 | 0.3824 | 1.50 | 74.1 | 带宽低但精度不足 |
| 分支 | ROI-only | 0.5406 | 0.0020 | 0.03 | 23.1 | 单独 ROI 分支覆盖不足 |
| 分支 | Frame-only | 0.5829 | 0.4238 | 26.53 | 79.6 | 当前主要收益来源 |
| 分支 | Frame+ROI | 0.5830 | 0.4246 | 26.55 | 79.7 | ROI 是有限补充 |
| 云尺寸 | 512 | 0.5830 | 0.4246 | 26.55 | 79.7 | 默认计算平衡 |
| 云尺寸 | 640 | 0.5880 | 0.4246 | 26.55 | 111.8 | +0.0050 mAP，+40.3% 计算 |
| JPEG | Q70 | 0.5744 | 0.4270 | 20.44 | 80.1 | 更低带宽 |
| JPEG | Q80 | 0.5765 | 0.4246 | 26.55 | 79.7 | 默认编码平衡 |

各消融分组使用对应的独立复跑作为组内参照，不跨组计算微小差值。主表
Balanced 快照与成本/云尺寸消融快照分别接纳 2,124 和 2,123 个任务；
一个阈值边界 ROI 的差异使四舍五入后的 mAP50–95 分别为 0.5829 和
0.5830，不影响组内结论。

成本项将上传比例从 0.5184 降到 0.4246，mAP50-95 仅下降 0.0026，
说明学习收益与显式上传量惩罚具有互补作用。普通 ROI-rule 虽然每图
仅 1.50 KB，但 mAP50–95 为 0.5351，低于 CAFRU-Balanced。
这回答了“为何不直接 ROI+云复核”：端侧漏检没有候选框可裁剪，过小
ROI 会丢失上下文，且 ROI 坐标回映射和局部误检会增加融合难度。

分支消融同时表明，当前收益主要来自帧级选择。Frame+ROI 相比
Frame-only 只增加约 0.0001 mAP50-95，因此本文不将 ROI 描述为主要
精度来源；其价值在于为未来更强的候选生成和极端带宽场景提供细粒度
接口。云端 640 输入仅提升 0.0050 mAP50-95，却使均摊计算增加约
40.3%，因此默认采用 512。

### 5.3 弱网与队列实时性

**表 3 置信阈值 0.25、10 Mbps 上行和 50 ms 网络往返时延下的**
**JPEG 部署仿真。**

| 方法 | 部署输出 mAP50–95 | 上传比 | KB/image | GFLOPs/image | Mean E2E (ms) | P95 E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Edge-only | 0.5350 | 0.0000 | 0.00 | 22.8 | 29.5 | 29.5 |
| Cloud-only@512 Q80 | 0.5996 | 1.0000 | 62.52 | 134.1 | 128.3 | 128.3 |
| ROI-rule strict | 0.5351 | 0.3824 | 1.50 | 74.1 | 62.7 | 122.6 |
| CAFRU Q70 | 0.5744 | 0.4270 | 20.44 | 80.1 | 79.0 | 162.5 |
| CAFRU Q80 | 0.5765 | 0.4246 | 26.55 | 79.7 | 83.6 | 179.0 |
| CAFRU Q90 | 0.5799 | 0.4238 | 37.73 | 79.6 | 92.5 | 207.3 |

JPEG 质量因子为 80 时，CAFRU 相比 Edge-only 提升 0.0415
mAP50–95；相比 Cloud-only@512，CAFRU 的 mAP50–95 低 0.0231，
但平均 E2E 从 128.3 ms 降至 83.6 ms，上传图像比例从 100% 降至
42.46%，均摊计算量从 134.1 降至 79.7 GFLOPs/image。Q90 相比 Q80
再提升 0.0034 mAP50–95，但每图上传量增加
42.1%，P95 超过 200 ms；Q70 则更偏向弱网上传成本。

在 15 FPS、200 ms 时限的队列模拟中，Cloud-only 在无等待队列时丢弃
50.00% 的上传候选。CAFRU Q70 在无等待队列时丢弃 31.99%；使用容量为
1 的 keep-latest 队列后，丢弃率降至 8.34%，但过期结果增加到每图
0.0860，云端 E2E 的 P95 从 173.2 ms 增至 240.7 ms。这说明增加队列
容量能够减少任务丢弃，但会提高等待和结果过期风险。

**[图 4：10 Mbps 上行和 50 ms RTT 下各方法的平均与 P95 E2E。]**

### 5.4 选择性、困难样本与指标公平性

在 CAFRU Q80 实际上传子集上，统一部署阈值下 Edge、Cloud 和 Fused
的 mAP50–95 分别为 0.4756、0.5536 和 0.5447；在微小目标子集上分别
为 0.4413、0.5050 和 0.4834。上传子集的端侧精度低于全体数据，说明
上传器确实偏向较困难样本；融合结果相对端侧分别提高 0.0691 和 0.0421，
但仍低于对应的全云端结果。未上传子集的端侧和融合 mAP50–95 分别为
0.6192 和 0.6220，仅相差 0.0028，构成“零上传不应产生显著云端收益”
的负对照。

这些困难样本子集由 COCO 标注几何和上传事件构建，
不是 COCO 官方新划分。完整子集定义、样本数量和指标公平性表放入
Online Resource 1。公平性检查还表明，官方置信口径和部署置信口径会
改变绝对 mAP；正文主结论只比较相同评估口径下的方法。

### 5.5 T41 真实原型

**表 4 T41 原型的实测组件成本。各行来自对应组件实验，不表示同一帧**
**的可加总延迟。**

| 组件 | 设置 | 平均值 | P95/补充量 | 解释 |
|---|---|---:|---:|---|
| YOLO26s NNA | 成功样本 | 585.971 ms | 701.825 ms | 端侧主要计算瓶颈 |
| JPEG 编码 | Q70, width=640 | 82.069 ms | 40.053 KB/upload | 真实板端编码 |
| HTTP 往返 | Q70, width=640 | 1503.373 ms | - | 包含网络与服务器等待 |
| 云端 YOLO26x | 同上 | 1123.246 ms | - | 当前服务端实测 |
| 规则选择性闭环 | 100 帧 | 37% 上传 | 8.462 KB/frame | 37/37 上传成功 |
| 板端线性决策器 | 50 帧在线决策 | 0.676 ms | 0.817 ms | 近似特征原型 |

规则选择性闭环在 100 个样本中上传 37 个，全部返回成功；单次上云平均
22.870 KB，按全部帧均摊为 8.462 KB/frame。这说明选择性上传可以在
真实 JPEG/HTTP 链路中直接降低通信量。YOLO26s 成功样本平均推理
585.971 ms，明显高于上传器计算，端侧 NNA 和云端服务才是主要瓶颈。

**表 5 T41 同步与异步调度隔离实验。该实验采用旁路本地 YOLO26s**
**的仅传输模式，用于测量传输和队列行为。**

| 模式 | 队列 | 时限 | 调度循环 FPS | 上传成功 | 丢弃任务 | 及时结果帧 | Effective FPS@D |
|---|---:|---:|---:|---:|---:|---:|---:|
| 同步全帧 | 1 | 1000 ms | 0.614 | 19/19 | 0 | 0 | 0.000 |
| 异步 | 1 | 1000 ms | 6.623 | 11/11 | 89 | 62 | 4.106 |
| 异步 | 2 | 500 ms | 6.828 | 12/12 | 88 | 31 | 2.117 |
| 异步 | 2 | 1000 ms | 6.613 | 12/12 | 88 | 62 | 4.100 |
| 异步 | 2 | 2000 ms | 6.663 | 12/12 | 88 | 65 | 4.331 |
| 异步 | 4 | 1000 ms | 6.751 | 15/15 | 85 | 63 | 4.253 |

同步每帧上传受约 1.5 s HTTP 往返阻塞，调度循环只达到 0.614 FPS，
且没有结果在 1000 ms 内可用。异步队列将不含本地检测的调度循环提高到
6.6–6.8 FPS，但为保持新鲜度丢弃了 85–89 个候选任务。队列容量为 4、
时限为 1000 ms 时获得 4.253 effective FPS@D；将时限从 500 ms
放宽到 2000 ms 可增加及时结果帧数，但也允许更旧结果进入缓存。上述
FPS 只反映采集、编码、HTTP 和队列调度，不代表包含 YOLO26s 推理的
完整端云检测帧率。

**[图 5 插图位置：T41 主线程、上传线程与时限门控。按
`FIGURE_DRAWING_SPEC_ZH.md` 绘制。]**

SpeciesNet 辅助伪参考检查包含 100 帧和 108 个自动 `person` 框。在该
非人工真值口径下，端侧、云端和当前规则闭环最终结果的 AP50 分别为
0.2973、0.4882 和 0.4835。它只能初步提示云端对端侧 person 漏检具有
补偿作用。由于最终结果来源近似全上传，且参考框未经人工核验，该结果不
用于证明 CAFRU 的真实场景 AP，也不进入本文核心结论。

## 6 讨论

### 6.1 与普通 ROI+云复核的比较

普通 ROI 方案只有在端侧已经提出候选框时才有裁剪中心，因此无法修复
端侧完全漏检；较小裁剪会损失上下文，较大裁剪又逐渐接近整帧成本。
此外，固定低置信、小目标或贴边规则只描述“可能困难”，没有估计云端
复核是否真正改变 TP/FP。CAFRU 通过复核前后的效用差训练门控，并在
整帧与 ROI 之间比较收益和上传数据量。实验中 ROI-rule strict 虽极省
带宽，但比 CAFRU-Balanced 低 0.0478 mAP50–95；同时 Frame-only
消融表明，当前实现的主要优势来自学习型整帧选择，而非 ROI 复核分支。

### 6.2 精度、带宽和实时性的部署选择

CAFRU 输出的是一组帕累托（Pareto）工作点。精度优先场景可考虑
Accuracy；Balanced 是一般资源约束下的候选配置；Efficient 和
Bandwidth 分别面向弱网及更严格的上传约束。阈值和 \(\lambda_b\) 应在
与部署域独立的校准集上选择，
而不能在最终测试集调参。运行时队列进一步根据服务能力控制任务新鲜度。
这两层控制分别回答“值不值得”和“来不来得及”，有助于保持决策的
可解释性，并避免网络状态变化后必须重新训练收益预测头。

### 6.3 有效性威胁与局限

第一，上传器在 COCO train2017 上训练并在 val2017 上验证；跨到野外
相机、交通或低照度场景时，端侧错误分布可能变化，需要独立校准或再训练。
第二，ROI 分支在默认工作点只带来约 0.0001 mAP50-95，后续需要更好的
漏检候选生成、上下文自适应裁剪或时序区域记忆。第三，网络与队列大规模
实验主要为可复现仿真，T41 实测只有有限场景和短序列。第四，T41 NNA
存在间歇性 SIGILL，报告的端侧推理延迟为成功样本统计，因此当前原型
不能被表述为稳定量产系统。第五，板端线性头使用近似特征，尚需完成
Python/C++ 特征一致性测试。第六，100 帧 SpeciesNet 辅助框是伪参考，
必须人工核验后才能形成正式真实场景 AP 结论。

## 7 结论

本文提出 CAFRU，一种面向低功耗智能摄像头的成本感知帧级-感兴趣区域级选择性
上传方法。该方法不额外引入深度策略网络，而是利用四个轻量线性头预测
整帧和 ROI 复核的正收益概率及期望收益，并将其与上传数据量代价、有界
队列和时限门控结合。在统一部署阈值和 JPEG 质量因子 80（Q80）条件下，
CAFRU 以 42.46% 上传
比例获得 0.5765 mAP50–95，介于端侧的 0.5350 和全云端的 0.5996
之间；相对全云端，均摊计算量从 134.1 降至 79.7 GFLOPs/image，
平均端到端延迟从 128.3 ms 降至 83.6 ms。
弱网实验表明，选择性上传需要同时评价检测精度、端到端延迟和结果
新鲜度。T41 原型为 JPEG/HTTP、异步队列和线性决策器的组件级可行性
提供了实测证据，同时暴露出 NNA 稳定性、ROI 有限收益和真实标注不足
等边界。
后续工作将优先完成板端特征等价测试、人工真实场景标注、长视频闭环
评价，并研究时序记忆和动态网络代价校准。

## Statements and Declarations

### Funding

待作者补充。若无资助，建议写：The authors declare that no funds,
grants, or other support were received during the preparation of this
manuscript.

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

待作者按 CRediT 角色补充，至少包括 Conceptualization、Methodology、
Software、Validation、Writing-original draft 和 Supervision。

### Data availability

COCO 可从其官方渠道获取。本文生成的汇总表、参数配置和匿名化日志计划
在论文接收后通过公开仓库提供。包含可识别人物的 T41 原始图像在未完成
同意确认和匿名化前不公开；可提供去标识统计结果和复现实验脚本。

### Code availability

代码仓库与永久链接待补。投稿前应提供匿名审稿链接或在说明中写明代码
将在接收后发布。

### Ethics and consent

T41 室内图像仅用于目标检测系统验证，不涉及身份识别。投稿前由作者确认
被摄对象同意、图像匿名化和所在机构的伦理要求；未经确认时，不应公开
可识别原始图像。

## 参考文献

[1] Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You Only Look
Once: Unified, Real-Time Object Detection. In: CVPR, pp. 779-788 (2016).
https://doi.org/10.1109/CVPR.2016.91

[2] Redmon, J., Farhadi, A.: YOLO9000: Better, Faster, Stronger. In:
CVPR, pp. 6517-6525 (2017). https://doi.org/10.1109/CVPR.2017.690

[3] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., Ding, G.:
YOLOv10: Real-Time End-to-End Object Detection. Adv. Neural Inf. Process.
Syst. 37 (2024)

[4] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y.,
Chen, J.: DETRs Beat YOLOs on Real-time Object Detection. In: CVPR,
pp. 16965-16974 (2024). https://doi.org/10.1109/CVPR52733.2024.01605

[5] Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., Kalfaoglu, M.E.:
Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.
arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748

[6] Shuvo, M.M.H., Islam, S.K., Cheng, J., Morshed, B.I.: Efficient
Acceleration of Deep Learning Inference on Resource-Constrained Edge
Devices: A Review. Proc. IEEE 111(1), 42-91 (2023).
https://doi.org/10.1109/JPROC.2022.3226481

[7] Zhou, Z., Chen, X., Li, E., Zeng, L., Luo, K., Zhang, J.: Edge
Intelligence: Paving the Last Mile of Artificial Intelligence With Edge
Computing. Proc. IEEE 107(8), 1738-1762 (2019).
https://doi.org/10.1109/JPROC.2019.2918951

[8] Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J.,
Tang, L.: Neurosurgeon: Collaborative Intelligence Between the Cloud and
Mobile Edge. In: ASPLOS, pp. 615-629 (2017).
https://doi.org/10.1145/3037697.3037698

[9] Teerapittayanon, S., McDanel, B., Kung, H.T.: BranchyNet: Fast
Inference via Early Exiting from Deep Neural Networks. In: ICPR,
pp. 2464-2469 (2016). https://doi.org/10.1109/ICPR.2016.7900006

[10] Matsubara, Y., Levorato, M., Restuccia, F.: Split Computing and Early
Exiting for Deep Learning Applications: Survey and Research Challenges.
ACM Comput. Surv. 55(5), 1-30 (2023).
https://doi.org/10.1145/3527155

[11] Ran, X., Chen, H., Zhu, X., Liu, Z., Chen, J.: DeepDecision: A Mobile
Deep Learning Framework for Edge Video Analytics. In: IEEE INFOCOM,
pp. 1421-1429 (2018). https://doi.org/10.1109/INFOCOM.2018.8485905

[12] Yi, S., Hao, Z., Zhang, Q., Zhang, Q., Shi, W., Li, Q.: LAVEA:
Latency-Aware Video Analytics on Edge Computing Platform. In: ACM/IEEE
SEC, pp. 1-13 (2017). https://doi.org/10.1145/3132211.3134459

[13] Xu, R., Razavi, S., Zheng, R.: Edge Video Analytics: A Survey on
Applications, Systems and Enabling Techniques. IEEE Commun. Surv. Tutor.
25(4), 2951-2982 (2023). https://doi.org/10.1109/COMST.2023.3323091

[14] Zhang, B., Jin, X., Ratnasamy, S., Wawrzynek, J., Lee, E.A.:
AWStream: Adaptive Wide-Area Streaming Analytics. In: ACM SIGCOMM,
pp. 236-252 (2018). https://doi.org/10.1145/3230543.3230554

[15] Li, Y., Padmanabhan, A., Zhao, P., Wang, Y., Xu, G.H., Netravali, R.:
Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video
Analytics. In: ACM SIGCOMM, pp. 359-376 (2020).
https://doi.org/10.1145/3387514.3405874

[16] Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M.: NoScope:
Optimizing Neural Network Queries over Video at Scale. Proc. VLDB Endow.
10(11), 1586-1597 (2017). https://doi.org/10.14778/3137628.3137664

[17] Chen, T.Y.-H., Ravindranath, L., Deng, S., Bahl, P., Balakrishnan, H.:
Glimpse: Continuous, Real-Time Object Recognition on Mobile Devices. In:
ACM SenSys, pp. 155-168 (2015).
https://doi.org/10.1145/2809695.2809711

[18] Liu, L., Li, H., Gruteser, M.: Edge Assisted Real-time Object Detection
for Mobile Augmented Reality. In: ACM MobiCom, pp. 1-16 (2019).
https://doi.org/10.1145/3300061.3300116

[19] Chen, T., Bu, Y., Zeng, Y., Xie, L., Lu, S.: RegionFilter:
Region-aware Video Filtering Mechanism on Resource-Constrained Edge
Nodes. Comput. Netw. 251, 110624 (2024).
https://doi.org/10.1016/j.comnet.2024.110624

[20] Said, N., Landsiedel, O.: EdgeBoost: Confidence Boosting for Resource
Constrained Inference via Selective Offloading. Comput. Netw. 269,
111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437

[21] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of
Modern Neural Networks. In: ICML, PMLR 70, pp. 1321-1330 (2017).
https://proceedings.mlr.press/v70/guo17a.html

[22] Setyanto, A., Sasongko, T.B., Fikri, M.A., Kim, I.K.: Near-Edge
Computing Aware Object Detection: A Review. IEEE Access 12, 2989-3011
(2024). https://doi.org/10.1109/ACCESS.2023.3347548

[23] Sudha, S.K., Aji, S.: TA-EnVODT: A Cloud-Edge Integrated
Temporal-Aware Ensemble for Real-Time Video Object Detection and Tracking
Using Knowledge Projection. Knowl.-Based Syst. 326, 114067 (2025).
https://doi.org/10.1016/j.knosys.2025.114067

[24] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common Objects
in Context. In: ECCV, pp. 740-755 (2014).
https://doi.org/10.1007/978-3-319-10602-1_48
