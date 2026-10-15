# 论文定位与任务优先级计划

更新时间：2026-06-16  
第一目标期刊：Journal of Real-Time Image Processing  
论文阶段：定位冻结 + 边写边补实验

## 1. 一句话定位

本文研究的是：

```text
面向低功耗智能摄像头的实时端云协同目标检测系统。
```

更具体地说：

```text
端侧轻量模型负责实时检测；
云端强模型只复核高价值帧或 ROI；
上传决策器在检测收益、带宽、延迟和计算量之间做权衡；
目标是在弱网和低功耗摄像头条件下获得更合理的精度-成本工作点。
```

本文不是：

```text
不是提出新的 YOLO 主干网络；
不是单纯追求最高 mAP；
不是证明 T41 一定能完整运行 YOLO26s；
不是把所有图片上传云端的检测系统；
不是只做离线 COCO 指标对比。
```

## 2. 核心研究问题

论文围绕一个主问题展开：

```text
在低功耗摄像头和网络带宽受限条件下，
如何选择最值得上云的帧或局部区域，
使目标检测精度提升，同时控制上传量、延迟和云端计算成本？
```

这个问题可以拆成四个子问题：

1. 哪些图像或目标区域需要云端复核？
2. 整图上传和 ROI 上传如何选择？
3. 上传决策如何同时考虑检测收益和通信代价？
4. 端侧结果和云端结果如何融合并满足实时性约束？

## 3. 论文标题

### 3.1 中文题目

首选：

```text
面向低功耗智能摄像头的成本感知端云协同目标检测方法
```

如果更强调系统：

```text
面向低功耗智能摄像头的实时端云协同目标检测系统
```

如果更强调上传策略：

```text
一种帧级与 ROI 级联合上传的端云协同目标检测方法
```

### 3.2 英文题目

首选：

```text
Cost-Aware Edge-Cloud Collaborative Object Detection for Low-Power Smart Cameras
```

备选：

```text
A Real-Time Edge-Cloud Collaborative Detection System for Low-Power Smart Cameras
```

```text
Resource-Aware Frame-ROI Offloading for Edge-Cloud Collaborative Object Detection
```

## 4. 目标读者和审稿人关心什么

JRTIP 审稿人不会只看 mAP。他们更关心：

| 审稿人问题 | 我们必须回答 |
| --- | --- |
| 这篇文章解决了什么 real-time issue？ | 弱网、低功耗摄像头、端到端延迟、上传队列、实时复核 |
| 为什么不能直接端侧检测？ | YOLO26s 精度受限，误检漏检明显 |
| 为什么不能全部上云？ | 带宽、RTT、云端计算和功耗成本高 |
| 为什么不是普通低置信上传？ | 本文用收益-代价联合决策，并有消融证明 |
| 为什么需要 ROI？ | ROI 可降低上传量，对局部困难目标复核有价值 |
| 真实系统有没有跑通？ | T41 摄像头链路、云端复核、延迟拆解 |
| 实验是否公平？ | Edge-only、Cloud-only、Rule、Learned v1/v2 同口径比较 |
| 方法是否可复现？ | 给出架构、公式、参数、阈值、数据流和代码说明 |

因此论文必须避免只写：

```text
我们的 mAP 提升了。
```

而要写成：

```text
在实时约束下，我们用更少上传量和更低均摊计算成本获得了更好的检测质量。
```

## 5. 论文主线

建议按照下面的故事线写：

### 5.1 问题背景

低功耗摄像头部署目标检测时，端侧轻量模型速度快但精度不足；云端大模型精度高但全量上传代价高。

### 5.2 核心矛盾

```text
Edge-only: 快，但误检漏检多；
Cloud-only: 准，但带宽、延迟和计算成本高；
Rule-based offloading: 简单，但容易上传低价值样本；
```

### 5.3 本文思路

让端侧模型先检测，再由上传决策器判断当前帧或 ROI 是否值得云端复核。判断标准不是单一置信度，而是：

```text
是否可能提升检测结果
提升幅度有多大
上传要花多少带宽和时间
当前网络和队列是否允许
```

### 5.4 方法落点

本文提出：

```text
端云协同检测框架
帧级 + ROI 级上传机制
成本感知学习型上传器
端云结果融合策略
真实摄像头链路验证
```

### 5.5 实验结论

希望最终支撑这样的结论：

```text
相比 Edge-only，本文方法显著提升检测精度；
相比 Cloud-only，本文方法显著降低上传量和均摊计算量；
相比普通规则上传，本文方法获得更好的精度-成本折中；
T41 摄像头链路验证了系统在低功耗摄像头场景下的可行性。
```

## 6. 贡献点和所需证据

### 6.1 贡献 1：实时端云协同检测框架

论文表述：

```text
构建一种面向低功耗智能摄像头的端云协同目标检测框架，集成端侧轻量检测、云端强模型复核、上传决策和结果融合。
```

必须有的证据：

- 系统总体架构图；
- 端侧、云端、传输、融合模块说明；
- T41 摄像头链路跑通；
- 端到端延迟拆解。

优先级：P0。

### 6.2 贡献 2：帧级与 ROI 级双粒度上传

论文表述：

```text
提出帧级与 ROI 级双粒度上传机制，使系统既能处理整帧困难样本，也能复核局部困难目标。
```

必须有的证据：

- 帧级上传流程图；
- ROI 生成流程图；
- ROI 融合说明；
- frame-only、ROI-only、frame+ROI 消融。

优先级：P1。

说明：

如果 ROI 分支最终收益不强，论文里不要硬夸。可以写成：

```text
ROI 分支作为低带宽局部复核机制，在部分局部困难目标中有效；
当前整体收益主要来自帧级选择性上传。
```

### 6.3 贡献 3：成本感知学习型上传决策器

论文表述：

```text
设计一种基于预期收益和上传代价的学习型上传决策器，根据端侧检测特征估计云端复核价值。
```

必须有的证据：

- 决策公式；
- 学习器结构图；
- 训练标签生成方式；
- 去掉代价项的消融；
- Rule-based 与 Learned v2 对比。

优先级：P0。

### 6.4 贡献 4：多目标实验验证

论文表述：

```text
从检测精度、上传量、均摊计算量和端到端延迟等角度验证所提系统。
```

必须有的证据：

- COCO full 主结果表；
- 不同阈值工作点；
- 上传量和 GFLOPs；
- 弱网延迟；
- T41 摄像头链路；
- 可视化案例。

优先级：P0。

## 7. 任务优先级

### 7.1 P0：决定能不能投稿

这些任务必须完成，否则不建议投 JRTIP。

| 编号 | 任务 | 输出物 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 冻结论文定位和贡献点 | 本文档 | 已完成初版 |
| P0-2 | 整理 COCO full 主结果 | 主结果表 + 结果分析 | 已有基础数据，需转论文格式 |
| P0-3 | 实现 T41 摄像头到云端链路 | demo + 延迟日志 | 待做 |
| P0-4 | 端到端延迟拆解 | latency table | 待做 |
| P0-5 | 弱网实验 | 1/5/10/20 Mbps 表格 | 待做 |
| P0-6 | 代价项消融 | no-cost vs cost-aware 表格 | 待做 |
| P0-7 | 普通 ROI baseline | rule ROI vs learned 表格 | 待做 |
| P0-8 | 论文英文骨架 | manuscript v0.1 | 待做 |

### 7.2 P1：提高录用概率

这些任务强烈建议做，会明显增强说服力。

| 编号 | 任务 | 输出物 |
| --- | --- | --- |
| P1-1 | cloud512 vs cloud640 | 输入尺寸对比表 |
| P1-2 | frame-only/ROI-only/frame+ROI | 双粒度消融表 |
| P1-3 | JPEG quality 对比 | 编码质量-上传量-延迟表 |
| P1-4 | 可视化案例 | 3-5 组图 |
| P1-5 | 英文架构图 | Fig. 1-4 |

### 7.3 P2：有时间再做

| 编号 | 任务 | 价值 |
| --- | --- | --- |
| P2-1 | 第二数据集 | 提升泛化性 |
| P2-2 | 视频时序记忆 | 增强创新 |
| P2-3 | 网络状态输入学习器 | 增强方法深度 |
| P2-4 | T41 NPU 端侧模型完整部署 | 提升硬件实验完整性 |

## 8. 近期执行计划

### 阶段 1：定位和论文骨架，1-2 天

目标：

```text
把论文讲什么固定下来。
```

任务：

1. 确认题目；
2. 确认摘要主线；
3. 确认 4 个贡献点；
4. 把中文初稿压缩成 JRTIP 结构；
5. 开始英文 manuscript v0.1。

输出：

```text
PAPER_POSITIONING_AND_TASK_PLAN_ZH.md
英文论文骨架
图表清单
```

### 阶段 2：COCO 结果论文格式化，2-3 天

目标：

```text
把已有结果变成论文表格和论述。
```

任务：

1. 整理 Edge-only、Cloud-only、Rule、Learned v1/v2 总表；
2. 整理阈值扫描工作点；
3. 画 mAP-upload-latency 曲线；
4. 写 Results 初稿；
5. 解释为什么协同 mAP50-95 可能高于 Cloud-only。

输出：

```text
Table 1 主结果
Table 2 阈值工作点
Fig. 5 精度-成本曲线
Results v0.1
```

### 阶段 3：T41 链路跑通，1 周

目标：

```text
拿到真实摄像头链路证据。
```

任务：

1. T41 摄像头取帧；
2. JPEG 编码；
3. 上传到云端；
4. 云端 YOLO26x 推理；
5. 返回检测框；
6. 记录每阶段时间戳；
7. 保存带框视频或图片。

输出：

```text
T41 demo
T41 latency table
T41 visualization cases
```

### 阶段 4：补核心消融，1 周

目标：

```text
证明方法各模块确实有用。
```

任务：

1. 代价项消融；
2. 普通 ROI baseline；
3. 弱网实验；
4. cloud512 vs cloud640；
5. frame-only vs frame+ROI。

输出：

```text
ablation tables
weak-network tables
cloud_imgsz comparison
```

### 阶段 5：英文稿整合，1 周

目标：

```text
形成第一版可内部评审的英文稿。
```

任务：

1. 完成 Introduction；
2. 完成 Related Work；
3. 完成 Method；
4. 完成 Experiments；
5. 完成 Results and Discussion；
6. 图表英文化；
7. 检查 12 页限制。

输出：

```text
manuscript v1
figures v1
tables v1
```

## 9. 当前写作口径

### 9.1 可以强调

- 真实低功耗摄像头场景；
- 实时端云协同；
- 帧级与 ROI 级双粒度上传；
- 成本感知学习型决策；
- 精度、上传量、延迟、GFLOPs 综合评估；
- COCO full + T41 摄像头链路。

### 9.2 需要谨慎

- 不要说“全面超过云端模型”；
- 不要说“端侧 YOLO26s 已部署在 T41 上”，除非确实完成；
- 不要把 ROI 分支收益夸大；
- 不要只用 COCO 结果证明真实部署；
- 不要把估算延迟当成真实链路延迟。

### 9.3 推荐表述

```text
The proposed system improves the accuracy of edge-only detection while substantially reducing the upload ratio and average cloud computation compared with cloud-only inference.
```

```text
The T41 camera experiment validates the feasibility of the camera acquisition, compression, transmission, cloud review, and result return pipeline under low-power edge-camera constraints.
```

如果 T41 没有跑 YOLO26s：

```text
The current T41 prototype validates the real-time camera-to-cloud pipeline, while the edge detector is evaluated using the same lightweight model in the offline benchmark. Full NPU deployment on T41 is left as an engineering optimization.
```

## 10. 论文成败判断标准

达到下面标准，可以考虑投稿 JRTIP：

```text
1. 英文稿 12 页以内；
2. 明确 real-time issue；
3. COCO full 主结果完整；
4. 至少一个真实 T41 摄像头链路实验；
5. 有端到端延迟拆解；
6. 有弱网实验；
7. 有 cost-aware 消融；
8. 有普通 ROI baseline；
9. 图表完整；
10. 讨论限制，不夸大。
```

如果 P0 实验补不齐，则建议先转投更工程应用或中文方向。

