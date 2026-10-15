# JRTIP 投稿完成路线

目标期刊：Journal of Real-Time Image Processing  
出版社：Springer Nature  
暂定题目：Cost-Aware Edge-Cloud Collaborative Object Detection for Low-Power Smart Cameras  
中文题目：面向低功耗智能摄像头的成本感知端云协同目标检测方法  
更新时间：2026-06-16

## 1. 为什么暂定 JRTIP

JRTIP 关注实时图像/视频处理算法、系统、架构和实际实时性能。本项目的核心不是提出一个新 YOLO 检测器，而是：

```text
在低功耗摄像头场景下，
通过端侧实时检测、学习型上传决策和云端复核，
在检测精度、上传量、端到端延迟和计算量之间取得折中。
```

这和 JRTIP 的方向匹配。投稿时必须把论文写成“实时图像处理系统与方法”，不要只写成“COCO 上 mAP 提升”。

## 2. 官方投稿约束

根据 JRTIP 官方投稿指南，需要特别注意：

1. 论文有页数限制：自 2020-01-01 起，稿件限制为双栏格式 12 页，包含参考文献和作者简介。
2. 不符合页数限制和双栏格式的稿件会被退回，不进入审稿流程。
3. 投稿时需要提供 cover letter，简要说明稿件背景和重要性。
4. cover letter 中必须说明论文解决了哪些具体 real-time issues。
5. 官方明确指出，只报告处理时间或速度并不算充分处理实时问题，论文必须完整讨论实时性问题。
6. 投稿需要上传可编辑源文件，通常为 Word 或 LaTeX。

因此本文写作必须围绕：

```text
实时性约束
弱网条件下的端到端延迟
端侧低功耗摄像头链路
上传队列和过期任务处理
精度-延迟-带宽-计算量折中
```

## 3. 论文最终定位

推荐定位：

```text
一种面向低功耗智能摄像头的实时端云协同目标检测系统。
```

避免定位：

```text
一种新的 YOLO 目标检测算法。
```

核心问题表述：

```text
端侧轻量模型实时性好但精度有限；
云端强模型精度高但全量上传代价高；
本文研究如何在实时约束下选择高价值帧或 ROI 上云复核。
```

## 4. 论文贡献点

JRTIP 版本建议保留 4 个贡献点：

1. 构建低功耗摄像头场景下的实时端云协同目标检测框架，包含端侧检测、上传决策、云端复核和结果融合。
2. 提出帧级与 ROI 级双粒度上传机制，使系统能够在整图复核和局部复核之间自适应选择。
3. 设计成本感知学习型上传决策器，以 `预测收益 - 通信代价` 的形式联合建模检测收益、上传量、网络状态和队列状态。
4. 在 COCO full 和 T41 摄像头链路上验证系统的精度、上传量、GFLOPs、端到端延迟和实时性。

## 5. 必须补齐的实验

### 5.1 P0 实验：没有这些不建议投稿

| 编号 | 实验 | 目的 | 完成标准 |
| --- | --- | --- | --- |
| E1 | COCO full 主结果整理 | 证明算法有效 | Edge-only、Cloud-only、Rule、Learned v1/v2 总表 |
| E2 | T41 摄像头链路 | 证明低功耗摄像头可行性 | 采集、JPEG、上传、云端检测、返回结果全链路跑通 |
| E3 | 端到端延迟拆解 | 满足 JRTIP 实时性要求 | capture、encode、upload、cloud、fusion、return、total |
| E4 | 弱网实验 | 证明实时约束下的优势 | 1/5/10/20 Mbps，RTT 20/50/100 ms |
| E5 | 代价项消融 | 证明 cost-aware 有用 | 有无代价项对比 mAP、KB/image、延迟 |
| E6 | 普通 ROI baseline | 证明优于普通 ROI+云复核 | 低置信/小目标/贴边 ROI baseline |
| E7 | 可视化案例 | 增强可解释性 | 3 类案例：漏检修正、误检抑制、不上传成功 |

### 5.2 P1 实验：强烈建议补齐

| 编号 | 实验 | 目的 |
| --- | --- | --- |
| E8 | cloud_imgsz=512 vs 640 | 回答为什么默认 512 |
| E9 | frame-only vs ROI-only vs frame+ROI | 证明双粒度设计边界 |
| E10 | JPEG quality 60/70/80/90 | 分析编码质量对带宽和检测的影响 |
| E11 | 队列长度 0/1/2/4 | 分析实时上传队列对延迟和丢弃的影响 |

### 5.3 P2 实验：有时间再做

| 编号 | 实验 | 目的 |
| --- | --- | --- |
| E12 | 第二数据集 | 提升泛化说服力 |
| E13 | 视频时序记忆 | 减少连续帧重复上传 |
| E14 | 网络状态直接输入学习器 | 形成更强方法创新 |

## 6. JRTIP 论文结构

建议控制在 12 页双栏内，优先压缩相关工作和实现细节，把空间留给实时性实验。

```text
Title
Abstract
Keywords

1. Introduction
2. Related Work
3. Proposed Edge-Cloud Real-Time Detection System
4. Cost-Aware Frame-ROI Offloading Method
5. Experimental Setup
6. Results and Discussion
7. Conclusion
References
```

### 6.1 各节写作重点

| 章节 | 页数建议 | 重点 |
| --- | ---: | --- |
| Abstract | 0.2 | 问题、方法、结果、实时性 |
| Introduction | 1.0 | 端侧精度不足、云端全量上传代价、本文贡献 |
| Related Work | 1.0 | 轻量检测、端云协同、实时视觉系统 |
| System | 1.5 | T41 摄像头、端侧模块、云端模块、通信模块 |
| Method | 2.0 | 图像/ROI特征、上传分数、学习器、融合 |
| Experiments | 1.5 | 数据集、设备、网络、baseline、指标 |
| Results | 3.0 | 主表、消融、弱网、T41延迟、可视化 |
| Conclusion | 0.3 | 总结和限制 |
| References | 1.0-1.5 | 控制数量，优先近五年和核心相关工作 |

## 7. 图表规划

### 7.1 必备图

| 图 | 内容 | 状态 |
| --- | --- | --- |
| Fig. 1 | 系统总体架构 | 待整理成英文图 |
| Fig. 2 | 成本感知学习型上传器结构 | 待整理成英文图 |
| Fig. 3 | 帧级/ROI级决策流程 | 待整理成英文图 |
| Fig. 4 | 端到端延迟链路 | 待补 T41 数据 |
| Fig. 5 | 上传比例-mAP-延迟曲线 | 待根据扫描结果绘图 |
| Fig. 6 | 可视化案例 | 待补真实摄像头案例 |

### 7.2 必备表

| 表 | 内容 | 状态 |
| --- | --- | --- |
| Table 1 | 与 Edge-only、Cloud-only、Rule、Learned v1/v2 对比 | 已有基础数据 |
| Table 2 | 不同阈值工作点 | 已有基础数据 |
| Table 3 | 消融实验 | 待补 |
| Table 4 | 弱网条件延迟 | 待补 |
| Table 5 | T41 真实链路延迟拆解 | 待补 |
| Table 6 | 普通 ROI baseline 对比 | 待补 |

## 8. 时间安排

如果按边写边实验推进，建议 4-6 周完成第一版投稿稿件。

### 第 1 周：论文骨架和实验脚本整理

任务：

- 固定 JRTIP 论文题目、摘要和贡献点；
- 把中文初稿改成英文论文骨架；
- 整理 COCO full 主结果；
- 确认 T41 端侧能完成哪些环节；
- 确认云端服务和上传接口。

交付：

```text
英文论文 v0.1
COCO 主结果表
T41 实验方案
```

### 第 2 周：T41 摄像头链路跑通

任务：

- T41 摄像头采集；
- JPEG 编码；
- 上传到云端；
- 云端 YOLO26x 推理；
- 返回检测框；
- 保存每阶段时间戳。

交付：

```text
T41 full-frame cloud review demo
T41 latency breakdown v1
```

### 第 3 周：弱网和消融实验

任务：

- 弱网 1/5/10/20 Mbps；
- RTT 20/50/100 ms；
- 代价项消融；
- 普通 ROI baseline；
- cloud512 vs cloud640。

交付：

```text
weak-network tables
ablation tables
cloud_imgsz comparison
```

### 第 4 周：图表和论文初稿

任务：

- 绘制全部英文图；
- 整理表格；
- 写 Results and Discussion；
- 写 Limitations；
- 准备 cover letter 初稿。

交付：

```text
manuscript v1
figures v1
cover letter v0
```

### 第 5 周：内部检查与英文润色

任务：

- 检查页数是否超过 12 页；
- 检查实时性论述是否充分；
- 检查所有实验是否可复现；
- 英文润色；
- 补参考文献。

交付：

```text
submission-ready manuscript
supplementary material
cover letter
```

### 第 6 周：投稿

任务：

- 根据 JRTIP 模板排版；
- 上传 Word 或 LaTeX 源文件；
- 上传图表；
- 填写 authorship、funding、competing interests、data availability；
- 提交到 Springer 投稿系统；
- 保存投稿记录。

## 9. T41 实验最低实现方案

为了尽快满足 JRTIP 的实时系统验证，T41 第一版不必一开始完整跑 YOLO26s。

最低可接受链路：

```text
T41 摄像头取帧
T41 JPEG 编码
T41 上传整图/ROI 到云端
云端 YOLO26x 检测
云端或端侧融合
返回检测结果
记录全链路延迟
```

如果 T41 暂时无法运行 YOLO26s，论文中必须明确区分：

```text
算法评估：COCO full，端侧模型 YOLO26s，云端模型 YOLO26x
硬件链路评估：T41 摄像头，验证采集/编码/传输/云端复核实时性
```

不能写成：

```text
YOLO26s 已完整部署在 T41 NPU 上
```

除非确实完成了 T41 NPU 推理和测量。

## 10. Cover Letter 要点

JRTIP 的 cover letter 必须强调 real-time issues。建议包含：

```text
1. 低功耗摄像头端侧实时检测面临精度与资源矛盾；
2. 全量云端检测无法满足弱网实时性；
3. 本文提出成本感知端云协同检测，在实时约束下选择高价值帧/ROI上云；
4. 论文不只报告速度，还系统分析了端到端延迟、上传队列、弱网带宽、计算量和检测精度之间的折中；
5. T41 摄像头链路验证了低功耗摄像头场景的可行性。
```

## 11. 投稿前检查清单

### 11.1 内容检查

- [ ] 是否明确 real-time issues，而不只是报告 FPS；
- [ ] 是否说明 T41 的真实角色和限制；
- [ ] 是否有端到端延迟拆解；
- [ ] 是否有弱网实验；
- [ ] 是否有代价项消融；
- [ ] 是否有普通 ROI baseline；
- [ ] 是否有可视化案例；
- [ ] 是否没有夸大 mAP 高于 Cloud-only 的原因；
- [ ] 是否如实说明 ROI 分支当前收益边界；
- [ ] 是否讨论跨数据集泛化和重训练问题。

### 11.2 格式检查

- [ ] 双栏格式；
- [ ] 12 页以内，包含参考文献和作者简介；
- [ ] 图表英文；
- [ ] 所有图表在正文中引用；
- [ ] 参考文献格式统一；
- [ ] 提供 Word 或 LaTeX 可编辑源文件；
- [ ] Cover letter 一页以内；
- [ ] Data availability statement；
- [ ] Competing interests；
- [ ] Funding；
- [ ] Author contributions。

### 11.3 投稿材料

- [ ] Manuscript PDF；
- [ ] Manuscript source files；
- [ ] Figures；
- [ ] Cover letter；
- [ ] Supplementary material；
- [ ] Code/data availability statement；
- [ ] Author information；
- [ ] ORCID；
- [ ] Funding information；
- [ ] Suggested reviewers，可选；
- [ ] Opposed reviewers，可选。

## 12. 风险和应对

| 风险 | 可能审稿意见 | 应对 |
| --- | --- | --- |
| T41 未跑 YOLO26s | 硬件实验不完整 | 如实区分算法实验和链路实验，补充轻量模型或 NPU 迁移计划 |
| 只在 COCO 上验证 | 泛化不足 | 补真实摄像头视频或第二数据集 |
| ROI 收益弱 | 双粒度贡献不足 | 把 ROI 作为补充机制，主贡献转为成本感知 frame/ROI unified policy |
| 实时性论述不足 | 不符合 JRTIP | 强化弱网、队列、延迟拆解、实时工作点 |
| 创新不足 | 只是工程集成 | 强调收益-代价学习型上传、目标级 utility 监督和多目标实验 |
| mAP 高于 Cloud-only 被质疑 | 评估不公平 | 明确 Cloud-only 不是 oracle，上报同口径 deployment baseline 和 mAP 统计细节 |

## 13. 当前下一步

立即执行顺序：

```text
1. 把中文初稿转成英文 JRTIP 结构；
2. 确认 T41 摄像头 SDK/取帧方式；
3. 实现 T41 -> 云端 HTTP/multipart 上传；
4. 云端增加 latency breakdown 返回；
5. 补普通 ROI baseline 和代价项消融；
6. 绘制第一版英文架构图和实验曲线。
```

第一版投稿前，最小闭环是：

```text
COCO full algorithm evidence
+ T41 camera real-time transmission evidence
+ weak-network latency evidence
+ ablation evidence
```

