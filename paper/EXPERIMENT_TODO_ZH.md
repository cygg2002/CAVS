# 端云协同目标检测论文补实验清单

更新时间：2026-06-16

## 1. 当前已有实验

已有结果可直接进入论文初稿：

- COCO full 上的 Edge-only、Cloud-only、Rule aggressive、Learned v1、Learned v2 对比；
- Precision、Recall、mAP50、mAP50-95；
- 上传图像比、KB/image、GFLOPs/image、E2E@10Mbps；
- 学习器 v2 多阈值工作点；
- 端云协同系统架构和延迟拆解文档；
- ROI 机制和单图可视化样例。

当前主结果：

```text
Edge-only YOLO26s:        mAP50-95 = 0.4733
Cloud-only YOLO26x:       mAP50-95 = 0.5656
Learned v2 f0.50/r0.05:   mAP50-95 = 0.5918, upload ratio = 0.5656
Learned v2 f0.60/r0.08:   mAP50-95 = 0.5829, upload ratio = 0.4248
Learned v2 f0.70/r0.10:   mAP50-95 = 0.5704, upload ratio = 0.2442
```

## 2. P0：必须补的实验

这些实验决定论文是否完整。

### 2.1 T41 + 摄像头真实链路

目标：证明系统不是只在离线 COCO 上跑，而是可以接入低功耗摄像头端侧设备。

最低要求：

```text
T41 摄像头采集
JPEG 编码
HTTP/multipart 上传
云端 YOLO26x 推理
结果返回
端侧或本地显示带框结果
```

需要记录：

| 指标 | 说明 |
| --- | --- |
| capture_ms | 摄像头取帧耗时 |
| jpeg_encode_ms | JPEG 编码耗时 |
| upload_ms | 图像上传耗时 |
| cloud_infer_ms | 云端 YOLO26x 推理耗时 |
| fusion_ms | 端云结果融合耗时 |
| download_ms | 结果返回耗时 |
| total_e2e_ms | 从取帧到获得最终框的总延迟 |

注意：

如果 T41 暂时跑不了 YOLO26s，不要宣称端侧 YOLO26s 已部署。可以写：

```text
T41 用于验证摄像头采集、编码、上传和云端复核链路；
端侧检测算法在 COCO 实验平台验证；
T41 NPU 模型部署作为后续优化。
```

### 2.2 弱网条件实验

目标：证明成本感知上传策略在弱网场景有意义。

建议带宽：

```text
1 Mbps
5 Mbps
10 Mbps
20 Mbps
```

建议 RTT：

```text
20 ms
50 ms
100 ms
```

需要输出表：

| 带宽 | 策略 | 上传比例 | KB/image | 平均延迟 | 95%延迟 |
| --- | --- | ---: | ---: | ---: | ---: |

### 2.3 普通 ROI+云复核 baseline

目标：证明本文方法不只是普通 ROI 上传。

baseline 设计：

```text
低置信框上传 ROI
小目标上传 ROI
贴边目标上传 ROI
max_roi_per_image = 1 或 2
```

比较指标：

- mAP50-95；
- 上传图像比；
- KB/image；
- E2E latency；
- ROI 上传次数。

### 2.4 去掉代价项的消融

目标：证明 `cost-aware` 不是装饰。

对比：

```text
score = p_gain * expected_gain
score = p_gain * expected_gain - cost_weight * payload_kb / 100
```

预期结论：

```text
无代价项会上传更多样本；
精度可能略高或相近；
但 KB/image、延迟和云端计算明显增加；
代价项提供更合理的精度-成本折中。
```

## 3. P1：强烈建议补的实验

这些实验会明显提升投稿说服力。

### 3.1 cloud_imgsz=512 vs 640

目标：回答审稿人可能提出的问题：为什么云端用 512 而不是 640。

需要同口径比较：

```text
Learned v2 f0.50/r0.05 cloud512 vs cloud640
Learned v2 f0.60/r0.08 cloud512 vs cloud640
Learned v2 f0.70/r0.10 cloud512 vs cloud640
```

指标：

- mAP50；
- mAP50-95；
- GFLOPs/image；
- E2E@10Mbps；
- cloud_infer_ms。

### 3.2 只整图上传 vs 整图+ROI

目标：证明双粒度设计有效，或者实事求是说明 ROI 当前收益有限。

对比：

```text
frame-only learned
roi-only learned
frame+roi learned
```

如果 ROI 贡献不明显，可以写：

```text
当前 COCO full 上收益主要来自帧级选择性上传；
ROI 分支在小目标和局部复杂场景中具有补充价值；
未来会针对视频和小目标场景进一步优化 ROI 分支。
```

### 3.3 真实视频可视化案例

至少准备 3 类案例：

| 案例 | 说明 |
| --- | --- |
| 漏检修正 | 端侧漏检，云端整图或 ROI 检出 |
| 误检抑制 | 端侧误检，云端复核后删除 |
| 不上传成功 | 端侧结果足够好，系统避免无效上传 |

每类准备：

```text
原图
端侧结果
上传区域
云端结果
融合结果
决策分数
```

### 3.4 T41 不同 JPEG quality

建议测试：

```text
quality = 60, 70, 80, 90
```

指标：

- JPEG 编码大小；
- 编码耗时；
- 云端检测 mAP 或单视频检测稳定性；
- 上传延迟。

## 4. P2：有时间再补的实验

### 4.1 第二数据集

如果时间允许，增加一个非 COCO 数据集：

| 数据集 | 优点 | 风险 |
| --- | --- | --- |
| VOC | 容易跑，类别和 COCO 接近 | 难度较低 |
| VisDrone | 小目标多，适合 ROI | 需要模型适配 |
| BDD100K | 自动驾驶摄像头场景 | 数据较大 |
| Caltech Camera Traps | 野外摄像头贴合应用 | 类别与 COCO 不完全一致，需要训练 |

如果时间紧，不强求第二数据集，但至少要有 T41 真实视频链路。

### 4.2 网络状态感知学习器

把以下变量加入决策特征：

```text
uplink_mbps
RTT_ms
queue_len
packet_loss
```

当前可以先用公式代价项实现，不一定重新训练学习器。

### 4.3 视频时序记忆

减少连续帧重复上传：

```text
同一目标连续稳定出现 -> 降低上传优先级
上一帧刚上传整图 -> 当前帧提高上传阈值
目标位置变化剧烈 -> 提高上传优先级
```

这适合作为后续工作或扩展实验。

## 5. 论文图表清单

### 5.1 必备图

| 图编号 | 内容 |
| --- | --- |
| Fig. 1 | 系统总体架构图 |
| Fig. 2 | 学习型上传决策器结构 |
| Fig. 3 | 端云数据流与延迟拆解 |
| Fig. 4 | ROI 生成与融合流程 |
| Fig. 5 | 精度-上传量-延迟曲线 |
| Fig. 6 | T41 摄像头链路实物/流程图 |
| Fig. 7 | 可视化案例 |

### 5.2 必备表

| 表编号 | 内容 |
| --- | --- |
| Table 1 | 方法对比总表 |
| Table 2 | 消融实验 |
| Table 3 | 不同阈值工作点 |
| Table 4 | 不同带宽延迟 |
| Table 5 | T41 真实链路延迟拆解 |
| Table 6 | 与普通 ROI+云复核对比 |

## 6. 论文最低完成标准

如果目标是“稳妥够二类”，最低需要完成：

```text
1. COCO full 主结果表
2. Rule / Learned v1 / Learned v2 对比
3. 代价项消融
4. 普通 ROI+云复核 baseline
5. T41+摄像头链路延迟
6. 弱网带宽实验
7. 3 个可视化案例
8. 完整系统架构和公式说明
```

完成这些后，可以先投：

```text
Journal of Real-Time Image Processing
或
Machine Vision and Applications
```

如果被拒，再根据审稿意见转：

```text
The Journal of Supercomputing
IEEE Access
Sensors
国内中文核心/科技核心
```

