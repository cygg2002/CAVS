# 端云协同目标检测补充实验执行计划

更新时间：2026-06-16  
目标论文：JRTIP 中文主稿 v0.1  
目标：先补齐 JRTIP 投稿所需的实时性、消融和系统证据

## 1. 实验总原则

所有补充实验都必须服务论文主线：

```text
在低功耗摄像头和弱网条件下，
通过成本感知端云协同，
提升端侧检测精度，同时降低上传量、端到端延迟和均摊云端计算量。
```

实验不追求越多越好，而是优先回答审稿人最可能问的关键问题：

1. 这个系统是否只是离线 COCO 实验，还是有真实摄像头链路？
2. 方法是否真的解决 real-time issue，而不是只报告 mAP？
3. 学习型上传器是否优于普通规则上传？
4. 成本项是否真的有用？
5. ROI 分支和整图分支分别贡献什么？
6. 弱网下为什么比 Cloud-only 更合理？

## 2. 优先级总览

### P0：投稿前必须完成

| 编号 | 实验 | 对应论文问题 | 输出 |
| --- | --- | --- | --- |
| E0 | 整理已有 COCO full 主结果 | 方法是否有效 | Table 1、Table 2、曲线图 |
| E1 | T41 摄像头端云链路 | 是否有真实低功耗摄像头证据 | Table 5、链路图、demo 图 |
| E2 | 端到端延迟拆解 | 是否解决 real-time issue | latency breakdown |
| E3 | 弱网实验 | 弱网下是否优于 Cloud-only | Table 4、延迟曲线 |
| E4 | 成本项消融 | cost-aware 是否有效 | ablation table |
| E5 | 普通 ROI baseline | 是否优于普通 ROI+云复核 | baseline table |
| E6 | 可视化案例 | 是否可解释 | Fig. 6/7 |

### P1：强烈建议完成

| 编号 | 实验 | 价值 |
| --- | --- | --- |
| E7 | cloud_imgsz=512 vs 640 | 回答云端输入尺寸选择 |
| E8 | frame-only / ROI-only / frame+ROI | 证明双粒度边界 |
| E9 | JPEG quality 60/70/80/90 | 分析编码质量和带宽 |
| E10 | 上传队列长度 0/1/2/4 | 分析实时队列机制 |

### P2：有时间再做

| 编号 | 实验 | 价值 |
| --- | --- | --- |
| E11 | 第二数据集或真实视频标注 | 泛化性 |
| E12 | 视频时序记忆 | 减少连续帧重复上传 |
| E13 | 网络状态输入学习器 | 增强方法创新 |
| E14 | T41 NPU 完整 YOLO26s 部署 | 完整硬件闭环 |

## 3. E0：已有 COCO full 主结果整理

### 3.1 目的

把已经得到的 COCO full 结果整理成论文表格和曲线，形成论文第一组核心证据。

### 3.2 需要回答

```text
相比 Edge-only，协同系统是否提升精度？
相比 Cloud-only，协同系统是否降低均摊计算和上传成本？
不同阈值是否能形成精度-成本可调工作点？
```

### 3.3 输入数据

已有结果来自：

```text
runs/edge_cloud/*/edge_cloud_eval_summary.json
runs/edge_cloud/*/strategy_summary.csv
examples/YOLO26-Edge-Cloud/EDGE_CLOUD_ARCHITECTURE_LATENCY_REPORT_ZH.md
```

当前主结果：

| 方法 | Precision | Recall | mAP50 | mAP50-95 | 上传图像比 | KB/image | GFLOPs/image | E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge-only YOLO26s | 0.6913 | 0.5781 | 0.6346 | 0.4733 | 0 | 0 | 22.8 | 18.2 ms |
| Cloud-only YOLO26x | 0.7383 | 0.6700 | 0.7327 | 0.5656 | 1.0 | 全图 | 209.5 | 取决于网络 |
| Rule aggressive | 0.7318 | 0.5832 | 0.6817 | 0.5486 | 0.6406 | 22.94 | 168.1 | 84.1 ms |
| Learned v1 f0.60/r0.08 | 0.7622 | 0.6108 | 0.7071 | 0.5744 | 0.6686 | 34.77 | 112.5 | 97.0 ms |
| Learned v2 f0.50/r0.05 | 0.7835 | 0.6202 | 0.7251 | 0.5918 | 0.5656 | 34.90 | 98.7 | 85.4 ms |
| Learned v2 f0.60/r0.08 | 0.7758 | 0.6092 | 0.7163 | 0.5829 | 0.4248 | 26.55 | 79.8 | 71.7 ms |
| Learned v2 f0.70/r0.10 | 0.7671 | 0.5919 | 0.7032 | 0.5704 | 0.2442 | 15.87 | 55.6 | 49.9 ms |
| Learned v2 f0.80/r0.10 | 0.7519 | 0.5748 | 0.6861 | 0.5518 | 0.0920 | 6.28 | 35.2 | 32.1 ms |

### 3.4 具体要做

1. 统一所有结果的小数位，论文表格保留 3 或 4 位。
2. 画一张 trade-off 图：

```text
x 轴: upload ratio 或 KB/image
y 轴: mAP50-95
点大小或颜色: E2E@10Mbps / GFLOPs/image
```

3. 画一张 latency-cost 图：

```text
x 轴: KB/image
y 轴: E2E@10Mbps
标注不同策略
```

4. 写一段结果分析：

```text
Learned v2 f0.60/r0.08 是默认平衡点；
Learned v2 f0.50/r0.05 是精度优先点；
Learned v2 f0.70/r0.10 是弱网优先点；
```

### 3.5 输出文件

建议输出：

```text
paper/figures/coco_tradeoff_map_upload_latency.png
paper/figures/coco_cost_latency_curve.png
paper/tables/coco_main_results.csv
```

### 3.6 成功标准

论文中能够形成：

```text
Table 1: Main results on COCO full
Fig. 5: Accuracy-cost-latency trade-off
```

## 4. E1：T41 摄像头端云链路实验

### 4.1 目的

证明系统不是只有离线 COCO 结果，而是能够接入低功耗摄像头端侧设备，完成真实采集、编码、上传、云端复核和结果返回。

### 4.2 当前现实约束

现阶段只要求 T41 完成：

```text
摄像头采集
JPEG 编码
上传整图或 ROI
接收云端返回结果
保存延迟日志
```

如果 YOLO26s 尚未部署到 T41 NPU，不能在论文中写成“YOLO26s 已在 T41 上运行”。应写成：

```text
T41 prototype validates the real-time camera-to-cloud pipeline.
The lightweight edge detector is evaluated in the COCO benchmark.
Full T41 NPU deployment is left as an engineering optimization.
```

### 4.3 实验链路

最低链路：

```text
T41 camera
  -> frame capture
  -> JPEG encode
  -> HTTP multipart upload
  -> cloud server receives image
  -> YOLO26x inference
  -> return boxes + latency
  -> T41/local client logs result
```

如果端侧检测可用，则完整链路为：

```text
T41 camera
  -> YOLO26s/T41-light detector
  -> upload decision
  -> JPEG full-frame or ROI upload
  -> cloud YOLO26x
  -> fusion
  -> final boxes
```

### 4.4 云端需要实现或确认

云端接口建议：

```text
POST /infer_multipart
```

请求包含：

```text
metadata.json
image.jpg
```

metadata 字段：

```text
frame_id
timestamp_capture
payload_type
orig_shape
crop_xyxy
edge_boxes
jpeg_quality
policy_name
decision_score
```

云端返回：

```json
{
  "frame_id": 1,
  "cloud_boxes": [],
  "final_boxes": [],
  "latency": {
    "cloud_receive_ms": 0.0,
    "decode_ms": 0.0,
    "cloud_infer_ms": 0.0,
    "fusion_ms": 0.0,
    "server_total_ms": 0.0
  }
}
```

### 4.5 T41 端需要记录

每一帧记录：

```text
frame_id
capture_start_ts
capture_end_ts
jpeg_start_ts
jpeg_end_ts
upload_start_ts
upload_end_ts
response_recv_ts
payload_bytes
payload_type
cloud_boxes_count
final_boxes_count
```

计算：

```text
capture_ms = capture_end - capture_start
jpeg_encode_ms = jpeg_end - jpeg_start
upload_roundtrip_ms = response_recv - upload_start
total_e2e_ms = response_recv - capture_start
```

如果 T41 端侧检测可用，额外记录：

```text
edge_infer_ms
postprocess_ms
policy_ms
roi_crop_ms
```

### 4.6 测试样本

至少测试：

```text
30 秒实时摄像头视频
或
300 帧连续采集
```

场景至少 3 类：

1. 简单场景：目标清晰、背景简单；
2. 复杂场景：背景复杂、多人/多物体；
3. 低质量场景：运动、遮挡、远距离、小目标。

### 4.7 输出表格

论文 Table 5：

| 阶段 | 平均延迟 ms | P50 ms | P95 ms | 说明 |
| --- | ---: | ---: | ---: | --- |
| capture | 待补 | 待补 | 待补 | T41 摄像头取帧 |
| jpeg_encode | 待补 | 待补 | 待补 | T41 JPEG 编码 |
| upload_roundtrip | 待补 | 待补 | 待补 | 上传+返回 |
| cloud_decode | 待补 | 待补 | 待补 | 云端解码 |
| cloud_infer | 待补 | 待补 | 待补 | YOLO26x |
| fusion | 待补 | 待补 | 待补 | 结果融合 |
| total_e2e | 待补 | 待补 | 待补 | 端到端 |

### 4.8 成功标准

最低成功标准：

```text
T41 能稳定采集图像；
能上传 JPEG 到云端；
云端能返回检测框；
能保存每帧 latency CSV；
能给出平均延迟和 P95 延迟；
能生成 2-3 张带框示例图。
```

## 5. E2：端到端延迟拆解实验

### 5.1 目的

JRTIP 要求论文讨论 real-time issues。端到端延迟拆解用于证明系统不是只报告模型推理时间，而是分析完整实时链路。

### 5.2 实验对象

至少比较：

```text
Edge-only
Cloud-only full-frame
Learned v2 f0.60/r0.08
Learned v2 f0.70/r0.10
```

如果 T41 端侧检测未完成，则 T41 链路只比较：

```text
Cloud review full-frame
Selective upload simulated by preselected frames/ROI
```

COCO 离线评估中则继续比较完整策略。

### 5.3 记录字段

```text
capture_ms
edge_infer_ms
postprocess_ms
policy_ms
jpeg_encode_ms
queue_wait_ms
upload_ms
cloud_decode_ms
cloud_infer_ms
fusion_ms
download_ms
render_ms
total_e2e_ms
```

### 5.4 输出图

建议画堆叠柱状图：

```text
x 轴: strategy
y 轴: latency ms
堆叠: capture / edge / encode / upload / cloud / fusion / return
```

输出：

```text
paper/figures/latency_breakdown_stacked.png
```

### 5.5 成功标准

论文能够说明：

```text
Cloud-only 的延迟瓶颈主要来自上传和云端推理；
Learned v2 通过降低上传比例降低平均端到端延迟；
Edge-only 延迟最低但精度不足；
```

## 6. E3：弱网条件实验

### 6.1 目的

证明本文方法在弱网条件下比 Cloud-only 更适合实时摄像头。

### 6.2 实验设置

带宽：

```text
1 Mbps
5 Mbps
10 Mbps
20 Mbps
```

RTT：

```text
20 ms
50 ms
100 ms
```

策略：

```text
Cloud-only
Rule aggressive
Learned v2 f0.50/r0.05
Learned v2 f0.60/r0.08
Learned v2 f0.70/r0.10
```

### 6.3 实现方式

优先级从易到难：

1. 先用已有 payload size 和 cloud inference time 做离线估算；
2. 再在 Linux 云端或中间路由器上用 `tc netem` 模拟带宽和 RTT；
3. 最后在 T41 实际 WiFi/4G 网络下测量。

估算公式：

```text
upload_ms = payload_kb * 8 / uplink_mbps
network_ms = RTT_ms + upload_ms
e2e_ms = edge_ms + encode_ms + queue_ms + network_ms + cloud_ms + fusion_ms
```

注意单位：

```text
KB * 8 / Mbps = ms 近似成立，因为 1 KB = 8 Kb，1 Mbps = 1000 Kb/s
```

### 6.4 输出表格

| 带宽 | RTT | 策略 | 上传比例 | KB/image | 平均延迟 ms | P95 延迟 ms | mAP50-95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

### 6.5 输出图

建议图：

```text
Fig. weak_network_latency.png
x 轴: bandwidth
y 轴: average E2E latency
不同线: Cloud-only / Rule / Learned v2
```

```text
Fig. weak_network_map_latency.png
x 轴: E2E latency
y 轴: mAP50-95
点大小: KB/image
```

### 6.6 成功标准

应能支撑结论：

```text
带宽越低，Cloud-only 延迟越高；
Learned v2 由于上传比例低，在弱网下平均延迟明显低于 Cloud-only；
f0.70/r0.10 适合弱网，f0.50/r0.05 适合精度优先。
```

## 7. E4：成本项消融实验

### 7.1 目的

证明 `score = p_gain * expected_gain - cost` 中的 cost 项是有效的，不是公式装饰。

### 7.2 对比方法

至少比较：

```text
Learned probability only:
  score = p_gain

Learned gain only:
  score = p_gain * expected_gain

Learned v2 cost-aware:
  score = p_gain * expected_gain - λ * payload_kb / 100
```

如果代码实现时间有限，最低比较：

```text
no-cost: score = p_gain * expected_gain
cost-aware: score = p_gain * expected_gain - λ * payload_kb / 100
```

### 7.3 固定条件

```text
数据集: COCO val2017
端侧模型: YOLO26s
云端模型: YOLO26x
cloud_imgsz: 512
frame_threshold: 0.60
roi_threshold: 0.08
```

### 7.4 需要记录

| 指标 | 说明 |
| --- | --- |
| Precision/Recall/mAP | 精度变化 |
| upload ratio | 是否上传更多 |
| KB/image | 带宽变化 |
| GFLOPs/image | 计算变化 |
| E2E@10Mbps | 弱网延迟变化 |
| ROI/full upload count | 上传类型变化 |

### 7.5 输出表格

| 策略 | Precision | Recall | mAP50 | mAP50-95 | 上传比例 | KB/image | GFLOPs/image | E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### 7.6 预期结果

合理预期：

```text
no-cost 上传更多；
no-cost 可能 mAP 略高，也可能接近；
cost-aware 在很小精度损失下显著降低 KB/image 和延迟；
```

如果结果不是这样，也要如实分析，可能说明 `λ` 需要重新扫描。

### 7.7 成功标准

能支撑以下任一结论即可：

```text
cost-aware 用更低成本达到接近精度；
或
cost-aware 在相同成本下精度更高；
或
cost-aware 形成更可控的精度-成本工作点。
```

## 8. E5：普通 ROI+云复核 baseline

### 8.1 目的

证明本文方法不是普通 ROI 上传，而是更合理的收益-代价决策。

### 8.2 baseline 设计

普通 ROI baseline 使用固定规则：

```text
如果端侧框 conf < 0.45 -> ROI 上传
如果框面积 area_ratio < 0.015 -> ROI 上传
如果 edge_margin < 0.05 -> ROI 上传
每张图最多上传 max_roi_per_image = 1 或 2
```

变体：

```text
ROI-rule-1: low confidence only
ROI-rule-2: low confidence + small object
ROI-rule-3: low confidence + small object + edge object
```

### 8.3 对比方法

```text
Edge-only
Rule ROI baseline
Rule aggressive
Learned v2 f0.60/r0.08
Learned v2 f0.70/r0.10
```

### 8.4 输出指标

```text
mAP50-95
upload ratio
ROI upload count
full-frame upload count
KB/image
E2E@10Mbps
GFLOPs/image
```

### 8.5 输出表格

| 方法 | mAP50-95 | 上传比例 | ROI次数/image | 整图次数/image | KB/image | E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

### 8.6 预期结果

可能出现两种情况：

1. Learned v2 明显优于普通 ROI baseline：

```text
说明学习器能减少低价值 ROI 上传，获得更好的精度-成本折中。
```

2. 普通 ROI baseline 某些场景接近 Learned v2：

```text
说明规则在冷启动有价值，但学习器仍提供更灵活的阈值和代价建模。
```

### 8.7 成功标准

论文中能明确回答：

```text
为什么本文方法比“低置信 ROI 上传 + 云端复核”更合理？
```

## 9. E6：可视化案例实验

### 9.1 目的

让审稿人直观看到系统如何工作。

### 9.2 案例类型

至少准备 4 组：

| 案例 | 说明 |
| --- | --- |
| Case 1 | 端侧漏检，云端整图补检 |
| Case 2 | 端侧误检，云端复核后抑制 |
| Case 3 | ROI 裁剪后云端检测到局部目标 |
| Case 4 | 不上传成功，端侧结果可靠且节省带宽 |

### 9.3 每个案例需要输出

```text
original
edge result
upload decision overlay
cloud result
fused result
decision score text
```

### 9.4 输出文件

建议目录：

```text
paper/figures/cases/case1_*.png
paper/figures/cases/case2_*.png
paper/figures/cases/case3_*.png
paper/figures/cases/case4_*.png
```

### 9.5 成功标准

论文中能放一张 2x4 或 4x4 可视化图，说明：

```text
端侧、上传区域、云端和融合结果之间的关系。
```

## 10. E7：cloud_imgsz=512 vs 640

### 10.1 目的

回答为什么默认云端输入尺寸为 512，而不是直觉上更高精度的 640。

### 10.2 对比配置

至少比较：

```text
Learned v2 f0.50/r0.05 cloud512
Learned v2 f0.50/r0.05 cloud640
Learned v2 f0.60/r0.08 cloud512
Learned v2 f0.60/r0.08 cloud640
Learned v2 f0.70/r0.10 cloud512
Learned v2 f0.70/r0.10 cloud640
```

### 10.3 记录指标

```text
mAP50
mAP50-95
cloud_infer_ms
GFLOPs/image
E2E@10Mbps
```

### 10.4 预期分析

可能结果：

```text
cloud640 精度略高，但计算和延迟明显更高；
cloud512 在协同系统中提供更好的综合折中；
```

也可能：

```text
cloud640 mAP 没有明显提升，因为融合和阈值改变了 AP 排序。
```

都可以接受，关键是有同口径证据。

## 11. E8：frame-only / ROI-only / frame+ROI

### 11.1 目的

明确双粒度上传机制的贡献边界。

### 11.2 对比方法

```text
frame-only: 只允许整图上传
roi-only: 只允许 ROI 上传
frame+roi: 默认完整策略
```

### 11.3 输出指标

```text
mAP50-95
upload ratio
full upload count
ROI upload count
KB/image
E2E@10Mbps
```

### 11.4 结果解释原则

如果 frame-only 接近 frame+roi：

```text
说明 COCO full 当前收益主要来自帧级复核；
ROI 分支作为局部低带宽复核机制保留；
```

如果 ROI 有明显提升：

```text
强调双粒度上传有效；
```

不要夸大数据不支持的结论。

## 12. E9：JPEG quality 实验

### 12.1 目的

分析图像压缩对上传量、延迟和云端检测稳定性的影响。

### 12.2 配置

```text
quality = 60, 70, 80, 90
策略 = Learned v2 f0.60/r0.08
cloud_imgsz = 512
```

### 12.3 输出指标

```text
KB/image
cloud mAP 或协同 mAP
jpeg_encode_ms
upload_ms
E2E latency
```

### 12.4 预期结论

```text
quality 低可以降低上传量和延迟；
quality 过低可能影响云端检测；
quality=70 或 80 可能是较好折中。
```

## 13. E10：上传队列实验

### 13.1 目的

证明实时系统需要队列限制和过期任务丢弃，不是简单阻塞上传。

### 13.2 配置

```text
queue_size = 0, 1, 2, 4
带宽 = 5 Mbps 或 10 Mbps
策略 = Learned v2 f0.60/r0.08
```

### 13.3 指标

```text
平均延迟
P95 延迟
丢弃任务比例
过期返回比例
有效返回比例
```

### 13.4 预期结论

```text
无限或过大队列会提高 P95 延迟；
小队列能保持实时性；
queue_size=2 是较合理工程点。
```

## 14. 实验执行顺序

建议按以下顺序执行：

```text
Step 1: E0 整理已有 COCO 结果和图表
Step 2: E4 成本项消融
Step 3: E5 普通 ROI baseline
Step 4: E7 cloud512 vs cloud640
Step 5: E8 frame-only/ROI-only/frame+ROI
Step 6: E1/E2 T41 摄像头链路和延迟拆解
Step 7: E3 弱网实验
Step 8: E6 可视化案例
Step 9: E9 JPEG quality
Step 10: E10 上传队列
```

原因：

```text
先做离线 COCO 消融，能快速补论文核心表格；
再做 T41 链路，补真实系统证据；
最后做弱网、JPEG 和队列，强化 JRTIP 实时性。
```

## 15. 论文中对应位置

| 实验 | 论文位置 |
| --- | --- |
| E0 | 6.1 COCO full 主结果、6.3 工作点分析 |
| E1 | 6.5 T41 摄像头链路实验 |
| E2 | 6.5 T41 摄像头链路实验、Fig. latency breakdown |
| E3 | 6.6 弱网实时性实验 |
| E4 | 6.4 消融实验 |
| E5 | 6.4 消融实验 |
| E6 | 6.7 可视化案例，后续新增 |
| E7 | 6.4 消融实验 |
| E8 | 6.4 消融实验 |
| E9 | 6.5 或 6.6 |
| E10 | 6.6 弱网实时性实验 |

## 16. 当前最小投稿闭环

如果时间有限，至少完成：

```text
E0: COCO 主结果
E1: T41 链路跑通
E2: 延迟拆解
E3: 弱网实验
E4: 成本项消融
E5: 普通 ROI baseline
E6: 可视化案例
```

完成以上，论文就具备 JRTIP 初投的基本实验闭环。

如果 E1/E2 无法按时完成，则不建议投 JRTIP，应转向更偏离线算法/应用验证的期刊或中文方向。

