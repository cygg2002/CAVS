# JRTIP V0.6 流程图绘制规范

本文档用于在 Visio 或 draw.io 中重绘正文流程图。所有图均采用白底、黑灰线条和少量低饱和强调色，确保灰度打印仍可区分。图内不写“图 1”“Figure 1”等编号，也不放图题；图号和图题只放在论文正文中。

## 统一视觉规范

- 画布：横向，正文跨双栏图宽 174 mm；单栏图宽 84 mm。
- 字体：Arial，最终打印尺寸 8.5-9 pt，模块标题 9-10 pt 加粗。
- 线宽：主数据流 1.2 pt；控制流 1.0 pt；离线反馈虚线 1.0 pt。
- 箭头：实心三角箭头；双向通信使用两条方向相反的箭头，不使用模糊的双箭头。
- 圆角：最多 2 mm，小圆角矩形；禁止大面积阴影、渐变和立体效果。
- 颜色：
  - 端侧模块：浅灰蓝 `#DCE6F1`；
  - 传输/队列：浅青灰 `#DDEBF7`；
  - 云端模块：浅橙 `#FCE4D6`；
  - 融合/输出：浅绿 `#E2F0D9`；
  - 离线训练：浅灰 `#E7E6E6`；
  - 关键代价或 deadline：浅红边框 `#C00000`，不填充红色。
- 图例：实线 = 在线数据流；虚线 = 离线训练或参数更新；点划线 = 异步迟到结果。

## 图 1：端云协同在线闭环架构

### 目的

说明一帧视频从采集到本地结果、选择性上传、云端复核、异步融合及过期结果丢弃的完整在线流程。该图只画推理架构，不画训练数据生成和模型重训。

### 版式

采用从左到右的五列布局：

1. 输入与端侧检测；
2. CAFRU 决策；
3. 有界传输队列；
4. 云端复核；
5. 结果融合与输出。

在图最下方增加一条“运行时状态反馈”细线，但不要把它画成学习器在线训练。

### 节点

第一列“低功耗摄像头/端侧”：

- `视频帧采集`
- `解码 / NV12-RGB 转换`
- `YOLO26s 端侧检测`
- 输出小标签：`B_e = {class, confidence, box}`

第二列“CAFRU 上传决策”：

- `帧级特征提取 (15维)`
- `ROI级特征提取 (每框15维)`
- `概率头 + 收益头`
- `收益-代价评分`
- 菱形判断：`整帧 / ROI / 不上传`

第三列“传输与实时调度”：

- `JPEG 编码`
- `有界队列 Q`
- `keep-latest / drop-new`
- `deadline 检查`
- `HTTP multipart 请求`
- 从队列旁边画一个状态小框：`payload、queue length、result age`

第四列“云端复核”：

- `请求解析与解码`
- `YOLO26x 复核`
- `ROI 坐标回映射`
- `JSON 检测结果`

第五列“融合与输出”：

- `类别一致性与 IoU 匹配`
- `端侧框保留 / 云端框替换或补充`
- `类别感知 NMS`
- `带框视频 / 事件结果`
- 菱形判断：`result age <= deadline?`

### 箭头逐条说明

1. `视频帧采集 -> 解码`：原始摄像头帧进入端侧预处理。
2. `解码 -> YOLO26s`：同一解码帧进入端侧检测器。
3. `解码 -> 帧级特征提取`：图像复杂度需要像素信息，因此帧图像也进入特征模块。
4. `YOLO26s -> 帧级特征提取`：检测数量、覆盖率和置信度统计来自端侧框。
5. `YOLO26s -> ROI级特征提取`：每个端侧候选框生成局部几何与置信度特征。
6. `两类特征 -> 概率头+收益头 -> 收益-代价评分`。
7. `评分 -> 整帧/ROI/不上传`。
8. `不上传 -> 结果融合`：当前帧直接复用端侧结果，不发生网络请求。
9. `整帧或ROI -> JPEG编码 -> 有界队列`。
10. `有界队列 -> HTTP请求 -> 云端解析 -> YOLO26x`。
11. `YOLO26x -> ROI坐标回映射`：仅 ROI 请求经过该节点；整帧请求绕过。
12. `云端检测结果 -> JSON -> 结果年龄判断`。
13. `未过期 -> 结果融合`；`已过期 -> 丢弃`，并以点划线表示迟到结果。
14. `结果融合 -> 带框视频/事件结果`。
15. `队列长度、result age -> 调度器`：仅用于运行时准入和丢弃，不连接到“训练”节点。

### 可直接用于 draw.io AI 的提示词

```text
Create a publication-quality horizontal system diagram for a real-time
edge-cloud object detection system. Use five columns: low-power camera edge,
CAFRU selective uploader, bounded communication queue, cloud reviewer, and
result fusion/output. Show video capture, NV12/RGB conversion, YOLO26s edge
detection, 15-D frame features and 15-D per-box ROI features, four lightweight
probability/value heads, utility-minus-cost scoring, a three-way decision
(full frame, ROI, skip), JPEG encoding, bounded queue with keep-latest/drop-new
and deadline checks, HTTP multipart upload, YOLO26x cloud review, ROI coordinate
remapping, JSON return, result-age validation, class-aware fusion/NMS, and
annotated video output. Use solid arrows for online data, dash-dot arrows for
late asynchronous results, and no offline training loop. White background,
Arial, thin black lines, low-saturation blue for edge, orange for cloud, green
for fusion, no gradients, no figure number or title inside the image.
```

## 图 2：CAFRU 离线训练与在线决策

### 目的

清楚区分“离线训练”和“在线推理”。左半部分生成收益监督并训练四个轻量头；右半部分只加载参数并执行推理，绝不在摄像头运行过程中继续训练。

### 版式

上下两条泳道：

- 上泳道：`离线训练阶段`
- 下泳道：`在线部署阶段`

左上到右上为训练数据流；训练结束后用一条向下虚线箭头连接到在线阶段的“参数文件/编译常量”。

### 离线训练节点

1. `COCO train2017 + 人工标注`
2. 并行节点：`YOLO26s 预测 B_e`、`YOLO26x 整帧预测 B_c`
3. `端侧效用 U(B_e,Y)`
4. `云端/融合效用 U(B_c,Y) 或 U(B_f,Y)`
5. `收益标签 g = U_after - U_before`
6. `正收益标签 z = 1[g > 0]`
7. 两条分支：
   - `15维帧特征 x_f -> 帧概率头 p_f -> 帧收益头 g_f`
   - `候选ROI + 云端ROI复核 -> 15维ROI特征 x_r -> ROI概率头 p_r -> ROI收益头 g_r`
8. `90%训练 / 10%验证，seed=42`
9. `逻辑回归 + 类别平衡 + L2`
10. `Huber线性回归 + L2`
11. `导出 JSON / C++ 常量`

### 在线决策节点

1. `当前帧 I_t + 端侧框 B_e`
2. `计算 x_f 和 {x_r,i}`
3. `四个线性头推理`
4. `S_f = p_f max(0,g_f) - lambda_b C_b`
5. `S_r,i = p_r,i max(0,g_r,i) - lambda_b C_r,i`
6. 菱形判断：
   - 若帧分数超过阈值且队列可用：`上传整帧`
   - 否则选择满足阈值的最高分 ROI：`上传 ROI`
   - 否则：`不上云`
7. `运行时调度器`接收 `queue length / deadline / result age`，可否决上传或丢弃迟到结果。

### 必须在图中标出的边界

- 概率和收益头的输入是图像/检测统计特征。
- payload 与 delay 是显式代价或调度变量，不应画成已经训练进模型的网络状态特征。
- `参数导出`到`在线四头推理`使用虚线箭头，标签写“离线部署”。
- 在线路径不得返回训练集或优化器。

### 可直接用于 draw.io AI 的提示词

```text
Draw a two-swimlane scientific workflow named only by lane labels, without a
figure title. Top lane: offline CAFRU training. Start with COCO train2017 and
manual annotations; run YOLO26s edge predictions and YOLO26x full-frame/ROI
review; compute object-level utility U = TP - alpha FP, normalized gain
g = U_after - U_before, and binary positive-gain label z. Extract 15 frame
features and 15 ROI features. Train four lightweight heads: frame gain
probability (balanced logistic regression), frame expected gain (Huber linear
regression), ROI gain probability, and ROI expected gain, using a deterministic
90/10 train-validation split. Export JSON/C++ parameters. Bottom lane: online
deployment. Compute features from the current frame and YOLO26s boxes, evaluate
the four frozen heads, calculate frame and ROI utility-minus-payload-cost scores,
then choose full-frame upload, best ROI upload, or skip. A separate bounded
queue/deadline scheduler may reject uploads or discard stale returns. Connect
offline export to online frozen heads with one dashed arrow labeled offline
deployment. Never connect online inference back to training. White background,
Arial, vector style, no gradients, no caption inside.
```

## 图 5：T41 有界队列与 deadline 处理

### 目的

解释为什么“云端返回成功”不等于“实时有效”。该图突出主采集/检测线程与上传工作线程解耦，以及结果年龄、丢弃率和 effective FPS 的计算位置。

### 版式与节点

画两个并行线程泳道和一个云端泳道：

- `T41 主线程`
- `T41 上传工作线程`
- `云端服务`

主线程节点：

1. `采集帧 t`
2. `YOLO26s`
3. `CAFRU / 规则准入`
4. `队列是否有容量?`
5. `写入候选任务或本地跳过`
6. `立即输出端侧结果`
7. `进入下一帧`

上传线程节点：

1. `取队首 / 保留最新`
2. `JPEG Q70 编码`
3. `记录 submit timestamp`
4. `HTTP 上传`
5. `接收 cloud boxes`
6. `计算 result_age = return_time - frame_time`
7. 菱形判断 `result_age <= deadline`
8. `及时结果进入融合缓存`或`过期结果丢弃`

云端节点：

1. `解码`
2. `YOLO26x`
3. `返回框与 server timing`

图右侧加一个小指标框：

```text
main-loop FPS = processed frames / wall time
deadline-hit ratio = timely results / returned results
effective FPS@D = deadline hits / wall time
drop ratio = discarded candidates / upload candidates
```

### 可直接用于 draw.io AI 的提示词

```text
Create a three-swimlane real-time scheduling diagram for a T41 edge-cloud
camera prototype. Lanes: T41 main capture/inference thread, T41 upload worker,
and cloud service. The main thread captures frame t, runs YOLO26s, applies
selective admission, inserts only admissible tasks into a bounded queue, emits
the edge result immediately, and continues to the next frame. The worker uses
keep-latest/drop-new, JPEG Q70 encoding, records timestamps, sends HTTP
multipart requests, receives YOLO26x boxes, computes result age, and accepts
only returns within deadline D; stale returns are dropped. The cloud decodes,
runs YOLO26x, and returns boxes plus timing fields. Add a metric box defining
main-loop FPS, deadline-hit ratio, effective FPS at D, and drop ratio. Use
solid arrows for active tasks and dash-dot red arrows for stale returns. White
background, compact vector style, Arial, no gradients or caption inside.
```

## 导出检查

1. Visio/draw.io 源文件保留可编辑版本。
2. 投稿导出优先 PDF 或 EPS；若用 PNG，组合图至少 600 dpi。
3. 所有字体嵌入；缩到 174 mm 后文字仍不小于约 8 pt。
4. 图内不包含图号、完整图题、实验内部编号或路径。
5. 灰度打印检查：不同模块仍能靠边框、泳道和线型区分。

