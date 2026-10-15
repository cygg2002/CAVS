# VOC 跨检测器架构 CAVS 论文级实验规格（冻结前注册）

## 1. 目的与可声明边界

本实验检验 CAVS 是否能从同构 YOLO 组合迁移到公开的异构检测器组合：

- Edge：GluonCV `center_net_resnet18_v1b_voc`；
- Cloud：GluonCV `faster_rcnn_resnet50_v1b_voc`；
- 不训练、微调或挑选任何检测器权重；
- 只从头训练 CAVS、ORIC 与 MORIC 的轻量决策学习器。

本实验是“固定外部预训练模型的跨架构稳健性基准”，不是新的隐藏终评。两个公开权重的发布者均在 VOC2007 test 上报告过结果，因此正文必须披露 detector-level test exposure；但本实验的路由学习器、校准和选择均不得访问 VOC2007 test。

## 2. 冻结数据协议

- 学习器开发总体：VOC2007 trainval + VOC2012 trainval，共 16,551 张；
- fit：15,629 张；calibration：922 张；按 `split_seed=20260824` 对 `year:image_id` 做 SHA-256 排序后固定划分；
- final：VOC2007 test，共 4,952 张，只用于端点评测和最终结果；
- fit 内做 5-fold OOF，fold 由 `fold_seed=20260824` 冻结；
- 三个学习器种子：20260824、20260825、20260826；
- final 不参与 reward、epoch、阈值、动作规则或模型选择。

VOC `difficult=1` 标注不得简单删除。主指标使用 0.50:0.05:0.95 IoU、101 点插值、按类宏平均且正确忽略 difficult 的 VOC-AP50–95；同时报告 AP50、AP75、VOC2007 11-point AP50 和 VOC2012 integral AP50。公开 GluonCV 数值只与 VOC2007 AP50 对照，不能与 AP50–95 混用。

## 3. 冻结检测协议

### 3.1 权重

- CenterNet 官方模型 SHA-1：`38c509d456a6e14e4c54e961ed43bffe8cf62840`；
- Faster R-CNN 官方模型 SHA-1：`447328d89d70ae1e2ca49226b8d834e5a5456df3`；
- 下载后额外记录 ZIP 与解压参数文件 SHA-256；任何哈希漂移均中止实验。

### 3.2 推理

- CenterNet：官方发布精度所用的 `CenterNetDefaultValTransform(512,512)` 固定方形仿射预处理、无 flip/TTA；预测框通过官方 `get_post_transform` 逆变换回原图；
- Faster R-CNN：官方 preset，short=600、max_size=1000、无 flip/TTA；
- 使用模型自带后处理；保留 score >= 0.0001 的有限框，坐标映射回原图并裁剪；
- 类别严格按 20 个 VOC 类顺序映射；
- 非有限、越界后退化或未知类别框必须计数并剔除；
- Original 直接读取原始 JPEG；Q90 使用 Pillow `quality=90, optimize=True, subsampling=2`，实际文件字节作为成本；
- Edge、Original、Q90 对同一图片均使用逐图确定性推理，禁止混用旧缓存。

在生成任何学习器结果之前，必须完成 4,952 张 final 的 Edge/Cloud 端点门禁；门禁只检查实现完整性、类别/几何和与官方 VOC07 AP50 的合理一致性，不据此修改超参数。

实现勘误（2026-08-24）：初版文字误将 CenterNet 单图演示接口的 `short=512,max_size=1024` 写为发布精度协议。端点门禁在任何 fit/calibration reward 或学习器训练开始前检测到该错误（错误路径 VOC07 AP50=0.384959），因此按 GluonCV 官方 512×512 验证脚本更正；错误缓存隔离保留，不进入任何论文结果。

## 4. 动作空间与预算

- O：Edge / Original；
- OQ：Edge / Original / Q90；
- Fixed-byte 容量：`floor(budget × final 原始 JPEG 总字节)`；
- Fixed-call 容量：`floor(N × budget + 0.5)` 个云调用；
- 预算：10%、25%、40%；
- 同一图片至多选择一个云动作；Edge 成本为 0；
- 主比较 CAVS-O 与所有外部方法均使用 O，保证同动作公平；
- CAVS-OQ 必须独立按 OQ reward 重训，不能由 OQ/OQR 模型屏蔽动作冒充；
- CAVS-OQ − CAVS-O 是多质量动作梯级的主要消融。

## 5. 学习器与方法矩阵

训练方法：

1. `CAVS-O-Core`：O 动作空间，Action-conditioned contextual AP50–95 reward；
2. `CAVS-OQ-Core`：OQ 动作空间，独立重训；
3. `ORIC-AP5095`、`MORIC-AP5095`：O 主公平基线；
4. `ORIC-AP50`、`MORIC-AP50`：目标度量敏感性分析。

固定/规则基线：

- Edge-only、Cloud-only；
- DCSB-Score-Adapted；
- Confidence；
- Random-Mean50。

对需要 OQ 结果的标量基线，仅允许使用 fit 训练、calibration 冻结的 SharedAction-Q90 规则；名称必须带 `SharedAction-OQ`，不得称为原始 ORIC/MORIC。

每个训练方法报告三个单模型结果、均值 ± sample SD、三模型平均分 ensemble 结果及 ensemble 相对单模型均值的变化。ORIC/MORIC 与 CAVS 使用相同三个种子。

## 6. Reward 与训练隔离

- reward 只为 fit 和 calibration 生成，final reward 明确禁止；
- fit reward 的上下文只能来自目标所属 OOF fold，context_size=32；
- contextual reward 为替换目标图片动作前后 AP50–95 差值乘上下文总图数；
- O 与 OQ 分别构建 reward 矩阵并独立训练；
- 每个 CAVS 方法执行 5-fold OOF × 3 seeds，加 full-fit × 3 seeds；
- full-fit 的 epoch/early-stop 只使用 fit 内部确定性验证子集；
- calibration 只用于冻结 SharedAction/报告，不用于反复挑选 final 结果。

## 7. 统计协议

- 点结果：两种预算模式 × 三档预算 × 完整方法矩阵；
- 主要推断：Fixed-byte、final、1,000 次 image-level paired bootstrap；
- 主比较：
  - CAVS-O vs ORIC-AP5095、MORIC-AP5095、DCSB、Confidence、Random-Mean50；
  - CAVS-OQ vs CAVS-O；
  - OQ SharedAction 结果仅作为补充比较；
- Random 必须在每个 bootstrap 重采样中分别计算 50 套 selection 的 AP，再对 50 个 AP 求均值；禁止平均 selection；
- 报告 delta、95% percentile CI、双侧符号概率与 replicate 数；
- point、selection、bootstrap 输入和输出均记录 SHA-256。

## 8. 失败门禁与停止规则

任一条件不满足即停止后续阶段：

- 数据角色数量、身份集合或 archive SHA-256 不符；
- 权重 SHA-1/SHA-256 不符；
- 任一 action cache 不完整、重复、缺图或含非法框；
- final 被 reward/training/calibration 访问；
- O/OQ 模型复用旧 checkpoint；
- Fixed-byte 超容量或 Fixed-call 数量错误；
- Random 少于 50 seeds 或 bootstrap 少于 1,000 replicates；
- 结果矩阵存在缺行、重复键或协议语义别名。

## 9. 资源与执行

- 新实验目录与旧 COCO/Oregon 任务完全隔离；
- GluonCV/MXNet 使用独立环境，不修改 `yolo26` 环境；
- GPU 推理仅在空闲显存达到门槛时启动；CPU reward/bootstrap 使用 `nice=19`；
- 所有阶段支持 completion + hash 断点，但禁止跨协议复用缓存；
- 正式运行前必须通过本地单元测试和服务器 1/10 张 smoke test。
