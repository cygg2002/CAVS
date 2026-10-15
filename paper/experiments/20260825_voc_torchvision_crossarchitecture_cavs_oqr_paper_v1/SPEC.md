# VOC Torchvision 异构检测器 CAVS-O/OQ/OQR 论文级实验规格

## Objective

在固定公开 COCO 权重、完全不训练检测器的条件下，检验 CAVS 是否能迁移到
SSDLite-MobileNetV3-Large → Faster R-CNN-ResNet50-FPN-V2 的异构检测器组合。
主方法为 `CAVS-OQR`，并独立重训 `CAVS-OQ` 与 `CAVS-O` 作为动作空间消融。

VOC2007 test 是公开标准 benchmark，且此前实验已经查看过，不能称 hidden final；
但是 reward、路由器训练、SharedAction、阈值和模型选择严禁读取 test。

## Frozen actions

- `edge`：SSDLite 官方 COCO 权重，零上传字节。
- `original`：原始 JPEG，Faster R-CNN 官方 transform。
- `q90`：Pillow `quality=90,optimize=True,subsampling=2` 后由同一云模型推理。
- `current@official`：从 SSDLite 框构建 Current merged ROI，以 Q90 编码；
  Faster R-CNN 保持官方 transform，映射回原图后以 MAUR 与 Edge 融合。

ROI 使用 `edge_conf=0.25`、最大面积比例 `0.65`。MAUR 删除中心位于 crop 内的
Edge 框，加入映射后的 Cloud 框，再执行 class-aware NMS。ROI 字节不小于原始
JPEG 时不作为可选动作。动作名不得写成 `current@640`，因为本实验不改变
Faster R-CNN 官方输入协议。

## Data and model protocol

- fit：15,629；calibration：922；VOC2007 test：4,952。
- 五个确定性 OOF folds；三个学习器种子。
- CAVS-O、CAVS-OQ、CAVS-OQR 必须分别按各自 reward 从头训练。
- ORIC/MORIC 使用相同三个种子；Confidence 固定；Random 使用 50 seeds。
- fit/calibration 的 Edge/Original/Q90 候选只复用已哈希的新 Torchvision Gate
  输出，不复用任何旧模型预测或学习器。

## Evaluation

- Primary：Fixed-byte 10% / 25% / 40%。
- Secondary：Fixed-call 10% / 25% / 40%。
- Primary metric：VOC AP50-95；同时报告 AP50、AP75、VOC07 AP50、VOC12 AP50。
- 公平动作比较：CAVS-O 对 Original-only ORIC/MORIC/Confidence/Random；动作消融为
  CAVS-OQ 对 CAVS-O、CAVS-OQR 对 CAVS-OQ；主系统比较还包括 CAVS-OQR 对
  各基线最强的 SharedAction-OQ 注册版本。这样同时回答同动作学习器优势和完整
  OQR 系统优势，不用不对称动作空间掩盖结论。
- Fixed-byte test 上运行 1,000 次 image-level paired bootstrap；Random 在每个
  replicate 内分别评价 50 套 selection 后求 AP 均值，禁止平均 selection。
- 报告三个单模型均值±sample SD、三模型 ensemble、动作计数、真实 payload、
  容量利用率和所有哈希。

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --stage data
python scripts/run_pipeline.py --stage actions
python scripts/run_pipeline.py --stage endpoint
python scripts/run_pipeline.py --stage common
python scripts/run_pipeline.py --stage rewards
python scripts/run_pipeline.py --stage train
python scripts/run_pipeline.py --stage point
python scripts/run_pipeline.py --stage bootstrap
python scripts/run_pipeline.py --stage report
```

## Project structure

```text
configs/protocol.json       全部冻结合同、哈希和方法矩阵
scripts/cache_actions.py    导入已冻结候选并生成 final/ROI
scripts/build_common.py     构建 O/OQ/OQR learner records
scripts/build_rewards.py    fold-contained contextual AP rewards
scripts/train_learners.py   三种子、五折和 full-fit 训练
scripts/evaluate_point.py   calibration/test 统一容量调度与点结果
scripts/bootstrap.py        paired bootstrap 与 nested Random
tests/                      纯逻辑与小型集成回归测试
runs/                       断点、哈希和运行证据
```

## Testing strategy

- 每项新逻辑先写失败测试；重点覆盖 ROI 几何、crop 坐标映射、MAUR、OQR action
  record、独立 reward map、训练任务矩阵和 test 禁止进入 reward/training。
- 本地及服务器测试全部通过后执行 10 张 ROI smoke，再启动完整缓存。
- 每一阶段只有 completion 与输入/输出哈希一致才允许续跑。

## Boundaries

- Always：使用官方权重与 transform；重训所有学习器；保留 difficult-ignore 语义；
  使用同一 evaluator 和容量求解器。
- Ask first：改变 ROI profile、融合规则、动作空间、预算、种子或训练超参数。
- Never：训练检测器；复用旧 learner checkpoint；用 test 生成 reward 或选择配置；
  将本实验并发运行的推理时间写成正式系统延迟。

## Success criteria

1. 数据、权重、Gate 源候选和协议哈希全部通过。
2. 四动作缓存人口完整；ROI 坐标、字节、映射和融合无非法输出。
3. O/OQ/OQR reward 分别覆盖 fit/calibration，test reward 不存在。
4. 三个 CAVS 方法完成 5-fold×3 seeds 和 full-fit×3 seeds；ORIC/MORIC 全部重训。
5. 两种预算模式、三档预算、完整方法矩阵无缺行、重复或超容量。
6. 1,000 次 paired bootstrap 和 Random-50 统计口径完整。

## Open questions

无。上述 `current@official+MAUR` 是本实验唯一注册 ROI 语义。
