# COCO RK3588–RTX 因果在线 HIL 最终实验 v2 规格

## 1. 目标与论文定位

本实验冻结并执行论文最终的 COCO 因果在线端云协同评测。每张图像到达后，
Orange Pi 5 Ultra（RK3588）必须完成 RKNN-FP16 端侧检测、当前帧动作决策、
必要时立即上传、接收 RTX 云端检测结果并提交当前帧，之后才允许读取下一张图像。

本实验与离线全局 Fixed-byte 调度严格分开：离线结果是算法上限；本实验是
`zero_lookahead=true`、`global_plan=false`、`buffer_frames=0` 的真实硬件在环结果。

## 2. 冻结对象

- 数据集：COCO val2017，共 5,000 张，顺序清单、标注文件与图像哈希冻结。
- 端模型：`yolo26n-rk3588-fp16.rknn`，SHA-256
  `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`。
- 云模型：`yolo26x.pt`，SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`。
- CAVS 学习器：只使用 RKNN-FP16 域的 fit=9,678、calibration=571 训练产物；
  final=5,000 不参与训练、校准、阈值或控制器选择。
- 预算：JPEG payload 相对原始 JPEG 总字节的 10% / 25% / 40%。同时独立报告
  HTTP body bytes；两种口径不得混称。
- 动作：Edge、Original、Q90，以及 CAVS-OQR 的 Current@640 ROI + MAUR。
- 云端与端侧检测阈值、NMS、图像预处理和融合语义写入运行合同并做哈希校验。

## 3. v2 相对旧 v1 的必要修正

1. 请求体由 JSON + Base64 改为版本化二进制帧，避免约 33% Base64 膨胀。
2. 服务端在返回前裁剪边界框并删除非有限、越界后零面积或类别非法的框；
   客户端继续严格校验，不以放宽校验掩盖错误。
3. 端侧 Edge 原始框仍用于冻结策略的特征与 ROI 构造，但提交给 COCO evaluator
   的最终框统一执行相同的确定性清洗。
4. 每帧记录 JPEG payload bytes、HTTP body bytes、动作、控制器状态、端/云/网络/
   端到端耗时和失败回退原因。
5. 新 run-id、新协议版本、新 bundle 哈希；禁止覆盖或续写 v1 run。

## 4. 二进制请求协议

媒体类型为 `application/vnd.cavs-hil.v2+octet-stream`。请求体按以下顺序编码：

1. 8 字节 magic：`CAVSHIL2`；
2. 4 字节无符号大端 metadata JSON 长度；
3. UTF-8 紧凑 JSON metadata；
4. 原始 JPEG payload。

metadata 必须包含协议版本、动作、图像标识、payload 长度与必要的 ROI 几何。
服务端必须核验 Content-Type、总长度、metadata 上限、payload 长度、动作字段和 ROI
字段，任何不一致均安全失败。Bearer token 只从环境变量读取，不写入日志或归档。

## 5. 正式实验矩阵

### 5.1 COCO 主矩阵（32 个独立 run）

| 方法 | 预算/种子 | run 数 |
|---|---|---:|
| RKNN Edge-only | 无上传 | 1 |
| Always-Original | 100% | 1 |
| CAVS-OQ（calibration 预选逐帧 dual step） | 10% / 25% / 40% | 3 |
| CAVS-OQR（calibration 预选逐帧 dual step） | 10% / 25% / 40% | 3 |
| ORIC（RKNN 域，3-seed ensemble） | 3 budgets | 3 |
| MORIC（RKNN 域，3-seed ensemble） | 3 budgets | 3 |
| Confidence | 10% / 25% / 40% | 3 |
| Random（预注册 5 seeds） | 3 budgets × 5 seeds | 15 |

上表合计 32 个硬件 run。ORIC/MORIC 必须各自独立训练三个 fit-only seed，论文主方法
使用三种子分数平均的预冻结 ensemble；三个单模型只做 calibration/final 的离线因果
重放以报告均值与样本 SD，不各自重复硬件延迟 run。该口径在 `RUN_MATRIX.json` 中冻结，
不得在看过 final AP 后改变。

Random 的五个种子在任何新 final 结果产生前固定为：`20260720`、`20260721`、
`20260722`、`20260723`、`20260724`。

### 5.2 在线控制器消融（6 个独立 run）

CAVS-OQ 与 CAVS-OQR 各自额外运行三档静态 calibration λ。主矩阵控制器在
`{0, 0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02}`
中仅用 calibration 选择逐帧 projected primal-dual step；其中 `step=0` 严格退化为
静态 λ。该消融用于量化无缓冲、无未来信息条件下在线 dual update 能减少多少调度
损失。两种控制器均为 `buffer_frames=0`，不会等待若干帧后再上传。

### 5.3 网络敏感性（8 个独立 run）

固定 25% 预算，CAVS-OQ、CAVS-OQR、ORIC、MORIC 各在两种额外受控网络条件下
运行。网络条件、实现工具、RTT/带宽/丢包实测值必须预先写入运行矩阵。

## 6. 运行隔离与顺序

- 正式延迟 run 串行执行，不允许多个方法共享 NPU、GPU 或链路并发。
- RK3588 关闭 XFCE/桌面负载，固定 NPU core、CPU governor 和温度记录。
- RTX 云端分配隔离 GPU；CPU bootstrap 可以继续，但不得争用云推理所需 CPU 核。
- 方法/预算/种子顺序由固定随机种子生成并写入 `RUN_MATRIX.json`，避免时间漂移
  与温度漂移总是偏向同一方法。
- 每次正式 run 使用全新目录；中断 run 保留为失败证据，重跑必须新建 run-id。

## 7. 验收门槛

50 张冒烟和每个 5,000 张正式 run 均检查：

1. 顺序位置唯一且完整，正式 run 为 5,000/5,000；
2. `cloud_failures=0`；
3. 最终预测中无非有限、负面积或零面积框；
4. prefix budget violation 为 0，最终 payload 容量不超预算；
5. HTTP body / JPEG payload 比值不高于 1.02（小 payload 的固定头开销单独解释）；
6. 运行合同明确 `zero_lookahead=true`、`global_plan=false`、`buffer_frames=0`；
7. 模型、策略、控制器、序列、代码 bundle 和服务协议哈希均与冻结清单一致；
8. 无 kernel Oops、OOM、热降频失控或后台任务污染；
9. AP50-95、AP50、AP75，动作数、字节、成功率、吞吐与 P50/P95/P99 延迟可重算。

任何硬门槛失败的 run 不进入论文主表，必须保留日志并用新 run-id 重跑。

## 8. 统计与报告

- COCO AP 使用官方 pycocotools 统一重算。
- CAVS 对每个基线、CAVS-OQR 对 CAVS-O 做三预算 paired image bootstrap。
- Random 主表报告 5 个预注册控制种子的均值和 SD；bootstrap 在每个图片重采样中
  先分别计算 5 套 Random，再对 AP 取均值，避免单 Random seed 与主表口径不一致。
- 离线全局调度与在线 HIL 分表报告，差值只在相同模型域、动作空间和预算口径下解释。

## 9. 范围边界

- 本目录只实现并执行 COCO final-v2；不修改 20260817/18 的 v1 历史目录。
- Oregon 使用独立规格与独立 bundle，在 COCO v2 协议通过后执行。
- VOC 保持服务器端跨架构泛化实验，不在本轮强行移植到 RKNN。
- 不重新训练检测器；只有在冻结资产缺失或训练域不匹配时才训练轻量策略学习器。
