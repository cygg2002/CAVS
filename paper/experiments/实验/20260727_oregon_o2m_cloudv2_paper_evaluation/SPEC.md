# Oregon Cloud‑V2 论文级复评规范

## 1. 研究问题

在保持 Oregon 数据划分、Edge YOLO26n、Original/Q90/ROI 动作空间、评估器和预算协议不变时，只将 Cloud YOLO26x 从旧 epoch‑18 检查点替换为 O2M 继续微调后的 `best.pt`，判断：

1. 新模型在 Original、Q90、ROI 三种载荷上的收益是否一致；
2. 九个 Ranker × 三种动作策略的相对排序是否改变；
3. 校准式成本感知选择器能否仅凭 calibration/audit 冻结到 final；
4. Fixed-call、Online、Fixed-byte 三种预算语义下的精度—通信成本折中是否改善。

## 2. 实验性质

- 这是独立的、探索性论文级 Cloud‑V2 复评，不替换或覆盖 V1。
- Oregon 的原 registered final 已经在 V1 中访问过，因此 Cloud‑V2 final 只能标为
  `posthoc_detector_update_final_reuse`，不能声称是新的隐藏 final。
- 新模型检查点由 detector_val 选择；detector_val 同时也是旧协议的 audit。
  因此 audit 风险门禁结果属于 post-hoc 诊断证据，不是严格独立的安全保证。

## 3. 唯一允许改变的变量

| 项目 | V1 | Cloud‑V2 |
|---|---|---|
| Cloud 模型 | YOLO26x epoch‑18 | O2M 继续微调 best.pt |

以下项目全部冻结：

- Edge 模型及其预测/FPN 特征；
- same-camera event-disjoint 数据划分及图像 ID；
- `imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True, end2end=False`；
- Q90 编码文件、ROI 几何、ROI JPEG 和请求字节；
- Original/Q90/ROI 动作定义；
- Ranker、动作头、预算点、随机种子和评估器。

云预测、gain/quality、所有 gain-dependent Ranker、共享动作头、GBDT 动作头、
Online 阈值和 Fixed-byte λ 必须重新生成或训练。

## 4. 数据角色

| 角色 | 图像数 | 用途 |
|---|---:|---|
| fit | 9,923 | Ranker 与动作头训练 |
| calibration | 4,961 | 阈值、Online 和 Fixed-byte 校准 |
| audit | 9,923 | 风险门禁与 detector parity 诊断 |
| final | 4,962 | 冻结后的 post-hoc final 评估 |

开发阶段禁止读取或生成 `final` 的 Cloud‑V2 预测。

## 5. 评测矩阵

- Ranker：Random、Confidence、ClassGainPrior、M2E、
  Confidence-Residual M2E、Current 49D SemanticRank、MORIC-style、
  GEC-style、Current-Controlled-32D。
- 动作策略：SharedAction、CostEligible-AllROI、GBDT-Hierarchical。
- Fixed-call：5/10/15/25/40/50/65/80%。
- Online：10/25/40%。
- Fixed-byte：5/10/20/40% Original-byte budget。
- Random：50 seeds。
- 主 paired bootstrap：200 次；投稿敏感性补充为 500/1000/2000 次。

## 6. 资源与隔离约束

- 只写入本实验目录，不修改 V1、三数据集消融或正在运行的 bootstrap。
- GPU 推理串行；默认 batch=4、workers=2。
- 当 GPU 可用显存不足 6 GiB 时等待，不强行启动。
- CPU 大矩阵等既有 `run_bootstrap_workpoint.py` 任务结束后再启动，
  并使用 nice=10、线程上限 4。
- 所有阶段可恢复，有阶段 marker、日志、输入哈希和失败状态。

## 7. 验收标准

1. 新 Cloud SHA256 必须为
   `8944137e1a1324b064ed487ab780c1a410ac55f5be36d0c95192bdfdb52f1fdd`。
2. audit Original/Q90/ROI 预检必须复现已完成审计文件及其 9,923 个图像 ID。
3. 旧与新实验的 GT、Edge、FPN、Q90 字节、ROI xyxy/字节逐项一致。
4. dev 阶段没有 `final` Cloud‑V2 产物。
5. 冻结前记录模型、动作门禁、阈值、λ 与代码哈希；final 后哈希不变。
6. 产出 canonical、九 Ranker × TriAction、风险选择器以及 V1/V2 对比表。

