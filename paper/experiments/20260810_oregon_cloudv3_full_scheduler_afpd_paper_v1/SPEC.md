# Oregon Cloud‑V3 全学习器与多动作评测规格

## 1. 目标

在冻结的 Oregon 45 类、same-camera/event-disjoint 协议上，将云端检测器唯一替换为 Cloud‑V3 epoch‑28 `best.pt`，重新生成所有受云模型影响的动作缓存、监督信号、学习器和阈值，并一次性完成与当前 COCO 方法栈等价的完整评测。

本实验回答四个问题：

1. Cloud‑V3 相比 Cloud‑V2 是否提高 Original、Q90 与 Profile×Scale ROI 的检测上限。
2. 从头训练后，哪一种图片排序器或图片—动作联合调度器在 Oregon 最强。
3. AFPD/Occupancy 与 12 个 Adaptive ROI 动作能否把 Teacher ceiling 转化为可部署收益。
4. 结论在 fixed-byte、fixed-call、事件块 bootstrap、类别/相机/事件分组下是否稳定。

## 2. 冻结边界

- 数据：canonical Oregon event-disjoint，`fit=9,923`、`calibration=4,961`、`final=4,962`。
- Edge：沿用 canonical Edge 模型及其预测、FPN 特征和 split manifests；它们不受云模型替换影响。
- Cloud：只允许 epoch‑28 Cloud‑V3 `best.pt`，SHA256 为 `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`。
- Cloud‑V2/旧 Cloud‑V1 的 Original、Q90、ROI predictions、reward、checkpoint、threshold、lambda 和 learner 一律禁止作为输入。
- 所有需要学习的 Ranker、action head、safe gate、AFPD、Occupancy 与校准器均从头训练。
- Oregon final 已在历史工作中使用，因此本实验是 `posthoc_exploratory_final_reuse`，不是新的隐藏 confirmatory final；final 不得参与训练、超参数、阈值、lambda、方法选择或 gate。
- 正在运行的 Cloud‑V4 训练不是本实验输入，不得中断，也不得在完成后替换 Cloud‑V3。

## 3. 动作空间

基础动作：

- `edge`
- `original@640`
- `q90@640`

Adaptive ROI 使用真实 crop geometry，共 12 个动作：

- Profile：`current`、`moderate`、`historical_cost`、`fixed_1_8`
- Cloud imgsz：`160`、`320`、`640`

每个 ROI 动作保存真实 `xyxy`、crop 尺寸、面积率、JPEG bytes、profile 和 imgsz。不得使用 COCO 旧 ActionValue 中由 area ratio 伪造左上角正方形 ROI 的兼容表示。

融合同时评测：

- `MAUR`：历史破坏式替换，仅作消融。
- `Reversible`：source-aware 可逆融合，作为 ROI 主结果。

## 4. 方法矩阵

### 4.1 系统与随机/置信度基线

- Edge-only
- Cloud-only Original
- Cloud-only Q90
- Random（50 个点估计种子）
- Confidence

### 4.2 历史内部 9 Ranker

- Random
- Confidence
- ClassGainPrior
- M2E
- Confidence-Residual M2E
- SemanticRank
- MORIC-style
- GEC-style
- Current-Controlled-32D

### 4.3 外部基线

- ORIC-AP50
- MORIC-AP50
- ORIC-AP5095
- MORIC-AP5095

外部方法使用相同 split、Cloud‑V3 reward、容量、动作缓存和 evaluator 重新训练/评测，不复用旧模型。

### 4.4 当前内部学习器与诊断模型

- Current-BudgetSet
- SetValue
- DualHead（部署只使用 value head，cost head 仅作辅助监督）
- Raw-ActionValue-OQ
- PerAction-Isotonic-OQ
- Value+Risk-LCB
- GlobalRank
- RelSetValue
- RelActionRank
- FactorizedGlobalActionRank

### 4.5 Frontier 与 Adaptive ROI

- CurrentActionValue-OQ
- CurrentActionValue-OQR12
- AFPD-OQ
- AFPD-OQR12
- OccupancyFree-AFPD-OQR12
- OccupancyAware-AFPD-OQR12

AFPD 的 Value 网络不接收 payload；真实 bytes 只进入 exact-capacity scheduler。Occupancy v2 只作冻结消融，不允许根据 final 改成主方法。

### 4.6 动作策略消融

所有适用的 scalar Ranker 评测：

- Original-only
- SharedAction-NoROI
- SharedAction-R12
- CostEligible-AllROI-R12
- GBDT-Hierarchical-R12

SharedAction、CostEligible 和 GBDT 必须用 Cloud‑V3 fit/calibration 从头训练与冻结。Action-conditioned 方法使用 `Joint-OQ` 或 `Joint-OQR12`。

## 5. 训练与校准

- `fit`：5-fold OOF，fold 按 event group 分配；同一事件不得跨训练/held-out fold。
- 每个神经学习器：3 个冻结种子，单 worker `torch.set_num_threads(1)`。
- contextual AP50-95 reward：fold-contained，fit OOF context 不得跨 fold；calibration context 只在 calibration role 内生成。
- full-fit checkpoint：仅由全量 fit 训练，必须保存并绑定 Cloud‑V3、类别表、split 与 action cache digest。
- calibration：只评一次，用于冻结 fixed-byte lambda、动作阈值、可逆融合参数和预注册 gate。
- final：只使用冻结 checkpoint/阈值/lambda；即使 calibration gate 失败也为完整性运行 post-hoc final，但不得产生 confirmatory claim。

## 6. 指标

主指标：Fixed-byte 10%/25%/40%，容量相对所有 final Original 总 bytes。

辅助指标：Fixed-call 10%/25%/40%。

每个单元格至少报告：

- Official AP50-95、AP50、AP75
- KB/image、实际调用率、Original/Q90/各 ROI 动作比例
- Cloud GFLOPs/image 与 10 Mbps E2E 估算
- ROI harmful-action rate、capacity utilization
- class、camera、event、目标尺寸分组

Teacher 报告 O/Q、最佳单 ROI、R12 Oracle、MAUR/Reversible 的 AP/KB frontier。

## 7. 统计检验

每项 1,000 次 paired bootstrap：

- 事件块 bootstrap 为主结果。
- image bootstrap 为敏感性分析。

固定 27 个比较：

- AFPD-OQR12 vs Raw ActionValue-OQ：3 个预算
- AFPD-OQR12 vs ORIC：3 个预算
- AFPD-OQR12 vs MORIC：3 个预算
- OccupancyFree vs AFPD：3 个预算
- OccupancyAware vs AFPD：3 个预算
- AFPD-OQR12 vs CurrentActionValue-OQ：3 个预算
- AFPD-OQR12 vs CurrentActionValue-OQR12：3 个预算
- AFPD-OQR12 vs Confidence：3 个预算
- AFPD-OQR12 vs Random：3 个预算

合计 `27 × 2 bootstrap units = 54` 个冻结比较。

## 8. 可恢复性与完成条件

每个 cache chunk、reward shard、fold/seed checkpoint、evaluation cell 和 bootstrap comparison 都写入输入签名和 completion hash；重启只跳过签名完全一致的产物。

完成必须同时满足：

- Cloud‑V3 模型 hash/epoch 校验通过。
- 3 roles 的 Original/Q90/R12 缓存完整且没有 Cloud‑V2 prediction hash。
- 所有学习方法存在全量 fit 新 checkpoint 或确定性无参数证明。
- Fixed-byte 容量无超限。
- final 未参与训练/选择。
- 54 个 bootstrap 比较完整。
- 生成 `RESULTS_ZH.md`、机器可读 CSV/JSON 和 `PIPELINE_COMPLETED.json`。

