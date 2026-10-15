# ORIC/MORIC × TriAction 三数据集独立扩展实验规范

## 1. 目的

在不修改以下两个已完成实验的前提下，为忠实重实现的 `ORIC-MLP` 与
`MORIC-MLP` 补齐两种后续动作策略：

- `CostEligible-AllROI`
- `GBDT-Hierarchical`

覆盖 COCO、ENA24、Oregon，以及 `Fixed-call`、`Online`、`Fixed-byte`
三种预算语义。

## 2. 冻结来源

- 排序模型、输出特征定义和分数：
  `20260728_three_dataset_dcsb_oric_moric_external_baselines`
- 数据划分、检测器缓存、Official AP 评估器和 SharedAction：
  `20260720_frozen_detector_multiranker_contextm2e`
- SafeROI GBDT、共享动作阈值及动作语义：
  `20260723_three_dataset_nineranker_triaction_paper_exploration`

所有来源只读，并在运行前后校验 SHA-256。

## 3. 严格控制

1. ORIC/MORIC 不重新训练；直接加载 20260728 已冻结的模型。
2. `GBDT-Hierarchical` 不因加入 ORIC/MORIC 而重新选阈值；使用既有
   8-Ranker 共享 calibration 阈值。
3. `CostEligible-AllROI` 对所有“ROI 可用且比实际 SharedAction 基础动作更省
   字节”的图片使用 ROI。
4. Fixed-call 与 Online 中，动作策略不得改变上传图片集合，只能改变载荷。
5. Online 上传阈值必须复现 20260728 的冻结阈值。
6. Fixed-byte 的载荷改变会改变可容纳图片集合，因此只允许在 calibration
   上为每个 Ranker × 动作策略 × 预算重新校准 λ；final 仅评估一次。
7. 先完整复现 20260728 的 `Original` 与 `SharedAction` 40 个控制点，指标漂移
   超过 `1e-12` 时拒绝产生新结果。
8. Oregon 继续标记为 `posthoc_exploratory_final_reuse`。

## 4. 结果矩阵

每个数据集：

- 2 个 Ranker；
- 2 个新增动作策略；
- 3 个 Fixed-call + 3 个 Online + 4 个 Fixed-byte 工作点。

共 `2 × 2 × 10 = 40` 条唯一结果；三个数据集共 120 条。

## 5. 成功标准

- 三个数据集均生成 `completion.json`，且 `complete=true`。
- 每个 `results.csv` 恰好 40 条唯一数据行。
- 20260728 Original/SharedAction 控制矩阵复现通过。
- Fixed-call/Online 两个新增动作策略的上传图片 ID 完全一致。
- GBDT 模型和共享阈值来源哈希被记录。
- final 不参与模型训练、Online 阈值、ROI 阈值或 Fixed-byte λ 的选择。
- 两个既有实验目录的来源哈希在运行前后完全一致。

