# Legacy-Style Rebuilt-32D 三数据集对照计划

## 冻结决策

- 新目录独立；不修改 20260720 canonical、20260722 Legacy control 或正在运行的
  20260723 九 Ranker TriAction 实验。
- 主矩阵加入 `Legacy-Style-Rebuilt-32D`，不把 COCO/Oregon 标为
  `Legacy-Frozen-32D`。
- Legacy 配方固定为：当前 fit-only pca32、当前 ClassGainPrior 分数作 anchor、
  32D budget-residual pairwise ranker、`rank_scale=0.25`、80 epochs、
  `lr=0.03`、`l2=0.001`、`max_pairs=20000`、主预算 10/25/40%。
- ENA24 真正的 `Legacy-Frozen-32D` 只在最终报告作为已完成 bridge reference。

## 任务

- [x] 任务1：写独立实验规格与计划。
  - 验证：明确三方 Ranker、输入、输出、不能做的操作。

- [x] 任务2：先写十 Ranker / Legacy-style 失败合同测试。
  - 验证：旧实现无法通过十 Ranker 与新的 score-contract 测试。
  - 文件：`tests/`、`scripts/evaluation_core.py`。

- [x] 任务3：接入 fit-only Legacy-style residual ranker。
  - 验证：测试通过；模型文件为 32 维，校准/final score 有限且 ID 对齐。
  - 文件：`scripts/evaluation_core.py`、`scripts/run_dataset.py`。

- [x] 检查点A：本地单测、编译、三个远端 preflight。
  - 验证：canonical 输入没有改动，矩阵规模为 429/135/1200/27。

- [~] 任务4：启动三个低优先级、独立输出的可恢复 pipeline。
  - 验证：每个数据集有 `pipeline_status.json` 和独立日志；不向现有目录写入。
  - 文件：`notes/launch_status_zh.md`、`runs/{dataset}/`。

- [ ] 任务5：完成后检查核心复现、输入 SHA、行数、bootstrap 完整性。
  - 验证：任一门禁失败则保留失败并不以 final 回调参数。

- [ ] 任务6：写三数据集中文对照报告与 ENA24 Frozen bridge 附录。
  - 验证：明确“rebuild”与“frozen”区别，结论只使用同协议三方比较。

## 风险与控制

| 风险 | 控制 |
|---|---|
| 旧 Frozen 输入与当前 COCO/Oregon 不兼容 | 只重建 Legacy 配方；名称明确为 Rebuilt |
| 新 ranker 改变共享 GBDT 阈值 | 新实验内重新校准，旧实验保持不变 |
| final 后调参 | score scale 和超参先写入 protocol，final 仅评估 |
| 当前任务资源受影响 | 独立目录、`nice +10`、64 核机器上独立进程 |
