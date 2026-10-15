# 执行计划

1. **P0 隔离与资源审计**
   - 校验新旧模型、数据划分和源缓存哈希。
   - 记录当前 CPU/GPU 任务；不终止、不降优先级、不改写。

2. **P1 开发角色动作缓存**
   - 复用 GT、Edge、FPN、Q90 JPEG、ROI crop/几何/字节。
   - 串行生成 fit、calibration 的新 Original/Q90/ROI 预测。
   - 导入并核验已完成的 audit Original/Q90/ROI 预测。

3. **P2 Canonical Dev**
   - 重建 fit/calibration/audit common cache 和 PCA。
   - 重训七个 canonical Ranker 与共享动作头。
   - 输出 audit 系统基线、Fixed-call/Online/Fixed-byte、随机种子和 bootstrap。
   - 写入 `DEV_FROZEN.json`，冻结模型与门禁。

4. **P3 Final 动作缓存**
   - 只有 P2 成功后才生成 final Original/Q90/ROI。
   - 重建 final common cache，检查 P2 冻结哈希不变。

5. **P4 九 Ranker × TriAction**
   - 重训 Current-Controlled-32D 与 GBDT-Hierarchical。
   - 跑 384 行主矩阵、1,200 行 Random raw、120 行 per-Ranker 消融。
   - 跑 Fixed-call 主预算的 24 个 paired-bootstrap 对比。

6. **P5 校准式成本感知选择器**
   - calibration 从 24 个候选中产生 provisional 配置。
   - audit 执行 200 次 paired bootstrap 风险门禁；失败直接回退 reference。
   - 冻结后读取 final 一次，输出 10 个工作点和 final-oracle regret 诊断。

7. **P6 投稿补充**
   - full / no-gate / no-cost / Confidence+SharedAction 消融。
   - global、dataset-adaptive、reference、final-oracle 对比。
   - Random Online/Fixed-byte、bootstrap 敏感性、Oregon event-cluster bootstrap。
   - 汇总 V1 与 Cloud‑V2 的 AP、KB、E2E、动作比例与 Pareto 差异。

