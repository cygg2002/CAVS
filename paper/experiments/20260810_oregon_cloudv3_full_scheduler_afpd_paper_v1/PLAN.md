# 实施计划

1. 冻结 Cloud‑V3、canonical manifests、Edge/FPN caches 与 event assignments 的 hash；验证 role identity 互斥（允许不同 role 的数值 image_id 重复）。
2. 并行预生成四个 Profile 的真实 ROI geometry；等待 GPU 资源门槛后依次生成 Cloud‑V3 Original、Q90 与 12 个 ROI prediction caches。
3. 生成 MAUR/Reversible 候选与 fold-contained contextual AP50-95 reward；完成 O/Q/R12 Teacher ceiling。
4. 用 event-group 5-fold OOF 并行训练所有历史、外部、当前与 frontier 学习器；保存 3-seed full-fit checkpoint。
5. calibration 一次冻结 action gates、lambda、fusion 与 protocol gate。
6. 在 post-hoc final 上运行完整 fixed-byte/fixed-call 方法矩阵、子组审计与 action/fusion 消融。
7. 并行执行 27 个 event-block 与 27 个 image-level paired bootstrap。
8. 汇总中文报告、CSV/JSON、hash provenance 和完成标记。

资源策略：64 核机器上最多 40 个 CPU worker；每个学习器 worker 固定 1 个 BLAS/Torch thread。GPU cache 使用小 batch 和显存/利用率门控，不中断 Cloud‑V4 或其他用户进程。

