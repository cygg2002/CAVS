# 执行计划

## Task 1：纯实验契约

- [ ] 定义 4 Profile × 3 imgsz 的唯一候选集合。
- [ ] 定义 fixed‑1.8 几何、tie-break 和 AP-safe working-point 选择。
- 验收：单测先失败，最小实现后通过。

## Task 2：共享几何与分尺度 cache

- [ ] 每个 role/profile 只编码一次 ROI Q90 JPEG。
- [ ] 每个 profile 的三个 imgsz 独立生成可恢复 prediction chunks。
- [ ] signature 包含 profile、imgsz、模型和输入 hash。
- 验收：dry-run 显示 4 个 geometry job、12 个 inference job，且没有 final。

## Checkpoint A

- [ ] 单测全部通过。
- [ ] 远端 preflight 校验模型、role 数量和源 cache。
- [ ] registered final accessed=false。

## Task 3：calibration 评测与冻结

- [ ] 生成 calibration 的 12 个候选。
- [ ] 输出单候选与多候选 Official AP/KB sweep。
- [ ] 写入 `CALIBRATION_FROZEN.json`，记录候选集合、最佳单候选和冻结 ROI ratio。
- 验收：freeze 文件 hash 固定，audit 阶段不得修改。

## Task 4：audit 一次性验证

- [ ] 生成 audit 的相同 12 个候选。
- [ ] 应用 calibration 冻结 ratio。
- [ ] 输出 frozen result、diagnostic ceiling 和类别/尺寸/放大率分解。
- 验收：报告明确区分 frozen point 与 GT-aware ceiling，不读取 final。

## Checkpoint B

- [ ] Official AP50-95/AP50/AP75、KB/image、eligible/safe rate 完整。
- [ ] audit `ΔAP >= -0.001` 且 KB 下降时才建议进入 ROI expert。
- [ ] 若不通过，定位为 profile、scale、Cloud crop domain 或融合问题并停止。

## 风险

| 风险 | 处理 |
|---|---|
| GPU 与其他任务争用 | 沿用至少 6 GiB 空闲门禁，不终止现有进程 |
| 12 候选重复 crop 占空间 | 同 profile 三尺度共享 geometry/JPEG |
| 不同尺度错误复用 prediction | imgsz 写入 signature 与独立目录 |
| audit 被用于调参 | candidate set 与 ratio 先写 freeze，audit 只读取 |
| GT-aware Oracle 被误报为部署结果 | 所有表中显式标注 diagnostic ceiling |
| 事件映射文件包含 final 行 | 本轮不读取；待 role-specific manifest 后补 |
