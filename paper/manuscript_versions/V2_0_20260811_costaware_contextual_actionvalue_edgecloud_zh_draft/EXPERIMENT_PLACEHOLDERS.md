# V2 投稿前待补实验与信息清单

本文件集中登记 `MANUSCRIPT_V2_0_ZH_DRAFT.md` 中的空缺。优先级含义：P0 为支撑论文主结论的必要项；P1 为显著增强可信度或系统性但可移入补充材料的项目；P2 为排版、声明和复现元数据。

## P0：主结果必须完成

### P0-1 Oregon Cloud-V3 统计闭环（点结果已完成）

- 已冻结 Cloud-V3 epoch-28 `best.pt`（SHA-256 `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`）、类别映射和动作缓存。
- Cloud-V3 的 fit/calibration/final Original、Q90 与 ROI 缓存已生成；依赖云输出的内部/外部学习器已从头训练，未复用 Cloud-V1/V2 学习器。
- CAVS-OQ-Core、Random、Confidence、ORIC/MORIC、Current-BudgetSet、SetValue、DualHead 的 Fixed-byte/Fixed-call 点结果已写入正文。
- 完成 CAVS 对最强外部方法、Confidence、Random 和 DualHead 的冻结 paired bootstrap，并核对 image/event 两种单位。
- 完成 camera/event-cluster bootstrap 与类别/目标尺寸/相机/事件分解。
- 点结果已替换摘要、第 7.7 节表 10–11 和结论；只待补统计区间与分解结果。

### P0-2 COCO CAVS-OQ-Core 的冻结统计确认（点结果已完成）

- CAVS-OQ-Core、同骨干 TwoStage、CostInValue、三种子 ORIC/MORIC、Confidence 和 Random-Mean50 点结果已在统一协议下完成。
- 以冻结 CAVS-OQ-Core、冻结比较器和相同 Fixed-byte cache/evaluator 完成正在运行的 paired bootstrap。
- 三个预算分别比较：最强 ORIC/MORIC、Confidence、Random-Mean50、TwoStage 和 CostInValue。
- 报告 bootstrap repetitions、seed、均值差、95% CI 与 `Pr(Δ>0)`。
- Random 统计对象必须是每个 bootstrap replicate 内对 50 套 selection 分别评 AP 后取均值，不能退回单个 base seed。
- 不使用 val2017 结果重新选择方法、阈值或比较器。

### P0-3 新注册确认集

- 新建与已有 COCO train/val 调参历史和 Oregon 已访问 final 互斥的确认协议，或选择可公开复现的外部测试域。
- 在访问前冻结 CAVS、所有比较器、超参数、动作缓存和成功门槛。
- 报告 Fixed-byte 主结果与 Fixed-call 边界。
- 若无法建立，应把论文结论和题目明确限定为 post-hoc/exploratory study，不能声称确认性 SOTA。

### P0-4 主部署实例与动作公平性冻结

- 当前方法名称固定为 CAVS；OQ 是不依赖 ROI 的核心实例，OQR 是加入 ROI 的完整动作实例。
- 只有当 Current@640+MAUR 在 COCO 和 Oregon Cloud-V3 的冻结统计都通过时，CAVS-OQR 才进入摘要和主表首行，CAVS-OQ 作为动作空间消融。
- OQR 主比较必须让 ORIC/MORIC、Confidence 和 Random 使用同一 ROI 可用性、payload 和冻结动作政策，或把“同动作调度公平比较”与“完整系统比较”分成两张表。
- 不把新增 ROI 动作上界直接归因于学习器，也不把区域选择/拼接声称为首次提出。

## P0：真机系统证据

### P0-5 RK3588 模型闭环

- 将 Edge detector 转换为 RKNN INT8 或 FP16，冻结量化配置和 SHA。
- 重新生成所有数据角色的 Edge 检测集合；量化后不得复用桌面 Edge 特征训练学习器。
- 基于新 Edge 输出重新训练 CAVS、Confidence 和 ORIC/MORIC。
- 记录 RKNN Toolkit、runtime、kernel、驱动、系统镜像和散热方案。

### P0-6 端到端多带宽实验

- 在 1/5/10/20 Mbps 下测量组件和端到端 P50/P95/P99。
- 同时记录 payload bytes 与 wire bytes、FPS、deadline miss、队列长度、重传和失败率。
- 测量 Edge-only、Always-Q90、Confidence、ORIC/MORIC 与 CAVS。
- 增加 Cloud-first/weak-skipping 边界，量化高上传预算下重复运行 Edge detector 的成本；若不实现完整 weak-skipping router，至少报告跳过 Edge 的理想时延上界。
- 报告空闲/负载功率、J/frame、温度和降频；确定长时间稳定运行时长。
- 实现时间窗口或令牌桶 Fixed-byte，而不是用离线批次结果冒充在线调度。
- 替换第 5 节表 3、第 7.9 节表 12 和摘要；真机结果完成前不列为已完成贡献。

## P1：必要消融与机制补强

### P1-1 检测器基础上下界

- 在与 CAVS 主表相同 evaluator 下重算 COCO 和 Oregon Cloud-V3 的 Edge-only、Cloud-only Original、Cloud-only Q90。
- 报告 Official AP、AP50、AP75、KB/图，以及 Q90 相对 Original 的精度和字节变化。

### P1-2 最小归因消融（主要点结果已完成）

- 同骨干 CAVS-OQ-Core-TwoStage vs Joint-OQ：点结果已完成；待补冻结 bootstrap。
- CostInValue-OQ vs 去除 cost 的 CAVS-OQ-Core：点结果已完成；待补冻结 bootstrap。
- Original-only、Q90-only、OQ joint。
- Raw vs per-action calibration。
- OQ vs same-run OQR、同 selection Reversible vs MAUR、Profile×Fusion 2×2：点结果已完成；严格 2×2 v2 bootstrap 待完成。
- 保持相同 seeds、reward、特征、scheduler 与 evaluator。

### P1-3 调度器最优差距

- 在可由动态规划或 MILP 求解的小规模子集上比较当前拉格朗日+贪心修复与精确多选背包。
- 报告 value objective gap、Official AP gap、运行时间和容量利用率。
- 若差距不可忽略，考虑只在离线窗口使用更精确求解，或弱化“近最优”表述。

### P1-4 Reward 定义与可学习性闭环

- 报告 Oracle/Teacher、learned value 与最终 scheduler 三层结果，避免只给 Teacher ceiling 或最终 AP。
- 汇总 learned score 对真实 image-action reward 的 Spearman、Teacher top-K overlap、正收益动作 precision/recall 和 Teacher gain recovery。
- 在不访问 final 的 fit OOF/calibration 上检查 context size 与采样种子稳定性；建议比较 8/16/32/64 或在同等计算量下选择代表性子集。
- 报告 Edge 优于 Cloud 的图片比例，并使用 TIDE 或等价的分类、定位、重复、背景和漏检分解说明端云互补性。

### P1-5 最近邻方法与图表

- 正文结构表必须包含 Bandwidth-Adaptive DCSB、ICDM 2025 Selective Cloud Offloading 和 Budget-Adaptive Routing。
- 若无法忠实复现 DCSB/Selective Cloud Offloading，明确说明协议不可直接等价，并在 Fixed-call 补充材料给出 DCSB-style/不确定性 ROI 规则基线或结构性比较；不得将其省略后声称全面优于既有 ROI 方法。

- 图 1：CAVS 架构，突出 bytes 不进入 Value head。
- 图 2：RK3588—网络—云端测试床与回退链路。
- 精度—KB Pareto 曲线：COCO 与 Oregon Cloud-V3。
- 动作比例随预算变化的堆叠图。
- 可选：按目标尺寸/类别的 CAVS 相对 MORIC 增益图。

## P2：复现与稿件信息

- COCO Edge/Cloud 型号、输入尺寸、checkpoint SHA。
- Oregon Cloud-V3 与 RKNN Edge 的型号、量化类型、SHA。
- 三个训练 seed、batch size、reward 标准化、tie-break、payload 单位、参数量。
- CPU/GPU 型号、并行度、训练时间、Python/PyTorch/Ultralytics/RKNN 版本。
- Oregon 正式数据集名称、论文引文、许可证、下载/申请方式。
- 作者、单位、通讯作者、基金、CRediT、伦理声明。
- 匿名仓库、代码归档 DOI、数据/缓存发布边界。
- 参考文献格式按目标期刊统一；确认 YOLO26 预印本出版状态。

## 建议执行顺序

1. 先让 COCO 20260813、严格 2×2 v2 和 Oregon 的冻结 bootstrap 正常完成，不再改变方法。
2. 将统计区间、单模型均值±标准差和分解结果写回正文与补充材料。
3. 冻结 RKNN Edge，重新生成真机特征并做 1/5/10/20 Mbps 测量。
4. 在任何新 final 前锁定 manuscript claim、成功门槛与比较器。
5. 最后补 reward/context 稳定性、精确调度器差距、最近邻结构对照、图表、环境和声明信息。

## 明确不再重复的探索

- 不继续开发未通过 gate 的 Risk/LCB、Factorized Global Action Ranker、Selection-context Teacher 和 Occupancy v3。
- 不用 COCO val2017 继续选择新模型。
- 不把 Cloud-V2 GT-aware Profile×Scale teacher 当作可部署结果。
- 不把不同 detector、不同 training run 或不同 payload protocol 的点估计直接做方法归因。
