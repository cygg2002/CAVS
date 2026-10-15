# CAVS 论文服务器最终补全实验规格

## 目标

一次性闭合论文服务器侧统计证据，不重新训练检测器或学习器，不修改既有点结果、
checkpoint、score、selection 或候选预测。所有 bootstrap 都严格复用已冻结的 final
selection 与原 evaluator。

## 注册矩阵

| 实验 | 重采样单位 | 比较任务 | 每任务重复 | Random 口径 |
|---|---|---:|---:|---|
| COCO CAVS 最终比较 | image | 36 | 1,000 | 每次重采样内 Random-50 AP 均值 |
| Oregon CAVS 最终比较 | event + image | 72 | 1,000 | 每次重采样内 Random-50 AP 均值 |
| VOC Torchvision 跨架构 | image | 36 | 1,000 | 每次重采样内 Random-50 AP 均值 |
| COCO Original-only 原生基线 | image | 12 | 1,000 | 每次重采样内 Random-50 AP 均值 |

总计 156 个任务、156,000 个 bootstrap replicate。Oregon 的正文主统计单位为
event cluster，image bootstrap 只作敏感性分析。

## 不变量

- COCO：fit=9,678、calibration=571、final=5,000。
- Oregon：fit=9,923、calibration=4,961、final=4,962。
- VOC：fit=15,629、calibration=922、final=4,952。
- 每个学习器的三个 full-fit seed 和 ensemble 均沿用已冻结点结果。
- final/test 不进入训练、超参数选择、动作选择或在线控制校准。
- 只允许从合法 `.partial.json` 的连续前缀恢复；不得跳过、重排或重复 replicate。
- 每个 job 的随机数生成器仍由原实验的 `bootstrap_seed + job_id` 派生。
- 分片 worker 必须为 replicate `r` 重放原 RNG 的前 `r` 次抽样，因此并行化不改变样本序列。
- 统计汇总继续调用源实验的 summary/atomic writer；完成文件仍写入源实验既定输出位置。

## 源协议

源脚本及协议的 SHA-256 固定在 `configs/protocol.json`。加速器启动前逐项校验；任一
哈希变化都必须失败，不能在未知代码上继续。

## 并行策略

原实现按比较 job 并行，进入少量 nested Random-50 尾部时只能使用 3--6 个 CPU。
本实验仅把尚未完成的 replicate 区间切成互不重叠的连续分片，在 fork worker 中运行
原 evaluator。父进程只提交已经连续拼接的前缀，并原子更新断点。

VOC 的原始 Python evaluator 还提供一个等价的预索引执行路径；它必须先在既有 canonical
partial replicate 上逐值对齐，未通过时自动禁用并失败，不允许静默换 evaluator。

## 完成门槛

1. 四个源实验的 bootstrap `completion.json` 均存在且 `complete=true`。
2. 任务数严格为 36/72/36/12；每任务 `replicates=1000`。
3. nested Random 任务均记录 `random_seed_count=50` 和 nested-per-resample 语义。
4. Oregon 同时存在 event 与 image 两套结果。
5. 三数据集点结果均可提取 3 个单模型、均值、样本 SD、ensemble AP 和 ensemble 增益。
6. 生成统一审计报告，记录输入/输出哈希、进程资源、失败与恢复历史。

