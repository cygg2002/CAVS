# 投稿补强扩展实验状态

- 启动时间：2026-07-27 11:43 CST
- 服务器：`fzu111`（SSH `10.193.114.121:10022`）
- 运行环境：原论文实验记录的 `yolo26` Python 与 `cameratrapai` site-packages
- 原则：只读取 20260720/20260723/20260725 冻结产物；所有新增输出仅写入本实验目录

## 已通过的启动前检查

- 本地与远端 14 项测试通过。
- COCO、ENA24、Oregon 均成功加载冻结 calibration/audit/final 缓存。
- 三个数据集各 10 个门禁动作对均重建成功，动作对 SHA-256 与 20260725 原门禁记录完全一致。
- Random Online 的阈值只使用同种子的 calibration 随机分数。
- Random Fixed-byte 使用冻结 SharedAction 载荷图和随机优先级贪心装箱，不使用学习型 lambda。

## 已完成

- ENA24 Bootstrap 200/1000/2000 稳定性。
- COCO、ENA24、Oregon 的 Random Online/Fixed-byte 50 seeds。

## 2026-07-27 14:36 CST 恢复状态

原单核任务停止后，所有 raw JSON checkpoint 均通过结构校验。远端与本地
18 项测试通过，并用 COCO 真实数据验证 1-worker 与 2-worker 的两个
replicate 结果逐项完全相同。

| 任务 | 控制器 PID | 执行方式 | 完成条件 |
|---|---:|---|---|
| COCO Bootstrap | 15182 | 10 工作点并行，每工作点 2 workers | `runs/bootstrap_sensitivity/coco/COMPLETED` |
| Oregon Bootstrap | 15183 | 10 工作点并行，每工作点 2 workers | `runs/bootstrap_sensitivity/oregon/COMPLETED` |

每个工作点只有一个父进程写唯一 raw checkpoint；worker 仅计算，结果按
原 replicate 序号回收。当前总 CPU 占用约 41 核，任务以 nice=5 运行。
控制器会等待全部工作点成功后调用原串行入口统一生成总表、provenance
和 `COMPLETED`。在两个完成标记与来源哈希复核通过前，论文不得引用中间值。

Oregon Fixed-byte 40% 是基准与候选动作完全相同的快捷点，不需要 raw
checkpoint；已修复旧远端脚本在该情况下错误读取不存在 raw 文件的问题。

## 本轮明确不冒充完成的项目

- BCa：Official AP 非逐图可加，尚无预注册且可承受的 acceleration 估计方案。
- LTT fixed-sequence：尚未预注册有效 p 值构造与候选检验顺序。
- ENA24 独立门禁扩样：仍需要新独立数据或完整重划分重训。
- 真机端到端测试床：仍需要冻结载荷回放清单、独占/安静 GPU 时段和原始链路日志。
