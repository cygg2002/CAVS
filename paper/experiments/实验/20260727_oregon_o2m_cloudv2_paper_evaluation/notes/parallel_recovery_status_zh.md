# 九 Ranker 并行恢复状态

- 恢复时间：2026-07-27 22:25 CST
- 服务器：`10.193.114.121:10022`
- 后台调度 PID：`59432`
- 当前阶段：`nineranker_core`
- 并行策略：core 保持单写者；core 冻结后，将 24 个注册 bootstrap
  工作点按唯一 action-pair SHA-256 分发到 8 个独立进程。
- 汇总策略：仅父进程按注册 Ranker × budget 顺序写入 checkpoint、CSV、summary
  和完成标记。
- 随机协议：每个工作点仍使用原协议的 200 次 paired bootstrap、seed
  `20260720` 和 image-level 抽样单位。
- 数值线程：每个进程限制为 1，避免 8 个进程内部再次过度并行。
- 恢复日志：`logs/recovery_parallel_nineranker_selector_20260727.log`
- 状态入口：`runs/pipeline_status.json`

验证：

- 本地测试：8/8 通过。
- 远端原环境测试：8/8 通过。
- 注册网格：8 个非随机 Ranker × 3 个主预算。
- 冻结输入预检：通过。
- `models.json` SHA-256：
  `2b2a9f0e7d186d94bb20505e4e7ae28891e9a4f84765573ae1e81c46fe9bb584`。
- `action_gate.json` SHA-256：
  `f3ae7c0eeccb76285cbaf0c838c3a14b62cfb3aa9f5dd5fceab59452fa6100b9`。

本次新增恢复器只位于当前 Cloud-V2 独立实验目录，不修改
20260720、20260723 或 20260725 的冻结实验。完成前不得将中间结果写入论文。
