# 启动状态

- 启动时间：2026-07-27 17:21 CST
- 服务器：`10.193.114.121:10022`
- 后台 worker PID：`18967`
- 调度 shell PID：`18960`
- nice：`10`
- 当前阶段：资源门禁等待
- 既有 `run_bootstrap_workpoint.py`：25 个匹配进程（父/子进程合计），未修改
- GPU 启动门槛：至少 6,144 MiB 可用显存
- final Cloud‑V2 文件数：0
- 远端核心契约测试：6/6 通过
- 预检：模型 SHA、四角色规模和源 manifest 通过

状态入口：

- `runs/pipeline_status.json`
- `runs/preflight.json`
- `logs/launcher.log`

流水线会在既有 CPU bootstrap 自然结束后，按 dev → freeze → final →
nine-Ranker TriAction → calibrated selector/risk gate 顺序续跑。
