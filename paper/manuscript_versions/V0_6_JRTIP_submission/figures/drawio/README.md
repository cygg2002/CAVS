# Draw.io Figure Sources

本目录保存 JRTIP V0.6 三张系统流程图的可编辑 draw.io 源文件：

| 图号 | 中文可编辑源 | 英文可编辑源 |
|---|---|---|
| Fig. 1 | `Fig1_online_edge_cloud_architecture.drawio` | `Fig1_online_edge_cloud_architecture_en.drawio` |
| Fig. 2 | `Fig2_cafru_offline_training_online_decision.drawio` | `Fig2_cafru_offline_training_online_decision_en.drawio` |
| Fig. 5 | `Fig5_t41_bounded_queue_deadline.drawio` | `Fig5_t41_bounded_queue_deadline_en.drawio` |

中文导出使用原文件名，英文导出在文件名后增加 `_en`；PNG、PDF 位于
上一级 `figures/` 目录。

同时保留 `.svg` 预览源。PNG 为 4800 px 宽、约 701 dpi，适合以
174 mm 双栏宽度插入论文。

重新生成命令：

```bash
/Users/cygg_macmini/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  paper/tools/build_jrtip_drawio_figures.py
```

生成脚本：

`paper/tools/build_jrtip_drawio_figures.py`

这些 `.drawio` 文件采用未压缩 mxGraph XML，可直接在 draw.io 桌面版
打开并继续编辑。PNG/PDF 预览由生成脚本使用同一坐标和样式模型导出。
