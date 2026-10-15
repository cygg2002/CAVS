# 启动状态

- 实验：`three_dataset_legacy_style_rebuilt32d_control_v1`
- 目录：本实验专属 `runs/{coco,ena24,oregon}`
- 启动方式：`yolo26` 解释器、继承既有任务的
  `PYTHONPATH=/home/111_wcy/anaconda3/envs/cameratrapai/lib/python3.10/site-packages`
  及 `nice +10`。
- 依赖故障：首次 ENA24 启动未继承上述 `PYTHONPATH`，无法导入已有的
  `scikit-learn`。失败日志保存在新目录
  `runs/ena24/logs/pipeline.missing_sklearn.log`；未产生任何结果。
- 重启后的三个 pipeline 已进入 `core_running`。所有 current canonical 与
  20260723 九 Ranker TriAction 输入、输出目录和进程均未改动。
