# DCSB 与 ORIC/MORIC 外部基线运行说明

## 范围

本目录独立评测 COCO、ENA24 和 Oregon 上的 `DCSB-Native`、
`DCSB-Score-Adapted`、`ORIC-MLP` 与 `MORIC-MLP`。程序只读
`20260720_frozen_detector_multiranker_contextm2e` 的冻结输入，所有新产物只写入
本目录的 `runs/`。不要把 `SharedAction` 结果表述为外部方法的原生载荷；它只是
控制载荷差异的诊断。

## 服务器前置条件

- 仓库位于 `/home/111_wcy/work/yolo26/ultralytics-main`。
- 20260720 冻结实验的三数据集角色、预测缓存、`models.json` 和
  `action_gate.json` 已完整同步。
- 当前 Python 环境可导入 NumPy 和 PyTorch。
- 运行期间不要修改协议、实现代码或冻结输入。来源哈希变化会使 checkpoint 或
  完成标志失效，这是预期保护行为。

## 推荐运行顺序

先进入实验目录：

```bash
cd /home/111_wcy/work/yolo26/ultralytics-main/paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines
```

先运行本地合同测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

三数据集预检可以并行：

```bash
python3 scripts/run_pipeline.py \
  --datasets coco ena24 oregon \
  --stage preflight \
  --max-parallel-datasets 3
```

ORIC 上下文奖励是最耗时阶段。建议先串行运行，降低 CPU 和内存竞争；中断后重跑
同一命令会从 `reward_checkpoint.npz` 继续：

```bash
python3 scripts/run_pipeline.py \
  --datasets coco ena24 oregon \
  --stage rewards \
  --max-parallel-datasets 1
```

奖励完成后训练模型并评测。资源充足时可并行三个数据集：

```bash
python3 scripts/run_pipeline.py \
  --datasets coco ena24 oregon \
  --stage evaluate \
  --max-parallel-datasets 3
```

若服务器内存不足，将最后一个参数改为 `1`。不推荐首次直接使用 `--stage all`，
因为分阶段运行更容易检查来源合同和恢复长时间奖励计算。

## 完成条件与输出

每个数据集必须同时存在：

```text
runs/<dataset>/preflight/completion.json
runs/<dataset>/rewards/completion.json
runs/<dataset>/evaluation/completion.json
runs/<dataset>/evaluation/results.csv
runs/<dataset>/completion.json
```

`results.csv` 必须恰好有 65 个数据行。管线日志位于 `runs/logs/`。核心输出包括：

- `rewards/reward_checkpoint.npz`：可恢复的逐图 ORIC 奖励进度；
- `rewards/fit_artifacts.npz`：fit-only 特征与奖励；
- `rewards/dcsb_model.json`：fit-only DCSB 阈值；
- `evaluation/oric_mlp.pkl` 与 `moric_mlp.pkl`：最终 all-fit 模型；
- `evaluation/frozen_controls.json`：Online 阈值和 Fixed-byte 参数；
- `evaluation/results.csv`：三类预算下的 Official AP 与实际请求字节结果。

## 结果使用边界

- `DCSB-Native` 只报告自然上传率，不强行对齐十个预算点。
- `DCSB-Score-Adapted` 是为统一预算曲线增加的调度适配，不是原始 DCSB。
- `ORIC-MLP` 和 `MORIC-MLP` 是论文与官方代码驱动的忠实重实现，不是原作者数值复现。
- `Original` 才是外部方法主结果；`SharedAction` 只能作为受控诊断。
- Oregon 使用已访问终评的 post-hoc 复用，不能称为新的隐藏测试结果。
