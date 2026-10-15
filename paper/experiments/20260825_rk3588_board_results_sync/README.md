# RK3588 板端实验结果同步归档

同步日期：2026-08-25

来源设备：`orangepi@10.133.10.18`（Orange Pi 5 Ultra / RK3588）

本目录从板端实验盘只读同步，不覆盖原实验目录。同步后使用 `rsync --dry-run` 对四组来源逐项复核，待传文件均为 0。

## 1. 完整批处理 HIL 结果

### CAVS-OQR / RKNN-FP16

目录：`batch_hil/cavs_oqr_fp16_complete`

- 协议：两阶段全局 Fixed-byte 调度，真实 RK3588 推理和真实 HTTP 云请求。
- 数据：COCO val2017，5,000 张。
- 云请求：4,548 / 4,548 成功。
- Edge-only AP50-95：0.405264。

| 预算 | AP50-95 | AP50 | AP75 |
|---|---:|---:|---:|
| 10% | 0.461068 | 0.643721 | 0.494693 |
| 25% | 0.503569 | 0.693798 | 0.542448 |
| 40% | 0.530318 | 0.723362 | 0.572609 |

### 原始 ORIC / MORIC / Confidence / RKNN-FP16

目录：`batch_hil/oric_moric_confidence_fp16_complete`

- 协议：两阶段全局 Fixed-byte 调度，Edge/Original 动作空间。
- 数据：COCO val2017，5,000 张。
- 唯一云请求：3,478 / 3,478 成功。

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| ORIC | 0.439772 | 0.470365 | 0.489860 |
| MORIC | 0.430747 | 0.461580 | 0.486673 |
| Confidence | 0.427594 | 0.461527 | 0.486238 |

精确指标、动作数、上传字节和延迟分位数见各目录的 `summary.json`。

## 2. 因果在线 HIL 原始运行记录

目录：`online_hil/runs`

已同步 11 个 formal run：

| 方法/链路 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-OQR / Mac 云 | — | 5,000 | 5,000 |
| CAVS-OQR / 远程 GPU 云 | 5,000 | 5,000 | 5,000 |
| ORIC / 远程 GPU 云 | 5,000 | 5,000 | 5,000 |
| MORIC / Mac 云 | — | — | 5,000 |
| MORIC / 远程 GPU 云 | 5,000 | **963（未完成）** | — |

每个 run 保留 `frames.jsonl` 和 `run_contract.json`。`online_hil/board_bundle` 同时保留运行合同、策略、控制器参数和执行脚本。

注意：板端没有发现已完成的 Confidence 或 Random 因果在线 formal run。二者只有控制配置，不能写成已完成结果。MORIC 25% 的 963 帧运行也不能用于正式比较。

## 3. 未完成的部署对齐运行

目录：`partial_runs/deployment_aligned_fp16_4719_of_4720`

- 阶段：upload。
- 成功请求：4,719 / 4,720。
- 仍缺 1 个请求，没有最终 `summary.json`。
- 仅作故障恢复和审计，不得进入论文主表。

## 4. 完整性说明

- 本地归档总量约 2.1 GB，共 126 个文件（不计本说明）。
- 正式同步排除了 `.run.lock`、`._*` 和 `__pycache__`。
- 未同步 smoke run 和明确标为 `PARTIAL_DO_NOT_USE` 的旧包。
- 四组来源在同步后的 `rsync --dry-run` 中均显示 `Number of files transferred: 0`。
