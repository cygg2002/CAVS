# V3 结果证据台账

更新时间：2026-08-27。

## 1. COCO 离线全局 Fixed-byte

来源：`paper/experiments/20260813_coco_oregon_cavs_final_completion_v1/runs/coco/point/results.csv`（服务器归档）；Profile×Fusion bootstrap 来源为 `20260812_coco_profile_fusion_2x2_controlled_retrain_v2/runs/bootstrap/report.json`。

| 方法 | 10% | 25% | 40% | 证据等级 |
|---|---:|---:|---:|---|
| CAVS-OQ-Core | 0.478224 | 0.528884 | 0.559286 | 三种子 ensemble，事后完整点结果 |
| CAVS-OQR Current+MAUR | 0.480277 | 0.532900 | 0.560178 | 严格全重训，三种子 ensemble，事后完整点结果 |
| MORIC-AP5095 SharedAction-OQR-MAUR | 0.452230 | 0.496914 | 0.528170 | 三种子 ensemble，同协议重实现 |
| ORIC-AP5095 SharedAction-OQR-MAUR | 0.449174 | 0.494970 | 0.527069 | 三种子 ensemble，同协议重实现 |
| Confidence SharedAction-OQR-MAUR | 0.447029 | 0.489546 | 0.520311 | 同缓存、同容量 |
| Random-50 SharedAction-OQR-MAUR | 0.433586 | 0.468908 | 0.500443 | 50 种子 AP 均值 |

同骨干 CAVS-OQ Joint 相对 TwoStage 的提升为 +0.027850/+0.036146/+0.034582 AP。

OQ-only 公平表中的 MORIC-AP5095 为 0.452755/0.496918/0.528256，Confidence 为 0.447376/0.489725/0.520410；正文 OQR 主表不混用这两行，而采用上表 OQR-MAUR 执行口径。

Profile×Fusion 的 15 组、每组 1,000 次 image bootstrap 已完成。Current 下 MAUR−Reversible 的均值差和 95% CI 为：10% +0.001162 [−0.001916,+0.003460]；25% +0.001097 [−0.000061,+0.002212]；40% +0.000216 [−0.000607,+0.000946]。因此融合收益为正点估计，但三档区间均未严格排除 0。

## 2. Oregon Cloud-V3

来源：`20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/runs/report/RESULTS_ZH.md`；最新 Current@640+MAUR 点结果来源为 `20260814_oregon_cloudv3_current640_maur_point_only_v1/runs/point/results.csv`。

| 方法 | 10% | 25% | 40% | 证据等级 |
|---|---:|---:|---:|---|
| CAVS-OQ-Core | 0.499366 | 0.517611 | 0.519335 | Cloud-V3 从头重训，三种子点结果 |
| CAVS-OQR Current+MAUR | 0.506028 | 0.517405 | 0.516663 | 最新点结果；三档 ROI count 均为 0 |
| ORIC-AP5095 | 0.483153 | 0.502061 | 0.509599 | 同 Cloud-V3 重训 |
| MORIC-AP5095 | 0.479826 | 0.493286 | 0.506171 | 同 Cloud-V3 重训 |
| Confidence | 0.489643 | 0.502499 | 0.510449 | 同缓存 |
| Random-50 | 0.471087 | 0.484147 | 0.495973 | 50 种子 |

Oregon 已完成 AFPD/控制项的 1,000 次 event/image bootstrap，但尚无最新 CAVS-OQ 或 Current+MAUR 对全部主基线的统一 event-cluster bootstrap。正文因此只写点估计，不写“显著”。

## 3. VOC 跨检测器架构

来源：`20260825_voc_torchvision_crossarchitecture_cavs_oqr_paper_v1/runs/point/point_results.csv`（服务器）；协议为 SSDLite320-MobileNetV3-Large → Faster R-CNN-ResNet50-FPN-V2，均使用 Torchvision 官方 COCO 权重，不训练检测器。

| 方法 | 10% | 25% | 40% | 证据等级 |
|---|---:|---:|---:|---|
| CAVS-O | 0.520249 | 0.557280 | 0.589427 | 三种子 ensemble |
| CAVS-OQ | 0.530787 | 0.579142 | 0.614054 | 三种子 ensemble |
| CAVS-OQR | 0.531486 | 0.579678 | 0.612721 | 三种子 ensemble |
| MORIC-AP5095 SharedAction-OQ | 0.504416 | 0.554956 | 0.592212 | 三种子 ensemble |
| Confidence SharedAction-OQ | 0.495439 | 0.532334 | 0.569091 | 同缓存 |
| Random-50 SharedAction-OQ | 0.482501 | 0.503851 | 0.532850 | 50 种子均值 |

端点：Edge 0.473322；Original 0.657511；Q90 0.653684；ROI+MAUR 0.499819。OQ−O 为 +0.010538/+0.021862/+0.024627；OQR−OQ 为 +0.000699/+0.000537/−0.001333。1,000 次 paired bootstrap 尚未完成，不能写“显著”。

## 4. RK3588 硬件在环

来源：`paper/experiments/20260825_rk3588_board_results_sync/PAPER_READINESS_AUDIT_20260826.md`。

### 两阶段全局调度 HIL

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-OQR | 0.461068 | 0.503569 | 0.530318 |
| ORIC | 0.439772 | 0.470365 | 0.489860 |
| MORIC | 0.430747 | 0.461580 | 0.486673 |
| Confidence | 0.427594 | 0.461527 | 0.486238 |

这些运行使用真实 RKNN 与真实 HTTP，但在看完整数据集后进行全局调度，不是因果在线系统。

### 因果在线 HIL 先导结果（远程 GPU 云）

| 方法 | 10% | 25% | 40% |
|---|---:|---:|---:|
| CAVS-OQR | 0.454842 | 0.505302 | 0.529707 |
| ORIC | 0.439013 | 0.468441 | 0.488893 |

旧在线运行满足 `zero_lookahead=true` 和 `global_plan=false`，但有 1–5 次云回退、临时零面积框过滤、JSON Base64 约 1.34× wire 膨胀，且缺少 MORIC/Confidence/Random 完整统一矩阵；只能作为先导结果。

25% Mac 云先导运行：5,000 帧，AP 0.505289，3.76 FPS，平均 E2E 253.91 ms，P95 462.37 ms，JPEG payload 容量利用率 99.02%，前缀预算违规 0。该延迟带运行环境与协议 caveat，不进入最终系统主表。

## 5. 不得混写的口径

- COCO 论文缓存域、部署对齐离线域、RKNN 批处理 HIL、因果在线 HIL 是四个不同问题。
- ORIC/MORIC 为论文定义的同协议重实现，不称为作者官方 checkpoint 的逐位复现。
- Fixed-byte 约束 JPEG payload；若传输仍使用 Base64，必须另报 wire bytes。
- 任何仍未完成的统计或统一端测只保留占位符。
