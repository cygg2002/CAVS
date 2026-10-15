# Online Resource 1

## Supplementary Experiments and Reproducibility Details for
## CAFRU: Cost-Aware Frame-ROI Selective Offloading for Real-Time Edge-Cloud Object Detection on Low-Power Cameras

Author names, affiliations, and corresponding-author e-mail to be completed.

This resource contains full operating-point mappings, secondary ablations,
difficulty-oriented analyses, metric-fairness checks, T41 prototype details,
and source-file provenance removed from the 12-page main manuscript.

## S1 Reproducibility scope

### S1.1 Software and data

- Detection framework: the project-local Ultralytics YOLO26 implementation.
- Edge detector: `yolo26s.pt`.
- Cloud reviewer: `yolo26x.pt`.
- Uploader training data: COCO train2017, 118,287 images.
- Uploader internal validation: deterministic 10% split, seed 42.
- Final algorithm evaluation: COCO val2017, 5,000 images.
- Edge/cloud prediction confidence: 0.25 unless otherwise stated.
- Utility match threshold: class-consistent IoU >= 0.5.
- False-positive utility penalty: \(\alpha=0.25\).
- Gain normalization: divide by `max(1, number of labels)`.
- Feature-noise standard deviation: 0.02.
- Default payload-cost weight: \(\lambda_b=0.02\).
- Default cloud input: 512 for deployment ablations.
- Default JPEG quality: Q80 for deployment ablations.

### S1.2 Lightweight heads

The policy contains 15 frame features and 15 ROI features. Positive-gain
classifiers are class-balanced logistic regressors. Gain estimators are linear
regressors trained with Huber loss. The saved metadata reports:

| Head | Samples | Positive rate | Validation metric |
|---|---:|---:|---:|
| Frame probability | 118,287 | 0.4603 | accuracy 0.7110; Brier 0.1916 |
| Frame value | 118,287 | - | MAE 0.1118 |
| ROI probability | 114,827 | 0.0501 | accuracy 0.9498; Brier 0.0453 |
| ROI value | 114,827 | - | MAE 0.0304 |

The high ROI classification accuracy must be interpreted with the 5.01%
positive rate and the reported Brier score; accuracy alone would be
misleading for this imbalanced branch.

## S2 Formal operating-point mapping

The main paper uses semantic names. The exact probability thresholds are:

| Main-paper name | Threshold pair | Frame threshold \(\tau_f\) | ROI threshold \(\tau_r\) | Intended use |
|---|---|---:|---:|---|
| CAFRU-Accuracy | \(\tau_f=0.50,\tau_r=0.05\) | 0.50 | 0.05 | Accuracy-priority |
| CAFRU-Balanced | \(\tau_f=0.60,\tau_r=0.08\) | 0.60 | 0.08 | Default balance |
| CAFRU-Efficient | \(\tau_f=0.70,\tau_r=0.10\) | 0.70 | 0.10 | Weak-network/low-compute |
| CAFRU-Bandwidth | \(\tau_f=0.80,\tau_r=0.10\) | 0.80 | 0.10 | Severe uplink constraint |

These labels are operating points of one trained uploader. They are not model
versions. Thresholds were fixed before final result synthesis; future studies
should select them on an independent calibration set.

### Table S1 Complete principal operating-point sweep at deployment confidence 0.25

This sweep disables JPEG codec simulation. The deployment codec and latency
results are reported separately in Table S7.

| Method | Precision | Recall | mAP50 | mAP50-95 | Upload ratio | KB/image | GFLOPs/image | Codec-disabled E2E@10Mbps (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edge-only YOLO26s | 0.7147 | 0.5621 | 0.6672 | 0.5350 | 0.0000 | 0.00 | 22.8 | 18.2 |
| Rule aggressive | 0.7318 | 0.5832 | 0.6817 | 0.5486 | 0.6406 | 22.94 | 168.1 | 84.1 |
| Earlier probability-only uploader | 0.7622 | 0.6108 | 0.7071 | 0.5744 | 0.6686 | 34.77 | 112.5 | 97.0 |
| CAFRU-Accuracy | 0.7835 | 0.6202 | 0.7251 | 0.5918 | 0.5656 | 34.90 | 98.7 | 85.4 |
| CAFRU-Balanced | 0.7758 | 0.6092 | 0.7163 | 0.5829 | 0.4248 | 26.55 | 79.8 | 71.7 |
| CAFRU-Efficient | 0.7671 | 0.5919 | 0.7032 | 0.5704 | 0.2442 | 15.87 | 55.6 | 49.9 |
| CAFRU-Bandwidth | 0.7519 | 0.5748 | 0.6861 | 0.5518 | 0.0920 | 6.28 | 35.2 | 32.1 |

## S3 Full ablations

### Table S2 Payload-cost weight

| Setting | \(\lambda_b\) | mAP50-95 | Upload ratio | KB/image | GFLOPs/image | E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|
| No cost | 0.00 | 0.5856 | 0.5184 | 31.44 | 92.3 | 99.6 |
| Low cost | 0.01 | 0.5832 | 0.4324 | 26.79 | 80.8 | 84.8 |
| Balanced | 0.02 | 0.5830 | 0.4246 | 26.55 | 79.7 | 83.2 |
| High cost | 0.05 | 0.5822 | 0.4152 | 25.62 | 78.5 | 81.6 |

The 0.02 setting removes most low-value requests found by the no-cost policy.
The change from 0.02 to 0.05 yields limited additional bandwidth reduction,
which supports 0.02 as the balanced value.

### Table S3 Ordinary ROI baselines

| Setting | mAP50-95 | Upload tasks/image | Upload-image ratio | ROI uploads | KB/image | GFLOPs/image | E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROI-rule strict | 0.5351 | 0.3824 | 0.3824 | 1,912 | 1.50 | 74.1 | 62.7 |
| ROI-rule moderate | 0.5309 | 0.5950 | 0.5950 | 2,975 | 3.82 | 102.6 | 82.6 |
| ROI-rule aggressive | 0.5231 | 1.1078 | 0.6480 | 5,539 | 7.47 | 171.3 | 90.8 |
| CAFRU-Balanced | 0.5830 | 0.4246 | 0.4246 | 4 | 26.55 | 79.7 | 83.2 |

Uploading more rule-selected ROIs does not monotonically improve accuracy.
Aggressive ROI review increases duplicate/local false detections and cloud
compute while remaining unable to recover edge misses without proposals.

### Table S4 Cloud input size

| Cloud input | Precision | Recall | mAP50 | mAP50-95 | Cloud-only official-path mAP50-95 | Upload ratio | GFLOPs/image | Cloud mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.7758 | 0.6092 | 0.7163 | 0.5830 | 0.5501 | 0.4246 | 79.7 | 28.0 |
| 640 | 0.7602 | 0.6294 | 0.7224 | 0.5880 | 0.5656 | 0.4246 | 111.8 | 29.1 |

The cloud-only reference column is retained from the original low-confidence
evaluation path and is not subtracted from the deployment-output CAFRU column.
Table S10 reports the explicit metric-fairness re-evaluation.

### Table S5 Frame/ROI branch ablation

| Method | mAP50-95 | Upload ratio | ROI/image | Full/image | KB/image | GFLOPs/image |
|---|---:|---:|---:|---:|---:|---:|
| Edge-only | 0.5350 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 22.8 |
| ROI-only | 0.5406 | 0.0020 | 0.0020 | 0.0000 | 0.03 | 23.1 |
| Frame-only | 0.5829 | 0.4238 | 0.0000 | 0.4238 | 26.53 | 79.6 |
| Frame+ROI | 0.5830 | 0.4246 | 0.0008 | 0.4238 | 26.55 | 79.7 |

This ablation is retained in full because it constrains the novelty claim:
frame-level gating is currently responsible for nearly all balanced-point
accuracy, while the ROI branch is an extensibility mechanism.

### Table S6 JPEG quality

| JPEG quality | mAP50-95 | Upload ratio | ROI uploads | Full uploads | Mean payload/upload (KB) | KB/image | E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q50 | 0.5662 | 0.4312 | 37 | 2,119 | 34.60 | 14.92 | 74.4 |
| Q60 | 0.5717 | 0.4286 | 24 | 2,119 | 40.16 | 17.21 | 76.5 |
| Q70 | 0.5744 | 0.4270 | 16 | 2,119 | 47.86 | 20.44 | 79.0 |
| Q80 | 0.5765 | 0.4246 | 4 | 2,119 | 62.53 | 26.55 | 83.6 |
| Q90 | 0.5799 | 0.4238 | 0 | 2,119 | 89.03 | 37.73 | 92.5 |

## S4 Network and queue details

### Table S7 Weak-network profiles at 10 Mbps/50 ms RTT

| Method | mAP50-95 | Upload ratio | KB/image | Mean E2E | P95 E2E | >100 ms | >200 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edge-only | 0.5350 | 0.0000 | 0.00 | 29.5 | 29.5 | 0.0000 | 0.0000 |
| Cloud-only@512 Q80 | 0.5996 | 1.0000 | 62.52 | 128.3 | 128.3 | 1.0000 | 0.0000 |
| ROI-rule strict | 0.5351 | 0.3824 | 1.50 | 62.7 | 122.6 | 0.3818 | 0.0000 |
| CAFRU Q70 | 0.5744 | 0.4270 | 20.44 | 79.0 | 162.5 | 0.4270 | 0.0004 |
| CAFRU Q80 | 0.5765 | 0.4246 | 26.55 | 83.6 | 179.0 | 0.4246 | 0.0080 |
| CAFRU Q90 | 0.5799 | 0.4238 | 37.73 | 92.5 | 207.3 | 0.4238 | 0.0712 |

### Table S8 Legacy queue simulation at 15 FPS and 200-ms deadline

| Method | Queue | Policy | Original mAP | RT mAP est. | Timely/image | Stale/image | Drop/candidate | P95 cloud E2E |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CAFRU Q70 | 0 | drop-new | 0.5744 | 0.5420 | 0.2902 | 0.0002 | 0.3199 | 173.2 |
| CAFRU Q70 | 1 | keep-latest | 0.5744 | 0.5456 | 0.3054 | 0.0860 | 0.0834 | 240.7 |
| CAFRU Q80 | 0 | drop-new | 0.5765 | 0.5373 | 0.2632 | 0.0060 | 0.3660 | 189.8 |
| CAFRU Q80 | 1 | keep-latest | 0.5765 | 0.5331 | 0.2460 | 0.1260 | 0.1239 | 265.8 |
| Cloud-only Q80 | 0 | drop-new | 0.5501 | 0.2751 | 0.5000 | 0.0000 | 0.5000 | 128.3 |
| Cloud-only Q80 | 1 | keep-latest | 0.5501 | 0.2860 | 0.5198 | 0.0002 | 0.4800 | 191.6 |

The real-time mAP estimate discounts cloud results that miss the deadline and
falls back to the edge result. It is a simulation-derived system metric from
the original low-confidence evaluation path, not a COCO standard metric and
not part of the primary accuracy claim. The main paper therefore interprets
this experiment through directly observed drop, stale-return, and P95-latency
statistics.

## S5 Difficulty-oriented analysis and fairness

### Table S9 Difficulty-oriented subsets under the original low-confidence metric path

| Subset | Images | Objects/image | Upload ratio | Edge mAP | Cloud mAP | Fused mAP | Gain vs edge | Gain vs cloud | KB/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crowded objects | 1,332 | 17.19 | 0.8934 | 0.3898 | 0.4696 | 0.5116 | 0.1218 | 0.0420 | 58.67 |
| Edge-touch objects | 3,780 | 8.47 | 0.4995 | 0.4648 | 0.5439 | 0.5720 | 0.1072 | 0.0281 | 31.34 |
| High complexity | 1,250 | 8.46 | 0.4592 | 0.4461 | 0.5188 | 0.5374 | 0.0913 | 0.0187 | 38.81 |
| CAFRU-uploaded Q80 | 2,123 | 12.38 | 1.0000 | 0.4169 | 0.5027 | 0.5447 | 0.1278 | 0.0420 | 62.53 |
| Multi-scale objects | 2,874 | 10.63 | 0.6180 | 0.4336 | 0.5111 | 0.5467 | 0.1130 | 0.0355 | 39.21 |
| Small objects | 2,160 | 12.14 | 0.6829 | 0.4023 | 0.4792 | 0.5149 | 0.1126 | 0.0357 | 43.40 |
| High-coverage objects | 1,238 | 8.25 | 0.4952 | 0.4630 | 0.5716 | 0.5639 | 0.1009 | -0.0077 | 30.29 |
| CAFRU-not-uploaded Q80 | 2,877 | 3.49 | 0.0000 | 0.5536 | 0.6274 | 0.6220 | 0.0684 | -0.0054 | 0.00 |
| Tiny objects | 1,204 | 14.59 | 0.7409 | 0.3673 | 0.4415 | 0.4834 | 0.1161 | 0.0419 | 47.68 |

The negative cloud gains on high-coverage and non-uploaded subsets reinforce
that cloud-only is not an oracle. The uploader can benefit from retaining edge
output on images where the cloud model is not better. These values are retained
for subset characterization; primary same-threshold interpretations use the
deployment columns in Table S10.

### Table S10 Metric-fairness check

| Subset | Upload | Edge official | Edge deploy | Cloud official | Cloud deploy | Fused deploy | Fused-edge official | Fused-edge deploy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 0.4246 | 0.4733 | 0.5350 | 0.5501 | 0.5996 | 0.5765 | 0.1033 | 0.0415 |
| Tiny objects | 0.7409 | 0.3673 | 0.4413 | 0.4415 | 0.5050 | 0.4834 | 0.1161 | 0.0421 |
| High-coverage objects | 0.4952 | 0.4630 | 0.5124 | 0.5716 | 0.6134 | 0.5639 | 0.1009 | 0.0515 |
| CAFRU-uploaded Q80 | 1.0000 | 0.4169 | 0.4756 | 0.5027 | 0.5536 | 0.5447 | 0.1278 | 0.0691 |
| CAFRU-not-uploaded Q80 | 0.0000 | 0.5536 | 0.6192 | 0.6274 | 0.6665 | 0.6220 | 0.0684 | 0.0028 |

The `official` and `deploy` columns use different confidence/evaluation paths.
Only within-column differences are interpreted.

## S6 T41 prototype details

### S6.1 Measured JPEG and selective-link costs

At JPEG Q70 and width 640, the measured payload is 40.053 KB/upload,
board encoding is 82.069 ms, HTTP round trip is 1503.373 ms, and reported cloud
inference is 1123.246 ms. In a 100-sample rule-selective run:

| Quantity | Result |
|---|---:|
| Samples | 100 |
| Uploaded samples | 37 |
| Successful requests | 37/37 |
| Upload ratio | 0.370 |
| Payload per upload | 22.870 KB |
| Payload per input frame | 8.462 KB |
| Mean edge inference | 616.395 ms |
| Mean HTTP round trip | 1532.018 ms |
| Mean cloud inference | 1114.861 ms |

Decision reasons were `edge_confident:63`, `keyframe:17`, `low_conf:11`,
`edge_failed:6`, and `small_object:3`. These are rule-uploader diagnostics,
not CAFRU feature attributions.

### S6.2 NNA stability

| Model | Successful attempts | Failed attempts | Mean inference, success-only | P95 | Mean postprocess |
|---|---:|---:|---:|---:|---:|
| YOLO26s | 4 | 6 | 585.971 ms | 701.825 ms | 218.331 ms |
| YOLO26n | 5 | 5 | 228.497 ms | 373.451 ms | available in raw log |

Failures are intermittent `SIGILL` events near the NNA runtime call. The
success-only latency must not be reported as an unconditional throughput
measurement.

### S6.3 Board-side linear evaluator

| Run | Samples | Suggested upload | Actual upload | Edge success | Mean policy | P95 policy | Mean HTTP |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO static probe | 20 | 14/20 | 20/20 | 90% | 0.649 ms | - | 1678.218 ms |
| Real-camera shadow | 100 | 36/100 | 100/100 | not applicable | 4.679 ms | - | 1520.671 ms |
| Real-camera online | 50 | 11/50 | 11/50 | 98% | 0.676 ms | 0.817 ms | 1548.606 ms |

Shadow mode logs suggested decisions but does not control upload; actual upload
therefore remains 100%. Online mode controls requests. The real-camera online
upload ratio is 22%, but this run is not an exact replication of
CAFRU-Balanced because the current C++ feature extractor differs from Python
for several features.

### S6.4 Python/C++ feature-equivalence gap

The following differences were found during the manuscript audit:

| Feature | Python training extractor | Current T41 prototype |
|---|---|---|
| Image complexity | pixel edge density | box-derived complexity proxy |
| Detection count | `min(1, N/20)` | raw `N` |
| Relative confidence | confidence minus frame mean | confidence divided by maximum |

Before claiming exact uploader portability, a deterministic test should feed
the same image and `edge_boxes.csv` to both implementations, export all 30
features, and require a per-feature absolute tolerance. The board run remains
valid as proof that four linear heads can execute with sub-millisecond overhead.

### S6.5 Measured deadline queue

These scheduling-isolation runs use a transport-only mode: local YOLO26s
loading and inference are bypassed. The reported FPS is therefore capture/JPEG/HTTP/
queue-loop throughput, not full detector throughput. A returned result can
remain timely for multiple subsequent frames, so timely-result frames may
exceed completed requests.

| Run | Queue | Deadline | Frames | Scheduling-loop FPS | Upload success | Dropped | Mean age | Timely-result frames | Effective FPS@D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sync every-frame | 1 | 1000 ms | 19 | 0.614 | 19/19 | 0 | blocking | 0 | 0.000 |
| Async | 1 | 1000 ms | 100 | 6.623 | 11/11 | 89 | 503.375 ms | 62 | 4.106 |
| Async | 2 | 500 ms | 100 | 6.828 | 12/12 | 88 | 245.647 ms | 31 | 2.117 |
| Async | 2 | 1000 ms | 100 | 6.613 | 12/12 | 88 | 510.449 ms | 62 | 4.100 |
| Async | 2 | 2000 ms | 100 | 6.663 | 12/12 | 88 | 699.024 ms | 65 | 4.331 |
| Async | 4 | 1000 ms | 100 | 6.751 | 15/15 | 85 | 471.275 ms | 63 | 4.253 |

### S6.6 SpeciesNet-assisted pseudo-reference check

SpeciesNet 4.0.2a was run offline on 100 T41 JPEG frames. Only automatically
generated human/person boxes with confidence at least 0.25 were mapped to COCO
class `person`, yielding 108 pseudo-reference boxes. These boxes were not
manually verified.

| Prediction source | Frames | Pseudo-reference boxes | TP | FP | FN | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Edge | 100 | 108 | 44 | 58 | 64 | 0.4314 | 0.4074 | 0.4190 | 0.2973 |
| Cloud | 100 | 108 | 75 | 39 | 33 | 0.6579 | 0.6944 | 0.6757 | 0.4882 |
| Rule-loop Final | 100 | 108 | 75 | 40 | 33 | 0.6522 | 0.6944 | 0.6726 | 0.4835 |

This table is a sanity check against automated reference annotations. It is
not a formal real-scene ground-truth evaluation. Rule-loop Final is close to
Cloud because the source run effectively uploads all evaluated frames.

## S7 Artifact-to-paper map

| Artifact label | Paper role |
|---|---|
| COCO main operating-point results | Main operating points |
| Payload-cost ablation results | Payload-cost ablation |
| Ordinary ROI comparison results | Ordinary ROI comparison |
| Cloud input-size ablation results | Cloud input size |
| Frame-vs-ROI branch ablation results | Branch contribution |
| JPEG quality ablation results | Codec quality |
| Network-condition simulation results | Network simulation |
| Queue-scheduling simulation results | Queue simulation |
| Difficulty-oriented subset results | Difficulty-oriented subsets |
| Qualitative case pool | Qualitative examples |
| Uploaded/non-uploaded selectivity controls | Selection behavior |
| Evaluation-path fairness check | Metric fairness |
| Consolidated paper tables | Result synthesis |
| T41 JPEG/cloud-path measurements | Measured JPEG/cloud path |
| T41 asynchronous transport measurements | Asynchronous transport |
| T41 YOLO26s NNA stability measurements | NNA runtime stability |
| T41 rule-selective closed-loop run | Real rule-selective closed loop |
| T41 board-policy, deadline, and pseudo-reference run | Board policy, deadline, pseudo-reference check |

The raw archive retains local folder names for reproducibility, but the main
manuscript reports only paper-facing method and artifact names.

## S8 Artifact provenance

Authoritative paper-facing files are listed in `CONTENT_AUDIT.md`. Raw logs,
scripts, and derived tables are retained under `paper/experiments`. Before
submission, the authors should create an anonymized archive containing:

1. policy JSON and feature schema;
2. COCO result tables and generation scripts;
3. network/queue simulator configuration;
4. de-identified T41 logs;
5. a script that verifies Python/C++ feature equivalence;
6. manual-annotation status for any real-camera accuracy claim.
