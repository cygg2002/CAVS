# V0.6 Content and Evidence Audit

## 1. Audit conclusion

The reviewed V0.5 Word document is an evidence archive rather than a
submission manuscript. It contains 8 figures, 19 body tables, internal
experiment identifiers, and several claims that exceed the evidence boundary.
The V0.6 package therefore uses a compact main paper plus a separate Online
Resource. The original reviewed document remains unchanged.

The scientific evidence is sufficient for the following claims:

1. On COCO val2017, a learned, cost-aware frame/ROI uploader provides
   configurable accuracy-upload-compute operating points.
2. Under the common deployment threshold and JPEG Q80, CAFRU improves
   mAP50-95 from 0.5350 to 0.5765 while uploading 42.46% of images and using
   79.7 amortized GFLOPs/image. Cloud-only@512 reaches 0.5996 at
   134.1 GFLOPs/image.
3. Cost regularization, JPEG quality, cloud input size, and bounded queues
   materially affect the accuracy-latency-bandwidth trade-off.
4. T41 component measurements validate edge inference, JPEG/HTTP transport,
   cloud review, linear-head execution, and isolated asynchronous
   deadline-aware scheduling. The 6.6-6.8 FPS scheduling runs bypass local
   YOLO26s and are not full detector throughput.
5. The four exported linear policy heads execute on T41 with sub-millisecond
   measured overhead.

The evidence does not support the following stronger claims:

1. SpeciesNet-generated boxes are manual ground truth.
2. The T41 runtime is production-stable; intermittent NNA `SIGILL` failures
   remain.
3. The 50-frame on-device uploader run proves real-scene mAP improvement.
4. Network state is a learned input feature. In the current implementation,
   payload/delay costs and queue admission are analytic runtime terms.
5. The ROI branch is the dominant source of accuracy gain. At the balanced
   setting, frame-only and frame+ROI mAP50-95 are 0.5829 and 0.5830.

## 2. Terminology normalization

| Internal term in V0.5 | Submission term |
|---|---|
| Learned v2 | Cost-Aware Frame-ROI Uploader (CAFRU) |
| f0.50/r0.05 | CAFRU-Accuracy |
| f0.60/r0.08 | CAFRU-Balanced |
| f0.70/r0.10 | CAFRU-Efficient |
| f0.80/r0.10 | CAFRU-Bandwidth |
| E18-E23 | Descriptive experiment names only |
| SpeciesNet GT | SpeciesNet-assisted pseudo-reference annotations |
| cloud strong model | cloud reviewer with higher-capacity detector |
| policy network | four lightweight linear prediction heads |

The symbols `tau_f` and `tau_r` denote frame- and ROI-level decision
thresholds. Their numerical values are disclosed in the reproducibility
supplement, not encoded into method names.

## 3. Main-paper content budget

### Figures retained in the main paper

| No. | Content | Source/status | Width |
|---|---|---|---|
| Fig. 1 | Online edge-cloud architecture and asynchronous feedback loop | Completed Chinese/English draw.io variants | two columns |
| Fig. 2 | Offline label generation, four-head training, and online CAFRU decision | Completed Chinese/English draw.io variants | two columns |
| Fig. 3 | COCO accuracy-upload-compute Pareto plot | Reuse V0.5 Fig. 5 after relabeling | one column |
| Fig. 4 | Weak-network mean/P95 latency comparison | Reuse V0.5 Fig. 6 after relabeling | one column |
| Fig. 5 | T41 bounded-queue and deadline processing path | Completed Chinese/English draw.io variants | two columns |

The difficult-subset chart, metric-fairness chart, pseudo-reference evaluation
diagram, and extended visual cases move to Online Resource 1.

### Tables retained in the main paper

| No. | Content | Rows | Purpose |
|---|---|---:|---|
| Table 1 | COCO principal comparison and four CAFRU operating points | 7 | Main accuracy-cost claim |
| Table 2 | Consolidated method ablation | 10 | Cost, ROI baseline, resolution, JPEG, and branch roles |
| Table 3 | Weak-network and simulated queue results | 6 | Real-time issue beyond model inference |
| Table 4 | T41 measured pipeline costs | 6 | Real-device feasibility and bottlenecks |
| Table 5 | T41 deadline queue isolation | 6 | Scheduling and freshness boundary |

### Tables moved to Online Resource 1

1. Complete threshold sweep.
2. Full cost-weight sweep.
3. Full ROI-rule sweep.
4. Complete JPEG Q50-Q90 sweep.
5. Complete network-condition matrix.
6. Complete queue-size/deadline matrix.
7. All difficulty-oriented subset results.
8. Metric-fairness check.
9. Detailed T41 JPEG and asynchronous pipeline repetitions.
10. NNA attempt-level stability results.
11. Rule-uploader reason counts.
12. On-device policy shadow/online logs.
13. SpeciesNet-assisted pseudo-reference evaluation protocol and per-source
    metrics.

## 4. Evidence-to-source map

| Claim | Authoritative source |
|---|---|
| COCO main operating points | `paper/experiments/E0_coco_main_results/tables/coco_main_results.csv` |
| Cost regularization | `paper/experiments/E4_cost_ablation/tables/e4_cost_ablation_summary.csv` |
| Ordinary ROI baselines | `paper/experiments/E5_roi_baseline/tables/e5_roi_baseline_summary.csv` |
| Cloud 512/640 comparison | `paper/experiments/E7_cloud_imgsz_ablation/tables/e7_cloud_imgsz_summary.csv` |
| Frame/ROI branch roles | `paper/experiments/E8_frame_roi_ablation/tables/e8_frame_roi_ablation_summary.csv` |
| JPEG quality | `paper/experiments/E9_jpeg_quality_ablation/tables/e9_jpeg_quality_summary.csv` |
| Weak-network latency | `paper/experiments/E16_paper_result_synthesis/tables/e16_table3a_network_realtime.csv` |
| Simulated queue freshness | `paper/experiments/E16_paper_result_synthesis/tables/e16_table3b_queue_realtime.csv` |
| Difficult-subset selectivity | `paper/experiments/E16_paper_result_synthesis/tables/e16_table4_difficult_selectivity.csv` |
| Metric fairness | `paper/experiments/E16_paper_result_synthesis/tables/e16_table5_metric_fairness.csv` |
| T41 policy overhead | `paper/experiments/E23_t41_real_eval_learned_policy_queue/tables/current_usb_summary/policy/e23_policy_summary.md` |
| T41 deadline queue | `paper/experiments/E23_t41_real_eval_learned_policy_queue/tables/current_usb_summary/deadline/e23_deadline_queue_summary.md` |
| Pseudo-reference check | `paper/experiments/E23_t41_real_eval_learned_policy_queue/tables/current_usb_summary/speciesnet_pseudo_real100/speciesnet_pseudo_real100_summary.md` |

## 5. Metric compatibility rules

The following numbers must not be compared without explicitly naming their
different settings:

1. The primary comparison re-evaluates edge, cloud, and collaborative
   deployment outputs at confidence 0.25. Values produced by the original
   low-confidence metric path may appear only in explicitly labeled
   supplementary legacy tables.
2. Under confidence 0.25 and JPEG Q80, edge-only, cloud-only@512, and CAFRU
   mAP50-95 are 0.5350, 0.5996, and 0.5765, respectively.
3. Edge model latency differs between workstation runs and T41 NNA runs.
4. COCO mAP uses manual COCO annotations; the T41 real100 check uses
   SpeciesNet-assisted pseudo-reference boxes for the `person` class only.
5. Simulated end-to-end latency and measured T41 HTTP latency are separate
   evidence layers and must remain in separate table blocks.
6. T41 deadline-queue FPS is capture/JPEG/HTTP/queue scheduling throughput
   with local YOLO26s bypassed; it must not be reported as full-pipeline FPS.

## 6. Newly identified implementation validity boundary

The exported JSON policy was trained with 15 frame and 15 ROI features.
However, the current T41 C++ evaluator uses approximations for several
features: its image-complexity term is box-derived rather than pixel-edge
density, its detection count is not normalized identically, and its relative
confidence definition differs from the Python extractor. Consequently:

1. the measured 0.676 ms policy overhead is valid for the board evaluator;
2. the measured 22% upload ratio is valid for that 50-frame prototype run;
3. numerical equivalence with the Python CAFRU operating point is not yet
   established;
4. the main paper reports this as an on-device execution feasibility result,
   not as a full replication of the COCO operating point.

Before a later revision claims exact policy portability, the C++ and Python
feature vectors must be checked on the same saved detections with a
per-feature tolerance test.

## 7. Submission-size decision

The main manuscript targets 5 figures and 5 tables. Internal run identifiers,
full sweeps, field schemas, and raw log descriptions are excluded from the
main text. The English PDF must be rendered in the journal's double-column
format and remain within 12 pages including references and biographies.
