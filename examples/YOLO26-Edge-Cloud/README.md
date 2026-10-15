# YOLO26 Edge-Cloud Detection Demo

This demo implements a first runnable edge-cloud collaborative detection system:

- Edge model: `yolo26n.pt` (the exact hash-pinned detector used to train Current+MAUR)
- Cloud model: `yolo26x.pt`
- Transport: local HTTP, JSON metadata, base64 JPEG payloads
- Fusion: asynchronous cloud correction logs

The design borrows the selective-offloading idea from EdgeBoost but adapts it to
object detection with detection-aware uncertainty, ROI/keyframe upload, and
missed-detection probes.

## Run

For the packaged local-edge/remote-cloud demo, copy the environment template and use the launch scripts:

```bash
cp examples/YOLO26-Edge-Cloud/demo.env.example examples/YOLO26-Edge-Cloud/demo.env
bash examples/YOLO26-Edge-Cloud/start_cloud.sh      # remote server
bash examples/YOLO26-Edge-Cloud/start_edge_demo.sh  # local edge machine
```

The safe default binds the cloud verifier to `127.0.0.1`; expose it to the local machine with an SSH tunnel. See
`EDGE_CLOUD_OPERATION_GUIDE_ZH.md` for the full setup and `DEMO_SYSTEM_SPEC_ZH.md` for acceptance criteria.

### Paper-aligned budget modes

The Gradio demo now separates byte budget from frame upload ratio:

- `Paper Fixed-byte` for uploaded video performs two passes. The first pass predicts all
  Edge/Original/Q90/Current@640 values and exact JPEG costs, then freezes one action per frame with the
  paper scheduler. The second pass performs cloud requests and MAUR fusion. Choose 10%, 25%, or 40%.
- Camera mode uses a strict online prefix budget. At every processed frame, cumulative raw JPEG request bytes
  remain below `budget * cumulative Q95 Original-equivalent bytes`. This is deployable but is not equivalent to
  the offline global optimum used by the paper.
- `Online per-frame` video and a disabled camera budget are legacy lambda modes. They are useful for latency
  demonstrations, but their upload bytes are not a paper Fixed-byte result.

The budget counts raw JPEG payload bytes before Base64 and HTTP framing. The UI reports both byte utilization
and uploaded-frame ratio because a 10% byte budget does not imply that only 10% of frames are uploaded. The
launcher verifies the hash-pinned edge model, the three policy checkpoints, the cloud checkpoint SHA256, and
the cloud detector threshold before collaboration starts.

Use the `yolo26` conda environment from the repository root.

```powershell
conda activate yolo26
cd F:\IDE\ultralytics-main
```

Start the cloud verifier:

```powershell
python examples\YOLO26-Edge-Cloud\cloud_server.py --model yolo26x.pt --host 127.0.0.1 --port 8765
```

In another terminal, run the edge client:

```powershell
python examples\YOLO26-Edge-Cloud\edge_client.py --model yolo26s.pt --source F:\IDE\codex\yoloe-new\datasets\bear.mp4 --cloud-url http://127.0.0.1:8765/infer --max-frames 300
```

For a formal webcam real-time demo on the local machine, first start the cloud server on the remote host:

```bash
conda run -n yolo26 python examples/YOLO26-Edge-Cloud/cloud_server.py --model yolo26x.pt --host 0.0.0.0 --port 8765 --device 0
```

Then run the local camera demo and switch modes with `1/2/3`:

```powershell
python examples\YOLO26-Edge-Cloud\webcam_demo.py --edge-model yolo26s.pt --source 0 --cloud-url http://<server-ip>:8765/infer --learned-policy-model runs\edge_cloud\learned_policy_coco_full_v2_20260424_1225\learned_upload_policy.json --mode collab --learned-frame-threshold 0.60 --learned-roi-threshold 0.08 --learned-cost-weight 0.02 --enable-temporal-memory --enable-adaptive-roi
```

For a browser UI with webcam, image upload, mode switching, and on-frame latency overlays:

```powershell
python examples\YOLO26-Edge-Cloud\gradio_demo.py --edge-model yolo26n.pt --cloud-url http://127.0.0.1:8765/infer --policy-path examples\YOLO26-Edge-Cloud\policies\current_maur_v2\manifest.json --learned-cost-weight 0.02 --cloud-imgsz 640 --classes person --server-port 7861 --inbrowser
```

If the cloud model runs on the remote server, keep the UI local and either point `--cloud-url` to the reachable remote address or create an SSH tunnel first:

```powershell
ssh -L 8765:127.0.0.1:8765 -p 10022 111_wcy@10.193.114.121
```

For automation without activating the shell:

```powershell
conda run -n yolo26 python examples\YOLO26-Edge-Cloud\edge_client.py --max-frames 30
```

The Gradio UI supports camera, image upload, and video upload. Use the `Classes`
selector in the Runtime panel to choose `person` or any subset of COCO classes.
Class filtering reduces NMS/display/fusion overhead and cloud response size, but
it does not remove the main YOLO backbone computation; for a large compute
reduction use a smaller `cloud_imgsz` or a task-specific detector.

## Outputs

Default output directory:

```text
runs/edge_cloud/bear/
```

Files:

- `events.jsonl`: one cloud response per line with request metadata, cloud detections, latency, and fusion results.
- `summary.json`: aggregate frame count, upload ratio, payload size, edge/cloud latency, and correction counts.
- `preview.mp4`: optional edge preview when `--save-video` is enabled.

## Upload strategies

The current implementation has these offloading and fusion policies:

1. Detection uncertainty ROI: upload low-confidence, small, edge-area, or temporally unstable boxes as cropped ROIs.
2. Complexity full-frame probe: upload the full frame when Canny edge density is high but edge detections are sparse or cover little image area. This targets missed detections.
3. Risk-aware ROI fusion: separate thresholds for cloud verification and unmatched cloud box insertion, so ROI crops can correct edge boxes without freely adding weak extra boxes.
4. Adaptive ROI context: expand small and edge-near boxes more than ordinary boxes; this gives the cloud model enough context while avoiding the large false-positive increase seen with aggressive expansion.
5. Scale-stability probe: optionally run a second edge pass at a smaller input size and upload frames whose boxes are unstable across scales.
6. Suppressed low-confidence proposal probe and uncovered tile probe are implemented for experiments, but they are not recommended defaults on COCO128 because they did not improve mAP.
7. Learned upload policy v2: train a small dual-head policy from labeled data. One head predicts upload utility probability, the other predicts normalized utility gain; online scoring then uses `probability * expected_gain - lambda * payload_cost`. This keeps inference lightweight while making the decision closer to the real optimization objective.
8. Learned adaptive cloud resolution: for learned uploads, use the learned utility probability to choose a lower or higher YOLO26x input size. This keeps high-risk samples at higher cloud compute and sends marginal samples to a cheaper cloud pass.
9. Temporal memory upload suppression: for video or webcam streams, a lightweight IoU track cache marks recently cloud-verified stable targets and suppresses repeated ROI or full-frame uploads until confidence drops, the object drifts, or enough frames have passed.

For still-image datasets, run offline evaluation instead of the async demo:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --upload-threshold 0.35 --max-roi-per-image 1 --keyframe-interval 30 --save-dir runs\edge_cloud\strategy_baseline_best
```

Recommended balanced strategy on COCO128:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --upload-threshold 0.35 --max-roi-per-image 1 --keyframe-interval 30 --enable-adaptive-roi --roi-small-gain 1.6 --roi-edge-gain 1.45 --roi-large-gain 1.15 --roi-extra-cloud-conf 0.55 --roi-veto-edge-conf 0.25 --enable-complexity-probe --complexity-threshold 0.14 --empty-complexity-threshold 0.06 --complexity-max-detections 2 --complexity-coverage-threshold 0.04 --save-dir runs\edge_cloud\strategy_adaptive_roi_mild_risk_complex
```

Higher-recall strategy on COCO128:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --upload-threshold 0.35 --max-roi-per-image 1 --keyframe-interval 30 --enable-adaptive-roi --roi-small-gain 1.6 --roi-edge-gain 1.45 --roi-large-gain 1.15 --roi-extra-cloud-conf 0.55 --roi-veto-edge-conf 0.25 --enable-complexity-probe --complexity-threshold 0.10 --empty-complexity-threshold 0.05 --complexity-max-detections 3 --complexity-coverage-threshold 0.06 --save-dir runs\edge_cloud\strategy_adaptive_roi_risk_complex_aggressive
```

Train a learned upload policy:

```powershell
python examples\YOLO26-Edge-Cloud\train_learned_uploader.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --enable-adaptive-roi --roi-small-gain 1.6 --roi-edge-gain 1.45 --roi-large-gain 1.15 --roi-extra-cloud-conf 0.55 --roi-veto-edge-conf 0.25 --max-roi-candidates 2 --save-dir runs\edge_cloud\learned_policy_v1
```

The current trainer fits a probability head and a value head with train/validation splitting and feature-noise regularization:

```powershell
python examples\YOLO26-Edge-Cloud\train_learned_uploader.py --source F:\IDE\datasets\coco\train2017.txt --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --half --edge-batch-size 8 --cloud-batch-size 8 --roi-batch-size 16 --enable-adaptive-roi --feature-noise-std 0.02 --gain-normalizer labels --save-dir runs\edge_cloud\learned_policy_v2
```

Evaluate a learned policy with a balanced threshold:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --enable-learned-policy --learned-policy-model runs\edge_cloud\learned_policy_v1\learned_upload_policy.json --learned-frame-threshold 0.60 --learned-roi-threshold 0.65 --learned-cost-weight 0.02 --learned-max-roi-per-image 1 --enable-adaptive-roi --roi-small-gain 1.6 --roi-edge-gain 1.45 --roi-large-gain 1.15 --roi-extra-cloud-conf 0.55 --roi-veto-edge-conf 0.25 --save-dir runs\edge_cloud\strategy_learned_policy_thr060
```

Recommended learned strategy with adaptive cloud compute:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\trae\datasets\coco128 --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --cloud-imgsz 512 --enable-learned-policy --learned-policy-model runs\edge_cloud\learned_policy_v1\learned_upload_policy.json --learned-frame-threshold 0.60 --learned-roi-threshold 0.65 --learned-cost-weight 0.02 --learned-max-roi-per-image 1 --enable-learned-adaptive-cloud-imgsz --learned-cloud-imgsz-low 416 --learned-cloud-imgsz-high 512 --learned-cloud-imgsz-switch 0.75 --enable-adaptive-roi --roi-small-gain 1.6 --roi-edge-gain 1.45 --roi-large-gain 1.15 --roi-extra-cloud-conf 0.55 --roi-veto-edge-conf 0.25 --save-dir runs\edge_cloud\strategy_learned_thr060_adaptive_cloud416_512_switch075
```

Evaluate a sanity upper bound with edge + cloud full-frame fusion on every image:

```powershell
python examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py --source F:\IDE\datasets\coco\val2017.txt --edge-model yolo26s.pt --cloud-model yolo26x.pt --device 0 --conf 0.25 --cloud-conf 0.25 --metric-conf 0.001 --metric-cloud-conf 0.001 --eval-cloud-only --eval-full-frame-fusion-upper --save-dir runs\edge_cloud\full_frame_fusion_upper
```

Use the learned policy in the online edge client:

```powershell
python examples\YOLO26-Edge-Cloud\edge_client.py --model yolo26s.pt --source F:\IDE\datasets\coco\val2017.txt --cloud-url http://127.0.0.1:8765/infer --enable-learned-policy --learned-policy-model runs\edge_cloud\learned_policy_coco_full_v2_20260424_1225\learned_upload_policy.json --learned-frame-threshold 0.60 --learned-roi-threshold 0.08 --learned-cost-weight 0.02 --cloud-imgsz 512 --enable-adaptive-roi
```

Summarize strategy runs:

```powershell
python examples\YOLO26-Edge-Cloud\summarize_strategy_results.py runs\edge_cloud\strategy_baseline_best runs\edge_cloud\strategy_complexity_probe_relaxed --save-dir runs\edge_cloud\strategy_comparison
```

Analyze bandwidth, latency, parameter, and FLOPs costs:

```powershell
python examples\YOLO26-Edge-Cloud\analyze_edge_cloud_costs.py --summary-csv runs\edge_cloud\strategy_comparison_learned\strategy_summary.csv --save-dir runs\edge_cloud\cost_analysis_learned --edge-model yolo26s.pt --cloud-model yolo26x.pt --learned-policy runs\edge_cloud\learned_policy_v1\learned_upload_policy.json
```

## COCO full v2 validation

The COCO full run uses train2017 to train the upload policy and val2017 for
evaluation. COCO segmentation labels are converted to bounding boxes during
evaluation, which fixes the earlier incorrect low mAP numbers.

Reference detector baselines on COCO val2017:

| Mode | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Edge-only YOLO26s | 0.6913 | 0.5781 | 0.6346 | 0.4733 |
| Cloud-only YOLO26x | 0.7383 | 0.6700 | 0.7327 | 0.5656 |
| Edge + full-frame cloud fusion upper | 0.9993 | 0.9399 | 0.9695 | 0.8008 |

Selected COCO full v2 working points at 10 Mbps uplink and 50 ms RTT:

| Strategy | Precision | Recall | mAP50 | mAP50-95 | Uploaded images | KB/image | GFLOPs/image | E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule aggressive | 0.7318 | 0.5832 | 0.6817 | 0.5486 | 0.6406 | 22.94 | 168.1 | 84.1 ms |
| Learned v1 f0.60/r0.08 | 0.7622 | 0.6108 | 0.7071 | 0.5744 | 0.6686 | 34.77 | 112.5 | 97.0 ms |
| Learned v2 f0.50/r0.05 | 0.7835 | 0.6202 | 0.7251 | 0.5918 | 0.5656 | 34.90 | 98.7 | 85.4 ms |
| Learned v2 f0.60/r0.08 | 0.7758 | 0.6092 | 0.7163 | 0.5829 | 0.4248 | 26.55 | 79.8 | 71.7 ms |
| Learned v2 f0.70/r0.10 | 0.7671 | 0.5919 | 0.7032 | 0.5704 | 0.2442 | 15.87 | 55.6 | 49.9 ms |
| Learned v2 f0.80/r0.10 | 0.7519 | 0.5748 | 0.6861 | 0.5518 | 0.0920 | 6.28 | 35.2 | 32.1 ms |

Default online demo setting:

```text
policy = runs/edge_cloud/learned_policy_coco_full_v2_20260424_1225/learned_upload_policy.json
cloud_imgsz = 512
learned_frame_threshold = 0.60
learned_roi_threshold = 0.08
learned_cost_weight = 0.02
```

Use `f0.50/r0.05` for accuracy-first runs and `f0.70/r0.10` for weak uplink
or low-latency runs.

COCO128 observations from this workspace:

| Strategy | Upload tasks | Uploaded images | Payload MB | Recall | Precision | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge-only YOLO26s, same deployment threshold | 0 | 0.0000 | 0.000 | 0.6100 | 0.7535 | 0.7196 | 0.5995 |
| Cloud-only YOLO26x full image upper bound | 128 | 1.0000 | full stream | 0.7262 | 0.7875 | 0.7939 | 0.6689 |
| ROI + keyframe baseline | 52 | 0.3828 | 0.450 | 0.6225 | 0.7877 | 0.7359 | 0.6148 |
| Adaptive ROI + risk fusion + aggressive complexity | 91 | 0.6250 | 3.345 | 0.6438 | 0.7730 | 0.7457 | 0.6250 |
| Learned v1, threshold 0.70, cloud 512 | 34 | 0.2656 | 1.921 | 0.6519 | 0.8076 | 0.7532 | 0.6332 |
| Learned v1, threshold 0.60, cloud 640 | 55 | 0.4297 | 3.111 | 0.6967 | 0.7943 | 0.7750 | 0.6478 |
| Learned v1, threshold 0.60, cloud 512 | 55 | 0.4297 | 3.111 | 0.6813 | 0.7877 | 0.7693 | 0.6514 |
| Learned v1, threshold 0.60, adaptive cloud 416/512, switch 0.75 | 55 | 0.4297 | 3.111 | 0.6765 | 0.8101 | 0.7717 | 0.6555 |
| Learned v1, threshold 0.60, adaptive cloud 416/512, switch 0.80 | 55 | 0.4297 | 3.111 | 0.6771 | 0.8055 | 0.7721 | 0.6538 |
| Learned v1, threshold 0.60, cloud 512, JPEG q60 simulated | 55 | 0.4297 | 2.609 | 0.6700 | 0.8083 | 0.7680 | 0.6407 |
| Learned v2 cost-sensitive, threshold 0.45, cloud 512 | 36 | 0.2812 | 2.082 | 0.6509 | 0.7744 | 0.7465 | 0.6325 |

On COCO128, the best rule-based default is adaptive ROI + risk-aware fusion + aggressive complexity probing. The learned policy is stronger because it learns which full frames deserve cloud review and avoids many low-value ROI uploads. The current best prototype is learned v1 with adaptive cloud `416/512` and switch `0.75`: mAP50-95 improves from edge-only `0.5995` to `0.6555`, closing about `80.0%` of the gap to full-image YOLO26x cloud-only `0.6689` while uploading only `42.97%` of images. The policy was trained and evaluated on COCO128 for rapid prototyping; a publishable or deployable result should train on one split and report on a separate validation/test split.

## Cost Model

The cost analysis treats selective upload as a constrained optimization problem:

```text
maximize detection quality
subject to edge latency, cloud workload, uplink bandwidth, and response latency budgets
```

For one image, the blocking-response estimate is:

```text
T = T_edge + upload_bytes/uplink + upload_tasks * RTT + upload_tasks * T_cloud
```

The asynchronous workload estimate separates local edge latency from remote cloud/network work. This matters because a deployed system may show edge results immediately and apply cloud corrections later.

The learned uploader approximates the Lagrangian form of selective offloading:

```text
score(x) = P(cloud upload improves detection | x) - lambda * estimated_payload_KB(x)
```

The rule baseline uses fixed uncertainty thresholds. The learned policy instead trains two tiny logistic heads, one for full-frame upload and one for ROI upload, from edge detections, image complexity, box geometry, and cloud/ground-truth utility labels. In this prototype the learned model adds only 26 linear coefficients, so the edge detector itself is unchanged. Adaptive cloud resolution then allocates cloud compute by learned utility: marginal samples use `416`, high-utility samples use `512`.

Local model profiles at `imgsz=640`:

| Model | Params | Weight MB | GFLOPs |
| --- | ---: | ---: | ---: |
| YOLO26s edge | 10.01M | 19.477 | 22.84 |
| YOLO26x cloud | 58.99M | 113.170 | 209.53 |
| Learned upload policy | 26 linear params | 0.003 MB JSON | negligible |

Measured decision overhead on COCO128:

| Decision logic | Avg ms/image | P95 ms/image |
| --- | ---: | ---: |
| Learned policy features + linear scoring | 1.50 | 2.44 |
| Complexity rule | 1.05 | 2.04 |
| ROI/adaptive ROI rules | 0.06 | 0.12 |

At a 10 Mbps uplink and 50 ms RTT, selected working points were:

| Strategy | mAP50-95 | KB/image | System GFLOPs/image | Est. blocking ms |
| --- | ---: | ---: | ---: | ---: |
| ROI + keyframe baseline | 0.6148 | 3.60 | 108.0 | 115.2 |
| Rule best, aggressive | 0.6250 | 26.76 | 171.8 | 155.5 |
| Learned v1, threshold 0.70, cloud 512 | 0.6332 | 15.37 | 58.5 | 80.6 |
| Learned v2 cost-sensitive, threshold 0.45, cloud 512 | 0.6325 | 16.66 | 60.5 | 69.8 |
| Learned v1, threshold 0.60, cloud 640 | 0.6478 | 24.89 | 112.9 | 105.3 |
| Learned v1, threshold 0.60, cloud 512 | 0.6514 | 24.89 | 80.5 | 104.2 |
| Learned v1, threshold 0.60, adaptive cloud 416/512, switch 0.75 | 0.6555 | 24.89 | 71.6 | 101.5 |
| Learned v1, threshold 0.60, adaptive cloud 416/512, switch 0.80 | 0.6538 | 24.89 | 67.6 | 98.0 |
| Learned v1, threshold 0.60, cloud 512, JPEG q60 simulated | 0.6407 | 20.87 | 80.5 | 100.2 |

For weak field uplink around 2 Mbps, the practical default in this run is `learned v1 threshold 0.70, cloud 512`: it reaches `0.6332` mAP50-95 with about `143.1 ms` estimated blocking latency. If the link and cloud budget can tolerate about `200 ms` blocking response, `learned v1 threshold 0.60 + adaptive cloud 416/512` is the accuracy-first setting. The cost-sensitive v2 learner is a useful negative result: it reduces uploads, but its mAP does not beat v1 threshold tuning. JPEG q60 reduces bytes but loses too much accuracy on COCO128, so it is only a bandwidth-priority fallback.

## Current v1 limits

- This is a local HTTP prototype, not a production network service.
- Confidence calibration is not trained yet; the client logs enough data for a later calibration stage.
- Without labeled video annotations, the demo reports correction candidates and system metrics, not mAP/Recall.
- ROI uploads can verify false positives and class confusion, while keyframe probes are used for missed detections.
