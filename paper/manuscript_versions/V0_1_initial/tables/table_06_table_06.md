# Table 06

Source: `JRTIP_MANUSCRIPT_DRAFT_ZH.md`, lines 388-394

| Method | Branch | Precision | Recall | mAP50 | mAP50-95 | Upload image ratio | ROI/img | Full/img | KB/image | GFLOPs/image | E2E@10Mbps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge-only YOLO26s | edge_baseline | 0.6913 | 0.5781 | 0.6346 | 0.4733 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 22.8 | 29.5 |
| Cloud-only YOLO26x@512 | cloud_baseline | 0.7528 | 0.6448 | 0.7179 | 0.5501 | 1.0000 | 0.0000 | 1.0000 | - | 134.1 | - |
| ROI-only | roi_only | 0.7401 | 0.5619 | 0.6748 | 0.5406 | 0.0020 | 0.0020 | 0.0000 | 0.03 | 23.1 | 29.7 |
| Frame-only | frame_only | 0.7758 | 0.6092 | 0.7162 | 0.5829 | 0.4238 | 0.0000 | 0.4238 | 26.53 | 79.6 | 83.4 |
| Frame+ROI | frame_roi | 0.7758 | 0.6092 | 0.7163 | 0.5830 | 0.4246 | 0.0008 | 0.4238 | 26.55 | 79.7 | 84.2 |
