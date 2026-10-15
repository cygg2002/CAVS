# Table 11

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_4_ZH.md`, lines 544-551

| Method | Queue | Policy | Orig mAP | RT mAP est. | Timely/img | Stale/img | Drop/cand | P95 cloud E2E(ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned v2 Q70 | 0 | drop_new | 0.5744 | 0.5420 | 0.2902 | 0.0002 | 0.3199 | 173.2 |
| Learned v2 Q70 | 1 | keep_latest | 0.5744 | 0.5456 | 0.3054 | 0.0860 | 0.0834 | 240.7 |
| Learned v2 Q80 | 0 | drop_new | 0.5765 | 0.5373 | 0.2632 | 0.0060 | 0.3660 | 189.8 |
| Learned v2 Q80 | 1 | keep_latest | 0.5765 | 0.5331 | 0.2460 | 0.1260 | 0.1239 | 265.8 |
| Cloud-only YOLO26x@512 Q80 | 0 | drop_new | 0.5501 | 0.2751 | 0.5000 | 0.0000 | 0.5000 | 128.3 |
| Cloud-only YOLO26x@512 Q80 | 1 | keep_latest | 0.5501 | 0.2860 | 0.5198 | 0.0002 | 0.4800 | 191.6 |
