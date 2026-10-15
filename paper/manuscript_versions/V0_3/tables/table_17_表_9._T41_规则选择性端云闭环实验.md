# 表 9. T41 规则选择性端云闭环实验。

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_3_ZH.md`, lines 638-642

| Setting | Samples | Uploads | Upload ratio | Upload success | Payload/upload | Payload/frame | Edge infer | Encode | HTTP | Cloud infer | Decision reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cloud-only JPEG q70 | 20 | 20 | 1.000 | 20/20 | 39.378 KB | 39.378 KB | - | 83.826 ms | 1615.952 ms | 1078.786 ms | all upload |
| Rule fixed v2 | 20 | 20 | 1.000 | 20/20 | 22.199 KB | 22.199 KB | 636.450 ms | 77.722 ms | 1426.343 ms | 1083.800 ms | edge_touch/low_conf/small_object |
| Rule selective | 100 | 37 | 0.370 | 37/37 | 22.870 KB | 8.462 KB | 616.395 ms | 73.407 ms | 1532.018 ms | 1114.861 ms | edge_confident/keyframe/low_conf/edge_failed/small_object |
