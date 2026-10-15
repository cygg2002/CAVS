# 表 7. T41 异步云端管线与队列实验。

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_4_ZH.md`, lines 622-627

| Setting | Async | Frames | FPS | Upload ok | Dropped | Payload | HTTP | Result age | Cloud infer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| async_jpeg_q70_every_frame | 1 | 100 | 6.644 | 5/5 | 95 | 32.800 KB | 4544.885 ms | 1382.996 ms | 1080.896 ms |
| async_jpeg_q70_keyframe5 | 1 | 100 | 10.605 | 4/4 | 16 | 31.255 KB | 4105.243 ms | 1367.170 ms | 1081.045 ms |
| async_ppm128_keyframe5 | 1 | 100 | 12.395 | 5/5 | 15 | 28.889 KB | 1837.533 ms | 749.655 ms | 1166.030 ms |
| sync_ppm128_keyframe5 | 0 | 27 | 0.896 | 6/6 | 0 | 28.889 KB | 4602.477 ms | - | 1166.120 ms |
