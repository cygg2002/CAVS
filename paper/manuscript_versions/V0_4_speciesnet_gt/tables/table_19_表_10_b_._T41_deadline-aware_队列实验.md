# 表 10(b). T41 deadline-aware 队列实验。

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_4_ZH.md`, lines 664-671

| Setting | Async | Queue | Frames | FPS | Upload ok | Dropped | Avg age | 1000ms hit | Effective FPS@1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sync jpeg q70 every-frame | 0 | 1 | 19 | 0.614 | 19/19 | 0 | - | 0 | 0.000 |
| async q1 deadline1000 | 1 | 1 | 100 | 6.623 | 11/11 | 89 | 503.375 ms | 62 | 4.106 |
| async q2 deadline1000 | 1 | 2 | 100 | 6.613 | 12/12 | 88 | 510.449 ms | 62 | 4.100 |
| async q2 deadline500 | 1 | 2 | 100 | 6.828 | 12/12 | 88 | 245.647 ms | 31 | 2.117 |
| async q2 deadline2000 | 1 | 2 | 100 | 6.663 | 12/12 | 88 | 699.024 ms | 65 | 4.331 |
| async q4 deadline1000 | 1 | 4 | 100 | 6.751 | 15/15 | 85 | 471.275 ms | 63 | 4.253 |
