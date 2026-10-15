# 表 10(a). T41 Learned v2 上传器上板结果。

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_4_ZH.md`, lines 654-658

| Run | Samples | Learned upload | Actual upload | Edge ok | SIGILL | Avg policy | P95 policy | Avg HTTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO20 static | 20 | 70.00% | 20/20 | 90.00% | 31 | 0.649 ms | 0.789 ms | 1678.218 ms |
| Real100 shadow | 100 | 36.00% | 100/100 | n/a | 0 | 4.679 ms | 6.104 ms | 1520.671 ms |
| Real50 online | 50 | 22.00% | 11/11 | 98.00% | 52 | 0.676 ms | 0.817 ms | 1548.606 ms |
