# 表 5. official mAP 与部署阈值口径的公平性检查。

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_4_ZH.md`, lines 585-591

| Subset | Upload | Edge official | Edge deploy | Cloud official | Cloud deploy | Fused deploy | Fused-Edge official | Fused-Edge deploy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.4246 | 0.4733 | 0.5350 | 0.5501 | 0.5996 | 0.5765 | 0.1033 | 0.0415 |
| tiny_gt | 0.7409 | 0.3673 | 0.4413 | 0.4415 | 0.5050 | 0.4834 | 0.1161 | 0.0421 |
| high_coverage_gt | 0.4952 | 0.4630 | 0.5124 | 0.5716 | 0.6134 | 0.5639 | 0.1009 | 0.0515 |
| learned_uploaded_q80 | 1.0000 | 0.4169 | 0.4756 | 0.5027 | 0.5536 | 0.5447 | 0.1278 | 0.0691 |
| learned_not_uploaded_q80 | 0.0000 | 0.5536 | 0.6192 | 0.6274 | 0.6665 | 0.6220 | 0.0684 | 0.0028 |
