# Table 01

Source: `EDGE_CLOUD_DETECTION_PAPER_DRAFT_ZH.md`, lines 241-247

| 方法 | 说明 |
| --- | --- |
| Edge-only | 仅使用端侧 YOLO26s |
| Cloud-only | 全部上传云端 YOLO26x |
| Rule-based | 基于低置信、小目标、贴边目标等规则上传 |
| Learned v1 | 使用学习型概率上传器 |
| Learned v2 | 使用概率收益 + 代价联合上传器 |
