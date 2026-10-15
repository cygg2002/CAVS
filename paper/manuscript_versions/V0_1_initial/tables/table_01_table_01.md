# Table 01

Source: `JRTIP_MANUSCRIPT_DRAFT_ZH.md`, lines 291-298

| 方法 | 说明 |
| --- | --- |
| Edge-only | 仅使用端侧 YOLO26s，完全不上云 |
| Cloud-only | 所有图像上传云端 YOLO26x 检测 |
| Ordinary ROI+Cloud | 只根据低置信、小目标、贴边目标等固定规则裁剪 ROI 并上传云端复核 |
| Rule aggressive | 基于低置信、小目标、贴边和复杂场景等规则上传 |
| Learned v1 | 使用学习型概率上传器 |
| Learned v2 | 使用概率收益和上传代价联合决策 |
