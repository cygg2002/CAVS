# Table 03

Source: `JRTIP_MANUSCRIPT_DRAFT_V0_3_ZH.md`, lines 430-435

| 实验 | 目的 | 主要输出 |
| --- | --- | --- |
| E18 JPEG upload | 验证板端 JPEG 编码和真实云端 YOLO26x 上传链路 | JPEG 质量、payload、encode ms、HTTP ms、cloud infer ms、FPS |
| E19 async pipeline | 验证异步上传和有界队列是否能避免主循环被云端阻塞 | 同步/异步 FPS、drop 数、result age、queue ablation |
| E20 NNA stability | 验证 YOLO26s/YOLO26n 在 T41 NNA 上的延迟与稳定性 | infer ms、post ms、SIGILL、success-only latency |
| E21 closed loop | 验证同一帧上端侧 YOLO26s、上传决策、JPEG 上传、云端 YOLO26x 和结果返回是否可闭环 | upload success、edge infer、cloud infer、final boxes、日志一致性 |
