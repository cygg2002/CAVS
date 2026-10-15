# V3 投稿前待补实验

## P0：直接决定正文结论

1. VOC Fixed-byte 1,000 次 paired bootstrap，含 nested Random-50；完成后替换表 7 的 CI 占位。
2. COCO CAVS-OQR 对 ORIC/MORIC、Confidence、Random-50 的同一冻结 selection 统一 paired bootstrap。
3. Oregon CAVS-OQ 与最新 Current+MAUR 对 ORIC/MORIC、Confidence、Random 的 1,000 次 event-cluster 主 bootstrap；image bootstrap 仅作敏感性分析。
4. 修正协议后的统一零前视在线矩阵：CAVS、ORIC、MORIC、Confidence、Random × 10/25/40%，同一 RK3588、云 GPU、网络路径和二进制/multipart 协议。

## P1：系统论文完整性

1. 每组在线 run：5,000/5,000、cloud failure=0、非法框=0、prefix violation=0、完整 SHA 合同。
2. 分解 Edge、CAVS、JPEG/ROI 编码、上传、云推理、融合、E2E 的 mean/P50/P95/P99；报告 FPS。
3. 同时报 JPEG payload 与 HTTP wire bytes；若仍为 Base64，不能把 payload 百分比称为链路字节百分比。
4. 外接功率计测 Edge-only/CAVS/上传动作的 J/frame；记录温度、频率与降频。
5. 至少三种带宽/RTT 条件及抖动或丢包；报告 deadline miss 与回退。

## P2：增强而非最低投稿门槛

1. 新注册、未被调参访问的确认集。
2. high-budget weak-skipping/Cloud-first 对照。
3. TIDE 或类别/尺寸误差分解、Teacher ceiling、score-reward Spearman 与动作 overlap。
4. 在线控制器的 paired bootstrap；当前 primal-dual/limited-burst 只收回少量离线差距，不能据 final 事后选控制器。

