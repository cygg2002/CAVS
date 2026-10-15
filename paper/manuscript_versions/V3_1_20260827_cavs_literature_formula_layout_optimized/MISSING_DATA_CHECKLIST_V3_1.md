# 投稿前缺失数据与证据清单

## P0：正文主张的硬门槛

- [ ] COCO：最新 CAVS-OQR 对 ORIC、MORIC、Confidence、Random-50 的三预算 1,000 次 paired bootstrap。
- [ ] VOC：三预算 paired bootstrap；Random 使用每个重采样内 50 套 selection 求均值的 nested 口径。
- [ ] Oregon：以 event cluster 为主单位的三预算 1,000 次 paired bootstrap；图片 bootstrap 仅作敏感性分析。
- [ ] TwoStage 对 CAVS-OQ 的最新三种子、三预算 paired CI。
- [ ] 三个数据集的单模型三 seed AP、均值±SD、ensemble AP 和 ensemble 增益。
- [ ] 正式零前视 15 组：CAVS、ORIC、MORIC、Confidence、Random × 10/25/40%，同设备、同云端、同网络、同二进制/multipart 协议。
- [ ] 每个正式在线 run：完整帧数、cloud failure=0、非法框=0、prefix violation=0、运行合同与模型/策略/服务哈希。
- [ ] 作者、单位、通讯邮箱、基金、CRediT、竞争利益、Oregon 数据许可/伦理、代码仓库声明。

## P0：可复现性元数据

- [ ] COCO fit/calibration 的抽样规则、完整 image ID 和 SHA-256。
- [ ] Oregon 端模型架构/checkpoint/hash；事件时间阈值；各 split 的相机数、事件数、事件大小分布与类别频次。
- [ ] VOC COCO→VOC 20 类映射、忽略类别、输入归一化、score/NMS/max-det、Torchvision 版本与权重哈希。
- [ ] YOLO/RKNN/云模型、推理尺度、conf、IoU、max-det、rect/scaleup/end2end、编码器版本和完整哈希。
- [ ] 服务器 GPU/CPU/内存、CUDA/驱动/框架版本，以及板端 OS、RKNN runtime/driver、频率模式。

## P1：系统论文说服力

- [ ] CAVS Value 网络参数量、MAC/FLOPs、RK3588 CPU mean/P50/P95/P99、峰值内存。
- [ ] Edge、router、编码、排队、wire RTT、云推理、回传、融合的 mean/P50/P95/P99 分解。
- [ ] JPEG payload、HTTP body、真实 wire bytes 与返回字节。
- [ ] 每档云动作图像数、HTTP 请求数、ROI crop 数、云 GPU 总毫秒；不要把上传图片数等同于云计算量。
- [ ] FPS、deadline 定义与 miss ratio；不能只写“实时”而不给目标频率/时限。
- [ ] 外接功率计 J/frame、平均/峰值功率、温度、NPU/CPU 频率与 throttling。
- [ ] 至少 3–4 个带宽/RTT/抖动条件，最好加入丢包；同协议报告 AP—wire—latency Pareto。
- [ ] 高预算 cloud-first / weak-skipping 对照，定位 weak-forward tax 拐点。

## P1：算法诊断与图表

- [ ] AP50/AP75：三数据集所有主基线，而非仅 CAVS 或少数表格。
- [ ] 各动作 value 的 Spearman/Kendall、top-value recall、动作选择重合率。
- [ ] 小规模 exact multiple-choice knapsack 与拉格朗日—修复的目标值/动作/容量差距。
- [ ] Teacher/oracle frontier、Always-Edge、Always-Original、Always-Q90 与可实现上界。
- [ ] TIDE 或类别/小中大目标分解，解释 Q90、ROI 和 Oregon 饱和。
- [ ] 图 5 正式系统 Pareto/时延分解/网络条件。
- [ ] 图 6 按 calibration 预定义规则抽取的成功与失败案例，包含 ROI 有益、ROI 无益、Q90 性价比和在线误选。

## P2：提高投稿稳定性

- [ ] 新注册、未被开发访问的确认集或时间后移 Oregon 测试。
- [ ] 网络分布漂移与不同序列顺序的 online controller 稳健性。
- [ ] 第二类边缘硬件或第二条网络路径。
- [ ] 在线 regret / distributionally robust price / bandit knapsack 的独立冻结验证。
