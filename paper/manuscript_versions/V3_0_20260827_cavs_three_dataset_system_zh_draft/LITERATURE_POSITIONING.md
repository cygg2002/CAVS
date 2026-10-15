# 文献定位与终稿写法（2026-08-27）

## 1. 结论

最稳妥的论文定位是：

> CAVS 将 ORIC/MORIC 的“图像级、单一云动作、调用比例约束”扩展为“AP 对齐的图像—像素载荷动作价值学习”，并把逐图真实字节留给约束调度器；离线严格字节调度给出批量上限，零前视控制器给出可部署在线结果。

不要把 JPEG、ROI、上下文奖励、集合编码器、拉格朗日价格或背包求解单独写成首次提出。最有辨识度的组合是：

1. object-detection AP 的 contextual image–action reward；
2. value 与 byte cost 解耦；
3. Original/Q90/ROI 等异构载荷的联合成员—动作调度；
4. 同时报告全局容量上限和零前视在线代价。

## 2. 借鉴 ORIC/MORIC 的论证骨架

ORIC/MORIC 的正文顺序是 Introduction → Background → Problem Formulation → ideal ORIC reward → deployable estimator/MORIC → Evaluation → Related Work → Conclusion。值得直接借鉴的不是措辞，而是四个论证动作：

- 在引言中只指出两个可验证缺口，并用小表比较 prior work；
- 把组合优化问题、理想 reward、可部署 estimator 分开；
- 先用 Oracle/teacher 证明 reward 定义合理，再评估学习器；
- 在真实 Jetson Xavier NX–RTX 3090/1-GbE 测试台上给出弱检测、估计器、传输和强检测的时延拆分。

CAVS 应沿用这一逻辑，但把“单图是否上传”改为“当前图像应采用哪种像素载荷”，并新增全局与因果在线两种决策语义。

## 3. 借鉴系统论文的证据链

### DCSB

DCSB 从多目标和小目标容易漏检的观察出发，构造 difficult-case discriminator，再报告上传比例、检测对象比例、端到端 mAP 与带宽。其带宽自适应版本已经包含 regional sampling 和 discriminator zoo。因此本文只能把 DCSB 当难例路由基线，不能把区域上传本身写成首创。

### OffloaD

OffloaD 以 detection failure metric 联合 introspective、golden 和 latency score，并在物理测试台及退化网络下报告 precision–latency。它提醒本文必须交代断网回退、云请求失败、P95/P99 和网络条件，而不能只给 AP/KB。

### AccMPEG

AccMPEG 先定义准确率、端到端时延、摄像头计算三项不可兼得目标，再用 Motivation → Overview → Online encoding → Offline training → Evaluation 的结构展开。它主动承认标准 codec 和 ROI encoding 并非新组件，把贡献收紧为 accuracy gradient 与低开销训练。本文也应主动承认通用组件，把贡献集中在 image–action AP value 与实际字节调度。

### Token-bucket 在线路由

Real-Time Edge Classification under Token Bucket Constraints 明确区分平均速率与短时 burst，并根据本地置信度和 token state 逐帧决策。CAVS 的在线章节应采用同样的因果性定义：`buffer_frames=0`、`zero_lookahead=true`，决定后立即上传；不能把全量评分后全局求解称为实时在线。

### Budget-Adaptive Routing（2026）

该工作指出 weak-conditioned router 在高上传预算仍为最终上传帧支付 weak forward cost，并提出 weak-skipping 分支。CAVS 当前依赖 Edge 框来估值和构建 ROI，因此属于 weak-conditioned 系统。论文应把高预算计算税作为边界，并将 weak-skipping 作为未来/附加对照，而不是声称全预算端侧计算最优。

## 4. 推荐终稿结构

1. Introduction：场景、缺口、核心反例、三项贡献、结果摘要。
2. Related Work：选择性卸载、任务感知传输、多动作预算路由、在线资源控制。
3. System Model and Problem Formulation：端/云/载荷、非可加 AP、Fixed-byte、offline/online 两种问题。
4. CAVS：contextual reward、集合编码器、action-conditioned value、成本解耦、训练。
5. Scheduling and System Realization：全局严格容量、零前视控制、ROI/MAUR、失败回退。
6. Experimental Methodology：三数据集、检测器、数据角色、比较器、统计、硬件。
7. Results：三数据集主表、核心归因、O/OQ/OQR、系统 HIL、在线差距。
8. Discussion：可学习性、ROI 非普适、离线—在线差距、部署边界。
9. Threats to Validity and Limitations。
10. Conclusion。

## 5. 与最近邻的差异表

| 方法 | AP/mAP 对齐 | 多像素载荷 | 逐图真实字节 | 图像—动作联合 | 零前视在线 | 真实硬件 |
|---|---:|---:|---:|---:|---:|---:|
| DCSB | 否/启发式 | 区域扩展版有 | 部分 | 否 | 是 | 是 |
| ORIC/MORIC | 是 | 否 | 否，主约束为上传率 | 否 | 阈值式 | 是 |
| OffloaD | 窗口 mAP/失败 | 否 | 非主目标 | 否 | 滑动窗口 | 是 |
| AccMPEG | 任务代理 | 编码块质量 | 带宽/时延 | 空间质量选择 | 在线编码 | 是 |
| Selective Cloud Offloading | 不确定性/precision-recall | ROI packing | 传输/API 成本 | 区域选择 | 可在线 | 有系统评测 |
| CAVS | AP50–95 contextual reward | Original/Q90/ROI | 是 | 是 | 是，另有全局上限 | RK3588 HIL |

## 6. 写作边界

- “优于”可用于点估计；“显著优于”只在相应 paired/bootstrap CI 排除 0 后使用。
- COCO、Oregon、VOC 均为事后完整评测，不写 confirmatory SOTA。
- ORIC/MORIC 是忠实定义下的统一系统重实现，不写成官方代码逐位复现。
- 当前在线结果是先导 HIL；完成统一协议矩阵前，不把其时延放入摘要主结论。
- Oregon 和 VOC 证明 ROI 并非普适增益，恰好支持“动作由价值竞争决定，而不是强制使用 ROI”的框架叙事。

