# V3.6 审稿问题处置矩阵（1–245）

本文档记录用户提供的 245 项审查意见如何进入 V3.6。状态分为：**已修订**、**核实后保留**、**明确披露**和**作者待填**。V3.5 未被覆盖。

## 1. 标题、摘要与关键词（1–18）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 1–3 | 已修订 | 中文摘要与引言首次出现时均展开 Contextual Image–Action Value Scheduling；中文统一为“调度”，英文统一为 Scheduling，不再混用 Scheduler。 |
| 4 | 已修订 | 删除“多质量编码”，改为 Original、单一 Q90 与 ROI 等异构载荷。 |
| 5–6 | 已修订 | 双语摘要首次定义 COCO-style AP50–95，并在全文统一使用 AP50–95。 |
| 7–8 | 已修订 | 摘要删除运行组数和完整设备串，仅保留“RK3588–GPU 零前视端云测试床”。 |
| 9–10 | 已修订 | 按每档最强外部基线重算为 +0.016278/+0.036928/+0.037235 AP，摘要写为 0.0163/0.0369/0.0372。 |
| 11–12 | 已修订 | 摘要与问题定义均限定为 prefix payload-byte 和 Always-Original JPEG payload。 |
| 13–16 | 已修订 | 同骨干结果限定为 COCO 受控结构消融；三数据集结果限定为冻结事后评测；时延限定为该测试床测量。 |
| 17–18 | 已修订 | 摘要改为“问题—方法—离线结果—系统验证—结论”结构；关键词统一为端云协同等正文术语。 |

## 2. 引言（19–39）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 19–21 | 已修订 | 正文重新展开 CAVS 与 ORIC；MORIC 只写 modified ORIC values，不编造独立全称。 |
| 22–24 | 已修订 | AF 准确描述为快/慢检测器级联并说明其为适配基线；不再把 AF、DCSB、ORIC 的原生预算统一说成调用比例。 |
| 25–29 | 已修订 | 明确 Original 是数据集原始 JPEG bitstream，Q90 是固定参数重编码，ROI 首次展开；所有覆盖优势限定在相同 payload-byte budget 下。 |
| 30–33 | 已修订 | 全文将“算法/批量/离线上限”改为 full-information feasible reference，并明确求解器不保证 MCKP 精确最优；图 1 同步修改。 |
| 34–36 | 已修订 | 从贡献中删除未正式定义的 Fixed-call；把价值预测与容量调度分为两个模块描述。 |
| 37–39 | 已修订 | 贡献删除运行组数，修正在线增益，并限定为全部前缀 payload-byte 约束。 |

## 3. 相关工作（40–49）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 40–44 | 已修订 | 展开 AF，保留并准确描述 DCSB/DCCI；ORIC 写为 representative context set，CAVS 写为从图像级二元卸载扩展到图像—动作价值。 |
| 45–49 | 已修订 | “充分说明”改为“表明”；ROI 已在引言展开；解释 weak-conditioned；用 mandatory weak-detector inference overhead 替代非正式“计算税”；保留克制创新边界。 |

## 4. 系统模型与问题定义（50–67）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 50 | 已修订 | 区分图像编码模块与神经检测集合编码器。 |
| 51–54 | 已修订 | 明确 ROI 是多个扩张框合并后的一个外接 crop、一个动作和一个 payload；严格定义 O/OQ/OQR。 |
| 55–58 | 已修订 | 规定失败前已发送 payload 计入预算、检测回退 Edge；说明主运行无失败；在 3.1 即定义主预算只约束 JPEG payload。 |
| 59–60 | 已修订 | 式（3）明确称 AP-aligned surrogate objective，不再说直接优化可加 AP。 |
| 61–62 | 已修订 | 正文与附录统一为“至多前 K 个正价值样本，余者保留 Edge”。 |
| 63–65 | 已修订 | 零前视改成自然语言学术定义；配置字段移除；删除无对应结果的 paced-burst。 |
| 66–67 | 已修订 | 指明正式 trace 按 image ID 升序，并给 sequence SHA-256；局限中说明顺序依赖。 |

## 5. CAVS 方法（68–90）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 68–74 | 已修订 | 展开 MLP；首次给出 top-64；逐项定义框、场景与 ROI 几何特征；只声称排除显式成本坐标，不作过强独立性断言。 |
| 75–78 | 已修订 | 给出 context=32、同折、排除目标、无放回、SHA-256 确定性采样；同图动作与不同训练 seed 共享冻结 context 表。 |
| 79–80 | 已修订 | 展开 OOF，并区分 OOF 机制诊断与三个 full-fit 最终学习器。 |
| 81–88 | 已修订 | 将 MAUR 正式定义为本文的 masked-area update-and-replacement，给出中心判定、duplicate IoU、NMS；完整定义 Current、Historical-cost 和 Reversible。 |
| 89–90 | 已修订 | 明确 O/OQ/OQR 独立生成 reward 并重训；相邻差值只称配置差异，不称单动作纯因果效应。 |

## 6. 调度与系统实现（91–107）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 91、96 | 已修订 | 所有“上限/最优”措辞改为 full-information feasible reference。 |
| 92–95 | 已修订 | 给出 lambda 上下界、80 次二分、净值 tie-breaking、repair 排序与停止方式。 |
| 97–98 | 已修订 | 说明 calibration 目标、lambda 网格、投影区间、初始价格与三档 OQR step size；静态价格作为 eta=0 消融。 |
| 99–102 | 已修订 | 主文只称紧凑二进制 HTTP 协议与端云测试床；内部协议名、运行合同和 QA 语言移出主叙事。 |
| 103–107 | 已修订 | 明确 YOLO26n RKNN-FP16、板端 OS/kernel/Python/RKNN 版本及云端 PyTorch/CUDA/cuDNN/Ultralytics；正式化 mandatory weak-detector inference overhead。 |

## 7. 实验设置（108–139）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 108–109 | 已修订 | 解释 COCO 12,000 图候选池、SHA-256 image-ID 划分、seed、角色数量与清单摘要。 |
| 110–112 | 明确披露 | Oregon 改称作者持有集合，不再称公开数据集；外部正式引用与许可仍依赖作者提供。 |
| 113–115 | 已修订 | 明确 Oregon 端/云均为 YOLO26 架构、输入、epoch、指标与 checkpoint 哈希；删除内部 Cloud-V3 作为唯一模型定义。 |
| 116–117 | 已修订 | 给出 Torchvision 版本和两个 weights enum，并增加官方文档及 MobileNetV3/Faster R-CNN 原始文献。 |
| 118–120 | 已修订 | 全文统一为 frozen post-hoc evaluation；角色名由 final 改为 evaluation。 |
| 121–123 | 已修订 | 删除没有全表支持的“并报告 AP50/AP75”；Current 和 MAUR 已在方法中定义。 |
| 124 | 核实后保留 | DCSB 三阈值规则按原文保留。 |
| 125–128 | 已修订 | 表格统一标 Adapted AF/Adapted DCSB；列出 ORIC/MORIC 保留部分与 Fixed-byte 适配部分。 |
| 129 | 已修订 | TwoStage SharedAction 的两阶段作用与受控差异在 6.2/7.3 明确。 |
| 130–133 | 已修订 | Random 改为正式 seed/selection 描述；bootstrap 明确 paired image、1,000 次及重采样单位。 |
| 134–136 | 已修订 | 删除运行组数与“验收条件”，改为实验矩阵和跨运行观测事实。 |
| 137–139 | 已修订 | 给出升序 image-ID trace 与哈希，并在局限中说明单序列顺序依赖。 |

## 8. 结果与分析（140–182）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 140–144 | 已修订 | 主表列名为 Delta AP；保持 point-estimate 措辞；Oregon 明确 25% 到 40% 下降 0.000742；影子价格解释降格为机制一致性。 |
| 145–149 | 已修订 | 标题改“受控同骨干结构消融”；“唯一变化”改“主要受控差异”；结论限定为支持而非证明全部归因。 |
| 150–156 | 已修订 | 明确五折 OOF 与三个 full-fit seeds；所有 Q90/ROI 增益改为 OQ−O/OQR−OQ 独立重训配置差。 |
| 157–158 | 已修订 | 四种 profile/fusion 均已在 4.4 定义；表注说明 1,000 次 paired image bootstrap。 |
| 159–163 | 已修订 | 离线到在线阶梯改为描述性，不再用 Mac 单图结果作主因果证据；列出未隔离的域、信息结构、ROI 和软件栈因素。 |
| 164–167 | 已修订 | 修正 40% 最强基线为 Adapted AF；摘要、贡献和结果统一为 0.0163/0.0369/0.0372 AP。 |
| 168–169 | 已修订 | 在线 OQR−OQ 明确为无 CI 的配置点差；删除运行组数主叙事。 |
| 170–176 | 已修订 | 表 7 和 B6 明确阶段均值以全部输入为分母；Cloud RTT 改为 Cloud HTTP round-trip；FPS 定义为串行 pipeline；0.43% 仅限定上行额外字节。 |
| 177–182 | 已修订 | 图 6 改 transport sensitivity；明确策略不读取网络状态；控制器表新增静态/自适应利用率和前缀超额；定义 ping 端点。 |

## 9. 讨论、局限与结论（183–209）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 183–188 | 已修订 | 讨论删除运行次数；VOC 改为“超出 YOLO-only 的证据”并明确不足以证明普遍性；保留 Oregon 未执行 ROI 的边界。 |
| 189–193 | 已修订 | 理论式绝对断言改为“一般不能复现”；未实现未来方法不再用连续未定义术语；基线适配边界前移到实验设置。 |
| 194–199 | 已修订 | 三个 evaluation set 均按事后冻结口径表述；cost-aware 首次即限定 JPEG payload；静态输入不再称真实视频帧；板端仅 COCO 明确列为局限。 |
| 200–204 | 已修订 | 结论不重复无必要缩写定义；“严格”只修饰可行性而非最优性；同骨干机制结论限定 COCO；在线优势限定所评测重实现/适配基线。 |
| 205–209 | 已修订 | 删除“多质量 Q90”；改为 VOC 中单一 Q90 配置贡献；网络结论与无丢包/无失败/无 deadline fallback/动作固定条件同句出现。 |

## 10. Declarations（210–219）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 210–215 | 作者待填 | 按用户要求保留作者、单位、邮箱、Funding、Competing interests 和 CRediT 占位，不虚构。 |
| 216–217 | 作者待填/已修订 | Oregon Data availability 仍需作者填写；正文已改为作者持有集合，不再称公开评测集。 |
| 218–219 | 作者待填 | Code availability 与 ethics approval 保留明确占位。上述项目是投稿前阻塞项。 |

## 11. 附录与复现（220–236）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 220–228 | 已修订 | 表 B1 给出三个数据集 seeds、context protocol、lambda 区间/网格、在线 step size；主文给出 ROI/MAUR/Reversible；表 B8 给出完整软件环境。 |
| 229–230 | 已修订 | B2 明确为三个训练 seed 间 mean±sample SD，并声明只是描述性稳定性指标。 |
| 231–233 | 已修订 | B3/B5 明确 VOC 的 DCSB/AF not evaluated 及原因。 |
| 234–236 | 已修订 | 内部工程语言改为 Supplementary Reproducibility Checklist；删除“表格应自动生成”等作者工作流表述。 |

## 12. 参考文献（237–245）

| 编号 | 状态 | V3.6 处置 |
|---|---|---|
| 237–238 | 已修订 | ORIC 正文首次展开；MORIC 只使用原论文的 modified ORIC values。 |
| 239–240 | 已修订 | 新增 Torchvision 两个具体模型文档、MobileNetV3 与 Faster R-CNN 原始论文。 |
| 241 | 作者待填 | Oregon 为作者持有集合；不能虚构外部引用。若已有 dataset paper，投稿前应补充；否则 Data availability 必须给出正式来源与访问说明。 |
| 242 | 已修订 | 新增 Rockchip RKNN Toolkit2 官方来源，并在正文/表 B8 给出版本。 |
| 243 | 已修订 | 参考文献统一为编号制作者—题名—来源—年份—卷页—DOI/URL 的基本 Springer 形式。 |
| 244–245 | 核实后保留 | DCSB、AF、ORIC、DCCI 的真实原文与方法事实保留，不因陌生而删除或质疑。 |

## 投稿前仍需作者完成

1. 作者、单位、通讯邮箱及全部 Declarations。
2. Oregon 数据许可、正式来源/引用、访问方式、伦理与敏感地理信息说明。
3. 匿名代码/模型/数据角色清单归档链接或 DOI。
4. 若目标期刊提供专用 Word/LaTeX 模板，需在内容定稿后做最后一次模板映射；V3.6 当前采用 Springer/JRTIP 通用作者稿规范。
