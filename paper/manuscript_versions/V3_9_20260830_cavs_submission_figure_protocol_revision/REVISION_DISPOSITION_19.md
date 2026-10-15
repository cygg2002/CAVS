# V3.7：19 项复核意见处理记录

本文件是作者侧审计记录，不进入投稿正文。状态分为“已修正”“已核实并精简”“待作者信息”“待模板排版”。

| # | 状态 | V3.7 处理 |
|---:|---|---|
| 1 | 已修正 | 摘要不再使用未定义的 MORIC；改为“上下文图像级奖励基线 / context-reward image-level baseline”。CAVS、ROI 与 AP50–95 的现有定义保持不变。 |
| 2 | 已修正 | 补充 COCO train2017 去重 image ID 经 `SHA256(sample:3407:<image_id>)` 排名取前 12,000 的确定性规则。 |
| 3 | 已修正 | 明确 630 张 calibration-audit 与 1,121 张 internal-diagnostic 的用途，并说明它们不进入冻结主实验的拟合、在线参数校准或主结果。 |
| 4 | 已核实并精简 | Oregon 99,231 张按相机事件分成 detector training/validation 与 selector fit/calibration/evaluation 五个互斥角色，计数完整相加为 99,231。 |
| 5 | 已核实并精简 | 明确 Oregon selector evaluation 不参与检测器拟合或 checkpoint 选择。 |
| 6 | 已核实并精简 | 删除 epoch 和内部模型名；仅说明 checkpoint 依据 detector-validation 角色确定。 |
| 7 | 已修正 | 补充 COCO→VOC 的 6 组异名映射、14 组同名映射及非 VOC 类过滤规则。 |
| 8 | 已修正 | 以式（12）正式定义 TwoStage/SharedAction：先对 O/Q 动作价值取 argmax，再固定动作并执行标量容量调度。 |
| 9 | 已修正 | 说明 TwoStage 不另训标量网络；标量是同一 CAVS-OQ 动作价值的最大值，因此不存在额外、未定义的标量监督目标。 |
| 10 | 已修正 | 给出完整在线初始价格与步长候选集合。 |
| 11 | 已修正 | 表 B1 同时列出 CAVS-OQ 与 CAVS-OQR 三档冻结 `(lambda_1, eta)`。 |
| 12 | 已修正 | 定义所有在线二元比较器统一写成 `v_edge=0, v_original=u_t`，并进入同一前缀可行控制器；DCSB/AF 的预算匹配分数先做 calibration 经验分位变换。 |
| 13 | 已修正 | 表 B4 只对 6 个预算匹配工作点声明前缀违规为 0；两个 Native 行仅声明完整性、云失败和非法框统计。 |
| 14 | 待作者信息 | Funding、Competing interests、Author contributions、Data availability、Code availability 与 Ethics approval 已改成明确的“投稿前必须由作者确认”，未代作者虚构声明。 |
| 15 | 待作者信息 | 作者、单位和通讯邮箱继续留空，符合作者要求；这些字段补齐前不能称为可直接提交版本。 |
| 16 | 已修正 | OneAdapt 改为 SoCC 2023 正式版本，页码 158–176，DOI `10.1145/3620678.3624653`。 |
| 17 | 已修正 | YOLO26 改引 Ultralytics 官方在线文档，不再把 arXiv-only 记录作为正式出版物条目。 |
| 18 | 待模板排版 | V3.7 是内容主稿，尚未套 JRTIP 官方 iicol 双栏模板；需在作者信息和声明完成后生成投稿版，并以实际 PDF 核验 12 页限制。当前单栏页数不能用于判断最终是否超页。 |
| 19 | 已修正 | 45 条参考文献统一为编号制 Springer 风格：作者—年份—题名—期刊/会议—卷页—完整 DOI/在线访问日期。 |

## 事实边界

- Oregon 只保留复现所需的角色边界、模型家族、输入尺寸和 checkpoint 哈希，不写 epoch 或内部实验代号。
- 正文使用中性的“冻结协议下事后评测”界定论断强度，不使用带有定性判断的内部数据审计术语。
- 第 14、15 和 18 项不能靠自动编辑真实完成；缺少作者事实或官方模板 PDF 页数时，任何“已投稿就绪”的表述都不可靠。
