# V0.8 论文图设计审计

## 参考图形语言

本次重绘检查了本地归档的 DeepDecision、EdgeBoost、Neurosurgeon、Split Computing and Early Exiting Survey、Edge Intelligence Survey 及既有 V0.6 draw.io 图稿。只提炼视觉组织规律，不复制任何论文图形或图标。

1. DeepDecision 把设备决策、端侧模型和服务器模型组织成短主链，选择位置一眼可见。
2. EdgeBoost 用一个明确菱形表达路由条件，两条结果分支分别落到 Edge 与 Cloud，避免把决策隐藏在长文本中。
3. Neurosurgeon 和 Split Computing Survey 用物理责任域区分设备、无线链路与服务器，通信方向和返回方向分别绘制。
4. Edge Intelligence Survey 使用并列小面板表达互斥架构；这种方式适合综述，不适合本文同时包含在线执行与离线控制器的单一方法图。
5. 既有 V0.6 图的信息密度过高，包含与当前论文无关的队列、deadline 和历史 CAFRU 细节，不能直接复用。

## V0.8 图 1 设计决策

- 图内不放图号和完整图题，图题只由论文排版系统生成。
- 上泳道只画在线部署，且在线只执行一个冻结配置。
- 在线责任域分为 Edge、Routing & Budget、Network Payload 和 Cloud。
- 三个核心决策用编号标明：1 样本 Ranker、2 预算调度、3 载荷动作。
- 明确画出 Edge-only 旁路，防止读者误解所有样本都经过云端。
- 下泳道只画 fit/selector/gate/freeze/final 关系。
- final 只有从冻结配置进入的单向箭头；final oracle 标记为诊断且无反馈。
- selector 目标区分 Fixed-call/Online 的低 KB 选择与 Fixed-byte 的高 AP 选择。
- gate 明确写出 200 次 paired bootstrap，并把失败回退写入冻结节点。
- 离线部署使用绿色虚线；在线数据流使用深灰实线；不使用渐变、阴影、3D 和装饰图标。

## 定量图边界

图 2-4 继续从冻结 CSV/JSON 自动生成。它们是定量证据图，不在 draw.io 中手工录入数据。手工重画会破坏数值可追溯性和后续复现，因此 draw.io 仅负责架构类示意图。

## 出图要求

- 可编辑源文件：`.drawio`。
- 投稿矢量文件：PDF 和 SVG。
- Word 预览文件：至少 300 dpi PNG。
- 缩放至 174 mm 双栏宽后，正文不小于约 8 pt。
- 灰度打印时仍能依靠泳道、边框和线型区分模块。
