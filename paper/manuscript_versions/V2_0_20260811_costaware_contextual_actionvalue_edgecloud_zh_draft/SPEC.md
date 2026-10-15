# V2 中文初稿写作规格

## 目标

在不修改 V1 的前提下，新建一份完整中文论文初稿，将论文主线从“离线上传配置选择”重构为“通信成本感知的图像—载荷动作联合调度”。初稿面向 Journal of Systems Architecture 风格，同时保留改投 Journal of Real-Time Image Processing 或 Engineering Applications of Artificial Intelligence 的空间。

## 核心论点

端云协同目标检测的正确调度单元不是孤立图像，而是图像—载荷动作。现有 Top-K 路由是二元、等成本动作下的特殊情况；面对 Original/Q90 等异构载荷与逐图真实字节，应学习动作条件的 contextual AP 价值，并在严格字节容量下联合选择图片和动作。

## 主方法与范围

- 论文方法暂称 CAVS（Contextual Action-Value Scheduler）。
- 当前主实例为独立 Original/Q90 动作空间的 `CAVS-OQ-Core`：5-fold OOF、3 seeds、全量 fit 重训；不得与历史 Raw ActionValue 或 OQR checkpoint 的 OQ-mask 混名。
- ROI 作为可选扩展和消融，不作为标题级主张。
- COCO 与 Oregon 为主数据集；ENA24 和历史九 Ranker 不进入主叙事。
- Fixed-byte 是主预算语义；Fixed-call 是机制边界和辅助实验。
- Orange Pi 5 Ultra / RK3588 是计划中的真机系统验证，不把尚未完成的测量写成结果。

## 证据等级

1. `已核验`：本地/服务器报告、结果文件和代码语义可以相互核对，可写入正文；若 bootstrap 未完成，只写点估计并明确标注。
2. `事后探索`：已消费评测集上的结果，只能用 post-hoc/exploratory 措辞。
3. `诊断`：OOF、calibration、teacher/oracle 或机制实验，只能用于归因。
4. `待完成`：使用 `【待补：...】` 占位，不填猜测数字。

## 文件结构

- `MANUSCRIPT_V2_0_ZH_DRAFT.md`：完整中文源稿。
- `SPEC.md`：本写作规格。
- `PLAN.md`：章节与核验计划。
- `EXPERIMENT_PLACEHOLDERS.md`：所有待补实验的集中清单。

## 写作风格

- 先陈述问题和可验证命题，再介绍组件。
- 区分事实、机制推断和研究判断。
- 不把通用组件（JPEG、ROI、双头网络、背包优化、contextual reward）单独声称为首创。
- 不使用“精确最优求解器”“实时系统”“新隐藏 final”这类尚无证据的措辞。
- 数字统一报告 Official mAP50–95、请求 KB/图、预算与证据层级。
- 相关工作优先引用论文或正式项目页，不引用搜索结果或二手解读。

## 验证方式

- 检查一级、二级和三级标题完整性。
- 检查所有 `【待补：...】` 均可检索，并集中登记。
- 检查结果数字能在本地实验总结中找到。
- 检查 COCO val2017 与 Oregon final 的 post-hoc 边界被明确披露。
- 检查主结果没有把 Fixed-byte 优势扩写为 Fixed-call 全面优势。
- 检查 CAVS 的调度器被描述为容量可行的拉格朗日调度，而非已证明的离散全局最优算法。

## 不在本轮范围

- 不生成或修改检测器、学习器、缓存与实验结果。
- 不修改 V1 文稿、图表或构建脚本。
- 不生成最终投稿格式的 DOCX/PDF。
- 不补写尚未得到的 RK3588、网络、能耗或新终评数字。
