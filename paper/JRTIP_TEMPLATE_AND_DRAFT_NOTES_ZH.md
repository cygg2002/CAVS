# JRTIP 投稿模板与当前初稿说明

更新时间：2026-06-23

## 1. 当前采用的模板口径

当前论文初稿文件：

`paper/JRTIP_MANUSCRIPT_DRAFT_V0_2_ZH.md`

该文件按 Springer/JRTIP 期刊论文结构组织：

```text
Title
Abstract
Keywords
1 Introduction
2 Related Work
3 System Architecture
4 Proposed Method
5 Experimental Setup
6 Results and Analysis
7 Discussion
8 Conclusion
Declarations
References
```

这是中文主稿，不是最终投稿文件。正式投稿 JRTIP 时，应转成英文，并使用 Springer Nature LaTeX 或 Word 模板排版。

## 2. JRTIP 当前关键要求

根据 JRTIP 官方页面和 Springer 作者说明，当前写作需要注意：

1. JRTIP 明确要求稿件讨论 real-time issues，否则可能不进入审稿范围。
2. JRTIP 从 2020 年开始对投稿实施 12 页限制，包含参考文献和作者简介，且按期刊双栏格式计算。
3. Springer 建议期刊作者使用 Springer Nature LaTeX authoring template，通常为 `sn-jnl.cls` 模板。
4. 当前初稿必须压缩成“方法清楚、实验集中、图表精简”的版本，否则转成双栏后容易超页。

官方页面：

- JRTIP journal page: https://link.springer.com/journal/11554
- JRTIP submission guidelines: https://link.springer.com/journal/11554/submission-guidelines

## 3. 当前初稿的投稿状态判断

当前 v0.2 初稿已经具备论文骨架：

- 有明确问题定义：低功耗摄像头下 Edge-only 精度不足，Cloud-only 上传和计算成本高。
- 有方法主线：帧级与 ROI 级上传、收益-代价决策、Learned v2 上传器、端云融合。
- 有 COCO full 主实验和消融实验。
- 有真实 T41 板端 JPEG、异步上传、NNA 稳定性和闭环原型实验。

当前还不适合直接投稿，主要缺口是：

1. 参考文献未正式整理，Related Work 仍需补充真实引用。
2. E21 规则选择性闭环已有 100 样本有效结果，但仍不能作为真实场景 mAP 证据。
3. T41 真实视频还没有人工标注，因此不能报告真实场景 mAP。
4. 中文主稿需要转英文，并按 12 页双栏压缩。
5. 图表需要重新排版，建议最终保留 5-6 个核心图表。

## 4. 建议最终投稿版图表

建议 JRTIP 投稿版保留：

| 编号 | 内容 | 来源 |
|---|---|---|
| Fig. 1 | 系统总体架构图 | 方法部分重新绘制 |
| Fig. 2 | Learned v2 上传器与收益-代价决策流程 | 方法部分重新绘制 |
| Fig. 3 | mAP - 上传量 - GFLOPs 气泡图 | E16 |
| Table 1 | COCO 主结果 | E16 Table 1 |
| Table 2 | 核心消融：成本项、ROI baseline、cloud imgsz、JPEG quality | E16 Table 2 精简 |
| Table 3 | 弱网与队列实时性 | E16 Table 3a/3b 精简 |
| Table 4 | T41 真实系统成本 | E18/E19/E20/E21 精简 |
| Fig. 4 | 2-3 个可视化案例 | E13 重新排版 |

如果页数紧张，可以把困难子集和指标公平性放入 supplementary material。

## 5. 下一步写作任务

优先级从高到低：

1. 补 Related Work 的正式参考文献和 BibTeX。
2. 修复并重跑 E21 规则闭环，更新 T41 Table 4。
3. 把 v0.2 中文稿压缩成 12 页英文 JRTIP 版本。
4. 生成 Springer Nature LaTeX 工程。
5. 按 JRTIP 双栏模板检查页数、图表、引用和 declarations。
