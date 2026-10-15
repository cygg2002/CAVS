# CAVS V3.0 中文完整初稿

本目录是 2026-08-27 新建的独立稿件版本，不覆盖 V1/V2。

## 文件

- `MANUSCRIPT_V3_0_ZH_DRAFT.md`：按终稿章节组织的中文完整初稿。
- `RESULT_EVIDENCE_LEDGER.md`：正文数字、来源文件与证据等级台账。
- `LITERATURE_POSITIONING.md`：与 ORIC/MORIC、DCSB、OffloaD、AccMPEG 等工作的定位与写作结构对照。
- `MISSING_EXPERIMENTS.md`：所有尚不能填入终稿的实验和验收条件。
- `scripts/build_figures.py`：从冻结点结果生成论文结果图。
- `figures/`：可直接插入论文的 SVG/PNG 图。

## 证据纪律

1. COCO val2017、Oregon final 和 VOC2007 test 都已被开发过程访问，正文统一称为事后完整评测，不称为新的 hidden final。
2. 未完成的 paired/bootstrap、统一在线矩阵、能耗与网络退化实验保留 `【待补：...】`，不使用推测值。
3. RK3588 结果严格区分两阶段全局调度 HIL 与零前视因果在线 HIL。
4. 当前 OQR 并未在三个数据集、三个预算上一致优于 OQ，因此主张对象是 CAVS 框架，OQ/OQR 是动作空间实例。

