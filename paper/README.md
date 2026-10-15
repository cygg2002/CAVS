# Paper Workspace

本目录用于整理端云协同目标检测小论文。

当前文件：

- `TARGET_JOURNALS_ZH.md`：投稿方向、目标期刊、二类可行性判断。
- `EDGE_CLOUD_DETECTION_PAPER_DRAFT_ZH.md`：中文论文初稿模板。
- `EXPERIMENT_TODO_ZH.md`：边写边补的实验清单。
- `JRTIP_SUBMISSION_ROADMAP_ZH.md`：第一目标 JRTIP 的完成和投稿路线。
- `PAPER_POSITIONING_AND_TASK_PLAN_ZH.md`：论文定位、核心故事线、贡献证据和任务优先级。
- `JRTIP_MANUSCRIPT_DRAFT_ZH.md`：面向 JRTIP 的中文主稿 v0.1，非上板实验已汇总，T41 链路待补。
- `JRTIP_MANUSCRIPT_DRAFT_V0_2_ZH.md`：面向 JRTIP 的中文主稿 v0.2，已纳入 T41 真实板端实验。
- `JRTIP_MANUSCRIPT_DRAFT_V0_3_ZH.md`：当前优先使用的中文可投稿初稿，已补图表清单、系统图、流程图、E16/E21 表格、图像引用、局限性和参考文献骨架。
- `EXPERIMENT_EXECUTION_PLAN_ZH.md`：补充实验的执行级计划，包含目的、步骤、输出和成功标准。

实验目录约定：

- `experiments/E0_coco_main_results/`：COCO full 主结果整理。
- `experiments/E4_cost_ablation/`：成本项消融实验。
- `experiments/E5_roi_baseline/`：普通 ROI+云复核 baseline。
- `experiments/E7_cloud_imgsz_ablation/`：云端输入尺寸消融。
- `experiments/E8_frame_roi_ablation/`：帧级/ROI 级分支消融。
- `experiments/E9_jpeg_quality_ablation/`：上传 JPEG 质量消融。
- `experiments/E10_network_condition_ablation/`：弱网条件模拟。
- `experiments/E11_queue_scheduling_ablation/`：队列调度与实时 deadline 模拟。
- `experiments/E12_difficult_subset_analysis/`：困难子集定量分析。
- `experiments/E13_visual_case_study/`：定性可视化案例。
- `experiments/E14_uploader_selectivity_control/`：上传选择性与未上传负对照实验。
- `experiments/E15_metric_fairness_check/`：official mAP 与部署阈值 mAP 的公平性检查。
- `experiments/E16_paper_result_synthesis/`：论文总表、总图、实验章节和实验遗漏审查。
- `experiments/E17_t41_board_experiments/`：T41 上板实验完整规划，包括端侧、云端链路、端云协同和结果回传要求。

当前建议路线：

```text
第一目标：Journal of Real-Time Image Processing
第二目标：Machine Vision and Applications
第三目标：The Journal of Supercomputing
备选：IEEE Access / Sensors / 国内中文核心或科技核心
```

写作原则：

```text
实事求是；
有理论依据；
依据数据说话；
不夸大 T41 当前部署能力；
边写论文，边补 T41 真实链路实验。
```
