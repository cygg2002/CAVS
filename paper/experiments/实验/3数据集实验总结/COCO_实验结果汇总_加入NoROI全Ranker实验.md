# COCO 数据集实验结果汇总（含 No-ROI 全 Ranker、ActionValue 与 AFPD）

> 最后更新：2026-08-10  
> 阅读提示：第 2–19 节保留原三动作与 No-ROI 全 Ranker 的完整历史结果；第 20 节起补充后续 DualHead、ActionValue、跨动作校准、AFPD/Occupancy 和 21 组 paired bootstrap。不同证据层不能直接混合排名。

## 1. 实验范围与指标口径

本文档汇总截至 2026-08-10 的 COCO 主要实验，包括原三动作实验、No-ROI 全 Ranker 扩展、DualHead/ActionValue 学习器进化、多项决定性诊断，以及 AFPD/Occupancy 独立重训和完整 val2017 评测。各实验使用的预算语义、数据划分与证据等级不完全相同，必须分别解释，不能直接把同为“10%、25%、40%”的结果视为相同约束。

### 1.1 原三动作实验

原三动作实验采用以下口径：

- 数据集：COCO final，共 5,000 张图像；
- 主评价指标：Official mAP50–95；
- 同时记录：Official mAP50、Official mAP75；
- 主预算模式：`fixed_call`；
- 固定调用预算：10%、25%、40%；
- 通信成本指标：`Request KB/image`；
- 内部排序方法：9 个 Ranker；
- 外部学习器：ORIC-MLP、MORIC-MLP；
- 动作策略：
  - SharedAction；
  - CostEligible-AllROI；
  - GBDT-Hierarchical；
- 外部方法另外保留严格 Original 整图上传结果。

原三动作实验中的 10%、25%、40% 表示被选中上云的图像比例。

### 1.2 No-ROI canonical 全 Ranker 扩展实验

新增实验采用以下口径：

- Ranker fit：9,678 张 COCO train2017 图像；
- Calibration：571 张互斥的 train2017 图像；
- Evaluation：已被使用过的 5,000 张 COCO val2017 图像；
- 状态：`post-hoc canonical val2017 No-ROI completeness extension`；
- 方法数量：16 个；
- 动作策略：
  - Original-only；
  - SharedAction-NoROI；
- ROI 被硬禁用，所有结果的 `roi_ratio = 0`；
- 主要预算模式：`fixed_byte`；
- 次要预算模式：`fixed_call`；
- 字节预算：全 Original val2017 总字节的 10%、25%、40%；
- fixed-byte 的 λ 和比较器均只由 calibration 冻结；
- val2017 不参与训练、λ 选择或比较器选择。

新增实验中的 10%、25%、40% 在 `fixed_byte` 下表示通信字节容量，而不是固定上传图像比例。

### 1.3 表格记号

下列表格中，`AP / KB` 表示：

`Official mAP50–95 / Request KB·image⁻¹`

### 1.4 动作缩写

- `OQ`：隐含 Edge 不上传动作，上传动作为 Original 和 Q90；
- `OQR`：隐含 Edge，上传动作为 Original、Q90 和 ROI；
- `Raw ActionValue-OQ`：对每个“图片—动作”联合估值，由外部 exact-capacity scheduler 根据真实字节选择动作；
- `AFPD`：Action-Frontier Path Distillation，学习完整 `reward - λ × bytes` 动作上包络及切换路径；
- `OccupancyFree/Aware`：AFPD 之上的 ROI frontier occupancy gate 消融。

### 1.5 证据等级

| 证据层 | 用途 | 能否当作新的隐藏 final |
| --- | --- | --- |
| fit 5-fold OOF | 训练目标、表征和动作机制诊断 | 否 |
| calibration=571 | 一次性冻结 gate/阈值，不用于反复调参 | 否 |
| registered val2017 访问 | 某一冻结协议的论文级检查 | 仅对当时预注册协议有效 |
| post-hoc val2017 reuse | 完整性、归因与探索比较 | 否；val2017 已被历史实验消费 |
| GT-aware Teacher/Oracle | 不可部署上界与瓶颈分解 | 否 |


## 2. COCO 系统基线

| 系统基线 | mAP50–95 | mAP50 | mAP75 | Request KB/image | 动作构成 |
| --- | ---: | ---: | ---: | ---: | --- |
| Edge-only | 0.4079 | 0.5684 | 0.4426 | 0.00 | 不上云 |
| Cloud-only Original | **0.5742** | 0.7497 | 0.6272 | 159.12 | 100% 原图 |
| Always 640-Q90 | 0.5699 | 0.7457 | 0.6221 | 84.08 | 99.36% Q90 |
| Always ROI fallback | 0.5594 | 0.7320 | 0.6119 | 71.16 | 70.52% Q90 + 29.08% ROI |
| Shared ContextM2E all | 0.5732 | 0.7486 | 0.6266 | 134.83 | 63.18% 原图 + 32.70% Q90 + 4.12% ROI |

其中两个最重要的系统基准为：

- Edge-only：mAP50–95 = **0.4079**
- Cloud-only Original：mAP50–95 = **0.5742**

---

## 3. SharedAction：所有 Ranker 结果

SharedAction 使用同一个冻结动作头，因此该设置主要用于比较不同 Ranker 对“哪些图像值得上云”的排序能力。

### 3.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4325 / 14.51 | 0.4742 / 38.37 | 0.5057 / 62.03 |
| 内部 | ClassGainPrior | 0.4336 / 14.79 | 0.4711 / 35.37 | 0.4971 / 57.72 |
| 内部 | M2E | 0.4372 / 14.86 | 0.4728 / 37.55 | 0.5034 / 59.41 |
| 内部 | Confidence-Residual M2E | 0.4365 / 14.76 | 0.4732 / 37.39 | 0.5030 / 59.04 |
| 内部 | SemanticRank | 0.4438 / 16.29 | 0.4836 / 39.95 | 0.5142 / 62.64 |
| 内部 | MORIC-style | 0.4363 / 15.95 | 0.4721 / 39.59 | 0.5012 / 61.92 |
| 内部 | GEC-style | 0.4419 / 14.76 | 0.4756 / 37.69 | 0.5080 / 60.91 |
| 内部 | Current-Controlled-32D | 0.4457 / 16.76 | 0.4852 / 38.92 | 0.5147 / 60.93 |
| 内部 | Random | 0.4245 / 13.50 | 0.4490 / 33.77 | 0.4742 / 54.06 |
| 外部 | **ORIC-MLP** | **0.4491 / 16.07** | 0.4860 / 38.44 | 0.5111 / 60.25 |
| 外部 | **MORIC-MLP** | 0.4472 / 16.36 | **0.4890 / 39.32** | **0.5187 / 61.13** |

### 3.2 SharedAction 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 7.48% | 2.50% | 0.02% |
| ClassGainPrior | 8.30% | 1.64% | 0.06% |
| M2E | 8.20% | 1.80% | 0.00% |
| Confidence-Residual M2E | 8.16% | 1.84% | 0.00% |
| SemanticRank | 9.78% | 0.22% | 0.00% |
| MORIC-style | 9.18% | 0.80% | 0.02% |
| GEC-style | 8.68% | 1.32% | 0.00% |
| Current-Controlled-32D | 9.64% | 0.36% | 0.00% |
| ORIC-MLP | 8.94% | 1.02% | 0.04% |
| MORIC-MLP | 9.38% | 0.62% | 0.00% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 20.96% | 4.02% | 0.02% |
| ClassGainPrior | 19.16% | 5.78% | 0.06% |
| M2E | 20.76% | 4.24% | 0.00% |
| Confidence-Residual M2E | 20.60% | 4.40% | 0.00% |
| SemanticRank | 23.58% | 1.42% | 0.00% |
| MORIC-style | 22.34% | 2.56% | 0.10% |
| GEC-style | 21.44% | 3.56% | 0.00% |
| Current-Controlled-32D | 22.78% | 2.22% | 0.00% |
| ORIC-MLP | 20.92% | 3.84% | 0.24% |
| MORIC-MLP | 21.78% | 3.08% | 0.14% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 34.36% | 5.62% | 0.02% |
| ClassGainPrior | 30.50% | 9.24% | 0.26% |
| M2E | 32.96% | 6.96% | 0.08% |
| Confidence-Residual M2E | 32.64% | 7.26% | 0.10% |
| SemanticRank | 36.46% | 3.44% | 0.10% |
| MORIC-style | 33.92% | 5.94% | 0.14% |
| GEC-style | 34.34% | 5.66% | 0.00% |
| Current-Controlled-32D | 34.86% | 5.06% | 0.08% |
| ORIC-MLP | 32.04% | 7.50% | 0.46% |
| MORIC-MLP | 33.44% | 6.32% | 0.24% |

### 3.3 SharedAction 结果总结

- 10% 预算下，ORIC-MLP 的 mAP50–95 最高，为 **0.4491**。
- 25% 预算下，MORIC-MLP 的 mAP50–95 最高，为 **0.4890**。
- 40% 预算下，MORIC-MLP 的 mAP50–95 最高，为 **0.5187**。
- 内部方法中，Current-Controlled-32D 在三个预算点均表现最强：
  - 10%：0.4457；
  - 25%：0.4852；
  - 40%：0.5147。
- SemanticRank 是内部方法中另一组较强结果：
  - 10%：0.4438；
  - 25%：0.4836；
  - 40%：0.5142。
- Random 明显低于主要学习型 Ranker，说明排序策略能够有效提升固定预算下的上云选择质量。

---

## 4. CostEligible-AllROI：所有 Ranker 结果

CostEligible-AllROI 将满足成本条件的候选尽量改为 ROI 上传，因此能够显著降低 Request KB/image，但在 COCO 上通常伴随一定 AP 损失。

### 4.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4312 / 13.09 | 0.4711 / 33.58 | 0.5007 / 53.66 |
| 内部 | ClassGainPrior | 0.4327 / 12.85 | 0.4676 / 31.93 | 0.4925 / 51.00 |
| 内部 | M2E | 0.4314 / 9.94 | 0.4645 / 27.56 | 0.4939 / 46.04 |
| 内部 | Confidence-Residual M2E | 0.4307 / **9.83** | 0.4648 / 27.70 | 0.4931 / **45.58** |
| 内部 | SemanticRank | 0.4404 / 12.12 | 0.4772 / 31.73 | 0.5056 / 50.75 |
| 内部 | MORIC-style | 0.4350 / 11.86 | 0.4676 / 31.05 | 0.4958 / 49.41 |
| 内部 | GEC-style | 0.4380 / 12.10 | 0.4688 / 30.84 | 0.4991 / 49.07 |
| 内部 | Current-Controlled-32D | 0.4438 / 15.02 | 0.4786 / 33.49 | 0.5062 / 50.99 |
| 内部 | Random | 0.4234 / 10.89 | 0.4464 / 27.25 | 0.4697 / 43.60 |
| 外部 | **ORIC-MLP** | **0.4480 / 14.82** | 0.4826 / 34.20 | 0.5059 / 51.76 |
| 外部 | **MORIC-MLP** | 0.4458 / 15.12 | **0.4855 / 34.94** | **0.5129 / 53.24** |

### 4.2 AllROI 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 6.16% | 2.46% | 1.38% |
| ClassGainPrior | 6.66% | 1.48% | 1.86% |
| M2E | 4.56% | 1.50% | 3.94% |
| Confidence-Residual M2E | 4.54% | 1.52% | 3.94% |
| SemanticRank | 6.48% | 0.14% | 3.38% |
| MORIC-style | 5.88% | 0.78% | 3.34% |
| GEC-style | 6.34% | 1.26% | 2.40% |
| Current-Controlled-32D | 8.30% | 0.36% | 1.34% |
| ORIC-MLP | 8.00% | 0.82% | 1.18% |
| MORIC-MLP | 8.38% | 0.54% | 1.08% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 16.86% | 3.92% | 4.22% |
| ClassGainPrior | 16.32% | 5.20% | 3.48% |
| M2E | 13.22% | 3.52% | 8.26% |
| Confidence-Residual M2E | 13.24% | 3.74% | 8.02% |
| SemanticRank | 16.88% | 1.22% | 6.90% |
| MORIC-style | 15.46% | 2.48% | 7.06% |
| GEC-style | 15.72% | 3.32% | 5.96% |
| Current-Controlled-32D | 18.26% | 1.90% | 4.84% |
| ORIC-MLP | 17.56% | 3.32% | 4.12% |
| MORIC-MLP | 18.30% | 2.68% | 4.02% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 27.38% | 5.46% | 7.16% |
| ClassGainPrior | 25.10% | 8.24% | 6.66% |
| M2E | 22.64% | 5.74% | 11.62% |
| Confidence-Residual M2E | 22.26% | 5.98% | 11.76% |
| SemanticRank | 26.74% | 3.00% | 10.26% |
| MORIC-style | 24.06% | 5.54% | 10.40% |
| GEC-style | 24.88% | 5.12% | 10.00% |
| Current-Controlled-32D | 26.92% | 4.08% | 9.00% |
| ORIC-MLP | 25.56% | 6.26% | 8.18% |
| MORIC-MLP | 27.16% | 5.44% | 7.40% |

### 4.3 CostEligible-AllROI 结果总结

- 10% 预算下，ORIC-MLP 最优，mAP50–95 = **0.4480**。
- 25% 预算下，MORIC-MLP 最优，mAP50–95 = **0.4855**。
- 40% 预算下，MORIC-MLP 最优，mAP50–95 = **0.5129**。
- 内部 Current-Controlled-32D 仍保持较强精度：
  - 10%：0.4438；
  - 25%：0.4786；
  - 40%：0.5062。
- CostEligible-AllROI 能够明显增加 ROI 使用比例并降低通信量。
- 以内部 M2E 和 Confidence-Residual M2E 为例，40% 预算时 ROI 比例分别达到 11.62% 和 11.76%，Request KB/image 分别降低到 46.04 和 45.58。
- 与 SharedAction 相比，COCO 上 AllROI 虽然降低通信量，但通常伴随可观察的 AP 下降，因此更偏向“通信成本优先”工作点。

---

## 5. GBDT-Hierarchical：所有 Ranker 结果

GBDT-Hierarchical 只将预测为安全的候选切换为 ROI，因此相较 CostEligible-AllROI 更保守。

### 5.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4324 / 14.37 | 0.4731 / 37.54 | 0.5041 / 60.39 |
| 内部 | ClassGainPrior | 0.4335 / 14.38 | 0.4697 / 34.70 | 0.4955 / 56.62 |
| 内部 | M2E | 0.4370 / 14.66 | 0.4712 / 36.35 | 0.5017 / 57.27 |
| 内部 | Confidence-Residual M2E | 0.4363 / 14.60 | 0.4717 / 36.40 | 0.5010 / 56.93 |
| 内部 | SemanticRank | 0.4435 / 16.13 | 0.4820 / 39.14 | 0.5123 / 60.71 |
| 内部 | MORIC-style | 0.4363 / 15.54 | 0.4705 / 38.29 | 0.4993 / 59.53 |
| 内部 | GEC-style | 0.4408 / 14.62 | 0.4743 / 36.90 | 0.5063 / 59.03 |
| 内部 | Current-Controlled-32D | 0.4457 / 16.74 | 0.4834 / 38.42 | 0.5130 / 59.70 |
| 内部 | Random | 0.4242 / 13.04 | 0.4484 / 32.62 | 0.4731 / 52.19 |
| 外部 | **ORIC-MLP** | **0.4488 / 15.84** | 0.4847 / 37.81 | 0.5096 / 58.96 |
| 外部 | **MORIC-MLP** | 0.4471 / 16.28 | **0.4886 / 38.89** | **0.5172 / 60.01** |

### 5.2 GBDT 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 7.38% | 2.50% | 0.12% |
| ClassGainPrior | 7.98% | 1.54% | 0.48% |
| M2E | 8.08% | 1.76% | 0.16% |
| Confidence-Residual M2E | 8.06% | 1.80% | 0.14% |
| SemanticRank | 9.68% | 0.20% | 0.12% |
| MORIC-style | 8.86% | 0.80% | 0.34% |
| GEC-style | 8.58% | 1.28% | 0.14% |
| Current-Controlled-32D | 9.62% | 0.36% | 0.02% |
| ORIC-MLP | 8.76% | 0.94% | 0.30% |
| MORIC-MLP | 9.30% | 0.60% | 0.10% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 20.30% | 4.00% | 0.70% |
| ClassGainPrior | 18.60% | 5.62% | 0.78% |
| M2E | 19.86% | 4.10% | 1.04% |
| Confidence-Residual M2E | 19.90% | 4.26% | 0.84% |
| SemanticRank | 22.90% | 1.38% | 0.72% |
| MORIC-style | 21.28% | 2.54% | 1.18% |
| GEC-style | 20.80% | 3.50% | 0.70% |
| Current-Controlled-32D | 22.36% | 2.20% | 0.44% |
| ORIC-MLP | 20.40% | 3.70% | 0.90% |
| MORIC-MLP | 21.44% | 3.00% | 0.56% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 33.00% | 5.58% | 1.42% |
| ClassGainPrior | 29.60% | 8.98% | 1.42% |
| M2E | 31.38% | 6.64% | 1.98% |
| Confidence-Residual M2E | 31.08% | 6.92% | 2.00% |
| SemanticRank | 34.86% | 3.30% | 1.84% |
| MORIC-style | 32.04% | 5.82% | 2.14% |
| GEC-style | 32.86% | 5.54% | 1.60% |
| Current-Controlled-32D | 33.90% | 4.84% | 1.26% |
| ORIC-MLP | 31.04% | 7.12% | 1.84% |
| MORIC-MLP | 32.60% | 6.10% | 1.30% |

### 5.3 GBDT-Hierarchical 结果总结

- 10% 预算下，ORIC-MLP 最优，mAP50–95 = **0.4488**。
- 25% 预算下，MORIC-MLP 最优，mAP50–95 = **0.4886**。
- 40% 预算下，MORIC-MLP 最优，mAP50–95 = **0.5172**。
- 内部 Current-Controlled-32D 仍是内部 Ranker 中的最佳或接近最佳方案：
  - 10%：0.4457；
  - 25%：0.4834；
  - 40%：0.5130。
- 相比 CostEligible-AllROI，GBDT-Hierarchical 的 ROI 使用比例明显更低，因此通信节省较小。
- 但 GBDT-Hierarchical 的 AP 更接近 SharedAction，说明其属于更保守的 ROI 使用策略。

---

## 6. 外部方法严格 Original 结果

该组实验中，被选中的图像全部执行原始整图上传，不使用 SharedAction、CostEligible-AllROI 或 GBDT-Hierarchical。

### 6.1 固定调用预算结果

| 外部方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| DCSB-Score-Adapted | 0.4467 / 17.80 | 0.4820 / 42.59 | 0.5088 / 66.24 |
| ORIC-MLP | **0.4493 / 16.57** | 0.4863 / 40.60 | 0.5116 / 64.59 |
| MORIC-MLP | 0.4472 / 16.61 | **0.4893 / 40.76** | **0.5191 / 64.25** |

### 6.2 DCSB-Native 自然工作点

DCSB-Native 不采用固定 10%、25%、40% 预算，而由原生规则自然产生上云比例。

| Payload | 实际上传率 | mAP50–95 | Request KB/image |
| --- | ---: | ---: | ---: |
| Original | 73.18% | 0.5566 | 116.79 |
| SharedAction | 73.18% | 0.5561 | 108.39 |

---

## 7. 不同动作策略下的最优结果

### 7.1 10% 固定调用预算

| 动作策略 | 第一名 | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | ORIC-MLP | **0.4493** | 16.57 |
| SharedAction | ORIC-MLP | **0.4491** | 16.07 |
| CostEligible-AllROI | ORIC-MLP | **0.4480** | 14.82 |
| GBDT-Hierarchical | ORIC-MLP | **0.4488** | 15.84 |

10% 预算下，四种载荷口径中均由 ORIC-MLP 获得最高 mAP50–95。

### 7.2 25% 固定调用预算

| 动作策略 | 第一名 | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | MORIC-MLP | **0.4893** | 40.76 |
| SharedAction | MORIC-MLP | **0.4890** | 39.32 |
| CostEligible-AllROI | MORIC-MLP | **0.4855** | 34.94 |
| GBDT-Hierarchical | MORIC-MLP | **0.4886** | 38.89 |

25% 预算下，四种载荷口径中均由 MORIC-MLP 获得最高 mAP50–95。

### 7.3 40% 固定调用预算

| 动作策略 | 第一名 | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | MORIC-MLP | **0.5191** | 64.25 |
| SharedAction | MORIC-MLP | **0.5187** | 61.13 |
| CostEligible-AllROI | MORIC-MLP | **0.5129** | 53.24 |
| GBDT-Hierarchical | MORIC-MLP | **0.5172** | 60.01 |

40% 预算下，四种载荷口径中同样由 MORIC-MLP 获得最高 mAP50–95。

---

## 8. 原三动作阶段的 COCO 结果分析

### 8.1 外部 ORIC/MORIC 的排序优势较稳定

COCO 上的排序结果具有较强一致性：

- 10% 预算：ORIC-MLP 最优；
- 25% 预算：MORIC-MLP 最优；
- 40% 预算：MORIC-MLP 最优。

这一规律在 Original、SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical 四种载荷策略下均保持不变。

说明在 COCO 数据集上，外部 ORIC/MORIC 学习器对高价值上云样本的排序能力整体优于当前内部 Ranker。

### 8.2 Current-Controlled-32D 是原三动作阶段最强内部 Ranker

在 SharedAction 下：

- 10%：0.4457；
- 25%：0.4852；
- 40%：0.5147。

在 GBDT-Hierarchical 下：

- 10%：0.4457；
- 25%：0.4834；
- 40%：0.5130。

在 CostEligible-AllROI 下：

- 10%：0.4438；
- 25%：0.4786；
- 40%：0.5062。

因此，Current-Controlled-32D 可以作为原三动作阶段内部 Ranker 的主要代表方法。

SemanticRank 也表现较强，特别是在中高预算下接近 Current-Controlled-32D。

### 8.3 SharedAction 的精度表现最稳定

与另外两种 ROI 扩展动作相比，SharedAction 的 AP 通常最高或接近最高。

以 MORIC-MLP 为例：

| 预算 | SharedAction | AllROI | GBDT |
| --- | ---: | ---: | ---: |
| 10% | 0.4472 | 0.4458 | 0.4471 |
| 25% | 0.4890 | 0.4855 | 0.4886 |
| 40% | 0.5187 | 0.5129 | 0.5172 |

说明 SharedAction 在 COCO 上能够较好地保留检测精度。

### 8.4 CostEligible-AllROI 更偏向通信成本优化

CostEligible-AllROI 能明显降低 Request KB/image，但 AP 通常低于 SharedAction。

以 MORIC-MLP 为例：

| 预算 | SharedAction AP / KB | AllROI AP / KB |
| --- | ---: | ---: |
| 10% | 0.4472 / 16.36 | 0.4458 / 15.12 |
| 25% | 0.4890 / 39.32 | 0.4855 / 34.94 |
| 40% | 0.5187 / 61.13 | 0.5129 / 53.24 |

因此 AllROI 更适合作为低通信成本工作点，而不是精度优先方案。

### 8.5 GBDT-Hierarchical 提供更保守的折中

GBDT-Hierarchical 的通信成本通常略低于 SharedAction，同时 AP 损失明显小于 AllROI。

仍以 MORIC-MLP 为例：

| 预算 | SharedAction AP / KB | GBDT AP / KB |
| --- | ---: | ---: |
| 10% | 0.4472 / 16.36 | 0.4471 / 16.28 |
| 25% | 0.4890 / 39.32 | 0.4886 / 38.89 |
| 40% | 0.5187 / 61.13 | 0.5172 / 60.01 |

因此，在 COCO 上，GBDT-Hierarchical 比 CostEligible-AllROI 更偏向精度保持。

### 8.6 与 Edge-only 和 Cloud-only 的关系

Edge-only 的 mAP50–95 为 0.4079，而固定预算方法在 10%、25%、40% 三个预算点都可以持续提高检测精度。

以严格 Original 最佳结果为例：

| 预算 | 最佳方法 | mAP50–95 | 相比 Edge-only 提升 |
| --- | --- | ---: | ---: |
| 10% | ORIC-MLP | 0.4493 | +0.0414 |
| 25% | MORIC-MLP | 0.4893 | +0.0814 |
| 40% | MORIC-MLP | 0.5191 | +0.1112 |

但是，即使在 40% 预算下，当前最佳 mAP50–95 = 0.5191，仍低于 Cloud-only Original 的 0.5742。

这说明 COCO 上当前方法形成的是从 Edge-only 向 Cloud-only 逐步逼近的精度—通信成本折中，而不是在固定低上传预算下完全达到 Cloud-only 精度。

---

## 9. 原三动作实验阶段性结论

1. **外部 ORIC/MORIC 在 COCO 上具有稳定的排序优势。**
   - 10% 预算由 ORIC-MLP 最优；
   - 25% 和 40% 预算由 MORIC-MLP 最优。

2. **Current-Controlled-32D 是原三动作阶段最强的内部 Ranker。**
   - 在三个预算点和三种动作策略下均具有较强稳定性。

3. **SemanticRank 是第二梯队中较强的内部方法。**

4. **SharedAction 的精度表现最稳定。**
   - 适合用于公平比较 Ranker 本身的排序能力。

5. **CostEligible-AllROI 能够显著降低通信量，但在 COCO 上通常伴随 AP 损失。**

6. **GBDT-Hierarchical 比 AllROI 更保守。**
   - 通信节省有限；
   - 但 AP 更接近 SharedAction。

7. **固定预算增加会持续改善 COCO 检测精度。**
   - 10% 最佳 AP：0.4493；
   - 25% 最佳 AP：0.4893；
   - 40% 最佳 AP：0.5191。

8. **当前固定预算方案仍未达到 Cloud-only Original 的 0.5742 mAP50–95。**

9. **COCO 上最清晰的实验结论不是“内部方法全面超过外部基线”，而是不同 Ranker 和动作策略形成了可解释的精度—通信成本折中。**

---

## 10. 原三动作实验使用建议

如果用于论文正文，建议优先保留以下结果。

### 10.1 Ranker 公平比较

采用：

- SharedAction；
- fixed_call；
- 10%、25%、40%；
- 内部 Ranker；
- ORIC-MLP；
- MORIC-MLP；
- Random。

该设置能够最大程度隔离 Ranker 本身的排序能力。

### 10.2 外部方法基线

单独报告：

- DCSB-Score-Adapted + Original；
- ORIC-MLP + Original；
- MORIC-MLP + Original；
- DCSB-Native 自然工作点。

### 10.3 动作策略消融

重点比较：

- SharedAction；
- CostEligible-AllROI；
- GBDT-Hierarchical。

建议至少选取以下代表 Ranker：

- Current-Controlled-32D；
- SemanticRank；
- ORIC-MLP；
- MORIC-MLP；
- Random。

这样既能展示内部方法与外部方法的差异，也能清晰分析 ROI 动作策略对通信成本和检测精度的影响。

---

## 11. No-ROI canonical 全 Ranker 扩展实验

### 11.1 实验目的

该实验回答的问题是：

> 在 canonical COCO 划分上，移除 ROI 动作后，冻结的 DualHead 是否仍能超过外部 ORIC/MORIC 和历史内部 Ranker？

与原三动作实验相比，该实验的主要改进包括：

1. ROI 被完全移除，不再让 ROI 裁剪质量混入 Ranker 对比；
2. 同时纳入 9 个历史内部 Ranker、3 个新内部 Ranker和 4 个外部 ORIC/MORIC 变体；
3. 以 `fixed_byte` 作为主要预算口径，更贴近真实通信容量限制；
4. fixed-byte λ 和比较器只使用 calibration 冻结；
5. val2017 不参与训练或选择。

### 11.2 方法矩阵

#### 历史内部 Ranker

- Random
- Confidence
- ClassGainPrior
- M2E
- Confidence-Residual M2E
- SemanticRank
- MORIC-style
- GEC-style
- Current-Controlled-32D

#### 新内部 Ranker

- Current-BudgetSet
- SetValue
- DualHead

#### 外部 Ranker

- ORIC-AP50
- MORIC-AP50
- ORIC-AP5095
- MORIC-AP5095

共计：

`16 个方法 × 2 个动作策略 × 2 种预算模式 × 3 个预算 = 192 行结果`

### 11.3 协议完整性

- final 结果：192 行；
- calibration 结果：192 行；
- 唯一结果键：192 个；
- 重复结果：0；
- 所有 `roi_ratio = 0`；
- fixed-byte 超预算：0；
- fixed-byte 容量基本全部利用；
- 所有学习器复用 fit=9,678 的冻结 checkpoint；
- calibration 和 val2017 均未用于重新训练 Ranker；
- 该实验属于 post-hoc completeness extension，不是新的隐藏 final。

---

## 12. Primary fixed-byte 全部结果

### 12.1 Original-only

被选中的图像全部上传 Original，未选中的图像保留 Edge 结果。

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Random | 0.434200 / 15.91 | 0.468143 / 39.78 | 0.497769 / 63.64 |
| 内部 | Confidence | 0.443833 / 15.90 | 0.483089 / 39.78 | 0.502723 / 63.64 |
| 内部 | ClassGainPrior | 0.437961 / 15.91 | 0.477248 / 39.78 | 0.507977 / 63.64 |
| 内部 | M2E | 0.444339 / 15.91 | 0.482543 / 39.78 | 0.510151 / 63.64 |
| 内部 | Confidence-Residual M2E | 0.444140 / 15.91 | 0.482018 / 39.78 | 0.509525 / 63.65 |
| 内部 | SemanticRank | 0.447894 / 15.91 | 0.488121 / 39.78 | 0.518768 / 63.65 |
| 内部 | MORIC-style | 0.443708 / 15.91 | 0.480348 / 39.78 | 0.510240 / 63.65 |
| 内部 | GEC-style | 0.444737 / 15.91 | 0.485820 / 39.78 | 0.512922 / 63.64 |
| 内部 | Current-Controlled-32D | 0.446960 / 15.91 | 0.490696 / 39.78 | 0.516251 / 63.65 |
| 内部 | Current-BudgetSet | 0.441561 / 15.91 | 0.489484 / 39.78 | 0.522171 / 63.65 |
| 内部 | SetValue | 0.442252 / 15.90 | 0.490240 / 39.78 | 0.523473 / 63.65 |
| 内部 | DualHead | 0.446943 / 15.90 | 0.491117 / 39.78 | 0.523762 / 63.65 |
| 外部 | ORIC-AP50 | 0.445398 / 15.91 | 0.486944 / 39.78 | 0.514394 / 63.65 |
| 外部 | MORIC-AP50 | 0.441302 / 15.91 | 0.482221 / 39.78 | 0.517500 / 63.64 |
| 外部 | ORIC-AP5095 | 0.439991 / 15.90 | 0.487081 / 39.77 | 0.514891 / 63.64 |
| 外部 | MORIC-AP5095 | 0.446326 / 15.91 | 0.489531 / 39.78 | 0.517684 / 63.64 |

#### Original-only 结果要点

- 10% 字节预算：
  - 全局最高为 SemanticRank：**0.447894**；
  - DualHead：**0.446943**；
  - 二者相差 **0.000952**。
- 25% 字节预算：
  - DualHead 全局最高：**0.491117**。
- 40% 字节预算：
  - DualHead 全局最高：**0.523762**；
  - SetValue：0.523473；
  - Current-BudgetSet：0.522171；
  - MORIC-AP5095：0.517684。

DualHead 在 Original-only 下的优势随字节预算增大而增强，特别是在 40% 字节预算下，对外部 Ranker 和多数历史内部 Ranker 的优势最明显。

### 12.2 SharedAction-NoROI

被选中的图像只能在 Original 和 Q90 之间选择，ROI 被硬禁用。

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Random | 0.432610 / 15.90 | 0.472373 / 39.77 | 0.503358 / 63.63 |
| 内部 | Confidence | 0.438580 / 15.90 | 0.481590 / 39.78 | 0.511007 / 63.64 |
| 内部 | ClassGainPrior | 0.436229 / 15.91 | 0.476991 / 39.77 | 0.510342 / 63.63 |
| 内部 | M2E | 0.445237 / 15.91 | 0.483961 / 39.78 | 0.513311 / 63.65 |
| 内部 | Confidence-Residual M2E | 0.444302 / 15.91 | 0.484466 / 39.78 | 0.513229 / 63.65 |
| 内部 | SemanticRank | 0.438249 / 15.91 | 0.491572 / 39.78 | 0.519520 / 63.65 |
| 内部 | MORIC-style | 0.437013 / 15.91 | 0.476011 / 39.78 | 0.511421 / 63.64 |
| 内部 | GEC-style | 0.437745 / 15.91 | 0.488243 / 39.78 | 0.515897 / 63.65 |
| 内部 | Current-Controlled-32D | 0.437443 / 15.90 | 0.489140 / 39.78 | 0.519502 / 63.65 |
| 内部 | Current-BudgetSet | 0.446646 / 15.91 | 0.493005 / 39.78 | 0.519143 / 63.65 |
| 内部 | SetValue | 0.445763 / 15.91 | 0.491608 / 39.78 | 0.521764 / 63.64 |
| 内部 | DualHead | 0.446715 / 15.91 | 0.492770 / 39.78 | 0.521186 / 63.64 |
| 外部 | ORIC-AP50 | 0.436378 / 15.91 | 0.490146 / 39.78 | 0.518286 / 63.65 |
| 外部 | MORIC-AP50 | 0.437698 / 15.91 | 0.483601 / 39.78 | 0.521262 / 63.64 |
| 外部 | ORIC-AP5095 | 0.436955 / 15.91 | 0.483229 / 39.77 | 0.517819 / 63.65 |
| 外部 | MORIC-AP5095 | 0.437333 / 15.90 | 0.493428 / 39.78 | 0.521645 / 63.65 |

#### SharedAction-NoROI 结果要点

- 10% 字节预算：
  - DualHead 全局最高：**0.446715**。
- 25% 字节预算：
  - MORIC-AP5095 全局最高：**0.493428**；
  - Current-BudgetSet：0.493005；
  - DualHead：0.492770；
  - DualHead 与最优只差 0.000658。
- 40% 字节预算：
  - SetValue 全局最高：**0.521764**；
  - MORIC-AP5095：0.521645；
  - MORIC-AP50：0.521262；
  - DualHead：0.521186；
  - DualHead 与最优只差 0.000578。

### 12.3 DualHead fixed-byte 动作构成

| 策略 | 字节预算 | mAP50–95 | Request KB/image | 实际上传率 | Original | Q90 | ROI | 选中图像数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original-only | 10% | 0.446943 | 15.90 | 18.16% | 18.16% | 0.00% | 0.00% | 908 |
| Original-only | 25% | 0.491117 | 39.78 | 35.00% | 35.00% | 0.00% | 0.00% | 1,750 |
| Original-only | 40% | 0.523762 | 63.65 | 46.92% | 46.92% | 0.00% | 0.00% | 2,346 |
| SharedAction-NoROI | 10% | 0.446715 | 15.91 | 23.22% | 8.08% | 15.14% | 0.00% | 1,161 |
| SharedAction-NoROI | 25% | 0.492770 | 39.78 | 41.34% | 22.84% | 18.50% | 0.00% | 2,067 |
| SharedAction-NoROI | 40% | 0.521186 | 63.64 | 60.82% | 34.02% | 26.80% | 0.00% | 3,041 |

相同字节预算下，Q90 的引入能够扩大实际上传覆盖率：

- 10%：18.16% → 23.22%；
- 25%：35.00% → 41.34%；
- 40%：46.92% → 60.82%。

但覆盖率增加并不必然提高 AP：

- 10%：两种策略基本相当；
- 25%：SharedAction-NoROI 更高；
- 40%：Original-only 更高。

这说明 Original 保真度与 Q90 扩大覆盖范围之间存在预算相关的最优平衡。

### 12.4 DualHead 与全局最优的关系

| 策略 | 字节预算 | DualHead AP | 全部方法实际最优 | 最优 AP | DualHead 与最优差值 |
| --- | ---: | ---: | --- | ---: | ---: |
| Original-only | 10% | 0.446943 | SemanticRank | 0.447894 | -0.000952 |
| Original-only | 25% | 0.491117 | DualHead | 0.491117 | +0.000000 |
| Original-only | 40% | 0.523762 | DualHead | 0.523762 | +0.000000 |
| SharedAction-NoROI | 10% | 0.446715 | DualHead | 0.446715 | +0.000000 |
| SharedAction-NoROI | 25% | 0.492770 | MORIC-AP5095 | 0.493428 | -0.000658 |
| SharedAction-NoROI | 40% | 0.521186 | SetValue | 0.521764 | -0.000578 |

DualHead 在六个 primary fixed-byte 工作点中：

- 3 个工作点获得全局第一；
- 其余 3 个工作点距离第一名均不足 0.001 AP；
- 最差差距为 −0.000952。

因此最准确的表述是：

> DualHead 在 fixed-byte 条件下具有很强的综合稳定性，在所有主工作点保持全局最优或近似最优。

不能表述为：

> DualHead 在所有策略和预算下均为第一。

---

## 13. Secondary fixed-call 全部结果

Fixed-call 固定上传图像数量，主要用于检验纯 top-K 排序能力。它与 primary fixed-byte 的预算语义不同。

### 13.1 Original-only fixed-call

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Random | 0.424343 / 16.01 | 0.450459 / 39.68 | 0.477682 / 63.50 |
| 内部 | Confidence | 0.432386 / 15.94 | 0.474227 / 40.42 | 0.505723 / 64.77 |
| 内部 | ClassGainPrior | 0.433566 / 15.59 | 0.471343 / 37.90 | 0.497322 / 62.52 |
| 内部 | M2E | 0.437363 / 15.74 | 0.472899 / 39.70 | 0.504046 / 62.90 |
| 内部 | Confidence-Residual M2E | 0.436582 / 15.66 | 0.473498 / 39.60 | 0.503567 / 62.66 |
| 内部 | SemanticRank | 0.443855 / 16.34 | 0.483576 / 40.52 | 0.514155 / 64.07 |
| 内部 | MORIC-style | 0.436355 / 16.44 | 0.472204 / 41.25 | 0.501470 / 65.47 |
| 内部 | GEC-style | 0.441844 / 15.42 | 0.475750 / 39.38 | 0.508225 / 63.44 |
| 内部 | Current-Controlled-32D | 0.445789 / 16.92 | 0.485517 / 39.83 | 0.515119 / 63.06 |
| 内部 | Current-BudgetSet | 0.442126 / 12.20 | 0.479398 / 32.19 | 0.514472 / 55.20 |
| 内部 | SetValue | 0.446274 / 15.50 | 0.485400 / 39.25 | 0.514858 / 63.00 |
| 内部 | DualHead | 0.446776 / 15.52 | 0.486132 / 38.84 | 0.514328 / 62.69 |
| 外部 | ORIC-AP50 | 0.443634 / 15.46 | 0.483037 / 39.96 | 0.512525 / 63.70 |
| 外部 | MORIC-AP50 | 0.446411 / 16.58 | 0.481178 / 40.39 | 0.510258 / 63.39 |
| 外部 | ORIC-AP5095 | 0.442595 / 16.54 | 0.479287 / 40.37 | 0.510346 / 63.47 |
| 外部 | MORIC-AP5095 | 0.448858 / 16.85 | 0.483423 / 40.14 | 0.512527 / 63.60 |

### 13.2 SharedAction-NoROI fixed-call

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Random | 0.424245 / 13.70 | 0.450143 / 34.28 | 0.477504 / 55.15 |
| 内部 | Confidence | 0.432433 / 14.54 | 0.474177 / 38.40 | 0.505651 / 62.05 |
| 内部 | ClassGainPrior | 0.433620 / 14.84 | 0.471116 / 35.42 | 0.497139 / 57.94 |
| 内部 | M2E | 0.437164 / 14.86 | 0.472771 / 37.55 | 0.503438 / 59.53 |
| 内部 | Confidence-Residual M2E | 0.436458 / 14.76 | 0.473245 / 37.39 | 0.503005 / 59.20 |
| 内部 | SemanticRank | 0.443825 / 16.29 | 0.483577 / 39.95 | 0.514115 / 62.76 |
| 内部 | MORIC-style | 0.436277 / 15.98 | 0.472067 / 39.72 | 0.501216 / 62.09 |
| 内部 | GEC-style | 0.441911 / 14.76 | 0.475635 / 37.69 | 0.507979 / 60.91 |
| 内部 | Current-Controlled-32D | 0.445670 / 16.76 | 0.485218 / 38.92 | 0.514650 / 60.98 |
| 内部 | Current-BudgetSet | 0.442001 / 11.70 | 0.478986 / 30.41 | 0.513875 / 51.76 |
| 内部 | SetValue | 0.446324 / 15.28 | 0.485132 / 37.84 | 0.514456 / 60.10 |
| 内部 | DualHead | 0.446783 / 15.21 | 0.485914 / 37.50 | 0.513813 / 59.53 |
| 外部 | ORIC-AP50 | 0.443585 / 15.01 | 0.482698 / 38.13 | 0.511956 / 59.98 |
| 外部 | MORIC-AP50 | 0.446498 / 16.43 | 0.481096 / 39.07 | 0.509683 / 60.35 |
| 外部 | ORIC-AP5095 | 0.442533 / 16.17 | 0.478933 / 38.68 | 0.509858 / 59.95 |
| 外部 | MORIC-AP5095 | 0.448805 / 16.63 | 0.483190 / 38.67 | 0.512261 / 60.32 |

### 13.3 Fixed-call 结果总结

- Original-only 10%：
  - MORIC-AP5095 = **0.448858**；
  - DualHead = 0.446776。
- Original-only 25%：
  - DualHead = **0.486132**，全局第一。
- Original-only 40%：
  - Current-Controlled-32D = **0.515119**；
  - SetValue = 0.514858；
  - DualHead = 0.514328。
- SharedAction-NoROI 10%：
  - MORIC-AP5095 = **0.448805**；
  - DualHead = 0.446783。
- SharedAction-NoROI 25%：
  - DualHead = **0.485914**，全局第一。
- SharedAction-NoROI 40%：
  - Current-Controlled-32D = **0.514650**；
  - SetValue = 0.514456；
  - DualHead = 0.513813。

因此，DualHead 的优势在 fixed-byte 下比 fixed-call 更明显。其方法定位更接近：

> 面向通信字节容量的预算感知 Ranker

而不是：

> 在所有固定 top-K 数量下都占优的通用 Ranker。

---

## 14. DualHead、Current-BudgetSet 与 SetValue 消融

### 14.1 Original-only fixed-byte

| 预算 | Current-BudgetSet | SetValue | DualHead |
| ---: | ---: | ---: | ---: |
| 10% | 0.441561 | 0.442252 | **0.446943** |
| 25% | 0.489484 | 0.490240 | **0.491117** |
| 40% | 0.522171 | 0.523473 | **0.523762** |

在 Original-only 下，DualHead 三个字节预算点均高于 Current-BudgetSet 和 SetValue。

### 14.2 SharedAction-NoROI fixed-byte

| 预算 | Current-BudgetSet | SetValue | DualHead |
| ---: | ---: | ---: | ---: |
| 10% | 0.446646 | 0.445763 | **0.446715** |
| 25% | **0.493005** | 0.491608 | 0.492770 |
| 40% | 0.519143 | **0.521764** | 0.521186 |

SharedAction-NoROI 下没有统一赢家：

- 10%：DualHead 与 Current-BudgetSet 几乎持平；
- 25%：Current-BudgetSet 略高；
- 40%：SetValue 略高。

因此该消融支持：

> DualHead 在 Original-only，尤其在低字节和中高字节预算下具有较稳定价值。

但不支持：

> DualHead 在所有动作策略下均优于其他新内部变体。

---

## 15. 外部 ORIC/MORIC 变体结果

本实验将外部 Ranker 细分为：

- ORIC-AP50；
- MORIC-AP50；
- ORIC-AP5095；
- MORIC-AP5095。

Primary fixed-byte Original-only 中，MORIC-AP5095 在三个预算下均是实际最强外部方法：

| 字节预算 | MORIC-AP5095 mAP50–95 |
| ---: | ---: |
| 10% | 0.446326 |
| 25% | 0.489531 |
| 40% | 0.517684 |

SharedAction-NoROI 中：

- 10%：MORIC-AP50 = 0.437698，略高于其他外部变体；
- 25%：MORIC-AP5095 = **0.493428**；
- 40%：MORIC-AP5095 = **0.521645**。

结果说明：

1. MORIC 的收益分位变换在 COCO 上整体比 ORIC 更稳定；
2. AP50–95 奖励目标在中高预算下更匹配最终 Official mAP50–95；
3. 外部 Ranker 的表现会受到载荷策略影响；
4. 外部 Original 收益学习器未必能直接适配 Original/Q90 混合载荷下的单位字节收益排序。

第 4 点属于由结果支持的机制推断，不是该实验直接验证的因果结论。

---

## 16. Paired Bootstrap 结果

实验对 DualHead 与 calibration 冻结的：

- 最强外部比较器；
- 最强历史内部比较器；
- SetValue；

进行了 1,000 次配对图像 Bootstrap。

| 策略 | 预算 | 比较对象 | 比较类型 | 均值 ΔAP | 95% CI | P(Δ>0) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Original-only | 10% | ORIC-AP50 | strongest_external | +0.001555 | [-0.000968, +0.004284] | 0.886 |
| Original-only | 10% | MORIC-style | strongest_historical | +0.003297 | [-0.000094, +0.006501] | 0.971 |
| Original-only | 10% | SetValue | SetValue | +0.004472 | [+0.002188, +0.006807] | 1.000 |
| Original-only | 25% | ORIC-AP5095 | strongest_external | +0.004186 | [-0.000501, +0.009653] | 0.957 |
| Original-only | 25% | Current-Controlled-32D | strongest_historical | +0.000485 | [-0.004337, +0.005551] | 0.549 |
| Original-only | 25% | SetValue | SetValue | +0.000922 | [-0.001971, +0.004753] | 0.680 |
| Original-only | 40% | ORIC-AP50 | strongest_external | +0.009483 | [+0.004499, +0.014873] | 1.000 |
| Original-only | 40% | SemanticRank | strongest_historical | +0.005058 | [+0.000331, +0.010156] | 0.982 |
| Original-only | 40% | SetValue | SetValue | +0.000399 | [-0.000964, +0.001685] | 0.717 |
| SharedAction-NoROI | 10% | MORIC-AP5095 | strongest_external | +0.009109 | [+0.005488, +0.013048] | 1.000 |
| SharedAction-NoROI | 10% | MORIC-style | strongest_historical | +0.009550 | [+0.005170, +0.013696] | 0.998 |
| SharedAction-NoROI | 10% | SetValue | SetValue | +0.000940 | [+0.000137, +0.001880] | 0.991 |
| SharedAction-NoROI | 25% | ORIC-AP50 | strongest_external | +0.002564 | [-0.001251, +0.006980] | 0.903 |
| SharedAction-NoROI | 25% | SemanticRank | strongest_historical | +0.001077 | [-0.003964, +0.005669] | 0.670 |
| SharedAction-NoROI | 25% | SetValue | SetValue | +0.001137 | [-0.000858, +0.003944] | 0.820 |
| SharedAction-NoROI | 40% | ORIC-AP50 | strongest_external | +0.002910 | [-0.000741, +0.006966] | 0.940 |
| SharedAction-NoROI | 40% | Current-Controlled-32D | strongest_historical | +0.001589 | [-0.001148, +0.004358] | 0.867 |
| SharedAction-NoROI | 40% | SetValue | SetValue | -0.000555 | [-0.001561, +0.000478] | 0.144 |

### 16.1 95% 区间完全高于 0 的比较

| 策略 | 预算 | 对手 | 均值 ΔAP | 95% CI |
| --- | ---: | --- | ---: | ---: |
| Original-only | 10% | SetValue | +0.004472 | [+0.002188, +0.006807] |
| Original-only | 40% | ORIC-AP50 | +0.009483 | [+0.004499, +0.014873] |
| Original-only | 40% | SemanticRank | +0.005058 | [+0.000331, +0.010156] |
| SharedAction-NoROI | 10% | MORIC-AP5095 | +0.009109 | [+0.005488, +0.013048] |
| SharedAction-NoROI | 10% | MORIC-style | +0.009550 | [+0.005170, +0.013696] |
| SharedAction-NoROI | 10% | SetValue | +0.000940 | [+0.000137, +0.001880] |

统计证据最明确的区域为：

- SharedAction-NoROI 10%；
- Original-only 40%。

其他多数比较的 95% 区间跨越 0，因此不能将点估计差异描述为确定或显著优势。

此外，`P(Δ>0)` 表示 Bootstrap 重采样中差值为正的比例，不应直接称为传统假设检验的 p-value。

---

## 17. 新旧 COCO 实验的综合解释

### 17.1 为什么旧三动作实验与新 No-ROI 实验的最优方法不同

旧三动作实验主要使用：

- fixed-call；
- 9 个历史内部 Ranker；
- ORIC-MLP、MORIC-MLP；
- SharedAction、AllROI、GBDT；
- 部分动作包含 ROI。

新实验主要使用：

- fixed-byte；
- 16 个 Ranker；
- Original-only、SharedAction-NoROI；
- ROI 完全禁用；
- 新增 Current-BudgetSet、SetValue、DualHead；
- 外部方法细分为 AP50 和 AP50–95 奖励版本。

因此，两组实验的最优方法不能简单并排后宣布矛盾。它们回答的是不同问题：

- 旧实验：固定上云调用比例下，不同 Ranker 和三动作策略如何权衡精度与通信成本；
- 新实验：在严格字节容量和无 ROI 条件下，哪个 Ranker 的预算感知选择更稳定。

### 17.2 原三动作实验仍然支持的结论

- 固定调用预算下，COCO 的 ORIC/MORIC 排序能力较强；
- 10% 由 ORIC 更强；
- 25%/40% 由 MORIC 更强；
- AllROI 能降低通信量，但通常损失 AP；
- GBDT 比 AllROI 更保守；
- Current-Controlled-32D 是历史内部 Ranker 中较稳定的代表。

### 17.3 No-ROI fixed-byte 实验新增的结论

- 去除 ROI 后，DualHead 的优势仍然存在；
- DualHead 在 6 个主工作点中取得 3 个第一；
- 其余 3 个点距最优不足 0.001 AP；
- DualHead 的优势主要体现在固定字节预算，而不是所有 fixed-call top-K；
- Original/Q90 的最优组合随字节预算变化；
- 25% 时 Q90 扩大覆盖有利；
- 40% 时过多 Q90 会损失载荷质量；
- MORIC-AP5095 仍是非常强的外部基线。

---

## 18. No-ROI 扩展阶段的 COCO 结论（历史阶段性）

> 本节仅概括截至 No-ROI 全 Ranker 扩展时的结论。后续 ActionValue 和 AFPD 已改变“当前最强学习器”的判断，请以第 28 节的截至当前结论为准。

1. **COCO 上不存在一个在所有协议、预算和动作策略下绝对占优的 Ranker。**

2. **在旧 fixed-call 三动作实验中，外部 ORIC/MORIC 具有稳定优势。**
   - 10%：ORIC 更强；
   - 25%/40%：MORIC 更强。

3. **在新 primary fixed-byte No-ROI 实验中，DualHead 的综合稳定性最强。**
   - 6 个主工作点中 3 个第一；
   - 另外 3 个点距最优均不足 0.001 AP。

4. **DualHead 的主要价值是字节预算感知，而不是固定 top-K 下全面统治。**

5. **去除 ROI 后 DualHead 仍然有效。**
   - 因此新 Ranker 的表现不能简单归因于 ROI 动作质量。

6. **Original-only 与 SharedAction-NoROI 没有统一优劣。**
   - 25% 字节预算下 Q90 扩大覆盖有效；
   - 40% 时 Original-only 更高，说明载荷保真度开始更重要。

7. **MORIC-AP5095 是需要保留的强外部基线。**
   - 尤其在 SharedAction-NoROI 25%/40% 和 fixed-call 10% 下具有竞争力。

8. **Current-Controlled-32D、SemanticRank、SetValue 和 Current-BudgetSet 仍然是重要内部比较器。**

9. **统计显著性不能由点估计排名替代。**
   - DualHead 的明确 Bootstrap 优势集中在 SharedAction-NoROI 10% 和 Original-only 40%；
   - 其他多数工作点应表述为相当或近似最优。

10. **新实验仍属于 post-hoc val2017 完整性扩展。**
    - 不能描述为新的隐藏 final 或独立确认性验证。

---

## 19. No-ROI 扩展阶段的论文组织建议（历史）

> 本节保留当时的组织建议供溯源，已被第 29 节的当前建议更新。

### 19.1 主 Ranker 表

若论文核心强调通信容量约束，建议把以下结果作为主表：

- No-ROI canonical；
- `fixed_byte`；
- Original-only；
- SharedAction-NoROI；
- 10%、25%、40% 字节预算；
- 16 个 Ranker，或正文保留代表方法、补充材料列全。

建议正文至少保留：

- Random；
- Confidence；
- SemanticRank；
- Current-Controlled-32D；
- Current-BudgetSet；
- SetValue；
- DualHead；
- ORIC-AP50；
- MORIC-AP50；
- ORIC-AP5095；
- MORIC-AP5095。

### 19.2 次要纯排序表

将 `fixed_call` 作为次要结果，用于说明：

- DualHead 并非所有 top-K 预算下第一；
- 其主要优势来自字节预算感知；
- fixed-call 10% 的 MORIC-AP5095 和 40% 的 Current-Controlled-32D 仍然很强。

### 19.3 动作策略消融

原三动作实验继续用于分析：

- SharedAction；
- CostEligible-AllROI；
- GBDT-Hierarchical。

但应明确：

> 该部分主要研究载荷动作对 AP 和通信成本的影响，不应与 No-ROI fixed-byte Ranker 主表混为同一协议。

### 19.4 外部方法对比

建议区分两类外部结果：

1. 原三动作实验中的 ORIC-MLP / MORIC-MLP；
2. canonical No-ROI 中的 ORIC-AP50、MORIC-AP50、ORIC-AP5095、MORIC-AP5095。

不同训练奖励版本必须写全名称，不能都简称 ORIC/MORIC。

### 19.5 推荐论文措辞

可以写：

> 在主要的固定字节协议下，DualHead 在六个策略—预算工作点中的三个获得最高 AP，并在其余三个工作点与最优方法保持在 0.001 AP 以内。其优势在 Original-only 的 40% 字节预算以及 SharedAction-NoROI 的 10% 字节预算下最为明确。

不应写：

- DualHead 在所有设置下第一；
- DualHead 显著超过全部基线；
- SharedAction-NoROI 始终优于 Original-only；
- 上传更多图像必然提高 AP；
- 该实验是新的独立 COCO final 验证。

---

## 20. 后续实验的协议与证据索引

下表是第 20 节以后所有新增数字的来源。“通过/失败”指对应实验自己的冻结 gate，不是事后主观判断。

| 实验 | 数据层 | 主要问题 | 协议结果 |
| --- | --- | --- | --- |
| `20260804_coco_budget_contextual_ranker_v2` | calibration + 已消费 val2017 | 预算条件化 pairwise/listwise 能否超过 ORIC/MORIC | calibration 通过；final 点估计小幅提升，CI 全部跨 0 |
| `20260804_coco_budget_sharedaction_costeligible_v1` | 4,000/1,000 development holdout | BudgetSet 和 SharedAction/ROI 的机制 | 诊断正结果，但 reward context 是 transductive，不是最终证据 |
| `20260804_coco_budgetset_value_cost_oof_v1` | 5-fold OOF | Value 与 Cost 拆分是否改善排序 | DualHead 信号质量更高，但所有候选的当轮 gate 均失败 |
| `20260804_coco_dualhead_canonical9678_val2017_paper_v1` | fit=9,678 / cal=571 / registered val=5,000 | canonical DualHead 论文级确认 | 点估计三点高于 ORIC-AP50，仅 40% CI 高于 0，总 gate 失败 |
| `20260805_coco_dualhead_canonical9678_triaction_extension_v1` | post-hoc val2017 reuse | DualHead 与三动作策略 | 补充完成，不是新隐藏 final |
| `20260805_coco_noroi_fullranker_canonical9678_v1` | post-hoc val2017 reuse | 16 Ranker × 2 No-ROI 策略的完整性 | 192 主行；已在第 11–19 节详列 |
| `20260805_coco_actionconditioned_valuerisk_oof_v1` | fit OOF + calibration | 两阶段排序还是图片—动作联合估值 | ActionValue-Joint 通过；Risk/LCB 崩溃 |
| `20260805_coco_actionvalue_roi_contribution_diagnostic_v1` | fit OOF + calibration | ActionValue 的收益是否来自 ROI | ROI 仅有小且不稳定的额外收益 |
| `20260805_coco_selection_context_jointaction_teacher_v1` | GT-aware fit/cal | 已选集合条件化的 Teacher ceiling | gate 失败 |
| `20260806_coco_dualheadv2_noroi_oof_v1` | fit OOF + sealed calibration | RiskMean/DualHead-v2 能否稳定超过 ActionValue | gate 失败 |
| `20260806_coco_relational_actionvalue_oof_v1` | strict fit OOF | 关系集合编码与跨图排序 | 10% 提升，总 gate 失败，calibration 未访问 |
| `20260806_coco_factorized_global_actionrank_oof_v1` | strict fit OOF | 图片价值与动作 advantage 因子化 | gate 失败，calibration 未访问 |
| `20260809_three_dataset_cross_action_calibration_v1` | fit OOF + calibration | Original/Q90 的跨动作尺度是否是瓶颈 | calibration 低/中预算改善；主瓶颈仍是 membership/图片选择 |
| `20260809_coco_oregon_full_scheduler_benchmark_v1` | post-hoc val2017 reuse | 当前学习器、外部基线、Confidence/Random 全比较 | Raw ActionValue-OQ 是该协议点估计最强 |
| `20260809_coco_frontier_aligned_actionvalue_exploration_v1` 至 `v6` | strict fit OOF | AFPD 与 5 个后续机制消融 | AFPD v1 通过 OOF；v2–v6 均不足以推广 |
| `20260810_coco_afpd_occupancy_paper_protocol_v1` | one-shot calibration | AFPD/Occupancy 能否进入 final | calibration gate 失败 |
| `20260810_coco_afpd_forced_fullval_posthoc_v1` | 独立重训 + post-hoc full val | 强制完整检查 AFPD/Occupancy | 已完成；不是确认性 final |
| `20260810_coco_afpd_same_run_baseline_bootstrap_v1` | post-hoc same-run bootstrap | AFPD 相对同次重训 Current 的纯归因 | 6 组全完成；仅 40% vs Current-OQ 明确为正 |

---

## 21. 从 BudgetSet 到 canonical DualHead 的完整结果

### 21.1 Budget-conditioned contextual ranker v2

该诊断沿用 ORIC 的 top-25 特征和 contextual AP50 reward，仅改为预算条件化 pairwise/listwise 目标。它证明“超过 ORIC/MORIC 的排序边界是可能的”，但不是独立创新的最终实现。

| 阶段 | 方法 | 10% AP | 25% AP | 40% AP |
| --- | --- | ---: | ---: | ---: |
| calibration | 最强外部基线 | 0.551489 | 0.593854 | 0.626858 |
| calibration | Budget-conditioned candidate | 0.553851 | 0.602469 | 0.631916 |
| calibration | ΔAP | +0.002362 | +0.008614 | +0.005058 |
| frozen val2017 | ORIC/MORIC comparator | 0.449341 | 0.489265 | 0.519131 |
| frozen val2017 | Budget-conditioned candidate | 0.450031 | 0.490677 | 0.520974 |
| frozen val2017 | ΔAP | +0.000690 | +0.001412 | +0.001843 |

| 预算 | Candidate KB | Comparator KB | Bootstrap 均值 ΔAP | 95% CI | Pr(Δ>0) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 17.113 | 16.574 | +0.000826 | [-0.002254, +0.004321] | 0.660 |
| 25% | 40.516 | 40.760 | +0.001330 | [-0.002591, +0.005370] | 0.740 |
| 40% | 63.792 | 64.252 | +0.001833 | [-0.001382, +0.006228] | 0.860 |

**结论：** calibration gate 通过，但三个 val2017 bootstrap CI 全部跨 0；10% 还同时增加了字节。这是机制证据，不是确认性超越。

### 21.2 BudgetSet + SharedAction/CostEligible development holdout

| 方法 | Fixed-call 10% | Fixed-call 25% | Fixed-call 40% | Fixed-byte 10% | Fixed-byte 25% | Fixed-byte 40% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORIC-Full | 0.586242 | 0.599077 | 0.612945 | 0.585685 | 0.602377 | 0.613260 |
| MORIC-Full | 0.596034 | 0.611673 | 0.624274 | 0.588760 | 0.608210 | 0.623652 |
| BudgetSet-Full | 0.598055 | 0.615132 | 0.628342 | 0.594316 | 0.615225 | 0.628575 |
| BudgetSet-SharedAction | 0.598055 | 0.615132 | 0.628342 | 0.593838 | 0.614936 | 0.628100 |

BudgetSet-Full 相对 MORIC-Full 的 Fixed-call 变化为：

| 预算 | ΔAP | ΔKB |
| ---: | ---: | ---: |
| 10% | +0.002021 | -12.47% |
| 25% | +0.003459 | -9.71% |
| 40% | +0.004068 | -5.17% |

BudgetSet-SharedAction 相对 BudgetSet-Full 的 Fixed-byte AP 变化为 `-0.000478 / -0.000288 / -0.000475`，说明当时 ROI/SharedAction 并非增益来源。注意：该轮为 4,000/1,000 development holdout，reward context 具有 transductive 性质，不可冒充 canonical 结果。

### 21.3 Value/Cost 拆分的 5-fold OOF

| 方法 | Fixed-call 10% | 25% | 40% | Gain Spearman | Payload Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORIC-AP50 | 0.500635 | 0.541669 | 0.565663 | — | — |
| MORIC-AP50 | 0.498887 | 0.542848 | 0.571840 | — | — |
| ORIC-AP5095 | 0.501819 | 0.542776 | 0.570809 | — | — |
| MORIC-AP5095 | 0.499374 | 0.543474 | 0.573548 | — | — |
| Current-BudgetSet | 0.499301 | 0.538839 | 0.571812 | 0.4104 | -0.2274 |
| SetValue | 0.503212 | 0.542255 | 0.574096 | 0.4565 | +0.0342 |
| DualHead | 0.502471 | 0.543675 | 0.575868 | **0.4614** | +0.0153 |

| 方法 | Teacher Top-K overlap 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 29.20% | 46.00% | 58.05% |
| SetValue | 35.80% | 49.76% | 61.60% |
| DualHead | **36.00%** | **50.32%** | **62.10%** |

**结论：** 去掉成本混入 Value 后，与真实 gain 的相关性和 Teacher overlap 均提高；DualHead 在这两项诊断上最好。但当轮所有预注册 gate 均为 false，且 reward context 可能跨 OOF fold，所以只能用于选择下一步方向。

### 21.4 Canonical DualHead registered val2017

统一划分为 train2017 `fit=9,678 / calibration=571 / registered val2017=5,000`，动作为 Edge/Original，Fixed-byte 为主指标。

| 方法 | 10% AP | 25% AP | 40% AP |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.448114 | 0.491589 | 0.519428 |
| SetValue | 0.450506 | **0.492992** | **0.523626** |
| DualHead | **0.452287** | 0.492888 | 0.523205 |
| 冻结 ORIC-AP50 | 0.449543 | 0.488787 | 0.517391 |
| DualHead - ORIC | +0.002744 | +0.004101 | +0.005815 |

| 预算 | Bootstrap 均值 ΔAP | 95% CI | Pr(Δ>0) |
| ---: | ---: | ---: | ---: |
| 10% | +0.002634 | [-0.001825, +0.007017] | 0.887 |
| 25% | +0.004164 | [-0.000990, +0.009884] | 0.937 |
| 40% | +0.005983 | [+0.000569, +0.012008] | 0.983 |

**结论：** 三个点估计全部高于冻结 ORIC，但仅 40% 的 95% CI 完全高于 0，预注册 gate 要求至少两个优势点，因此 `passed=false`。

### 21.5 DualHead 三动作 post-hoc 扩展

| 冻结动作策略 | 10% AP / Δ外部 | 25% AP / Δ外部 | 40% AP / Δ外部 |
| --- | ---: | ---: | ---: |
| SharedAction | 0.446814 / +0.003401 | 0.492979 / +0.002457 | 0.521679 / +0.003284 |
| CostEligible-AllROI | 0.450197 / +0.004640 | 0.499297 / -0.000241 | 0.531495 / +0.002650 |
| GBDT-Hierarchical | 0.450197 / +0.004640 | 0.499297 / -0.000241 | 0.531495 / +0.002650 |

| 策略 | 预算 | Bootstrap 均值 ΔAP | 95% CI | Pr(Δ>0) |
| --- | ---: | ---: | ---: | ---: |
| SharedAction | 10% | +0.003514 | [+0.000989, +0.006425] | 0.998 |
| SharedAction | 25% | +0.002424 | [-0.001514, +0.006830] | 0.891 |
| SharedAction | 40% | +0.003306 | [-0.000211, +0.007500] | 0.961 |
| CostEligible-AllROI | 10% | +0.004950 | [+0.002421, +0.007560] | 1.000 |
| CostEligible-AllROI | 25% | -0.000245 | [-0.002967, +0.002912] | 0.412 |
| CostEligible-AllROI | 40% | +0.002617 | [-0.000159, +0.005400] | 0.970 |
| GBDT-Hierarchical | 10% | +0.004950 | [+0.002421, +0.007560] | 1.000 |
| GBDT-Hierarchical | 25% | -0.000245 | [-0.002967, +0.002912] | 0.412 |
| GBDT-Hierarchical | 40% | +0.002617 | [-0.000159, +0.005400] | 0.970 |

AllROI 与 GBDT 数字完全相同，说明在该冻结配置下 GBDT 没有改变实际选择。该补充为 val2017 post-hoc reuse，不应与第 21.4 节的注册主结果混为同一次 confirmatory 证据。

---

## 22. Action-conditioned Value 联合调度的决定性诊断

### 22.1 两阶段 DualHead+SharedAction 对比 ActionValue-Joint

旧架构先给图片排序，再由 SharedAction 选 Original/Q90/ROI；ActionValue-Joint 直接对每个“图片—动作”进行估值，然后在真实字节约束下联合选择。

#### Fixed-byte 主结果

| 证据层 | 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| fit 5-fold OOF | CurrentDualHead + SharedAction | 0.508873 / 15.93 | 0.557670 / 39.82 | 0.590531 / 63.72 |
| fit 5-fold OOF | **ActionValue-Joint** | **0.538453 / 15.93** | **0.595035 / 39.82** | **0.627713 / 63.72** |
| fit 5-fold OOF | Value+Risk/LCB | 0.460497 / 0.88 | 0.460497 / 0.88 | 0.460497 / 0.88 |
| frozen calibration | CurrentDualHead + SharedAction | 0.547589 / 15.46 | 0.589121 / 38.71 | 0.621195 / 61.88 |
| frozen calibration | **ActionValue-Joint** | **0.572072 / 15.49** | **0.629529 / 38.71** | **0.656786 / 61.95** |
| frozen calibration | Value+Risk/LCB | 0.504993 / 1.76 | 0.504993 / 1.76 | 0.504993 / 1.76 |

ActionValue-Joint 相对两阶段基线的 Fixed-byte ΔAP：

| 证据层 | 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| fit 5-fold OOF | +0.029580 | +0.037365 | +0.037183 |
| frozen calibration | +0.024484 | +0.040407 | +0.035591 |

#### Fixed-call 辅助结果

| 证据层 | 方法 | 10% AP | 25% AP | 40% AP |
| --- | --- | ---: | ---: | ---: |
| fit 5-fold OOF | CurrentDualHead + SharedAction | 0.504447 | 0.543254 | 0.576323 |
| fit 5-fold OOF | ActionValue-Joint | 0.503625 | 0.545071 | 0.577329 |
| fit 5-fold OOF | Value+Risk/LCB | 0.483537 | 0.519042 | 0.549301 |
| frozen calibration | CurrentDualHead + SharedAction | 0.544203 | 0.587155 | 0.618742 |
| frozen calibration | ActionValue-Joint | 0.549262 | 0.589358 | 0.622603 |
| frozen calibration | Value+Risk/LCB | 0.529445 | 0.565099 | 0.594854 |

**核心结论：** 最大提升来自把“选哪张图”和“用哪个动作”改为联合估值，而不是继续扩大 DualHead 或增加新 ROI 动作。当时 contextual reward 的 32 图 context 可能跨 OOF fold，所以该轮只是机制诊断，`open_three_seed_paper_protocol=false`。

### 22.2 Risk/LCB 为什么失败

- fit OOF 中只有 127/22,041 个云动作的 LCB 大于 0；
- calibration 只有 14/1,299 个云动作的 LCB 大于 0；
- Edge 效用设为 0 时，调度器宁愿几乎不上传；
- fit OOF 三个字节预算下均只使用 0.88 KB/图，calibration 均只使用 1.76 KB/图。

因此 LCB 不是“更安全的 Value”，而是一个过度保守、几乎放弃容量的策略。Risk 后续不再直接替换 Value。

### 22.3 ROI 收益归因：OQ vs OQR vs GT Oracle gate

#### fit 5-fold OOF

| 预算 | ActionValue-OQ | Learned OQR | Learned ΔOQ | GT-relative OQR | Oracle ΔOQ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.536741 | 0.538453 | +0.001712 | 0.538169 | +0.001428 |
| 25% | 0.592833 | 0.595035 | +0.002202 | 0.595078 | +0.002245 |
| 40% | 0.627062 | 0.627713 | +0.000651 | 0.628502 | +0.001440 |

#### frozen calibration

| 预算 | ActionValue-OQ | Learned OQR | Learned ΔOQ | GT-relative OQR | Oracle ΔOQ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.573900 | 0.572072 | -0.001827 | 0.574809 | +0.000910 |
| 25% | 0.626870 | 0.629529 | +0.002659 | 0.629976 | +0.003106 |
| 40% | 0.656807 | 0.656786 | -0.000021 | 0.657625 | +0.000818 |

| 证据层 | Learned ROI 相对次优率 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| fit OOF | 75.22% | 70.96% | 74.84% |
| calibration | 95.24% | 74.00% | 68.89% |

**结论：** COCO 的 ROI 有可恢复收益，但尾部小且不稳定；当时的 learned ROI 大多数实际不如 Original/Q90/不上传。这个结论后来促成了 AFPD 的完整 frontier 监督，而不是简单 AllROI。

### 22.4 Selection-context Teacher ceiling

| 预算 | Static JointAction AP | Selection-context AP | ΔAP | 图片 Jaccard | 精确动作 Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.596227 | 0.593598 | -0.002629 | 0.825 | 0.799 |
| 25% | 0.650354 | 0.652037 | +0.001683 | 0.859 | 0.789 |
| 40% | 0.673916 | 0.671540 | -0.002375 | 0.904 | 0.762 |

该结果是 GT-aware Teacher，即使如此也只在 25% 小幅提升，10%/40% 退化，`passed=false`。因此没有继续实现 Selection-Contextual SharedAction 完整学习器。

### 22.5 DualHead-v2 / RiskMean No-ROI

| 证据层 | 方法 | Fixed-byte 10% | 25% | 40% |
| --- | --- | ---: | ---: | ---: |
| fit OOF | ActionValue-OQ | 0.536741 | 0.592833 | 0.627062 |
| fit OOF | RiskMean / DualHead-v2 | 0.519637 | 0.583221 | 0.624555 |
| calibration | ActionValue-OQ | 0.573900 | 0.626870 | 0.656807 |
| calibration | RiskMean / DualHead-v2 | 0.557834 | 0.614149 | 0.655647 |

该分支所有 gate 均失败。它排除了“只需把风险平均进价值就能增强 DualHead”这一假设。

---

## 23. ActionValue 之后的表征、目标与校准诊断

### 23.1 Relational Action-Value Ranker

此轮用四个消融区分“跨图片动作排序目标”和“检测框关系集合编码”的作用。全部为 strict fit OOF，No-ROI，calibration 在 gate 失败后未访问。

#### Fixed-byte

| 方法 | 集合表征 | 训练目标 | 10% AP | 25% AP | 40% AP | 三点均值 ΔCurrent |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CurrentActionValue | mean/max | 原回归+图内排序 | 0.539103 | 0.594713 | **0.627965** | — |
| GlobalRank | mean/max | 跨图片动作排序 | 0.540462 | **0.595829** | 0.627260 | +0.000590 |
| RelSetValue | relation set encoder | 原目标 | 0.540173 | 0.594570 | 0.626938 | -0.000034 |
| RelActionRank | relation set encoder | 跨图片动作排序 | **0.541631** | 0.595683 | 0.627153 | +0.000895 |

RelActionRank 相对 CurrentActionValue 为 `+0.002528 / +0.000970 / -0.000812`。预注册要求至少两个 Fixed-byte 点 `>=+0.001`；25% 只差 0.000030，但仍必须判为 `passed=false`。

#### Fixed-call

| 方法 | 10% AP | 25% AP | 40% AP |
| --- | ---: | ---: | ---: |
| CurrentActionValue | 0.507512 | 0.550215 | 0.582960 |
| GlobalRank | 0.506475 | 0.550279 | 0.583026 |
| RelSetValue | 0.508344 | 0.550770 | 0.581111 |
| RelActionRank | **0.509011** | **0.551506** | 0.583011 |

**结论：** 跨图片排序目标在低预算真实有效；单独增加 relation/Transformer 集合表征没有稳定改善。40% Fixed-byte 的退化主要与 Original/Q90 跨动作价值尺度及昂贵 Original 选得过多有关。

### 23.2 Factorized Global Action Ranker

该方法将总价值拆成图片是否值得上云的 `G_i` 和 Original/Q90 图内相对 advantage `A_i,a`。

| 模式 | 方法 | 10% AP | 25% AP | 40% AP |
| --- | --- | ---: | ---: | ---: |
| Fixed-byte | CurrentActionValue | 0.539103 | 0.594713 | 0.627965 |
| Fixed-byte | FactorizedGlobalActionRank | 0.538357 | 0.589807 | 0.617611 |
| Fixed-byte | ΔCurrent | -0.000746 | -0.004906 | -0.010355 |
| Fixed-call | CurrentActionValue | 0.507512 | 0.550215 | 0.582960 |
| Fixed-call | FactorizedGlobalActionRank | 0.508431 | 0.550777 | 0.582494 |

**结论：** 因子化使高预算价值校准更差，总 gate 失败，calibration 保持封存。不继续扩大该分支。

### 23.3 Cross-action monotonic calibration

该实验不换网络，只对 Original 和 Q90 分别做 fold-local monotonic/isotonic calibration，用于检验跨动作数值尺度是否是主瓶颈。

#### Fixed-byte

| 证据层 | 方法 | 10% AP | 25% AP | 40% AP |
| --- | --- | ---: | ---: | ---: |
| fit OOF | Raw ActionValue-OQ | **0.536741** | 0.592833 | 0.627062 |
| fit OOF | Per-action isotonic OQ | 0.535781 | **0.593035** | **0.627137** |
| frozen calibration | Raw ActionValue-OQ | 0.573900 | 0.626870 | 0.656807 |
| frozen calibration | Per-action isotonic OQ | **0.578733** | **0.630260** | **0.656961** |

#### Fixed-call

| 证据层 | 方法 | 10% AP | 25% AP | 40% AP |
| --- | --- | ---: | ---: | ---: |
| fit OOF | Raw ActionValue-OQ | **0.503625** | **0.545071** | **0.577329** |
| fit OOF | Per-action isotonic OQ | 0.503044 | 0.544280 | 0.576196 |
| frozen calibration | Raw ActionValue-OQ | 0.549262 | 0.589358 | **0.622603** |
| frozen calibration | Per-action isotonic OQ | **0.549277** | **0.589956** | 0.621183 |

#### calibration Fixed-byte Oracle 瓶颈分解

| 方法 | 10% AP | 25% AP | 40% AP |
| --- | ---: | ---: | ---: |
| Learned calibrated membership + Oracle O/Q action | 0.578701 | 0.630488 | 0.657921 |
| Oracle membership + Calibrated action | 0.588142 | 0.643826 | 0.663857 |
| Full O/Q Oracle | 0.589671 | 0.646866 | 0.670733 |

**结论：** 校准能修复 calibration 低/中预算的一部分跨动作误差，但 Oracle membership 带来的提升明显更大。主瓶颈仍是“选哪些图/动作候选进容量”，不是仅靠 Original/Q90 标定就能解决。

### 23.4 Advantage ROI Gate 跨数据集诊断中的 COCO 结果

| 证据层 | OQ AP 10%/25%/40% | Naive OQR AP 10%/25%/40% | AdvantageGate AP 10%/25%/40% |
| --- | --- | --- | --- |
| fit OOF | 0.536741 / 0.592833 / 0.627062 | 0.538398 / 0.595041 / 0.627698 | 0.536741 / 0.592833 / 0.627062 |
| calibration | 0.573900 / 0.626870 / 0.656807 | 0.572072 / 0.629529 / 0.656786 | 0.573900 / 0.626870 / 0.656807 |

COCO 上冻结阈值为 `-0.0017557`，最终没有任何 ROI 通过 gate，策略完全退化为 OQ。它说明当时的 advantage gate 只学会了“不用 ROI”，并未学会安全 ROI 尾部。

---

## 24. COCO 完整调度器基准：学习器、外部基线、Confidence 与 Random

该轮生成 197 个主结果行和 600 个 Random seed 行。以下为 Fixed-byte 论文核心对比；原 9 Ranker 和三策略的详细结果已在第 3–16 节列全，不再重复整张 197 行。该轮是 `post-hoc final reuse`，val2017 未用于训练、阈值或比较器选择，但也不是新的隐藏 final。

### 24.1 Fixed-byte 主结果

| 方法 | 策略 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| **Raw ActionValue-OQ** | Joint-OQ | **0.472192 / 15.91** | **0.525455 / 39.78** | **0.559262 / 63.65** |
| Per-action Isotonic OQ | Joint-OQ | 0.471139 / 15.91 | 0.525039 / 39.78 | 0.558810 / 63.65 |
| DualHead | Original-only | 0.446943 / 15.90 | 0.491117 / 39.78 | **0.523762 / 63.65** |
| DualHead | SharedAction-NoROI | 0.446715 / 15.91 | **0.492770 / 39.78** | 0.521186 / 63.64 |
| Confidence | Original-only | **0.443833 / 15.90** | **0.483089 / 39.78** | 0.502723 / 63.64 |
| Confidence | SharedAction-NoROI | 0.438580 / 15.90 | 0.481590 / 39.78 | **0.511007 / 63.64** |
| Random | Original-only | **0.434290 / 15.91** | 0.468621 / 39.77 | 0.497239 / 63.64 |
| Random | SharedAction-NoROI | 0.432729 / 15.91 | **0.469860 / 39.77** | **0.501508 / 63.64** |
| ORIC-MLP | Original-only | **0.445248 / 15.91** | 0.482116 / 39.77 | 0.514506 / 63.64 |
| ORIC-MLP | SharedAction-NoROI | 0.437167 / 15.90 | **0.484735 / 39.76** | **0.519733 / 63.64** |
| MORIC-MLP | Original-only | 0.446639 / 15.91 | 0.483809 / 39.77 | 0.518209 / 63.65 |
| MORIC-MLP | SharedAction-NoROI | **0.446856 / 15.91** | **0.494336 / 39.78** | **0.522596 / 63.64** |

同一方法类别内的加粗值表示 Original-only/SharedAction-NoROI 中较高者；整表的三个最高值均为 Raw ActionValue-OQ。

### 24.2 Raw ActionValue 相对代表基线

| 对手 | 对手最优 10%/25%/40% AP | Raw ActionValue ΔAP 10% | 25% | 40% |
| --- | --- | ---: | ---: | ---: |
| 最强外部 ORIC/MORIC | 0.446856 / 0.494336 / 0.522596 | +0.025336 | +0.031119 | +0.036666 |
| Confidence 最优策略 | 0.443833 / 0.483089 / 0.511007 | +0.028359 | +0.042366 | +0.048255 |
| Random 最优策略 | 0.434290 / 0.469860 / 0.501508 | +0.037902 | +0.055595 | +0.057754 |
| DualHead 最优策略 | 0.446943 / 0.492770 / 0.523762 | +0.025249 | +0.032686 | +0.035500 |

**这是目前最清楚的广泛 post-hoc 比较结论：** Raw ActionValue-OQ 在三个 Fixed-byte 工作点的点估计均超过外部 ORIC/MORIC、Confidence、Random 和原 DualHead。这说明联合图片—动作估值与精确字节容量调度的结构改进是真实的；但因为是已消费 val2017，不能用“新隐藏测试上确认超越”来描述。

### 24.3 Raw 与 calibrated ActionValue 动作组成

| 方法 | 预算 | AP | AP50 | AP75 | KB/图 | Original | Q90 | 上传率 | 选中图片 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw ActionValue-OQ | 10% | 0.472192 | 0.647457 | 0.513013 | 15.91 | 0.74% | 24.24% | 24.98% | 1,249 |
| Raw ActionValue-OQ | 25% | 0.525455 | 0.704769 | 0.574096 | 39.78 | 1.78% | 52.02% | 53.80% | 2,690 |
| Raw ActionValue-OQ | 40% | 0.559262 | 0.738440 | 0.611470 | 63.65 | 5.40% | 73.44% | 78.84% | 3,942 |
| Per-action Isotonic OQ | 10% | 0.471139 | 0.645890 | 0.512082 | 15.91 | 0.66% | 23.96% | 24.62% | 1,231 |
| Per-action Isotonic OQ | 25% | 0.525039 | 0.704411 | 0.573955 | 39.78 | 1.36% | 51.98% | 53.34% | 2,667 |
| Per-action Isotonic OQ | 40% | 0.558810 | 0.738100 | 0.611290 | 63.65 | 4.04% | 74.18% | 78.22% | 3,911 |

校准在开发 calibration 上有改善，但在这次完整 val2017 上三点均略低于 Raw：`-0.001053 / -0.000416 / -0.000451 AP`。因此不能将 isotonic calibration 定为必然更强的升级。

### 24.4 Fixed-call 中 Raw/Calibrated ActionValue

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Raw ActionValue-OQ | 0.444032 / 15.93 | 0.483683 / 39.46 | **0.510823 / 63.05** |
| Per-action Isotonic OQ | **0.444075 / 15.49** | 0.482652 / 33.95 | 0.509834 / 55.90 |
| ORIC-MLP Original-only | **0.449341 / 16.57** | 0.486328 / 40.60 | 0.511614 / 64.59 |
| MORIC-MLP Original-only | 0.447244 / 16.61 | **0.489265 / 40.76** | **0.519131 / 64.25** |

Fixed-call 不考虑每张图动作的真实字节容量，ActionValue 并未超过 ORIC/MORIC。它的主要优势明确集中在 Fixed-byte 调度。

### 24.5 冻结 calibrated candidate 的 9 组 paired bootstrap

Bootstrap 候选是当时预先冻结的 `PerAction-Isotonic-OQ`，不是事后点估计更高的 Raw ActionValue。外部对手也按冻结比较器选择，因此 bootstrap 均值差不必等于第 24.2 节事后“每个预算取表内最大值”的简单相减。

| 对手 | 预算 | Bootstrap 均值 ΔAP | 95% CI | Pr(Δ>0) |
| --- | ---: | ---: | ---: | ---: |
| 冻结最强外部 | 10% | +0.034822 | [+0.028191, +0.041172] | 1.000 |
| 冻结最强外部 | 25% | +0.030687 | [+0.025370, +0.036716] | 1.000 |
| 冻结最强外部 | 40% | +0.036114 | [+0.031719, +0.040778] | 1.000 |
| Confidence | 10% | +0.032093 | [+0.025551, +0.038402] | 1.000 |
| Confidence | 25% | +0.043509 | [+0.037566, +0.049879] | 1.000 |
| Confidence | 40% | +0.047907 | [+0.043092, +0.053062] | 1.000 |
| Random | 10% | +0.038281 | [+0.031953, +0.044423] | 1.000 |
| Random | 25% | +0.054946 | [+0.048647, +0.061047] | 1.000 |
| Random | 40% | +0.060022 | [+0.054578, +0.065429] | 1.000 |

### 24.6 完整基准的扩展覆盖状态

| 扩展 | 状态 |
| --- | --- |
| SharedAction | `completed_imported` |
| CostEligible-AllROI | `completed_imported` |
| GBDT-Hierarchical | `completed_imported` |
| ROI | `completed_imported` |
| MAUR/Reversible | `completed_imported` |
| DualHead-v2 / RelActionRank / Factorized | `coco_completed_imported` |
| Cloud-V2 Profile×Scale | COCO 无对应 Cloud-V2 cache，`not_applicable` |
| DualHead-v3 | 没有冻结可复现定义，`not_run` |

---

## 25. AFPD 创新探索：v1–v6 的全部 development OOF 结果

### 25.1 AFPD 实际改了什么

普通 ActionValue 对每个图片—动作回归一个静态 contextual reward，外部调度器再计算 `value - λ × bytes`。AFPD 将监督对象改为每张图的完整参数化动作上包络：

1. 枚举 Edge/Original/Q90/ROI 动作效用线的两两交点；
2. 在每个交点前后监督真正应该胜出的动作；
3. 强化接近动作切换边界的样本；
4. payload 不输入 Value 网络，只在 frontier Teacher 解析和部署调度时使用；
5. 被支配动作自动从 Teacher path 中消失；
6. 部署仍使用 exact-capacity scheduler，四动作在同一字节容量中竞争。

所以 AFPD 的核心不是“再加一个 head”，而是把训练目标从单点 reward 回归改成与下游预算决策一致的整条动作路径蒸馏。

### 25.2 AFPD v1 strict fit OOF 主结果

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| CurrentActionValue-OQ | 0.540011 / 15.929 | 0.595700 / 39.823 | 0.627999 / 63.717 |
| **AFPD-OQR** | **0.542323 / 15.929** | **0.599302 / 39.823** | **0.630157 / 63.717** |
| ΔAP | **+0.002312** | **+0.003602** | **+0.002158** |

- 三个单独种子在三个 Fixed-byte 预算上均至少提升 `+0.001 AP`；
- Fixed-call ensemble ΔAP 为 `-0.000795 / +0.001232 / -0.000268`；
- seed 20260810 的 Fixed-call 40% 为 `-0.001262`，因此不能声称 AFPD 在每个 Fixed-call 种子都提升；
- 该阶段仅使用 canonical `fit=9,678` 的 strict 5-fold OOF，没有访问 calibration 或 val2017。

### 25.3 ROI frontier 与不可部署上界

| 指标 | 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| AFPD 选择 ROI 数 | 857 | 1,534 | 1,809 |
| 与当前预算 Teacher ROI 的精确 precision | 36.99% | 41.92% | 41.07% |
| Learned ROI 位于任意真实 frontier | 约 62.9% | 约 63.3% | 约 61.1% |

其他总体统计：

- fit 中 ROI 可用图片：2,731/9,678（28.22%）；
- ROI 至少在某个真实 reward-byte frontier 上：1,563/2,731（57.23%）。

| GT-aware ROI 修正（相对 AFPD） | 10% ΔAP | 25% ΔAP | 40% ΔAP |
| --- | ---: | ---: | ---: |
| 任意真实 frontier gate | +0.002766 | +0.003230 | +0.002831 |
| 当前预算 utility gate | +0.002221 | +0.003199 | +0.002773 |
| 直接替换为真实 ROI residual | +0.004239 | +0.004475 | +0.004051 |

这些 Oracle 只证明安全 ROI 尾部仍然存在，不证明当前静态 Edge 特征可以把它稳定学会。

### 25.4 v1–v6 六轮机制实验

| 版本 | 核心假设 | Fixed-byte 相对基线/AFPD 10% / 25% / 40% | 关键诊断 | 决策 |
| --- | --- | --- | --- | --- |
| v1 AFPD | 完整 reward-byte frontier 监督比标量回归更匹配调度 | **+0.002312 / +0.003602 / +0.002158** | 三种子三预算方向一致 | OOF 通过，冻结候选 |
| v2 Occupancy Gate | 先判断 ROI 是否曾位于 frontier | +0.000158 / +0.000338 / +0.000154 | 收益小于 0.001 门槛 | 仅作消融 |
| v3 Shadow-price Gate | 给 ROI gate 显式预算价格 | -0.001593 / -0.002405 / -0.001528 | ROI recall 仅 2.45%–3.48% | 拒绝 |
| v4 Dual-conditioned residual | 用 price、AFPD margin 和竞争动作预测 ROI residual | -0.000303 / +0.000105 / -0.000138 | 安全但没有实际收益 | 拒绝 |
| v5 Transition audit | ROI crossing 可能欠采样 | 不重训 | ROI 可用条件下 crossing 权重 51.26%，目标权重 35.87% | 假设不成立 |
| v6 Frontier consensus | 三种子分歧可过滤错误 ROI | 0 / 0 / -0.0000046 | 2/3 共识保留 2,670/2,731 ROI，模型会一致犯错 | 拒绝 |

### 25.5 创新性边界

已有文献中存在 decision-focused learning、SPO loss、shadow price、uncertainty set、set-valued prediction 和 ROI offloading，所以不能把这些通用概念单独声称为原创。当前可以严谨描述的 AFPD 创新组合是：

- 成本不进入的静态图片—动作 value；
- 对完整 `reward - λ × bytes` 上包络进行蒸馏；
- 对动作交点两侧做显式路径监督；
- Edge/Original/Q90/ROI 在 exact-capacity scheduler 中联合竞争。

在当时已检索的相邻文献中未发现与这个完整组合同构的方法，但这不等于已证明“全球首创”。AFPD 的创新性要与后续独立评测的实证强度分开表述。

---

## 26. AFPD/Occupancy 论文协议：一次性 calibration 与独立重训 full val

### 26.1 一次性 calibration=571

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| **Current-OQ** | **0.580552 / 15.48** | 0.633421 / 38.72 | 0.659326 / 61.94 |
| **Current-OQR** | 0.579637 / 15.48 | **0.635091 / 38.71** | 0.656812 / 61.94 |
| AFPD-OQ | 0.579995 / 15.47 | 0.628837 / 38.71 | 0.659610 / 61.95 |
| AFPD-OQR | 0.578163 / 15.49 | 0.633142 / 38.72 | **0.659660 / 61.93** |
| AFPD-OQR + OccupancyFree | 0.578608 / 15.48 | 0.632072 / 38.71 | 0.659109 / 61.94 |
| AFPD-OQR + OccupancyAware | 0.578354 / 15.48 | 0.632465 / 38.71 | 0.659178 / 61.95 |

- `calibration primary gate=false`；
- 当时的预注册规则下 `post-hoc final opened=false`；
- Occupancy Gate 始终是次要消融，不能在 calibration 上替代 AFPD-OQR；
- AFPD development OOF 的三点正收益没有在 571 张 calibration 上复现成稳定优势。

### 26.2 独立从头重训的 full val2017 Fixed-byte

用户要求完整检查后，实验在不修改冻结定义的前提下从头重训所有 AFPD/Occupancy 模型，并强制评测 5,000 张 val2017。由于 calibration gate 再次失败且 val2017 已被历史使用，这是 `posthoc_full_val_paper_format_independent_retrain`，不是 confirmatory hidden final。

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-OQ | 0.478224 / 15.91 | 0.528884 / 39.78 | 0.559286 / 63.65 |
| **Current-OQR** | **0.480491 / 15.91** | **0.532639 / 39.78** | 0.561182 / 63.65 |
| AFPD-OQ | 0.476406 / 15.91 | 0.529313 / 39.78 | 0.560062 / 63.65 |
| AFPD-OQR | 0.475686 / 15.91 | 0.530840 / 39.78 | 0.562336 / 63.65 |
| AFPD-OQR + OccupancyFree | 0.475620 / 15.91 | 0.532182 / 39.78 | **0.562572 / 63.65** |
| AFPD-OQR + OccupancyAware | 0.475572 / 15.91 | 0.531829 / 39.78 | 0.562388 / 63.65 |

**点估计排名：** 10%/25% 均是 Current-OQR 最高；40% 是 OccupancyFree 最高，其次为 OccupancyAware 和 AFPD-OQR。因此 AFPD 不能取代 Current-OQR 成为三预算统一主方法。

### 26.3 Full val AP50 / AP75

| 方法 | 10% AP50 / AP75 | 25% AP50 / AP75 | 40% AP50 / AP75 |
| --- | ---: | ---: | ---: |
| Current-OQ | 0.653405 / 0.521689 | 0.708478 / 0.578464 | 0.738728 / 0.612046 |
| Current-OQR | **0.656241 / 0.522598** | **0.712583 / 0.582853** | 0.739557 / 0.613715 |
| AFPD-OQ | 0.651593 / 0.519484 | 0.709737 / 0.578767 | 0.739047 / 0.612406 |
| AFPD-OQR | 0.651122 / 0.518955 | 0.709199 / 0.582142 | 0.738897 / 0.615942 |
| AFPD-OQR + OccupancyFree | 0.650994 / 0.518968 | 0.711608 / 0.582816 | **0.739396 / 0.616290** |
| AFPD-OQR + OccupancyAware | 0.650932 / 0.518828 | 0.711424 / 0.582535 | 0.739230 / 0.616246 |

40% 时 AFPD/Occupancy 的 AP75 更高，与 ROI 对局部定位的改善一致；但 10% 低预算的 AP50 和 AP75 都低于 Current-OQR。

### 26.4 Full val Fixed-call

| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-OQ | 0.447997 / 16.07 | **0.489166 / 39.13** | **0.519119 / 62.49** |
| Current-OQR | 0.447951 / 15.89 | 0.487230 / 39.04 | 0.517359 / 62.90 |
| **AFPD-OQ** | **0.448980 / 16.43** | 0.487541 / 39.55 | 0.518123 / 61.66 |
| AFPD-OQR | 0.447829 / 16.25 | 0.487254 / 39.40 | 0.516912 / 56.28 |
| AFPD-OQR + OccupancyFree | 0.447829 / 16.25 | 0.487254 / 39.40 | 0.516912 / 56.28 |
| AFPD-OQR + OccupancyAware | 0.447829 / 16.25 | 0.487254 / 39.40 | 0.516912 / 56.28 |

Occupancy 两变体在 Fixed-call 与 AFPD-OQR 完全相同，即 gate 没有改变 top-K 选择。Fixed-call 25%/40% 仍由 Current-OQ 最高。

#### Fixed-call AP50 / AP75

| 方法 | 10% AP50 / AP75 | 25% AP50 / AP75 | 40% AP50 / AP75 |
| --- | ---: | ---: | ---: |
| Current-OQ | 0.618464 / 0.489224 | **0.664371 / 0.535036** | **0.698116 / 0.571110** |
| Current-OQR | 0.619310 / 0.488308 | 0.662295 / 0.533077 | 0.696461 / 0.568263 |
| AFPD-OQ | **0.619377 / 0.489337** | 0.663919 / 0.533764 | 0.697178 / 0.568883 |
| AFPD-OQR | 0.617638 / 0.488650 | 0.663304 / 0.533417 | 0.695439 / 0.568945 |
| AFPD-OQR + OccupancyFree | 0.617638 / 0.488650 | 0.663304 / 0.533417 | 0.695439 / 0.568945 |
| AFPD-OQR + OccupancyAware | 0.617638 / 0.488650 | 0.663304 / 0.533417 | 0.695439 / 0.568945 |

### 26.5 Full val Fixed-byte 动作数量

| 方法 | 预算 | Original | Q90 | ROI | 总上传图片 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current-OQ | 10% | 38 | 1,189 | 0 | 1,227 |
| Current-OQ | 25% | 81 | 2,611 | 0 | 2,692 |
| Current-OQ | 40% | 313 | 3,647 | 0 | 3,960 |
| Current-OQR | 10% | 28 | 1,111 | 218 | 1,357 |
| Current-OQR | 25% | 58 | 2,452 | 386 | 2,896 |
| Current-OQR | 40% | 259 | 3,511 | 427 | 4,197 |
| AFPD-OQ | 10% | 23 | 1,163 | 0 | 1,186 |
| AFPD-OQ | 25% | 40 | 2,620 | 0 | 2,660 |
| AFPD-OQ | 40% | 156 | 3,816 | 0 | 3,972 |
| AFPD-OQR | 10% | 19 | 978 | 383 | 1,380 |
| AFPD-OQR | 25% | 28 | 2,213 | 793 | 3,034 |
| AFPD-OQR | 40% | 166 | 3,295 | 927 | 4,388 |
| AFPD-OQR + OccupancyFree | 10% | 20 | 993 | 351 | 1,364 |
| AFPD-OQR + OccupancyFree | 25% | 28 | 2,242 | 720 | 2,990 |
| AFPD-OQR + OccupancyFree | 40% | 167 | 3,354 | 796 | 4,317 |
| AFPD-OQR + OccupancyAware | 10% | 20 | 984 | 366 | 1,370 |
| AFPD-OQR + OccupancyAware | 25% | 28 | 2,237 | 729 | 2,994 |
| AFPD-OQR + OccupancyAware | 40% | 167 | 3,355 | 793 | 4,315 |

AFPD-OQR 相比 Current-OQR 更激进地使用 ROI：10%/25%/40% 分别多 165/407/500 个 ROI。这在 40% 有利，但在 10% 明显伤害 AP。

### 26.6 2×2 归因消融

#### Fixed-byte ΔAP

| 归因项 | 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| ROI effect under Current：Current-OQR - Current-OQ | +0.002267 | +0.003755 | +0.001896 |
| AFPD objective under OQ：AFPD-OQ - Current-OQ | -0.001818 | +0.000430 | +0.000776 |
| AFPD objective under OQR：AFPD-OQR - Current-OQR | -0.004805 | -0.001799 | +0.001154 |
| AFPD total：AFPD-OQR - Current-OQ | -0.002538 | +0.001956 | +0.003050 |
| OccupancyFree - AFPD-OQR | -0.000066 | +0.001342 | +0.000236 |
| OccupancyAware - AFPD-OQR | -0.000115 | +0.000989 | +0.000052 |

#### Fixed-call ΔAP

| 归因项 | 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| Current-OQR - Current-OQ | -0.000047 | -0.001936 | -0.001760 |
| AFPD-OQ - Current-OQ | +0.000983 | -0.001626 | -0.000996 |
| AFPD-OQR - Current-OQR | -0.000121 | +0.000025 | -0.000447 |
| AFPD-OQR - Current-OQ | -0.000168 | -0.001912 | -0.002207 |
| OccupancyFree/Aware - AFPD-OQR | 0 | 0 | 0 |

**归因结论：**

- ROI 动作本身并不是 COCO 上必然有害；在同次 Current 学习器中，OQR 的三个 Fixed-byte 点全部高于 OQ；
- AFPD 目标单独使用 OQ 没有稳定增益；
- AFPD 与 ROI 的交互只在 40% 较好，10% 明显失配；
- OccupancyFree 25% 点估计有 +0.001342，但 bootstrap 尚不支持确定优势；
- AFPD development OOF 的三点提升没有转化为 full val 上对 Current-OQR 的统一提升。

### 26.7 同协议冻结外部参考

| 方法 | 10% AP | 25% AP | 40% AP |
| --- | ---: | ---: | ---: |
| Raw ActionValue-OQ（历史冻结） | 0.472192 | 0.525455 | 0.559262 |
| ORIC-MLP SharedAction-NoROI | 0.437167 | 0.484735 | 0.519733 |
| MORIC-MLP SharedAction-NoROI | 0.446856 | 0.494336 | 0.522596 |

独立重训 Current-OQR 相对历史冻结 Raw ActionValue-OQ 的点差为 `+0.008299 / +0.007184 / +0.001920 AP`。它的绝对点估计在三个预算都更高，但因为训练 run 不同，不应将差异纯粹归因为“加 ROI”或某个网络结构。这正是后续 6 组 same-run bootstrap 必须补做的原因。

---

## 27. AFPD/Occupancy 完整 21 组 paired bootstrap

所有比较均为 1,000 次 image-level paired bootstrap。`Pr(Δ>0)` 是重采样中差值为正的比例，不是传统 p-value。

### 27.1 原 15 组：AFPD 对历史冻结参考及 Occupancy 对 AFPD

| 候选 | 对手 | 预算 | 均值 ΔAP | 95% CI | Pr(Δ>0) | CI 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| AFPD-OQR | Raw ActionValue-OQ（历史冻结） | 10% | +0.003923 | [-0.001068, +0.009208] | 0.937 | 跨 0 |
| AFPD-OQR | Raw ActionValue-OQ（历史冻结） | 25% | +0.005521 | [+0.001296, +0.009420] | 0.993 | 正 |
| AFPD-OQR | Raw ActionValue-OQ（历史冻结） | 40% | +0.003001 | [+0.001067, +0.004869] | 0.998 | 正 |
| AFPD-OQR | MORIC-MLP | 10% | +0.029013 | [+0.023436, +0.034838] | 1.000 | 正 |
| AFPD-OQR | MORIC-MLP | 25% | +0.036609 | [+0.030680, +0.043003] | 1.000 | 正 |
| AFPD-OQR | MORIC-MLP | 40% | +0.039525 | [+0.035322, +0.044456] | 1.000 | 正 |
| AFPD-OQR | ORIC-MLP | 10% | +0.038403 | [+0.032296, +0.044204] | 1.000 | 正 |
| AFPD-OQR | ORIC-MLP | 25% | +0.046293 | [+0.039975, +0.052705] | 1.000 | 正 |
| AFPD-OQR | ORIC-MLP | 40% | +0.042517 | [+0.037999, +0.047508] | 1.000 | 正 |
| OccupancyFree | AFPD-OQR | 10% | -0.000083 | [-0.000679, +0.000361] | 0.407 | 跨 0 |
| OccupancyFree | AFPD-OQR | 25% | +0.001168 | [-0.000329, +0.002935] | 0.928 | 跨 0 |
| OccupancyFree | AFPD-OQR | 40% | +0.000240 | [-0.000310, +0.000790] | 0.800 | 跨 0 |
| OccupancyAware | AFPD-OQR | 10% | -0.000113 | [-0.000368, +0.000107] | 0.191 | 跨 0 |
| OccupancyAware | AFPD-OQR | 25% | +0.000802 | [-0.000674, +0.002565] | 0.776 | 跨 0 |
| OccupancyAware | AFPD-OQR | 40% | +0.000052 | [-0.000472, +0.000547] | 0.585 | 跨 0 |

这 15 组说明 AFPD-OQR 明确高于冻结 ORIC/MORIC，对历史 Raw ActionValue 在 25%/40% 也是正 CI。但后一个结论混合了“新一次重训”与“AFPD 目标”的差异，不能单独归因给 AFPD。OccupancyFree/Aware 对 AFPD 的 6 个 CI 全部跨 0，没有形成可确认的额外提升。

### 27.2 后补 6 组：同次重训纯归因

| 候选 | 同次对手 | 预算 | 点估计 ΔAP | Bootstrap 均值 ΔAP | 95% CI | Pr(Δ>0) | CI 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AFPD-OQR | Current-OQ | 10% | -0.002538 | -0.002432 | [-0.007504, +0.001917] | 0.156 | 跨 0 |
| AFPD-OQR | Current-OQ | 25% | +0.001956 | +0.002129 | [-0.001589, +0.005576] | 0.887 | 跨 0 |
| AFPD-OQR | Current-OQ | 40% | +0.003050 | +0.002936 | [+0.001474, +0.004534] | 1.000 | **正** |
| AFPD-OQR | Current-OQR | 10% | -0.004805 | -0.004497 | [-0.008982, -0.000263] | 0.017 | **负** |
| AFPD-OQR | Current-OQR | 25% | -0.001799 | -0.001681 | [-0.005107, +0.001577] | 0.163 | 跨 0 |
| AFPD-OQR | Current-OQR | 40% | +0.001154 | +0.001049 | [-0.000531, +0.002637] | 0.922 | 跨 0 |

### 27.3 21 组 bootstrap 合并解读

1. **外部基线结论依然很强：** AFPD 对 ORIC/MORIC 六个预算比较的 CI 均明确高于 0。
2. **跨运行 Raw 对比不能用来纯归因 AFPD：** 25%/40% 虽为正 CI，但它同时包含重训差异。
3. **同次归因才直接回答 AFPD 目标是否更强：** 仅 40% vs Current-OQ 明确为正；10% vs Current-OQR 反而明确为负。
4. **Current-OQR 是必须保留的强内部基线：** AFPD 在 10% 使用过多 ROI，25% 也没有证明超越它。
5. **Occupancy Gate 没有定稿为主方法的统计依据：** 所有对 AFPD 的 CI 均跨 0，且点收益最大仅出现在 25% OccupancyFree。

---

## 28. 截至 2026-08-10 的 COCO 总结论

### 28.1 目前最强学习器必须按问题回答

| 问题 | 当前答案 | 数据依据 |
| --- | --- | --- |
| 广泛 post-hoc Fixed-byte 基准中，最强 No-ROI 学习器是谁？ | **Raw ActionValue-OQ** | 0.472192 / 0.525455 / 0.559262，三点均高于 ORIC/MORIC、Confidence、Random 和 DualHead |
| 同次独立重训中，低/中预算最强是谁？ | **Current-OQR** | 10% 0.480491；25% 0.532639 |
| 同次独立重训中，40% 点估计最高是谁？ | **AFPD-OQR + OccupancyFree** | 0.562572，但对 AFPD 的 bootstrap CI 跨 0 |
| AFPD 是否已稳定取代 ActionValue/Current-OQR？ | **否** | calibration gate 失败；10% vs Current-OQR 为明确负 CI；仅 40% vs Current-OQ 为明确正 CI |
| DualHead 是否已被否定？ | **否，但已降为重要消融/中间版本** | canonical registered 三点高于 ORIC 点估计，但总 gate 失败；ActionValue 明显更强 |
| 是否有新隐藏 final 已确认最终超越？ | **没有** | 后续完整 val2017 结果都是 post-hoc reuse |

### 28.2 学习器与策略的当前定位

| 方法/组件 | 当前定位 | 是否建议作论文主方法 | 原因 |
| --- | --- | --- | --- |
| Raw ActionValue-OQ | 当前最强 No-ROI Fixed-byte 候选 | **是，当前首选** | 图片—动作联合估值，完整基准三预算均最高 |
| Current-OQR | 最强简单 ROI 同次基线 | **是，必须保留** | 独立重训 10%/25% 最高，ROI 三个 Fixed-byte 点均增益 |
| AFPD-OQR | 创新目标候选/高预算候选 | **可作研究主线，不能宣称全面替代** | OOF 三点通过；full val 仅高预算较好，低预算失败 |
| OccupancyFree/Aware | AFPD 的次要消融 | 否 | 6 个对 AFPD 的 CI 全跨 0 |
| Per-action Isotonic OQ | 校准消融 | 否 | calibration 有效，full val 三点均略低于 Raw |
| Canonical DualHead | 价值/成本解耦的重要中间版 | 作消融 | 信号质量优于 Current-BudgetSet，但两阶段架构不如 ActionValue |
| SetValue / Current-BudgetSet | DualHead 的结构消融 | 否 | 用于证明成本不应混入 Value |
| RelActionRank | 低预算训练目标诊断 | 暂不作主方法 | 10% 有 +0.002528，但总 gate 失败 |
| FactorizedGlobalActionRank | 失败消融 | 否 | 25%/40% 明显退化 |
| Risk/LCB / DualHead-v2 | 失败消融 | 否 | 过度保守或全面低于 ActionValue |
| SharedAction | 旧两阶段动作基线 | 作消融 | 用于证明联合 ActionValue 的改进 |
| CostEligible-AllROI | 激进 ROI 动作基线 | 作消融 | COCO 上通常降 AP，但能降字节 |
| GBDT-Hierarchical | 保守 ROI gate 基线 | 作消融 | 部分协议中与 AllROI 完全同选择 |
| ORIC/MORIC | 强外部学习基线 | **必须保留** | Fixed-call 仍强，是证明我们方法价值的主要对手 |
| Confidence / Random | 经典可解释基线 | **必须保留** | 提供非学习和弱学习参考 |

### 28.3 已经可以确定的技术结论

1. **两阶段架构是主要早期瓶颈。** 将图片 ranking 与动作选择改成 action-conditioned joint value，在 OOF/calibration Fixed-byte 上带来约 +0.024–+0.040 AP，幅度远大于其他结构小改。
2. **成本应与 Value 解耦。** payload 混入 Value 会让模型偏向便宜图；更稳定的做法是 Value 预测精度价值，exact-capacity scheduler 在推理时管理 bytes。
3. **COCO 的主协议应以 Fixed-byte 为主。** Raw ActionValue 在 Fixed-byte 明显超过外部方法，但 Fixed-call 仍不如 ORIC/MORIC，不能混为“全协议全面超越”。
4. **ROI 不是简单的好/坏二分。** 历史 AllROI 通常伤 AP；但同次 Current-OQR 在 Fixed-byte 三点都高于 Current-OQ，证明受控 ROI 有真实价值。
5. **AFPD 的训练目标有研究新意，但实证尚不够稳定。** development OOF 三预算一致提升，但 calibration 失败，full val 只在高预算展现优势。
6. **继续增加静态 gate/head 的预期价值已很低。** Occupancy、shadow-price、residual、consensus、Risk/LCB、DualHead-v2 都没有稳定突破，共同指向静态 Edge 检测集的可观测性上限。
7. **所有“当前最强”都是已消费 val2017 上的 post-hoc 结论。** 真正的论文确认仍需要新的互斥 holdout/数据集或严格跨数据集证据。

### 28.4 事实、推断和研究判断分层

#### 已验证事实

- Raw ActionValue-OQ 在完整 post-hoc Fixed-byte 表的三个点均是最高点估计；
- AFPD OOF v1 在三种子、三个 Fixed-byte 预算都高于当时 CurrentActionValue-OQ；
- AFPD one-shot calibration gate 失败，独立重训 full val 也未统一超过 Current-OQR；
- same-run bootstrap 中，AFPD-OQR 只在 40% vs Current-OQ 获得明确正 CI，10% vs Current-OQR 获得明确负 CI；
- Occupancy 对 AFPD 的六个 CI 全部跨 0。

#### 基于多轮证据的推断

- 当前剩余误差主要来自哪张图/哪个动作进容量的 membership 判断，不是简单 O/Q 校准；
- 低预算下 ROI 可用性很稀疏，AFPD 对 ROI 的激进选择超过了当前静态特征可靠辨别的范围；
- 40% 时更高 ROI 覆盖对 AP75 有帮助，这是 AFPD 高预算收益的合理机制，但仍需新数据证实。

#### 研究判断

- 若论文主题是 No-ROI 泛化与最完整的外部基线对比，应选 **Raw ActionValue-OQ**；若追求这次独立重训的最高低/中预算点，应同时保留 **Current-OQR**；
- AFPD 适合作为“更创新但尚未完全确认”的主要研究候选，并与 Current-OQR 做强归因；
- 不建议继续在 COCO val2017 上开发 v7 或新 gate，否则只会增加 post-hoc 选择偏差。

---

## 29. 当前论文组织与后续实验建议

### 29.1 主结果建议

建议将 COCO 结果组织成三层：

1. **主方法表：Raw ActionValue-OQ Fixed-byte**
   - 对比 ORIC、MORIC、Confidence、Random、DualHead；
   - 报告 10%/25%/40% AP50–95、AP50、AP75、KB、动作分布和容量利用率；
   - Fixed-call 作为辅助，如实披露其不超过 ORIC/MORIC。
2. **架构消融表：Current-BudgetSet → SetValue → DualHead → ActionValue**
   - 证明价值/成本解耦和联合图片—动作调度是主要改进；
   - SharedAction、CostEligible-AllROI、GBDT-Hierarchical 作为动作框架消融。
3. **创新目标/风险分析表：Current-OQ/OQR × AFPD-OQ/OQR × Occupancy**
   - OOF 用于说明 AFPD frontier-path 监督的动机；
   - calibration 和 full val 必须同时报告，不能只报 OOF 正结果；
   - 21 组 bootstrap 中，same-run 6 组应用于主归因，历史冻结 15 组用于外部参考。

### 29.2 可以使用的严谨措辞

> 在已消费的 COCO val2017 post-hoc Fixed-byte 基准上，Raw ActionValue-OQ 在 10%、25% 和 40% 三个字节预算的点估计均高于冻结 ORIC/MORIC、Confidence、Random 与 DualHead 基线。AFPD 在 strict development OOF 上三预算一致提升，但一次性 calibration 未通过，独立重训 full val 显示其收益集中于 40% 高预算，未形成对 Current-OQR 的全预算稳定超越。

### 29.3 不应使用的措辞

- “AFPD 已全面取代 Raw ActionValue”；
- “Occupancy Gate 显著提高 AFPD”；
- “我们在新隐藏 COCO final 上确认超过 ORIC/MORIC”；
- “ROI 在 COCO 必然有害”或“ROI 在 COCO 必然有益”；
- “Fixed-byte 的优势等价于 Fixed-call 全面优势”；
- “在相邻文献未发现同构方法”等价于“已证明全球首创”。

### 29.4 唯一值得的下一步确认

COCO val2017 不应再用于选择新版本。下一步应当：

1. 冻结 Raw ActionValue-OQ、Current-OQR 和 AFPD-OQR 的完整定义；
2. 不再根据 COCO val2017 调参；
3. 在新的互斥 holdout 或严格跨数据集协议上同时报告三者；
4. 主门槛保持 Fixed-byte 至少两个预算 `>=+0.001 AP`，其余 `>=-0.001 AP`，并使用 paired bootstrap；
5. 若 AFPD 仍只在 40% 提升，将它定位为高预算/ROI 消融，不继续开发 v7 静态 gate。

### 29.5 主要结果溯源路径

| 内容 | 源实验/报告 |
| --- | --- |
| 原 9 Ranker 三动作 | `paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration/` |
| No-ROI 16 Ranker 全表 | `paper/experiments/20260805_coco_noroi_fullranker_canonical9678_v1/` |
| Budget-conditioned contextual ranker | `paper/experiments/20260804_coco_budget_contextual_ranker_v2/notes/results_zh.md` |
| BudgetSet holdout | `paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/notes/results_zh.md` |
| Value/Cost OOF | `paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/notes/results_zh.md` |
| Canonical registered DualHead | `paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/` |
| DualHead 三动作扩展 | `paper/experiments/20260805_coco_dualhead_canonical9678_triaction_extension_v1/` |
| ActionValue/Risk OOF | `paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1/notes/results_zh.md` |
| ROI 贡献诊断 | `paper/experiments/20260805_coco_actionvalue_roi_contribution_diagnostic_v1/notes/results_zh.md` |
| Selection-context Teacher | `paper/experiments/20260805_coco_selection_context_jointaction_teacher_v1/notes/results_zh.md` |
| DualHead-v2 | `paper/experiments/20260806_coco_dualheadv2_noroi_oof_v1/` |
| Relational ActionValue | `paper/experiments/20260806_coco_relational_actionvalue_oof_v1/notes/results_zh.md` |
| Factorized Global ActionRank | `paper/experiments/20260806_coco_factorized_global_actionrank_oof_v1/notes/results_zh.md` |
| 跨动作校准 | `paper/experiments/20260809_three_dataset_cross_action_calibration_v1/` |
| COCO 完整 scheduler benchmark | `paper/experiments/20260809_coco_oregon_full_scheduler_benchmark_v1/` |
| AFPD v1–v6 综合 | `paper/experiments/20260810_coco_actionvalue_innovation_synthesis_v1/RESULTS_ZH.md` |
| AFPD/Occupancy calibration | `paper/experiments/20260810_coco_afpd_occupancy_paper_protocol_v1/notes/RESULTS_ZH.md` |
| AFPD 独立重训 full val + 15 bootstrap | `paper/experiments/20260810_coco_afpd_forced_fullval_posthoc_v1/` |
| AFPD 同次归因 6 bootstrap | `paper/experiments/20260810_coco_afpd_same_run_baseline_bootstrap_v1/` |

---

**最终一句话：** COCO 上已经确立的最强 **No-ROI 方向**是 **Raw ActionValue-OQ 的图片—动作联合价值学习 + exact Fixed-byte 调度**；同次重训显示 Current-OQR 是强而简单的 ROI 基线；AFPD 提供了更有创新性的 frontier-path 监督，但目前只有高预算局部优势，仍需新互斥数据确认，不能替代 Current-OQR/Raw ActionValue 成为全预算已证实最优方法。
