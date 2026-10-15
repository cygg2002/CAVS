# COCO 数据集实验结果汇总

## 1. 实验口径

本文档汇总 COCO 数据集上的主要可比实验结果，统一采用以下评估口径：

- 数据集：COCO final，共 5,000 张图像；
- 主评价指标：Official mAP50–95；
- 同时记录：Official mAP50、Official mAP75；
- 预算模式：`fixed_call`；
- 固定调用预算：10%、25%、40%；
- 通信成本指标：`Request KB/image`；
- 内部排序方法：9 个 Ranker；
- 外部学习器：ORIC-MLP、MORIC-MLP；
- 统一动作策略：
  - SharedAction；
  - CostEligible-AllROI；
  - GBDT-Hierarchical；
- 外部方法另外保留严格 Original 整图上传结果。

下列表格中，`AP / KB` 表示：

`Official mAP50–95 / Request KB·image⁻¹`

---

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

## 8. COCO 结果总体分析

### 8.1 外部 ORIC/MORIC 的排序优势较稳定

COCO 上的排序结果具有较强一致性：

- 10% 预算：ORIC-MLP 最优；
- 25% 预算：MORIC-MLP 最优；
- 40% 预算：MORIC-MLP 最优。

这一规律在 Original、SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical 四种载荷策略下均保持不变。

说明在 COCO 数据集上，外部 ORIC/MORIC 学习器对高价值上云样本的排序能力整体优于当前内部 Ranker。

### 8.2 Current-Controlled-32D 是当前最强内部 Ranker

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

因此，Current-Controlled-32D 可以作为当前内部 Ranker 的主要代表方法。

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

## 9. COCO 当前主要结论

1. **外部 ORIC/MORIC 在 COCO 上具有稳定的排序优势。**
   - 10% 预算由 ORIC-MLP 最优；
   - 25% 和 40% 预算由 MORIC-MLP 最优。

2. **Current-Controlled-32D 是当前最强的内部 Ranker。**
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

## 10. 结果使用建议

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
