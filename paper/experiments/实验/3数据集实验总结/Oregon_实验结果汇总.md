# Oregon 数据集实验结果汇总

## 1. 实验范围与指标口径

本文档包含原 fixed-call 三动作实验、Cloud-V2 teacher ceiling，以及
2026-08-07 新增的 fixed-byte 跨数据集学习器诊断。不同实验的
10%/25%/40% 预算语义不同，不能直接混表比较。

### 1.1 原三动作 fixed-call 实验

- 数据集：Oregon final，共 4,962 张图像；
- 主评价指标：Official mAP50–95；
- 同时记录：Official mAP50、Official mAP75；
- 预算模式：`fixed_call`；
- 固定调用预算：10%、25%、40%；
- 对应选中图像数与实际上传率：
  - 10%：496 张，实际上传率约 9.996%；
  - 25%：1,241 张，实际上传率约 25.010%；
  - 40%：1,985 张，实际上传率约 40.004%；
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

### 1.2 最新 fixed-byte 学习器诊断

- fit：9,923 张；calibration：4,961 张；
- 主模式：`fixed_byte`，预算表示全 Original 字节容量的 10%/25%/40%；
- 学习器：Current-BudgetSet、SetValue、DualHead、ActionValue-OQ-Joint；
- 动作：Edge、Original、Q90，主诊断不包含 ROI；
- Track A 不访问 final，Track B 继续采用 post-hoc exploratory final reuse；
- 详细结果见第 19 节。

### Oregon 结果使用限制

当前协议中 Oregon 被标记为：

`posthoc_exploratory_final_reuse`

因此，Oregon 结果应作为探索性跨域证据使用，不能描述为新的独立 hidden test 或新的确认性 final 结果。

---

## 2. Oregon 系统基线

| 系统基线 | mAP50–95 | mAP50 | mAP75 | Request KB/image | 动作构成 |
| --- | ---: | ---: | ---: | ---: | --- |
| Edge-only | 0.4548 | 0.6398 | 0.5041 | 0.00 | 不上云 |
| Cloud-only Original | **0.4991** | 0.6976 | 0.5521 | 1595.49 | 100% Original |
| Always 640-Q90 | 0.4711 | 0.6607 | 0.5315 | 97.98 | 100% Q90 |
| Always ROI fallback | 0.4599 | 0.6454 | 0.5168 | 96.03 | 92.87% Q90 + 7.13% ROI |
| Shared ContextM2E all | 0.4796 | 0.6746 | 0.5334 | 1037.00 | 59.61% Original + 40.39% Q90 |

其中两个主要端点为：

- Edge-only：mAP50–95 = **0.4548**
- Cloud-only Original：mAP50–95 = **0.4991**

二者的 mAP50–95 差距约为：

`0.4991 - 0.4548 = 0.0443`

---

## 3. SharedAction：所有 Ranker 结果

SharedAction 使用同一冻结动作策略，因此该设置主要用于比较不同 Ranker 对“哪些图像值得上云”的排序能力。

### 3.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4603 / **11.92** | 0.4687 / 163.11 | **0.4842 / 396.85** |
| 内部 | ClassGainPrior | 0.4502 / 143.30 | 0.4509 / 369.85 | 0.4554 / 572.21 |
| 内部 | M2E | 0.4638 / 138.57 | 0.4743 / 334.40 | 0.4778 / 538.60 |
| 内部 | Confidence-Residual M2E | 0.4633 / 147.72 | 0.4739 / 337.02 | 0.4774 / 537.70 |
| 内部 | SemanticRank | 0.4626 / 193.95 | 0.4704 / 452.55 | 0.4783 / 666.77 |
| 内部 | MORIC-style | 0.4620 / 124.51 | 0.4731 / 275.17 | 0.4822 / 448.15 |
| 内部 | **GEC-style** | **0.4652 / 95.92** | **0.4759 / 282.53** | 0.4796 / 494.57 |
| 内部 | Current-Controlled-32D | 0.4590 / 186.67 | 0.4642 / 452.28 | 0.4754 / 667.16 |
| 内部 | Random | 0.4580 / 103.68 | 0.4639 / 258.20 | 0.4673 / 414.75 |
| 外部 | ORIC-MLP | 0.4536 / 115.26 | 0.4537 / 293.20 | 0.4795 / 465.08 |
| 外部 | **MORIC-MLP** | **0.4677 / 89.73** | 0.4700 / 254.82 | 0.4840 / 429.05 |

### 3.2 SharedAction 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 0.06% | 9.94% | 0.00% |
| ClassGainPrior | 8.38% | 1.61% | 0.00% |
| M2E | 5.30% | 4.70% | 0.00% |
| Confidence-Residual M2E | 5.54% | 4.45% | 0.00% |
| SemanticRank | 7.70% | 2.30% | 0.00% |
| MORIC-style | 5.93% | 4.07% | 0.00% |
| GEC-style | 3.57% | 6.43% | 0.00% |
| Current-Controlled-32D | 6.57% | 3.43% | 0.00% |
| Random | 5.95% | 4.05% | 0.00% |
| ORIC-MLP | 6.25% | 3.75% | 0.00% |
| MORIC-MLP | 5.40% | 4.59% | 0.00% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 8.28% | 16.73% | 0.00% |
| ClassGainPrior | 20.33% | 4.68% | 0.00% |
| M2E | 14.25% | 10.76% | 0.00% |
| Confidence-Residual M2E | 14.53% | 10.48% | 0.00% |
| SemanticRank | 18.48% | 6.53% | 0.00% |
| MORIC-style | 13.72% | 11.29% | 0.00% |
| GEC-style | 12.21% | 12.80% | 0.00% |
| Current-Controlled-32D | 18.18% | 6.83% | 0.00% |
| Random | 14.86% | 10.15% | 0.00% |
| ORIC-MLP | 15.74% | 9.27% | 0.00% |
| MORIC-MLP | 14.47% | 10.54% | 0.00% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 21.89% | 18.12% | 0.00% |
| ClassGainPrior | 32.14% | 7.86% | 0.00% |
| M2E | 24.49% | 15.52% | 0.00% |
| Confidence-Residual M2E | 24.63% | 15.38% | 0.00% |
| SemanticRank | 29.22% | 10.78% | 0.00% |
| MORIC-style | 22.73% | 17.27% | 0.00% |
| GEC-style | 23.16% | 16.85% | 0.00% |
| Current-Controlled-32D | 29.58% | 10.42% | 0.00% |
| Random | 23.84% | 16.16% | 0.00% |
| ORIC-MLP | 25.49% | 14.51% | 0.00% |
| MORIC-MLP | 24.81% | 15.20% | 0.00% |

### 3.3 SharedAction 结果总结

- 10% 预算下，MORIC-MLP 最优，mAP50–95 = **0.4677**。
- 25% 预算下，GEC-style 最优，mAP50–95 = **0.4759**。
- 40% 预算下，Confidence 最优，mAP50–95 = **0.4842**。
- 40% 下 Confidence 与 MORIC-MLP 非常接近：
  - Confidence = 0.484184；
  - MORIC-MLP = 0.484006；
  - 差约 0.000178。
- SharedAction 在 Oregon 上没有产生 ROI 动作，所有被选中的图像均采用 Original 或 Q90。

---

## 4. CostEligible-AllROI：所有 Ranker 结果

CostEligible-AllROI 将满足成本条件的候选更积极地切换为 ROI。Oregon 上该策略可以降低通信成本，但通常会伴随一定 AP 损失。

### 4.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4603 / **11.92** | 0.4664 / 153.06 | **0.4777 / 372.70** |
| 内部 | ClassGainPrior | 0.4466 / 135.93 | 0.4457 / 351.72 | 0.4492 / 543.17 |
| 内部 | M2E | 0.4633 / 132.74 | 0.4717 / 319.19 | 0.4745 / 515.26 |
| 内部 | Confidence-Residual M2E | 0.4629 / 142.32 | 0.4712 / 321.30 | 0.4738 / 513.50 |
| 内部 | SemanticRank | 0.4619 / 185.63 | 0.4693 / 437.47 | 0.4747 / 640.06 |
| 内部 | MORIC-style | 0.4591 / 112.80 | 0.4688 / 255.91 | 0.4765 / 422.01 |
| 内部 | **GEC-style** | **0.4648 / 90.61** | **0.4733 / 266.68** | 0.4749 / 469.66 |
| 内部 | Current-Controlled-32D | 0.4587 / 182.45 | 0.4632 / 435.64 | 0.4725 / 640.20 |
| 内部 | Random | 0.4570 / 99.19 | 0.4612 / 246.69 | 0.4627 / 396.12 |
| 外部 | ORIC-MLP | 0.4526 / 109.80 | 0.4515 / 281.00 | 0.4749 / 444.63 |
| 外部 | **MORIC-MLP** | **0.4650 / 83.80** | 0.4641 / 240.36 | 0.4760 / 404.05 |

### 4.2 CostEligible-AllROI 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 0.06% | 9.94% | 0.00% |
| ClassGainPrior | 7.42% | 1.49% | 1.09% |
| M2E | 4.82% | 4.70% | 0.48% |
| Confidence-Residual M2E | 5.10% | 4.45% | 0.44% |
| SemanticRank | 7.03% | 2.30% | 0.67% |
| MORIC-style | 4.70% | 3.91% | 1.39% |
| GEC-style | 3.08% | 6.43% | 0.48% |
| Current-Controlled-32D | 6.29% | 3.43% | 0.28% |
| Random | 5.30% | 4.01% | 0.68% |
| ORIC-MLP | 5.54% | 3.71% | 0.75% |
| MORIC-MLP | 4.72% | 4.49% | 0.79% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 7.01% | 16.57% | 1.43% |
| ClassGainPrior | 17.90% | 4.55% | 2.56% |
| M2E | 12.45% | 10.68% | 1.87% |
| Confidence-Residual M2E | 12.66% | 10.48% | 1.87% |
| SemanticRank | 17.05% | 6.49% | 1.47% |
| MORIC-style | 11.39% | 11.12% | 2.50% |
| GEC-style | 10.44% | 12.68% | 1.89% |
| Current-Controlled-32D | 16.63% | 6.83% | 1.55% |
| Random | 13.19% | 10.08% | 1.74% |
| ORIC-MLP | 14.11% | 9.15% | 1.75% |
| MORIC-MLP | 12.54% | 10.38% | 2.10% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 18.68% | 17.96% | 3.37% |
| ClassGainPrior | 28.17% | 7.70% | 4.13% |
| M2E | 21.58% | 15.38% | 3.04% |
| Confidence-Residual M2E | 21.58% | 15.22% | 3.20% |
| SemanticRank | 26.14% | 10.62% | 3.24% |
| MORIC-style | 19.39% | 17.11% | 3.51% |
| GEC-style | 20.05% | 16.69% | 3.26% |
| Current-Controlled-32D | 26.70% | 10.42% | 2.88% |
| Random | 21.15% | 16.03% | 2.82% |
| ORIC-MLP | 22.67% | 14.31% | 3.02% |
| MORIC-MLP | 21.36% | 15.01% | 3.63% |

### 4.3 CostEligible-AllROI 结果总结

- 10% 预算下，MORIC-MLP 最优，mAP50–95 = **0.4650**。
- 10% 下 MORIC-MLP 与 GEC-style 非常接近：
  - MORIC-MLP = 0.464950；
  - GEC-style = 0.464821；
  - 差约 0.000129。
- 25% 预算下，GEC-style 最优，mAP50–95 = **0.4733**。
- 40% 预算下，Confidence 最优，mAP50–95 = **0.4777**。
- AllROI 能提高 ROI 使用比例并降低通信成本。
- 但与 ENA24 不同，Oregon 上 AllROI 通常会带来一定 AP 下降，因此更偏向通信成本优化。

---

## 5. GBDT-Hierarchical：所有 Ranker 结果

GBDT-Hierarchical 对 ROI 使用更加保守。在 Oregon 上，其结果通常与 SharedAction 非常接近。

### 5.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.4603 / **11.92** | 0.4681 / 162.39 | **0.4835 / 395.11** |
| 内部 | ClassGainPrior | 0.4499 / 142.83 | 0.4506 / 369.31 | 0.4552 / 571.18 |
| 内部 | M2E | 0.4637 / 138.28 | 0.4743 / 333.41 | 0.4777 / 537.13 |
| 内部 | Confidence-Residual M2E | 0.4634 / 147.65 | 0.4739 / 336.36 | 0.4772 / 536.21 |
| 内部 | SemanticRank | 0.4625 / 193.49 | 0.4704 / 451.72 | 0.4781 / 665.02 |
| 内部 | MORIC-style | 0.4614 / 124.01 | 0.4725 / 273.58 | 0.4815 / 446.56 |
| 内部 | **GEC-style** | **0.4651 / 95.63** | **0.4758 / 281.55** | 0.4789 / 493.08 |
| 内部 | Current-Controlled-32D | 0.4590 / 186.50 | 0.4641 / 451.45 | 0.4754 / 665.63 |
| 内部 | Random | 0.4580 / 103.47 | 0.4638 / 257.62 | 0.4672 / 413.79 |
| 外部 | ORIC-MLP | 0.4535 / 115.01 | 0.4530 / 292.60 | 0.4788 / 464.20 |
| 外部 | **MORIC-MLP** | **0.4672 / 89.38** | 0.4693 / 253.94 | 0.4832 / 427.53 |

### 5.2 GBDT-Hierarchical 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 0.06% | 9.94% | 0.00% |
| ClassGainPrior | 8.34% | 1.51% | 0.14% |
| M2E | 5.26% | 4.70% | 0.04% |
| Confidence-Residual M2E | 5.52% | 4.45% | 0.02% |
| SemanticRank | 7.64% | 2.30% | 0.06% |
| MORIC-style | 5.84% | 3.91% | 0.24% |
| GEC-style | 3.53% | 6.43% | 0.04% |
| Current-Controlled-32D | 6.55% | 3.43% | 0.02% |
| Random | 5.92% | 4.03% | 0.04% |
| ORIC-MLP | 6.23% | 3.71% | 0.06% |
| MORIC-MLP | 5.36% | 4.49% | 0.14% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 8.16% | 16.57% | 0.28% |
| ClassGainPrior | 20.27% | 4.57% | 0.16% |
| M2E | 14.11% | 10.68% | 0.22% |
| Confidence-Residual M2E | 14.43% | 10.48% | 0.10% |
| SemanticRank | 18.34% | 6.49% | 0.18% |
| MORIC-style | 13.52% | 11.12% | 0.36% |
| GEC-style | 12.09% | 12.68% | 0.24% |
| Current-Controlled-32D | 18.08% | 6.83% | 0.10% |
| Random | 14.78% | 10.12% | 0.11% |
| ORIC-MLP | 15.68% | 9.15% | 0.18% |
| MORIC-MLP | 14.35% | 10.38% | 0.28% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 21.66% | 17.96% | 0.38% |
| ClassGainPrior | 32.02% | 7.76% | 0.22% |
| M2E | 24.30% | 15.38% | 0.32% |
| Confidence-Residual M2E | 24.45% | 15.22% | 0.34% |
| SemanticRank | 29.00% | 10.62% | 0.38% |
| MORIC-style | 22.53% | 17.11% | 0.36% |
| GEC-style | 22.97% | 16.69% | 0.34% |
| Current-Controlled-32D | 29.40% | 10.42% | 0.18% |
| Random | 23.71% | 16.10% | 0.19% |
| ORIC-MLP | 25.39% | 14.37% | 0.24% |
| MORIC-MLP | 24.61% | 15.03% | 0.36% |

### 5.3 GBDT-Hierarchical 结果总结

- 10% 预算下，MORIC-MLP 最优，mAP50–95 = **0.4672**。
- 25% 预算下，GEC-style 最优，mAP50–95 = **0.4758**。
- 40% 预算下，Confidence 最优，mAP50–95 = **0.4835**。
- GBDT 在 Oregon 上实际使用的 ROI 比例非常低，最大约为总图像的 0.38%。
- 因此 GBDT-Hierarchical 的行为非常接近 SharedAction，通信优化幅度也较有限。

---

## 6. 外部方法严格 Original 结果

该组实验中，被选中的图像全部执行 Original 整图上传，不使用 SharedAction、CostEligible-AllROI 或 GBDT-Hierarchical。

### 6.1 固定调用预算结果

| 外部方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| DCSB-Score-Adapted | 0.4620 / 166.99 | 0.4766 / 402.54 | 0.4794 / 642.87 |
| ORIC-MLP | 0.4593 / 166.87 | 0.4667 / 420.39 | 0.4897 / 660.80 |
| **MORIC-MLP** | **0.4695 / 153.96** | **0.4793 / 404.07** | **0.5001 / 647.90** |

严格 Original 下，MORIC-MLP 在三个固定预算点均获得最高 mAP50–95。

特别是 40%：

- MORIC-MLP = **0.500100**
- Cloud-only Original = **0.499095**

差约：

`+0.001005`

但这一差距很小，而且 Oregon 是 post-hoc exploratory reuse，因此不能据此宣称 MORIC-MLP 显著超过 Cloud-only。

更稳妥的表述是：

> 40% Original MORIC-MLP 在 Oregon 探索性结果中达到与 Cloud-only 相当的 AP。

---

## 7. DCSB-Native 自然工作点

DCSB-Native 不采用固定 10%、25%、40% 预算，而由原生规则自然产生上传比例。

| Payload | 自然上传率 | mAP50–95 | Request KB/image | 动作构成 |
| --- | ---: | ---: | ---: | --- |
| Original | 11.21% | **0.4653** | 189.31 | 11.21% Original |
| SharedAction | 11.21% | 0.4642 | 177.56 | 10.16% Original + 1.05% Q90 |

相比 Edge-only：

`0.4653 - 0.4548 ≈ +0.0105`

因此，与 ENA24 不同，DCSB-Native 在 Oregon 上具有正收益。

---

## 8. 内部 Ranker 与外部 ORIC/MORIC 的直接比较

### 8.1 10% 固定调用预算

| 动作策略 | 最佳内部 | AP | 最佳外部 | AP | 结果 |
| --- | --- | ---: | --- | ---: | --- |
| SharedAction | GEC-style | 0.4652 | MORIC-MLP | **0.4677** | 外部更高 |
| CostEligible-AllROI | GEC-style | 0.4648 | MORIC-MLP | **0.4650** | 基本持平 |
| GBDT-Hierarchical | GEC-style | 0.4651 | MORIC-MLP | **0.4672** | 外部更高 |

10% 低预算下，MORIC-MLP 整体最强，说明其对最头部少量高收益图像的识别能力较强。

### 8.2 25% 固定调用预算

| 动作策略 | 最佳内部 | AP | 最佳外部 | AP | 内部优势 |
| --- | --- | ---: | --- | ---: | ---: |
| SharedAction | **GEC-style** | **0.4759** | MORIC-MLP | 0.4700 | +0.0058 |
| CostEligible-AllROI | **GEC-style** | **0.4733** | MORIC-MLP | 0.4641 | +0.0092 |
| GBDT-Hierarchical | **GEC-style** | **0.4758** | MORIC-MLP | 0.4693 | +0.0065 |

25% 是 Oregon 上内部方法优势最明确的预算点。

GEC-style 在三种共享动作策略下全部超过 ORIC-MLP 和 MORIC-MLP。

### 8.3 40% 固定调用预算

| 动作策略 | 最佳内部 | AP | 最佳外部 | AP | 内部优势 |
| --- | --- | ---: | --- | ---: | ---: |
| SharedAction | **Confidence** | **0.4842** | MORIC-MLP | 0.4840 | +0.0002 |
| CostEligible-AllROI | **Confidence** | **0.4777** | MORIC-MLP | 0.4760 | +0.0017 |
| GBDT-Hierarchical | **Confidence** | **0.4835** | MORIC-MLP | 0.4832 | +0.0003 |

40% 下 Confidence 表面上在三种动作策略下均为第一，但 SharedAction 和 GBDT-Hierarchical 中的差距都非常小。

因此，更准确的结论是：

> 40% 下内部 Confidence 与外部 MORIC-MLP 基本达到相近水平，AllROI 下 Confidence 的优势相对更明显。

---

## 9. 不同动作策略下的最优结果

### 9.1 10% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | MORIC-MLP | **0.4695** | 153.96 |
| SharedAction | MORIC-MLP | 0.4677 | 89.73 |
| CostEligible-AllROI | MORIC-MLP | 0.4650 | **83.80** |
| GBDT-Hierarchical | MORIC-MLP | 0.4672 | 89.38 |

### 9.2 25% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | MORIC-MLP | **0.4793** | 404.07 |
| SharedAction | GEC-style | 0.4759 | 282.53 |
| CostEligible-AllROI | GEC-style | 0.4733 | **266.68** |
| GBDT-Hierarchical | GEC-style | 0.4758 | 281.55 |

### 9.3 40% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | MORIC-MLP | **0.5001** | 647.90 |
| SharedAction | Confidence | 0.4842 | 396.85 |
| CostEligible-AllROI | Confidence | 0.4777 | **372.70** |
| GBDT-Hierarchical | Confidence | 0.4835 | 395.11 |

总体规律非常清楚：

> Original 通常获得最高 AP，而共享三动作策略显著降低通信成本，但 Oregon 上会损失一定 AP。

---

## 10. Oregon 三种动作策略的总体特点

### 10.1 SharedAction

SharedAction 在当前三种统一动作策略中精度总体最好。

代表性结果：

- MORIC-MLP 10%：0.4677 / 89.73 KB；
- GEC-style 25%：0.4759 / 282.53 KB；
- Confidence 40%：0.4842 / 396.85 KB。

### 10.2 CostEligible-AllROI

CostEligible-AllROI 可以降低通信成本，但 Oregon 上不像 ENA24 那样形成明显的 Pareto 改善。

以 MORIC-MLP 为例：

| 预算 | SharedAction AP / KB | AllROI AP / KB |
| --- | ---: | ---: |
| 10% | 0.4677 / 89.73 | 0.4650 / 83.80 |
| 25% | 0.4700 / 254.82 | 0.4641 / 240.36 |
| 40% | 0.4840 / 429.05 | 0.4760 / 404.05 |

说明 AllROI 在降低通信量的同时也带来了 AP 下降。

### 10.3 GBDT-Hierarchical

GBDT-Hierarchical 的结果几乎贴近 SharedAction。

以 MORIC-MLP 为例：

| 预算 | SharedAction AP / KB | GBDT AP / KB |
| --- | ---: | ---: |
| 10% | 0.4677 / 89.73 | 0.4672 / 89.38 |
| 25% | 0.4700 / 254.82 | 0.4693 / 253.94 |
| 40% | 0.4840 / 429.05 | 0.4832 / 427.53 |

原因在于当前 GBDT 实际允许的 ROI 数量非常少。

因此：

> Oregon 上当前 GBDT-Hierarchical 的风险控制较保守，但通信优化作用也十分有限。

---

## 11. Oregon 独有：Cloud-V2 ROI Profile × Scale 扫描

该组实验与前面的可部署 Ranker 结果口径不同，不能直接作为主方法结果混入上述表格。

该实验使用：

- GT-aware proxy；
- 多种 ROI profile；
- 多种 cloud inference scale；
- Profile × Scale 联合候选。

因此，该实验反映的是：

> ROI 动作空间的 teacher ceiling / 可学习上限。

它不是已经可部署的学习器性能。

该实验的数据角色为：

- calibration：4,961 张；
- audit：9,923 张；
- registered final 未访问。

---

## 12. Cloud-V2 Calibration 基础点

| Official AP | AP50 | AP75 | KB/image |
| ---: | ---: | ---: | ---: |
| 0.501968 | 0.715654 | 0.551097 | 1482.11 |

---

## 13. Cloud-V2 单候选关键结果

| 候选 | 工作点 | ROI | AP | ΔAP | KB/image | ΔKB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical@320 | 最大 AP | 15% | **0.522691** | +0.020723 | 1299.21 | −182.90 |
| Historical@320 | 40% | 40% | 0.511411 | +0.009444 | 995.61 | −486.50 |
| Moderate@320 | 最大 AP | 10% | 0.519400 | +0.017432 | 1347.87 | −134.24 |
| Moderate@320 | 冻结 AP-safe | 40% | 0.505948 | +0.003981 | **961.90** | −520.21 |
| Historical@640 | 最大 AP | 10% | 0.515686 | +0.013719 | 1357.52 | −124.59 |
| Historical@640 | 40% | 40% | 0.482670 | **−0.019298** | 1005.52 | −476.59 |

该结果表明：

> 640 并不是 ROI 云端推理的默认最佳尺度。

Historical@320 明显优于 Historical@640。

---

## 14. Calibration 多候选 teacher ceiling

| 工作点 | ROI | Official AP | AP50 | AP75 | ΔAP | KB/image | ΔKB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 最大 AP | 30% | **0.547861** | 0.734240 | 0.598647 | **+0.045893** | 1084.95 | −397.16 |
| 最低 KB AP-safe | 40% | 0.545551 | 0.732127 | 0.596823 | +0.043584 | **954.56** | −527.55 |

40% 工作点的尺度比例：

- 160：58.0%
- 320：31.3%
- 640：10.8%

Profile 比例：

- Current：51.8%
- Historical：23.7%
- fixed-1.8：14.4%
- Moderate：10.0%

说明收益并非来自单一 ROI 策略，而是来自：

> ROI profile 与 cloud inference scale 的联合适配。

---

## 15. Audit teacher ceiling

Audit 基础结果：

| Official AP | AP50 | AP75 | KB/image |
| ---: | ---: | ---: | ---: |
| 0.518897 | 0.728753 | 0.560833 | 1470.61 |

冻结后的 teacher 结果：

| 配置 | ROI | AP | AP50 | AP75 | ΔAP | KB/image | ΔKB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Moderate@320 单候选 teacher | 40% | 0.523224 | 0.716405 | 0.572331 | +0.004327 | 950.43 | −520.18 |
| **12 候选联合 teacher** | 40% | **0.567346** | **0.755201** | **0.615554** | **+0.048449** | **938.41** | **−532.20** |

其中 12 候选联合 teacher 相比 Audit 基础点：

- mAP50–95：0.518897 → **0.567346**
- ΔAP = **+0.048449**
- Request KB/image：1470.61 → **938.41**
- ΔKB = **−532.20**

这是 Oregon 最近实验中非常重要的结果。

但必须保持严格边界：

> 这是 GT-aware teacher ceiling，不是实际部署学习器性能。

因此不能直接作为“本文方法最终结果”写入主实验表。

它证明的是：

> 现有 ROI Profile × Scale 动作空间仍有很大的未开发潜力。

---

## 16. Oregon 原阶段结果总体分析

### 16.1 低预算下 MORIC-MLP 更强

10% 固定调用预算下：

- SharedAction：MORIC-MLP = 0.4677；
- AllROI：MORIC-MLP = 0.4650；
- GBDT：MORIC-MLP = 0.4672。

说明 MORIC-MLP 对最头部少量高收益样本的识别能力较强。

### 16.2 中等预算下 GEC-style 优势最明确

25% 固定调用预算下：

- SharedAction：GEC-style = 0.4759；
- AllROI：GEC-style = 0.4733；
- GBDT：GEC-style = 0.4758。

三种统一动作策略下均超过外部 ORIC/MORIC。

因此 Oregon 25% 是内部 GEC-style 最有说服力的工作点。

### 16.3 高预算下 Confidence 与 MORIC-MLP 基本相当

40%：

- SharedAction：Confidence 0.4842 vs MORIC 0.4840；
- GBDT：Confidence 0.4835 vs MORIC 0.4832；
- AllROI：Confidence 0.4777 vs MORIC 0.4760。

SharedAction 和 GBDT 中的优势非常小，因此不宜强调“显著超过”。

### 16.4 Original 能获得最高 AP，但通信成本明显更高

例如 40%：

- MORIC-MLP + Original：0.5001 / 647.90 KB；
- Confidence + SharedAction：0.4842 / 396.85 KB；
- Confidence + AllROI：0.4777 / 372.70 KB；
- Confidence + GBDT：0.4835 / 395.11 KB。

这体现了 Oregon 上明显的精度—通信成本折中。

### 16.5 当前 ROI 学习策略还没有发挥动作空间潜力

主三动作实验中：

- AllROI 会明显损失 AP；
- GBDT 又极度保守，ROI 比例过低。

但 Profile × Scale teacher ceiling 表明：

- AP 可以提高约 0.048；
- 同时通信量可以降低约 532 KB/image。

因此问题并不是“ROI 本身没有价值”，而更可能是当前系统尚未充分联合建模：

- ROI profile；
- crop context；
- cloud inference scale；
- ROI safe decision。

---

## 17. Oregon 原阶段主要结论

1. **10% 低预算下，外部 MORIC-MLP 最强。**

2. **25% 预算下，内部 GEC-style 最强。**
   - SharedAction、AllROI、GBDT 三种共享动作下均超过外部 ORIC/MORIC。

3. **40% 预算下，内部 Confidence 与外部 MORIC-MLP 基本达到相近水平。**

4. **严格 Original 下 MORIC-MLP 表现非常强。**
   - 40% 达到 0.5001；
   - 与 Cloud-only Original 的 0.4991 基本相当。

5. **Original 能获得更高 AP，但通信成本明显高于共享三动作。**

6. **SharedAction 是当前三种统一动作策略中精度最稳定的方案。**

7. **CostEligible-AllROI 能降低通信量，但 Oregon 上通常伴随 AP 损失。**

8. **GBDT-Hierarchical 当前过于保守。**
   - ROI 使用比例极低；
   - 通信优化效果有限；
   - 结果非常接近 SharedAction。

9. **Oregon 的主三动作实验尚未充分发挥 ROI 潜力。**

10. **Cloud-V2 ROI Profile × Scale teacher ceiling 表明动作空间具有很高上限。**
    - Audit AP：0.518897 → **0.567346**
    - Request KB/image：1470.61 → **938.41**

11. **当前下一步更值得关注的是联合学习 ROI profile、crop context 和 cloud inference scale，而不是简单扩大 ROI 使用比例。**

12. **所有 Oregon 主实验结果必须保留 `posthoc_exploratory_final_reuse` 限制。**
    - 不能将其表述为新的独立 hidden test。

---

## 18. 原阶段结果使用建议

如果用于论文正文，建议优先保留以下结果。

### 18.1 Ranker 公平比较

采用：

- SharedAction；
- fixed_call；
- 10%、25%、40%；
- 内部 9 个 Ranker；
- ORIC-MLP；
- MORIC-MLP；
- Random。

这一设置能够最大程度隔离 Ranker 本身的排序能力。

### 18.2 外部方法基线

单独报告：

- DCSB-Score-Adapted + Original；
- ORIC-MLP + Original；
- MORIC-MLP + Original；
- DCSB-Native 自然工作点。

### 18.3 动作策略消融

重点比较：

- SharedAction；
- CostEligible-AllROI；
- GBDT-Hierarchical。

建议至少保留以下代表 Ranker：

- GEC-style；
- Confidence；
- MORIC-MLP；
- ORIC-MLP；
- Random。

### 18.4 ROI Profile × Scale teacher ceiling

该组结果建议作为：

- 动作空间上限分析；
- Oracle / teacher ceiling；
- 后续方法设计依据。

不应与可部署 Ranker 主结果放在同一张主表中直接比较。

### 18.5 Oregon 的论文措辞限制

涉及 Oregon 时建议明确写成：

> exploratory post-hoc Oregon evaluation

或：

> post-hoc exploratory evaluation on Oregon

避免使用：

- independent hidden test；
- untouched final test；
- confirmatory external validation。

Cloud-V2 joint action + reversible fusion audit 已完成，但 registered final 仍未
访问。该结果见第 19.6 节，并与本节的原 fixed-call final 表分开报告。

---

## 19. 最新跨数据集学习器诊断（2026-08-07）

本节补充 `20260807_ena24_oregon_dualhead_crossdataset_v1` 和
`20260807_three_dataset_actionvalue_roi_gate_v1` 的结果。Track A 使用
fit OOF 与独立 calibration，不访问 final；Track B 复用已经使用过的 Oregon
final，因此继续遵循 `posthoc_exploratory_final_reuse` 限制。

### 19.1 Track A 协议

- 学习器训练：canonical fit=9,923 张 Oregon 图像；5-fold OOF，3 个 seeds；
- calibration：4,961 张互斥图像；
- 动作：Edge、Original、Q90；Track A 有意排除 ROI；
- 主模式：`fixed_byte`，次模式：`fixed_call`；
- final：Track A 未访问 final；
- 指标：Official mAP50–95 和 Request KB/image。

### 19.2 Track A：Fixed-byte OOF 与 calibration

#### 5-fold OOF

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.490038 / 148.02 | 0.499167 / 370.04 | 0.506361 / 592.06 |
| SetValue | 0.491507 / 148.01 | 0.499988 / 370.01 | 0.505976 / 592.05 |
| DualHead | 0.491459 / 148.01 | 0.500979 / 370.01 | 0.508267 / 592.07 |
| ActionValue-OQ-Joint | **0.511769 / 148.02** | **0.511184 / 370.04** | **0.514295 / 592.07** |

#### Frozen calibration

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.468057 / 154.45 | 0.474473 / 386.02 | 0.478787 / 617.83 |
| SetValue | 0.462707 / 154.46 | 0.475274 / 386.13 | 0.479161 / 617.78 |
| DualHead | 0.462685 / 154.46 | 0.476772 / 386.14 | 0.478955 / 617.83 |
| ActionValue-OQ-Joint | **0.471968 / 154.45** | **0.479579 / 386.14** | **0.480083 / 617.83** |

ActionValue-OQ-Joint 在 Oregon 三个 fixed-byte 工作点均为四个新学习器中的
最高结果。相对 DualHead 的提升为：

- 10%：`+0.009283 AP`；
- 25%：`+0.002807 AP`；
- 40%：`+0.001128 AP`。

相对 Current-BudgetSet 的提升为：

- 10%：`+0.003910 AP`；
- 25%：`+0.005106 AP`；
- 40%：`+0.001296 AP`。

### 19.3 Track A：Fixed-call calibration

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | **0.445211 / 71.36** | **0.456251 / 225.02** | 0.473865 / 412.36 |
| SetValue | 0.433718 / 80.94 | 0.454675 / 262.51 | **0.476193 / 463.99** |
| DualHead | 0.436507 / 83.82 | 0.454488 / 269.52 | 0.471081 / 475.22 |
| ActionValue-OQ-Joint | 0.433287 / 163.43 | 0.454844 / 411.32 | 0.464568 / 662.37 |

ActionValue 在 Oregon fixed-call 下没有优势，并且更倾向选择昂贵 Original。
因此它的正结果来自 fixed-byte 条件下的图片—动作—成本联合调度，不能描述为
通用 top-K Ranker 优势。

### 19.4 Track B：DualHead-Value 接入既有动作框架

Track B 将 DualHead-Value 的图像分数接入原有冻结的 SharedAction、
CostEligible-AllROI 和 GBDT-Hierarchical。每个工作点进行 200 次 paired
bootstrap；所有结果属于 post-hoc exploratory final reuse。

| 动作框架 | 预算 | DualHead AP / KB | 历史最强比较器 | 比较器 AP | ΔAP | ΔKB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| SharedAction | 10% | 0.463902 / 97.24 | GEC-style | 0.465205 | -0.001302 | +1.32 |
| CostEligible-AllROI | 10% | 0.462529 / 88.66 | GEC-style | 0.464821 | -0.002293 | -1.95 |
| GBDT-Hierarchical | 10% | 0.463367 / 97.00 | GEC-style | 0.465108 | -0.001741 | +1.36 |
| SharedAction | 25% | 0.475685 / 280.97 | GEC-style | 0.475861 | -0.000176 | -1.56 |
| CostEligible-AllROI | 25% | 0.471245 / 263.66 | GEC-style | 0.473315 | -0.002070 | -3.03 |
| GBDT-Hierarchical | 25% | 0.474993 / 280.08 | GEC-style | 0.475830 | -0.000837 | -1.47 |
| SharedAction | 40% | 0.471235 / 475.86 | Confidence | 0.484184 | -0.012949 | +79.01 |
| CostEligible-AllROI | 40% | 0.463457 / 450.79 | Confidence | 0.477718 | -0.014260 | +78.09 |
| GBDT-Hierarchical | 40% | 0.470455 / 474.34 | Confidence | 0.483519 | -0.013064 | +79.23 |

#### Paired bootstrap

| 动作框架 | 预算 | 比较器 | 均值 ΔAP | 95% CI | P(Δ>0) |
| --- | ---: | --- | ---: | ---: | ---: |
| SharedAction | 10% | GEC-style | -0.001113 | [-0.006728, +0.005111] | 0.375 |
| CostEligible-AllROI | 10% | GEC-style | -0.002177 | [-0.007879, +0.003886] | 0.260 |
| GBDT-Hierarchical | 10% | GEC-style | -0.001627 | [-0.007281, +0.004735] | 0.325 |
| SharedAction | 25% | GEC-style | +0.000350 | [-0.006034, +0.006497] | 0.555 |
| CostEligible-AllROI | 25% | GEC-style | -0.001596 | [-0.007908, +0.004074] | 0.285 |
| GBDT-Hierarchical | 25% | GEC-style | -0.000350 | [-0.006828, +0.005760] | 0.430 |
| SharedAction | 40% | Confidence | -0.010796 | [-0.020198, +0.000300] | 0.030 |
| CostEligible-AllROI | 40% | Confidence | -0.012156 | [-0.022599, -0.000785] | 0.010 |
| GBDT-Hierarchical | 40% | Confidence | -0.010903 | [-0.020352, +0.000202] | 0.030 |

`P(Δ>0)` 表示 Bootstrap 重采样中 ΔAP 为正的比例，不是传统假设检验的 p-value。

10% 和 25% 的 paired bootstrap 区间均跨越 0，说明 DualHead 最多只能与
GEC-style 接近，不能证明更强。40% 的三个点估计均下降约 0.013 至 0.014 AP，
且通信量增加约 78 至 79 KB/image；其中 AllROI 的 95% ΔAP 区间完全低于 0。

### 19.5 OQ/OQR ROI gate 诊断

该诊断使用 canonical Oregon v1 动作缓存，不使用 Cloud-V2 Profile × Scale
候选，也不访问 final。fit OOF 冻结 AdvantageGate，calibration 只评测一次。

| 预算 | OQ AP | OQR-Naive AP / ΔAP | Naive ROI | AdvantageGate AP / ΔAP | Gate ROI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.471968 | 0.473128 / +0.001160 | 0 | 0.473128 / +0.001160 | 0 |
| 25% | 0.479579 | 0.480165 / +0.000586 | 0 | 0.480165 / +0.000586 | 0 |
| 40% | 0.480083 | 0.479663 / -0.000420 | 0 | 0.479663 / -0.000420 | 0 |

OQR-Naive 和 AdvantageGate 都没有选择任何 ROI。表中的微小 ΔAP 来自
四动作训练改变了 Original/Q90 的分数与排序，不能解释为 ROI 收益。因此
canonical Oregon v1 不支持将 ROI 直接加入统一 ActionValue 动作空间。

该结论不否定前文 Cloud-V2 Profile × Scale teacher ceiling。两者的区别是：

- canonical v1 只使用单一历史 ROI 动作；
- Cloud-V2 teacher 使用多 Profile × 多推理尺度，并由 GT-aware Oracle 选动作；
- teacher ceiling 证明动作空间存在潜力，当前学习结果证明该潜力尚未转化为
  可部署调度收益。

### 19.6 Cloud-V2 Joint Action + Reversible Fusion audit

`20260728_oregon_cloudv2_joint_action_reversible_fusion` 已完成 audit；三个
10%/25%/40% 预算结果相同，因为学习器只触发了 7.66% 的 MAUR ROI，预算上限
没有成为约束。registered final 未访问。

| 配置 | Official AP | AP50 | AP75 | KB/image | ROI 比例 | ΔAP | ΔKB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q90/Original Base | 0.518897 | 0.728753 | 0.560833 | 1470.61 | 0.00% | 0.000000 | 0.00 |
| Joint Action + MAUR | 0.517851 | 0.728205 | 0.560141 | 1392.42 | 7.66% | -0.001046 | -78.20 |
| Joint Action + Reversible | 0.518771 | 0.728769 | 0.560734 | 1462.36 | 0.86% | -0.000126 | -8.25 |

注册主结果 MAUR 未通过 `ΔAP >= -0.001` 门槛，仅差 0.000046 AP；Reversible
满足 AP-safe，但通信节省只有约 0.56%。该结果说明可逆融合能够缓解破坏式
替换，却没有解决学习器过于保守和动作选择不准的问题。

### 19.7 Oregon 更新后的结论

1. `ActionValue-OQ-Joint` 是 Oregon fixed-byte 下最强的新学习器，三个预算
   均超过 Current-BudgetSet、SetValue 和 DualHead。
2. ActionValue 在 fixed-call 下明显不强，其创新价值应限定为面向真实字节容量的
   图片—动作联合调度，而不是通用图像 Ranker。
3. `DualHead` 接入 SharedAction、AllROI、GBDT 后没有超过历史 Ranker；40%
   出现明显 AP 与通信成本双重退化，应仅保留为消融。
4. canonical OQR gate 没有触发任何 ROI；若继续 Oregon ROI 研究，应单独使用
   Cloud-V2 Profile × Scale 扩展动作空间，不能把两套动作混为一个结果。
5. COCO 的 16-Ranker No-ROI completeness extension 尚未在 Oregon 上执行，
   Track A 只是四个新学习器的泛化诊断。
6. 所有 Track B 和历史 final 表仍必须标记为 exploratory post-hoc Oregon
   evaluation，不能称为独立 hidden test。

### 19.8 论文使用建议

- Oregon 的新方法表优先报告 fixed-byte ActionValue-OQ-Joint；
- fixed-call 表用于说明 ActionValue 的优势来自成本约束联合调度；
- DualHead、SetValue、Current-BudgetSet 作为内部消融；
- 旧九 Ranker + 三动作表继续作为历史排序和动作策略对比；
- Cloud-V2 Profile × Scale 仅作为 teacher ceiling，不能与可部署结果混表；
- Joint Action + MAUR/Reversible 作为 audit 消融，不能表述为 registered final；
- OQR gate 作为 canonical ROI 负结果报告。
