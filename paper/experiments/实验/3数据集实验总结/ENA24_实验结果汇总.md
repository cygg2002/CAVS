# ENA24 数据集实验结果汇总

## 1. 实验范围与指标口径

本文档包含原 fixed-call 三动作实验和 2026-08-07 新增的 fixed-byte
跨数据集学习器诊断。两组实验的 10%/25%/40% 预算语义不同，必须分别解释。

### 1.1 原三动作 fixed-call 实验

- 数据集：ENA24 final，共 1,758 张图像；
- 主评价指标：Official mAP50–95；
- 同时记录：Official mAP50、Official mAP75；
- 预算模式：`fixed_call`；
- 固定调用预算：10%、25%、40%；
- 对应选中图像数与实际上传率：
  - 10%：176 张，实际上传率 10.0114%；
  - 25%：440 张，实际上传率 25.0284%；
  - 40%：703 张，实际上传率 39.9886%；
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

- fit：1,400 张；calibration：400 张；
- 主模式：`fixed_byte`，预算表示全 Original 字节容量的 10%/25%/40%；
- 学习器：Current-BudgetSet、SetValue、DualHead、ActionValue-OQ-Joint；
- 动作：Edge、Original、Q90，主诊断不包含 ROI；
- Track A 不访问 final，Track B 为 post-hoc final 表扩展；
- 详细结果见第 13 节。

---

## 2. ENA24 系统基线

| 系统基线 | mAP50–95 | mAP50 | mAP75 | Request KB/image | 动作构成 |
| --- | ---: | ---: | ---: | ---: | --- |
| Edge-only | 0.7127 | 0.9560 | 0.8341 | 0.00 | 不上云 |
| Cloud-only Original | 0.7418 | 0.9591 | 0.8589 | 433.29 | 100% Original |
| Always 640-Q90 | 0.7369 | 0.9545 | 0.8584 | 94.34 | 100% Q90 |
| Always ROI fallback | 0.7373 | 0.9542 | 0.8563 | 89.91 | 83.90% Q90 + 16.10% ROI |
| Shared ContextM2E all | **0.7459** | **0.9606** | **0.8662** | 343.68 | 66.84% Original + 33.16% Q90 |

其中两个基础端点为：

- Edge-only：mAP50–95 = **0.7127**
- Cloud-only Original：mAP50–95 = **0.7418**

Edge-only 与 Cloud-only Original 的 mAP50–95 差距约为：

`0.7418 - 0.7127 = 0.0291`

因此，ENA24 的端侧检测器本身已经较强，端云协同可获得的剩余精度空间相对有限。

---

## 3. SharedAction：所有 Ranker 结果

SharedAction 使用同一个冻结动作策略，因此该设置主要用于比较不同 Ranker 对“哪些图像值得上云”的排序能力。

### 3.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.7099 / 40.96 | 0.7175 / 103.56 | 0.7217 / 162.26 |
| 内部 | ClassGainPrior | 0.7215 / 36.04 | 0.7269 / 93.30 | 0.7341 / 148.31 |
| 内部 | M2E | 0.7176 / 41.92 | 0.7247 / 100.39 | 0.7252 / 151.48 |
| 内部 | Confidence-Residual M2E | 0.7169 / 42.94 | 0.7256 / 101.26 | 0.7258 / 153.69 |
| 内部 | SemanticRank | 0.7200 / 45.58 | 0.7213 / 107.72 | 0.7277 / 163.79 |
| 内部 | MORIC-style | 0.7149 / 41.71 | 0.7248 / 104.71 | 0.7225 / 156.36 |
| 内部 | GEC-style | 0.7185 / 42.29 | 0.7263 / 102.15 | 0.7265 / 158.01 |
| 内部 | **Current-Controlled-32D** | **0.7215 / 43.80** | **0.7302 / 103.53** | **0.7355 / 162.53** |
| 内部 | Random | 0.7158 / 34.30 | 0.7209 / 86.07 | 0.7253 / 137.38 |
| 外部 | ORIC-MLP | 0.7149 / 36.67 | 0.7223 / 95.64 | 0.7281 / 154.09 |
| 外部 | MORIC-MLP | 0.7098 / 36.45 | 0.7181 / 90.26 | 0.7265 / 142.54 |

### 3.2 SharedAction 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 8.19% | 1.82% | 0.00% |
| ClassGainPrior | 6.71% | 3.30% | 0.00% |
| M2E | 8.19% | 1.82% | 0.00% |
| Confidence-Residual M2E | 8.19% | 1.82% | 0.00% |
| SemanticRank | 8.82% | 1.19% | 0.00% |
| MORIC-style | 8.76% | 1.25% | 0.00% |
| GEC-style | 8.42% | 1.59% | 0.00% |
| Current-Controlled-32D | 8.76% | 1.25% | 0.00% |
| Random | 6.64% | 3.37% | 0.00% |
| ORIC-MLP | 7.74% | 2.28% | 0.00% |
| MORIC-MLP | 7.57% | 2.45% | 0.00% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 21.56% | 3.47% | 0.00% |
| ClassGainPrior | 18.20% | 6.83% | 0.00% |
| M2E | 20.82% | 4.21% | 0.00% |
| Confidence-Residual M2E | 21.05% | 3.98% | 0.00% |
| SemanticRank | 21.73% | 3.30% | 0.00% |
| MORIC-style | 22.01% | 3.01% | 0.00% |
| GEC-style | 21.05% | 3.98% | 0.00% |
| Current-Controlled-32D | 20.53% | 4.49% | 0.00% |
| Random | 16.73% | 8.30% | 0.00% |
| ORIC-MLP | 19.23% | 5.80% | 0.00% |
| MORIC-MLP | 18.15% | 6.88% | 0.00% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 33.90% | 6.09% | 0.00% |
| ClassGainPrior | 27.93% | 12.06% | 0.00% |
| M2E | 30.72% | 9.27% | 0.00% |
| Confidence-Residual M2E | 31.17% | 8.82% | 0.00% |
| SemanticRank | 33.45% | 6.54% | 0.00% |
| MORIC-style | 32.54% | 7.45% | 0.00% |
| GEC-style | 32.20% | 7.79% | 0.00% |
| Current-Controlled-32D | 32.48% | 7.51% | 0.00% |
| Random | 26.66% | 13.33% | 0.00% |
| ORIC-MLP | 30.49% | 9.50% | 0.00% |
| MORIC-MLP | 28.33% | 11.66% | 0.00% |

### 3.3 SharedAction 结果总结

- 10% 预算下，Current-Controlled-32D 的 mAP50–95 最高，为 **0.721522**。
- 10% 下 ClassGainPrior 为 **0.721507**，与 Current-Controlled-32D 仅相差约 **0.000015**，两者实际上近似持平。
- 25% 预算下，Current-Controlled-32D 最优，为 **0.730205**。
- 40% 预算下，Current-Controlled-32D 最优，为 **0.735506**。
- 在统一 SharedAction 条件下，内部 Ranker 在三个预算点均超过 ORIC-MLP 和 MORIC-MLP。
- SharedAction 在 ENA24 上没有产生 ROI 动作，所有被选中的图像均采用 Original 或 Q90。

---

## 4. CostEligible-AllROI：所有 Ranker 结果

CostEligible-AllROI 将满足成本条件的候选更积极地切换为 ROI。在 ENA24 上，该策略不仅能够降低通信成本，而且部分工作点的 AP 还略有提高。

### 4.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.7111 / 32.90 | 0.7189 / 79.95 | 0.7223 / 128.81 |
| 内部 | **ClassGainPrior** | **0.7225 / 28.16** | 0.7279 / 74.40 | **0.7358 / 119.03** |
| 内部 | M2E | 0.7184 / 30.68 | 0.7249 / 71.26 | 0.7254 / 111.06 |
| 内部 | Confidence-Residual M2E | 0.7180 / 30.74 | 0.7259 / 72.52 | 0.7264 / 115.99 |
| 内部 | SemanticRank | 0.7211 / 31.71 | 0.7215 / 75.68 | 0.7279 / 122.12 |
| 内部 | MORIC-style | 0.7165 / 28.39 | 0.7260 / 76.51 | 0.7230 / 123.49 |
| 内部 | GEC-style | 0.7193 / 31.71 | 0.7270 / 73.27 | 0.7271 / 118.30 |
| 内部 | **Current-Controlled-32D** | 0.7216 / 32.10 | **0.7305 / 80.41** | 0.7354 / 127.95 |
| 内部 | Random | 0.7158 / 29.18 | 0.7208 / 72.86 | 0.7253 / 116.32 |
| 外部 | ORIC-MLP | 0.7156 / 27.77 | 0.7227 / 75.45 | 0.7279 / 124.38 |
| 外部 | MORIC-MLP | 0.7108 / 27.61 | 0.7177 / 72.25 | 0.7265 / 113.98 |

### 4.2 CostEligible-AllROI 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 6.20% | 1.59% | 2.22% |
| ClassGainPrior | 4.49% | 3.19% | 2.33% |
| M2E | 5.29% | 1.71% | 3.01% |
| Confidence-Residual M2E | 5.12% | 1.71% | 3.19% |
| SemanticRank | 5.46% | 1.08% | 3.47% |
| MORIC-style | 5.12% | 1.02% | **3.87%** |
| GEC-style | 5.75% | 1.48% | 2.79% |
| Current-Controlled-32D | 5.92% | 1.19% | 2.90% |
| Random | 5.29% | 3.15% | 1.57% |
| ORIC-MLP | 5.35% | 2.16% | 2.50% |
| MORIC-MLP | 5.23% | 2.33% | 2.45% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 15.42% | 3.19% | 6.43% |
| ClassGainPrior | 13.14% | 6.14% | 5.75% |
| M2E | 13.20% | 3.87% | 7.96% |
| Confidence-Residual M2E | 13.48% | 3.64% | 7.91% |
| SemanticRank | 13.59% | 3.07% | **8.36%** |
| MORIC-style | 14.62% | 2.73% | 7.68% |
| GEC-style | 13.48% | 3.70% | 7.85% |
| Current-Controlled-32D | 14.68% | 4.38% | 5.97% |
| Random | 13.22% | 7.80% | 4.01% |
| ORIC-MLP | 13.94% | 5.57% | 5.52% |
| MORIC-MLP | 13.42% | 6.20% | 5.40% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 25.03% | 5.63% | 9.33% |
| ClassGainPrior | 20.25% | 11.21% | 8.53% |
| M2E | 20.14% | 8.87% | 10.98% |
| Confidence-Residual M2E | 21.27% | 8.48% | 10.24% |
| SemanticRank | 22.58% | 6.03% | **11.38%** |
| MORIC-style | 23.83% | 6.77% | 9.39% |
| GEC-style | 21.79% | 7.45% | 10.75% |
| Current-Controlled-32D | 23.55% | 7.34% | 9.10% |
| Random | 21.09% | 12.51% | 6.38% |
| ORIC-MLP | 22.70% | 9.10% | 8.19% |
| MORIC-MLP | 20.82% | 10.75% | 8.42% |

### 4.3 CostEligible-AllROI 结果总结

- 10% 预算下，ClassGainPrior 最优：
  - mAP50–95 = **0.722521**
  - Request KB/image = **28.16**
- 25% 预算下，Current-Controlled-32D 最优：
  - mAP50–95 = **0.730518**
  - Request KB/image = **80.41**
- 40% 预算下，ClassGainPrior 最优：
  - mAP50–95 = **0.735844**
  - Request KB/image = **119.03**
- 与 SharedAction 相比，AllROI 显著增加 ROI 比例并降低通信成本。
- 与 COCO 不同，ENA24 上 AllROI 并未表现出系统性的精度下降，部分工作点的 AP 反而略有提高。
- 因此，CostEligible-AllROI 是 ENA24 上目前最有价值的动作策略之一。

---

## 5. GBDT-Hierarchical：所有 Ranker 结果

GBDT-Hierarchical 对 ROI 使用更保守，通信成本和检测精度通常处于 SharedAction 与 CostEligible-AllROI 之间。

### 5.1 主结果

| 来源 | Ranker | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | --- | ---: | ---: | ---: |
| 内部 | Confidence | 0.7102 / 35.90 | 0.7177 / 86.65 | 0.7213 / 138.03 |
| 内部 | **ClassGainPrior** | **0.7219 / 30.73** | 0.7263 / 80.34 | 0.7336 / 127.84 |
| 内部 | M2E | 0.7179 / 33.95 | 0.7245 / 79.74 | 0.7248 / 122.99 |
| 内部 | Confidence-Residual M2E | 0.7172 / 34.47 | 0.7255 / 80.57 | 0.7256 / 127.15 |
| 内部 | SemanticRank | 0.7201 / 36.43 | 0.7211 / 85.26 | 0.7270 / 135.07 |
| 内部 | MORIC-style | 0.7156 / 32.97 | 0.7250 / 84.46 | 0.7222 / 132.74 |
| 内部 | GEC-style | 0.7185 / 34.82 | 0.7260 / 82.17 | 0.7262 / 129.81 |
| 内部 | **Current-Controlled-32D** | 0.7213 / 35.76 | **0.7295 / 87.85** | **0.7351 / 140.11** |
| 内部 | Random | 0.7157 / 30.66 | 0.7208 / 76.80 | 0.7250 / 122.59 |
| 外部 | ORIC-MLP | 0.7154 / 30.84 | 0.7221 / 82.09 | 0.7273 / 132.76 |
| 外部 | MORIC-MLP | 0.7102 / 30.81 | 0.7171 / 77.42 | 0.7255 / 122.21 |

### 5.2 GBDT-Hierarchical 动作构成

#### 10% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 6.94% | 1.71% | 1.37% |
| ClassGainPrior | 5.18% | 3.24% | 1.59% |
| M2E | 6.14% | 1.82% | 2.05% |
| Confidence-Residual M2E | 6.09% | 1.82% | 2.10% |
| SemanticRank | 6.60% | 1.19% | 2.22% |
| MORIC-style | 6.43% | 1.14% | 2.45% |
| GEC-style | 6.54% | 1.59% | 1.88% |
| Current-Controlled-32D | 6.83% | 1.25% | 1.93% |
| Random | 5.71% | 3.24% | 1.06% |
| ORIC-MLP | 6.09% | 2.22% | 1.71% |
| MORIC-MLP | 5.97% | 2.39% | 1.65% |

#### 25% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 17.24% | 3.30% | 4.49% |
| ClassGainPrior | 14.79% | 6.48% | 3.75% |
| M2E | 15.64% | 4.04% | 5.35% |
| Confidence-Residual M2E | 15.81% | 3.81% | 5.40% |
| SemanticRank | 16.21% | 3.19% | 5.63% |
| MORIC-style | 16.84% | 2.84% | 5.35% |
| GEC-style | 15.98% | 3.81% | 5.23% |
| Current-Controlled-32D | 16.61% | 4.44% | 3.98% |
| Random | 14.33% | 7.96% | 2.74% |
| ORIC-MLP | 15.64% | 5.69% | 3.70% |
| MORIC-MLP | 14.73% | 6.43% | 3.87% |

#### 40% 预算

| Ranker | Full | Q90 | ROI |
| --- | ---: | ---: | ---: |
| Confidence | 27.65% | 5.80% | 6.54% |
| ClassGainPrior | 22.64% | 11.55% | 5.80% |
| M2E | 23.49% | 9.04% | 7.45% |
| Confidence-Residual M2E | 24.40% | 8.65% | 6.94% |
| SemanticRank | 26.17% | 6.20% | 7.62% |
| MORIC-style | 26.51% | 6.94% | 6.54% |
| GEC-style | 25.03% | 7.62% | 7.34% |
| Current-Controlled-32D | 26.91% | 7.39% | 5.69% |
| Random | 22.84% | 12.77% | 4.38% |
| ORIC-MLP | 24.97% | 9.33% | 5.69% |
| MORIC-MLP | 23.04% | 11.09% | 5.86% |

### 5.3 GBDT-Hierarchical 结果总结

- 10% 预算下，ClassGainPrior 最优，mAP50–95 = **0.721858**。
- 25% 预算下，Current-Controlled-32D 最优，mAP50–95 = **0.729456**。
- 40% 预算下，Current-Controlled-32D 最优，mAP50–95 = **0.735059**。
- GBDT-Hierarchical 的 ROI 比例低于 CostEligible-AllROI，因此通信节省更保守。
- 在 ENA24 上，由于 AllROI 本身没有造成明显精度惩罚，因此 GBDT 的保守性并未带来明显精度优势。

---

## 6. 外部方法严格 Original 结果

该组实验中，被选中的图像全部执行 Original 整图上传，不使用 SharedAction、CostEligible-AllROI 或 GBDT-Hierarchical。

### 6.1 固定调用预算结果

| 外部方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| DCSB-Score-Adapted | 0.7125 / 50.55 | 0.7197 / 102.63 | 0.7266 / 169.12 |
| **ORIC-MLP** | **0.7142 / 42.94** | **0.7207 / 112.10** | **0.7267 / 181.37** |
| MORIC-MLP | 0.7095 / 42.96 | 0.7170 / 108.38 | 0.7249 / 173.81 |

严格 Original 下，三个固定预算点均由 ORIC-MLP 获得最高 mAP50–95。

需要注意：

- 10% ORIC-MLP = 0.7142；
- Edge-only = 0.7127；

仅提升约 **0.0016**。

而 10% MORIC-MLP = **0.7095**，低于 Edge-only 的 0.7127。

---

## 7. DCSB-Native 自然工作点

DCSB-Native 不采用固定 10%、25%、40% 预算，而由其原生规则自然产生上传比例。

| Payload | 自然上传率 | mAP50–95 | Request KB/image | 动作构成 |
| --- | ---: | ---: | ---: | --- |
| Original | 4.04% | 0.7078 | 18.50 | 4.04% Original |
| SharedAction | 4.04% | 0.7082 | 16.48 | 3.36% Original + 0.68% Q90 |

DCSB-Native 的两个结果均低于 Edge-only：

- Edge-only：0.7127；
- DCSB-Native + Original：0.7078；
- DCSB-Native + SharedAction：0.7082。

说明 DCSB-Native 在 ENA24 上选择的少量上云样本并没有形成稳定正收益。

---

## 8. 不同动作策略下的最优结果

### 8.1 10% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | ORIC-MLP | 0.7142 | 42.94 |
| SharedAction | Current-Controlled-32D | 0.7215 | 43.80 |
| CostEligible-AllROI | ClassGainPrior | **0.7225** | **28.16** |
| GBDT-Hierarchical | ClassGainPrior | 0.7219 | 30.73 |

10% 下，从精度与通信量综合看，ClassGainPrior + CostEligible-AllROI 是当前最突出工作点。

### 8.2 25% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | ORIC-MLP | 0.7207 | 112.10 |
| SharedAction | Current-Controlled-32D | 0.7302 | 103.53 |
| CostEligible-AllROI | Current-Controlled-32D | **0.7305** | **80.41** |
| GBDT-Hierarchical | Current-Controlled-32D | 0.7295 | 87.85 |

25% 下，Current-Controlled-32D + CostEligible-AllROI 同时获得最高 AP 和明显更低的通信量。

### 8.3 40% 固定调用预算

| 动作策略 | 最优 Ranker | mAP50–95 | Request KB/image |
| --- | --- | ---: | ---: |
| Original | ORIC-MLP | 0.7267 | 181.37 |
| SharedAction | Current-Controlled-32D | 0.7355 | 162.53 |
| CostEligible-AllROI | ClassGainPrior | **0.7358** | **119.03** |
| GBDT-Hierarchical | Current-Controlled-32D | 0.7351 | 140.11 |

40% 下，ClassGainPrior + CostEligible-AllROI 是当前固定调用实验中精度最高且通信成本较低的组合。

---

## 9. 内部 Ranker 与外部 ORIC/MORIC 的直接比较

在公平的 SharedAction 条件下：

| 预算 | 最佳内部 | AP | 最佳外部 | AP | 内部优势 |
| ---: | --- | ---: | --- | ---: | ---: |
| 10% | Current-Controlled-32D | **0.7215** | ORIC-MLP | 0.7149 | +0.0066 |
| 25% | Current-Controlled-32D | **0.7302** | ORIC-MLP | 0.7223 | +0.0080 |
| 40% | Current-Controlled-32D | **0.7355** | ORIC-MLP | 0.7281 | +0.0074 |

因此，与 COCO 不同：

> 在 ENA24 上，当载荷动作统一为 SharedAction 后，内部 Ranker 在 10%、25%、40% 三个预算点全部超过 ORIC-MLP 和 MORIC-MLP。

此外，这种优势并不能简单归因于更高通信量。

例如 10%：

- Current-Controlled-32D：0.7215 / 43.80 KB；
- ClassGainPrior：0.7215 / 36.04 KB；
- ORIC-MLP：0.7149 / 36.67 KB。

ClassGainPrior 与 ORIC-MLP 的通信成本接近，但 AP 高约 0.0066。

---

## 10. ENA24 原三动作结果总体分析

### 10.1 内部 Ranker 在 ENA24 上明显强于外部 ORIC/MORIC

SharedAction 公平比较中：

- 10%：内部最佳 0.7215，外部最佳 0.7149；
- 25%：内部最佳 0.7302，外部最佳 0.7223；
- 40%：内部最佳 0.7355，外部最佳 0.7281。

说明 ENA24 上内部 Ranker 对真正具有端云收益的样本识别更有效。

### 10.2 Current-Controlled-32D 是中高预算最稳定的内部 Ranker

其结果为：

| 动作策略 | 10% | 25% | 40% |
| --- | ---: | ---: | ---: |
| SharedAction | 0.7215 | 0.7302 | 0.7355 |
| CostEligible-AllROI | 0.7216 | 0.7305 | 0.7354 |
| GBDT-Hierarchical | 0.7213 | 0.7295 | 0.7351 |

说明 Current-Controlled-32D 对动作策略变化具有较好的稳定性。

### 10.3 ClassGainPrior 在 ENA24 上具有明显竞争力

特别是：

- AllROI 10%：**0.7225**
- AllROI 40%：**0.7358**
- GBDT 10%：**0.7219**

因此 ClassGainPrior 不应只作为简单启发式基线看待，它在 ENA24 上实际形成了多个最优工作点。

### 10.4 CostEligible-AllROI 是 ENA24 上最有价值的动作策略

与 COCO 上“AllROI 用精度换通信量”的情况不同，ENA24 上 AllROI 可以在降低通信量的同时保持甚至略微提高 AP。

代表性最佳结果：

| 预算 | 组合 | mAP50–95 | Request KB/image |
| ---: | --- | ---: | ---: |
| 10% | ClassGainPrior + AllROI | **0.7225** | **28.16** |
| 25% | Current-Controlled-32D + AllROI | **0.7305** | **80.41** |
| 40% | ClassGainPrior + AllROI | **0.7358** | **119.03** |

因此 ENA24 是当前三个数据集中最支持 ROI 策略的数据集。

### 10.5 GBDT-Hierarchical 更保守，但优势有限

GBDT-Hierarchical：

- ROI 使用比例低于 AllROI；
- 通信量高于 AllROI；
- AP 与 SharedAction 接近。

由于 ENA24 上 AllROI 并没有产生明显 AP 惩罚，因此 GBDT 的保守性在该数据集上没有体现出明显优势。

### 10.6 严格 Original 外部方法整体竞争力有限

Original 下最好的 ORIC-MLP：

- 10%：0.7142；
- 25%：0.7207；
- 40%：0.7267。

这些结果均低于相同预算下内部 Ranker + 三动作策略的最佳结果。

尤其 10% 时，MORIC-MLP 的 0.7095 甚至低于 Edge-only 的 0.7127。

### 10.7 DCSB-Native 在 ENA24 上表现失败

DCSB-Native 自然上传率只有 4.04%，但：

- Original：0.7078；
- SharedAction：0.7082；

都低于 Edge-only 的 0.7127。

因此 ENA24 上不能仅依靠传统阈值式困难样本选择来保证端云收益。

---

## 11. ENA24 原三动作阶段主要结论

1. **内部 Ranker 在 ENA24 上整体强于外部 ORIC/MORIC。**
   - SharedAction 公平比较下，三个预算点全部由内部方法获胜。

2. **Current-Controlled-32D 是中高预算最稳定的内部 Ranker。**

3. **ClassGainPrior 在 ENA24 上非常强。**
   - 10% 和 40% 的 AllROI 最佳结果均来自 ClassGainPrior。

4. **CostEligible-AllROI 是 ENA24 当前最有价值的动作策略。**
   - 能显著降低通信量；
   - 同时保持甚至略微提高 mAP50–95。

5. **GBDT-Hierarchical 更保守，但在 ENA24 上没有明显优于 AllROI。**

6. **严格 Original 的 ORIC/MORIC 整体弱于内部三动作方案。**

7. **DCSB-Native 在 ENA24 上产生负收益。**

8. **ENA24 的 Edge-only 已经很强。**
   - Edge-only = 0.7127；
   - Cloud-only Original = 0.7418；
   - 可提升空间仅约 0.0291。

9. **当前固定调用最佳结果为：**

| 预算 | 最佳组合 | mAP50–95 | Request KB/image |
| ---: | --- | ---: | ---: |
| 10% | ClassGainPrior + CostEligible-AllROI | **0.7225** | **28.16** |
| 25% | Current-Controlled-32D + CostEligible-AllROI | **0.7305** | **80.41** |
| 40% | ClassGainPrior + CostEligible-AllROI | **0.7358** | **119.03** |

---

## 12. 原三动作结果使用建议

如果用于论文正文，建议优先保留以下结果。

### 12.1 Ranker 公平比较

采用：

- SharedAction；
- fixed_call；
- 10%、25%、40%；
- 内部 9 个 Ranker；
- ORIC-MLP；
- MORIC-MLP；
- Random。

这一设置能够最大程度隔离 Ranker 本身的排序能力。

### 12.2 外部方法基线

单独报告：

- DCSB-Score-Adapted + Original；
- ORIC-MLP + Original；
- MORIC-MLP + Original；
- DCSB-Native 自然工作点。

### 12.3 动作策略消融

重点比较：

- SharedAction；
- CostEligible-AllROI；
- GBDT-Hierarchical。

建议至少保留以下代表 Ranker：

- Current-Controlled-32D；
- ClassGainPrior；
- ORIC-MLP；
- MORIC-MLP；
- Random。

ENA24 的动作消融尤其值得突出 CostEligible-AllROI，因为这是目前三个数据集中 ROI 策略表现最明确的场景。

---

## 13. 最新跨数据集学习器诊断（2026-08-07）

本节补充 `20260807_ena24_oregon_dualhead_crossdataset_v1` 和
`20260807_three_dataset_actionvalue_roi_gate_v1` 的结果。该节与前文的
`fixed_call` 九个 Ranker 实验不是同一个预算协议，不能直接把两组表格的
10%/25%/40% 数值混为一谈。

### 13.1 Track A 协议

- 学习器训练：canonical fit=1,400 张 ENA24 图像；5-fold OOF，3 个 seeds；
- calibration：400 张互斥图像，仅用于冻结阈值和比较器；
- 动作：Edge、Original、Q90；Track A 有意排除 ROI；
- 主模式：`fixed_byte`，次模式：`fixed_call`；
- final：Track A 未访问 final；
- 指标：Official mAP50–95 和 Request KB/image。

### 13.2 Track A：Fixed-byte OOF 与 calibration

#### 5-fold OOF

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.710073 / 43.04 | 0.721103 / 107.66 | 0.729973 / 172.31 |
| SetValue | 0.710912 / 43.07 | 0.720724 / 107.68 | **0.733678 / 172.29** |
| DualHead | 0.711242 / 43.07 | 0.724418 / 107.70 | 0.731117 / 172.30 |
| ActionValue-OQ-Joint | **0.718890 / 43.08** | **0.725845 / 107.67** | 0.725798 / 172.31 |

#### Frozen calibration

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.706552 / 43.06 | 0.703230 / 107.77 | **0.710527 / 172.45** |
| SetValue | 0.701428 / 43.10 | 0.696361 / 107.77 | 0.695452 / 172.45 |
| DualHead | 0.698450 / 43.01 | 0.692448 / 107.78 | 0.694766 / 172.38 |
| ActionValue-OQ-Joint | **0.708708 / 43.02** | **0.712084 / 107.73** | 0.698244 / 172.41 |

相对 DualHead，ActionValue-OQ-Joint 在三个字节预算分别提升
`+0.010258 / +0.019636 / +0.003478 AP`。相对 Current-BudgetSet，
ActionValue 在 10% 和 25% 提升，但 40% 下降 `0.012283 AP`。

### 13.3 Track A：Fixed-call calibration

| 学习器 | 10% AP / KB | 25% AP / KB | 40% AP / KB |
| --- | ---: | ---: | ---: |
| Current-BudgetSet | 0.701858 / 43.83 | 0.693705 / 94.48 | 0.693624 / 146.61 |
| SetValue | 0.697520 / 32.09 | 0.693814 / 95.39 | 0.697357 / 148.32 |
| DualHead | 0.698675 / 34.72 | 0.691275 / 93.12 | 0.690275 / 149.97 |
| ActionValue-OQ-Joint | **0.703800 / 39.08** | **0.703907 / 102.74** | **0.698048 / 166.94** |

ActionValue 在 ENA24 的 fixed-call 三个点也较强，但通信量随动作选择变化，
因此论文主结论仍应以 fixed-byte 为准。

### 13.4 Track B：DualHead-Value 接入既有动作框架

Track B 将 DualHead-Value 的图像分数接入原有冻结的 SharedAction、
CostEligible-AllROI 和 GBDT-Hierarchical。该部分是 `posthoc_canonical_final_reuse`，
不是新的 hidden final；每个工作点进行 200 次 paired bootstrap。

| 动作框架 | 预算 | DualHead AP / KB | 历史最强比较器 | 比较器 AP | ΔAP | ΔKB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| SharedAction | 10% | 0.708193 / 31.95 | Current-Controlled-32D | 0.721522 | -0.013329 | -11.86 |
| CostEligible-AllROI | 10% | 0.709159 / 27.37 | ClassGainPrior | 0.722521 | -0.013362 | -0.79 |
| GBDT-Hierarchical | 10% | 0.708281 / 28.99 | ClassGainPrior | 0.721858 | -0.013577 | -1.74 |
| SharedAction | 25% | 0.719599 / 86.42 | Current-Controlled-32D | 0.730205 | -0.010607 | -17.10 |
| CostEligible-AllROI | 25% | 0.720568 / 69.62 | Current-Controlled-32D | 0.730518 | -0.009950 | -10.79 |
| GBDT-Hierarchical | 25% | 0.719409 / 74.94 | Current-Controlled-32D | 0.729456 | -0.010047 | -12.90 |
| SharedAction | 40% | 0.724966 / 142.94 | Current-Controlled-32D | 0.735506 | -0.010539 | -19.59 |
| CostEligible-AllROI | 40% | 0.725772 / 116.11 | ClassGainPrior | 0.735844 | -0.010072 | -2.92 |
| GBDT-Hierarchical | 40% | 0.723965 / 124.64 | Current-Controlled-32D | 0.735059 | -0.011093 | -15.46 |

#### Paired bootstrap

| 动作框架 | 预算 | 比较器 | 均值 ΔAP | 95% CI | P(Δ>0) |
| --- | ---: | --- | ---: | ---: | ---: |
| SharedAction | 10% | Current-Controlled-32D | -0.011629 | [-0.026471, +0.000240] | 0.030 |
| CostEligible-AllROI | 10% | ClassGainPrior | -0.012245 | [-0.027607, -0.001081] | 0.020 |
| GBDT-Hierarchical | 10% | ClassGainPrior | -0.012402 | [-0.027806, -0.001157] | 0.015 |
| SharedAction | 25% | Current-Controlled-32D | -0.009294 | [-0.022645, +0.002299] | 0.080 |
| CostEligible-AllROI | 25% | Current-Controlled-32D | -0.008614 | [-0.021273, +0.003300] | 0.095 |
| GBDT-Hierarchical | 25% | Current-Controlled-32D | -0.008694 | [-0.021755, +0.003140] | 0.095 |
| SharedAction | 40% | Current-Controlled-32D | -0.009080 | [-0.022981, +0.003505] | 0.085 |
| CostEligible-AllROI | 40% | ClassGainPrior | -0.008852 | [-0.022890, +0.000854] | 0.050 |
| GBDT-Hierarchical | 40% | Current-Controlled-32D | -0.009602 | [-0.023470, +0.002169] | 0.070 |

`P(Δ>0)` 表示 Bootstrap 重采样中 ΔAP 为正的比例，不是传统假设检验的 p-value。

九个工作点的点估计和 Bootstrap 均值全部低于各自历史最强比较器。10% 的
AllROI 与 GBDT 区间完全低于 0；其余区间跨零，但没有一个工作点提供
DualHead 更强的证据。因此不能把 DualHead 描述为 ENA24 上的泛化优势。

### 13.5 OQ/OQR ROI gate 诊断

该诊断使用三动作 OQ 与四动作 OQR 的缓存，fit OOF 冻结 AdvantageGate，
calibration 只评测一次。`OQR-Naive` 不限制 ROI，`AdvantageGate` 只在预测
ROI 优于 Original/Q90 时开放 ROI。

| 预算 | OQ AP | OQR-Naive AP / ΔAP | Naive ROI / 有害率 | AdvantageGate AP / ΔAP | Gate ROI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.708708 | 0.703794 / -0.004913 | 22.84% / 48.65% | 0.704400 / -0.004307 | 0 |
| 25% | 0.712084 | 0.704417 / -0.007667 | 19.73% / 45.45% | 0.708805 / -0.003280 | 0 |
| 40% | 0.698244 | 0.706781 / +0.008537 | 15.21% / 45.00% | 0.705343 / +0.007099 | 0 |

AdvantageGate 在 calibration 三个预算都关闭了 ROI。其非零 ΔAP 来自 OQR
四动作模型重新校准 Original/Q90 分数，而不是 ROI 带来的增益。因此当前
实验不支持将 canonical ROI 直接加入 ENA24 的统一 ActionValue 动作空间。

### 13.6 ENA24 更新后的结论

1. `ActionValue-OQ-Joint` 是当前 ENA24 上最强的新联合学习器，在 10%/25%
   fixed-byte 下超过 Current-BudgetSet 和 DualHead，但 40% 应回退到
   Current-BudgetSet。
2. `DualHead` 在 Track A 明显弱于 ActionValue，在 Track B 接入三个旧动作框架
   后九个点也全部低于历史最强 Ranker，适合作为消融而不是主方法。
3. ENA24 的旧 fixed-call 结果仍然支持 `CostEligible-AllROI` 的通信—精度折中，
   但 OQR gate 说明当前学习器还不能可靠识别 ROI 增益。
4. COCO 的 16-Ranker No-ROI completeness extension 尚未在 ENA24 上执行，
   本文档不把 Track A 四模型误称为全 Ranker 扩展。

### 13.7 论文使用建议

- 以 fixed-byte 的 ActionValue-OQ-Joint 作为新方法候选；ENA24 40% 报告
  Current-BudgetSet 回退工作点；
- 将 DualHead、SetValue、Current-BudgetSet 作为内部消融；
- 将原 fixed-call 九 Ranker + 三动作表作为历史动作策略分析；
- 将 OQR gate 作为 ROI 失败诊断，不作为部署收益。
