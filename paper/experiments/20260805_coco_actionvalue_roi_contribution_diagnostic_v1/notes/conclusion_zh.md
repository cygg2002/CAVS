# COCO ROI 贡献诊断结论

## 结果

### 5-fold OOF

相对禁用 ROI 的 `ActionValue-OQ`，完整 learned ROI 的 AP 增益为：

- Fixed-byte 10%/25%/40%：`+0.001712 / +0.002202 / +0.000651`。
- Oracle-relative ROI gate：`+0.001428 / +0.002245 / +0.001440`。

OOF 中 ROI 有小幅可见收益，但只有前两个 learned 点达到 +0.001 门槛。

### Frozen calibration

- Learned ROI：`-0.001827 / +0.002659 / -0.000021`。
- Oracle-relative ROI gate：`+0.000910 / +0.003106 / +0.000818`。

即使使用不可部署的 GT-aware gate，calibration 也只有 25% 达到 +0.001，
没有形成稳定的三预算收益。

## 诊断含义

1. COCO 的 ActionValue 增益主要来自 Original/Q90 的联合图片—动作调度，
   而不是稳定的 ROI 收益。
2. Learned ROI 的相对次优率很高：OOF 约 71%–75%，calibration 约 69%–95%。
   这表示当前 contextual reward/静态特征无法可靠判断 ROI 是否优于可用的
   Original、Q90 或不上传。
3. Oracle gate 将 ROI 数量大幅压低后，AP 只小幅变化；这说明 COCO 上可恢复的
   ROI 尾部很小，不值得继续堆叠新的 ROI Ranker。Oracle 结果仅用于上界诊断，
   不能作为部署结果。
4. Fixed-call 三个预算的 full/OQ/Oracle selection 完全一致，说明差异只发生在
   byte-constrained 调度，不是调用数量口径造成的。

## 决策

`decision=drop_coco_roi_mainline`

COCO 主方法应收敛为 Action-conditioned Value + Original/Q90 + Fixed-byte
CostEligible 调度。ROI 保留为消融或其他数据集的可选动作，不再在 COCO 上继续
训练静态 ROI eligibility/ranker。

本诊断使用 canonical fit=9,678 的 OOF 分数和 calibration=571，未访问 audit、
val2017 final 或 registered final；Oracle 使用 GT contextual reward，结果不是论文
最终结果，也不能直接宣称超过 ORIC/MORIC。

