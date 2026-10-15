# Action-conditioned Value/Risk 决定性诊断结论

## 冻结协议

- canonical train2017 fit=9,678：5-fold OOF。
- calibration=571：只做一次冻结验证，不参与训练。
- 固定 Edge/Original/Q90/ROI 动作缓存，不增加动作、不重新推理检测器。
- 比较 Current DualHead + SharedAction、Action-conditioned Value-only、
  Action-conditioned Value + 0.1 quantile LCB。
- 未访问 audit、val2017 final 或 registered final。

## 相对 Current DualHead + SharedAction 的 AP50-95 变化

| 数据 | 方法 | Fixed-call 10%/25%/40% | Fixed-byte 10%/25%/40% |
|---|---|---|---|
| fit 5-fold OOF | Value-only | -0.000823 / +0.001817 / +0.001006 | +0.029580 / +0.037365 / +0.037183 |
| fit 5-fold OOF | Value+Risk/LCB | -0.020911 / -0.024212 / -0.027022 | -0.048376 / -0.097173 / -0.130034 |
| frozen calibration | Value-only | +0.005060 / +0.002204 / +0.003861 | +0.024484 / +0.040407 / +0.035591 |
| frozen calibration | Value+Risk/LCB | -0.014758 / -0.022055 / -0.023889 | -0.042596 / -0.084128 / -0.116202 |

## 关键判断

1. Action-conditioned Value-only 的正式 AP/capacity gate 在 OOF 与 calibration
   均通过。两阶段的“先排图片、再套 SharedAction”确实是当前瓶颈；不需要先发明新的
   像素动作，应该先把现有 Original/Q90/ROI 改成逐图片—动作联合估值。
2. Value-only 在 Fixed-call 下几乎总选 Original；在 Fixed-byte 下会自动转向 Q90，
   并选择少量 ROI。这说明 Value 与实际字节约束分开后，动作会随约束自然变化。
3. 0.1 quantile LCB 不能直接替换 Value。fit OOF 只有 127/22,041 个可用云动作
   的 LCB 为正，calibration 只有 14/1,299 个为正。由于 Edge 的效用为 0，
   Fixed-byte 调度器只上传 69/7 张图片，三个预算都无法用满，导致 AP 崩溃。
4. ROI 风险尚未被彻底解决。Value-only 的有害 ROI 率在 OOF 三个 Fixed-byte 点
   有所下降，但 calibration 10% 从 28.57% 上升到 61.90%；因此不能把当前
   Value-only 直接描述成“风险受控的 Adaptive ROI”。
5. 冻结的总开关仍为 `open_three_seed_paper_protocol=false`，因为预注册主候选是
   Value+Risk/LCB，而它在 OOF 和 calibration 均失败。Value-only 的正结果应另立
   新协议确认，不能在当前 calibration 上继续调参后宣称论文结果。

## 后续路线

- 主干：保留现有三动作，使用 Action-conditioned Value 做联合图片—动作评分；
  Fixed-byte 由真实 payload 做容量约束。
- SharedAction、AllROI、GBDT-Hierarchical 保留为消融和对照，不再作为最终串联主干。
- Risk 不再作为替代 Value 的原始 LCB；若继续使用，只作为独立的 ROI accept/reject
  约束，并要求预算利用率下限，且必须在新的 development 协议中预先冻结。
- 下一项论文级评测应冻结 Value-only，使用新 audit/registered final、3 seeds、
  官方 ORIC/MORIC 同协议对照和 paired bootstrap；不得继续使用本次 calibration 调参。

