# Implementation Plan: VOC Torchvision Gate 0–2

## Architecture Decisions

- 新目录隔离，不修改失败的 GluonCV 实验。
- 旧实验仅作为已哈希的 VOC 数据/指标来源，不复用预测。
- 所有风险门禁先于大规模推理；推理按 action/role 断点保存。
- Static COCO→VOC 类别映射只允许 20 个一对一别名。

## Task List

### Phase 1: Contract

- [x] 写纯逻辑失败测试。
  - Acceptance：测试覆盖 final 禁止、训练数据来源、类别映射、框过滤、门禁判定。
  - Verify：测试因 `gate_core.py` 尚不存在而失败。
- [x] 实现 `gate_core.py`。
  - Acceptance：上述测试通过。
  - Verify：`python -m unittest discover -s tests -v`。

### Phase 2: Gate 0–1

- [x] 实现合同审计和可断点推理。
  - Acceptance：模型/权重/数据/源码哈希完整；禁止 final；2 张真实模型 smoke 通过。
  - Verify：服务器运行 audit 与 `--limit 2`。
- [ ] 执行冻结 200 张冒烟。
  - Acceptance：三动作 200/200、非法输出为 0、摘要确定。
  - Verify：`evaluate_gate.py --stage smoke`。

### Phase 3: Gate 2

- [x] 实现五折、端点和 paired bootstrap 评测。
  - Acceptance：报告每条停止规则及 `proceed_to_reward`。
  - Verify：合成测试和服务器完整报告。
- [ ] 在 GPU 空闲门禁后运行 fit/calibration。
  - Acceptance：不访问 final，不影响现有 GPU 任务，工件哈希完整。
  - Verify：完成报告与进程/资源审计。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| COCO→VOC 性能方向仍不稳定 | 高 | 仅看 fit/calibration 的 Gate 2 失败即停 |
| Torchvision 两模型默认后处理不同 | 中 | 显式统一 score threshold/max detections，保留各自官方 NMS |
| 当前 GPU 被其他任务占用 | 中 | 显存与利用率门禁后再启动，不并发测延迟 |
| 小上下文 reward 不稳定 | 高 | 本轮不生成 reward；留给 Gate 3 单独验证 |

## Checkpoints

- Contract：单元测试全绿。
- Gate 1：200 张三动作全绿后才准许 Gate 2。
- Gate 2：只有完整报告 `proceed_to_reward=true` 才可新建后续论文实验。
