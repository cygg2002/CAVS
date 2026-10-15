# Implementation Plan: VOC Torchvision CAVS-O/OQ/OQR

## Architecture decisions

- 复用 20260824 论文级 VOC 数据、reward、训练、评测与 bootstrap 骨架。
- fit/calibration 的三种基础动作读取 20260825 Torchvision Gate 哈希输出。
- 新增、独立生成 `current@official+MAUR`，不修改已完成 Gate 实验。
- O、OQ、OQR 分别构建 records/reward/checkpoints；禁止动作屏蔽冒充消融。

## Tasks

### Phase 1: contracts

- [ ] 冻结 protocol、源行哈希、模型权重和动作语义。
  - Acceptance：协议验证禁止旧 learner、禁止 test reward、固定三种动作空间。
  - Verify：`python -m unittest discover -s tests -v`。

### Phase 2: ROI vertical slice

- [ ] 实现 ROI 几何、编码、官方 Cloud 推理、坐标映射和 MAUR。
  - Acceptance：合成测试通过，10 张真实 smoke 无非法框且可断点。
  - Verify：服务器运行 `cache_actions.py --role calibration --action roi --limit 10`。

### Phase 3: development artifacts

- [ ] 导入 fit/calibration 基础动作并构建完整 ROI。
- [ ] 构建 O/OQ/OQR records 和 fold-contained rewards。
  - Acceptance：人口与角色哈希一致，test reward 路径不存在。
  - Verify：common/reward completion 与审计报告。

### Phase 4: learners

- [ ] 从头训练 CAVS-O/OQ/OQR 和 ORIC/MORIC。
  - Acceptance：全部 jobs、OOF、full-fit、单模型和 ensemble scores 完整。
  - Verify：job matrix 与 artifacts SHA-256。

### Phase 5: benchmark and statistics

- [ ] 生成 test 四动作，运行点结果、paired bootstrap 和报告。
  - Acceptance：两模式×三预算完整、byte-safe、1,000 replicates。
  - Verify：结果矩阵审计与最终报告。

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Faster R-CNN 官方 transform 与 YOLO 640 不同 | 高 | 动作命名为 current@official，禁止改模型 transform |
| SSDLite 低阈值框过多导致 ROI 过大 | 高 | ROI 几何固定使用 score≥0.25 和 area≤0.65 |
| ROI 对多数图不可用 | 中 | 不回退伪造 ROI；记录 availability，由 OQR 学习器显式 mask |
| test 影响模型选择 | 高 | 所有配置预注册；reward/training 代码拒绝 test |
| GPU 并发污染延迟 | 中 | 本实验只使用准确率，不声明正式延迟 |
