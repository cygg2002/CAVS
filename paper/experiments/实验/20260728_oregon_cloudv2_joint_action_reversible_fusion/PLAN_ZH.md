# 实施计划

## 阶段 1：协议和纯函数

- [ ] 冻结 role、候选动作、特征维度、融合和调度契约。
- [ ] 先写动作特征、可逆融合、预算选择和 final 拒绝测试。
- [ ] 实现最小纯函数并通过本地单测。

## 阶段 2：fit 候选缓存

- [ ] 只在本实验目录生成 fit 的四种 crop geometry。
- [ ] GPU 空闲门禁后生成 12 个 scale prediction cache。
- [ ] 校验 12 个完成 marker、哈希和 9,923 图像集合。

## 阶段 3：训练与校准

- [ ] 从 fit 构建 MAUR/reversible action-conditioned 标签。
- [ ] 训练 safe classifier 和 gain regressor。
- [ ] calibration 冻结 reversible 参数和 10%/25%/40% 调度配置。
- [ ] 写入模型、输入哈希和 `CALIBRATION_FROZEN.json`。

## 阶段 4：audit

- [ ] 只读取冻结文件，运行一次 audit。
- [ ] 计算 Official AP、AP50、AP75、KB、动作比例和安全精度。
- [ ] 计算 class AP 与 200 次 paired image bootstrap。
- [ ] 确认 registered final 未访问。

## 阶段 5：复核与报告

- [ ] 复核测试、代码、缓存签名和选择隔离。
- [ ] 对比 Q90/Original base、MAUR、reversible 和 teacher ceiling。
- [ ] 写中文结果说明，明确部署结果与 Oracle 上界边界。
