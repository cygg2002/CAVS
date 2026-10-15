# Oregon Cloud-V2 Joint Action + Reversible Fusion 规范

## 1. 目标

在不改变 Oregon Cloud-V2、Edge proposal、数据划分、Q90 编码和 Official
evaluator 的前提下，把上一阶段 12 个 `ROI Profile x imgsz` teacher ceiling
变成可部署学习器，并独立回答可逆融合是否优于破坏式 MAUR。

本实验训练 action-conditioned GBDT：

- 安全头预测候选动作相对冻结 Q90/Original base 是否满足
  `quality >= base_quality - 0.02`；
- 增益头预测候选动作相对 base 的逐图 proxy gain；
- 调度器在 `10%/25%/40%` ROI 上限下，联合决定图片、Profile 和 imgsz；
- Q90/Original 是所有图片的低成本/高保真 fallback，ROI 未通过门禁时不替换。

同时评测两个融合分支：

1. `maur`：复现上一阶段的破坏式替换；
2. `reversible`：保留 Edge shadow base，只有同类匹配且 ROI 证据充分时才替换，
   未匹配 ROI 仅在分数门槛以上加入。

## 2. 数据角色与隔离

| Role | 图片数 | 用途 |
|---|---:|---|
| fit | 9,923 | 生成 12 候选标签并训练学习器 |
| calibration | 4,961 | 冻结可逆融合参数、模型门禁和预算调度参数 |
| audit | 9,923 | 冻结后一次性验证 |

- calibration/audit 的 12 候选原始预测只读复用
  `20260728_oregon_cloudv2_roi_profile_scale_scan`；
- fit 的候选缓存在本实验目录重新生成；
- 禁止读取、生成或评估 `final`；
- 不读取同时包含 final 的全局 assignments 文件；
- registered final 继续封存。

## 3. 冻结项

- Cloud-V2 SHA256：
  `8944137e1a1324b064ed487ab780c1a410ac55f5be36d0c95192bdfdb52f1fdd`；
- Edge、Original、Q90 prediction 和冻结 Q90 action head；
- Profile：Current、Moderate、Historical-cost、fixed-1.8；
- imgsz：160、320、640；
- ROI JPEG Q90、Edge conf=0.25、最大合并 ROI 面积=65%；
- Official `conf=0.001, iou=0.7, max_det=300`；
- fit/calibration/audit 图像集合及 evaluator；
- AP tolerance=0.001、safe epsilon=0.02。

## 4. 可观测特征与标签

每个 `(image, action)` 样本只使用上传前可得信息：

- 冻结的 54D Edge/action feature；
- 动作类型、Profile、imgsz；
- ROI crop 面积、宽高比、相对 Full 的线性放大率；
- 实际 JPEG bytes、相对 Original/base payload ratio；
- candidate availability。

不允许使用 GT 类别、GT 框、逐图真实 gain 或 audit 标签作为输入特征。
GT 只用于 fit 标签、calibration 配置选择和 audit 最终评测。

模型只用 fit 训练。模型结构和随机种子在 calibration 前冻结；calibration 只选择：

- 可逆融合参数；
- safe probability threshold；
- cost weight；
- hierarchical-base 或 joint Q90/Original fallback；
- 每个注册预算的工作点。

## 5. 融合契约

可逆融合必须满足：

1. Edge detection 默认保留；
2. ROI 与 Edge 同类且 IoU 达门槛时，只有 ROI score 不低于
   `edge_score + score_margin` 才允许替换；
3. 未匹配 ROI 只有 score 达 `add_threshold` 才加入；
4. 最后执行冻结的 class-aware NMS；
5. 每个输出框记录 source，仅用于诊断，不参与评价。

可逆参数候选仅在 calibration 比较，audit 不回调。

## 6. 调度与评测

注册预算为 `10%/25%/40%`，含义是 ROI 动作最多占全部图片的比例。
每张图始终从 Q90、Original 或一个 ROI Profile@imgsz 中选择一个请求动作。

calibration 对每个预算选择最低 KB 的 AP-safe 工作点：

1. Official AP 不低于冻结 Q90/Original base AP `-0.001`；
2. ROI 数量不超过预算；
3. 未通过 safe threshold 的 ROI 必须回退；
4. 相同 KB 时选择 AP 更高、阈值更保守的配置。

audit 只应用冻结模型和配置，并报告：Official AP/AP50/AP75、KB/image、
ROI/Q90/Original 比例、动作分布、安全精度、class AP 和 200 次 paired image
bootstrap。

现有缓存没有 event_id。只有获得不含 final 的 role-specific event manifest 后
才能补 event bootstrap；本轮不得从全局 assignments 推导。

## 7. 验收标准

工程验收：

1. fit 的 12 候选完整且 geometry/prediction hash 通过；
2. 模型只读取 fit，冻结文件在 audit 前写入；
3. audit 代码路径拒绝任何未冻结配置和任何 final 路径；
4. 全部单测、缓存签名和 Official evaluator 检查通过。

研究验收：

1. 至少一个预算的 audit `Delta AP >= -0.001` 且 KB 低于 base；
2. 报告三个预算，不只报告最佳点；
3. reversible 与 MAUR 使用相同候选、特征、训练和 calibration 规则；
4. 任何 Oracle 只标为 ceiling，不冒充部署结果；
5. class AP 或 paired bootstrap 出现明显负组时必须显式披露。

## 8. 命令

```bash
python3 -m unittest discover \
  -s paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/tests \
  -p 'test_*.py'
```

```bash
/home/111_wcy/anaconda3/envs/yolo26/bin/python \
  paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/scripts/run_experiment.py \
  --stage preflight

/home/111_wcy/anaconda3/envs/yolo26/bin/python \
  paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/scripts/run_experiment.py \
  --stage fit-cache

/home/111_wcy/anaconda3/envs/yoloe/bin/python \
  paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/scripts/run_experiment.py \
  --stage calibrate

/home/111_wcy/anaconda3/envs/yoloe/bin/python \
  paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/scripts/run_experiment.py \
  --stage audit
```

`preflight` 和 `fit-cache` 使用 `yolo26` 推理环境；`calibrate` 和 `audit`
使用已有 `yoloe` 分析环境（scikit-learn 1.7.0 + pycocotools），两者的
解释器和库版本必须记录到冻结文件。

## 9. 边界

- Always：输入哈希、role 数量、模型哈希、特征有限性和 action availability 校验。
- Ask first：改变检测器、数据划分、Q90 codec、预算定义或访问 final。
- Never：使用 audit 调阈值、覆盖旧实验、终止其他用户进程、伪造 event bootstrap。
