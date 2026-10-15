# Oregon Cloud‑V2 ROI Profile × Scale 扫描规范

## 1. 目标

冻结 Oregon Cloud‑V2 的 Edge proposal、Cloud 模型、Original/Q90 base、
数据划分、Q90 编码、MAUR 融合和 Official evaluator，只改变 ROI 扩框
Profile 与 Cloud 推理尺度，回答：

1. 多 Profile/Scale 是否严格扩大单一 Historical@640 的 AP–KB 候选空间；
2. Cloud‑V2 ROI 退化主要来自扩框上下文还是 resize 尺度；
3. calibration 冻结的 AP-safe ROI 比例能否在 audit 保持
   `Official ΔAP >= -0.001`。

本实验是 GT-aware 动作空间诊断，不训练学习器，也不把 Oracle 结果描述为可部署结果。

## 2. 冻结与变量

冻结：

- Cloud‑V2 SHA256：
  `8944137e1a1324b064ed487ab780c1a410ac55f5be36d0c95192bdfdb52f1fdd`；
- Edge predictions、GT、Original/Q90 predictions 和 Q90 base action head；
- ROI JPEG quality=90、Edge conf=0.25、最大合并 ROI 面积=65%；
- `conf=0.001, iou=0.7, max_det=300, rect=True, end2end=False`；
- MAUR replacement + class-aware NMS；
- epsilon=0.02、Official AP tolerance=0.001。

唯一变量：

- Profile：`current / moderate / historical_cost / fixed_1_8`；
- Cloud imgsz：`160 / 320 / 640`；
- 共 12 个预注册 ROI 候选。

每个 role/profile 只生成一份 Q90 crop 与 manifest，三个尺度共享相同 JPEG，
仅 prediction cache 分开。

## 3. 数据角色与隔离

| Role | 图片数 | 用途 |
|---|---:|---|
| calibration | 4,961 | 选择最佳单候选、冻结多候选集合和 ROI 比例 |
| audit | 9,923 | 一次性冻结验证与诊断上限 |

禁止：

- 读取、生成或评估 `final`；
- 用 audit 修改候选集合、融合、阈值或 ROI 比例；
- 修改 20260720/20260727 冻结实验的脚本或产物。

## 4. 评测

每个候选报告：

- ROI generation / payload-eligible / proxy-safe / strict-safe；
- median ROI/base payload ratio；
- teacher Oracle sweep 的 Official AP50-95/AP50/AP75 与 KB/image；
- AP-safe 最大 ROI 比例及最大 AP 点。

多候选报告：

- 每图在 12 个 eligible 候选中按 proxy quality 选最优动作，
  quality 相同则选 payload 更低、candidate ID 字典序更小者；
- calibration 冻结 AP-safe 最低 KB 工作点；
- audit 使用同一候选集合与冻结 ROI 比例；
- audit unrestricted sweep 只标记为 diagnostic ceiling。

分解维度：

- Profile、imgsz、GT 面积桶、类别、linear magnification；
- AP50 与 AP75 分开报告。

现有 common cache 不含 `event_id`，而全局 assignments 文件同时包含 final。
本实验不读取该文件；事件分解只允许在获得预先导出的 calibration/audit
role-specific event manifest 后补充。

## 5. 命令

本地测试：

```bash
python3 -m unittest discover \
  -s paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/tests \
  -p 'test_*.py'
```

服务器预检和执行：

```bash
/home/111_wcy/anaconda3/envs/yolo26/bin/python \
  paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/run_scan.py \
  --stage preflight

/home/111_wcy/anaconda3/envs/yolo26/bin/python \
  paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/run_scan.py \
  --stage calibration

/home/111_wcy/anaconda3/envs/yolo26/bin/python \
  paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/run_scan.py \
  --stage audit
```

## 6. 代码与产物

```text
configs/protocol.json       冻结协议和远端路径
scripts/scan_core.py        候选、几何和冻结选择纯函数
scripts/build_cache.py      共享 crop 几何与分尺度预测
scripts/evaluate_scan.py    Official AP/KB 与分解
scripts/run_scan.py         可恢复 GPU 门禁与阶段调度
tests/                      纯函数和隔离契约测试
runs/                       仅本实验产物
```

## 7. 测试策略

- 小型单测：12 候选枚举、fixed‑1.8 几何、确定性 tie-break、AP-safe 选择；
- 契约测试：dev roles 只能是 calibration/audit，拒绝任何 final；
- dry-run：预检所有输入存在、模型 hash 正确、输出路径不指向冻结实验；
- 真实验证：calibration 完成并写入 freeze 后才允许 audit。

## 8. 成功标准

1. calibration/audit 均完整生成 12 个候选，四个 profile 内三个尺度共享相同
   geometry manifest SHA；
2. 没有读取或生成 final；
3. audit 多候选 frozen point 满足 `Official ΔAP >= -0.001` 且 KB/image
   低于 base；
4. audit 多候选 ceiling 明显优于最佳单候选，否则停止后续学习器实验；
5. 所有代码和产物仅位于本实验目录。

## 9. 边界

- Always：校验模型 hash、输入规模、cache signature、Official evaluator。
- Ask first：改变 detector、融合语义、Q90 codec、数据划分或 registered final。
- Never：覆盖旧实验、终止其他任务、用 audit 二次调参。
