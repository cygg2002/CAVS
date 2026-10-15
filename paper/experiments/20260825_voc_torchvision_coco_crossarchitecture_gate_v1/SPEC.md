# VOC Torchvision COCO 跨架构端点门禁规格

## Objective

在不训练检测器、不访问 VOC2007 test 的前提下，判断以下公开 COCO 权重是否适合进入完整 CAVS 跨架构实验：

- Edge：Torchvision `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`；
- Cloud：Torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`。

本实验只执行 Gate 0–2：权重与数据溯源、200 张实现冒烟、calibration 先行端点门禁、fit 五折与 calibration paired bootstrap。不得生成 contextual reward、训练路由器或读取 final。

## Tech Stack

- Python 3.10；
- PyTorch 2.6.0+cu124；
- Torchvision 0.21.0+cu124；
- RTX 4090，单图确定性推理；
- 复用上一实验冻结的 VOC fit/calibration 身份和标注，必须逐文件核对 SHA-256；
- 复用冻结的 VOC metric 实现，必须核对源码 SHA-256。

官方依据：

- Torchvision 检测模型与 COCO_V1 指标：https://docs.pytorch.org/vision/master/models.html#object-detection
- SSDLite COCO 训练说明：https://pytorch.org/blog/torchvision-ssdlite-implementation/

## Commands

```bash
# 单元测试
python -m unittest discover -s tests -v

# Gate 0：数据、依赖、权重与类别合同
python scripts/audit_contract.py

# Gate 1：fit 中冻结的 200 张冒烟
python scripts/run_inference.py --role fit --stage smoke --action all
python scripts/evaluate_gate.py --stage smoke

# Gate 2：仅 fit/calibration 端点
python scripts/run_inference.py --role calibration --stage full --action all
python scripts/evaluate_gate.py --stage calibration
python scripts/run_inference.py --role fit --stage full --action all
python scripts/evaluate_gate.py --stage full
```

## Project Structure

```text
configs/protocol.json      冻结模型、数据、类别、推理与门禁合同
scripts/gate_core.py       纯合同、映射、折分、框过滤与门禁逻辑
scripts/audit_contract.py  Gate 0 审计并下载/哈希官方权重
scripts/run_inference.py   可断点的 Edge/Original/Q90 单图推理
scripts/evaluate_gate.py   VOC 指标、五折、paired bootstrap 与停止判定
tests/                     纯逻辑和小型集成测试
runs/                      运行时证据；不得提交或跨协议复用
```

## Code Style

```python
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

保持函数短小、显式验证输入；不建立通用实验框架，不修改旧实验。

## Testing Strategy

- 先写失败测试，再实现纯逻辑；
- 单元测试覆盖：禁止 final、训练集溯源、20 类映射、确定性抽样/折分、非法框过滤、门禁通过与拒绝；
- 服务器先执行 2 张模型 smoke，再执行 200 张 Gate 1；
- Gate 1 通过后才允许运行完整 fit/calibration；
- 每个完成工件记录模型、协议、输入与输出 SHA-256。

## Boundaries

- Always：只读 fit/calibration；使用官方 transforms；记录实际权重哈希；每张图单独推理；保留 difficult 并在指标阶段忽略。
- Ask first：改变模型组合、门槛、类别映射、阈值或允许访问 final。
- Never：使用 VOC 微调权重；读取 final；复用 GluonCV 检测缓存；生成 reward；训练 CAVS/ORIC/MORIC；根据 final 选择模型。

## Success Criteria

1. Gate 0：所有数据/源码/权重/版本/类别合同通过且 `final_accessed=false`。
2. Gate 1：200/200 图像、三动作完成；无未知类别、非有限框或退化框；重复推理摘要一致。
3. Gate 2：
   - calibration 点结果先行；任一门槛失败即不运行 15,629 张 fit；
   - fit 总体及 calibration 的 `Cloud Original − Edge AP50–95 >= 0.02`；
   - fit 五折的差值全部大于 0；
   - calibration 1,000 次 paired bootstrap 的 95% CI 下界大于 0；
   - Cloud AP75 不低于 Edge AP75；
   - Q90 平均节省字节至少 20%，且相对 Original 的 AP50–95 损失不超过 0.01。
4. 任一失败即输出 `proceed_to_reward=false`，不得继续完整论文实验。

## Open Questions

无。VOC07 test 已在旧实验中被查看，因此未来只能称标准 benchmark test，不能称新的 hidden final；本门禁仍禁止新模型访问该角色。
