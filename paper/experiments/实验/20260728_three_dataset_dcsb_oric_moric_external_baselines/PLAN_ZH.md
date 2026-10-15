# DCSB 与 ORIC/MORIC 三数据集外部基线计划

## 阶段 0：方法与口径冻结

- [x] 核对 DCSB 原文与阈值规则。
- [x] 核对 ORIC/MORIC 原文、公式与官方代码。
- [x] 区分 DCSB 原生二值点和连续预算适配分数。
- [x] 区分 Original 外部方法口径和 SharedAction 受控诊断。

## 阶段 1：规范与失败测试

- [x] 写 `SPEC.md`、协议和文献来源清单。
- [x] 写 top-25 特征、DCSB、ORIC、CDF 与矩阵合同测试。
- [x] 按失败测试、最小实现、完整回归的顺序完成逻辑验证。

## 阶段 2：纯算法核心

- [x] 实现 top-25 检测输出特征。
- [x] 实现 DCSB 三阈值拟合、原生决策和连续适配分数。
- [x] 实现检测匹配、AP 与上下文 ORIC 奖励。
- [x] 实现 MORIC 经验 CDF 与加权 MLP。

验证：全部纯函数测试通过；固定随机种子结果完全复现。

## 阶段 3：论文协议接入

- [x] 只读加载三数据集 fit/calibration/final 冻结角色。
- [x] 加载 Original 与冻结 SharedAction 载荷映射。
- [x] 实现 Fixed-call、Online 和 Fixed-byte 十个工作点。
- [x] 复用 Official AP 与真实请求字节评估。

验证：每数据集 65 行，键唯一，final 不参与任何拟合或冻结。

## 阶段 4：可恢复运行与来源保护

- [x] ORIC 奖励按图片 checkpoint，可安全续跑。
- [x] 每阶段记录实际读取文件的 SHA-256。
- [x] 阶段结束复查来源哈希，变化则不写完成标志。
- [x] 奖励完成标志绑定协议、fit 来源摘要、奖励 artifact 和 DCSB 模型。
- [x] 三数据集输出目录互相隔离。

## 阶段 5：验证与服务器运行

- [x] 本地单元测试与合成烟雾测试（33 项测试；9 个 Python 文件语法检查通过）。
- [ ] 服务器三数据集 preflight。
- [ ] 并行生成三个数据集 fit-only ORIC 奖励。
- [ ] 奖励完成后并行训练和终评。
- [ ] 合并外部基线与冻结主结果，生成 AP-KB 主图输入表。

## 结果解释边界

- `DCSB-Native` 是最忠实结果，但只有自然上传率一个点。
- `DCSB-Score-Adapted` 可以画预算曲线，但必须保留 Adapted 后缀。
- `ORIC-MLP`/`MORIC-MLP` 是论文忠实重实现，不是作者原始数值复现。
- SharedAction 结果只用于控制载荷差异，不能称为原 ORIC/MORIC 系统。
- Oregon 终评仍是已访问终评的 post-hoc 复用，不能称新隐藏测试。

服务器运行命令和完成条件见 `README_ZH.md`。本地没有服务器预测缓存和
PyTorch 环境，因此尚未生成或声称任何三数据集真实结果。
