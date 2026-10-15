# 执行计划

1. 冻结协议、来源、数据/权重哈希和完整结果矩阵。
2. 先写并运行协议、VOC difficult 评测、框映射与统计矩阵测试。
3. 在服务器建立隔离的 GluonCV/MXNet 环境，下载并核验两个官方权重。
4. 解压 VOC 官方归档，生成 fit/calibration/final 角色和双语义 GT。
5. 完成 1 张、10 张 smoke，随后执行 4,952 张 Edge/Cloud 端点门禁。
6. 生成三个角色的 Edge、Original、Q90 新缓存和 JPEG 成本清单。
7. 构建 fold-contained contextual rewards；分别构建 O 与 OQ 训练矩阵。
8. 运行 CAVS 5-fold OOF/full-fit 三种子及 ORIC/MORIC 三种子全重训。
9. calibration 一次冻结 SharedAction-Q90；生成两种预算模式的完整点结果。
10. 对 Fixed-byte 主比较执行 1,000 次 paired/nested Random-50 bootstrap。
11. 校验行数、哈希、预算、数据隔离，生成中文结果报告和论文表格。

