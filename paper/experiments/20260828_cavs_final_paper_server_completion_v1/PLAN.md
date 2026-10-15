# 执行计划

1. 校验源脚本、协议、点结果和 selection 哈希。
2. 停止已淘汰的 GluonCV-VOC bootstrap，保留其断点但不纳入本论文。
3. 与现有 COCO 任务并行生成 Oregon 完整点结果矩阵。
4. 在历史 partial replicate 上验证分片结果与原实现一致。
5. 按资源可用量依次补完 COCO、COCO Original-only、VOC 和 Oregon bootstrap。
6. 调用各源实验原生 report 生成结果表与完成标记。
7. 汇总三 seed、ensemble、paired CI、任务数、哈希及统计口径并执行最终审计。

实验运行使用 `nice=19`，单 worker 限制 BLAS/OpenMP 线程为 1。并行度根据 64 核服务器
的实时负载和内存基线确定，保留系统与无关训练任务所需资源。

