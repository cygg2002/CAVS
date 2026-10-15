# 运行状态

更新时间：2026-07-23 16:45 CST，服务器 `10.193.114.121:10022`。

## 已通过

- 本地合同测试：9项通过。
- 远端论文环境测试：9项通过。
- COCO、ENA24、Oregon preflight：全部通过。
- 冻结输入完整性审计：120个文件、约3.26 GB，无缺失或哈希不一致。

## 运行中

- ENA24 core：PID 55546；完成后由 watcher 自动启动 bootstrap。
- COCO pipeline：PID 1392；core通过后自动启动 bootstrap。
- Oregon pipeline：PID 1394；core通过后自动启动 bootstrap。
- ENA24 bootstrap watcher：PID 1395。

每个数据集只写入本目录的 `runs/{dataset}`。只有通过矩阵规模、canonical
SharedAction 复现、ENA24 Controlled-32D锚点（仅ENA24）、输入前后哈希和
bootstrap唯一性检查后，才会写 `COMPLETED`。

Oregon 的 final 已在此前 canonical 实验访问过一次；本次始终标记为
post-hoc exploratory，不宣称新的 registered/hidden final。
