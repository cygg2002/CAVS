# 真实系统测量状态

## 当前结论

真实异机链路在硬件和代码层面可行，但本实验**不能声称已实测 E2E**。
本文件只记录 2026-07-25 已核实的可行性事实，不包含延迟、吞吐或能耗测量值。

## 已核实条件

- 端侧为 `Apple M2 Pro`，云端为 `NVIDIA GeForce RTX 4090`，两者是不同主机。
- 云端主机别名为 `fzu111`；GPU 当前为共享且繁忙的瞬时状态，不适合作为论文级计时环境。
- 现有端侧客户端：`examples/YOLO26-Edge-Cloud/edge_client.py`（存在：`true`）。
- 现有云端服务端：`examples/YOLO26-Edge-Cloud/cloud_server.py`（存在：`true`）。
- 真实网络链路可建立；1/5/10/20 Mbps 等受控限速需要系统权限，本次没有执行网络整形。
- 审计文件不保存登录凭据或远程地址。

## 为什么尚未进行论文级实测

必须先同时满足：

1. 获得空闲或独占的 RTX 4090 计时窗口，避免其他进程污染尾延迟；
2. 冻结端侧和云端模型权重，并记录各自 SHA-256；
3. 冻结 Original/Q90/ROI 的 payload 回放清单及其 SHA-256；
4. 对端侧检测、编码、传输往返、解码、云端检测和总 E2E 分阶段记录原始日志；
5. 对自然网络状态进行实测，或在取得系统权限后执行并记录受控带宽/RTT配置。

当前 measured-E2E 契约按预期拒绝该声明：

```text
An estimated latency cannot be labeled measured E2E
```

## 下一次可执行测量

在云 GPU 空闲后，冻结权重与 payload 回放；先做 warm-up，再对同一回放序列重复计时，
保存逐样本原始日志、环境信息和文件哈希。只有
`selector_core.validate_measured_e2e_claim` 对完整元数据校验通过后，才允许在论文中使用
“实测 E2E”表述。在此之前，既有 `E2E@10Mbps` 仍只能称为估算值。
