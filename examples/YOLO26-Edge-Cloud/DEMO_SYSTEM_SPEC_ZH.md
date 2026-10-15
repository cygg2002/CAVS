# 端云上传学习器演示系统规格

## 目标

将本机作为端侧、远程 GPU 服务器作为云端，演示以下完整闭环：

```text
本机图像/摄像头 -> YOLO26n -> Current+MAUR 决策 -> ROI/整图 HTTP 上传
              -> 服务器 YOLO26x -> 云端结果 -> 本机融合与指标展示
```

面向现场演示人员。系统应直观说明“是否上传、为什么上传、上传了多少、云端修正了什么”，而不是在线重新训练 YOLO 或上传学习器。

## 技术栈

- Python 3.11（`conda` 环境 `yolo26`）
- Ultralytics YOLO、PyTorch、OpenCV
- Gradio 5.x 本机 UI
- `ThreadingHTTPServer` 云端推理服务
- JSON metadata + Base64 JPEG HTTP 协议

## 命令

```bash
# 测试
/absolute/path/to/envs/yolo26/bin/python -m pytest -q -o addopts='' \
  tests/test_edge_cloud_demo.py tests/test_train_learned_uploader.py tests/test_edge_cloud_system.py

# 云端
bash examples/YOLO26-Edge-Cloud/start_cloud.sh

# 本机（直接访问服务器）
CLOUD_URL=http://SERVER_IP:8765/infer bash examples/YOLO26-Edge-Cloud/start_edge_demo.sh

# 本机（SSH 隧道）
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
bash examples/YOLO26-Edge-Cloud/start_edge_demo.sh
```

## 工程结构

```text
examples/YOLO26-Edge-Cloud/
  cloud_server.py          云端健康检查与 YOLO26x 推理
  gradio_demo.py           本机 UI、端侧推理、决策和融合
  demo_system.py           预检、健康检查和报告逻辑
  start_cloud.sh           云端启动入口
  start_edge_demo.sh       本机启动入口
  demo.env.example         可复制的演示配置模板
  DEMO_SYSTEM_SPEC_ZH.md   本规格
tests/
  test_edge_cloud_system.py  演示系统逻辑测试
runs/edge_cloud/gradio_demo/
  events.jsonl             逐帧事件
  summary.json             运行汇总
  demo_report.md           人类可读演示报告
```

## 代码风格

沿用当前 Python 风格：类型标注、短函数、显式返回结构，行宽不超过 120。

```python
def infer_url_to_health_url(infer_url: str) -> str:
    """Convert a validated /infer URL to its /health endpoint."""
    parsed = urlsplit(infer_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Cloud URL must be an absolute HTTP(S) URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
```

## 测试策略

- 单元测试：URL 校验、预检结果、云端健康响应、报告生成。
- 现有回归测试：上传学习器、ROI/整图协议与融合逻辑。
- 本机集成测试：启动 CPU 云端，检查 `/health`，发送一张真实图片到 `/infer`。
- UI 冒烟：构建 Gradio Blocks，启动后检查首页与无云降级状态。

## 边界

- 始终：验证 URL、模型和策略文件；请求设置超时；限制上传体积；云端失败时保留端侧结果；保存非敏感指标。
- 先询问：开放公网端口、添加鉴权、保存原始上传图像、修改服务器防火墙或安装系统服务。
- 禁止：提交服务器密码/密钥；将用户输入拼成 shell 命令；在错误中返回调用栈；把无鉴权服务描述为生产可用。

## 实施任务

1. 增加可测试的预检、健康检查和 Markdown 报告逻辑。
2. 增加显式使用 `yolo26` Python 的本机/云端启动脚本和配置模板。
3. 在现有 Gradio UI 中增加端云拓扑、连接状态、决策细节和累计指标。
4. 更新运行说明，完成单元测试和本机 HTTP 冒烟测试。

## 成功标准

- 本机一条命令可完成依赖、模型、策略和云端连通性预检并启动 UI。
- 端测模型必须是训练 Current+MAUR 时使用的 `yolo26n.pt`，并通过 SHA256 一致性校验。
- 云端一条命令可加载 `yolo26x.pt` 并提供 `/health` 与 `/infer`。
- UI 支持 `Edge Only`、`Cloud Only`、`Collaboration`，并显示连接状态。
- Collaboration 显示整图/ROI/不上云、学习器概率、预期收益、payload、端侧/云端/RTT/E2E 延迟。
- 云端不可达时 Edge Only 仍可运行，错误以简短状态呈现。
- 运行目录生成 `events.jsonl`、`summary.json` 和 `demo_report.md`。

## Current+MAUR 论文复现模式

上传视频提供两遍式 `Paper Fixed-byte` 模式：第一遍只运行端侧检测、Current ROI 构建、
O/Q/R 精确 JPEG 字节统计和三种子价值预测；随后按论文相同的多选拉格朗日调度器，
在全视频容量下冻结每帧动作；第二遍才执行云端请求和 MAUR 融合。

容量定义保持与论文一致：

```text
capacity_bytes = budget * sum(original_equivalent_bytes)
```

其中视频和摄像头没有可直接复用的源 JPEG，`Original` 使用当前部署协议的 Q95
`Original-equivalent`；Q90 和 Current@640 ROI 使用 Q90。界面必须同时报告原始 JPEG
载荷预算和实际上传帧比例，不得将两者混称为“上传比例”。

论文复现模式固定以下语义：Current@640+MAUR、三种子 ensemble、Cloud 640、全部类别、
Temporal Memory 关闭、Adaptive ROI/Cloud ImgSz 关闭。实时摄像头继续使用低延迟在线策略，
但必须明确标为在线近似，不能宣称与全局 Fixed-byte 调度等价。

新增验收条件：

- 10%/25%/40% 任一视频运行均满足 `request_bytes <= capacity_bytes`。
- 每帧恰好选择 Edge、Original、Q90、Current@640 中的一个动作。
- 调度器的动作顺序、代价、价值和 tie-breaking 与论文实现一致。
- 第二遍实际 JPEG 上传字节不得超过第一遍冻结容量。
- UI 显示预算、容量利用率、上传帧率以及 O/Q/R/Edge 数量。

## 未决项

- 远程服务器地址、SSH 用户和端口由运行者通过环境变量或 SSH 配置提供，不写入仓库。
- 公网部署所需的 TLS、鉴权和限流不在本次演示范围内。
