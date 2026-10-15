# CAVS-OQR 严格因果在线自适应控制实验

本实验在冻结的 RKNN-FP16 三种子 CAVS-OQR 上加入逐帧 projected
primal-dual byte price。它不缓存未来帧：当前帧选中云动作后，必须完成
HTTP 上传、云推理、MAUR 融合和提交，才读取下一张图片。

## 当前诊断结果

| 预算 | 固定 λ 在线 AP | 自适应 λ 诊断 AP | 变化 | η | 改变动作帧数 |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.454842 | 0.454842 | +0.000000 | 0 | 0 |
| 25% | 0.505289 | 0.505374 | +0.000085 | 0.0015 | 84 |
| 40% | 0.529706 | 0.530874 | +0.001168 | 0.002 | 153 |

10% 使用 RTX 在线源；25% 和 40% 使用同一个 Mac-MPS 云后端。自适应点结果
是严格顺序 selection 的诊断性动作补算，不是新的 RK3588 全流程延迟结果。
正式论文系统结果必须按下述命令在板端重新逐帧运行。

## 板端行为合同

每张图严格执行：

`打开当前图 -> RKNN -> 当前四动作打分 -> 当前帧决策 -> 立即 HTTP -> 融合 -> 提交 -> 下一图`

运行合同固定记录：

- `zero_lookahead=true`
- `global_plan=false`
- `buffer_frames=0`

控制器仍使用精确前缀容量，因此任何时刻都不会超过截至当前帧已获得的
10%/25%/40% JPEG payload credit。

## 板端运行

解压发布包后进入实验目录，设置：

```bash
export ONLINE_PYTHON="$HOME/venvs/rknn-yolo26/bin/python"
export CAVS_TOKEN='<与服务器一致的运行令牌>'

COMMON=(
  --sequence "$PWD/board_bundle/sequences/coco_val2017_sequence.json"
  --images-dir /mnt/usb/rk3588/yolo26n_coco_val/coco_val2017/images
  --rknn /mnt/usb/rk3588/yolo26n_coco_val/models/yolo26n-rk3588-fp16.rknn
  --rknn-helpers-dir /mnt/usb/rk3588/yolo26n_coco_val
  --cloud-url http://<SERVER-IP>:8768/infer
  --cloud-sha256 9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92
  --token-env CAVS_TOKEN
  --core 0
  --runs-root "$PWD/runs"
)
```

先做 50 张 40% 冒烟：

```bash
bash board_bundle/run_online.sh \
  --run-id "smoke-cavs-pd-40-$(date +%Y%m%d%H%M%S)" \
  --method cavs_oqr \
  --budget 0.4 \
  --limit 50 \
  --policy board_bundle/policies/cavs_oqr/manifest.json \
  --control board_bundle/controls/cavs_oqr_primal_dual/40.json \
  "${COMMON[@]}"
```

冒烟通过后严格串行跑三档：

```bash
TAG="formal-cavs-pd-$(date +%Y%m%d%H%M%S)"

for spec in "10 0.10" "25 0.25" "40 0.40"; do
  pct="${spec%% *}"
  budget="${spec##* }"
  bash board_bundle/run_online.sh \
    --run-id "${TAG}-${pct}" \
    --method cavs_oqr \
    --budget "$budget" \
    --policy board_bundle/policies/cavs_oqr/manifest.json \
    --control "board_bundle/controls/cavs_oqr_primal_dual/${pct}.json" \
    "${COMMON[@]}"
done
```

不要并行跑三档；否则 NPU、网络和服务器争用会污染端到端延迟。决策本身没有
等待窗口，也不会为了攒预算暂停当前帧。

## 验收

每档必须满足：5,000/5,000、`cloud_failures=0`、最大前缀违规为 0，且
run contract 中 `buffer_frames=0`。最后将 `runs/` 带回 Mac 做 COCO AP 与
延迟汇总。
