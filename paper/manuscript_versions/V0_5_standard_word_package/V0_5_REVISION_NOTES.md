# v0.5 同版本增强说明

本次修订仍保留版本号 v0.5，主要在原 v0.5 基础上做投稿前口径强化，而不是新增 v0.6。

## 主要增强点

1. 保留并强化 `SpeciesNet GT` 作为 real100 的真实标注口径。
2. 新增 `5.1.1 real100 SpeciesNet GT 标注协议`，明确采集、转换、GT 构建、类别映射和 IoU=0.50 评价流程。
3. 在对比方法和评价指标中区分：
   - `Learned v2 online`：证明上板执行和上传率控制；
   - `real100 Table 10(c)`：证明云端复核对端侧漏检的检测收益。
4. 强化表 10(a) 的统计口径：
   - `Learned upload` 是学习器建议上传比例；
   - `Actual upload` 是真实 HTTP 上传次数；
   - `shadow` 模式不控制真实上传；
   - `SIGILL` 是 attempt-level 统计，不是 frame-level。
5. 调整讨论和局限性：不再写“人工校正 GT”，改为样本规模、类别数、场景多样性和 Learned v2 online AP50 闭环仍待补强。
6. 增加“投稿版压缩与后续补充建议”，便于后续转成 JRTIP 英文 12 页稿。

## 下一步最优先补充

若继续补实验，最优先补：`Learned v2 online` 在同一批 `SpeciesNet GT` 帧上的 Precision / Recall / F1 / AP50。这样可以把“上板低开销 + 降上传率 + 真实精度收益”三条证据闭合。
