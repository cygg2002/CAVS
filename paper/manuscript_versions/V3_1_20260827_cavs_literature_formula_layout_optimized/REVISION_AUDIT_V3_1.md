# V3.1 文献、公式与版式审查记录

日期：2026-08-27

## 1. 本轮关键结论

1. CAVS 的创新边界应限定为：在单图目标检测路由中，学习 AP50–95 对齐的图像—像素载荷价值，并用每图实测编码字节联合选择样本成员与载荷。
2. 不能再把“多配置＋资源优化”本身写成首创。TURBO 已联合选择云模型、压缩配置与跨服务带宽；AccMPEG、TileClipper、Hawlader 等已覆盖任务感知质量、区域传输或多档压缩。
3. VOC 跨架构评测有价值，但不是首次。ORIC/MORIC 已用 SSD-MobileNetV3/Faster R-CNN 等检测器对验证其奖励估计；本文新增的是严格字节预算下的 O/Q/ROI 动作价值。
4. 当前题目由泛化的“成本感知”收紧为“严格字节预算”，避免在能耗、云 GPU 和真实 wire cost 未闭环时过度宣称。

## 2. 采用的最近邻文献

- ORIC/MORIC：contextual image reward、图像级 Edge/Original、offload ratio 与逐图阈值执行。  
  https://arxiv.org/html/2410.18919
- Budget-Adaptive Routing：weak-conditioned 与 weak-skipping 的预算依赖切换，明确高预算 weak-forward tax。  
  https://arxiv.org/abs/2606.30919
- TURBO：服务级 utility curve、云模型/压缩配置和带宽/SLO 的联合优化与真实部署。  
  https://doi.org/10.4230/OASIcs.NINeS.2026.18
- Hawlader et al.：真实硬件和网络仿真下的 JPEG/H.265 精度—时延权衡。  
  https://doi.org/10.1016/j.comcom.2023.11.025
- AccMPEG、TileClipper、RegenHance：任务感知编码、区域选择和区域增强的直接先例。

## 3. 已实施的技术修订

- 正式定义在线信息集 $\mathcal F_t$，用可测性表达零前视因果性。
- 将前缀约束写为 $S_t\le\lfloor\rho C_t^o\rfloor+\beta_t$，并明确主协议 $\beta_t=0$。
- 为零检测框情况定义集合池化的分段形式。
- 区分标准化网络输出 $\hat{\bar v}$ 与反标准化调度价值 $\hat v$。
- 明确三个 full-fit seed 分别反标准化后再集成。
- 使排序损失按有效 pair 权重和归一化，与代码一致。
- 将最大 epoch 从错误的 60 改为冻结协议中的 50，并补 batch size 192、gradient clip 5。
- 增加顺序系统时延分解式，明确均值公式不能替代尾时延和吞吐测量。

## 4. 已实施的版式与图形修订

- 缩短中英文题目，突出 Contextual Image–Action Value 与 Hard Byte Budgets。
- 最近邻比较表由密集的是/否矩阵改为“信号—配置—成本—粒度—部署语义”。
- 图 2 直接标注当前只有点估计、paired CI 待补。
- 图 3 改为“绝对 AP＋动作边际增益”双面板。
- 图 4 改为“协议阶梯＋离线到在线差值”双面板。
- 为图 3、图 4 补齐独立图注。

## 5. 当前最主要的论文风险

1. 三数据集最新主比较尚未形成统一 paired bootstrap 闭环。
2. 零前视系统矩阵尚未在同一二进制协议、同一服务器和同一网络路径下完整注册。
3. 三个公开 final/test 已被长期开发访问，不能写成确认性隐藏测试。
4. 目前主要优化成本是 JPEG payload，而非 wire bytes、能耗或云 GPU 成本的统一目标。
5. CAVS 的全局求解器是拉格朗日—修复启发式，尚缺小规模 exact solver 最优性间隙审计。
