# 冷启动外挂特征库开集yolo-segment
下面为你重新整理一份**完全剔除 `pseudo_boxes`、不用二轮 TAL、只用“旁路逐像素 Mask Loss”**的纯净版端到端完整方案。

这套架构改动最小、梯度最干净、训练最稳，是目前工业落地性价比最高的设计。

---

### 核心架构图景（先看一眼整体长什么样）

我们在标准 `YOLOv11-seg` 的 `Proto` 模块内部倒数第二层 **`feat_256`（`[B, 256, 160, 160]`）** 上，旁路接一个极轻量的分割小头 `AnomalyHead`（`3×3 Conv -> 1×1 Conv`），输出一张 **`[B, 1, 160, 160]` 的通用异常概率图 `anom_logits`**。
* **主路（原版 `8400` 头，`Nc` 类）**：专心检出并精细分类**已知缺陷**（只在负样本 `cls_loss` 上做 Ignore 屏蔽，不修改结构）。
* **旁路（`[1, 160, 160]` 异常分割头）**：直接受 WRN50 异常图 + 已知 GT 掩码的**逐像素 Mask Loss** 监督，负责抓出**一切未知/漏标缺陷**。

---

### Phase 1：离线建库（20 张 OK 正常图 $\rightarrow$ WRN50 裁判库，只需跑 1 次）

1. **加载冻结模型**：
   加载 ImageNet 预训练 `WideResNet-50-2`，设 `.eval()` 并冻结全部参数。
   *(注意：YOLO 默认输入是 `[0, 1]`，喂给 WRN50 前**必须做 ImageNet 均值方差归一化** `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`。)*
2. **抽取中层特征（坚决不用 `layer4`）**：
   输入 20 张 `640×640` 正常图，抽取 **`layer2`（`[20, 512, 80, 80]`）** 和 **`layer3`（`[20, 1024, 40, 40]`）**。
3. **聚合、拼接与无训练降维**：
   * 对 `layer2` 和 `layer3` 分别做 `AvgPool2d(3, stride=1, padding=1)`（容忍 1~2px 抖动）。
   * 将 `layer3` 双线性插值上采样到 `80×80`，拼接成 `[20, 1536, 80, 80]`。
   * 在通道维用 `AdaptiveAvgPool1d(256)` 压成 `256` 维并做 L2 归一化，得到 **`[20, 256, 80, 80]`**。
4. **按产线工况二选一存库**：
   * **情况 A（工位固定/已按产品外框裁正）**：保留空间先验，直接存为 **`Bank_Spatial` `[20, 256, 80, 80]`**。
   * **情况 B（位置随机/纯随机纹理如布匹、钢板）**：展平成 `[128000, 256]`，Coreset 最远点采样浓缩为 **`Bank_Global` `[4000, 256]`**。

---

### Phase 2：阈值标定与离线缓存（训练前一次性算完）

#### Step 2.1：留一法（Leave-One-Out）标定正常底噪阈值
用 20 张正常图做 20 折交叉验证（每次用 1 张图去查另外 19 张建的库），统计全图所有网格的异常分 $A(h, w) = 1 - \max(\text{cos\_sim})$：
* 取正常分数的 **99% 分位数** 加一点余量作为**真负上限 $\tau_{\text{low}}$**（如 $\tau_{\text{low}} = P_{99}(\text{OK}) + 0.05$）；
* 向上留一段缓冲带作为**异常确认下限 $\tau_{\text{high}}$**（如 $\tau_{\text{high}} = \tau_{\text{low}} + 0.15\sim 0.25$）。

#### Step 2.2：为每张训练图生成 `A_160`（只存一张小图！）
1. 每张训练图过一遍冻结 WRN50（同样提取 L2 归一化 `[256, 80, 80]`）。
2. 计算异常图 `A_80`（`[80, 80]`）：
   * **情况 A**：每个点 `(h, w)` 与 `Bank_Spatial` 对应 `3×3` 邻域 `[h-1:h+2, w-1:w+2]` 共 180 个向量点乘取最大相似度，得 $1 - S_{\max}(h, w)$。
   * **情况 B**：与 `Bank_Global` 矩阵乘取最大相似度，得 $1 - S_{\max}(h, w)$。
3. 双线性插值上采样到 **`160×160`**，存成一张轻量 `float16` 文件 **`A_160.npy`**（单张仅 ~50KB，**先抽查可视化几张，确认正常区黑、未知缺陷区亮，再开始训练！**）。

*(关键避坑：为什么只存 `A_160` 而不提前存 `prior_8400`？因为 YOLO 训练时有随机翻转/仿射/裁剪等数据增强！把 `A_160` 当作一张附加的灰度掩码放进 Dataloader，随图像和 GT 掩码一起做相同的翻转与几何增强，就能保证训练时 100% 像素级对齐！)*

---

### Phase 3：双路训练阶段（网络改造与 Loss 计算）

#### Step 3.1：在 `Segment` 头挂上旁路 `AnomalyHead`
打开 Ultralytics 的 `Segment` 与 `Proto` 模块，将 `Proto` 内部的倒数第二层输出 `feat_256`（`[B, 256, 160, 160]`）引出：
```python
# 在 Segment 头中新增旁路（仅多约 0.15ms 推理耗时）
self.anom_head = nn.Sequential(
    Conv(256, 64, k=3),        # 带 BN + SiLU，缓冲与适配特征
    nn.Conv2d(64, 1, kernel_size=1)
)
# 训练与推理均返回: anom_logits = self.anom_head(feat_256)  -> [B, 1, 160, 160]
```

#### Step 3.2：Loss 计算（改写 `v8SegmentationLoss`，分主路与旁路）

**【主路：原版 `8400` 已知缺陷检测与分割】**
1. 用人工标注的 `Nc` 个已知缺陷走原版 TAL，得到 `fg_mask`（已知正样本）和 `target_scores`。
2. `box_loss`、`dfl_loss`、`seg_loss`（`proto_32`）：**一行不用改**，只在 `fg_mask` 上算。
3. **动态生成 `prior_8400` 并屏蔽 `cls_loss`**：
   直接对当前增强对齐后的 `A_160`（`[B, 1, 160, 160]`）做 GPU 池化（耗时 <0.02ms）：
   * `p3 = F.max_pool2d(A_160, 2, 2)` $\rightarrow$ `80×80`
   * `p4 = F.max_pool2d(A_160, 4, 4)` $\rightarrow$ `40×40`
   * `p5 = F.max_pool2d(A_160, 8, 8)` $\rightarrow$ `20×20`
   * 展平拼接得 `prior_8400`（`[B, 8400]`）。
   * 构造分类权重掩码 `cls_weight`（`[B, 8400, 1]`）：
     * 已知正样本 (`fg_mask == True`) $\rightarrow$ `1.0`
     * 负样本且真负 (`~fg_mask & (prior_8400 < τ_low)`) $\rightarrow$ `1.0`（压向 0）
     * **负样本且可疑 (`~fg_mask & (prior_8400 >= τ_low)`) $\rightarrow$ `0.0`（Ignore 保护！绝不把漏标/未知缺陷压死成背景）**
   * `loss_cls = (F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none') * cls_weight).sum() / target_scores_sum`

**【旁路：逐像素 Mask Loss（在 `[B, 1, 160, 160]` 上算）】**
1. **构建 `160×160` 监督掩码 `gt_anom` 与有效掩码 `valid_mask`**：
   * 把本图所有人工已知缺陷的 GT 掩码合并并缩放到 `160×160` 得到 `gt_known_160`（0 或 1）。
   * **正像素（`Target = 1`）**：`is_pos = (gt_known_160 > 0.5) | (A_160 > τ_high)` $\rightarrow$ `gt_anom = 1.0`, `valid = 1.0`
   * **负像素（`Target = 0`）**：`is_neg = (~is_pos) & (A_160 < τ_low)` $\rightarrow$ `gt_anom = 0.0`, `valid = 1.0`
   * **模糊缓冲区**：`τ_low <= A_160 <= τ_high` 且不在已知 GT 内 $\rightarrow$ `valid = 0.0`（**Ignore 屏蔽，不参与反向传播**）
2. **计算 Masked BCE + Masked Dice Loss（务必加 Dice 防止极小缺陷被大背景淹没！）**：
   ```python
   # 1. Masked BCE
   bce_map = F.binary_cross_entropy_with_logits(anom_logits, gt_anom, reduction='none')
   loss_anom_bce = (bce_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

   # 2. Masked Dice（专治工业微小缺陷正负像素极度不平衡）
   prob = torch.sigmoid(anom_logits) * valid_mask
   target = gt_anom * valid_mask
   inter = (prob * target).sum(dim=(1, 2, 3))
   union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
   loss_anom_dice = (1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)).mean()

   loss_anom = loss_anom_bce + loss_anom_dice
   ```
3. **（可选加分项）`feat_256` 特征层逐像素聚拢/推开约束**：
   在 `feat_256`（`[B, 256, 160, 160]`）L2 归一化后，用同一个 `is_neg` 掩码算出当前 Batch 所有正常像素的均值中心 $c_{\text{ok}}$，让 `is_neg` 像素向 $c_{\text{ok}}$ 靠拢，让 `is_pos` 像素与 $c_{\text{ok}}$ 的余弦相似度小于 `0.2`，进一步防止复杂背景过杀。
4. **总损失汇总**：`total_loss = loss_box + loss_cls + loss_dfl + loss_seg + λ * loss_anom`（建议初期设 `λ = 1.0 ~ 2.0`）。

---

### Phase 4：产线推理与后处理（极速 TensorRT，零外挂）

1. **彻底丢弃 WRN50 与 20 张正常图特征库**。
2. **导出 ONNX / TensorRT FP16 引擎**：模型单次前向直接输出三个张量：
   * `output0`：`[B, 4 + Nc + 32, 8400]`
   * `proto_32`：`[B, 32, 160, 160]`
   * `anom_prob`：`Sigmoid(anom_logits)` `[B, 1, 160, 160]`
3. **后处理流水线（总耗时增加 < 0.3ms）**：
   * **Step 1（已知缺陷）**：对 `output0` 和 `proto_32` 跑标准 YOLOv11-seg 解码 + NMS，拿到所有已知缺陷的 `(Box, 类别名, 置信度, 160×160 实例掩码)`。
   * **Step 2（未知缺陷防重屏蔽）**：把 Step 1 检出的所有已知缺陷实例掩码（可向外膨胀 `2px`）合并成 `known_mask_160`，在 `anom_prob` 上执行 `anom_prob[known_mask_160 > 0] = 0`（防止同一处已知缺陷被重复报成未知缺陷）。
   * **Step 3（未知缺陷提取）**：对剩余的 `anom_prob > 0.5` 做二值化 + `cv2.connectedComponentsWithStats`：
     * 过滤掉面积极小的随机噪点（如 `area < 4` 像素）；
     * 直接得到每一个**未知新缺陷**的像素掩码、外接矩形框以及峰值异常分数，直接触发报警并存图留档，后续可人工标注并入下一代已知类别训练集。

---

### 总结：为什么这套纯逐像素版本是最优选？
1. **无任何伪框噪声污染**：不需要写连通域切 `pseudo_boxes`，也不会因为伪框不准把 YOLO 的框回归头训歪；
2. **数据增强完全免疫**：只存一张 `A_160`，跟着掩码同步做几何变换，GPU 内直接池化出 `prior_8400`，杜绝一切坐标对齐 Bug；
3. **工程闭环极快**：先跑完 Phase 1+2 肉眼看 `A_160` 效果，再加几十行代码改通 Phase 3 的 `AnomalyHead` 和 Loss，几天内就能在产线上验证“已知全分类 + 未知零漏检”。