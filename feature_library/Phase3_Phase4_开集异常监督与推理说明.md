# Phase 3 / Phase 4 开集异常监督与推理说明

配图：[`diagrams/phase3_phase4_osr_architecture.svg`](diagrams/phase3_phase4_osr_architecture.svg)

本文只记录与开集异常监督直接相关的实现：离线 `A_208` 先验、Phase 3 异常分支与损失、TAL 后的分类 Ignore 机制，以及 Phase 4 的未知异常后处理。YOLO11s-seg 的常规主干和常规已知类检测/实例分割不展开。

> **术语说明**：当前代码调用的是 Ultralytics 的 **TAL（Task-Aligned Assigner）**。本文将“异常先验驱动的分类 Ignore”称为 **ATL-Ignore 层**，它发生在 TAL 已完成正样本分配之后；它不替换 TAL，不生成伪框，也没有第二轮 TAL。

---

## 1. 固定尺度与关键张量

所有 Phase 2、3、4 的空间处理都使用直接拉伸（direct stretch），即原图直接变换到 `832×832`，不使用 letterbox。原因是缓存的 `A_208.npy` 就是在这个坐标系里产生的；如果训练或推理引入 Mosaic、翻转、随机透视或 letterbox，异常先验会与标签、预测发生错位。

| 名称 | 维度 / 空间大小 | 产生阶段 | 用途 |
|---|---:|---|---|
| 输入图像 | `B×3×832×832` | Phase 2/3/4 | 特征库、YOLO 训练与推理统一输入 |
| `Bank_Spatial` | `20×256×104×104`，FP16 | Phase 1 | 20 张正常图的 WRN50 固定工位空间特征库 |
| WRN 查询特征 | `B×256×104×104` | Phase 2 | 与相同位置 `3×3` 正常库邻域做局部相似度匹配 |
| `A_104` | `B×1×104×104` | Phase 2 | 局部匹配得到的异常分数图 |
| `A_208` / `anom_prior` | `B×1×208×208`，FP16 落盘、FP32 训练 | Phase 2/3 | 离线教师异常先验；由 `A_104` 上采样而来 |
| P3 检测点 | `104×104=10,816` | Phase 3 | stride 8 检测尺度 |
| P4 检测点 | `52×52=2,704` | Phase 3 | stride 16 检测尺度 |
| P5 检测点 | `26×26=676` | Phase 3 | stride 32 检测尺度 |
| 检测候选总数 | `14,196` | Phase 3 | `10,816 + 2,704 + 676` |
| Proto 中间特征 | `B×256×208×208` | Phase 3/4 | `proto.cv1 → upsample → proto.cv2` 的输出；异常头输入 |
| 分割 prototype | `B×32×208×208` | Phase 3/4 | `proto.cv3` 输出；沿用标准实例分割路径 |
| `anom_logits` | `B×1×208×208` | Phase 3/4 | `Conv(256→64, 3×3) → Conv(64→1, 1×1)` 输出 |
| `anom_prob` | `B×1×208×208` | Phase 4 | `sigmoid(anom_logits)`，用于未知异常连通域提取 |

当前配置阈值如下，来源于正常样本 leave-one-out 标定：

```yaml
tau_low:  0.07967019379138947
tau_high: 0.27967019379138947
```

---

## 2. 完整流程与启动命令

所有命令使用同一个 Python 环境：

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/OSR-YOLO
```

### 2.1 Phase 1：构建固定工位空间正常库

```bash
cd /home/fth/OSR-YOLO/feature_library
python build_feature_library.py --config config.yaml
```

输出：

```text
feature_library/outputs/bank/bank_spatial.pt
# 预期：20×256×104×104，FP16
```

### 2.2 Phase 2：标定阈值并缓存 `A_208`

```bash
cd /home/fth/OSR-YOLO/feature_library

# Leave-one-out 正常样本标定 tau_low / tau_high
python calibrate_thresholds.py --config config.yaml

# 正常图 QA：检查留一法下的异常响应
python visualize_results.py --config config.yaml --source normal

# 为训练图缓存 A_208.npy 与 manifest.json
python cache_anomaly_maps.py --config config.yaml --source train

# 训练图 Top-K 异常响应可视化
python visualize_results.py --config config.yaml --source train
```

关键产物：

```text
feature_library/outputs/calibration/thresholds.json
feature_library/outputs/anomaly_maps/train/manifest.json
feature_library/outputs/anomaly_maps/train/<image-stem>.npy
```

### 2.3 Phase 3：数据与反向传播检查

```bash
cd /home/fth/OSR-YOLO

# 一个真实 batch：检查 A_208 / anom_logits 尺寸、loss、反向梯度
python -m feature_library.phase3.verify_batch

# 生成输入、GT union、A_208 与 Ignore 区域的对齐图
python -m feature_library.phase3.visualize_batch
```

### 2.4 Phase 3：正式训练

```bash
cd /home/fth/OSR-YOLO
python -m feature_library.phase3.train
```

正式训练参数位于：

```text
feature_library/phase3/config.yaml
# imgsz: 832
# epochs: 100
# batch: 4
# enable_cls_ignore: true
```

训练前必须保持以下空间增强关闭：

```yaml
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
cutmix: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
fliplr: 0.0
flipud: 0.0
multi_scale: 0.0
```

### 2.5 Phase 4：已知类 + 未知异常推理

```bash
cd /home/fth/OSR-YOLO

# 使用 phase4/config.yaml 的 source、checkpoint 和阈值
python -m feature_library.phase4.infer

# 单图推理
python -m feature_library.phase4.infer \
  --source 8v6_front_test_data/images/val/000000_108.png \
  --output-dir feature_library/phase4/outputs_single

# 批量推理前 10 张
python -m feature_library.phase4.infer \
  --source 8v6_front_test_data/images/val \
  --output-dir feature_library/phase4/outputs_val10 \
  --limit 10

# 导出原始三输出 ONNX：known predictions / mask prototypes / anomaly logits
python -m feature_library.phase4.export_onnx \
  --output feature_library/phase4/exports/osr_phase4.onnx
```

> Phase 4 默认 checkpoint 当前指向 1 epoch smoke 产物，仅用于验证端到端接口。正式部署或评估前，必须把 `feature_library/phase4/config.yaml` 中的 `model.checkpoint` 替换为正式 Phase 3 训练的 `best.pt`。

---

## 3. Phase 3：异常监督构造

### 3.1 `A_208` 的来源与对齐约束

`A_208` 不是 YOLO 特征，不是分类标签，也不能自行重算 WRN 相似度；它是离线缓存的单通道异常分数图。

```text
832×832 输入
  ↓ WRN50 layer2 + layer3 融合
256×104×104 查询特征
  ↓ 与 Bank_Spatial 的相同位置 3×3 邻域局部匹配
A_104
  ↓ 双线性上采样
A_208
```

空间匹配的候选数为：

```text
20 张正常图 × 3×3 同位置邻域 = 180 个候选特征 / 查询位置
```

因此 Phase 3 的样本必须与 `A_208` 在同一像素坐标下。`OSRSegDataset` 对图像及分割标签执行固定 `832×832` 直接拉伸，同时根据 `manifest.json` 读取该图对应的 `A_208.npy`。

### 3.2 正、负、Ignore 像素构造

设：

```text
M_known：人工标注的所有已知缺陷实例 mask 的并集，尺寸 B×1×208×208
A：缓存的 A_208 异常先验
G：geom_valid，有效几何区域（本实现的直接拉伸情形通常为全 1）
```

则 anomaly branch 的监督定义为：

```text
positive = M_known OR (A > tau_high)
negative = NOT positive AND (A < tau_low)
valid    = (positive OR negative) AND G
target   = float(positive)
```

中间带：

```text
tau_low <= A <= tau_high
```

不进入 BCE/Dice，避免将“不确定的异常教师响应”硬压为负样本，或硬标为正样本。

当前实现使用严格比较：`A > tau_high` 和 `A < tau_low`；正好等于阈值的像素归入 Ignore。

### 3.3 Masked BCE

异常头原始输出：

```text
Z = anom_logits，尺寸 B×1×208×208
```

逐像素 BCE 后只保留 `valid` 区域：

```text
L_anom_bce = Σ[BCEWithLogits(Z, target) × valid] / max(Σ[valid], 1)
```

结果：人工已知缺陷始终为正监督；只有高置信教师像素会补充为正；低异常像素作为负监督；中间不确定区域不产生梯度。

### 3.4 Masked Dice

先计算：

```text
P = sigmoid(Z) × valid
T = target × valid
```

对每张图：

```text
DiceLoss = 1 - (2 × Σ(P×T) + 1e-5) / (ΣP + ΣT + 1e-5)
```

仅对具有至少一个正有效像素的图像求平均：

```text
positive_images = Σ(T) > 0
L_anom_dice = mean(DiceLoss for positive_images)
```

如果一个 batch 没有任何正有效像素，Dice 设为零梯度零值。这避免纯负图像的 Dice 分母行为主导训练，而 BCE 仍会稳定提供背景监督。

### 3.5 Phase 3 总损失

常规 YOLO-seg 路保留既有的 box、segmentation、classification、DFL 损失。异常分支额外加入 BCE 与 Dice：

```text
L_total = L_box + L_seg + L_cls + L_dfl
          + lambda_anom × (L_anom_bce + L_anom_dice)
```

当前：

```yaml
lambda_anom: 1.0
```

训练日志中会记录：

```text
box_loss, seg_loss, cls_loss, dfl_loss,
anom_bce, anom_dice, cls_ignore,
anom_known_ratio, anom_teacher_ratio, anom_ignore_ratio, anom_valid_ratio
```

### 3.6 当前教师正样本限制

当前 `tau_high=0.279670...` 高于已缓存训练图的最大异常分数（此前统计约 `0.1394`）。因此：

```text
A_208 > tau_high 的 teacher pseudo-positive 数量 = 0
anom_teacher_ratio = 0
```

这不是实现错误；表示目前 Phase 3 的 anomaly branch 正样本主要来自人工已知缺陷 mask。本文档不修改现有阈值、`A_208` 或训练逻辑。后续若要利用教师正样本，需要独立进行阈值再标定和正常图误报评估。

---

## 4. Phase 3：TAL 与 ATL-Ignore 三尺度机制

### 4.1 TAL 先完成已知类正样本分配

标准 YOLO 分割损失首先基于预测分类分数、预测框、GT 框和候选锚点，运行 Task-Aligned Assigner，产生：

```text
fg_mask：B×14196
```

`fg_mask=True` 的候选点是已经由人工已知类 GT 分配的正样本。它们保持标准分类监督，不被 anomaly prior 覆盖。

### 4.2 `A_208` 到 P3 / P4 / P5

为不漏掉一个检测格点覆盖区域内的高异常响应，使用 max pooling，而不是平均池化：

```text
A_208 --MaxPool(2,2)--> A_P3: B×1×104×104
A_208 --MaxPool(4,4)--> A_P4: B×1×52×52
A_208 --MaxPool(8,8)--> A_P5: B×1×26×26
```

三个尺度按检测头候选点顺序 flatten 并拼接：

```text
A_flat: B×14196
```

代码运行时断言 `A_flat.shape[1] == pred_scores.shape[1]`，防止尺度或候选点顺序不一致。

### 4.3 ATL-Ignore 分类权重

对非已知正样本，定义可疑负样本：

```text
suspicious_negative = (NOT fg_mask) AND (A_flat >= tau_low)
```

标准逐类别 BCE 为：

```text
BCE(pred_scores, target_scores)
```

然后使用分类权重：

```text
w = 0,  对 suspicious_negative
w = 1,  对普通背景负样本
w = 1,  对 fg_mask 已知正样本（强制保留）

L_cls = Σ[BCE × w] / max(Σ target_scores, 1)
```

含义是：若一个位置没有被 TAL 分配给人工已知类，但 `A_208` 认为它可能异常，则不再要求主路将它学习为“所有已知类的背景”。这降低未标注/未知缺陷被强行压制为背景的风险。

### 4.4 明确不做的内容

ATL-Ignore 不做以下事情：

```text
不创建第 9 个 YOLO 类
不创建 pseudo box
不创建 pseudo polygon
不将 A_208 作为 TAL 的 GT 框
不运行第二轮 TAL
不替代已知类人工标注
```

它只修改主路的**背景负样本分类 BCE 权重**；box、DFL、实例 mask 的正样本定义仍由 TAL 与人工已知类标签控制。

---

## 5. Phase 4：未知异常推理与去重

### 5.1 并行输出

Phase 4 从同一个 Phase 3 checkpoint 获得两组输出：

```text
已知路：标准 YOLO-seg → 已知类别 box / class / confidence / instance mask
异常路：anom_logits → sigmoid → anom_prob，尺寸 208×208
```

未知异常不是主检测头中的第 9 类。它是部署后处理中的独立结果：

```json
{
  "type": "unknown_anomaly",
  "class_id": null,
  "class_name": "unknown_anomaly"
}
```

### 5.2 先扣除已知实例

对于通过 `known_conf_threshold` 的已知实例：

```text
instance_mask_832
  ↓ nearest resize
instance_mask_208
  ↓ dilate (当前半径 2 px)
known_mask_208
```

然后：

```text
residual_anom = anom_prob
residual_anom[known_mask_208] = 0
```

这一步避免同一实体同时输出“已知缺陷”和“unknown_anomaly”。

### 5.3 未知实例提取

默认参数：

```yaml
known_conf_threshold: 0.25
known_iou_threshold: 0.70
known_mask_dilate_px: 2
unknown_prob_threshold: 0.50
unknown_min_area_208: 4
unknown_max_area_ratio: 0.50
morph_kernel_size: 3
morph_open_iterations: 1
morph_close_iterations: 1
```

处理顺序：

```text
residual_anom
  → threshold(residual >= unknown_prob_threshold)
  → morphological open
  → morphological close
  → 8-connectivity connected components
  → min / max area filter
  → 一个连通域对应一个 unknown_anomaly
```

每个 unknown 实例的分数：

```text
score = 连通域内部 residual 的最大值
peak_score = 最大值
mean_score = 平均值
```

其 `bbox_xyxy_208` 先在 `208×208` 空间产生，再按照横纵比例分别映射到原图；mask 使用 nearest resize 回原图尺寸。

### 5.4 Phase 4 输出布局

```text
feature_library/phase4/outputs/<image-stem>/
├── result.json
├── overlay.png
├── anomaly_heatmap.png
├── anomaly_residual.png
├── known_mask_208.png
├── unknown_mask_208.png
└── masks/
    ├── known_00_class_<id>.png
    └── unknown_00.png
```

批量推理目录额外包含：

```text
summary.json
summary.csv
gallery.html
```

---

## 6. 验证检查表

| 检查项 | 命令 / 观察点 | 通过条件 |
|---|---|---|
| A-map 对齐 | `python -m feature_library.phase3.visualize_batch` | 输入、GT union、`A_208`、Ignore 区域位置一致 |
| 异常头梯度 | `python -m feature_library.phase3.verify_batch` | `pred_anom_logits=[2,1,208,208]`，`anom_grad_norm` 有限且非零 |
| 训练 | `python -m feature_library.phase3.train` | `anom_bce/anom_dice` 被记录，checkpoint 含 `anom_head.*` 参数 |
| 已知实例输出 | Phase 4 `result.json` | known 实例具有类别、置信度、原图 box、mask 文件 |
| 未知实例输出 | Phase 4 `result.json` | unknown 实例 `class_id=null`，具有原图 box、面积和 mask 文件 |
| 去重 | 查看 `overlay.png` 与 `anomaly_residual.png` | 已知实例 mask 覆盖区域不应再产生 unknown 组件 |
| ONNX | `python -m feature_library.phase4.export_onnx` | 输出 `known_predictions`、`mask_prototypes`、`anom_logits` |

---

## 7. 当前状态与使用边界

1. Phase 3 / Phase 4 的端到端实现和 smoke 验证已完成；当前默认 Phase 4 checkpoint 是 1 epoch smoke checkpoint。
2. smoke checkpoint 的 anomaly probability 尚未收敛，可能产生大量 `unknown_anomaly` 连通域；这只证明数据流、维度与后处理闭环有效，不能作为业务指标。
3. 当前阈值下 `anom_teacher_ratio=0`，异常分支正监督主要来自人工已知 mask。不要将此误解为模型已经通过教师先验学会未知缺陷。
4. 正式部署前至少要完成：正式训练 checkpoint 选择、正常图误报率评估、未知缺陷留出集的图像级/区域级召回评估，以及 `unknown_prob_threshold` 与面积过滤阈值的独立校准。
