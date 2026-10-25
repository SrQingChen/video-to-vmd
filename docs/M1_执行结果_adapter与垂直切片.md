# M1 执行结果：canonical adapter + T5/T1/T2 垂直切片

> 日期：2026-10-25。工作目录：`E:\zcodeAI项目\视频转VMD`。
>
> **验收追加（2026-10-25）**：数值/PMX 求值通过；但 Blender 腿部视觉验收无效：腿部网格权重在 D 链，MMD Tools 导入 VMD 后 D 链未跟随 FK（T5 帧 26 右膝 51.2°，渲染帧 27 腿直）。修复见 `docs/开发指导_M1.5_BlenderD链与真实闭环.md`。
>
> 本文是 Step 0-2 的实测记录；规划口径见 `开发指导_M1_adapter与垂直切片.md` 与
> `开发路线图_v2.md` §2.8/§2.9/§4.3/§4.4.7。**没有写后馈网络，没有用 S 做门禁。**

## 1. 结论先行

1. **尺子已标定**：12/12 扰动变体只打中目标分量（`outputs/closed_loop/m0/baselines.json`
   的 `checks`），`summary_S` 降级为只读摘要。
2. **adapter 已落盘并跑通**：SMPL-X/PMX → canonical（23 关节世界朝向）→ 目标 PMX
   23 身骨 + 4 IK off，安全白名单、rest 补偿、根 delta、slow_factor 全部实现。
3. **同模型往返精确**：R2→祀 250 帧，18 个非手臂身体关节世界位置最大误差 **1.6e-15 unit**，
   世界朝向最大 **1e-5°**；手臂链（肘/手）因源 GT 的 4.5° 非轴向 twist 残留 **2.1e-3 unit**
   （0.17 mm，见 §5.1、§5.5）。
4. **跨模型方向精确**：R2→提弗洛斯/洛茜，段方向与 `D·源方向` 的偏差 **1.1e-5° / 5.7e-6°**，
   脚底校准 |趾尖−rest| < 0.2 unit。
5. **T5 第一版 VMD + 720p 渲染完成**：242 帧 / 8.07 s / 23 轨 / IK off，角色面朝镜头、
   脚在地面（起跳帧离地）、手臂没有反向、肩不塌、根不漂。T1（0.25×）与 T2（0.5×）
   同样导出 242 帧。
6. **三模型都能套**：祀（T5/T1/T2）、提弗洛斯/洛茜（R2 跨模型）都渲染过，无 helper 写入、
   无 IK 拖腿。**提弗洛斯/洛茜没有追 same-model GT**（按指导：R2 + 各自 safe_mode）。
7. 仍不做：手/表情（M3）、后馈/流式（M2）、twist 分布（M1.1，见 §5.1）、平滑（M2/F）。

## 2. 交付物

| 路径 | 内容 |
|---|---|
| `src/vtv/canonical_body.py` | SMPL-X 22 语义、canonical 23 关节树、段方向/腿长表、映射自检 |
| `src/vtv/adapter.py` | `canonical_from_smplx` / `canonical_from_pmx` / `adapt` / 四元数工具 / FK |
| `scripts/run_adapter.py` | `--npz`（GVHMR）或 `--vmd`（PMX 源）→ VMD + `<out>.json` 诊断报告 |
| `scripts/m1_slice_report.py` | 无 GT 时的绝对稳定性报告（接触/滑动/穿地/jerk/根/骨长） |
| `tests/test_adapter_roundtrip.py` | 同模型往返、跨模型方向不变式、slow_factor 三速 |
| `outputs/t5_m1/t5_242.vmd` + `t5_720p.mp4` + `front/` + `report.json` | T5 第一版切片 |
| `outputs/t1_m1/`、`outputs/t2_m1/` | T1（0.25×）全片 720p 渲染 + T2（0.5×）抽帧渲染；两段都有 242 帧 VMD + 报告 |
| `outputs/adapter/si_R2_to_{tiflos,luoxi}.vmd` + `render_tiflos/` | 跨模型适配证据（提弗洛斯抽帧渲染） |
| `outputs/gvhmr/T2/` | T2 的 GVHMR 结果（484 帧 30fps；预处理 + 推理同 T5 脚本） |

## 3. 算法（实现口径）

```
canonical motion = {每帧 23 关节世界朝向 G_src(j), 根（骨盆）世界位置, 源 rest 关节/方向}

G_tgt(j)  = Y · ( D_j · G_src(j) · D_j⁻¹ )        # 世界朝向
R_local   = inv( W(parent) · inherit(ratio) ) · G_tgt(j)   # 沿真实 PMX 链
root_t    = P_tgt(pelvis) − C_rest − R_root·(P_rest − C_rest)   # 只写センター
P_tgt(t)  = P_rest(pelvis) + s·(P_src(t) − P_src_rest)/unit + (0, ground_shift, 0)
```

- `D_j`：源 rest 段方向 → 目标 rest 段方向的最小旋转（A/T-pose 补偿）；叶骨（手/趾）继承父骨；
  **脚/趾不做 swing**（§5.2）。
- `Y`：模型 rest 朝向差（SMPL-X 模板面朝 +Z，PMX 面朝 −Z）= 整身 180° yaw，
  **左乘世界朝向 + 同时转根位移**；不能做共轭（共轭会逐骨重扭，§5.3）。
- 父级世界朝向按 MMD 语义算：付与親比例（肩C −1.0、捩1 0.5）、层级、全角骨名都走真实链，
  所以“父级是未写的 helper”这种坑不会错。
- 缩放 `s = 目标(骨盆→踝)/源(骨盆→踝)`：腿长决定“同一姿态下骨盆离地多高”，比全身体高比更稳。
- 脚底校准：`ground_shift` 是**全片一个常数**，保持源“最低趾尖离地多少”的比例，
  不逐帧贴地（逐帧会把跳跃吃掉）。
- 导出：23 轨（`safe_mode` 可写骨里属于 23 身骨的那部分）+ 4 个 IK off 帧（全角名，20 字节）；
  不写 D 链、捩1、肩C、肩P、`_dummy_/_shadow_`；不写 morph（body-only）。
- GVHMR 源：`S = diag(-1,1,1)`（模型系镜像，见 §5.1）先把世界卷积成 MMD-like；
  `slow_factor` 决定输出帧 = `round(k/slow_factor)`：T1 0.25、T2 0.5、T5 1.0。

## 4. 实测

### 4.1 M1 门禁

- [x] **R2→祀 往返**：18 个非手臂身体关节世界位置 max **1.57e-15 unit**；手臂链（肘/手）
      **2.15e-3 unit**（源 GT 的非轴向 twist，见 §5.5）；世界朝向除上臂/前臂 roll 外
      max **1e-5°**（手/头/脊柱/腿全部精确）；ground_shift **−1.8e-15 unit**。
      同一测试还断言：23 轨、无 helper、IK off、帧范围 0–249。
- [x] **T5 242 帧 VMD + Blender 720×1280 渲染**：角色面朝镜头；双脚在地面
      （跳跃帧离地）；手臂与源视频同向；肩不塌；根不漂；无 gross twisting。
- [x] **T1/T2 slow_factor 后 242 帧**：968→242（0.25）、484→242（0.5）、242→242（1.0），
      输出帧到源帧的映射经单测逐帧断言。
- [x] **三模型都能套**：祀（T5/T1/T2 渲染）、提弗洛斯（R2 帧 60 渲染）、洛茜（数值方向
      不变式 5.7e-6° + VMD 落盘）；均无 IK 拖腿/helper 写入。
- [x] **向量指标记录、S 不用于门禁**：`report.json` 只有绝对量；S 字段带
      `is_gate=false`，`m0_selftest` 会检查它。

### 4.2 导出稳定性（无 GT，自报绝对量）

`scripts/m1_slice_report.py`，接触阈值 = 该片自身最低趾尖 + 1% 身高。

| clip | 帧 | 轨 | IK off | 接触帧比 (L/R) | 着地滑动 unit/s | 穿地 %身高 | 趾尖最低 unit | jerk RMS | MAD 毛刺 | 根路径 %身高 | 根 Y 范围 unit |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|
| T5 | 242 | 23 | ✓ | 2.9% / 2.5% | 0.90 / 0.77 | 0.0 | 0.38 / 0.29 | 12995 | 15 | 133 | 3.58 |
| T1（0.25×） | 242 | 23 | ✓ | 4.1% / 2.1% | 2.30 / 4.89 | 0.0 | 0.45 / 0.04 | 23226 | 64 | 281 | 2.61 |
| T2（0.5×） | 242 | 23 | ✓ | 3.3% / 2.1% | 0.73 / 1.77 | 0.10 | 0.36 / −0.02 | 15649 | 35 | 207 | 3.83 |

读法：

- 接触帧比低是因为这三段是**高速/跳跃舞蹈**：两脚最低点中位 2.5 unit（20 cm），
  只有落地瞬间进 2.2 cm 带；趾尖最低 0.29–0.45 unit ≈ 贴地，**没有整体漂浮**。
- `jerk RMS` 与 M0 的 R2 GT（0.12）差 5 个数量级：T5 是高速舞（源自身手臂 10–14°/帧），
  且 M1 **故意不做平滑**。这是 M2 选择性清理的首要目标，不是 adapter 的误差。
- 着地滑动 0.7–4.9 unit/s：全局脚底校准没有逐帧 foot-lock，脚在着地区间会随根/腿漂几
  unit；M2 接触项 + 后馈修。
- T2 有 0.10% 身高（2 mm）的趾尖穿地，属校准常数与脚趾几何差，可接受。

### 4.3 slow_factor 三速一致性（限定）

T1/T2/T5 都产出 242 帧 / 8.07 s，时间映射精确。但把三段 canonical 姿态逐帧比，
上臂/前臂差 10–18°、根位移差 0.30 m——**它们不是同一次 take**（同名“同舞”但不同渲染/表演）。
所以“三速自一致”只验证到**帧数与时间轴**，不能当姿态一致性证据。

## 5. 实跑发现的坑（都已修，写清原因）

> 附：`scripts/render_synth.py` 新增 `--bbox-frames`（默认 8）：相机按**全片**包围盒取景，
> 否则 2 m 的根位移会走出画面。M0 的三机位 GT 相机已存在 `manifest.json` 里，不受影响。

### 5.1 SMPL-X 与 PMX 的模型系是镜像的

- 实测：SMPL-X 模板 `Lhip=+X`、`踝→趾=+Z`（面朝 +Z，右手系）；PMX `左足=+X`、`足首→つま先=−Z`
  （面朝 −Z，MMD 左手系）。两侧“左”都在 +X，但“前”反号 → 模型系差一个 **X 镜像**。
- 不做换算时，`D_j` 的 swing 会把“前伸的手臂”当成“向后”的 rest 差异来补偿：T5 首版渲染
  背对镜头、手臂反向。
- 修法：GVHMR 源在 `canonical_from_smplx` 里做 `S=diag(-1,1,1)`（位置 `x→−x`，姿态
  `(x,y,z,w)→(x,−y,−z,w)`），再叠加模型 rest 朝向差的 180° yaw。
- **附带结论**：T5 的渲染必须带这个 180°（`run_adapter` 对 SMPL-X 源默认 `stage_yaw=180`），
  它不是“调参”，是模型约定差。

### 5.2 脚的 rest 方向不是同一段解剖

SMPL-X 的 ankle→foot joint 是**踝到跖骨头**（近水平），PMX 的 足首→つま先 是**踝到趾尖**
（下前 45°）。按方向做 swing 会给出 ~100° 的补偿：脚掌朝天、角色整体下沉 4 unit
（T5 首版 ground_shift = −4.0）。修法：`NO_SWING = (foot_l/r, toe_l/r)`，脚只做世界朝向直通
（脚是“朝向”不是“肢段”）。

### 5.3 `stage_yaw` 必须左乘，不能共轭

第一版把它塞进 `D_j`（`G = D G_src D⁻¹` 里 D 含 yaw），等于给每根骨各扭一下：角色朝向变了但
姿态被拧坏（手臂抬到脸侧）。整身旋转的正确写法是 `G_tgt = Y · (…)` **并且根位移一起转**。

### 5.4 根平移单位与锚点

- `root_pos`（米）与目标 rest（MMD unit）混用会让根位移放大 12.5×；同模型往返因 scale=1 恰好
  自洽，只有跨速度/跨模型才暴露，已加单测（`slow_*` 的 x 映射与 `ground_shift≈0`）。
- 锚点用“源站立时骨盆离地高度”（`j0.y − min(rest 脚 y)`），不是 `transl` 绝对值；
  `BodyModelSMPLX` 的 rest 原点不在骨盆，用绝对 y 会差 0.3 m。

### 5.5 源 GT 的 twist 骨带 4.5° 非轴向分量

R2 的 safe GT 里 `腕捩/手捩` 有 54°/20° 的轴向旋转（正常），但含 ≤4.5° 的非轴向分量；
它会把肘的位置推 0.1 unit。canonical 用“段帧 + 方向修正”读 PMX 源：仍差 2.1e-3 unit
（骨段被非轴向分量轻微“折短”），单骨无法表示。MMD 用 `fixed_axis` 会夹掉非轴向部分，
Blender 不会——这是 M0 GT 的已知 Blender/MMD 语义差，记录在案。

## 6. 已知限制（M1 不做，留给后续）

1. **手臂 roll 分布**：源用 `腕捩/手捩` 分布扭转；body-only 导出把轴向扭转放在上臂/前臂，
   所以上臂世界 roll 差最多 54°、前臂 16–20°（位置与末端朝向不受影响）。
   `adapt(twist_split=True)` 是实验开关，**默认关**：世界对齐骨骼系下 swing-twist 分解会把
   “肘部弯曲”误判为轴向扭转（A-pose 手臂的弯曲轴是世界 X），需要解剖 twist 分解 + 权重
   验证后再开。
2. **手/表情**：GVHMR 不给手部姿态，23 身骨 body-only；morph 0。M3。
3. **抖动/脚滑**：M1 不做平滑、不做逐帧 foot-lock。M2/F。
4. **T3/T4** 不做单人验收（2D Q 版 / 三人场景）。
5. **三速**只验证帧数与时间映射（不是同一 take）。

## 7. 复现

```bash
cd /e/zcodeAI项目/视频转VMD
source .venv/Scripts/activate

# Step 0 尺子
PYTHONPATH=src python scripts/closed_loop.py          # 12/12 checks

# Step 1 adapter
python tests/test_adapter_roundtrip.py                 # 往返/跨模型/slow_factor

# Step 2 垂直切片
PYTHONPATH=src python scripts/run_adapter.py --npz outputs/gvhmr/T5/gvhmr_result.npz \
  --model si --out outputs/t5_m1/t5_242.vmd
PYTHONPATH=src python scripts/run_adapter.py --npz outputs/gvhmr/T1/gvhmr_result.npz \
  --model si --slow-factor 0.25 --out outputs/t1_m1/t1_242.vmd
PYTHONPATH=src python scripts/run_adapter.py --npz outputs/gvhmr/T2/gvhmr_result.npz \
  --model si --slow-factor 0.5 --out outputs/t2_m1/t2_242.vmd
PYTHONPATH=src python scripts/m1_slice_report.py --vmd outputs/t5_m1/t5_242.vmd \
  --out outputs/t5_m1/report.json
"D:/blender/blender.exe" --background "E:/zcodeAI项目/AIblender/三人_王牌竞速_20261024.blend" \
  --python scripts/render_synth.py -- --model si --vmd outputs/t5_m1/t5_242.vmd \
  --view front --out outputs/t5_m1/front --frames 1:242 --res 720x1280 --samples 8

# 跨模型
PYTHONPATH=src python scripts/run_adapter.py --vmd outputs/gt/si_R2_safe.vmd \
  --model tiflos --out outputs/adapter/si_R2_to_tiflos.vmd
```

## 8. 下一步（M2 之前先做的两件小事）

1. 把真实闭环接上：`outputs/synth/si_R2/*/gt_*.mp4` → GVHMR → adapter → VMD' →
   `closed_loop.py --pred` 打分，拿到「渲染→识别→重定向」的真实基线（M1 只做了 VMD 级往返）。
2. 开 M2 的第一刀：**选择性清理**（Hampel/中值只修 flagged 帧）先把 T5 的 jerk 从 1e4
   量级拉下来，再谈结构后馈；门禁仍用向量分量 + per-bone/P99，不看 S。
