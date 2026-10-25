# M1 开发指导：从 M0 复核到 adapter + T5/T1 垂直切片

> 用途：新窗口直接按本文开工。先读 `README.md` → `开发路线图_v2.md` 的 §2.8/§2.9/§4.3/§4.4.7 → `docs/M0_审计与闭环.md` → 本文。
> 工作目录：`E:\zcodeAI项目\视频转VMD`。日期：2026-10-25。
> 本文不重复环境的安装细节；环境/Day0-10 基线仍见 `docs/开发交接_新窗口.md`。
>
> **验收追加（2026-10-25）**：M1 数值/adapter 通过；但 Blender 腿部视觉验收无效（腿部网格权重在 D 链，VMD 只写 FK，D 链未跟随）。下一步先修 D 链，见 `docs/开发指导_M1.5_BlenderD链与真实闭环.md`。

## ✅ 执行状态（2026-10-25 收尾）

Step 0/1/2 已全部完成，结果与数值见
[`M1_执行结果_adapter与垂直切片.md`](M1_执行结果_adapter与垂直切片.md)。要点：

- Step 0：`summary_S` 降级（`is_gate=false`），补 world-orientation / jerk / spike / contact
  分项；12 个扰动变体 **12/12** 只打中目标分量（`baselines.json` 的 `checks`）。
- Step 1：`src/vtv/adapter.py` + `scripts/run_adapter.py` + `tests/test_adapter_roundtrip.py`；
  R2→祀 往返身体关节 **1.6e-15 unit**，手臂链 **2.1e-3 unit**（源 GT 非轴向 twist，§5.5）；
  跨模型段方向 **~1e-5°**；slow_factor 0.25/0.5/1.0 → 242 帧。
- Step 2：`outputs/t5_m1/t5_242.vmd` + `t5_720p.mp4`（242 帧 720×1280）+ T1/T2 同规格；
  提弗洛斯/洛茜跨模型 VMD + 抽帧渲染。
- 实跑修掉的坑（本文未预见）：SMPL-X/PMX 模型系镜像、脚的段定义不一致、`stage_yaw`
  必须左乘、根平移单位混用。详见执行结果的 §5。
- 仍未做（有意）：twist 分布（`twist_split` 实验开关默认关）、手/表情、平滑、逐帧 foot-lock、
  真实闭环（渲染→识别→重定向）——最后一项是 M2 第一件事。

---

## 0. 一句话

M0 的“尺子 + 真值 + safe 接口 + GVHMR 入口”已经完成；**下一步是 M1 adapter**，不是后馈网络。先用 R2 safe GT 做 PMX↔canonical 往返，再做 T5/T1 垂直切片；过了 M1 门禁再进 M2。

---

## 1. 已完成的是什么、相当于项目哪个阶段

### 1.1 对照 v2 里程碑

v2 的 M0 是“地基与尺子”（1-2 周）：git/环境、GVHMR 入口、mmd2gltf 求值、三模型 profile、兼容性审计、合成 GT、指标、朴素基线、T5 垂直切片。

当前完成的是 **M0 的核心部分**：

| v2 M0 任务 | 状态 |
|---|---|
| git init + .gitignore | ✅ 完成（未 commit） |
| E: venv + torch/GVHMR 入口 | ✅ 完成 |
| pytorch3d shim | ✅ 完成 + 单测 |
| mmd2gltf vendor + VMD writer | ✅ 完成；修了上游插值解析 bug |
| 三模型 profile | ✅ 完成 |
| 兼容性审计 + safe_mode | ✅ 完成 |
| R2/祀 已知-good 重放 | ✅ RMSE 0.03085 |
| 合成 GT（祀/R2，固定全身三机位） | ✅ 完成 |
| 指标 + 闭环 harness | ✅ 完成（M0 基线） |
| GVHMR T5/T1 | ✅ 完成（242/968 帧，无 NaN） |
| 朴素基线 | ✅ copy/identity/noise/spike/leg |
| T5 第一版 VMD + Blender 720p | ⏳ 未完成；**依赖 adapter，归入 M1** |
| 提弗洛斯/洛茜 GT | ⏳ 未完成；建议用 R2 跨模型 GT，见 §3.3 |

**结论**：项目处在 **M0→M1 的交接点**。M0 的“测量与真值基础”已经成立；M1 的 adapter 还没开始。T5 第一版 VMD 是 M1 的第一个用户可见里程碑。

### 1.2 M0 实际买到了什么

1. **排除了“模型/播放环境坏了”的嫌疑**：三套 PMX 与 Blender armature 是忠实 MMD Tools 导入；父级 0 差异、rest 误差 ~1e-5、多出的只有 `_dummy_/_shadow_`。现有 VTV 的“严重错配”应查 VMD 语义/重定向，不是模型不能接 VMD。
2. **有了真值**：R2→祀 是同模型已知-good；safe GT 250 帧、IK off、重放误差 4e-7。合成闭环可以逐骨/空间/时序打分。
3. **有了 safe 输出契约**：53 根可写骨（23 身体 + 30 手指）、四个 IK off、根位移只写 `センター/グルーブ`、不写 D 链/twist 辅助/肩C。adapter 的目标接口已经明确。
4. **有了 GVHMR 入口**：T5 242 帧、T1 968 帧，参数完整、无 NaN；只保存参数，不渲染。
5. **有了环境与测试**：venv、shim、VMD round-trip、M0 selftest 全通过。

### 1.3 还没买到什么

- 还没有 T5 的 VMD；
- 还没有 SMPL-X→PMX 的 adapter；
- 还没有跨模型渲染验证；
- 还没有结构后馈/流式/手/脸；
- 还没有 commit；还没有正式开源发布。

所以：**M0 是“能测、能看真值、能安全写出 VMD”的地基，不是 VTV 成品。** 下一步 adapter 才是把 GVHMR 参数变成可套模 VMD 的第一段真实管线。

---

## 2. 已落盘产物（复核用）

| 路径 | 内容 |
|---|---|
| `outputs/compatibility/compatibility_report.json` | PMX↔Blender、覆盖率、语义、identity、单骨、known-good |
| `outputs/compatibility/safe_mode_{tiflos,luoxi,si}.json` | 每模型 53 可写骨、IK off、根/平移、morph 白名单、坑位 |
| `outputs/gt/si_R2_safe.vmd` | R2→祀 safe GT，64 轨、250 帧、IK off、0 morph |
| `outputs/synth/si_R2/{front,three_quarter,side}/gt_*.mp4` | 固定全身三机位，1080×1920、30fps、250 帧 |
| `outputs/synth/si_R2/manifest.json` | 相机参数、隐藏集合、GT VMD SHA256 |
| `outputs/closed_loop/m0/baselines.json` | copy/identity/noise/spike/leg 向量分数与逐骨报告 |
| `outputs/gvhmr/T5|T1/` | `gvhmr_result.npz`、meta、check、check.png |
| `src/vtv/mmd/` | PMX/VMD 解析、FK/IK 求值、profile、safe bake、VMD writer |
| `src/vtv/metrics/core.py` | 逐骨/空间/接触/时序指标 + S |
| `src/vtv/audit/checks.py` | 兼容性审计 |
| `src/vtv/canonical.py` | canonical 关节与 PMX 候选名 |
| `scripts/` | 环境、profile、审计、bake、渲染、编码、闭环、GVHMR |
| `tests/` | VMD round-trip、pytorch3d shim |
| `docs/M0_审计与闭环.md` | M0 实测记录 |

**复现测试**（已通过）：

```bash
cd /e/zcodeAI项目/视频转VMD
.venv/Scripts/python.exe tests/test_vmd_roundtrip.py
PYTHONPATH=shim .venv/Scripts/python.exe tests/test_pytorch3d_shim.py
PYTHONPATH=src:shim .venv/Scripts/python.exe scripts/m0_selftest.py
```

---

## 3. 复核结论：哪些可信、哪些要修

### 3.1 可信

- 播放环境：三模型 PMX↔Blender 忠实导入；多出的只有 helper 骨。
- R2/祀：同模型 RMSE 0.03085（Blender 语义 + IK on）；safe GT 重放 4e-7。
- 53 根可写骨三模型同名存在；祀前臂 1.437× 提弗洛斯。
- 合成 GT 固定全身镜头、无相机 VMD、manifest 完整。
- GVHMR T5/T1 无 NaN、尺度正常；T5 242、T1 968。
- 测试全通过。

### 3.2 必须修（P0，接 adapter 前）

#### 3.2.1 打分器 S 不能跨错误类型比较

`src/vtv/metrics/core.py:249` 的 temporal 项是相对 jerk RMS 差，2° 噪声和 3 帧尖峰都会打满 100；`score_S` 被 temporal 主导。结果：

- identity 87.08、2° 噪声 58.54、尖峰 50.94、双腿 +10° 偏置 3.32；
- 腿偏置不是“误差小”：逐骨 thigh 是 10°/10°，p99 10°；只是它平滑、temporal 不爆；
- 角度 mean 0.87° 是被 23 根身体骨平均稀释。

**处理**：

- M1/M2 门禁用**向量分量 + per-bone/P99 + Pareto**，不要用 S 排序；
- `score_S` 改名为摘要/移除；补 `jerk_rms_ratio`、`spike_delta`、`contact_slide`、`contact_penetration`、`contact_recall/precision`；
- 增加针对性基线：脚抬起、脚滑、穿地、根漂移、rest-pose 偏置、twist、IK-on；每个变体应只打中对应分量；
- 当前 contact 在 GT 上 250/250 帧接触（阈值 0.05×height），需要 foot-lift/penetration 变体验证阈值是否合理。

#### 3.2.2 T1 时间重映射未实现

`run_gvhmr.py` 只做 60→30fps 降采样：T1 968、T5 242。T1 是 0.25× 慢速，导出 VMD 必须：

- T1：968 → 242（slow_factor 0.25）
- T2：484 → 242（slow_factor 0.5）
- T5：242 → 242（slow_factor 1.0）

在 adapter/导出层加 `slow_factor`，并加单测断言三条路径都是 242 帧、时间对齐。否则 T1 VMD 会是 4 倍时长。

#### 3.2.3 adapter 的根/朝向/rest 是最大风险

- GVHMR T5 `transl` y = 1.117–1.388m；祀 `センター` rest = 8.74 unit（≈0.699m）。**不要直接写 `transl×12.5` 到 `センター`**，否则角色浮起 0.4–0.7m。用源 rest→目标 rest 的 delta，或 foot-ground 校准；根位移只写 `センター/グルーブ`。
- GVHMR `orient_angle_deg` 均值 164–180°；adapter 必须处理 180° yaw/朝向；首版渲染先确认角色面朝镜头。
- A-pose vs T-pose 逐骨 rest 补偿；`左肩C` 的 -1.0 继承、D 链、twist 辅助、IK off 都必须按 safe_mode 处理。
- 先做 R2 safe GT 的 PMX↔canonical 往返，再上 GVHMR T5。

### 3.3 文档/语义小修（P1）

- `docs/M0_审计与闭环.md` 的 T1 状态已修正为“T5/T1 已跑通”。
- `docs/开发交接_新窗口.md` 第 11 节的下一步已修正为“打分器标定 → M1 adapter → T5/T1 垂直切片”。
- `outputs/gt/si_R2_safe.vmd` 已 dry-run 掉全零 helper/root 链 6 轨，现 58 轨；不再有零 `肩P` 轨（已处理）。
- safe GT 当前 0 morph（body-only）；M0 文档“R2 的 12 个 morph 留到 M3”建议明确写成“safe GT body-only 排除 morph”。
- **提弗洛斯/洛茜 GT 不要追“same-model”**：R1 是 Sunna、R3 是花火，对这两个模型也是跨模型；R1 重定向得到的 GT 会循环验证 adapter。直接用 R2 safe VMD + 各自 safe_mode 烘 **跨模型 GT**；R1/R3 留作 adapter 之后的第二动作测试。
- 仓库已 git init 但未 commit；建议在打分器/文档小修后做 M0 首次提交（outputs 已 ignore）。commit 需要你明确授权。

---

## 4. M1 开发任务（按顺序）

### Step 0：修尺子 + 文档（0.5–1 天）

1. `src/vtv/metrics/core.py`：S 降级为摘要；补 jerk/spike/contact 分项；
2. `scripts/closed_loop.py`：加 foot_lift/foot_slide/penetration/root_drift/rest_bias/twist/ik_on 变体；
3. `baselines.json` 重新生成；确认每个变体只打中对应分量；
4. 修 `M0_审计与闭环.md:196`、`开发交接_新窗口.md` 第 11 节；
5. 重跑测试；如授权，做 M0 首次 commit。

**Step 0 门禁**：向量分数能区分“角度/位置/根/接触/时序”；S 不再用于排序。

### Step 1：adapter 核心 + 往返测试（1–2 天）

新增：

- `src/vtv/adapter.py`：canonical（SMPL-X 22 语义）→ 目标 23 身体骨 + 30 手指；
- `scripts/run_adapter.py`：读 GVHMR `.npz` / canonical 序列 → VMD；
- `tests/test_adapter_roundtrip.py`。

实现要点：

1. **映射**：SMPL-X pelvis/spine1/2/3 → 下半身/上半身/上半身1/上半身2；センター 作为根；clavicle→肩、shoulder→腕、elbow→ひじ、wrist→手首；hip→足、knee→ひざ、ankle→足首、foot→つま先（或 foot 复制）。
2. **rest 补偿**：逐骨源 bind→目标 bind 的 delta；世界朝向 delta；A-pose/T-pose。
3. **根**：源 rest→目标 rest 的 delta；foot-ground 校准；只写 `センター/グルーブ`；不要绝对 transl。
4. **朝向**：180° yaw/手性/0.08 缩放；首版渲染检查面朝。
5. **safe_mode**：只写 53 可写骨；四 IK off；不写 D 链/twist 辅助/肩C；morph 暂不写。
6. **slow_factor**：T1 0.25、T2 0.5、T5 1.0；导出 242 帧。
7. **VMD**：30fps、xyzw、0 起、IK 段 20 字节全角名。

往返测试 A（同模型，最干净）：

```
R2 safe GT（祀）
  → PMX FK → canonical 关节角
  → adapter → PMX local
  → PMX FK
  → 与 R2 safe GT 逐关节比较（世界位置/旋转）
```

目标：祀上 <1e-3（非 T 帧，250 帧）。

往返测试 B（跨模型）：

```
R2 safe GT（祀）→ canonical → adapter(tiflos/luoxi)
→ 归一化角度/位置比较 + 渲染
```

这不是“same-model GT”，是跨模型适配测试；不要用 R1 重定向结果当 GT。

### Step 2：T5/T1 垂直切片（1–2 天）

1. T5：`outputs/gvhmr/T5/gvhmr_result.npz` → adapter → `outputs/t5_m1/t5_242.vmd` → Blender 720p 渲染；
2. T1：`outputs/gvhmr/T1/gvhmr_result.npz` → slow_factor 0.25 → 242 帧 VMD；T2 0.5 → 242；
3. 检查：脚贴地、面朝镜头、手臂无反向、肩不塌、根不漂、IK 不拖腿、无 twist 缠绕；
4. 用向量指标（2D 重投影/根/接触/jerk）记录；不用 S 当门禁。

**这一步的产物**：第一版 T5 VMD + 720p 渲染。M0 文档把它列为 M1 前置，它是用户第一次能看到“视频→VMD→套模”的完整闭环。

### Step 3：跨模型 + M1 门禁（1 天）

- R2 safe GT → 祀（参考）→ adapter → 提弗洛斯/洛茜；渲染固定全身；
- 逐骨/分组比较：肩/肘/腕/髋/膝/踝/脊柱；归一化；
- M1 门禁：
  - [x] R2→祀 往返：非手臂 <1e-6（实测 1.6e-15）；手臂 <3e-3 且记录源 GT twist 限制（实测 2.1e-3）；twist_split 留 M1.1；
  - [x] T5 242 帧 VMD + Blender 720p，脚贴地/面朝/无 gross twisting；
  - [x] T1/T2 slow_factor 后 242 帧，**帧数/时间轴**一致；三速不是同一次 take，不能当姿态一致性门禁（M1 实测）；
  - [x] 三模型都能套，无 IK 拖腿/无 helper 写入；
  - [x] 向量指标记录；S 不用于门禁。

### Step 4：之后才是 M2

M1 门禁过了，再做结构后馈/流式；不要在 adapter 往返和 T5 切片之前写后馈网络。

---

## 5. 新窗口开场提示（可直接用）

> 读 `README.md`、`开发路线图_v2.md` 的 §2.8/§2.9/§4.3/§4.4.7、`docs/M0_审计与闭环.md`、`docs/开发指导_M1_adapter与垂直切片.md`。
> 先按 Step 0 修打分器/文档，再做 Step 1 adapter 往返（R2→祀），然后 Step 2 T5/T1 垂直切片。
> 不要先写后馈网络；不要用 S 当门禁；不要为提弗洛斯/洛茜追 same-model GT；T1 必须 slow_factor 0.25；根位移不要写绝对 transl。

常用命令：

```bash
cd /e/zcodeAI项目/视频转VMD
source .venv/Scripts/activate

# 复核
python tests/test_vmd_roundtrip.py
PYTHONPATH=shim python tests/test_pytorch3d_shim.py
PYTHONPATH=src:shim python scripts/m0_selftest.py

# 已有产物
python scripts/closed_loop.py --help
"D:/blender/blender.exe" --background "E:/zcodeAI项目/AIblender/三人_王牌竞速_20261024.blend" \
  --python scripts/render_synth.py -- --help
```

---

## 6. 不要做什么

1. 不要先写后馈网络/流式/手/脸；M1 adapter 还没过。
2. 不要用 `score_S` 做 M1/M2 门禁或排序；用向量/per-bone/Pareto。
3. 不要为提弗洛斯/洛茜追“same-model GT”；R2 跨模型 GT 是当前最干净的。
4. 不要把 GVHMR `transl` 绝对值写进 `センター`；先 foot-ground 校准。
5. 不要写 D 链/twist 辅助/肩C/`_dummy_/_shadow_`；只写 53 可写骨 + 四个 IK off。
6. 不要用镜头 VMD 做合成 GT；固定全身镜头优先。
7. 不要提交 `outputs/`、`profiles/`、`third_party/`、venv、测试媒体/权重。

---

## 7. 项目进度对照（给规划用）

```
v2 M0  地基与尺子        ████████░  核心完成；剩扰动变体/跨模型 GT；T5 切片归 M1
v2 M1  canonical+adapter ░░░░░░░░░  下一步；adapter/往返/T5-T1 垂直切片
v2 M2  结构后馈+流式     ░░░░░░░░░  未开始
v2 M3  手/表情           ░░░░░░░░░  未开始
v2 M4  产线/集成         ░░░░░░░░░  未开始
```

已完成的 M0 是项目的**测量基础 + 真值基础 + safe 输出契约 + 前馈入口**；它不直接产出好看的 T5 VMD，但决定了后面每一步“有没有变好”能不能被量化。M1 adapter 是第一段真正把参数变成可套模 VMD 的管线。
