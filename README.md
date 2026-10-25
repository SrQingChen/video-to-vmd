# 高精度视频→VMD（VTV）v2

**目标**：把单目视频（AI 动漫舞蹈 / 真人舞蹈）转成可直接套到三套标准 PMX 的 VMD，尽量不需要手 K；后馈是**跨身份结构微反馈**，不是像素/剪影对比；支持流式补全、手/表情分级降级；开源研究工具。

## 核心判断（先读这 6 条）

1. **视频角色 ≠ 目标 PMX**，像素/全剪影不可比；后馈用 2D 关节、肢体方向、关节角、部件/光流、深度序、接触、时序、目标约束。
2. 源与目标之间用 **canonical body** 做中间语言；每模型 **adapter** 处理 rest pose、P/C、D 链、twist、IK、缩放、morph。
3. **VMD 不是跨模型可移植格式**；套模错配必须诊断到骨名/rest/层级/付与親/twist/IK/缩放/格式/Blender armature 的具体一项。
4. **合成闭环**（GT VMD→渲染→跑管线→VMD'→逐骨/空间/综合评分）是验证+优化+训练+回归的飞轮；**默认固定全身镜头，不用动作文件里的镜头配布**。
5. 后馈同时追求**精度、稳定、跨模型一致、完整度**；不是只修毛刺。
6. **开源**：代码 Apache-2.0；默认 research profile（GVHMR/SMPL-X 非商业）；权重/PMX/VMD/测试视频不入库。

## 当前状态（2026-10-25）

- [x] 调研/文献/九大根因/格式事实
- [x] 10 个代码仓库；GVHMR/HMR2/ViTPose/YOLO/DWPose/SAM2/DAv2；SMPL-X/SMPL；R1-R3；T1-T5
- [x] 规划 v2（执行文档）
- [x] M0：git/环境/兼容性审计/合成 GT（祀·R2，固定三机位）/指标+闭环/安全 VMD 烘焙/GVHMR T5/T1
  - 见 [docs/M0_审计与闭环.md](docs/M0_审计与闭环.md)
- [x] **Step 0 尺子标定**：`summary_S` 降级为只读摘要；补 world-orientation/jerk/spike/contact
  分项；12 个扰动变体 **12/12** 只打中目标分量（`outputs/closed_loop/m0/baselines.json`）
- [x] **Step 1 adapter**：`src/vtv/adapter.py`（SMPL-X/PMX → canonical → PMX 23 身骨 + IK off，
  rest/A-pose/根 delta/帧镜像/slow_factor）；R2→祀 往返身体关节 **1.6e-15 unit**，
  跨模型段方向 **~1e-5°**
- [x] **Step 2 垂直切片**：T5/T1(0.25×)/T2(0.5×) 各 242 帧 VMD + 720×1280 渲染；
  提弗洛斯/洛茜跨模型套模；见 [docs/M1_执行结果_adapter与垂直切片.md](docs/M1_执行结果_adapter与垂直切片.md)
  - ⚠️ **Blender 腿部视觉待修**：腿部网格权重在 D 链，VMD 只写 FK，Blender 里 D 链未跟随；T5 帧 26 右膝 51° 但渲染帧 27 腿直。数值/PMX 求值通过，视觉腿验收无效；见 [docs/开发指导_M1.5_BlenderD链与真实闭环.md](docs/开发指导_M1.5_BlenderD链与真实闭环.md)。
- [ ] M2：真实闭环基线（渲染→GVHMR→adapter→打分）、选择性清理（T5 抖动）、结构后馈
- 新窗口开工：先读 [docs/M1_执行结果_adapter与垂直切片.md](docs/M1_执行结果_adapter与垂直切片.md)；下一步见 §下一步（M2 入口）。

## 环境事实（实测）

| 项 | 值 |
|---|---|
| 硬件 | Windows + RTX 4060 Laptop 8GB + 15.6GB RAM |
| Python | 3.10.11（系统；venv 必须独立） |
| 系统 torch | 2.7.1+cu118（不要直接用；venv 用 2.3.0+cu121） |
| Blender | 5.2.2 LTS，`D:\blender\blender.exe` |
| MMD Tools | 已装：`C:\Users\asus\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\user_default\mmd_tools` |
| ffmpeg | `D:\ffmpeg\bin` |
| C: 空间 | 仅 ~8.7GB 空闲（99%）；venv/pip cache/TEMP/HF cache 必须放 E: |
| E: 空间 | ~158GB 空闲 |
| CUDA Toolkit | 未装；v2 不依赖 nvdiffrast/pytorch3d 生产编译 |
| VS2022 | 已装（仅在需要编译时用） |

## 核心文档

| 文档 | 用途 |
|------|------|
| [开发路线图_v2.md](开发路线图_v2.md) | **执行规划**：架构、模块、adapter、结构后馈、流式、合成闭环、里程碑 |
| [docs/开发交接_新窗口.md](docs/开发交接_新窗口.md) | **环境/Day 0-10 基线**：安装、命令、坑位；M0 已完成 |
| [docs/开发指导_M1_adapter与垂直切片.md](docs/开发指导_M1_adapter与垂直切片.md) | **M1 开工**：复核结论、修复项、adapter、R2→祀往返、T5/T1 垂直切片 |
| [docs/M1_执行结果_adapter与垂直切片.md](docs/M1_执行结果_adapter与垂直切片.md) | **M1 实测**：往返/跨模型/slow_factor 数值、三个实跑坑、T5-T1-T2 切片、已知限制 |
| [docs/开发前准备清单.md](docs/开发前准备清单.md) | 已有/待准备/缺失项，M0-M2 是否有阻塞 |
| [高精度VTV调研报告与实现方案_2026-10-25.md](高精度VTV调研报告与实现方案_2026-10-25.md) | 文献/根因/格式事实；顶部有 **v2 修订说明**，执行以 v2 为准 |
| [参考文献与资源汇总.md](参考文献与资源汇总.md) | 文献/仓库链接 |
| [素材与依赖清单.md](素材与依赖清单.md) | 素材/权重/环境/待办状态 |
| [开发路线图_v1.md](开发路线图_v1.md) | ⚠️ 已废弃，仅历史记录 |

## 目录结构

```
视频转VMD/
├── README.md
├── 开发路线图_v2.md              # 执行规划
├── 开发路线图_v1.md              # 已废弃
├── 高精度VTV调研报告...md        # 文献/根因（含 v2 修订说明）
├── 参考文献与资源汇总.md
├── 素材与依赖清单.md
├── docs/
│   ├── 开发交接_新窗口.md
│   ├── 开发前准备清单.md
│   ├── M0_审计与闭环.md
│   ├── 开发指导_M1_adapter与垂直切片.md
│   └── M1_执行结果_adapter与垂直切片.md
├── third_party/                  # 10 个仓库（shallow clone）
│   ├── GVHMR/                    # 前馈主模型（SMPL-X body）
│   ├── WHAM/                     # 前馈兜底（MIT 代码）
│   ├── mmd2gltf-gui/             # PMX/VMD 求值参考（MIT）
│   ├── reze-mipo/                # 骨名/VMD 写出参考（GPL，不复制代码）
│   ├── reze-rig/                 # rest-pose 重定向参考
│   ├── vmd_sizing/               # 体型适配参考
│   ├── PMX-VMD-Scripting-Tools/  # PMX/VMD 工具/格式文档
│   ├── ScoreHMR/                 # 运动先验/引导参考
│   ├── UnderPressure/            # 脚接触检测参考
│   └── MotionCritic/             # 动作质量评判参考
├── models/                       # DWPose / SAM2 / DAv2
├── body_models/                  # SMPL-X v1.1 / SMPL v1.1.0（junction 到 GVHMR）
├── testdata/                     # T1/T2/T5 主测试；T3/T4 非主验收；R1-R3 GT
└── src/vtv/                     # 已建：mmd/metrics/audit/canonical/adapter/synth
```

## 管线一览

```
视频
 → 证据层（2D 身体/手/脸、部件掩码、光流、相对深度序、相机、置信度）
 → canonical body（GVHMR 初始化 + 滑窗优化 + 运动先验 + 接触）
 → 目标 adapter（每模型 profile：rest/P-C/D 链/twist/IK/缩放/morph）
 → 结构后馈（目标关节/肢体投影 vs 源证据；低维残差；2-3 轮）
 → 流式/缺口补全（状态/先验/置信度/lookahead）
 → 选择性清理（毛刺/抖动/接触/四元数）
 → 手部/表情（L0-L3 置信度阶梯 + 自然 fallback）
 → VMD（30fps/xyzw/IK-off/每模型骨名/morph）→ Blender/MMD 验证
```

## 下一步（M1.5 入口）

**M0 + M1 数值/adapter 已完成**：尺子标定、adapter（R2→祀往返 1.6e-15 unit）、T5/T1/T2 242 帧 VMD、三模型套模。但 **Blender 腿部视觉待 D 链修复**：腿部网格权重在 D 链，VMD 只写 FK，Blender 里 D 链未跟随；T5 帧 26 膝 51° 渲染腿直。见 [docs/开发指导_M1.5_BlenderD链与真实闭环.md](docs/开发指导_M1.5_BlenderD链与真实闭环.md)。

1. **先修 Blender D 链**：bake/约束 D/twist/肩C，用 T5 帧 26/27 + 60° 测试 VMD 验证腿真的弯；数值/PMX 求值已通过，视觉腿验收在修复后才有效；
2. **再接真实闭环基线**：`outputs/synth/si_R2/{front,three_quarter,side}/gt_*.mp4` →
   `run_gvhmr.py` → `run_adapter.py` → `closed_loop.py --pred`，拿到「渲染→识别→重定向」的
   真实分数；M1 只做了 VMD 级往返，没有过前馈链路；
3. 再开 M2 第一刀：**选择性清理**（Hampel/中值只修 flagged 帧）先把 T5 的 jerk（1e4 量级，
   源自身 10–14°/帧）拉下来；然后才是结构后馈；
4. 门禁一律用向量分量 + per-bone/P99/Pareto，**不用 `summary_S`**；
5. **不要直接跑 `demo.py`**；用 `scripts/run_gvhmr.py`；**T3/T4 不做单人验收**；
   合成 GT 用固定全身镜头；T1 的 `slow_factor=0.25`；根位移不写绝对 transl。

## 开源许可（当前决定）

- 代码：Apache-2.0；
- 默认 research profile：GVHMR + SMPL-X（非商业研究依赖），README 不承诺商用；
- 权重、SMPL-X、三套 PMX、R1-R3 VMD、T1-T5 视频都**不入库**；用户自备；
- 不复制 reze-mipo（GPL）代码；mmd2gltf（MIT）可移植，保留声明；
- 开源发布前准备 CC0/CC-BY demo 模型 + 自生成 demo 视频 + `THIRD_PARTY_NOTICES.md` / `MODEL_LICENSES.md`。

## 缺失/待准备（详见 docs/开发前准备清单.md）

- **M0-M2：无阻塞缺失**；仓库、权重、body models、测试视频、Blender、ffmpeg 都在。
- **M3（手/脸）**：需选一个手部专家（HaMeR / WiLoR / SMPLer-X）和一个脸部方案（MediaPipe / anime face）；到时下载即可，不影响 M0-M2。
- **可选**：DA3（运动镜头）、RAFT/Sea-RAFT（光流；OpenCV DIS 可兜底）、人体解析（部件掩码）、AMASS（先验泛化）、CC0 demo 模型（开源发布）。
