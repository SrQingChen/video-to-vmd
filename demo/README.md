# 发布 demo（清单，不含受限素材）

本目录**只放清单**，不放视频、PMX、VMD、权重或渲染帧。原因见
`MODEL_LICENSES.md` / `THIRD_PARTY_NOTICES.md`：测试视频、三套 PMX、
SMPL-X、GVHMR 权重与 M3.5 checkpoint 都不随仓库分发。

## 目标 demo

| 资产 | 要求 | 许可 | 是否入库 |
|---|---|---|---|
| demo 视频 | 单机位、固定相机、10–120 秒、1080×1920 或更高 | **CC0-1.0 或 CC-BY-4.0**（自拍/自生成优先） | 否 |
| demo 模型 | 标准 MMD PMX（23 身骨 + 手指 + morph） | **CC0-1.0 或 CC-BY-4.0**；CC-BY 需在 `demo/manifest.json` 署名 | 否 |
| 运行产物 | VMD / 2K before-after / QC summary | 由运行生成，按 demo 资产许可分发 | 否（`outputs/` git-ignored） |
| 静帧 | 2–4 张 2K 抽帧 | 同上 | 否 |

`demo/manifest.json` 是机器可读的清单模板；发布时把实际资产路径、作者、
来源 URL、许可证填进去（`assets` 数组），并确认所有条目都是 CC0/CC-BY。
**不要**把任何非商业研究依赖（GVHMR/SMPL-X/R1-R3/T1-T5/AIblender PMX）当作 demo 素材。

## 复现命令

```bash
# 1) 环境
python scripts/check_env.py --models si
python scripts/check_env.py --long   # 额外检查 chunking/ffmpeg 长视频路径

# 2) 长视频清理（固定机位；首次会下载/加载 GVHMR）
python scripts/run_vtv.py --video demo/assets/<demo>.mp4 \
  --out outputs/m5/demo --models si --feedback default \
  --chunk-seconds 10 --overlap-seconds 1 --render --render-boundary 0

# 3) 短片段（同一模型）2K before/after
python scripts/run_vtv.py --video demo/assets/<short>.mp4 \
  --out outputs/m5/demo_short --models si --feedback default \
  --no-chunk --render --frames 70:230

# 4) 重新打印 QC
python scripts/qc_report.py --run outputs/m5/demo
```

## 长 demo 视频怎么来

仓库不带测试视频；可以用同分辨率/同帧率的自有片段拼接（脚本会校验并记录）：

```bash
python scripts/make_long_video.py \
  --inputs demo/assets/clip_a.mp4,demo/assets/clip_b.mp4,demo/assets/clip_c.mp4 \
  --out demo/assets/demo_long_80s.mp4 --fps 30
```

输出旁会生成 `demo_long_80s.json`：源文件 sha256、逐段帧数、总时长。
拼接口是**内容硬切**，QC 的 `source_max_step_deg` 会把它和分段边界区分开。

## 发布前检查

- [ ] `demo/manifest.json` 的每个 `assets[]` 都有 `license`（CC0/CC-BY）与 `source`；
- [ ] 仓库里没有 `*.mp4/*.pmx/*.vmd/*.pth/*.ckpt/*.blend`（`.gitignore` 已覆盖）；
- [ ] 没有本机绝对路径（`configs/pipeline.json` 的 Blender/ffmpeg/场景路径需改成占位符或用
      `VTV_BLENDER/VTV_FFMPEG/VTV_BLEND` 环境变量）；
- [ ] `THIRD_PARTY_NOTICES.md` / `MODEL_LICENSES.md` 与素材清单一致；
- [ ] 复现命令在干净 venv 中跑通（见 `docs/M5_使用指南.md`）。
