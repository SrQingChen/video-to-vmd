@echo off
rem ============================================================
rem 状态（2026-10-25）：body_models 已安置，GVHMR junction 已建；
rem 本脚本仅在重装/迁移/修复 junction 时使用。
rem SMPL-X 安置脚本：把下载的 SMPL-X v1.1 文件放好并接入 GVHMR
rem 用法：先把解压出的 npz/pkl 放到 body_models\smplx\ 与 body_models\smpl\
rem       （SMPL-X 的 npz 进 smplx；SMPL 的 pkl 进 smpl，如果以后下载）
rem       然后双击本脚本运行一次
rem ============================================================
chcp 936 >nul
cd /d "%~dp0"

if not exist "body_models\smplx" (echo [缺] body_models\smplx 不存在 & pause & exit /b 1)

rem GVHMR 期望: third_party\GVHMR\inputs\checkpoints\body_models\smplx\*.npz
if not exist "third_party\GVHMR\inputs\checkpoints\body_models" mkdir "third_party\GVHMR\inputs\checkpoints\body_models"
if not exist "third_party\GVHMR\inputs\checkpoints\body_models\smplx" mklink /J "third_party\GVHMR\inputs\checkpoints\body_models\smplx" "%~dp0body_models\smplx"
if exist "body_models\smpl" (if not exist "third_party\GVHMR\inputs\checkpoints\body_models\smpl" mklink /J "third_party\GVHMR\inputs\checkpoints\body_models\smpl" "%~dp0body_models\smpl")

echo.
echo [完成] 当前 smplx 目录内容:
dir /b "body_models\smplx"
echo.
echo GVHMR 侧检查:
dir /b "third_party\GVHMR\inputs\checkpoints\body_models\smplx" 2>nul
pause
