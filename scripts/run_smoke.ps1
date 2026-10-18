# 在 dl1 环境中运行一组不会下载真实数据集的快速验证。
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = "C:\Users\3229629598\miniconda3\envs\dl1\python.exe"

& $Python -m unittest discover -s tests -v
& $Python train.py --config configs/smoke_test.yaml --dry-run

