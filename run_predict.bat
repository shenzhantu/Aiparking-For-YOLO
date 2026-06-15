@echo off
echo 使用 yolov11 环境 (CUDA) 运行预测...
"C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" "%~dp0predict.py" %*
pause
