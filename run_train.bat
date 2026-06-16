@echo off
echo Building weighted YOLOv8 dataset...
"C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" "%~dp0build_yolov8_dataset.py"
if errorlevel 1 pause & exit /b %errorlevel%

echo Training YOLOv8 segmentation model...
"C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" "%~dp0train.py" %*
pause
