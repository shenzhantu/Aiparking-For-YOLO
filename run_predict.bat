@echo off
echo Running YOLOv8 auto-label on images（5）...
"C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" "%~dp0predict.py" %*
pause
