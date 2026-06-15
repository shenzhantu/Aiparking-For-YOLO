"""
清理之前自动生成的矩形格式 JSON 文件（保留手动多边形标注）
特征：shape_type 为 "rectangle"，或 shapes 为空列表
"""
import json
import os
import glob

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

json_files = glob.glob(os.path.join(TARGET_DIR, "*.json"))
print(f"扫描 {len(json_files)} 个 JSON 文件...")

auto_count = 0
manual_count = 0
error_count = 0

for jf in json_files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        is_auto = False
        if len(shapes) == 0:
            # 无标注 → 之前 predict.py 生成的空文件
            is_auto = True
        elif any(s.get("shape_type") == "rectangle" for s in shapes):
            # 矩形标注 → 之前 predict.py 生成的错误格式
            is_auto = True

        if is_auto:
            os.remove(jf)
            auto_count += 1
        else:
            manual_count += 1
    except Exception as e:
        print(f"  错误 {os.path.basename(jf)}: {e}")
        error_count += 1

print(f"\n清理完成:")
print(f"  删除自动生成: {auto_count} 个")
print(f"  保留手动标注: {manual_count} 个")
print(f"  错误: {error_count} 个")
print(f"\n现在可以运行 predict.py 重新生成多边形标注")
