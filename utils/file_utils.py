import os
import json


def ensure_directory_exists(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_json(data, output_path):
    """将字典保存为 JSON 文件，自动创建不存在的目录"""
    try:
        # --- 新增步骤：获取目录并创建 ---
        # 1. 获取文件所在的文件夹路径
        directory = os.path.dirname(output_path)

        # 2. 如果目录不存在，使用 makedirs 递归创建 (exist_ok=True 防止目录已存在报错)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📂 已自动创建目录: {directory}")
        # ---------------------------

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ JSON 结果已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")
        # 打印详细堆栈以便调试（可选）
        # import traceback
        # traceback.print_exc()