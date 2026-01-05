import os
import json


def ensure_directory_exists(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_json(data, output_path):
    """将字典保存为 JSON 文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ JSON 结果已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")