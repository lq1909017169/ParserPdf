import os
import sys
from utils.pdf_processor import convert_pdf_to_images
from utils.ocr_engine import img_to_md
from utils.file_utils import save_to_json


def process_single_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 -> {pdf_path}")
        return

    # 1. PDF 转 图片
    # 返回：所有图片路径列表，和图片所在的文件夹路径
    img_paths, output_dir = convert_pdf_to_images(pdf_path)

    # 准备 JSON 数据结构
    # pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    result_data = {
        "filename": os.path.basename(pdf_path),
        "total_pages": len(img_paths),
        "pages": []
    }
    print(result_data)

    print(f"\n🚀 开始 OCR 识别 ({len(img_paths)} 页)...")

    # 2. 遍历图片进行 OCR
    for idx, img_path in enumerate(img_paths):
        page_num = idx + 1
        print(f"[{page_num}/{len(img_paths)}] 处理中...")

        # 调用 Gemini
        md_content = img_to_md(img_path)

        # 拼装单页数据
        page_data = {
            "page": page_num,
            "image_path": img_path,
            "content": md_content
        }
        result_data["pages"].append(page_data)

    # 3. 保存为 JSON
    # JSON 将保存在 output/文件名/文件名.json
    save_json_path = str(pdf_path)[:-4].replace('upload', 'result')

    json_output_path = os.path.join(save_json_path, f"pdf_new.json")
    print(json_output_path)
    save_to_json(result_data, json_output_path)

    print("\n✨ 全部完成！")


if __name__ == '__main__':
    # 默认读取 input 文件夹下的 example.pdf，或者通过命令行传参
    target_pdf = '/usr/local/src/s3mnt/new_backend/upload/3b5c822b955a48deb83695ada1399252/24474f7d_404460_e62897a8.pdf'

    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
    process_single_pdf(target_pdf)
