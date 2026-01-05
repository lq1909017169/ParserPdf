import os
import fitz  # PyMuPDF
from PIL import Image
from .file_utils import ensure_directory_exists


def convert_pdf_to_images(pdf_path):
    """
    将 PDF 的每一页转换为图片。
    :param pdf_path: PDF 文件路径
    :return: (img_path_list, output_dir) 图片路径列表和图片所在文件夹
    """
    # 获取文件名（不带后缀），例如 'book'
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(str(pdf_path)[:-4], 'img').replace('upload', 'layout')

    img_path_list = []

    print('output_path创建img路径', output_path)
    ensure_directory_exists(output_path)

    print(f"📄 正在处理 PDF: {pdf_name} ...")

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(3, 3)

    for i, page in enumerate(doc):
        image_filename = f'{i + 1}.jpg'
        full_image_path = os.path.join(output_path, image_filename)

        # 将路径存入列表
        img_path_list.append(full_image_path)

        if os.path.exists(full_image_path):
            # 如果图片已存在，跳过生成，节省时间
            print(f"  - 跳过已存在图片: P{i + 1}")
            continue

        # 渲染页面为图像
        pix = page.get_pixmap(matrix=mat)

        # 使用 Pillow 保存 (PyMuPDF 的 pix 也可以直接 save，但转为 Pillow 对象更通用)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        img.save(full_image_path)
        print(f"  - 已生成图片: P{i + 1}")

    return img_path_list, output_path
