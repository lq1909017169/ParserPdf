import os
import fitz  # PyMuPDF
from PIL import Image
from .file_utils import ensure_directory_exists
import datetime
import pandas as pd
from dotenv import load_dotenv
from pymysql import Connect

# 加载环境变量
load_dotenv()


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


def pdf_balance(image_path, task_id, file_id, user_id, pdf_page_num, setting_sql):

    image_path_one = os.path.join(image_path, '1.jpg')
    success_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = f"/usr/local/src/s3mnt/new_backend/result/{task_id}/{file_id}/pdf_middle.json"
    with Connect(**setting_sql) as conn:
        cursor = conn.cursor()
        sql = f'UPDATE file_result SET success_time="{success_time}", ' \
              f'parser_time="{success_time}", result_path="{result_path}", ' \
              f'image_path="{image_path_one}" WHERE file_id="{file_id}" and task_id="{task_id}"'
        cursor.execute(sql)
        conn.commit()

        # 扣费详情
        se_sql = f'SELECT balance FROM user_balance where user_id="{user_id}"'
        balance = pd.read_sql(sql=se_sql, con=conn).iloc[-1]['balance']

        price = int(os.getenv("price", ""))

        residue_balance = int(balance) - (pdf_page_num * price)

        insert_sql = f'INSERT INTO user_balance(user_id, balance, change_amount, c_time, ' \
                     f'change_project, file_id) VALUES ' \
                     f'("{user_id}", {residue_balance}, {-pdf_page_num * price}, ' \
                     f'"{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", ' \
                     f'"pdfParser", "{file_id}");'
        print(insert_sql)
        cursor.execute(insert_sql)
        conn.commit()
