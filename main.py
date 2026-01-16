import datetime
import os
import sys
import time
import traceback

from utils.pdf_processor import convert_pdf_to_images, pdf_balance
from utils.ocr_engine import img_to_md
from utils.file_utils import save_to_json
import boto3
from pymysql import Connect
from concurrent.futures import ThreadPoolExecutor


from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# def process_single_pdf(pdf_path, lang):
#     if not os.path.exists(pdf_path):
#         print(f"错误: 文件不存在 -> {pdf_path}")
#         return
#
#     # 1. PDF 转 图片
#     # 返回：所有图片路径列表，和图片所在的文件夹路径
#     img_paths, output_dir = convert_pdf_to_images(pdf_path)
#
#     # 准备 JSON 数据结构
#     # pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     result_data = {
#         "filename": os.path.basename(pdf_path),
#         "total_pages": len(img_paths),
#         "pages": []
#     }
#     print(result_data)
#
#     print(f"\n🚀 开始 OCR 识别 ({len(img_paths)} 页，语言{lang})...")
#
#     # 2. 遍历图片进行 OCR
#     for idx, img_path in enumerate(img_paths):
#         page_num = idx + 1
#         print(f"[{page_num}/{len(img_paths)}] 处理中...")
#
#         # 调用 Gemini
#         md_content = img_to_md(img_path, lang)
#
#         # 拼装单页数据
#         page_data = {
#             "page": page_num,
#             "image_path": img_path,
#             "content": md_content
#         }
#         result_data["pages"].append(page_data)
#
#     # 3. 保存为 JSON
#     # JSON 将保存在 output/文件名/文件名.json
#     save_json_path = str(pdf_path)[:-4].replace('upload', 'result')
#
#     json_output_path = os.path.join(save_json_path, f"pdf_new.json")
#     print(json_output_path)
#     save_to_json(result_data, json_output_path)
#
#     print("\n✨ 全部完成！")
#     return output_dir, len(img_paths)


MAX_WORKERS = 5


def process_page_wrapper(args):
    """
    包装函数，用于在线程池中运行。
    接收一个元组参数 (索引, 图片路径, 语言, 总页数)
    """
    idx, img_path, lang, total_pages = args
    page_num = idx + 1

    print(f"⚡ [线程启动] 第 {page_num}/{total_pages} 页开始处理...")

    # 调用核心 OCR 函数
    # 注意：img_to_md 函数内部已经包含了重试机制，这里直接调用即可
    md_content = img_to_md(img_path, lang)

    print(f"✅ [线程完成] 第 {page_num}/{total_pages} 页处理完毕")

    # 返回结构化的单页数据
    return {
        "page": page_num,
        "image_path": img_path,
        "content": md_content
    }


def process_single_pdf(pdf_path, lang):
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 -> {pdf_path}")
        return

    # 1. PDF 转 图片
    # (假设 convert_pdf_to_images 已经在你的代码上下文中定义好了)
    try:
        img_paths, output_dir = convert_pdf_to_images(pdf_path)
    except Exception as e:
        print(f"PDF 转图片失败: {e}")
        return

    # 准备 JSON 数据结构
    result_data = {
        "filename": os.path.basename(pdf_path),
        "total_pages": len(img_paths),
        "pages": []  # 这里的数据稍后填充
    }

    print(result_data)
    print(f"\n🚀 开始多线程 OCR 识别 ({len(img_paths)} 页，并发数: {MAX_WORKERS})...")

    # 2. 准备多线程任务参数
    # 将需要的参数打包成元组列表
    tasks = [(idx, img_path, lang, len(img_paths)) for idx, img_path in enumerate(img_paths)]

    # 3. 执行多线程池
    # 使用 map 方法可以保证返回的结果顺序与 tasks 的顺序一致（即按页码排序）
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map 会阻塞主线程，直到所有任务完成，并返回一个迭代器
        results = list(executor.map(process_page_wrapper, tasks))

    # 将有序的结果赋值给 result_data
    result_data["pages"] = results

    # 4. 保存为 JSON
    save_json_path = str(pdf_path)[:-4].replace('upload', 'result')

    # 确保目录存在
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)

    json_output_path = os.path.join(save_json_path, f"pdf_new.json")
    print(f"\n💾 保存结果到: {json_output_path}")

    # (假设 save_to_json 已经在你的代码上下文中定义好了)
    save_to_json(result_data, json_output_path)

    print("\n✨ 全部完成！")
    return output_dir, len(img_paths)


if __name__ == '__main__':
    region_name = os.getenv("REGION", "")
    aws_access_key_id = os.getenv("aws_access_key_id", "")
    aws_secret_access_key = os.getenv("aws_secret_access_key", "")
    QUEUE_URL = os.getenv("QUEUE_URL", "")

    sqs = boto3.client('sqs', region_name=region_name,
                       aws_access_key_id=aws_access_key_id,
                       aws_secret_access_key=aws_secret_access_key)
    # s3 = boto3.client('s3', region_name=Parameter.Parameter.REGION)

    # 加载模型
    print('load model')

    while True:
        print('While loop ---->')
        time.sleep(5)
        response = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=1,
                                       WaitTimeSeconds=20)
        if 'Messages' in response:
            # snapshot1 = tracemalloc.take_snapshot()
            message = response['Messages'][0]
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=message['ReceiptHandle'])
            print(message)
            file_map = eval(message['Body'])
            try:
                file_id = file_map['file_id']
                task_id = file_map['task_id']
                # layout = file_map['layout']
                pdf_path = os.path.join('/usr/local/src/s3mnt/new_backend/upload', task_id, f"{file_id}.pdf")
                user_id = file_map['user_id']
                parameter = file_map['parameter']
                lang = file_map['lang']

                # 数据库配置
                setting_sql = {'host': os.getenv("host", ""), 'port': int(os.getenv("port", "")),
                               'user': os.getenv("user", ""),
                               'password': os.getenv("password", ""), 'database': os.getenv("database", "")}

                with Connect(**setting_sql) as conn:
                    cursor = conn.cursor()
                    sql = f'UPDATE file_result SET ' \
                          f'queue_time="{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}" ' \
                          f'WHERE file_id="{file_id}" and task_id="{task_id}"'
                    print(sql)
                    cursor.execute(sql)
                    conn.commit()

                # 解析pdf
                image_path, pdf_page_num = process_single_pdf(pdf_path=pdf_path, lang=lang)

                # 计费
                pdf_balance(image_path, task_id, file_id, user_id, pdf_page_num, setting_sql)

                print('扣费成功')

            except:
                print(traceback.format_exc())

