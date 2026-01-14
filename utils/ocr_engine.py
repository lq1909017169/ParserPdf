import os
import time
import traceback
import google.generativeai as genai
import random
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PIL.Image

# 加载环境变量
load_dotenv()

# 准备 API Keys
api_keys_str = os.getenv("API_KEYS", "")
genai_name = os.getenv("GENAI_NAME", "")
API_KEYS = [
    k.strip().replace("'", "").replace('"', "")  # 核心修改：强制替换掉单引号和双引号
    for k in api_keys_str.split(',')
    if k.strip()
]


def random_genai():
    """随机获取一个 API Key"""
    try:
        if not API_KEYS:
            raise ValueError("API key list is empty")
        api_key_index = random.randint(0, len(API_KEYS) - 1)
        api_key = API_KEYS[api_key_index]
        return api_key
    except Exception as e:
        print(f"Error selecting API key: {e}")
        raise


def create_generation_config():
    return {
        "temperature": 0.1,  # 保持低温度以确保 OCR 准确性
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }


def get_safety_settings():
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


def img_to_md(image_path, lang="en"):
    api_key = random_genai()

    if not os.path.exists(image_path):
        return "Error: Image file not found."

    # 定义重试次数
    max_retries = 2

    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            img = PIL.Image.open(image_path)

            # --- 动态调整配置 ---
            # 如果是重试（attempt > 0），说明第一次可能陷入死循环了
            # 我们使用更极端的配置：温度降为 0（绝对冷静），Prompt 极简
            current_temp = 0.1 if attempt == 0 else 0.0

            generation_config = {
                "temperature": current_temp,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            # 基础 System Instruction
            base_instruction = (
                f"你是一个 OCR 工具。识别图中的{lang}文字。"
                f"遇到目录页的引导点（......），**必须忽略**，直接输出文字和页码。"
                f"禁止输出连续的点号。"
            )

            # 如果是重试，加强语气
            if attempt > 0:
                print(f"[Warning] Retrying {os.path.basename(image_path)} with STRICT mode...")
                system_instruction = base_instruction + " **严重警告：不要输出任何点号！不要死循环！**"
                prompt = "提取文字。忽略所有符号点。"
            else:
                system_instruction = base_instruction
                prompt = "识别图片内容转 Markdown。"

            model = genai.GenerativeModel(
                model_name=genai_name,
                generation_config=generation_config,
                system_instruction=system_instruction,
                safety_settings=get_safety_settings()
            )

            # 发送请求
            response = model.generate_content([prompt, img])

            # --- 结果检查逻辑 ---
            if not response.candidates:
                if attempt < max_retries - 1: continue  # 重试
                return "Error: No candidates."

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # 情况 A: 成功拿到文本
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                # 如果是因为 Token 耗尽截断，但在有文本的情况下，通常是可以用的
                # 我们可以把末尾可能存在的连续点号切掉
                if finish_reason == 2:
                    print(f"[Info] Page truncated but text salvaged.")
                    # 简单的清洗，去掉末尾可能存在的 '....'
                    text = text.rstrip('. ')
                return text

            # 情况 B: Token 耗尽且没有文本 (死循环最坏情况)
            elif finish_reason == 2:
                print(f"[Debug] Hit Max Tokens loop on attempt {attempt + 1}.")
                # 这种情况下必须重试，因为没有内容返回
                if attempt < max_retries - 1:
                    time.sleep(1)  # 歇一秒再试
                    continue
                return "Error: OCR failed due to infinite loop (Max Tokens)."

            # 情况 C: 安全拦截或其他
            else:
                print(f"[Debug] Blocked or Empty. Reason: {finish_reason}")
                if attempt < max_retries - 1: continue
                return "Error: Content blocked."

        except Exception as e:
            print(f"[Exception] {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return 'Please parse again'

    return "Error: Failed after retries."

# def create_generation_config():
#     return {
#         "temperature": 0.1,
#         "top_p": 0.95,
#         "top_k": 40,
#         "max_output_tokens": 100000,
#         "response_mime_type": "text/plain",
#     }
#
#
# def get_safety_settings():
#     return {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
#
#
# def upload_to_gemini(api_key, path, mime_type=None):
#     """上传文件到 Gemini"""
#     genai.configure(api_key=api_key)
#     file = genai.upload_file(path, mime_type=mime_type)
#     print(f"Uploaded file '{file.display_name}' as: {file.uri}")
#     return file
#
#
# def wait_for_files_active(files):
#     """等待文件处理完毕（Gemini API 对于某些大文件需要处理时间）"""
#     print("Waiting for file processing...")
#     for name in (file.name for file in files):
#         file = genai.get_file(name)
#         while file.state.name == "PROCESSING":
#             print(".", end="", flush=True)
#             time.sleep(2)
#             file = genai.get_file(name)
#         if file.state.name != "ACTIVE":
#             raise Exception(f"File {file.name} failed to process")
#     print("...all files ready")
#
#
# def img_to_md(image_path, lang):
#     api_key = random_genai()
#     print('Using api_key ending in:', api_key[-4:])  # 打印Key的后四位用于调试，不打印全貌
#
#     if not api_key:
#         return "Error: No API key available."
#
#     try:
#         # 配置 API
#         genai.configure(api_key=api_key)
#
#         # 1. 上传图片
#         # 注意：如果图片非常小，其实可以直接转 bytes 发送，但这里保留你的 upload_file 逻辑
#         gemini_image = upload_to_gemini(api_key, image_path, mime_type="image/png")
#
#         # 确保文件已就绪（虽然图片通常很快，但加上这个逻辑更稳健）
#         wait_for_files_active([gemini_image])
#
#         # 2. 设置 System Instruction
#         # 这里的指令非常关键，要求它强制输出 Markdown，并处理公式
#         system_instruction = (
#             f"你是一个专业的 OCR 助手。请识别图片中的所有内容，图中语言为:{lang},请以{lang}语言返回并将其转换为标准的 "
#             f"Markdown 格式返回。如果是表格，请使用 Markdown 表格语法。如果是数学公式，请使用 LaTeX 格式（行内公式用 $ 包裹，独占一行用 $$ 包裹）。"
#             f"不要包含任何开场白或结束语，只返回转换后的内容。"
#         )
#
#         # 3. 创建模型
#         generation_config = create_generation_config()
#         model = genai.GenerativeModel(
#             model_name=genai_name,
#             generation_config=generation_config,
#             system_instruction=system_instruction,
#             safety_settings=get_safety_settings()
#         )
#
#         prompt = "请将这张图片的内容精准转换为 Markdown 格式。"
#
#         response = model.generate_content([gemini_image, prompt])
#
#         try:
#             return response.text
#         except:
#             print(response.candidates)
#             print(traceback.format_exc())
#             print(f"DEBUG: Finish Reason: {response.candidates[0].finish_reason}")
#             # 强行获取截断内容
#             if response.candidates and response.candidates[0].content.parts:
#                 return response.candidates[0].content.parts[0].text
#             return ""
#
#     except Exception:
#         print(traceback.format_exc())
#         return 'Please parse again'


if __name__ == '__main__':
    # 确保文件存在再运行
    img_path = 'E:\各类文件\图片\Snipaste_2026-01-14_11-03-10.png'
    lang = 'zh'
    if os.path.exists(img_path):
        result = img_to_md(img_path, lang)
        print("-" * 20 + " RESULT " + "-" * 20)
        print(result)
    else:
        print(f"文件不存在: {img_path}")
