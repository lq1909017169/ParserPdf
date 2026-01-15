# import os
# import time
# import traceback
# import google.generativeai as genai
#
# import traceback
from PIL import Image
import io
import random
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time
import mimetypes
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    FinishReason,
    HarmCategory,
    HarmBlockThreshold,
    GenerationConfig
)


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
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }


def load_image_part(image_path):
    """
    核心修复：
    不直接读取文件字节，而是用 PIL 打开，
    强制转换为 RGB 模式（防止 CMYK 报错），
    并重新保存为标准 JPEG 字节流。
    """
    try:
        # 1. 用 PIL 打开图片
        with Image.open(image_path) as img:
            # 2. 转换为 RGB (防止 PNG 透明通道或 CMYK 导致 400 错误)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. 保存到内存中的 BytesIO 对象
            img_byte_arr = io.BytesIO()
            # 强制指定格式为 JPEG，质量 95
            img.save(img_byte_arr, format='JPEG', quality=95)

            # 4. 获取字节数据
            image_data = img_byte_arr.getvalue()

        # 5. 返回 Vertex AI 需要的 Part 对象，明确指定 mime_type 为 image/jpeg
        return Part.from_data(data=image_data, mime_type="image/jpeg")

    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        raise


def img_to_md(image_path, lang="en"):
    try:
        KEY_PATH = '/usr/local/src/pypro/ParserPdf/utils/key_json/key.json'
        PROJECT_ID = "eyeweb-wb-ys"
        LOCATION = "us-central1"
        if os.path.exists(KEY_PATH):
            # 使用服务账号 JSON 文件进行认证
            credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            print(f"✅ Vertex AI initialized successfully: {PROJECT_ID} @ {LOCATION}")
        else:
            print(f"❌ Error: 找不到密钥文件 '{KEY_PATH}'。请去 GCP 后台下载 JSON 密钥。")
            # 如果你已经在环境变量里配好了，可以尝试无参初始化
            # vertexai.init(project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"❌ Vertex AI init failed: {e}")

    print(f"\n========== PROCESSING: {os.path.basename(image_path)} ==========")

    if not os.path.exists(image_path):
        return "Error: Image file not found."

    max_retries = 3

    for attempt in range(max_retries):
        try:
            image_part = load_image_part(image_path)

            # === 动态策略 ===
            temp = 0.1
            prompt_text = "识别图片内容并转换为 Markdown 格式。"
            sys_instruction = (
                f"你是一个专业的 OCR 工具。请识别图中的{lang}文字。"
                f"遇到目录页的引导点（......），**必须忽略**，直接输出文字和页码。"
                f"如果是数学公式，请使用 LaTeX。"
            )

            # 策略 1: 针对目录死循环 (Reason 2)
            if attempt == 1:
                print(f"[Warning] Retrying (Strict Mode)...")
                temp = 0.0
                sys_instruction += " **严重警告：绝对禁止输出任何连续的点号(......)！**"
                prompt_text = "提取文字。忽略所有装饰符号。"

            # 策略 2: 针对参考文献版权拦截 (Reason 4)
            if attempt == 2:
                print(f"[Warning] Retrying (Anti-Recitation Mode)...")
                temp = 0.4
                # 强行要求加粗标题，破坏文本指纹
                sys_instruction = (
                    f"You are a bibliographic data assistant. "
                    f"Extract references from the image into Markdown. "
                    f"**IMPORTANT RULE**: You MUST **bold** the title of every paper/section."
                    f"Example: Author Name. **Paper Title**. Publisher."
                )
                prompt_text = "Extract content. Remember to **bold** titles."

            model = GenerativeModel(genai_name, system_instruction=[sys_instruction])

            response = model.generate_content(
                [prompt_text, image_part],
                generation_config=GenerationConfig(
                    temperature=temp,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
                safety_settings=get_safety_settings()
            )

            if not response.candidates:
                if attempt < max_retries - 1: continue
                return "Error: No candidates."

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # === 成功 ===
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                if finish_reason == FinishReason.MAX_TOKENS:
                    text = text.rstrip('. ')
                return text

            # === 失败处理 ===
            print(f"[Debug] Attempt {attempt + 1} Failed. Reason Code: {finish_reason}")

            # Reason 4: RECITATION (版权) -> 下一次循环触发 Attempt 2
            if finish_reason == FinishReason.RECITATION:
                time.sleep(1)
                continue

            # Reason 2: MAX_TOKENS (死循环) -> 下一次循环触发 Attempt 1
            if finish_reason == FinishReason.MAX_TOKENS:
                time.sleep(1)
                continue

            if attempt < max_retries - 1:
                time.sleep(1)
                continue

            return f"Error: Blocked with reason {finish_reason}"

        except Exception as e:
            print(f"[Exception] {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return 'Please parse again'

    return "Error: Failed after retries."


# def get_safety_settings():
#     return {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
#
#
# def img_to_md(image_path, lang="en"):
#     # ... 前面的初始化代码不变 ...
#     api_key = random_genai()
#     max_retries = 5
#
#     for attempt in range(max_retries):
#         try:
#             genai.configure(api_key=api_key)
#             img = PIL.Image.open(image_path)
#
#             # === 默认配置 ===
#             temp = 0.1
#             sys_instruction = f"你是一个 OCR 工具。请识别图中的{lang}文字并转为 Markdown。"
#             prompt_text = "识别图片内容。"
#
#             # === 【关键策略修改】 ===
#
#             # 第一次重试 (Attempt 1): 严厉模式 (针对目录死循环)
#             if attempt == 1:
#                 temp = 0.0
#                 sys_instruction += " **忽略所有连续的点号(......)**。"
#
#             # 第二次重试 (Attempt 2): 【防版权模式 - 针对参考文献】
#             # 如果是参考文献页，强制要求改变格式，破坏指纹匹配
#             if attempt == 2:
#                 print(f"[Warning] 启用参考文献特殊模式 (Anti-Recitation Mode)...")
#                 temp = 0.3  # 稍微增加随机性
#
#                 # 核心 Trick：告诉模型这是一个“格式化任务”而不是“读取任务”
#                 sys_instruction = (
#                     f"You are a bibliographic data assistant. "
#                     f"The image contains a list of academic references. "
#                     f"Your task is to extract them into a Markdown list. "
#                     f"**IMPORTANT RULE**: To ensure readability, you MUST **bold** the title of every paper."
#                     f"For example: Author Name. **Paper Title**. Publisher."
#                 )
#                 prompt_text = "Extract references. Remember to **bold** the titles to differentiate them from authors."
#
#             model = genai.GenerativeModel(
#                 model_name=genai_name,
#                 generation_config={
#                     "temperature": temp,
#                     "top_p": 0.95,
#                     "max_output_tokens": 8192,
#                 },
#                 system_instruction=sys_instruction,
#                 safety_settings=get_safety_settings()
#             )
#
#             response = model.generate_content([prompt_text, img])
#
#             if not response.candidates:
#                 if attempt < max_retries - 1: continue
#                 return "Error: No candidates."
#
#             candidate = response.candidates[0]
#             finish_reason = candidate.finish_reason
#
#             # 成功获取
#             if candidate.content and candidate.content.parts:
#                 text = candidate.content.parts[0].text
#                 return text
#
#             # 失败处理
#             print(f"[Debug] Attempt {attempt + 1} Failed. Reason: {finish_reason}")
#
#             # 如果是版权拦截 (4)，让循环继续，自然会进入 attempt=2 的逻辑
#             if finish_reason == 4 or finish_reason == 3:
#                 time.sleep(1)
#                 continue
#
#             # 如果是死循环 (2)，也继续
#             if finish_reason == 2:
#                 time.sleep(1)
#                 continue
#
#         except Exception as e:
#             print(f"[Exception] {e}")
#             time.sleep(1)
#             continue
#
#     return "Error: Failed to parse page 19."


# 下面为旧版
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
