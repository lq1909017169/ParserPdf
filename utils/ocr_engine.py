# import os
# import time
# import traceback
# import google.generativeai as genai
#
# import traceback
# from PIL import Image
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
    GenerationConfig,
    Image
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


# 你的 JSON 密钥路径
KEY_PATH = "/usr/local/src/pypro/ParserPdf/utils/key_json/key.json"

# 你的项目 ID
PROJECT_ID = "eyeweb-wb-ys"

# 【关键修改】Gemini 3 Preview 通常需要 global 区域
# LOCATION = "global"
LOCATION = "us-central1"

# 使用你验证成功的模型
MODEL_NAME = "gemini-3-pro-preview"

# ================= 初始化 =================
try:
    print(f"🔄 Initializing Vertex AI ({LOCATION})...")
    if os.path.exists(KEY_PATH):
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        print(f"✅ Vertex AI initialized using {MODEL_NAME}")
    else:
        print(f"⚠️ Key file missing at {KEY_PATH}")
except Exception as e:
    print(f"❌ Init failed: {e}")


# =========================================

def get_safety_settings():
    """放宽安全限制"""
    return {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }


def img_to_md(image_path, lang="en"):
    """
    优化后的 OCR 函数：
    1. 使用 Gemini 3 Pro Preview
    2. 使用 Vertex AI Image 类加载
    3. 包含针对目录页和版权页的自动修复逻辑
    """
    # print(f"\n========== PROCESSING: {os.path.basename(image_path)} ==========")

    if not os.path.exists(image_path):
        return "Error: Image file not found."

    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 1. 使用 SDK 原生方式加载图片 (代码更简洁)
            img = Image.load_from_file(image_path)

            # 2. 动态 Prompt 策略 (应对死循环和版权拦截)

            # --- Attempt 0: 正常模式 ---
            prompt_parts = [
                f"你是一个专业的 OCR 工具。请识别图中的{lang}文字并转换为 Markdown。",
                "如果是数学公式，请严格使用 LaTeX 格式（如 $$...$$）。",
                "遇到目录页的引导点（......），**必须忽略**，直接输出文字和页码。",
                "如果图片中没有任何元素，返回""即可",
                img  # 图片对象直接放入列表
            ]

            # --- Attempt 1: 严格模式 (针对目录页死循环) ---
            if attempt == 1:
                print(f"[Warning] Retrying {os.path.basename(image_path)} (Strict Mode)...")
                prompt_parts = [
                    "提取文字。**严重警告：绝对禁止输出任何连续的点号(......)！遇到请直接删除！**",
                    "忽略所有装饰性符号，只保留文本和数字。",
                    img
                ]

            # --- Attempt 2: 防版权模式 (针对参考文献页) ---
            if attempt == 2:
                print(f"[Warning] Retrying {os.path.basename(image_path)} (Anti-Recitation Mode)...")
                prompt_parts = [
                    "You are a bibliographic data assistant.",
                    "Extract references from the image into Markdown.",
                    "**IMPORTANT RULE**: You MUST **bold** the title of every paper/section to create a structured dataset.",
                    "Example: Author. **Paper Title**. Year.",
                    img
                ]

            # 3. 加载模型
            model = GenerativeModel(MODEL_NAME)

            # 4. 发送请求
            # 注意：Gemini 3 通常不需要 System Instruction，直接写在 Prompt 里效果更好
            response = model.generate_content(
                prompt_parts,
                generation_config=GenerationConfig(
                    # 重试时降低温度，增加确定性
                    temperature=0.1 if attempt < 2 else 0.4,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
                safety_settings=get_safety_settings()
            )

            # 5. 结果校验
            if not response.candidates:
                if attempt < max_retries - 1: continue
                return "Error: No candidates."

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # === 成功获取文本 ===
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text

                # 如果是因为 Token 耗尽 (可能还在画点)，尝试截断修复
                if finish_reason == FinishReason.MAX_TOKENS:
                    text = text.rstrip('. ')

                return text

            # === 失败处理 ===
            # print(f"[Debug] Attempt {attempt+1} Failed. Reason Code: {finish_reason}")

            # 遇到版权(RECITATION=4) 或 死循环(MAX_TOKENS=2) -> 继续循环
            if finish_reason in [FinishReason.RECITATION, FinishReason.MAX_TOKENS, FinishReason.SAFETY]:
                time.sleep(1)
                continue

            if attempt < max_retries - 1:
                time.sleep(1)
                continue

            return f"Error: Blocked with reason {finish_reason}"

        except Exception as e:
            # print(f"[Exception] {e}")
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
    pass
    # import vertexai
    # from google.oauth2 import service_account
    # from vertexai.generative_models import GenerativeModel, Image
    #
    # KEY_PATH = "/usr/local/src/pypro/ParserPdf/utils/key_json/key.json"
    # PROJECT_ID = "eyeweb-wb-ys"
    # LOCATION = "global"
    #
    # credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    # vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    #
    # image = Image.load_from_file(r"/usr/local/src/pypro/ParserPdf/img/gongshi.png")
    # vision_model = GenerativeModel("gemini-3-pro-preview")
    #
    # vision_model.generate_content(["你是一个专业的 OCR 工具，识别图片内容并转换为 Markdown。", image])