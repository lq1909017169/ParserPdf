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
    print(f"\n========== STARTING OCR FOR: {os.path.basename(image_path)} ==========")
    api_key = random_genai()

    # 打印关键配置信息
    print(f"[DEBUG] Using Model Name: '{genai_name}'")
    print(f"[DEBUG] API Key (last 4 chars): ...{api_key[-4:]}")

    if not os.path.exists(image_path):
        print(f"[DEBUG] ERROR: Image path does not exist: {image_path}")
        return "Error: Image file not found."

    try:
        genai.configure(api_key=api_key)

        # 1. 加载并检查图片
        try:
            img = PIL.Image.open(image_path)
            print(f"[DEBUG] Image Loaded Successfully. Size: {img.size}, Format: {img.format}, Mode: {img.mode}")
        except Exception as e:
            print(f"[DEBUG] PIL Error loading image: {e}")
            return "Error: Invalid image file."

        # 2. 准备 Prompt
        system_instruction = (
            f"你是一个 OCR 专家。识别图中的{lang}内容并转为 Markdown。"
            f"规则 1：如果遇到目录中的引导点（如 'Introduction ...... 5'），**绝对不要**输出中间的点号，直接输出 'Introduction 5' 或使用空格分隔。"
            f"规则 2：不要输出任何 Markdown 代码块标记（如 ```markdown），直接输出内容。"
            f"规则 3：如果遇到数学公式，使用 LaTeX 格式。"
            f"规则 4：若遇到乱码或无法识别，直接跳过。"
        )

        model = genai.GenerativeModel(
            model_name=genai_name,
            generation_config=create_generation_config(),
            system_instruction=system_instruction,
            safety_settings=get_safety_settings()
        )

        prompt = "识别图片内容。注意：忽略目录中的所有点号（......）。"

        print("[DEBUG] Sending request to Gemini...")
        start_time = time.time()

        # 3. 发送请求
        response = model.generate_content([prompt, img])

        end_time = time.time()
        print(f"[DEBUG] Request finished in {end_time - start_time:.2f} seconds.")

        # 4. 深度分析响应结果
        if not response.candidates:
            print("[DEBUG] CRITICAL: Response has NO candidates. Usually means complete blockage or server error.")
            # 尝试打印 feedback
            if hasattr(response, 'prompt_feedback'):
                print(f"[DEBUG] Prompt Feedback: {response.prompt_feedback}")
            return "Error: No candidates returned."

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason

        # Finish Reason 字典映射
        reason_map = {
            0: "UNKNOWN",
            1: "STOP (Normal)",
            2: "MAX_TOKENS (Token limit reached - likely loop)",
            3: "SAFETY (Content blocked)",
            4: "RECITATION (Copyright/Memorization)",
            5: "OTHER"
        }

        print(f"[DEBUG] Finish Reason Code: {finish_reason} -> {reason_map.get(finish_reason, 'Unknown')}")

        if finish_reason == 3:  # SAFETY
            print(f"[DEBUG] Safety Ratings: {candidate.safety_ratings}")
            return "Error: Blocked by safety filters."

        # 5. 安全提取文本
        if candidate.content and candidate.content.parts:
            text_part = candidate.content.parts[0].text
            print(f"[DEBUG] Text extracted successfully. Length: {len(text_part)} chars.")
            print(f"[DEBUG] First 100 chars preview: {text_part[:100].replace(chr(10), ' ')}...")
            return text_part
        else:
            print("[DEBUG] CRITICAL: Candidate exists but has NO text parts.")
            return "Error: Empty content parts."

    except Exception:
        print("\n[DEBUG] EXCEPTION OCCURRED:")
        print(traceback.format_exc())
        return 'Please parse again'


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
