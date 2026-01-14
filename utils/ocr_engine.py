import os
import time
import traceback
import google.generativeai as genai
import random
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

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
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 100000,
        "response_mime_type": "text/plain",
    }


def get_safety_settings():
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


# def _extract_text_from_response(response):
#     """
#     专门用于从各种 finish_reason 中提取内容的辅助函数
#     """
#     try:
#         if not response.candidates:
#             return None, "No candidates returned"
#
#         candidate = response.candidates[0]
#         finish_reason = candidate.finish_reason
#
#         # 尝试提取文本 parts
#         text = ""
#         if candidate.content and candidate.content.parts:
#             text = candidate.content.parts[0].text
#
#         # 【关键修改2】如果是 STOP (1) 或者 MAX_TOKENS (2)，都视为有内容
#         # 注意：在某些 SDK 版本中 MAX_TOKENS 的枚举值可能是 2，SAFETY 是 3，具体视版本而定
#         # 我们这里直接判断是否有文本内容
#         if text and len(text.strip()) > 0:
#             if str(finish_reason) == "MAX_TOKENS" or finish_reason == 2:
#                 print("⚠️ Warning: Output triggered MAX_TOKENS (truncated), but returning partial content.")
#             return text, None
#
#         # 如果是因为安全原因被拦截且没有文本
#         return None, f"Blocked/Empty. Reason: {finish_reason}"
#
#     except Exception as e:
#         return None, f"Extraction Error: {str(e)}"
#
#
# def _do_gemini_ocr(api_key, image_path, lang, prompt_suffix=""):
#     genai.configure(api_key=api_key)
#     file = genai.upload_file(image_path, mime_type="image/png")
#
#     # 【关键修改3】Prompt 增加防死循环指令
#     system_instruction = (
#         f"你是一个OCR工具。识别图中所有文字，语种为{lang}。直接输出Markdown。不要重复输出相同的内容。如果遇到无法识别的乱码，直接跳过，不要循环输出符号。"
#     )
#
#     model = genai.GenerativeModel(
#         model_name=genai_name,
#         generation_config=create_generation_config(),
#         system_instruction=system_instruction,
#         safety_settings=get_safety_settings()
#     )
#
#     prompt = f"Extract text to Markdown.{prompt_suffix}"
#     response = model.generate_content([file, prompt], stream=False)
#     return response
#
#
# def img_to_md(image_path, lang):
#     api_key = random_genai()
#     print(f'Using api_key ending in: {api_key[-4:]}')
#
#     if not api_key: return "Error: No API key."
#
#     try:
#         # 1. 尝试直接识别
#         try:
#             response = _do_gemini_ocr(api_key, image_path, lang)
#             text, error = _extract_text_from_response(response)
#
#             if text:
#                 return text
#
#             # 如果没有文本，打印错误原因
#             print(f"⚠️ Primary Attempt Failed: {error}")
#
#             # 只有明确是 SAFETY 问题时才去切片，如果是 MAX_TOKENS 且没内容，说明模型死循环了，切片可能也没用，但也试一下
#
#         except Exception as e:
#             print(f"⚠️ Primary Attempt Exception: {e}")
#
#         # 2. 进入切片救场模式 (如果第一次失败)
#         print("🔄 激活救场模式（切片+黑白化）...")
#
#         try:
#             print(image_path)
#             original_img = Image.open(image_path)
#             gray_img = original_img.convert('L')  # 转黑白
#             width, height = gray_img.size
#             overlap = int(height * 0.1)
#             mid = height // 2
#
#             # 切割
#             top_img = gray_img.crop((0, 0, width, mid + overlap))
#             bottom_img = gray_img.crop((0, mid - overlap, width, height))
#
#             dir_name = os.path.dirname(image_path)
#             path_top = os.path.join(dir_name, "temp_rescue_top.png")
#             path_bot = os.path.join(dir_name, "temp_rescue_bot.png")
#
#             top_img.save(path_top)
#             bottom_img.save(path_bot)
#
#             final_output = []
#
#             # 上半部分
#             try:
#                 r1 = _do_gemini_ocr(api_key, path_top, lang, " (Part 1)")
#                 t1, _ = _extract_text_from_response(r1)
#                 final_output.append(t1 if t1 else "<!-- Top part failed -->")
#             except:
#                 final_output.append("<!-- Top part error -->")
#
#             # 下半部分
#             try:
#                 r2 = _do_gemini_ocr(api_key, path_bot, lang, " (Part 2)")
#                 t2, _ = _extract_text_from_response(r2)
#                 final_output.append(t2 if t2 else "<!-- Bottom part failed -->")
#             except:
#                 final_output.append("<!-- Bottom part error -->")
#
#             if os.path.exists(path_top): os.remove(path_top)
#             if os.path.exists(path_bot): os.remove(path_bot)
#
#             return "\n\n".join(final_output)
#
#         except Exception as e:
#             return f"Error in rescue mode: {str(e)}"
#
#     except Exception:
#         print(traceback.format_exc())
#         return 'Please parse again'

# def create_generation_config():
#     """创建生成配置"""
#     return {
#         "temperature": 0.1,  # 调低温度以获得更精准的OCR结果，减少幻觉
#         "top_p": 0.95,
#         "top_k": 40,
#         # "max_output_tokens": 8192,
#         "max_output_tokens": 100000,
#         "response_mime_type": "text/plain",
#     }
#
#
def upload_to_gemini(api_key, path, mime_type=None):
    """上传文件到 Gemini"""
    genai.configure(api_key=api_key)
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(files):
    """等待文件处理完毕（Gemini API 对于某些大文件需要处理时间）"""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
#
#
def img_to_md(image_path, lang):
    api_key = random_genai()
    print('Using api_key ending in:', api_key[-4:])  # 打印Key的后四位用于调试，不打印全貌

    if not api_key:
        return "Error: No API key available."

    try:
        # 配置 API
        genai.configure(api_key=api_key)

        # 1. 上传图片
        # 注意：如果图片非常小，其实可以直接转 bytes 发送，但这里保留你的 upload_file 逻辑
        gemini_image = upload_to_gemini(api_key, image_path, mime_type="image/png")

        # 确保文件已就绪（虽然图片通常很快，但加上这个逻辑更稳健）
        # wait_for_files_active([gemini_image])

        # 2. 设置 System Instruction
        # 这里的指令非常关键，要求它强制输出 Markdown，并处理公式
        system_instruction = (
            f"你是一个专业的 OCR 助手。请识别图片中的所有内容，图中语言为:{lang},请以{lang}语言返回并将其转换为标准的 "
            f"Markdown 格式返回。如果是表格，请使用 Markdown 表格语法。如果是数学公式，请使用 LaTeX 格式（行内公式用 $ 包裹，独占一行用 $$ 包裹）。"
            f"不要包含任何开场白或结束语，只返回转换后的内容。"
        )

        # safety_settings = {
        #     "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        #     "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        #     "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        #     "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        # }

        # 3. 创建模型
        generation_config = create_generation_config()
        model = genai.GenerativeModel(
            model_name=genai_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
            safety_settings=get_safety_settings()
        )

        # 4. 生成内容 (使用 generate_content 替代 chat，因为这是一次性任务)
        # 提示词这里再次强调，防止模型“忘记”
        prompt = "请将这张图片的内容精准转换为 Markdown 格式。"

        response = model.generate_content([gemini_image, prompt])

        # 5. 清理文件 (可选，但这能防止你的 Google Drive 存满垃圾文件)
        # try:
        #     genai.delete_file(gemini_image.name)
        # except:
        #     pass

        try:
            return response.text
        except:
            print(response.candidates)
            print(traceback.format_exc())
            print(f"DEBUG: Finish Reason: {response.candidates[0].finish_reason}")
            # 强行获取截断内容
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return ""

    except Exception:
        print(traceback.format_exc())
        return 'Please parse again'


if __name__ == '__main__':
    # 确保文件存在再运行
    img_path = 'img/gongshi.png'
    lang = 'zh'
    if os.path.exists(img_path):
        result = img_to_md(img_path, lang)
        print("-" * 20 + " RESULT " + "-" * 20)
        print(result)
    else:
        print(f"文件不存在: {img_path}")
