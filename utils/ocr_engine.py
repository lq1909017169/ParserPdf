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


def upload_to_gemini(api_key, path, mime_type=None):
    genai.configure(api_key=api_key)
    file = genai.upload_file(path, mime_type=mime_type)
    return file


def _run_gemini_inference(api_key, image_path, lang, is_retry=False):
    """
    实际执行 Gemini 推理的内部函数
    """
    genai.configure(api_key=api_key)

    # 上传图片
    gemini_image = upload_to_gemini(api_key, image_path, mime_type="image/png")

    # Prompt 强调数据提取
    context_desc = "上半部分" if is_retry else "完整"
    system_instruction = (
        f"你是一个OCR引擎。请读取图片({context_desc})中的文字。语言:{lang}。"
        "忽略图片中的照片、人脸或插图，仅输出文字。"
        "直接输出 Markdown，不要解释。"
    )

    model = genai.GenerativeModel(
        model_name=genai_name,
        generation_config=create_generation_config(),
        system_instruction=system_instruction,
        safety_settings=get_safety_settings()
    )

    prompt = f"Extract text to Markdown ({lang})."

    # 发送请求
    response = model.generate_content([gemini_image, prompt], stream=False)

    return response


def img_to_md(image_path, lang):
    api_key = random_genai()  # 假设你外部有这个函数
    print(f'Using api_key ending in: {api_key[-4:]} for {image_path}')

    if not api_key:
        return "Error: No API key available."

    try:
        # --- 第1次尝试：直接识别整图 ---
        response = _run_gemini_inference(api_key, image_path, lang)

        # 检查是否因为安全原因被拦截
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"WARN: Prompt blocked directly. Reason: {response.prompt_feedback.block_reason}")
            # 如果 Prompt 就被拦了，通常没救，但可以尝试切片

        try:
            return response.text
        except ValueError:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            print(f"WARN: 1st Attempt Blocked. Finish Reason: {finish_reason}")

            # 如果不是因为安全拦截（比如只是内容为空），直接返回空
            # 2 代表 SAFETY, 3 代表 RECITATION (版权)
            if str(finish_reason) not in ["2", "SAFETY", "3", "RECITATION"]:
                return "Error: Unknown Parsing Error"

            # --- 第2次尝试：启动“切片重试”策略 ---
            print("🔄 触发安全拦截，尝试【切片重试策略】...")

            # 1. 打开原图
            img = Image.open(image_path)
            width, height = img.size

            # 2. 切割图片（上下两半，中间重叠 50px 防止切断文字）
            overlap = 50
            mid_point = height // 2

            top_crop = img.crop((0, 0, width, mid_point + overlap))
            bottom_crop = img.crop((0, mid_point - overlap, width, height))

            # 保存临时文件
            temp_dir = os.path.dirname(image_path)
            top_path = os.path.join(temp_dir, "temp_top.png")
            bottom_path = os.path.join(temp_dir, "temp_bottom.png")

            top_crop.save(top_path)
            bottom_crop.save(bottom_path)

            try:
                # 3. 分别识别
                print("   Processing Top Half...")
                res_top = _run_gemini_inference(api_key, top_path, lang, is_retry=True)
                text_top = ""
                try:
                    text_top = res_top.text
                except:
                    text_top = "(Top half blocked)"

                print("   Processing Bottom Half...")
                res_bottom = _run_gemini_inference(api_key, bottom_path, lang, is_retry=True)
                text_bottom = ""
                try:
                    text_bottom = res_bottom.text
                except:
                    text_bottom = "(Bottom half blocked)"

                print("✅ Slicing Success!")
                return text_top + "\n\n" + text_bottom

            finally:
                # 清理临时切片文件
                if os.path.exists(top_path): os.remove(top_path)
                if os.path.exists(bottom_path): os.remove(bottom_path)

    except Exception:
        print(traceback.format_exc())
        return 'Please parse again'


# def create_generation_config():
#     """创建生成配置"""
#     return {
#         "temperature": 0.1,  # 调低温度以获得更精准的OCR结果，减少幻觉
#         "top_p": 0.95,
#         "top_k": 40,
#         "max_output_tokens": 8192,
#         "response_mime_type": "text/plain",
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
#         # wait_for_files_active([gemini_image])
#
#         # 2. 设置 System Instruction
#         # 这里的指令非常关键，要求它强制输出 Markdown，并处理公式
#         system_instruction = (
#             f"你是一个专业的 OCR 助手。请识别图片中的所有内容，图中语言为:{lang},请以{lang}语言返回并将其转换为标准的 Markdown 格式返回。"
#             "如果是表格，请使用 Markdown 表格语法。"
#             "如果是数学公式，请使用 LaTeX 格式（行内公式用 $ 包裹，独占一行用 $$ 包裹）。"
#             "不要包含任何开场白或结束语，只返回转换后的内容。"
#         )
#
#         safety_settings = {
#             "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
#             "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
#             "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
#             "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
#         }
#
#         # 3. 创建模型
#         generation_config = create_generation_config()
#         model = genai.GenerativeModel(
#             model_name=genai_name,
#             generation_config=generation_config,
#             system_instruction=system_instruction,
#             safety_settings=safety_settings
#         )
#
#         # 4. 生成内容 (使用 generate_content 替代 chat，因为这是一次性任务)
#         # 提示词这里再次强调，防止模型“忘记”
#         prompt = "请将这张图片的内容精准转换为 Markdown 格式。"
#
#         response = model.generate_content([gemini_image, prompt])
#
#         # 5. 清理文件 (可选，但这能防止你的 Google Drive 存满垃圾文件)
#         # try:
#         #     genai.delete_file(gemini_image.name)
#         # except:
#         #     pass
#
#         try:
#             return response.text
#         except ValueError:
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
    img_path = 'img/gongshi.png'
    lang = 'zh'
    if os.path.exists(img_path):
        result = img_to_md(img_path, lang)
        print("-" * 20 + " RESULT " + "-" * 20)
        print(result)
    else:
        print(f"文件不存在: {img_path}")
