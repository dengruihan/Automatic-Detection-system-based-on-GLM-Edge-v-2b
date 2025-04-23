import os
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import torch
from ov_convert.convert_v import OvGLMv
import cv2
import numpy as np
from threading import Thread





def detect_applesnails_with_two_ranges(image_path):
    print(f"DEBUG: 正在处理图像路径 {image_path}")
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"图像加载失败：{image_path}")
        return 0

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    lower_color1 = np.array([150, 75, 140])
    upper_color1 = np.array([200, 150, 250])

    lower_color2 = np.array([160, 140, 80])
    upper_color2 = np.array([190, 190, 140])

    lower_color3 = np.array([90, 126, 128])
    upper_color3 = np.array([180, 180, 160])

    # 生成掩膜
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask3 = cv2.inRange(hsv, lower_color3, upper_color3)

    # 合并掩膜
    combined_mask = cv2.bitwise_or(mask1, mask2, mask3)

    # 形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 统计福寿螺数量
    snail_count = sum(1 for contour in contours if cv2.contourArea(contour) > 80)

    return snail_count

def process_folder_and_save_results(image_folder):
    # 创建 output 文件夹（如果不存在）
    output_folder = os.path.join(image_folder, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建文件夹：{output_folder}")

    # 遍历文件夹并筛选图片文件
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []

    for filename in os.listdir(image_folder):
        # 检查文件是否为图片
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)  # 拼接完整路径
            
            if os.path.exists(image_path):
                print(f"正在处理：{filename}")
                detected_count = detect_applesnails_with_two_ranges(image_path)
                results.append((filename, detected_count))

                # 如果检测到福寿螺卵，保存图片到 output 文件夹
                if detected_count > 0:
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, cv2.imread(image_path))  # 保存原图
                    print(f"检测到福寿螺卵，已保存图片至：{output_path}")

            else:
                print(f"图片文件不存在：{image_path}")

    # 打印每张图片对应的卵的数量
    for result in results:
        print(f"图片 {result[0]} 对应的卵的数量：{result[1]}")

    # 计算并打印所有图片中卵的总和
    total_count = sum(count for _, count in results)
    print(f"所有图片中卵的总和：{total_count}")

def process_single_image(image_path, tokenizer, processor, model, temperature, top_p, max_length, backend):
    """处理单张图片并返回AI回答"""
    history = []
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = torch.tensor(processor(image).pixel_values).to(model.device)
    except Exception as e:
        print(f"Error opening image {image_path}: {str(e)}")
        return None

    # 固定问题（可根据需求修改）
    user_input = "how many eggs are there in the image"
    history.append([user_input, ""])

    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image"}
            ]})
            break
        if user_msg:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": user_msg}
            ]})

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )

    model_inputs = model_inputs.to(model.device)

    streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.2,
        "pixel_values": pixel_values,
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    full_response = []
    for new_token in streamer:
        if new_token:
            full_response.append(new_token)
    
    response = "".join(full_response).strip()
    return response

def initialize_models(model_path, backend, precision):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    if backend == "ov":
        model = OvGLMv(model_path, device="CPU")
    else:
        if precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if precision == "bfloat16" else torch.float16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

    return tokenizer, processor, model


