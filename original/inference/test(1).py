import argparse
from threading import Thread
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
import pandas as pd

###---------下面是颜色识别

def detect_applesnails_with_two_ranges(image_path):
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

###--------上面是颜色识别

###--------下面是颜色方案数据统计

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
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_path = os.path.join(image_folder, filename)
            print(f"正在处理：{filename}")
            detected_count, mask = detect_applesnails_with_two_ranges(image_path)
            results.append((filename, detected_count))

            # 如果检测到福寿螺卵，保存图片到 output 文件夹
            if detected_count > 0:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cv2.imread(image_path))  # 保存原图
                print(f"检测到福寿螺卵，已保存图片至：{output_path}")

    # 打印结果
    print("\n===== 检测结果 =====")
    for filename, count in results:
        print(f"文件：{filename} \t 福寿螺数量：{count}")
    print("===================")

###--------上面是颜色方案数据统计

# 支持的图片扩展名列表
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

###---------下面是AI回答

def process_folder(folder_path):
    """处理文件夹并返回排序后的图片文件列表"""
    image_files = []
    for filename in sorted(os.listdir(folder_path)):  # 按文件名排序
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            image_files.append(filename)
    return image_files

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

###--------上面是ai回答

###--------下面主程序

def main():
    # 获取文件夹路径
    folder_path = input("请输入图片文件夹路径: ").strip()
    if not os.path.isdir(folder_path):
        print("错误：输入的路径不是一个有效的文件夹")
        return

    # 设置图片文件夹路径
    image_folder = folder_path

    process_folder_and_save_results(image_folder)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run GLM-Edge-v DEMO Chat")
    parser.add_argument("--backend", type=str, choices=["transformers", "ov"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float16", "bfloat16", "int4"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=8)
    args = parser.parse_args()

    # 初始化模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if args.backend == "ov":
        model = OvGLMv(args.model_path, device="CPU")
    else:
        if args.precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if args.precision == "bfloat16" else torch.float16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

    # 处理所有图片
    image_files = process_folder(folder_path)
    results = {}

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, "/output", filename)
        print(f"\n处理图片 {idx}/{len(image_files)}: {filename}")
        
        response = process_single_image(
            image_path=image_path,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            backend=args.backend
        )
        
        if response:
            results[filename] = response
            print(f"结果: {response}")
        else:
            results[filename] = "处理失败"
            print("处理失败")

            

if __name__ == "__main__":
    main()