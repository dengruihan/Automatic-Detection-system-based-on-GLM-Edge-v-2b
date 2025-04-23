import cv2
import numpy as np
import os
import pandas as pd

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

def process_images_and_print_results(image_folder):
    # 初始化一个列表来存储每张图片的检测结果
    results = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 检查文件是否为图片
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)  # 拼接完整路径
            
            if os.path.exists(image_path):
                print(f"正在处理：{filename}")
                detected_count = detect_applesnails_with_two_ranges(image_path)
                results.append((filename, detected_count))
            else:
                print(f"图片文件不存在：{image_path}")

    # 打印每张图片对应的卵的数量
    for result in results:
        print(f"图片 {result[0]} 对应的卵的数量：{result[1]}")

    # 计算并打印所有图片中卵的总和
    total_count = sum(count for _, count in results)
    print(f"所有图片中卵的总和：{total_count}")

# 设置路径
image_folder = "/Users/jiework2022/Documents/CTB/model2/picture"  # 替换为图片文件夹路径


# 执行函数
process_images_and_print_results(image_folder)

