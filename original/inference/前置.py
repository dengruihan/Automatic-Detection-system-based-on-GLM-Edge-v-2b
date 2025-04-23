import os
from PIL import Image

# 用户输入文件夹路径
folder_path = input("请输入文件夹路径：")
new_folder = os.path.join(folder_path, 'a')

# 创建文件夹a，如果已存在则继续
try:
    os.mkdir(new_folder)
except FileExistsError:
    print("文件夹'a'已经存在，将继续使用。")

# 清空egg_count.txt文件
with open(os.path.join(new_folder, 'egg_count.txt'), 'w') as f:
    pass

# 获取文件夹中的所有图片文件
supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]

# 处理每个图片文件
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    try:
        img = Image.open(image_path)
        pixels = img.load()
        has_pink = False
        for i in range(img.width):
            for j in range(img.height):
                r, g, b = pixels[i, j]
                if r > 180 and g < 150 and b < 150 and r > b and r > g:
                    has_pink = True
                    break
            if has_pink:
                break
        if has_pink:
            if img.width >= 1100 and img.height >= 800:
                cropped_img = img.crop((0, 0, 1100, 800))
                save_path = os.path.join(new_folder, image_file)
                count = 1
                while os.path.exists(save_path):
                    name, ext = os.path.splitext(image_file)
                    save_path = os.path.join(new_folder, f"{name}_{count}{ext}")
                    count += 1
                cropped_img.save(save_path)
            else:
                print(f"{image_file} 的尺寸不足以截取1100x800，已跳过。")
        else:
            with open(os.path.join(new_folder, 'egg_count.txt'), 'a') as f:
                f.write(f"{image_file}: 0 eggs\n")
    except Exception as e:
        print(f"处理图片{image_file}时发生错误：{e}")