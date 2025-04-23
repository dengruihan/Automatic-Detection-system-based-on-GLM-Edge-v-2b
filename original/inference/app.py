# app.py
from flask import Flask, render_template, request, jsonify
import os
import uuid
import threading
from werkzeug.utils import secure_filename
import json

import re
from word2number import w2n

# 导入原有功能（需要将原有cli_demo_vision.py中的相关函数整理到vision.py）
from vision import detect_applesnails_with_two_ranges, process_folder_and_save_results, process_single_image, initialize_models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# 全局模型变量
models_initialized = False
tokenizer = None
processor = None
model = None

def initialize_models_once():
    global models_initialized, tokenizer, processor, model
    if not models_initialized:
        tokenizer, processor, model = initialize_models(
            model_path="C:/GLM-Edge-main/training_model",
            backend="transformers",
            precision="bfloat16"
        )
        models_initialized = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    # 创建临时目录
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(temp_dir, exist_ok=True)

    # 保存上传的文件
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            saved_files.append(file_path)

    # 启动后台处理任务
    thread = threading.Thread(target=process_images, args=(temp_dir, session_id))
    thread.start()

    return jsonify({'session_id': session_id}), 202

@app.route('/api/data')
def get_data():
    return jsonify({
        "message": "这是来自后端的字符串",
        "number": 42
    })

@app.route('/status/<session_id>')
def get_status(session_id):
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'result.json')
    if os.path.exists(result_file):
        return jsonify({'status': 'complete'})
    return jsonify({'status': 'processing'})

@app.route('/result/<session_id>')
def get_result(session_id):
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'result.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Result not ready'}), 404

def process_images(temp_dir, session_id):
    try:
        # 初始化模型
        initialize_models_once()

        # 运行图像处理
        process_folder_and_save_results(temp_dir)

        # 获取结果文件
        output_dir = os.path.join(temp_dir, 'output')
        image_files = [f for f in os.listdir(output_dir) if allowed_file(f)]

        # 处理每个图像
        results = {}
        for filename in image_files:
            image_path = os.path.join(output_dir, filename)
            response = process_single_image(
                image_path=image_path,
                tokenizer=tokenizer,
                processor=processor,
                model=model,
                temperature=0.1,
                top_p=0.1,
                max_length=8,
                backend="transformers"
            )
            results[filename] = response or "处理失败"

        # 保存结果
        result_path = os.path.join(temp_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'results': results,
                'output_dir': output_dir
            }, f)

    except Exception as e:
        print(f"处理错误: {str(e)}")

    total_eggs = 0
    error_files = []
    
    for filename, response in results.items():
        # 匹配所有数字和英文单词（不区分大小写）
        matches = re.findall(r'\b\d+\b|\b[a-zA-Z]+\b', response.lower())
        valid_numbers = []
        
        for item in matches:
            try:
                # 优先尝试转换英文单词
                if item.isdigit():
                    num = int(item)
                else:
                    num = w2n.word_to_num(item)
                valid_numbers.append(num)
            except (ValueError, AttributeError):
                continue
        
        if valid_numbers:
            # 取最后一个有效数字（假设最后出现的是最终答案）
            count = valid_numbers[-1]
            total_eggs += count
        else:
            error_files.append(filename)

    # 输出统计结果
    print("\n===== 统计结果 =====")
    print(f"检测到含卵图片数量: {len(results)} 张")
    print(f"成功解析图片数量: {len(results)-len(error_files)} 张")
    print(f"总卵数量: {total_eggs}")
    
    if error_files:
        print("\n以下文件未能解析有效数字：")
        for f in error_files:
            print(f" - {f} (原始响应: {results[f]})")


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)