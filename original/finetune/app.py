# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import uuid
import zipfile
from werkzeug.utils import secure_filename
from celery import Celery
import subprocess
from your_cli_demo_vision import process_folder_and_save_results, main as vision_main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'zip'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@celery.task(bind=True)
def process_upload(self, upload_id):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
    zip_path = os.path.join(upload_path, 'upload.zip')
    extract_path = os.path.join(upload_path, 'extracted')
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # 运行原有处理逻辑
    try:
        process_folder_and_save_results(extract_path)
        vision_main(extract_path)  # 需要调整原有main函数为可调用形式
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    
    return {'status': 'completed', 'path': upload_path}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            upload_id = str(uuid.uuid4())
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
            os.makedirs(upload_path, exist_ok=True)
            
            filename = secure_filename(file.filename)
            zip_path = os.path.join(upload_path, 'upload.zip')
            file.save(zip_path)
            
            task = process_upload.apply_async(args=[upload_id])
            return redirect(url_for('processing', task_id=task.id))
    
    return render_template('upload.html')

@app.route('/processing/<task_id>')
def processing(task_id):
    task = process_upload.AsyncResult(task_id)
    return render_template('processing.html', task=task)

@app.route('/results/<upload_id>')
def show_results(upload_id):
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'output')
    return send_from_directory(result_path, 'results.txt')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)