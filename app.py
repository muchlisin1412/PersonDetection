from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from ultralytics import YOLO  
from model import detect_people, models  
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')

# Definisikan models di sini
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']  # Sesuaikan dengan model yang Anda miliki

if not os.path.exists(app.config['UPLOAD_FOLDER']):
	os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
	return render_template('index.html', models=models)

def process_video(file_path, model):
    video = cv2.VideoCapture(file_path)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('static/uploads/output.mp4', fourcc, fps, (width, height))
    
    total_persons = 0
    total_time = 0
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        start_time = time.time()
        results = model(frame)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        frame_count += 1
        
        # Assuming the model returns a list of detections
        persons = [det for det in results[0].boxes.data if int(det[5]) == 0]  # Assuming person class is 0
        total_persons += len(persons)
        
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    
    video.release()
    out.release()
    
    average_persons = total_persons / frame_count if frame_count > 0 else 0
    average_time = total_time / frame_count if frame_count > 0 else 0
    
    return 'output.mp4', average_persons, average_time

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(video_path, mimetype='video/mp4', 
                     as_attachment=True, 
                     download_name=filename)

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        model_name = request.form.get('model', 'yolov8n.pt')
        model = YOLO(model_name)
        
        person_count = 0
        inference_time = 0
        
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            output_filename, person_count, inference_time = process_video(file_path, model)
        else:
            img = cv2.imread(file_path)
            img_result, person_count, inference_time, detections = detect_people(img, model)
            output_filename = f'result_{filename}'
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), img_result)
        
        return render_template('result.html', filename=output_filename, model_name=model_name, person_count=person_count, inference_time=inference_time)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return redirect(url_for('static', filename=f'uploads/{filename}'))
	
if __name__ == '__main__':
	app.run(debug=True)