import torch
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import imageio.v2 as imageio  # ImageIO v2를 명시적으로 사용
import cv2
import numpy as np


app = Flask(__name__)

# YOLOv5 모델 로드
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('C:/yolov5', 'custom', path='C:/yolov5/runs/train/exp/weights/best.pt', source='local', force_reload=True)

# 클래스 이름 출력
class_names = model.names
print("Class names:", class_names)  # 디버깅을 위한 클래스 이름 출력

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        try:
            file = request.files['image']
            if not file:
                return render_template('index.html', ml_label="파일이 업로드되지 않았습니다.")
            
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', 'uploads', filename)
            file.save(file_path)

            # 이미지 전처리
            original_image = preprocess_image(file_path)
            
            # YOLOv5 모델의 입력 크기에 맞춰 이미지 크기 조정
            stride = int(model.stride) if isinstance(model.stride, int) else int(max(model.stride))
            resized_image = cv2.resize(original_image, (stride * ((original_image.shape[1] + stride - 1) // stride),
                                                        stride * ((original_image.shape[0] + stride - 1) // stride)))

            # 모델 예측
            results = model(resized_image)

            # 예측 결과 후처리
            obj = postprocess_results(original_image, results, class_names)
            print("Detected objects:", obj)  # 디버깅을 위한 감지된 객체 출력

            output_image_path = os.path.join('static', 'uploads', filename)
            imageio.imwrite(output_image_path, original_image)

            return render_template('predict.html', ml_label=obj, print_image=filename)
        except Exception as e:
            return str(e)

def preprocess_image(image_path):
    img = imageio.imread(image_path)
    
    # 이미지 크기 조정
    img_cropped = img[0:800, 0:800]
    img_resized = cv2.resize(img_cropped, (640, 640))  # 모델의 입력 크기에 맞춤
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # 색상 채널을 RGB로 변환
    
    return img_rgb

def postprocess_results(image, results, class_names):
    boxes = results.xyxy[0].numpy()
    
    def objects_info(boxes, class_names):
        objects = []
        for box in boxes:
            obj = {
                'class': class_names[int(box[5])],
                'confidence': float(box[4]),
                'bbox': box[:4].tolist()
            }
            objects.append(obj)
        return objects
    
    def plot_boxes(image, boxes, class_names, plot_labels=True):
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if plot_labels:
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    plot_boxes(image, boxes, class_names)
    return objects_info(boxes, class_names)

if __name__ == "__main__":
    os.makedirs('static/uploads', exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
