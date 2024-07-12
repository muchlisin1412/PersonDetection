import cv2
import numpy as np
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']

def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_map(detections, num_classes=1, iou_threshold=0.5, confidence_threshold=0.5):
    if not detections:
        return 0  # Return 0 if there are no detections

    average_precisions = []
    
    for class_id in range(num_classes):
        class_detections = [d for d in detections if d['class'] == class_id and d['confidence'] >= confidence_threshold]
        if not class_detections:
            continue
        
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        num_gt = len(class_detections)  # Assuming each detection corresponds to a ground truth
        true_positives = np.zeros(len(class_detections))
        false_positives = np.zeros(len(class_detections))
        
        detected_gt = set()
        
        for i, detection in enumerate(class_detections):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_detections):
                if j in detected_gt:
                    continue
                
                iou = calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                if best_gt_idx not in detected_gt:
                    true_positives[i] = 1
                    detected_gt.add(best_gt_idx)
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1
        
        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        
        recalls = cumulative_tp / num_gt
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
        
        # Compute average precision
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11
        average_precisions.append(ap)
    
    if not average_precisions:
        return 0  # Return 0 if no class had any detections
    
    return np.mean(average_precisions)

def detect_people(frame, model, conf_threshold=0.5, iou_threshold=0.5):
    start_time = time.time()
    results = model(frame, conf=conf_threshold, iou=iou_threshold)
    inference_time = time.time() - start_time
    
    person_count = 0
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0:  # Class 0 adalah orang dalam dataset COCO
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                detections.append({
                    'class': 0,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
    
    cv2.putText(frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame, person_count, inference_time, detections

def process_webcam():
    cap = cv2.VideoCapture(0)
    
    for model_name in models:
        print(f"\nMenggunakan model: {model_name}")
        model = YOLO(model_name)
        
        frame_count = 0
        total_inference_time = 0
        total_person_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_result, person_count, inference_time, detections = detect_people(frame, model)
            
            total_inference_time += inference_time
            total_person_count += person_count
            all_detections.extend(detections)
            
            cv2.putText(frame_result, f"Model: {model_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_result, f"Inference Time: {inference_time:.4f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Webcam Detection', frame_result)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= 100:  # Process 100 frames
                break
        
        avg_inference_time = total_inference_time / frame_count
        avg_person_count = total_person_count / frame_count
        estimated_map = calculate_map(all_detections)
        
        print(f"Rata-rata jumlah orang terdeteksi per frame: {avg_person_count:.2f}")
        print(f"Rata-rata inference time: {avg_inference_time:.4f} detik")
        print(f"Estimated mAP: {estimated_map:.4f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        print("Tidak ada file yang dipilih.")
        return
    
    img = cv2.imread(file_path)
    
    for model_name in models:
        print(f"\nMenggunakan model: {model_name}")
        model = YOLO(model_name)
        
        start_time = time.time()
        img_result, person_count, inference_time, detections = detect_people(img.copy(), model)
        latency_time = time.time() - start_time
        
        estimated_map = calculate_map(detections)
        
        print(f"Jumlah orang terdeteksi: {person_count}")
        print(f"Inference time: {inference_time:.4f} detik")
        print(f"Latency time: {latency_time:.4f} detik")
        print(f"Estimated mAP: {estimated_map:.4f}")
        
        cv2.imshow(f'Detection Result - {model_name}', img_result)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def process_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        print("Tidak ada file yang dipilih.")
        return
    
    cap = cv2.VideoCapture(file_path)
    
    for model_name in models:
        print(f"\nMenggunakan model: {model_name}")
        model = YOLO(model_name)
        
        frame_count = 0
        total_inference_time = 0
        total_person_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            frame_result, person_count, inference_time, detections = detect_people(frame, model)
            
            total_inference_time += inference_time
            total_person_count += person_count
            all_detections.extend(detections)
            
            cv2.putText(frame_result, f"Model: {model_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_result, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Video Detection', frame_result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        avg_inference_time = total_inference_time / frame_count
        avg_person_count = total_person_count / frame_count
        estimated_map = calculate_map(all_detections)
        
        print(f"Rata-rata jumlah orang terdeteksi per frame: {avg_person_count:.2f}")
        print(f"Rata-rata inference time: {avg_inference_time:.4f} detik")
        print(f"Estimated mAP: {estimated_map:.4f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nPilih mode:")
        print("1. Deteksi menggunakan webcam")
        print("2. Deteksi pada gambar")
        print("3. Deteksi pada video")
        print("4. Keluar")
        
        choice = input("Masukkan pilihan (1/2/3/4): ")
        
        if choice == '1':
            process_webcam()
        elif choice == '2':
            process_image()
        elif choice == '3':
            process_video()
        elif choice == '4':
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

if __name__ == "__main__":
    main()