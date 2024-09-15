import random
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
import requests
import json
import sys

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

def detect_helmet(user_id):
    # AWS S3 구성 설정
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    # 모델 로드
    model = YOLO("C:/safeT/ai_integration/Person_Detection/best_last.pt")

    class_names = ['With Helmet', 'Without Helmet']

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    detection_start_time = None
    detection_time = 0
    detected_class = None
    required_detection_time = 3
    stoped_detection_time = 10
    count = 0
    record_done = False

    capture_dir = "penalty/"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    def get_current_datetime():
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        results = model(frame)
        annotated_frame = frame.copy()

        current_time = time.time()
        target_class_detected = False

        for result in results:
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    box = box.cpu().numpy()
                    conf = conf.cpu().numpy()
                    cls = int(cls.cpu().numpy())
                    restart_time = time.time() - detection_time

                    if cls == 0 and conf >= 0.8:
                        label = f'{class_names[cls]} {conf:.2f}'
                        plot_one_box(box, annotated_frame, label=label, color=(255, 0, 0), line_thickness=2)
                    if cls == 1 and conf >= 0.5 and int(stoped_detection_time - restart_time) < 0:
                        label = f'{class_names[cls]} {conf:.2f}'
                        plot_one_box(box, annotated_frame, label=label, color=(0, 0, 255), line_thickness=2)

                        if detection_start_time is None:
                            detection_start_time = time.time()
                        else:
                            elapsed_time = time.time() - detection_start_time
                            if elapsed_time >= required_detection_time:
                                if not record_done:
                                    file_name = f"user{user_id}_{current_time}.png"
                                    capture_filename = os.path.join(capture_dir, file_name)
                                    current_time_str = get_current_datetime()
                                    cv2.imwrite(capture_filename, frame)

                                    if s3_client:
                                        s3_url = upload_to_s3(capture_filename, S3_BUCKET_NAME, s3_client)
                                        if s3_url:
                                            print(f"S3에 이미지 업로드 성공 : {s3_url}")
                                        else:
                                            print("S3에 이미지 업로드 실패")
                                    else:
                                        print("S3 연결 오류로 이미지 업로드 실패")

                                    file_path = f"{capture_dir}{file_name}"
                                    os.remove(file_path)
                                    record_done = True
                                    detection_start_time = None
                                count += 1
                                detection_time = time.time()
                        target_class_detected = True

        if not target_class_detected:
            detection_start_time = None

        cv2.imshow('YOLOv8 Live Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"감지 횟수 : {count}")
            flask_url = "http://localhost:5000/send-detection"
            final_data = {
                "userId": user_id,
                "content": detected_class,
                "photo": s3_url if s3_url else "N/A",
                "date": current_time_str,
                "map": {
                    "latitude": 37.3,
                    "longitude": 127.457
                },
                "detectionCount": count
            }
            response = requests.post(flask_url, json=final_data)
            print("Flask 서버로 데이터 전송 완료:", response.status_code)
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용자 ID를 제공해주세요")
        sys.exit(1)

    user_id = sys.argv[1]
    detect_helmet(user_id)
