import cv2
from ultralytics import YOLO
import pandas as pd
import sys
import time
from datetime import datetime
import os
import requests

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

def detect_people(user_id):
    # AWS S3 구성 설정
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME

    # S3 연결 설정
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    # YOLOv8 모델 로드
    model = YOLO("C:/ai_integration/Person_Detection/yolov8n.pt")

    detection_start_time = None
    detection_time = 0
    detected_class = None
    required_detection_time = 3
    stoped_detection_time = 10
    count = 0
    record_done = False

    cap = cv2.VideoCapture(0)
    PERSON_CLASS_ID = 0

    def get_current_datetime():
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    capture_dir = "penalty/"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_time = time.time()

        results_df = pd.DataFrame(results[0].boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "class"])
        people = results_df[results_df["class"] == PERSON_CLASS_ID]
        person_count = len(people)

        restart_time = time.time() - detection_time
        if int(stoped_detection_time - restart_time) < 0:
            for _, person in people.iterrows():
                x1, y1, x2, y2, conf, cls = person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if person_count >= 2:
                if detection_start_time is None:
                    detection_start_time = time.time()
                    detected_class = "More than two people on board"
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
                            record_done = True
                            detection_start_time = None
                            file_path = f"{capture_dir}{file_name}"
                            os.remove(file_path)
                        count += 1
                        detection_time = time.time()
            elif person_count < 2:
                detection_start_time = None

        cv2.imshow('YOLOv8 Live People Counting', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"감지 횟수 : {count}")
            flask_url = "http://localhost:5000/send-detection"
            final_data = {
                "userId": user_id,
                "content": detected_class,
                "photo": s3_url if s3_url else "N/A",
                "date": current_time_str,
                "map": {
                    "latitude": 37.12233,
                    "longitude": 127.472
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
    detect_people(user_id)
