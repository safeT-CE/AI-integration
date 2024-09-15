import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import time
import requests

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

def detect_face(user_id):
    # AWS S3 구성 설정
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    # use_camera 변수 설정
    use_camera = False  # False면 img, True면 camera

    # CSV 파일 경로
    csv_filename = "C:/ai_integration/Face_Recogniton/face_rec_data/face_features.csv"

    # 저장된 얼굴 데이터 불러오기
    df = pd.read_csv(csv_filename)
    saved_encodings = df.values

    # 결과를 저장할 디렉토리 생성
    result_dir = 'face_rec_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if use_camera:
        # 카메라 캡처 시작
        cap = cv2.VideoCapture(0)
    else:
        # 테스트 이미지 불러오기
        test_image_path = 'face_pics/ai_hub_data/age/people1 (3).jpg'
        imgTest = face_recognition.load_image_file(test_image_path)
        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    min_distance = float('inf')
    best_match_text = ""

    while True:
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                break
            # 프레임을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = imgTest

        # 프레임에서 얼굴 인식 및 인코딩
        faceLocTest = face_recognition.face_locations(rgb_frame)
        encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

        if not encodeTest:
            print("No face found in the frame.")
        else:
            for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
                # 저장된 얼굴 데이터와 현재 프레임 얼굴 비교
                faceDis = face_recognition.face_distance(saved_encodings, face_encoding)
                current_min_distance = np.min(faceDis)

                # 디버깅: 유사도 값 출력
                print(f"Face distance: {current_min_distance}")

                # 최솟값 갱신
                if current_min_distance < min_distance:
                    min_distance = current_min_distance
                    if min_distance <= 0.4:
                        best_match_text = "동일인입니다."
                    else:
                        best_match_text = "동일인이 아닙니다."

                # 얼굴 주위에 사각형 그리기
                if use_camera:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                else:
                    cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

        # N초가 지나면 루프 종료
        if time.time() - start_time >= 5:
            break

        # 결과 보여주기
        if use_camera:
            cv2.imshow('Camera', frame)
        else:
            cv2.imshow('Test Image', imgTest)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 최종 결과 출력
    print(f"최종 결과: {best_match_text} (유사도 거리: {min_distance})")

    # 결과 이미지를 파일로 저장
    result_image_path = os.path.join(result_dir, 'result_image.jpg')
    if use_camera:
        cv2.imwrite(result_image_path, frame)
    else:
        cv2.imwrite(result_image_path, cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR))

    # AWS S3에 이미지 업로드
    if s3_client:
        s3_url = upload_to_s3(result_image_path, S3_BUCKET_NAME, s3_client)
        if s3_url:
            print(f"S3에 이미지 업로드 성공 : {s3_url}")
        else:
            print("S3에 이미지 업로드 실패")
    else:
        print("S3 연결 오류로 이미지 업로드 실패")

    # Flask 서버로 결과 전송
    flask_url = "http://localhost:5000/send-detection"
    final_data = {
        "userId": user_id,
        "result": best_match_text,
        "distance": min_distance,
        "image_url": s3_url if s3_url else "N/A"
    }
    response = requests.post(flask_url, json=final_data)
    print("Flask 서버로 데이터 전송 완료:", response.status_code)

    # 카메라 및 윈도우 종료
    if use_camera:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용자 ID를 제공해주세요")
        sys.exit(1)

    user_id = sys.argv[1]
    detect_face(user_id)