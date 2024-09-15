from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import time
import requests
from s3 import s3_connection, upload_to_s3
import secret_key

app = Flask(__name__)

def detect_face(user_id):
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    use_camera = False  # False면 img, True면 camera
    csv_filename = "C:/safeT/ai_integration/Face_Recogniton/face_rec_data/face_features.csv"
    df = pd.read_csv(csv_filename)
    saved_encodings = df.values

    result_dir = 'face_rec_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = imgTest

        faceLocTest = face_recognition.face_locations(rgb_frame)
        encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

        if not encodeTest:
            print("No face found in the frame.")
        else:
            for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
                faceDis = face_recognition.face_distance(saved_encodings, face_encoding)
                current_min_distance = np.min(faceDis)
                print(f"Face distance: {current_min_distance}")

                if current_min_distance < min_distance:
                    min_distance = current_min_distance
                    if min_distance <= 0.4:
                        best_match_text = "동일인입니다."
                    else:
                        best_match_text = "동일인이 아닙니다."

                if use_camera:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                else:
                    cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

        if time.time() - start_time >= 5:
            break

        if use_camera:
            cv2.imshow('Camera', frame)
        else:
            cv2.imshow('Test Image', imgTest)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"최종 결과: {best_match_text} (유사도 거리: {min_distance})")
    result_image_path = os.path.join(result_dir, 'result_image.jpg')
    if use_camera:
        cv2.imwrite(result_image_path, frame)
    else:
        cv2.imwrite(result_image_path, cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR))

    if s3_client:
        s3_url = upload_to_s3(result_image_path, S3_BUCKET_NAME, s3_client)
        if s3_url:
            print(f"S3에 이미지 업로드 성공 : {s3_url}")
        else:
            print("S3에 이미지 업로드 실패")
    else:
        print("S3 연결 오류로 이미지 업로드 실패")

    return {
        "userId": user_id,
        "result": best_match_text,
        "distance": min_distance,
        "image_url": s3_url if s3_url else "N/A"
    }

@app.route('/face-detection', methods=['POST'])
def face_detection():
    data = request.json
    user_id = data.get('userId')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    result = detect_face(user_id)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
