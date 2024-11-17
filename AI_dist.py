from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import differ
import cv2
import pickle
import io
from PIL import Image

# Flask 앱 생성
app = Flask(__name__)

# 저장된 모델 및 LabelEncoder 불러오기
with open('label_ver1.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

loaded_model = load_model('ver1.h5')

# 예측 함수
def predict_images(test_img, render_img):
    """
    두 이미지를 비교하여 모델로 예측 수행
    """
    ssim, dif = differ.image_dif(test_img, render_img)

    # 모델 입력에 맞게 데이터 전처리
    dif = np.expand_dims(dif, axis=0)  # 배치 차원 추가
    ssim = np.expand_dims(ssim, axis=-1)  # 채널 차원 추가 # 배치 차원 추가

    # 예측 수행
    predicted_probabilities = loaded_model.predict([dif, ssim])
    predicted_class = predicted_probabilities.argmax(axis=1)  # 가장 높은 확률의 클래스
    predicted_label = label_encoder.inverse_transform(predicted_class)  # 클래스 디코딩

    return predicted_label[0]  # 예측된 클래스 레이블 반환

# API 라우트 정의
@app.route('/process', methods=['POST'])
def prediction():
    """
    두 이미지를 받고 예측 결과 반환
    """
    try:
        # 이미지 파일 가져오기
        test_img_file = request.files['test_img']
        render_img_file = request.files['render_img']

        # 이미지를 OpenCV 형식으로 읽기
        test_img = Image.open(io.BytesIO(test_img_file.read())).convert('L')  # 흑백 변환
        render_img = Image.open(io.BytesIO(render_img_file.read())).convert('L')  # 흑백 변환

        test_img = np.array(test_img)
        render_img = np.array(render_img)

        # 예측 수행
        predicted_label = predict_images(test_img, render_img)

        # 결과 반환
        return jsonify({'res': predicted_label})

    except Exception as e:
        return jsonify({'res': None, 'message': "잘못된 이미지입니다."})

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True)