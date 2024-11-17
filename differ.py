from skimage.metrics import structural_similarity as ssim
import cv2

# 이미지 로드
def image_dif(real_image, rendered_image):

    if real_image.shape != rendered_image.shape:
        height, width = real_image.shape  # 기준 크기 설정
        rendered_image = cv2.resize(rendered_image, (width, height))
    mask = cv2.inRange(rendered_image, 0, 55)
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 외곽선을 기준으로 ROI 계산
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)  # ROI 좌표

        # 주변 영역을 포함하여 ROI 확장
        padding = int((mask.shape[0]) / 15)  # 추가할 여백 크기
        x_start = max(x - padding, 0)
        y_start = max(y - padding, 0)
        x_end = min(x + w + padding, rendered_image.shape[1])
        y_end = min(y + h + padding, rendered_image.shape[0])

        # 확장된 영역으로 크롭

        rendered_image_c = rendered_image[y_start:y_end, x_start:x_end]
        real_image_c = real_image[y_start:y_end, x_start:x_end]
    else:
        print("유효한 ROI를 찾을 수 없습니다.")
    # 이미지 크기 가져오기

    h, w = real_image_c.shape[:2]  # 이미지 높이와 너비
    crop_width, crop_height = int(real_image.shape[0]/16), int(real_image.shape[0]/16)
    # 이미지 중심 좌표 계산
    center_x, center_y = w // 2, h // 2

    # 크롭 영역 계산
    x_start = max(center_x - crop_width // 2, 0)
    y_start = max(center_y - crop_height // 2, 0)
    x_end = min(center_x + crop_width // 2, w)
    y_end = min(center_y + crop_height // 2, h)

    # 이미지 크롭
    real_image_c = real_image_c[y_start:y_end, x_start:x_end]
    rendered_image_c = rendered_image_c[y_start:y_end, x_start:x_end]

    # SSIM 계산
    score, diff = ssim(real_image_c, rendered_image_c, full=True)
    return score, diff