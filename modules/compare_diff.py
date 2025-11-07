import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 파일 경로 설정 ===
test_img_path = "./data/def_front/cast_def_0_0.jpeg"   # 테스트(이상) 이미지
ref_img_path  = "./data/ok_front/cast_ok_0_423.jpeg"   # 기준(정상) 이미지
heatmap_path  = "./out_heatmap_gray.png"                # PatchCore의 회색 히트맵

# === 이미지 로드 ===
img_test = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
img_ref  = cv2.imread(ref_img_path,  cv2.IMREAD_COLOR)
heat     = cv2.imread(heatmap_path,  cv2.IMREAD_GRAYSCALE)

# 크기 맞추기 (테스트/기준이 약간 다를 수도 있음)
if img_test.shape != img_ref.shape:
    img_ref = cv2.resize(img_ref, (img_test.shape[1], img_test.shape[0]))

# === (1) 단순 픽셀 차이 기반 세그멘테이션 ===
diff = cv2.absdiff(img_test, img_ref)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
diff_blur = cv2.GaussianBlur(diff_gray, (5,5), 0)

# 이상 후보 영역(밝은 부분만 추출)
_, mask_diff = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_BINARY)

# === (2) PatchCore 히트맵 기반 세그멘테이션 ===
# 히트맵 강한 부분만 추출 (상위 5% 픽셀)
thr_val = np.percentile(heat, 95)
_, mask_heat = cv2.threshold(heat, thr_val, 255, cv2.THRESH_BINARY)

# === (3) 두 마스크 결합 ===
mask_final = cv2.bitwise_and(mask_diff, mask_heat)

# === (4) 색상 오버레이 시각화 ===
overlay = img_test.copy()
overlay[mask_final > 0] = [0, 0, 255]  # 빨간색 표시
result = cv2.addWeighted(img_test, 0.7, overlay, 0.3, 0)

# === (5) 시각화 ===
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)); plt.title("Test Image")
plt.subplot(1,3,2); plt.imshow(mask_final, cmap='gray'); plt.title("Detected Differences (mask)")
plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Overlay result")
plt.tight_layout()
plt.show()
