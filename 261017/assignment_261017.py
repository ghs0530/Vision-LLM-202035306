
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 농구공 이미지 불러오기
# ==========================================

basketball = cv2.imread('src/images/basketball_crop.jpg')

if basketball is None:
    raise FileNotFoundError("농구공 이미지를 찾을 수 없습니다.")

basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(basketball_rgb)
plt.title("Original")
plt.axis("off")
plt.show()


# ==========================================
# 2. HSV 색공간 기반 마스킹
# ==========================================

basketball_hsv = cv2.cvtColor(basketball, cv2.COLOR_BGR2HSV)

lower_orange_hsv = np.array([0, 40, 40])
upper_orange_hsv = np.array([25, 255, 255])

mask_hsv = cv2.inRange(
    basketball_hsv,
    lower_orange_hsv,
    upper_orange_hsv
)

result_hsv = cv2.bitwise_and(
    basketball_rgb,
    basketball_rgb,
    mask=mask_hsv
)


# ==========================================
# 3. LAB 색공간 기반 마스킹
# ==========================================

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

mask_lab = cv2.inRange(
    basketball_lab,
    lower_basketball_lab,
    upper_basketball_lab
)

result_lab = cv2.bitwise_and(
    basketball_rgb,
    basketball_rgb,
    mask=mask_lab
)

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(basketball_rgb)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(result_hsv)
plt.title("HSV Mask Result")

plt.subplot(1, 3, 3)
plt.imshow(result_lab)
plt.title("LAB Mask Result")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 4. CLAHE를 이용한 명암 대비 개선
# ==========================================

# LAB 채널 분리
l, a, b = cv2.split(basketball_lab)

# CLAHE 생성
clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

# L 채널에 CLAHE 적용
l_clahe_enhanced = clahe.apply(l)

# LAB 채널 병합
basketball_enhanced_lab = cv2.merge(
    (l_clahe_enhanced, a, b)
)

# CLAHE가 적용된 LAB 이미지에서 마스크 생성
mask_enhanced = cv2.inRange(
    basketball_enhanced_lab,
    lower_basketball_lab,
    upper_basketball_lab
)

# LAB 이미지를 RGB로 변환
basketball_enhanced_rgb = cv2.cvtColor(
    basketball_enhanced_lab,
    cv2.COLOR_LAB2RGB
)

# 마스크를 이용해 관심 영역 추출
final_result = cv2.bitwise_and(
    basketball_enhanced_rgb,
    basketball_enhanced_rgb,
    mask=mask_enhanced
)

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(basketball_rgb)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(result_lab)
plt.title("LAB Segmentation Result")

plt.subplot(1, 3, 3)
plt.imshow(final_result)
plt.title("LAB (CLAHE) Enhanced Segmentation Result")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 5. Morphological Operation 비교
# ==========================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

# Opening 1회
ball_mask_opened = cv2.morphologyEx(
    mask_enhanced,
    cv2.MORPH_OPEN,
    kernel
)

# Closing 1회
ball_mask_closed = cv2.morphologyEx(
    mask_enhanced,
    cv2.MORPH_CLOSE,
    kernel
)

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(mask_enhanced, cmap="gray")
plt.title("Enhanced Mask")

plt.subplot(1, 3, 2)
plt.imshow(ball_mask_opened, cmap="gray")
plt.title("Mask after Opening")

plt.subplot(1, 3, 3)
plt.imshow(ball_mask_closed, cmap="gray")
plt.title("Mask after Closing")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 6. 최종 Morphological Operation
# ==========================================

# Opening 2회 적용
final_basketball_mask = cv2.morphologyEx(
    mask_enhanced,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


# ==========================================
# 7. Binary Mask에서 Contour 검출
# ==========================================

contours, _ = cv2.findContours(
    final_basketball_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("검출된 Contour 개수:", len(contours))


# ==========================================
# 8. 가장 큰 Contour 선택
# ==========================================

# 원본 RGB 이미지 복사
final_img = basketball_rgb.copy()

# 최종 농구공 마스크 초기화
final_mask = np.zeros_like(final_basketball_mask)

if len(contours) > 0:

    # 가장 큰 Contour 선택
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # 가장 큰 Contour만 채운 마스크 생성
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 초록색 Overlay 이미지 생성
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask

    alpha = 0.35

    final_img = cv2.addWeighted(
        final_img,
        1.0,
        mask_color,
        alpha,
        0
    )

    # 초록색 Contour 표시
    cv2.drawContours(
        final_img,
        [largest_contour],
        -1,
        (0, 255, 0),
        2
    )

    print("가장 큰 Contour 면적:", cv2.contourArea(largest_contour))

else:
    print("검출된 농구공 영역이 없습니다.")


# ==========================================
# 9. 최종 결과 시각화
# ==========================================

# 최종 마스크로 농구공 영역 추출
segmented_basketball = cv2.bitwise_and(
    basketball_rgb,
    basketball_rgb,
    mask=final_mask
)

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.imshow(basketball_rgb)
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(final_basketball_mask, cmap="gray")
plt.title("Opening (5x5, 2 iterations)")

plt.subplot(1, 4, 3)
plt.imshow(final_mask, cmap="gray")
plt.title("Largest Contour Mask")

plt.subplot(1, 4, 4)
plt.imshow(final_img)
plt.title("Final Basketball Contour")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 10. 최종 농구공 추출 결과
# ==========================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(segmented_basketball)
plt.title("Segmented Basketball")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(final_img)
plt.title("Final Result with Contour")
plt.axis("off")

plt.tight_layout()
plt.show()
