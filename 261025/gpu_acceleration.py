# ============================================================
# 1. 라이브러리 Import
# ============================================================

import numpy as np
import cupy as cp

import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import time
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 2. 글로벌 변수 정의
# ============================================================

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32


# ============================================================
# 3. 이미지 불러오기 및 전처리
# ============================================================

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");


# ============================================================
# 4. 잔상 효과 함수 정의 (CPU - NumPy)
# ============================================================

# TODO: NumPy를 사용하여 잔상 효과 함수 정의

def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """
    height, width = image.shape
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
          pixel_sum = 0.0

          for shift in range(TRAIL_LENGTH):
            input_x = output_x - shift
                # TODO: x값 shift
            if input_x >= 0:
              pixel_sum += image[output_y, input_x] * weights[shift]

                # TODO: shift한 픽셀을 누적
                # TODO: 출력 배열에 누적 픽셀 저장
          output[output_y, output_x] = pixel_sum
    return output


# ============================================================
# 5. 잔상 효과 함수 정의 (GPU - CuPy)
# ============================================================

# TODO: CuPy를 사용하여 잔상 효과 함수 정의

def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """
    height, width = image.shape
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(TRAIL_LENGTH):
      if shift >= len(weights) or shift >= width:
        break
        # TODO: 배열 slicing
      input_region = image[:, :width-shift]
      output_region = output[:, shift:]

      # TODO: GPU가 병렬 연산할 output_region 계산 코드
      output_region += input_region * weights[shift]

    return output


# ============================================================
# 6. 잔상 효과 가중치 설정 및 정규화
# ============================================================

# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)


# ============================================================
# 7. CPU(NumPy) 잔상 효과 처리 및 시간 측정
# ============================================================

cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start


# ============================================================
# 8. GPU(CuPy) 잔상 효과 처리
# ============================================================

# ------------------------------------------------------------
# 8-1. CPU -> GPU 데이터 전송
# ------------------------------------------------------------

# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start


# ------------------------------------------------------------
# 8-2. GPU 워밍업
# ------------------------------------------------------------

warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()


# ------------------------------------------------------------
# 8-3. GPU 병렬 연산 및 시간 측정
# ------------------------------------------------------------

cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start


# ------------------------------------------------------------
# 8-4. GPU -> CPU 데이터 전송
# ------------------------------------------------------------

# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start


# ============================================================
# 9. 잔상 효과 CPU / GPU 처리 속도 비교
# ============================================================

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time


# ============================================================
# 10. 잔상 효과 CPU / GPU 실행 시간 결과 출력
# ============================================================

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")


# ============================================================
# 11. 잔상 효과 이미지 결과 시각화
# ============================================================

plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")


# ============================================================
# 12. Convolution 실습 - 입력 이미지 확인
# ============================================================

plt.imshow(input_np, cmap="gray")
plt.axis("off");


# ============================================================
# 13. Convolution 실습에 필요한 기존 CPU 함수 및 필터 정의
#     (앞선 Feature Map 실습에서 정의했던 항목 추가)
# ============================================================

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """2차원 이미지에 하나의 CNN 필터를 적용한다."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]
            weighted_sum = np.sum(image_region * kernel)
            output[output_y, output_x] = weighted_sum + bias

    return output


def relu(x):
    return np.maximum(0, x)


dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

dx_bias = -0.2
dy_bias = -0.2


# ============================================================
# 14. Convolution 함수 정의 (GPU - CuPy)
# ============================================================

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
  """
  CuPy를 이용한 2차원 convolution.

  커널의 위치는 Python 반복문으로 순차 처리하고,
  각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
  """
  image_height, image_width = image.shape
  kernel_height, kernel_width = kernel.shape

  # Zero padding
  padded_image = cp.pad(
      image,
      ((padding, padding), (padding, padding)),
      mode="constant",
      constant_values=0
  )

  # 출력 이미지 크기 계산
  output_height = (padded_image.shape[0] - kernel_height) // stride + 1
  output_width = (padded_image.shape[1] - kernel_width) // stride + 1

  # 출력 배열 생성
  output = cp.zeros((output_height, output_width), dtype=cp.float32)

  # 커널의 각 원소를 순차적으로 처리
  for kernel_y in range(kernel_height):
    for kernel_x in range(kernel_width):
      # 현재 커널 위치에 대응하는 입력 이미지 영역
      input_region = padded_image[
          kernel_y:kernel_y + output_height * stride:stride,
          kernel_x:kernel_x + output_width * stride:stride
      ]
      # GPU 병렬 연산
      output += input_region * kernel[kernel_y, kernel_x]

  output += bias
  return output


# ============================================================
# 15. Convolution CPU 처리 및 시간 측정
# ============================================================

# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능
cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)
cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)


# ============================================================
# 16. Convolution GPU 처리
# ============================================================

# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

# ------------------------------------------------------------
# 16-1. NumPy 배열을 CuPy 배열로 변환
# ------------------------------------------------------------

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()
upload_time = upload_end - upload_start


# ------------------------------------------------------------
# 16-2. GPU 워밍업
# ------------------------------------------------------------

warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(
    image=warmup_image,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()


# ------------------------------------------------------------
# 16-3. GPU Convolution 처리 및 시간 측정
# ------------------------------------------------------------

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start


# ------------------------------------------------------------
# 16-4. ReLU 적용 및 GPU -> CPU 변환
# ------------------------------------------------------------

download_start = time.perf_counter()

relu_dx_cp_np = cp.asnumpy(cp.maximum(output_dx_cp, 0))
relu_dy_cp_np = cp.asnumpy(cp.maximum(output_dy_cp, 0))

download_end = time.perf_counter()
download_time = download_end - download_start


# ============================================================
# 17. Convolution CPU / GPU 처리 속도 비교 및 결과 출력
# ============================================================

# TODO: CPU/GPU 결과 비교 및 출력

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
print()




# ============================================================
# 18. Convolution 이미지 결과 시각화
# ============================================================

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()
