# GPU 가속 실습

NumPy와 CuPy를 사용하여 이미지 잔상 효과와 Convolution2D 연산을 구현하고, CPU와 GPU의 처리 속도를 비교하는 과제입니다.

## 1. 이미지 잔상 효과 (Motion Trail)

**코드 1~11번**

이미지를 오른쪽으로 조금씩 이동시키면서 가중치를 곱해 잔상 효과를 구현했습니다. 이동 거리가 멀어질수록 가중치가 작아지도록 설정했습니다.

- `motion_trail_np()`: NumPy와 3중 for문을 사용하여 CPU에서 픽셀별로 계산
- `motion_trail_cp()`: CuPy와 배열 슬라이싱을 사용하여 GPU에서 병렬 연산
- CPU와 GPU의 실행 시간을 측정하고 결과 이미지 비교
- CPU와 GPU 간 데이터 전송 시간도 함께 측정

## 2. Convolution2D 연산

**코드 12번부터 마지막까지**

기존 NumPy Convolution 함수를 CuPy로 구현하고, 같은 이미지에 Edge 검출 필터를 적용하여 CPU와 GPU의 결과를 비교했습니다.

- `convolution2d()`: NumPy를 사용하여 출력 픽셀을 하나씩 계산
- `convolution2d_cp()`: CuPy를 사용하여 커널의 각 원소를 이미지 배열 전체에 적용
- 세로 및 가로 Edge 검출 필터 적용
- ReLU를 적용하여 Feature Map 생성
- CPU와 GPU의 실행 시간 및 연산 결과 비교

## 3. 사용 라이브러리

- Python
- NumPy
- CuPy
- OpenCV
- Matplotlib
- time

GPU 연산에는 CUDA를 지원하는 환경이 필요합니다.

## 4. 파일 구성

```text
261025/
├── gpu_acceleration.py
├── README.md
└── src/
    └── images/
        └── seagull.jpg
```

## 5. 실행 방법

필요한 라이브러리가 설치된 환경에서 다음 명령어를 실행합니다.

```bash
python gpu_acceleration.py
```

입력 이미지는 `src/images/seagull.jpg`를 사용합니다.

실행하면 잔상 효과와 Convolution2D 연산의 CPU/GPU 처리 시간을 출력하고, 각 결과 이미지를 시각화합니다.

## 6. 참고 사항

- NumPy는 CPU 연산, CuPy는 GPU 연산에 사용했습니다.
- GPU 연산 전에는 워밍업을 수행했습니다.
- GPU 처리 시간은 데이터 전송 시간을 포함한 경우와 연산 시간만 측정한 경우로 나누어 비교했습니다.
- CPU는 Python 반복문으로 픽셀을 계산하고, GPU는 배열 단위로 병렬 연산하므로 처리 시간에 차이가 발생할 수 있습니다.