# MNIST 데이터셋 GPU 가속 학습

PyTorch를 사용하여 MNIST 손글씨 분류 모델을 CPU와 GPU에서 각각 학습하고, 학습 시간을 비교하는 과제입니다.

## 1. GPU 기반 MNIST 학습

기존 CPU 학습 코드를 바탕으로 `main_gpu()` 함수를 작성하여 GPU에서 MNIST 데이터셋을 학습하도록 구현했습니다.

- `torch.device("cuda")`로 GPU 장치 설정
- `model.to(device)`로 모델을 GPU로 이동
- `images.to(device)`, `labels.to(device)`로 데이터 이동
- `torch.cuda.synchronize()`로 GPU 연산 완료 후 시간 측정
- Epoch별 Loss, Accuracy 및 학습 시간 출력

## 2. CPU/GPU 학습 결과 비교

CPU와 GPU에서 동일한 CNN 모델을 3 Epoch 동안 학습하고 전체 학습 시간을 비교합니다.

- CPU 및 GPU 학습 시간
- 학습 시간 차이
- GPU 학습 속도 향상 비율

## 3. 사용 라이브러리

- PyTorch
- torchvision
- NumPy
- time
- random

## 4. 파일 구조

```text
261031/
├── mnist_gpu_training.py
├── README.md
└── src/
    └── datasets/
        └── MNIST/     # 실행 시 다운로드
```

MNIST 데이터셋은 코드 실행 시 `src/datasets` 경로에 자동으로 다운로드됩니다.

## 5. 실행 방법

`261031` 폴더에서 다음 명령어를 실행합니다.

```bash
python mnist_gpu_training.py
```

실행하면 CPU 학습을 먼저 수행한 뒤 GPU 학습을 진행하고, 학습 시간과 속도 향상 비율을 출력합니다.

## 6. 참고 사항

- Batch Size: 64
- Test Batch Size: 1000
- Epochs: 3
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- GPU 학습에는 CUDA 환경이 필요합니다.