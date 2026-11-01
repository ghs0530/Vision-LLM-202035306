# CIFAR-10 CNN 이미지 분류

PyTorch를 사용해 CIFAR-10 데이터셋을 학습하는 CNN 모델을 구현했습니다.

Convolution, Batch Normalization, ReLU, Max Pooling을 활용해 이미지의 특징을 추출하고, 총 10개의 클래스로 분류하도록 구성했습니다.

## 모델 구조

3개의 ConvBlock을 사용했으며, 각 블록은 다음과 같이 구성했습니다.

`Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool2d`

전체 모델 구조는 다음과 같습니다.

```text
Input (3 × 32 × 32)
    ↓
ConvBlock (64)
    ↓
ConvBlock (128)
    ↓
ConvBlock (256)
    ↓
Global Average Pooling
    ↓
Flatten
    ↓
Dropout (0.3)
    ↓
Linear (256 → 10)
```

## 학습 설정

- Dataset: CIFAR-10
- Framework: PyTorch
- Optimizer: AdamW
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Epochs: 5

## 모델 파일

학습한 모델은 `.pth` 파일로 저장했습니다.

```text
src/
└── models/
    └── CIFAR10/
        └── CIFAR10_CNN.pth
```

`cifar_model.py`에는 데이터셋 불러오기부터 모델 학습, 평가, 저장 및 테스트까지의 코드가 포함되어 있습니다.

이미 학습된 모델을 사용하려면 `src/models/CIFAR10/CIFAR10_CNN.pth` 파일을 불러오면 됩니다.

```python
checkpoint = torch.load(
    "src/models/CIFAR10/CIFAR10_CNN.pth",
    map_location="cpu",
    weights_only=True
)

model = CIFAR10CNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

모델 구조는 `cifar_model.py`의 `CIFAR10CNN` 클래스를 참고하면 됩니다.

학습된 가중치를 불러온 뒤 CIFAR-10 테스트 이미지로 분류 결과를 확인할 수 있습니다. 전체 코드를 실행하면 모델 학습부터 진행되므로, 기존 모델만 테스트할 경우에는 모델 정의와 불러오기, 테스트 부분을 사용하면 됩니다.

## 실행 환경

```bash
pip install torch torchvision numpy matplotlib
```

CIFAR-10 데이터셋은 코드 실행 시 자동으로 다운로드되도록 설정했습니다.

모델 학습에는 CUDA GPU를 사용했으며, 저장된 모델을 불러와 테스트하는 것은 CPU에서도 가능합니다.