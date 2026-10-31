# Object Detection 실습 (IoU & NMS)

Bounding Box를 이용한 IoU 계산과 NMS 알고리즘을 구현하는 과제입니다.

## 1. Intersection over Union (IoU)

**코드 1~6번**

두 Bounding Box가 얼마나 겹치는지 계산하는 IoU 함수를 구현했습니다.

- `convert_bbox_coord()`: BBox 좌표를 `(x, y, w, h)`에서 `(x1, y1, x2, y2)`로 변환
- `calculate_iou()`: 두 BBox의 교집합과 합집합 면적을 이용하여 IoU 계산
- 합집합 면적이 0인 경우 ZeroDivisionError 방지
- 사과 이미지에서 빨간 사과와 초록 사과의 윤곽선 검출
- `cv2.boundingRect()`로 BBox를 생성하고 이미지별 IoU 비교

## 2. Non-Maximum Suppression (NMS)

**코드 7~14번**

중복된 Bounding Box를 제거하는 NMS 알고리즘을 구현하고, OpenCV 내장 함수와 결과를 비교했습니다.

- `random_bbox()`: 랜덤 Bounding Box 생성
- `random_conf()`: 랜덤 Confidence Score 생성
- `prune_low_conf()`: Confidence Score가 임계값보다 낮은 BBox 제거
- `apply_nms()`: Confidence Score가 높은 BBox부터 선택하고 IoU를 기준으로 중복 BBox 제거
- `apply_nms_cv2()`: OpenCV의 `cv2.dnn.NMSBoxes()`를 이용한 NMS 적용
- 직접 구현한 NMS와 OpenCV NMS의 결과 시각화 및 비교

## 3. 사용 라이브러리

- NumPy
- OpenCV
- Matplotlib
- scikit-image
- random

## 4. 파일 구조

```text
Vision-LLM-202035306/
├── 261031_2/
│   ├── src/
│   │   └── images/
│   │       ├── two_apples1.png
│   │       ├── two_apples2.png
│   │       └── two_apples3.png
│   ├── iou_nms_assignment.py
│   └── README.md
```

사과 이미지는 IoU 실습에 사용하고, NMS 실습에서는 `skimage.data.hubble_deep_field()` 이미지를 사용합니다.

## 5. 실행 방법

`261031_2` 폴더로 이동한 후 다음 명령어를 실행합니다.

```bash
python iou_nms_assignment.py
```

실행하면 사과 이미지의 IoU를 계산하고, 랜덤 Bounding Box에 NMS를 적용한 결과를 시각화합니다.

## 6. 참고 사항

- IoU는 두 Bounding Box의 교집합 면적을 합집합 면적으로 나눈 값입니다.
- Confidence Threshold는 0.5를 사용했습니다.
- NMS 비교에서는 IoU Threshold를 0.2로 설정했습니다.
- 랜덤으로 BBox와 Confidence Score를 생성하므로 실행할 때마다 결과가 달라질 수 있습니다.
- 직접 구현한 NMS와 OpenCV 내장 NMS의 결과를 비교할 수 있습니다.