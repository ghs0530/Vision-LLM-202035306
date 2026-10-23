"""손동작으로 크기를 조절하는 2D 도자기 프로그램.

초록색 공의 크기로 시작과 종료를 판단하고, 두 손의 움직임으로 도자기의 폭과 높이를 바꾼다.
양손을 주먹 쥐면 편집을 멈추고, 다시 펴면 현재 위치를 기준으로 편집을 재개한다.

조작 방법
- 큰 초록색 공: 게임 시작
- 양손 펼치기: 손 사이 거리로 폭, 손의 평균 세로 위치로 높이 조절
- 양손 주먹: 일시정지 / 다시 펼치기: 기준 위치 갱신 후 재개
- 작은 초록색 공: 도자기 이미지 저장 후 종료
- S: 직접 저장 후 종료 / Q 또는 ESC: 저장 없이 종료
"""

import math
import time
from datetime import datetime
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import mediapipe as mp
import numpy as np

# -------------------------- 설정값 --------------------------
# 집 Windows 환경에서는 기본 웹캠(0번)을 사용합니다.
# 공 검출과 손 제스처 판별은 카메라 종류와 관계없이 같은 방식으로 처리합니다.
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CAMERA_INDEX = 0
# 모델은 현재 .py 파일이 있는 Project01/src/models/MediaPipe/ 안에 둡니다.
MODEL_RELATIVE_PATH = Path('src/models/MediaPipe/hand_landmarker.task')
START_BALL_RADIUS = 65                   # 640x480 카메라 영상 기준 픽셀 단위
FINISH_BALL_MIN_RADIUS = 10                 # 종료 반지름의 하한값(포함); 공이 검출되지 않으면 종료하지 않음
FINISH_BALL_MAX_RADIUS = 23                 # 종료 반지름의 상한값(포함)
# 단일 프레임에서 우연히 공이 검출되어 시작/종료되는 것을 방지하는 유지 시간(초)
START_HOLD_SECONDS = 0.55
END_HOLD_SECONDS = 1.20
END_COOLDOWN_SECONDS = 2.0          # 시작 직후 바로 종료되는 현상 방지
# 주먹/펼침 모양도 일정 시간 유지되어야 상태가 전환됩니다.
GESTURE_HOLD_SECONDS = 0.25

# LAB 색공간에서 초록 공에 해당하는 색상 범위입니다.
GREEN_LOWER_LAB = np.array([30, 60, 90], dtype=np.uint8)
GREEN_UPPER_LAB = np.array([230, 115, 180], dtype=np.uint8)
# 면적이 너무 작거나 원형에 가깝지 않은 영역은 제외합니다.
MIN_CONTOUR_AREA = 100
MIN_CIRCULARITY = 0.58

# 도자기의 처음 크기, 최소/최대 크기(픽셀)
POT_INITIAL_WIDTH = 170
POT_INITIAL_HEIGHT = 240
POT_MIN_WIDTH, POT_MAX_WIDTH = 70, 380
POT_MIN_HEIGHT, POT_MAX_HEIGHT = 95, 385
WIDTH_CHANGE_SPEED = 1.0            # 양손 좌우 간격 1px 변화당 도자기 폭 변화량
HEIGHT_CHANGE_SPEED = 1.0           # 양손을 함께 위로 움직일수록 도자기 높이 증가
SIZE_SMOOTHING_FACTOR = 0.35                 # 0~1 사이의 부드러움 계수: 클수록 빠르게 반응

# OpenCV에서 사용하는 BGR 색상값
POT_FILL_COLOR = (65, 190, 45)
POT_OUTLINE_COLOR = (25, 110, 30)
CANVAS_BACKGROUND_COLOR = (246, 244, 239)

# 손 랜드마크 연결선과 관절 각도 계산에 사용할 인덱스입니다.
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_ANGLE_POINTS = ((1, 2, 3), (5, 6, 7), (9, 10, 11),
                       (13, 14, 15), (17, 18, 19))
FINGER_STRAIGHT_ANGLE = 160.0  # 펼친 손가락 기준 각도


def find_model_path():
    """현재 .py 파일을 기준으로 손 인식 모델의 경로를 확인합니다."""
    project_folder = Path(__file__).resolve().parent
    model_path = project_folder / MODEL_RELATIVE_PATH

    if not model_path.is_file():
        raise FileNotFoundError(
            f'손 인식 모델을 찾지 못했습니다: {model_path}\n'
            'Project01/src/models/MediaPipe/hand_landmarker.task 위치를 확인하세요.'
        )
    return str(model_path)


def calculate_joint_angle(p1, p2, p3):
    """세 랜드마크로 이루어진 관절의 3차원 각도를 계산합니다."""
    vector1 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
    vector2 = np.array([p3.x-p2.x, p3.y-p2.y, p3.z-p2.z])
    first_vector_length, second_vector_length = np.linalg.norm(vector1), np.linalg.norm(vector2)
    if first_vector_length == 0 or second_vector_length == 0:
        return 0.0
    cosine_angle = np.clip(np.dot(vector1, vector2) / (first_vector_length * second_vector_length), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


# 계산한 도자기 크기가 최소·최대 범위를 넘지 않도록 제한합니다.
# 예: limit_range(500, 70, 380) -> 380
def limit_range(value, low, high):
    return max(low, min(high, value))


def enhance_frame(bgr):
    """LAB 색공간의 CLAHE를 사용하여 어둡거나 대비가 낮은 영상을 개선합니다."""
    # LAB의 L은 밝기, a/b는 색상 성분입니다. 밝기만 보정하여 색 왜곡을 줄입니다.
    lab_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness_channel, a_channel, b_channel = cv2.split(lab_frame)
    # CLAHE: 영상의 작은 영역마다 명암 대비를 보정하되 지나친 대비 증폭을 억제합니다.
    lightness_channel = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(lightness_channel)
    return cv2.cvtColor(cv2.merge((lightness_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def detect_green_ball(frame):
    """LAB 색상 분할과 윤곽선 검출로 공의 중심과 반지름을 구합니다.

0   색상과 원형 조건을 만족하는 물체가 없으면 None을 반환합니다.
    """
    # 이미 CLAHE를 적용한 영상이 아니라 원본 BGR 영상을 입력받습니다.
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness_channel, a_channel, b_channel = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge((clahe.apply(lightness_channel), a_channel, b_channel))
    shape_mask = cv2.inRange(enhanced_lab, GREEN_LOWER_LAB, GREEN_UPPER_LAB)
    detected_contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 컨투어를 기준으로 원을 맞춥니다.
    if not detected_contours:
        return None
    largest_contour = max(detected_contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < MIN_CONTOUR_AREA:
        return None
    contour_perimeter = cv2.arcLength(largest_contour, True)
    contour_circularity = (4 * math.pi * contour_area / (contour_perimeter ** 2)) if contour_perimeter else 0.0
    if contour_circularity < MIN_CIRCULARITY:
        return None
    (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(circle_center_x), int(circle_center_y), float(radius)) if radius > 0 else None


def get_hand_position_and_gesture(hand_landmarks, frame_width, frame_height):
    """Tasks API의 랜드마크 리스트를 받아 손 중심/제스처를 판정합니다.

    손가락 관절 각도가 160도 이상이면 펴진 것으로 판단합니다.
    주먹/펼침은 엄지를 제외한 4개 손가락으로 판정하여 엄지의 방향 오판을 줄입니다.
    """
    straight_finger_count = sum(calculate_joint_angle(hand_landmarks[i], hand_landmarks[j], hand_landmarks[k]) >= FINGER_STRAIGHT_ANGLE
                   for i, j, k in FINGER_ANGLE_POINTS[1:])
    hand_gesture = 'OPEN' if straight_finger_count == 4 else 'FIST' if straight_finger_count <= 1 else 'OTHER'
    palm_landmark_indices = (0, 5, 9, 13, 17)
    center_x = float(np.mean([hand_landmarks[i].x for i in palm_landmark_indices]) * frame_width)
    center_y = float(np.mean([hand_landmarks[i].y for i in palm_landmark_indices]) * frame_height)
    return (center_x, center_y), hand_gesture


def draw_hand_landmarks(frame, hand_landmarks):
    """손의 관절 연결선과 랜드마크를 영상에 표시합니다."""
    frame_height, frame_width = frame.shape[:2]
    points = [(int(p.x*frame_width), int(p.y*frame_height)) for p in hand_landmarks]
    for link in HAND_CONNECTIONS:
        cv2.line(frame, points[link.start], points[link.end], (0, 255, 0), 2)
    for i, point in enumerate(points):
        color = (0, 0, 255) if i in FINGER_TIPS else (255, 0, 0)
        cv2.circle(frame, point, 6 if i in FINGER_TIPS else 4, color, -1)


def get_two_hand_measurements(detected_hands):
    # 실제 왼손·오른손 구분이 아니라 화면상의 왼쪽·오른쪽 순서만 사용
    # 좌우가 바뀌어 감지되더라도 화면상의 x좌표로 두 손을 정렬합니다.
    hands_sorted_by_x = sorted(detected_hands, key=lambda item: item[0][0])
    (left_hand_x, left_hand_y), (right_hand_x, right_hand_y) = hands_sorted_by_x[0][0], hands_sorted_by_x[1][0]
    # 두 손의 가로 거리로 도자기 폭을 조절합니다.
    # 두 손 중심의 평균 y좌표로 도자기 높이를 조절합니다.
    # 영상의 y좌표는 아래쪽으로 갈수록 커집니다.
    return abs(right_hand_x - left_hand_x), (left_hand_y + right_hand_y) / 2.0


def create_pottery_mask(frame_width, frame_height, pot_width, pot_height):
    """폭과 높이, 두 매개변수로만 크기가 변하는 좌우 대칭 도자기 마스크를 만듭니다."""
    pot_width = int(pot_width)
    pot_height = int(pot_height)
    # 도자기의 중심 x좌표는 화면 중앙, 바닥은 화면 높이의 90% 지점입니다.
    center_x, bottom = frame_width // 2, int(frame_height * 0.90)
    top = bottom - pot_height
    # 도자기 윤곽선: (위에서 아래로의 높이 비율, 전체 폭에 대한 반쪽 너비 비율)
    pottery_profile = [
        (0.00, 0.32), (0.035, 0.34), (0.06, 0.26),
        (0.19, 0.25), (0.28, 0.35), (0.38, 0.47),
        (0.51, 0.50), (0.64, 0.47), (0.78, 0.39),
        (0.93, 0.31), (1.00, 0.30),
    ]
    # 도자기 오른쪽 윤곽선을 먼저 계산하고, 같은 점들을 좌우 반전해 왼쪽도 만듭니다.
    # 형태의 비율은 유지하고 전체 폭과 높이만 조절합니다.
    right_outline_points = [(int(center_x + half * pot_width), int(top + t * pot_height)) for t, half in pottery_profile]
    left_outline_points = [(int(center_x - half * pot_width), int(top + t * pot_height)) for t, half in reversed(pottery_profile)]
    points = np.array(right_outline_points + left_outline_points, dtype=np.int32)
    shape_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    # 다각형 내부를 흰색(255)으로 채운 뒤, 그 마스크를 실제 색 채우기에 사용합니다.
    cv2.fillPoly(shape_mask, [points], 255)
    return shape_mask, top, bottom, center_x


def draw_pottery_image(pot_width, pot_height, canvas_size=(480, 640), clean=False):
    frame_height, frame_width = canvas_size
    canvas = np.full((frame_height, frame_width, 3), CANVAS_BACKGROUND_COLOR, dtype=np.uint8)
    shape_mask, top, bottom, center_x = create_pottery_mask(frame_width, frame_height, pot_width, pot_height)
    # 배경 중 마스크가 켜진 위치만 초록색으로 칠합니다.
    canvas[shape_mask > 0] = POT_FILL_COLOR
    detected_contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, detected_contours, -1, POT_OUTLINE_COLOR, 2, cv2.LINE_AA)
    # 도자기 입구를 타원으로 그려 형태를 더 자연스럽게 표현
    lip_w = int(pot_width * 0.67)
    cv2.ellipse(canvas, (center_x, top), (max(4, lip_w // 2), 8), 0, 0, 360, POT_OUTLINE_COLOR, 2, cv2.LINE_AA)
    # clean=True일 때는 설명 글자를 제외한 도자기 그림만 저장할 수 있습니다.
    if not clean:
        cv2.putText(canvas, '2D POTTERY', (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 60, 50), 2, cv2.LINE_AA)
        cv2.putText(canvas, f'Width  : {int(pot_width)} px', (20, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Height : {int(pot_height)} px', (20, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (65, 65, 65), 1, cv2.LINE_AA)
    return canvas


def save_pottery_image(pot_width, pot_height):
    # 실행 파일이 있는 위치를 기준으로 저장 폴더를 만듭니다.
    output_folder = Path(__file__).resolve().parent / 'pottery_output'
    output_folder.mkdir(parents=True, exist_ok=True)
    # 실행 시각으로 파일명을 생성합니다. 같은 초에 반복 저장하면 이름이 겹칠 수 있습니다.
    output_path = output_folder / f'pottery_{datetime.now():%Y%m%d_%H%M%S}.png'
    pottery_image = draw_pottery_image(pot_width, pot_height, clean=True)
    if not cv2.imwrite(str(output_path), pottery_image):
        raise OSError(f'Failed to write image: {output_path}')
    return output_path


def check_hold_duration(is_condition_met, condition_started_at, current_time, required_seconds):
    """조건이 연속으로 유지된 시간을 검사하고 (시작 시각, 완료 여부, 진행률)을 반환합니다."""
    # 조건이 끊기는 순간 시작 시각을 지워 다음 유지 시간을 처음부터 셉니다.
    if not is_condition_met:
        return None, False, 0.0
    if condition_started_at is None:
        condition_started_at = current_time
    # 진행률은 0~1 사이로 계산되어 화면의 초록색 진행 막대에 사용됩니다.
    hold_progress = limit_range((current_time - condition_started_at) / required_seconds, 0, 1)
    return condition_started_at, hold_progress >= 1.0, hold_progress


def main():
    # 카메라 번호가 0이 아닌 경우 CAMERA_INDEX를 변경합니다.
    camera_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if camera_capture.isOpened():
        # 처리에 사용할 카메라 해상도를 지정합니다.
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera_capture.isOpened():
        camera_capture.release()
        raise RuntimeError(
            'Windows 웹캠을 열지 못했습니다. CAMERA_INDEX와 카메라 권한을 확인하세요.'
        )
    # 모델 경로를 확인하고 최대 두 손을 인식하는 객체를 생성합니다.
    try:
        hand_model_path = find_model_path()
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    except Exception:
        camera_capture.release()
        raise

    # 상태 머신: WAIT_START(시작 전) → ACTIVE(편집) ↔ PAUSED(일시정지)
    state = 'WAIT_START'        # 시작 대기 / 도자기 조절 / 일시정지
    pot_width, pot_height = float(POT_INITIAL_WIDTH), float(POT_INITIAL_HEIGHT)
    # 상대 제어를 시작한 순간의 도자기 크기와 손 위치를 저장하는 기준값입니다.
    # 편집을 다시 시작할 때 기준값을 갱신하므로 크기가 갑자기 튀지 않습니다.
    start_pot_width = start_pot_height = 0.0
    start_hand_distance = start_hand_y = None
    # True이면 이번에 양손이 펴졌을 때 새로운 상대 제어 기준을 한 번 설정합니다.
    reset_start_position = True
    # 시작/종료/주먹/펼침 각각의 "연속 유지 시작 시각"을 따로 관리합니다.
    start_condition_since = finish_condition_since = fist_condition_since = open_condition_since = None
    game_started_at = 0.0

    try:
        # 웹캠에서 프레임을 읽고 분석/그리기를 반복합니다.
        while True:
            frame_received, camera_frame = camera_capture.read()
            if not frame_received:
                print('[ERROR] 카메라 영상을 읽지 못했습니다.\n')
                break
            # 영상 크기를 일정하게 맞춘 다음 좌우를 뒤집어 거울처럼 표시합니다.
            # flip(..., 1)은 좌우 반전으로, 사용자가 손을 움직이는 방향과 화면 방향을 맞춥니다.
            camera_frame = cv2.flip(cv2.resize(camera_frame, (FRAME_WIDTH, FRAME_HEIGHT)), 1)
            # 실제 카메라 영상에서만 공을 찾으므로 화면의 도자기 그림은 검출 대상이 아닙니다.
            # LAB 변환과 CLAHE 보정, 윤곽선 검출은 detect_green_ball() 내부에서 수행합니다.
            detected_ball = detect_green_ball(camera_frame)
            # 시스템 시각 변경의 영향을 받지 않는 단조 증가 시간으로 유지 시간을 측정합니다.
            current_time = time.monotonic()

            # OpenCV의 BGR 영상을 RGB로 바꾼 뒤 MediaPipe에 전달합니다.
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            hand_detection_result = hand_landmarker.detect(mp_image)
            detected_hands = []
            # 인식된 각 손은 21개의 정규화된 랜드마크 좌표를 가집니다.
            if hand_detection_result.hand_landmarks:
                for hand_landmarks in hand_detection_result.hand_landmarks:
                    detected_hands.append(get_hand_position_and_gesture(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT))
                    draw_hand_landmarks(camera_frame, hand_landmarks)
                    (hand_center_x, hand_center_y), hand_gesture = detected_hands[-1]
                    cv2.putText(camera_frame, hand_gesture, (int(hand_center_x) - 28, int(hand_center_y) - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 240, 240), 2)
                    cv2.circle(camera_frame, (int(hand_center_x), int(hand_center_y)), 5, (255, 220, 50), -1)

            if detected_ball is not None:
                ball_center_x, ball_center_y, ball_radius = detected_ball
                cv2.circle(camera_frame, (ball_center_x, ball_center_y), int(ball_radius), (0, 255, 255), 2)
                cv2.circle(camera_frame, (ball_center_x, ball_center_y), 3, (0, 255, 255), -1)
                cv2.putText(camera_frame, f'GREEN r={ball_radius:.1f}', (max(5, ball_center_x - 55), max(25, ball_center_y - int(ball_radius) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # [상태 1] 시작 대기: 큰 초록 공을 일정 시간 보여줘야 게임 시작
            if state == 'WAIT_START':
                start_ball_detected = detected_ball is not None and detected_ball[2] >= START_BALL_RADIUS
                start_condition_since, start_hold_completed, hold_progress = check_hold_duration(
                    start_ball_detected, start_condition_since, current_time, START_HOLD_SECONDS)
                # 한 프레임만 커 보인다고 시작하지 않고 연속 유지 시간을 확인합니다.
                if start_hold_completed:
                    state = 'ACTIVE'
                    game_started_at = current_time
                    reset_start_position = True
                    start_condition_since = None
                    print('[START] 초록색 공 - 게임 시작\n')
            else:
                # [상태 2~3] 게임 중: 작은 공 종료 판정이 손 제스처 판정보다 우선합니다.
                # 실제 검출된 원형 초록 공의 반지름이 하한값과 상한값 사이일 때만 종료 후보로 판정
                # 공이 보이지 않으면 종료 타이머를 초기화하며 게임은 종료하지 않음
                finish_cooldown_passed = current_time - game_started_at >= END_COOLDOWN_SECONDS
                finish_ball_detected = (detected_ball is not None and
                              FINISH_BALL_MIN_RADIUS <= detected_ball[2] <= FINISH_BALL_MAX_RADIUS)
                finish_condition_since, finish_hold_completed, hold_progress = check_hold_duration(
                    finish_cooldown_passed and finish_ball_detected, finish_condition_since, current_time, END_HOLD_SECONDS)
                if finish_hold_completed:
                    saved_file_path = save_pottery_image(pot_width, pot_height)
                    print(f'[SAVE] 이미지 저장 완료: {saved_file_path.name}\n')
                    break

                # 양손 모두 같은 제스처일 때만 일시정지/재개 조건을 인정합니다.
                both_hands_fists = len(detected_hands) == 2 and all(g == 'FIST' for _, g in detected_hands)
                both_hands_open = len(detected_hands) == 2 and all(g == 'OPEN' for _, g in detected_hands)
                fist_condition_since, fists_held_long_enough, _ = check_hold_duration(
                    both_hands_fists, fist_condition_since, current_time, GESTURE_HOLD_SECONDS)
                open_condition_since, open_held_long_enough, _ = check_hold_duration(
                    both_hands_open, open_condition_since, current_time, GESTURE_HOLD_SECONDS)

                # ACTIVE에서 주먹을 일정 시간 유지하면 PAUSED로 전환
                # PAUSED에서 펼침을 유지하면 ACTIVE로 복귀하고 상대 제어 기준을 초기화
                if state == 'ACTIVE' and fists_held_long_enough:
                    state = 'PAUSED'
                    reset_start_position = True
                    print('[PAUSE] 양손 주먹 - 도자기 수정 중지\n')
                elif state == 'PAUSED' and open_held_long_enough:
                    state = 'ACTIVE'
                    reset_start_position = True
                    print('[RESUME] 양손 펼침 - 도자기 수정 재개\n')

                if state == 'ACTIVE':
                    if not both_hands_open:
                        # 손이 사라지거나 펼친 손 제스처가 풀리면 도자기 크기를 유지
                        # 다시 양손을 펼쳤을 때 상대 조절의 기준점을 새로 기록
                        reset_start_position = True
                    else:
                        # 현재 두 손의 가로 간격과 평균 세로 위치를 계산합니다.
                        hand_horizontal_distance, hands_average_y = get_two_hand_measurements(detected_hands)
                        if reset_start_position:
                            # 재개 직후 현재 도자기 크기 + 손 위치를 기준점으로 잡습니다.
                            # 이 프레임에는 크기를 변화시키지 않아 순간적인 점프를 방지합니다.
                            start_pot_width, start_pot_height = pot_width, pot_height
                            start_hand_distance, start_hand_y = hand_horizontal_distance, hands_average_y
                            reset_start_position = False
                        else:
                            # 상대 폭 = 기준 도자기 폭 + (현재 손 간격 - 시작 손 간격)
                            # 상대 높이 = 기준 도자기 높이 + (기준 평균 y - 현재 평균 y)
                            # 위로 올리면 y가 작아지므로 높이는 증가합니다.
                            desired_pot_width = limit_range(start_pot_width + (hand_horizontal_distance - start_hand_distance) * WIDTH_CHANGE_SPEED,
                                             POT_MIN_WIDTH, POT_MAX_WIDTH)
                            desired_pot_height = limit_range(start_pot_height + (start_hand_y - hands_average_y) * HEIGHT_CHANGE_SPEED,
                                             POT_MIN_HEIGHT, POT_MAX_HEIGHT)
                            # 목표 크기로 즉시 순간이동하지 않고 일부씩 가까워지게 하여 떨림을 줄입니다.
                            pot_width += SIZE_SMOOTHING_FACTOR * (desired_pot_width - pot_width)
                            pot_height += SIZE_SMOOTHING_FACTOR * (desired_pot_height - pot_height)

            # 카메라 상태 표시 영역(도자기 그림은 오른쪽의 별도 패널에 출력)
            cv2.rectangle(camera_frame, (0, 0), (FRAME_WIDTH, 88), (18, 22, 24), -1)
            cv2.putText(camera_frame, f'MODE: {state}', (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 240, 255), 2)
            if state == 'WAIT_START':
                instruction_text = f'Show BIG green ball (r >= {START_BALL_RADIUS}) to START'
                progress_fraction = hold_progress if start_condition_since is not None else 0.0
            else:
                instruction_text = f'Show SMALL green ball ({FINISH_BALL_MIN_RADIUS} <= r <= {FINISH_BALL_MAX_RADIUS}) to SAVE + EXIT'
                progress_fraction = hold_progress if finish_condition_since is not None else 0.0
            cv2.putText(camera_frame, instruction_text, (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.rectangle(camera_frame, (12, 68), (332, 77), (70, 70, 70), -1)
            cv2.rectangle(camera_frame, (12, 68), (12 + int(320 * progress_fraction), 77), (40, 220, 90), -1)
            cv2.putText(camera_frame, 'BOTH OPEN: edit | BOTH FISTS: pause | S: save | Q: quit',
                        (10, FRAME_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                        (250, 250, 250), 1, cv2.LINE_AA)

            pottery_panel = draw_pottery_image(pot_width, pot_height, canvas_size=(FRAME_HEIGHT, FRAME_WIDTH))
            cv2.putText(pottery_panel, f'STATE: {state}', (20, FRAME_HEIGHT - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.64, (60, 60, 60), 2)
            # 카메라 영상(좌측)과 도자기 결과(우측)를 하나의 창에 나란히 출력합니다.
            combined_screen = np.hstack((camera_frame, pottery_panel))
            cv2.imshow('Hand Pottery Studio', combined_screen)
            # 키 입력을 매 프레임 확인합니다. q/ESC: 저장 없이 종료, s: 저장 후 종료
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in (ord('q'), 27):
                print('[EXIT] 저장 없이 종료\n')
                break
            if pressed_key == ord('s') and state != 'WAIT_START':
                saved_file_path = save_pottery_image(pot_width, pot_height)
                print(f'[SAVE] 이미지 저장 완료: {saved_file_path.name} (수동 저장)\n')
                break
    finally:
        # 오류/종료 시 MediaPipe 객체, 카메라, OpenCV 창을 정리합니다.
        hand_landmarker.close()
        camera_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
