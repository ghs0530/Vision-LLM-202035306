"""VoiceSort: 음성 명령 기반 실시간 객체 분류 안내 시스템.

Project02 폴더에서 실행:
    python VoiceSort_Final.py
    python VoiceSort_Final.py --vision-only
    python VoiceSort_Final.py --no-mirror

R: 음성으로 분류 규칙 변경 / Q 또는 Esc: 종료
"""

# 1. 필요한 라이브러리 불러오기
import argparse
import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO
from llama_cpp import Llama


# 2. 모델 및 프로그램 경로 (Project02 폴더 기준 상대경로)
YOLO_MODEL = Path("src/models/YOLO/yolo11n_int8.engine")
GEMMA_MODEL = Path("src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
WHISPER_CLI = Path("whisper.cpp/build-cpu/bin/whisper-cli")
WHISPER_MODEL = Path("whisper.cpp/models/ggml-base.bin")
PIPER_PYTHON = Path(".piper_venv/bin/python")
PIPER_MODEL = Path("src/models/Piper/ko_KR-kss-medium.onnx")
AUDIO_DIR = Path("src/audio")
INPUT_WAV = AUDIO_DIR / "input.wav"
OUTPUT_WAV = AUDIO_DIR / "response.wav"

# 3. USB 카메라 영상 입력을 위한 GStreamer 파이프라인
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

MIC_DEVICE = "plughw:2,0"  # 마이크 카드 번호 2, 장치 번호 0
# 4. 객체 감지 / 음성 안내에 필요한 기본 설정
RECORD_SECONDS = 5
TARGET_CLASSES = {39: "bottle", 67: "cell phone"}  # COCO: bottle=39, cell phone=67
KOREAN_NAMES = {"bottle": "페트병", "cell phone": "핸드폰"}
DIRECTIONS = {"left": "왼쪽", "right": "오른쪽"}
CONFIDENCE = 0.60
IMAGE_SIZE = 640  # TensorRT INT8 엔진의 입력 크기
CENTER_RECT = (0.28, 0.22, 0.72, 0.78)
STABLE_WINDOW = 10
STABLE_COUNT = 7
LOST_SECONDS = 0.9


# 5. 실행 전에 필요한 모델과 프로그램 파일 확인
def verify_environment(vision_only):
    files = {"YOLO INT8 engine": YOLO_MODEL}
    if not vision_only:
        files.update({
            "Gemma GGUF": GEMMA_MODEL,
            "Whisper 실행 파일": WHISPER_CLI,
            "Whisper base model": WHISPER_MODEL,
            "Piper virtualenv Python": PIPER_PYTHON,
            "Piper Korean model": PIPER_MODEL,
            "Piper model config": Path(str(PIPER_MODEL) + ".json"),
        })
    missing = [(name, path) for name, path in files.items() if not path.is_file()]
    if missing:
        for name, path in missing:
            print("[MISSING] {}: {}".format(name, path))
        raise FileNotFoundError("Required course file(s) missing. Project02 폴더에서 실행 중인지 확인하세요.")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# 6. STT: 마이크 음성을 녹음하고 Whisper.cpp로 한국어 인식
def speech_to_text():
    print("[STT] 녹음 시작: {}초".format(RECORD_SECONDS))
    # 16kHz, 모노, 16비트 PCM 형식으로 5초간 녹음
    subprocess.run([
        "pasuspender", "--", "arecord", "-D", MIC_DEVICE,
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(RECORD_SECONDS), str(INPUT_WAV)
    ], check=True)
    result = subprocess.run([
        str(WHISPER_CLI), "-m", str(WHISPER_MODEL),
        "-f", str(INPUT_WAV), "-l", "ko", "--no-timestamps"
    ], text=True, capture_output=True, check=True)
    transcript = " ".join(line.strip() for line in result.stdout.splitlines()
                          if line.strip() and not line.startswith("whisper_"))
    print("[STT]", transcript)
    return transcript


# 7. TTS: Piper로 한국어 음성을 생성하고 aplay로 재생
class Speaker:
    """음성 안내 중복 방지; R 키 안내가 끝나야 음성 녹음을 시작함."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.busy = False
        self.lock = threading.Lock()

    def speak(self, text, wait=False):
        print("[TTS]", text)
        if not self.enabled:
            return True
        with self.lock:
            if self.busy:
                print("[TTS] Previous speech still playing; request skipped.")
                return False
            self.busy = True
        if wait:
            return self._run(text)
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()
        return True

    def _run(self, text):
        success = False
        try:
            # Piper 가상환경의 실행 파일을 사용해 음성 파일 생성
            subprocess.run([
                str(PIPER_PYTHON), "-m", "piper", "-m", str(PIPER_MODEL),
                "-f", str(OUTPUT_WAV), "--", text
            ], check=True)
            subprocess.run(["aplay", str(OUTPUT_WAV)], check=True)
            success = True
        except (OSError, subprocess.CalledProcessError) as exc:
            print("[TTS ERROR]", exc)
        finally:
            with self.lock:
                self.busy = False
        return success


# 8. Gemma: 사용자의 분류 규칙 명령을 JSON으로 해석 (YOLO 매 프레임에는 호출하지 않음)
class GemmaParser:
    def __init__(self):
        print("[Gemma] 모델 로딩:", GEMMA_MODEL)
        # GGUF 모델을 한 번 로딩하고 사용 가능한 GPU 레이어를 오프로딩
        self.llm = Llama(model_path=str(GEMMA_MODEL), n_gpu_layers=-1,
                        n_ctx=2048, n_batch=32, n_ubatch=32, verbose=False)

    def parse(self, command, current_rules):
        prompt = (
            "너는 음성으로 말한 물건 분류 규칙을 JSON으로 변환한다. "
            "물건은 페트병(bottle)과 핸드폰(cell phone)만 가능하다. "
            "방향은 left(왼쪽), right(오른쪽)만 가능하다. "
            "명령에서 명확하게 지정된 물건만 결과에 넣는다. "
            "명령이 불명확하면 {}를 출력한다. 설명 없이 JSON 객체만 출력한다.\n"
            "입력 예시: 페트병은 오른쪽, 핸드폰은 왼쪽으로 보내\n"
            '출력 예시: {"bottle":"right","cell phone":"left"}\n'
            "현재 규칙: {}\n입력: {}\n출력:".format(
                json.dumps(current_rules, ensure_ascii=False), command)
        )
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=100
        )
        content = response["choices"][0]["message"]["content"] or ""
        print("[Gemma]", content)
        matched = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if matched is None:
            return {}
        try:
            changes = json.loads(matched.group(0))
        except json.JSONDecodeError:
            return {}
        if not isinstance(changes, dict):
            return {}
        return {key: value for key, value in changes.items()
                if key in TARGET_CLASSES.values() and value in DIRECTIONS}


# 9. 중앙 감지 영역 안에서 페트병/핸드폰 중 가장 신뢰도 높은 객체 선택
def pick_detection(result, shape):
    h, w = shape[:2]
    left, top = w * CENTER_RECT[0], h * CENTER_RECT[1]
    right, bottom = w * CENTER_RECT[2], h * CENTER_RECT[3]
    choices = []
    for box in result.boxes:
        cls = int(box.cls[0].item())
        if cls not in TARGET_CLASSES:
            continue
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].cpu().tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if left <= cx <= right and top <= cy <= bottom:
            choices.append((conf, TARGET_CLASSES[cls],
                            (int(x1), int(y1), int(x2), int(y2))))
    return max(choices, key=lambda x: x[0]) if choices else None


# 10. 분류 방향을 OpenCV 화살표로 표시
def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    if direction == "left":
        start, end = (int(w * 0.23), h // 2), (int(w * 0.04), h // 2)
    else:
        start, end = (int(w * 0.77), h // 2), (int(w * 0.96), h // 2)
    cv2.arrowedLine(frame, start, end, (20, 220, 60), 12, tipLength=0.45)


# 11. 메인 실행: 카메라 입력 → YOLO 감지 → 규칙 조회 → 화면/음성 안내
def main():
    parser = argparse.ArgumentParser(description="VoiceSort 객체 분류 안내")
    parser.add_argument("--vision-only", action="store_true", help="영상 및 화살표 기능만 실행")
    parser.add_argument("--no-mirror", action="store_true", help="카메라 좌우 반전 비활성화")
    args = parser.parse_args()
    verify_environment(args.vision_only)

    rules = {"bottle": "left", "cell phone": "right"}
    speaker = Speaker(enabled=not args.vision_only)
    print("[YOLO]", YOLO_MODEL)
    model = YOLO(str(YOLO_MODEL))
    llm = None if args.vision_only else GemmaParser()

    # GStreamer 파이프라인으로 카메라 영상 수신
    cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        raise RuntimeError(
            "GStreamer camera failed. Check /dev/video0, YUY2 640x480@30, "
            "cv2 GStreamer support and webcam connection."
        )

    recent = deque(maxlen=STABLE_WINDOW)
    active_object = None
    last_seen = 0.0
    last_message = "Ready: bottle LEFT / phone RIGHT"
    print("[KEYS] R=음성 규칙 변경, Q=종료")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[CAMERA] 프레임을 읽지 못했습니다.")
                break
            # 화면을 거울 모드로 표시 (필요하면 --no-mirror 옵션)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            # TensorRT INT8 모델의 입력 크기인 640으로 객체 탐지
            results = model.predict(source=frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                    classes=[39, 67], verbose=False)
            target = pick_detection(results[0], frame.shape)
            name = target[1] if target else None
            # 최근 10프레임 중 같은 객체가 7번 이상 검출되면 안정적 인식으로 판정
            recent.append(name)
            now = time.monotonic()

            if target:
                conf, obj, (x1, y1, x2, y2) = target
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "{}: {:.2f}".format(obj, conf),
                            (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2)
                if obj == active_object:
                    last_seen = now

            # 일정 시간 객체가 보이지 않으면 다음 인식을 받을 수 있게 초기화
            if active_object is not None and now - last_seen >= LOST_SECONDS:
                active_object = None
                recent.clear()
                last_message = "Waiting for object"

            if active_object is None:
                for candidate in TARGET_CLASSES.values():
                    if recent.count(candidate) >= STABLE_COUNT:
                        active_object = candidate
                        last_seen = now
                        last_message = "{} -> {}".format(candidate, rules[candidate].upper())
                        speaker.speak("{}은 {}입니다.".format(
                            KOREAN_NAMES[candidate], DIRECTIONS[rules[candidate]]))
                        break

            # 화면에 중앙 인식 영역과 분류 규칙, 방향 화살표를 그림
            x1r, y1r = int(w * CENTER_RECT[0]), int(h * CENTER_RECT[1])
            x2r, y2r = int(w * CENTER_RECT[2]), int(h * CENTER_RECT[3])
            cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (255, 190, 0), 2)
            if active_object:
                draw_arrow(frame, rules[active_object])
            cv2.rectangle(frame, (0, 0), (w, 66), (32, 32, 32), -1)
            cv2.putText(frame, "R: Voice rule change | Q: Quit", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(frame, "Bottle: {} | Phone: {}".format(
                rules["bottle"].upper(), rules["cell phone"].upper()),
                (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)
            cv2.putText(frame, last_message, (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            cv2.imshow("VoiceSort", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            # R 키: 안내 TTS → 녹음 → STT → Gemma 해석 → 규칙 변경 → TTS
            if key == ord("r"):
                if args.vision_only:
                    print("[INFO] --vision-only 모드에서는 음성 규칙 변경이 비활성화됩니다.")
                    continue
                if speaker.busy:
                    print("[INFO] 음성 안내가 끝난 뒤 다시 R 키를 눌러주세요.")
                    continue
                try:
                    # 안내 음성이 끝나기 전에 녹음하면 Piper 음성까지 녹음될 수 있으므로 대기
                    if not speaker.speak("변경할 분류 규칙을 말씀해 주세요.", wait=True):
                        continue
                    command = speech_to_text()
                    if not command:
                        speaker.speak("음성 명령을 인식하지 못했습니다.")
                        continue
                    changes = llm.parse(command, rules)
                    if not changes:
                        speaker.speak("방향 변경 명령을 이해하지 못했습니다.")
                        continue
                    rules.update(changes)
                    last_message = "Bottle: {} / Phone: {}".format(
                        rules["bottle"], rules["cell phone"])
                    summary = ", ".join("{}은 {}".format(KOREAN_NAMES[k], DIRECTIONS[v])
                                        for k, v in rules.items())
                    print("[RULES]", rules)
                    speaker.speak("분류 규칙이 변경되었습니다. {}입니다.".format(summary))
                except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                    print("[RULE CHANGE ERROR]", exc)
    finally:
        # 종료 시 카메라 장치와 화면 자원을 정리
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
