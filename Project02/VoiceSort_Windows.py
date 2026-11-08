"""VoiceSort: 음성 명령 기반 실시간 객체 분류 안내 시스템.

Project02 폴더에서 실행:
    python VoiceSort_Windows.py
    python VoiceSort_Windows.py --vision-only

R: 음성으로 분류 규칙 변경 / Q 또는 Esc: 종료
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import winsound

import cv2
import sounddevice as sd
from llama_cpp import Llama
from ultralytics import YOLO

# 1. 모델 및 파일 경로 (Project02 폴더 기준)
YOLO_MODEL = Path("src/models/YOLO/yolo11n.pt")
GEMMA_MODEL = Path("src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
WHISPER_MODEL = Path("whisper.cpp/models/ggml-base.bin")
PIPER_MODEL = Path("src/models/Piper/ko_KR-kss-medium.onnx")
AUDIO_DIR = Path("src/audio")
INPUT_WAV = AUDIO_DIR / "input.wav"
OUTPUT_WAV = AUDIO_DIR / "response.wav"

# 2. 객체 탐지 클래스 및 분류 방향 설정
TARGET_CLASSES = {39: "bottle", 67: "cell phone"}
KOREAN_NAMES = {"bottle": "페트병", "cell phone": "핸드폰"}
ALLOWED_DIRECTIONS = {"left": "왼쪽", "right": "오른쪽"}

# 3. 카메라와 객체 탐지 설정
CAMERA_INDEX = 0
CONFIDENCE = 0.60
IMAGE_SIZE = 416  # CPU 추론 부담을 줄이기 위한 입력 크기
CENTER_RECT = (0.28, 0.22, 0.72, 0.78)  # 중앙 감지 영역(화면 비율)
STABLE_WINDOW = 10  # 최근 10프레임 기준
STABLE_COUNT = 7  # 7프레임 이상 인식되면 안정적인 탐지로 판정
LOST_SECONDS = 0.9  # 물체가 사라진 뒤 초기화하기까지의 시간
RECORD_SECONDS = 5
SAMPLE_RATE = 16000


# 4. 빌드된 Whisper 실행 파일 탐색
def find_whisper_cli():
    candidates = [
        Path("whisper.cpp/build/bin/whisper-cli.exe"),
        Path("whisper.cpp/build/bin/Release/whisper-cli.exe"),
        Path("whisper.cpp/build-cpu/bin/whisper-cli.exe"),
        Path("whisper.cpp/build-cpu/bin/Release/whisper-cli.exe"),
    ]
    override = os.environ.get("WHISPER_CLI")
    if override:
        candidates.insert(0, Path(override))
    return next((path for path in candidates if path.is_file()), candidates[0])


# 5. 실행 전에 필요한 모델 파일 확인
def show_required_files(vision_only):
    """통합 모드에서 필요한 모델/실행 파일의 존재 여부를 확인한다."""
    required = {"Gemma GGUF": GEMMA_MODEL,
                "Whisper CLI": find_whisper_cli(),
                "Whisper base": WHISPER_MODEL,
                "Piper Korean voice": PIPER_MODEL,
                "Piper voice config": Path(str(PIPER_MODEL) + ".json")}
    missing = {name: path for name, path in required.items() if not path.is_file()}
    if missing and not vision_only:
        print("\n[ERROR] The full audio/LLM pipeline needs these files:")
        for name, path in missing.items():
            print(f"  {name}: {path}")
        print("\nUse --vision-only to test the webcam and YOLO first.")
        raise SystemExit(2)


# 6. 마이크 음성을 16kHz 모노 WAV 파일로 녹음
def record_audio():
    """sounddevice로 5초간 음성을 녹음한다."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[STT] Speak now ({RECORD_SECONDS} seconds)")
    samples = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                     samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    with wave.open(str(INPUT_WAV), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


# 7. Whisper.cpp를 사용하여 한국어 음성을 텍스트로 변환
def speech_to_text():
    record_audio()
    proc = subprocess.run(
        [str(find_whisper_cli()), "-m", str(WHISPER_MODEL),
         "-f", str(INPUT_WAV), "-l", "ko", "--no-timestamps"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    # Whisper 실행 로그는 stderr, 인식된 음성 문장은 stdout에서 확인
    transcript = " ".join(line.strip() for line in proc.stdout.splitlines()
                          if line.strip() and not line.startswith("whisper_"))
    print("[STT]", transcript)
    return transcript


# 8. Gemma를 이용해 음성 분류 명령을 JSON 규칙으로 해석
class GemmaParser:
    """Gemma는 R 키로 규칙을 변경할 때만 호출한다."""
    def __init__(self):
        print("[Gemma] CPU 모델 로딩:", GEMMA_MODEL)
        self.llm = Llama(model_path=str(GEMMA_MODEL),
                        n_gpu_layers=0,
                        n_ctx=2048, n_batch=32, n_ubatch=32, verbose=False)

    def parse(self, command, current_rules):
        prompt = (
            "너는 물건 분류 규칙 변경 명령을 JSON으로 바꾸는 도우미다. "
            "대상은 페트병(bottle)과 핸드폰(cell phone) 두 종류뿐이다. "
            "방향은 left(왼쪽) 또는 right(오른쪽)만 허용한다. "
            "발화에서 명시적으로 바꾸라고 한 물건만 포함한다. "
            "불분명하거나 방향이 없는 경우 빈 객체 {}를 출력한다. "
            "JSON 객체만 출력하고 설명이나 마크다운은 쓰지 마라.\n"
            "예시 입력: 페트병은 오른쪽으로 두고 핸드폰은 왼쪽으로 둬\n"
            '예시 출력: {"bottle":"right","cell phone":"left"}\n'
            f"현재 규칙: {json.dumps(current_rules, ensure_ascii=False)}\n"
            f"입력: {command}\n출력:"
        )
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=100,
        )
        content = response["choices"][0]["message"]["content"] or ""
        print("[Gemma]", content)
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            proposed = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        if not isinstance(proposed, dict):
            return {}
        # 허용된 객체와 방향만 규칙 변경에 사용
        return {key: value for key, value in proposed.items()
                if key in TARGET_CLASSES.values()
                and isinstance(value, str) and value in ALLOWED_DIRECTIONS}


# 9. Piper에서 생성한 WAV 파일을 재생
def play_wav(path):
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


# 10. TTS 생성 및 재생; 중복 안내 방지
class Speaker:
    """음성을 별도 스레드에서 재생하고 중복 출력을 방지한다."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.busy = False

    def speak(self, text):
        print("[TTS]", text)
        if not self.enabled:
            return
        with self.lock:
            if self.busy:
                print("[TTS] Busy: skipped overlapping message")
                return
            self.busy = True
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()
        return thread

    def speak_and_wait(self, text):
        """음성 안내가 끝난 뒤에 마이크 녹음을 시작한다."""
        thread = self.speak(text)
        if thread is not None:
            thread.join()

    def _run(self, text):
        try:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            # Piper 모델로 한국어 음성 생성
            subprocess.run([sys.executable, "-m", "piper", "-m", str(PIPER_MODEL),
                            "-f", str(OUTPUT_WAV), "--", text],
                           check=True, capture_output=True, text=True)
            play_wav(OUTPUT_WAV)
        except Exception as exc:
            print("[TTS ERROR]", exc)
        finally:
            with self.lock:
                self.busy = False


# 11. 카메라 중앙에서 페트병과 핸드폰만 인식
def pick_detection(result, shape):
    """중앙 영역 안의 후보 중 신뢰도가 가장 높은 물체를 선택한다."""
    height, width = shape[:2]
    left = width * CENTER_RECT[0]
    top = height * CENTER_RECT[1]
    right = width * CENTER_RECT[2]
    bottom = height * CENTER_RECT[3]
    candidates = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in TARGET_CLASSES:
            continue
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if left <= cx <= right and top <= cy <= bottom:
            candidates.append((conf, TARGET_CLASSES[class_id],
                               (int(x1), int(y1), int(x2), int(y2))))
    return max(candidates, default=None, key=lambda item: item[0])


# 12. 현재 분류 방향을 화살표로 표시
def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    if direction == "left":
        start, end = (int(w * .23), h // 2), (int(w * .04), h // 2)
    else:
        start, end = (int(w * .77), h // 2), (int(w * .96), h // 2)
    cv2.arrowedLine(frame, start, end, (20, 220, 60), thickness=12, tipLength=.45)


# 13. 카메라 입력, YOLO 탐지, 규칙 변경 기능 통합
def main():
    parser = argparse.ArgumentParser(description="VoiceSort 객체 분류 안내")
    parser.add_argument("--vision-only", action="store_true",
                        help="카메라와 YOLO만 테스트")
    parser.add_argument("--no-mirror", action="store_true",
                        help="웹캠 좌우 반전 끄기")
    args = parser.parse_args()
    show_required_files(args.vision_only)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rules = {"bottle": "left", "cell phone": "right"}
    speaker = Speaker(enabled=not args.vision_only)

    print("[YOLO] Loading:", YOLO_MODEL)
    if not YOLO_MODEL.is_file() and YOLO_MODEL.suffix == ".pt":
        # 사전 학습 YOLO 모델이 없다면 처음 실행할 때 다운로드
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO(str(YOLO_MODEL))

    llm = None if args.vision_only else GemmaParser()
    # 일반 USB/내장 웹캠 입력
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera. Check permissions and CAMERA_INDEX.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recent = deque(maxlen=STABLE_WINDOW)
    active_object = None
    last_seen = 0.0
    last_message = "Ready: bottle=LEFT, cell phone=RIGHT"
    print("\nControls: R=record sorting instruction (5 s), Q=quit")
    print("Default rules:", rules)
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[CAMERA] Lost frame")
                break
            # 화면을 거울 모드로 보여주되 YOLO도 같은 좌표계를 사용
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            # 두 객체만 탐지하여 불필요한 탐지 결과 제외
            results = model.predict(frame, classes=[39, 67],
                                    conf=CONFIDENCE, imgsz=IMAGE_SIZE,
                                    verbose=False, device="cpu")
            target = pick_detection(results[0], frame.shape)
            detected = target[1] if target else None
            recent.append(detected)
            now = time.monotonic()

            if target:
                confidence, name, (x1, y1, x2, y2) = target
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{name}: {confidence:.2f}", (x1, max(y1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                if active_object == name:
                    last_seen = now
            if active_object and now - last_seen >= LOST_SECONDS:
                active_object = None
                recent.clear()
                last_message = "Waiting for object"

            if active_object is None:
                for name in TARGET_CLASSES.values():
                    if recent.count(name) >= STABLE_COUNT:
                        active_object = name
                        last_seen = now
                        direction = ALLOWED_DIRECTIONS[rules[name]]
                        message = f"{KOREAN_NAMES[name]}은 {direction}입니다."
                        last_message = f"{name} -> {rules[name].upper()}"
                        speaker.speak(message)
                        break

            x1r, y1r = int(w * CENTER_RECT[0]), int(h * CENTER_RECT[1])
            x2r, y2r = int(w * CENTER_RECT[2]), int(h * CENTER_RECT[3])
            cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (255, 190, 0), 2)
            if active_object:
                draw_arrow(frame, rules[active_object])
            cv2.rectangle(frame, (0, 0), (w, 66), (32, 32, 32), -1)
            cv2.putText(frame, "R: Change rules by voice  |  Q: Quit", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 2)
            cv2.putText(frame, f"Bottle: {rules['bottle'].upper()}  |  Phone: {rules['cell phone'].upper()}",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 1)
            cv2.putText(frame, last_message, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (230, 230, 230), 2)
            cv2.imshow("VoiceSort - YOLO / Gemma / STT / TTS", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                if args.vision_only:
                    print("[TEST MODE] Use full mode for STT + Gemma + TTS")
                    continue
                if speaker.busy:
                    print("[INFO] Finish TTS playback before recording to avoid feedback.")
                    continue
                try:
                    speaker.speak_and_wait("변경할 분류 규칙을 말씀해 주세요.")
                    command = speech_to_text()
                    if not command:
                        speaker.speak("음성 명령을 인식하지 못했습니다.")
                        continue
                    changes = llm.parse(command, rules)
                    if not changes:
                        speaker.speak("방향 변경 명령을 이해하지 못했습니다.")
                        continue
                    rules.update(changes)
                    summary = ", ".join(f"{KOREAN_NAMES[k]}은 {ALLOWED_DIRECTIONS[v]}"
                                        for k, v in rules.items())
                    last_message = f"Bottle: {rules['bottle']} / Phone: {rules['cell phone']}"
                    print("[RULES]", rules)
                    speaker.speak(f"분류 규칙이 변경되었습니다. {summary}입니다.")
                    # 현재 물체가 감지 중이라면 새 규칙으로 화살표가 갱신됨
                except Exception as exc:
                    print("[RULE CHANGE ERROR]", exc)
                    # 음성 인식이나 명령 해석이 실패하면 규칙을 변경하지 않음
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
