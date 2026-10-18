import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)
    
    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(ball_lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)

    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))


    final_frame = frame.copy()

    mask_lab = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)

    contours, _ = cv2.findContours(
    mask_lab,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )
    
    final_mask = np.zeros_like(mask_lab)
    
    if len(contours) > 0:
    # Binary mask를 반투명 초록색으로 overlay
        largest_contour = max(contours, key=cv2.contourArea)

        
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    final_frame_enhanced = frame.copy()

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)
    
    cv2.circle(final_frame_enhanced, center_circle, radius, (0,0,255), 3)
    cv2.circle(final_frame_enhanced, center_circle, 5, (255,0,0), -1)
    
    

    cv2.imshow("VideoCapture with GStreamer", final_frame_enhanced)

    




cap.release()
cv2.destroyAllWindows()


