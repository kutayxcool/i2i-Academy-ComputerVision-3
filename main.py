import cv2
import mediapipe as mp
import urllib.request
import os
import math

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Model indiriliyor...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Model indirildi!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

finger_tips = [8, 12, 16, 20]

def mesafe(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        finger_count = 0

        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks:
                h, w, _ = frame.shape

                for lm in hand_landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # El boyutunu referans al (bilek - orta parmak MCP arası)
                el_boyutu = mesafe(hand_landmark[0], hand_landmark[9])

                # Başparmak: uç ile işaret parmağı MCP arası mesafe
                if mesafe(hand_landmark[4], hand_landmark[5]) > el_boyutu * 0.5:
                    finger_count += 1

                # Diğer 4 parmak
                for tip in finger_tips:
                    if hand_landmark[tip].y < hand_landmark[tip - 2].y:
                        finger_count += 1

        cv2.putText(frame, f'Parmak Sayisi: {finger_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('i2i Finger Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()