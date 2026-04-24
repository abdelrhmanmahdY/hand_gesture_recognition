import mediapipe as mp
import cv2
import numpy as np
import time
from proccesing import clahe,denoise

mphand= mp.solutions.hands
cam = cv2.VideoCapture(0)
hand=mphand.Hands()
mp_drawing=mp.solutions.drawing_utils
fps_update_time = time.time()
display_fps = 0
frame_count = 0

while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    # Apply gaussian blur to clean /reduce noise in the image
    img_proccessing = clahe(denoise(img))
    # change color to rgb for mediapipe
    img_rgb = cv2.cvtColor(img_proccessing  , cv2.COLOR_BGR2RGB)
    result=hand.process(img_rgb)
    # flip the image for mirror view

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id,lm in enumerate(hand_landmarks.landmark):
                h,w,c= img.shape
                cx,cy= int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx,cy), 3, (255, 0, 255), cv2.FILLED)
            mp_drawing.draw_landmarks(img, hand_landmarks, mphand.HAND_CONNECTIONS)
    frame_count += 1
    current_time = time.time()
    
    # Update FPS every 0.75 seconds
    if current_time - fps_update_time >= 0.75:
        display_fps = round(frame_count / (current_time - fps_update_time), 2)
        fps_update_time = current_time
        frame_count = 0
        
    cv2.putText(img, f"fps: {display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("image",img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()