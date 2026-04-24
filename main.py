import mediapipe as mp
import cv2
import numpy as np
import time
from proccesing import clahe,denoise

mphand= mp.solutions.hands
cam = cv2.VideoCapture(0)
hand=mphand.Hands()
mp_drawing=mp.solutions.drawing_utils
while True:
    success, img = cam.read()
    
    # Apply Gaussian Blur to clean/reduce noise in the image
    img_proccessing = clahe(denoise(img))
    
    img_rgb = cv2.cvtColor(img_proccessing  , cv2.COLOR_BGR2RGB)
    result=hand.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mphand.HAND_CONNECTIONS)
    cv2.imshow("image",img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()