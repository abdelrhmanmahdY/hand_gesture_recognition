import cv2
import mediapipe as mp
import cv2
import numpy as np
import time
from proccesing import clahe,denoise

class display_image():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.hand=hand_tracking()
        self.fps_update_time = time.time()
        self.display_fps = 0
        self.frame_count = 0
    def display(self):
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)
            # Apply gaussian blur to clean /reduce noise in the image
            img_proccessing = clahe(denoise(img))
            # change color to rgb for mediapipe
            img_rgb = cv2.cvtColor(img_proccessing  , cv2.COLOR_BGR2RGB)
            # flip the image for mirror view
            self.hand.hand_detector(img_rgb,img)
            self.calcute_fps(img)
            cv2.imshow("image",img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cam.release()
        cv2.destroyAllWindows()

    def calcute_fps(self,img):
        self.frame_count += 1
        current_time = time.time()
        # Update FPS every 0.75 seconds
        if current_time - self.fps_update_time >= 0.75:
            self.display_fps = round(self.frame_count / (current_time - self.fps_update_time), 2)
            self.fps_update_time = current_time
            self.frame_count = 0
            
        cv2.putText(img, f"fps: {self.display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

class hand_tracking():
    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5):
               self.static_image_mode=static_image_mode
               self.max_num_hands=max_num_hands
               self.model_complexity=model_complexity
               self.min_detection_confidence=min_detection_confidence
               self.min_tracking_confidence=min_tracking_confidence
               self.mphand=mp.solutions.hands
               self.mp_drawing=mp.solutions.drawing_utils
               self.hand=self.mphand.Hands(static_image_mode=self.static_image_mode,max_num_hands=self.max_num_hands,model_complexity=self.model_complexity,min_detection_confidence=self.min_detection_confidence,min_tracking_confidence=self.min_tracking_confidence)
   
    def hand_detector(self,image_proccessed,image):
        result=self.hand.process(image_proccessed)
            # flip the image for mirror view

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for id,lm in enumerate(hand_landmarks.landmark):
                    h,w,c= image.shape
                    cx,cy= int(lm.x * w), int(lm.y * h)
                    if id ==0:
                        cv2.circle(image,(cx,cy), 20, (255,255,0), cv2.FILLED)
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mphand.HAND_CONNECTIONS)
        
    
    
if __name__ == "__main__":
    display_image().display()