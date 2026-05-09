from ultralytics import YOLO
import cv2
import time
model = YOLO("runs/detect/runs/asl_yolov8m/weights/best.pt")
cap = cv2.VideoCapture(0)
time_start= time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.6)
    annotated = results[0].plot()
    time_end=time.time()
    # Show detected letter big on screen
    if results[0].boxes and time_end-time_start >= 1 :
        cls_id = int(results[0].boxes.cls[0])
        letter = model.names[cls_id]
        cv2.putText(annotated, f"Sign: {letter}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 30)
        time_end=time_start
        time_start= time.time()
        
    cv2.imshow("ASL Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()