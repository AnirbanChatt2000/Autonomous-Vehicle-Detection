# pip install opencv-python ultralytics
# Run the above commands for installing the packages
# Downloads everything needed to run on your nvidea gpu

import cv2
from ultralytics import YOLO
model = YOLO("yolo11n.pt"); # Selects the Nano model 

# Had an issue where it was not working perfectly. Follow the next steps to troubleshoot, if encountered
# Download the model at https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
# Put it into same folder as the code and also put the video in the same folder for easy use

# Enter the video path, not sure about relative path address in windows
video_path = "videoplayback.mp4"  # Video name here
cap = cv2.VideoCapture(video_path)  # Path to the vid

# The below thing decides what YOLO detects
VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

while cap.isOpened():  # Get the frames from video 
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    # Run YOLOv11 detection
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            
            # Only process vehicles
            if cls in VEHICLE_CLASSES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, f"Vehicle ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show processed frame
    cv2.imshow("Vehicle Detection Prototype", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Click "q" to stop process, don't know the CTRL+C equivalent on windows
        break

cap.release()
cv2.destroyAllWindows() # DOne