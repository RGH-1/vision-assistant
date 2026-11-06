import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

rtsp_url = "rtsp://localhost:8554/mystream"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force drop of old frames

if not cap.isOpened():
    print("Error: Stream not reachable.")
    exit()

window_name = "Optimized Live Stream"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    # Grab the newest frame directly and skip backlogged queue
    if not cap.grab():
        break
    ret, frame = cap.retrieve()
    if not ret:
        break
        
    cv2.imshow(window_name, frame)
    
    # 15ms wait keeps CPU cool while maintaining ~60fps UI responsiveness
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()