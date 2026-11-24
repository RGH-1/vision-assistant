import cv2

rtsp_url = "rtsp://localhost:8554/phone"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"Error: Could not connect to RTSP stream at {rtsp_url}")
    exit(1)

print(f"Successfully connected to stream: {rtsp_url}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Reconnecting...")
        break

    cv2.imshow("M2 RTSP Stream Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()