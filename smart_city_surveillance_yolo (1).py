import cv2
from ultralytics import YOLO
import time

MODEL_PATH = "yolov8n.pt"
SOURCE = "road.mp4"
CONFIDENCE = 0.40
WIDTH = 640
HEIGHT = 384

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera or video source.")

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    start = time.perf_counter()

    result = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    latency = (time.perf_counter() - start) * 1000.0

    fps = 1000.0 / latency if latency > 0 else 0.0

    output = result.plot()

    cv2.putText(
        output,
        f"Latency: {latency:.1f} ms",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Smart City Roadside Object Detection",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()