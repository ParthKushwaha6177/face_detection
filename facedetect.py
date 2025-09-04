"""
Proctor system with Face + Phone detection alerts (Windows-ready).
- Uses MediaPipe for fast face detection
- Uses YOLOv5 (ultralytics) for phone detection
- Alerts if >1 face OR a phone appears
- Saves screenshots + logs + beeps
"""

import os
import time
import platform
from datetime import datetime
import cv2
import mediapipe as mp
from ultralytics import YOLO

# ---------- Config ----------
MIN_DETECTION_CONFIDENCE = 0.5
ALERT_COOLDOWN_SEC = 5.0
ALERT_DIR = "alerts"
LOG_FILE = "alert_log.csv"
CAM_INDEX = 0
WINDOW_NAME = "Proctor - Face + Phone Detection"
PHONE_CLASSES = {"cell phone"}  # YOLO class labels to trigger alert
# ----------------------------

os.makedirs(ALERT_DIR, exist_ok=True)

# cross-platform beep
if platform.system() == "Windows":
    try:
        import winsound
        def do_beep(): winsound.Beep(1200, 450)
    except Exception:
        def do_beep(): print("\a", end="", flush=True)
else:
    def do_beep(): print("\a", end="", flush=True)

mp_face = mp.solutions.face_detection

# load YOLOv5 pretrained COCO model (detects phones, people, etc.)
yolo_model = YOLO("yolov8n.pt")  # ultralytics latest small model

def log_alert(reason, faces_count, screenshot_path):
    header_needed = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp_iso,reason,faces_count,screenshot_path\n")
        f.write(f"{datetime.now().isoformat()},{reason},{faces_count},{screenshot_path}\n")

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE) as detector:
        last_alert_time = 0.0
        prev_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(frame_rgb)

            # ---- Face Detection ----
            faces = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    score = float(det.score[0]) if det.score else 0.0
                    faces.append((x, y, bw, bh, score))
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
                    cv2.putText(frame, f"{score:.2f}", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # ---- Phone Detection via YOLO ----
            phone_detected = False
            results_yolo = yolo_model.predict(frame, verbose=False)
            for r in results_yolo:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    conf = float(box.conf[0])
                    if label in PHONE_CLASSES and conf > 0.5:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ---- Alert Conditions ----
            now_time = time.time()
            reason = None
            if len(faces) > 1:
                reason = f"Multiple faces ({len(faces)})"
            elif phone_detected:
                reason = "Phone detected"

            if reason:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                cv2.putText(frame, f"ALERT: {reason}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                if now_time - last_alert_time > ALERT_COOLDOWN_SEC:
                    last_alert_time = now_time
                    do_beep()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"alert_{ts}.jpg"
                    filepath = os.path.join(ALERT_DIR, filename)
                    cv2.imwrite(filepath, frame)
                    log_alert(reason, len(faces), filepath)
                    print(f"[ALERT] {reason} — screenshot saved: {filepath}")

            # FPS display
            fps = 1.0 / (now_time - prev_time) if (now_time - prev_time) > 0 else 0
            prev_time = now_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

