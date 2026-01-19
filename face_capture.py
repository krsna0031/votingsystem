import cv2
import os
from insightface.app import FaceAnalysis

# -------------------------------------------------
# Config
# -------------------------------------------------
SAVE_DIR = "data/voters"
os.makedirs(SAVE_DIR, exist_ok=True)

PERSON_NAME = input("Enter person name (without spaces): ").strip()
SAVE_PATH = os.path.join(SAVE_DIR, f"{PERSON_NAME}.jpg")

# -------------------------------------------------
# Initialize InsightFace (same as face_match.py)
# -------------------------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------------------------------
# Camera
# -------------------------------------------------
cap = cv2.VideoCapture(0)

print("[INFO] Press 'c' to capture face | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible")
        break

    faces = app.get(frame)

    # Draw bounding box if face detected
    if faces:
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face detected - press C",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture (InsightFace)", frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture
    if key == ord("c") and faces:
        cv2.imwrite(SAVE_PATH, frame)
        print(f"[SUCCESS] Face saved at: {SAVE_PATH}")
        break

    # Quit
    if key == ord("q"):
        print("[INFO] Capture cancelled")
        break

cap.release()
cv2.destroyAllWindows()
