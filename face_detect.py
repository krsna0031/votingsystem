import cv2
from insightface.app import FaceAnalysis

# -------------------------------------------------
# Initialize InsightFace (same config everywhere)
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

if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        score = face.det_score

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Conf: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("InsightFace - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
