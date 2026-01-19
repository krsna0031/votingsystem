import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def check_liveness(frame_placeholder,
                   blink_threshold=0.23,
                   required_blinks=2,
                   max_time=10):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    blink_count = 0
    frame_counter = 0
    start_time = time.time()

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = np.array([[int(landmarks[i].x * w),
                                  int(landmarks[i].y * h)] for i in LEFT_EYE])
            right_eye = np.array([[int(landmarks[i].x * w),
                                   int(landmarks[i].y * h)] for i in RIGHT_EYE])

            ear = (eye_aspect_ratio(left_eye) +
                   eye_aspect_ratio(right_eye)) / 2.0

            if ear < blink_threshold:
                frame_counter += 1
            else:
                if frame_counter >= 2:
                    blink_count += 1
                frame_counter = 0

            cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✅ Send frame to Streamlit (not OpenCV GUI)
        frame_placeholder.image(frame, channels="BGR")

        if blink_count >= required_blinks:
            break

        if time.time() - start_time > max_time:
            break

    cap.release()
    return blink_count >= required_blinks
