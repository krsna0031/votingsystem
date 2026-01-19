import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def run_liveness(frame, state):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    status = {"blink": False, "head": False, "done": False}

    if not res.multi_face_landmarks:
        return frame, status

    h, w, _ = frame.shape
    lm = res.multi_face_landmarks[0].landmark

    left = np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT_EYE])
    right = np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT_EYE])

    ear_val = (ear(left) + ear(right)) / 2

    if ear_val < 0.23:
        state["frames"] += 1
    else:
        if state["frames"] > 2:
            state["blink"] = True
        state["frames"] = 0

    nose_x = lm[1].x * w
    if state["prev"] and abs(nose_x - state["prev"]) > 20:
        state["head"] = True
    state["prev"] = nose_x

    status["blink"] = state["blink"]
    status["head"] = state["head"]
    status["done"] = state["blink"] and state["head"]

    return frame, status
