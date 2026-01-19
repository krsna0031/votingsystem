import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Initialize InsightFace (ONNX Runtime)
# -------------------------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------------------------------
# Config
# -------------------------------------------------
BASE_DIR = "data/voters"
SIMILARITY_THRESHOLD = 0.5

# -------------------------------------------------
# Load reference embeddings ONCE
# -------------------------------------------------
reference_embeddings = []
reference_ids = []

def load_reference_faces():
    for file in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, file)
        img = cv2.imread(path)

        if img is None:
            continue

        faces = app.get(img)
        if faces:
            reference_embeddings.append(faces[0].embedding)
            reference_ids.append(file)

load_reference_faces()

# -------------------------------------------------
# PUBLIC API (Streamlit imports THIS)
# -------------------------------------------------
def authenticate_face(frame):
    """
    Returns:
        matched (bool)
        matched_id (str or None)
        score (float)
        bbox (tuple or None)
        face_count (int)
    """
    faces = app.get(frame)
    face_count = len(faces)

    if face_count != 1:
        return False, None, 0.0, None, face_count

    face = faces[0]
    emb = face.embedding
    bbox = tuple(map(int, face.bbox))

    scores = cosine_similarity([emb], reference_embeddings)[0]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return True, reference_ids[best_idx], float(best_score), bbox, face_count

    return False, None, float(best_score), bbox, face_count

