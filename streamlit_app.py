import streamlit as st
import cv2
import time
from face_match import authenticate_face
from auth_pipeline import run_liveness

st.set_page_config(page_title="Secure Voting Auth", layout="centered")

# ----------------------------------
# Session state
# ----------------------------------
if "cap" not in st.session_state:
    st.session_state.cap = None

if "mode" not in st.session_state:
    st.session_state.mode = "idle"   # idle | face | live | done

if "liveness_state" not in st.session_state:
    st.session_state.liveness_state = {
        "blink": False,
        "head": False,
        "frames": 0,
        "prev": None
    }

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "user_score" not in st.session_state:
    st.session_state.user_score = None

# ----------------------------------
# UI placeholders
# ----------------------------------
frame_box = st.empty()
status_box = st.empty()
identity_box = st.empty()
progress_box = st.empty()
button_box = st.empty()

# ----------------------------------
# Idle state
# ----------------------------------
if st.session_state.mode == "idle":
    identity_box.empty()
    progress_box.empty()

    if button_box.button("Start Authentication"):
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.mode = "face"

# ----------------------------------
# Streaming loop
# ----------------------------------
if st.session_state.mode in ["face", "live"]:
    cap = st.session_state.cap

    if not cap or not cap.isOpened():
        status_box.error("Camera not available")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                status_box.error("Failed to read camera")
                break

            # ---------------- FACE RECOGNITION ----------------
            if st.session_state.mode == "face":
                status_box.info("📸 Looking for registered face…")

                matched, name, score, bbox, face_count = authenticate_face(frame)

                # ❌ Multi-face rejection
                if face_count > 1:
                    status_box.error("❌ Multiple faces detected")
                    frame_box.image(frame, channels="BGR")
                    time.sleep(0.05)
                    continue

                # ❌ Unregistered face
                if face_count == 1 and not matched:
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "UNREGISTERED", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    status_box.error("❌ Unregistered face")
                    frame_box.image(frame, channels="BGR")
                    time.sleep(0.05)
                    continue

                # ✅ Registered face
                if matched:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    frame_box.image(frame, channels="BGR")

                    st.session_state.user_name = name
                    st.session_state.user_score = score

                    identity_box.success(f"👤 {name}")
                    progress_box.progress(min(score, 1.0))
                    progress_box.caption(f"Confidence: {score:.2f}")

                    st.session_state.mode = "live"
                    time.sleep(1)

            # ---------------- LIVENESS ----------------
            elif st.session_state.mode == "live":
                status_box.info("👁️ Blink & move head")

                frame, result = run_liveness(frame, st.session_state.liveness_state)
                frame_box.image(frame, channels="BGR")

                if result["done"]:
                    st.session_state.mode = "done"
                    break

            time.sleep(0.03)

# ----------------------------------
# Done
# ----------------------------------
if st.session_state.mode == "done":
    identity_box.success(f"✅ Authenticated: {st.session_state.user_name}")
    progress_box.progress(min(st.session_state.user_score, 1.0))
    progress_box.caption(f"Final confidence: {st.session_state.user_score:.2f}")
    status_box.success("Authentication successful")

    if button_box.button("Next Person"):
        if st.session_state.cap:
            st.session_state.cap.release()

        st.session_state.cap = None
        st.session_state.mode = "idle"
        st.session_state.user_name = None
        st.session_state.user_score = None
        st.session_state.liveness_state = {
            "blink": False,
            "head": False,
            "frames": 0,
            "prev": None
        }
