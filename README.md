# Secure Voting Authentication System
Face Recognition + Liveness Detection (InsightFace + ONNX)



## 1. Project Overview

This project is a real-time biometric authentication system designed for secure voting use cases.  
It combines **face recognition** and **liveness detection** to ensure that only authorized and live users can authenticate.

The system is built to be:
- Real-time
- Lightweight (TensorFlow-free)
- macOS compatible
- Interview & demo ready



## 2. Key Features

### Face Recognition
- Single-face enforcement
- Registered-user verification
- Confidence-based matching
- Face bounding box with name overlay
- Unregistered face rejection
- Multiple-face rejection

### Liveness Detection
- Eye blink detection
- Head movement verification
- Sequential challenge validation
- Anti-spoofing protection

### User Experience
- Live camera feed
- Persistent identity display
- Confidence progress bar
- Next-person authentication without restarting app



## 3. Technology Stack

- Language: Python 3.10
- UI: Streamlit
- Face Recognition: InsightFace
- Model Runtime: ONNX Runtime
- Computer Vision: OpenCV
- ML Framework: TensorFlow-free (CPU optimized)



## 4. Project Structure


votingsystem/
│
├── streamlit_app.py        # Main Streamlit application
├── face_match.py           # Face recognition logic
├── auth_pipeline.py        # Liveness detection (blink + head movement)
├── face_capture.py         # Face enrollment utility
├── face_detect.py          # Face detection helper
├── requirements.txt        # Frozen dependencies
│
├── data/
│   └── registered_faces/   # Stored face images / embeddings
│
└── README.md




## 5. Authentication Flow

1. Start Authentication
2. Face Detection & Recognition
   - Reject multiple faces
   - Reject unregistered faces
   - Accept registered face above confidence threshold
3. Liveness Verification
   - Blink detection
   - Head movement detection
4. Authentication Success
5. Reset for Next Person



## 6. Setup Instructions

### Step 1: Create Virtual Environment

python3.10 -m venv venv
source venv/bin/activate


### Step 2: Install Dependencies

pip install -r requirements.txt


### Step 3: Run the Application

streamlit run streamlit_app.py


Open in browser:

http://localhost:8501




## 7. Platform Compatibility

### Supported
- macOS (local execution)
- Linux (local or cloud VM)

### Docker Note (macOS)
Docker on macOS cannot access the system webcam due to OS-level virtualization limits.  
Camera-based features work fully on Linux-based systems.



## 8. Security Considerations

- Explicit unregistered face rejection
- Single-face enforcement
- Liveness verification before authentication
- Confidence thresholding
- No heavy ML runtime dependencies



## 9. Deployment Strategy

- Local Streamlit deployment for real-time demos
- Linux VM for full cloud deployment
- Browser-based camera (WebRTC) recommended for large-scale systems



## 10. Why InsightFace + ONNX?

- Faster CPU inference
- Lightweight runtime
- Better macOS compatibility
- Industry-grade embeddings
- Easy scalability



## 11. Future Enhancements

- WebRTC browser camera support
- Admin enrollment dashboard
- Failed-attempt logging
- GPU acceleration
- Cloud deployment (AWS / GCP)



## 12. Author

Developed by Krishna Yadav and Qaaid Badri      
Focus: Secure systems, computer vision, applied AI



## 13. License

This project is intended for educational and research purposes.
