import cv2
import mediapipe as mp
import numpy as np
import os

def find_perfect_representative_frames():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    # Eye indices for EAR
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def calculate_ear(landmarks, img_w, img_h):
        def get_coords(eye_indices):
            return [np.array([landmarks.landmark[idx].x * img_w, landmarks.landmark[idx].y * img_h]) for idx in eye_indices]
        l_coords = get_coords(LEFT_EYE)
        r_coords = get_coords(RIGHT_EYE)
        def ear(c):
            v1 = np.linalg.norm(c[1] - c[5])
            v2 = np.linalg.norm(c[2] - c[4])
            h = np.linalg.norm(c[0] - c[3])
            return (v1 + v2) / (2.0 * h)
        return (ear(l_coords) + ear(r_coords)) / 2.0

    # 1. ALERT: Find frame with MAX EAR (Wide open eyes)
    cap_alert = cv2.VideoCapture("Data/UBFC-rPPG/DATASET_1/10-gt/vid.avi")
    alert_frame = None
    best_alert_ear = 0
    for i in range(1000):
        ret, frame = cap_alert.read()
        if not ret: break
        if i % 5 == 0:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                ear = calculate_ear(results.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
                if ear > best_alert_ear:
                    best_alert_ear = ear
                    alert_frame = frame.copy()
    cap_alert.release()
    if alert_frame is not None:
        cv2.imwrite("logs/debug_frames_7gt/alert_10gt_wide.png", alert_frame)
        print(f"Saved Alert frame (Wide eyes) EAR: {best_alert_ear:.4f}")

    # 2. FATIGUED: Find frame with DROOPING EAR (approx 0.22 - 0.25)
    cap_fatigue = cv2.VideoCapture("Data/UBFC-rPPG/DATASET_1/6-gt/vid.avi")
    fatigue_frame = None
    # We want a mid-range EAR that isn't a blink
    target_ear = 0.24 
    min_diff = 1.0
    for i in range(1500):
        ret, frame = cap_fatigue.read()
        if not ret: break
        if i % 5 == 0:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                ear = calculate_ear(results.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
                if abs(ear - target_ear) < min_diff:
                    min_diff = abs(ear - target_ear)
                    fatigue_frame = frame.copy()
                    current_ear = ear
    cap_fatigue.release()
    if fatigue_frame is not None:
        cv2.imwrite("logs/debug_frames_7gt/fatigue_6gt_droop.png", fatigue_frame)
        print(f"Saved Fatigue frame (Drooping eyes) EAR: {current_ear:.4f}")

if __name__ == "__main__":
    find_perfect_representative_frames()
