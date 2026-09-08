import cv2
import mediapipe as mp
import numpy as np
import os

def find_representative_frames():
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

    # 1. Find ALERT frame (Subject 10-gt is very stable)
    cap_alert = cv2.VideoCapture("Data/UBFC-rPPG/DATASET_1/10-gt/vid.avi")
    alert_frame = None
    max_ear = 0
    for _ in range(500): # Scan first 500 frames
        ret, frame = cap_alert.read()
        if not ret: break
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            ear = calculate_ear(results.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
            if ear > max_ear:
                max_ear = ear
                alert_frame = frame.copy()
    cap_alert.release()
    if alert_frame is not None:
        cv2.imwrite("logs/debug_frames_7gt/alert_10gt.png", alert_frame)
        print(f"Saved Alert frame from 10-gt (EAR: {max_ear:.4f})")

    # 2. Find FATIGUED frame (Subject 6-gt shows some eye drooping)
    cap_fatigue = cv2.VideoCapture("Data/UBFC-rPPG/DATASET_1/6-gt/vid.avi")
    fatigue_frame = None
    target_ear = 0.22 # Looking for "semi-closed" eyelids
    closest_diff = 1.0
    for _ in range(1000):
        ret, frame = cap_fatigue.read()
        if not ret: break
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            ear = calculate_ear(results.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
            if abs(ear - target_ear) < closest_diff:
                closest_diff = abs(ear - target_ear)
                fatigue_frame = frame.copy()
    cap_fatigue.release()
    if fatigue_frame is not None:
        cv2.imwrite("logs/debug_frames_7gt/fatigue_6gt.png", fatigue_frame)
        print(f"Saved Fatigue frame from 6-gt (EAR: {target_ear + closest_diff:.4f})")

if __name__ == "__main__":
    find_representative_frames()
