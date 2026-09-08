import cv2
import mediapipe as mp
import numpy as np
import os

def find_closed_eyes_7gt():
    video_path = "Data/UBFC-rPPG/DATASET_1/7-gt/vid.avi"
    cap = cv2.VideoCapture(video_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    # Eye indices for EAR
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def calculate_ear(landmarks, eye_indices, img_w, img_h):
        coords = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            coords.append(np.array([lm.x * img_w, lm.y * img_h]))
        
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h)

    min_ear = 999
    best_frame = None
    best_idx = 0
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % 10 == 0: # Check every 10th frame for speed
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                ear_l = calculate_ear(landmarks, LEFT_EYE, frame.shape[1], frame.shape[0])
                ear_r = calculate_ear(landmarks, RIGHT_EYE, frame.shape[1], frame.shape[0])
                ear = (ear_l + ear_r) / 2.0
                
                if ear < min_ear:
                    min_ear = ear
                    best_frame = frame.copy()
                    best_idx = count
                    print(f"New Min EAR: {min_ear:.4f} at frame {best_idx}")
        
        count += 1
        if count > 2000: break # Scan first 2000 frames
        
    if best_frame is not None:
        output_path = "logs/debug_frames_7gt/best_closed_eyes.png"
        cv2.imwrite(output_path, best_frame)
        print(f"Saved best closed eye frame ({best_idx}) to {output_path}")

    cap.release()

if __name__ == "__main__":
    find_closed_eyes_7gt()
