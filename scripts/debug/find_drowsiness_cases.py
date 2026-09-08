import cv2
import mediapipe as mp
import numpy as np
import os

def find_best_drowsiness_images():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
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

    # 1. ALERT: Find image with MAX EAR in 'Non Drowsy'
    non_drowsy_dir = "drowsiness_dataset/Non Drowsy"
    best_alert_img = None
    max_ear = 0
    
    # Sample first 100 images for speed
    files = [f for f in os.listdir(non_drowsy_dir) if f.endswith('.png')][:100]
    for filename in files:
        img_path = os.path.join(non_drowsy_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            ear = calculate_ear(res.multi_face_landmarks[0], img.shape[1], img.shape[0])
            if ear > max_ear:
                max_ear = ear
                best_alert_img = img_path
    
    print(f"Best Alert Image: {best_alert_img} (EAR: {max_ear:.4f})")

    # 2. FATIGUED: Find image with Drooping EAR in 'Drowsy'
    drowsy_dir = "drowsiness_dataset/Drowsy"
    best_fatigued_img = None
    target_ear = 0.22 # Drooping but not closed
    min_diff = 1.0
    
    files = [f for f in os.listdir(drowsy_dir) if f.endswith('.png')][:100]
    for filename in files:
        img_path = os.path.join(drowsy_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            ear = calculate_ear(res.multi_face_landmarks[0], img.shape[1], img.shape[0])
            if abs(ear - target_ear) < min_diff:
                min_diff = abs(ear - target_ear)
                best_fatigued_img = img_path
                actual_fatigued_ear = ear
                
    print(f"Best Fatigued Image: {best_fatigued_img} (EAR: {actual_fatigued_ear:.4f})")

    # Save copies for easy access
    if best_alert_img:
        cv2.imwrite("logs/debug_frames_7gt/alert_final_open.png", cv2.imread(best_alert_img))
    if best_fatigued_img:
        cv2.imwrite("logs/debug_frames_7gt/fatigued_final_droop.png", cv2.imread(best_fatigued_img))

if __name__ == "__main__":
    find_best_drowsiness_images()
