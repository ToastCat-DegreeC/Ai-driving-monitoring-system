import cv2
import mediapipe as mp
import numpy as np
import os

def check_y_files():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    # Mouth indices for MAR
    def calculate_mar(landmarks, img_w, img_h):
        def get_pt(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * img_w, lm.y * img_h])
        v1 = np.linalg.norm(get_pt(82) - get_pt(87))
        v2 = np.linalg.norm(get_pt(13) - get_pt(14))
        v3 = np.linalg.norm(get_pt(312) - get_pt(317))
        h = np.linalg.norm(get_pt(78) - get_pt(308))
        return (v1 + v2 + v3) / (3.0 * h)

    drowsy_dir = "drowsiness_dataset/Drowsy"
    y_files = [f for f in os.listdir(drowsy_dir) if f.lower().startswith('y')]
    
    if not y_files:
        print("No files starting with 'Y' found.")
        return

    print(f"Checking {len(y_files)} 'Y' files for yawning...")
    max_mar = 0
    best_img = None

    for filename in y_files[:50]: # Check first 50
        img_path = os.path.join(drowsy_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            mar = calculate_mar(res.multi_face_landmarks[0], img.shape[1], img.shape[0])
            if mar > max_mar:
                max_mar = mar
                best_img = img_path
    
    print(f"Max MAR found in 'Y' files: {max_mar:.4f} (Image: {best_img})")

if __name__ == "__main__":
    check_y_files()
