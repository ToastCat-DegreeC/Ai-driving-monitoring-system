import cv2
import mediapipe as mp
import numpy as np
import os

def find_yawning_image():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    # Mouth indices for MAR
    MOUTH = [13, 14, 78, 308, 82, 312, 87, 317]
    
    def calculate_mar(landmarks, img_w, img_h):
        def get_pt(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * img_w, lm.y * img_h])
        
        # Vertical distances
        v1 = np.linalg.norm(get_pt(82) - get_pt(87))
        v2 = np.linalg.norm(get_pt(13) - get_pt(14))
        v3 = np.linalg.norm(get_pt(312) - get_pt(317))
        # Horizontal distance
        h = np.linalg.norm(get_pt(78) - get_pt(308))
        
        return (v1 + v2 + v3) / (3.0 * h)

    drowsy_dir = "drowsiness_dataset/Drowsy"
    best_mar_img = None
    max_mar = 0
    
    files = [f for f in os.listdir(drowsy_dir) if f.endswith('.png')]
    # Scan a wider range to find a good yawn
    for filename in files[:300]:
        img_path = os.path.join(drowsy_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            mar = calculate_mar(res.multi_face_landmarks[0], img.shape[1], img.shape[0])
            if mar > max_mar:
                max_mar = mar
                best_mar_img = img_path
    
    print(f"Best Yawn Image: {best_mar_img} (MAR: {max_mar:.4f})")
    
    if best_mar_img:
        cv2.imwrite("logs/debug_frames_7gt/yawn_final.png", cv2.imread(best_mar_img))

if __name__ == "__main__":
    find_yawning_image()
