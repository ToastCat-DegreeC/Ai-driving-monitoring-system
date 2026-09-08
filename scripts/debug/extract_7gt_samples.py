import cv2
import os

def extract_subject_7_frames():
    video_path = "Data/UBFC-rPPG/DATASET_1/7-gt/vid.avi"
    output_dir = "logs/debug_frames_7gt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Extract every 500 frames to find a good spot
    for i in range(0, total_frames, 500):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"frame_{i}.png"), frame)
            
    cap.release()
    print(f"Frames saved to {output_dir}")

if __name__ == "__main__":
    extract_subject_7_frames()
