import cv2
import os

def capture_self():
    output_dir = "logs/self_captures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(0) # Open laptop camera
    
    states = [
        ("Alert", "Look directly at camera, eyes wide open."),
        ("Fatigued", "Look at camera, eyelids drooping/half-closed."),
        ("Critical", "Eyes completely closed (micro-sleep)."),
        ("Distracted", "Turn your head sharply to the left or right.")
    ]
    
    print("=== RESEARCHER SELF-CAPTURE TOOL ===")
    print("Instructions:")
    print("- A window will open showing your camera.")
    print("- For each state, perform the action and press 'S' to save the frame.")
    print("- Press 'Q' to skip a state or exit.\n")

    for state_name, instruction in states:
        print(f"NEXT STATE: [{state_name}]")
        print(f"Action: {instruction}")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Mirror the frame for easier positioning
            display_frame = cv2.flip(frame, 1)
            
            # Add instruction overlay
            cv2.putText(display_frame, f"State: {state_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'S' to Capture", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Researcher Capture Tool", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_path = os.path.join(output_dir, f"{state_name.lower()}.png")
                cv2.imwrite(save_path, frame) # Save the original non-mirrored frame
                print(f"SUCCESS: Saved {state_name} to {save_path}\n")
                break
            elif key == ord('q'):
                print(f"Skipping {state_name}...\n")
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Capture session complete. Images are in 'logs/self_captures/'.")

if __name__ == "__main__":
    capture_self()
