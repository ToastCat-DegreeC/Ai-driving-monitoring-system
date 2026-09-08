import cv2

class VideoProcessor:
    def __init__(self):
        pass

    def load_video_source(self, source=0):
        """
        Loads a video source (webcam or file).
        :param source: 0 for default webcam, or path to video file.
        :return: cv2.VideoCapture object.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}.")
            return None
        return cap

    def display_frame(self, window_name, frame):
        """
        Displays a video frame.
        :param window_name: Name of the display window.
        :param frame: The frame to display.
        """
        cv2.imshow(window_name, frame)

    def release_video_source(self, cap):
        """
        Releases the video capture object and destroys all OpenCV windows.
        :param cap: cv2.VideoCapture object.
        """
        cap.release()
        cv2.destroyAllWindows()

    # Add other utility functions related to video processing here
    # e.g., frame resizing, color conversion, drawing overlays, etc.

def read_frames_from_video(video_path, max_frames=None):
    """
    Reads frames from a video file.
    :param video_path: Path to the video file.
    :param max_frames: Maximum number of frames to read. If None, reads all frames.
    :return: List of frames (numpy arrays), or empty list if video cannot be opened.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
            
    cap.release()
    return frames
