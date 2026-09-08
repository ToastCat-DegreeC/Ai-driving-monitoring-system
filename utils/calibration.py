import numpy as np
import config

class SystemCalibrator:
    def __init__(self, target_frames=config.CALIBRATION_FRAMES):
        self.target_frames = target_frames
        self.reset()

    def reset(self):
        self.is_active = True
        self.frame_count = 0
        self.sum_pitch = 0.0
        self.sum_yaw = 0.0
        self.sum_ear = 0.0
        self.offset_pitch = 0.0
        self.offset_yaw = 0.0
        self.offset_ear = 0.3 # Default baseline

    def update(self, current_pitch, current_yaw, current_ear):
        if not self.is_active:
            return False

        self.sum_pitch += current_pitch
        self.sum_yaw += current_yaw
        self.sum_ear += current_ear
        self.frame_count += 1

        if self.frame_count >= self.target_frames:
            self.offset_pitch = self.sum_pitch / self.frame_count
            self.offset_yaw = self.sum_yaw / self.frame_count
            self.offset_ear = self.sum_ear / self.frame_count
            self.is_active = False
            return True # Calibration completed
        return False

    def get_offsets(self):
        return self.offset_pitch, self.offset_yaw, self.offset_ear

    def get_progress(self):
        return int((self.frame_count / self.target_frames) * 100)
