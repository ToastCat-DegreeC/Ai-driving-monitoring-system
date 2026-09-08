import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from scipy.signal import detrend
import config

class FeatureExtractor:
    def __init__(self, rppg_method='POS'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE
        )
        print("Initialized MediaPipe Face Mesh.")
        
        if rppg_method.upper() not in ['CHROM', 'POS']:
            raise ValueError(f"Invalid rPPG method: {rppg_method}. Choose 'CHROM' or 'POS'.")
        self.rppg_method = rppg_method.upper()
        print(f"rPPG method set to: {self.rppg_method}")

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)


    def detect_and_get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        landmarks = None
        rois = {'forehead': None, 'left_cheek': None, 'right_cheek': None}

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            img_h, img_w, _ = frame.shape
            face_top = landmarks.landmark[10].y * img_h
            face_bottom = landmarks.landmark[152].y * img_h
            face_height = face_bottom - face_top
            forehead_height_offset = face_height * 0.12
            
            # 1. Forehead
            forehead_pts = np.array([
                (landmarks.landmark[105].x * img_w, landmarks.landmark[105].y * img_h),
                (landmarks.landmark[334].x * img_w, landmarks.landmark[334].y * img_h),
                (landmarks.landmark[298].x * img_w, landmarks.landmark[298].y * img_h - forehead_height_offset),
                (landmarks.landmark[68].x * img_w, landmarks.landmark[68].y * img_h - forehead_height_offset),
            ], dtype=np.int32)
            rois['forehead'] = cv2.boundingRect(forehead_pts)
            
            # 2. Left Cheek
            l_cheek_pts = np.array([
                (landmarks.landmark[447].x * img_w, landmarks.landmark[447].y * img_h),
                (landmarks.landmark[345].x * img_w, landmarks.landmark[345].y * img_h),
                (landmarks.landmark[346].x * img_w, landmarks.landmark[346].y * img_h),
                (landmarks.landmark[280].x * img_w, landmarks.landmark[280].y * img_h),
            ], dtype=np.int32)
            rois['left_cheek'] = cv2.boundingRect(l_cheek_pts)

            # 3. Right Cheek
            r_cheek_pts = np.array([
                (landmarks.landmark[227].x * img_w, landmarks.landmark[227].y * img_h),
                (landmarks.landmark[116].x * img_w, landmarks.landmark[116].y * img_h),
                (landmarks.landmark[117].x * img_w, landmarks.landmark[117].y * img_h),
                (landmarks.landmark[50].x * img_w, landmarks.landmark[50].y * img_h),
            ], dtype=np.int32)
            rois['right_cheek'] = cv2.boundingRect(r_cheek_pts)

        return landmarks, rois

    def _clip_roi(self, roi, frame_shape):
        if roi is None: return None
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_shape[1] - x), min(h, frame_shape[0] - y)
        if w <= 0 or h <= 0: return None
        return (x, y, w, h)

    def _get_skin_mask(self, roi_frame):
        """
        Creates a binary mask for skin pixels using YCrCb color space.
        """
        if roi_frame is None or roi_frame.size == 0:
            return None
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
        # Academic thresholds for skin: Cr [133, 173], Cb [77, 127]
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        return mask

    def get_roi_means(self, frame, landmarks, rois):
        """
        Extracts RGB means for all 3 ROIs.
        Returns a dict: {'forehead': [R,G,B], ...}
        """
        results = {}
        for name, roi in rois.items():
            if roi:
                roi = self._clip_roi(roi, frame.shape)
                if roi:
                    x, y, w, h = roi
                    roi_frame = frame[y:y+h, x:x+w]
                    if roi_frame.size > 0:
                        mask = self._get_skin_mask(roi_frame)
                        skin_pixels = roi_frame[mask > 0]
                        if len(skin_pixels) > 10:
                            results[name] = np.mean(skin_pixels, axis=0)
                            continue
            results[name] = np.zeros(3)
        return results

    def get_ear_value(self, frame, landmarks):
        if landmarks:
            img_h, img_w, _ = frame.shape
            left_ear = self._calculate_ear(landmarks, config.LEFT_EYE_LANDMARKS, img_w, img_h)
            right_ear = self._calculate_ear(landmarks, config.RIGHT_EYE_LANDMARKS, img_w, img_h)
            return (left_ear + right_ear) / 2.0
        return 0.35

    def get_mar_value(self, frame, landmarks):
        if landmarks:
            img_h, img_w, _ = frame.shape
            top = np.array([landmarks.landmark[13].x * img_w, landmarks.landmark[13].y * img_h])
            bottom = np.array([landmarks.landmark[14].x * img_w, landmarks.landmark[14].y * img_h])
            left = np.array([landmarks.landmark[78].x * img_w, landmarks.landmark[78].y * img_h])
            right = np.array([landmarks.landmark[308].x * img_w, landmarks.landmark[308].y * img_h])
            vertical_dist = np.linalg.norm(top - bottom)
            horizontal_dist = np.linalg.norm(left - right)
            if horizontal_dist == 0: return 0.0
            return vertical_dist / horizontal_dist
        return 0.0

    def get_head_pose(self, frame, landmarks):
        if not landmarks: return None, None
        img_h, img_w, _ = frame.shape
        idx = config.POSE_LANDMARKS
        image_points = np.array([
            (landmarks.landmark[idx[0]].x * img_w, landmarks.landmark[idx[0]].y * img_h),
            (landmarks.landmark[idx[1]].x * img_w, landmarks.landmark[idx[1]].y * img_h),
            (landmarks.landmark[idx[2]].x * img_w, landmarks.landmark[idx[2]].y * img_h),
            (landmarks.landmark[idx[3]].x * img_w, landmarks.landmark[idx[3]].y * img_h),
            (landmarks.landmark[idx[4]].x * img_w, landmarks.landmark[idx[4]].y * img_h),
            (landmarks.landmark[idx[5]].x * img_w, landmarks.landmark[idx[5]].y * img_h)
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success: return None, None
            axis_len = 150.0
            axis_points = np.array([(axis_len, 0.0, 0.0), (0.0, axis_len, 0.0), (0.0, 0.0, axis_len)], dtype=np.float64)
            (nose_tip_2d, _) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (axis_2d, _) = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            viz_data = ((int(nose_tip_2d[0][0][0]), int(nose_tip_2d[0][0][1])), (int(axis_2d[0][0][0]), int(axis_2d[0][0][1])), (int(axis_2d[1][0][0]), int(axis_2d[1][0][1])), (int(axis_2d[2][0][0]), int(axis_2d[2][0][1])))
            rmat, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
            if sy > 1e-6:
                p, y, r = np.arctan2(rmat[2,1], rmat[2,2]), np.arctan2(-rmat[2,0], sy), np.arctan2(rmat[1,0], rmat[0,0])
            else:
                p, y, r = np.arctan2(-rmat[1,2], rmat[1,1]), np.arctan2(-rmat[2,0], sy), 0
            return (p * 180 / np.pi, y * 180 / np.pi, r * 180 / np.pi), viz_data
        except: return None, None

    def calculate_sqi(self, rppg_signal, fs=30):
        """
        Calculates Signal Quality Index based on SNR.
        """
        if rppg_signal is None or len(rppg_signal) < 64: return -15.0
        
        sig = detrend(rppg_signal)
        n_fft = 1024
        freqs = np.fft.rfftfreq(n_fft, 1/fs)
        fft_mags = np.abs(np.fft.rfft(sig, n=n_fft))
        
        pulse_mask = (freqs >= 0.75) & (freqs <= 3.0)
        noise_mask = ~pulse_mask & (freqs > 0.3)
        
        pulse_power = np.sum(fft_mags[pulse_mask]**2)
        noise_power = np.sum(fft_mags[noise_mask]**2)
        
        if noise_power == 0: return 10.0
        snr = 10 * np.log10(pulse_power / noise_power)
        return snr

    def calc_spo2(self, rgb_means_buffer):
        """
        Calculates SpO2 from the provided rgb_means buffer.
        If the buffer is a list of dicts (multi-ROI), it uses the first ROI available.
        """
        if not rgb_means_buffer: return 98.0
        
        # Extract means from first available ROI if using multi-ROI
        if isinstance(rgb_means_buffer[0], dict):
            means = []
            for item in rgb_means_buffer:
                if np.any(item['forehead'] > 0): means.append(item['forehead'])
                elif np.any(item['left_cheek'] > 0): means.append(item['left_cheek'])
                else: means.append(item['right_cheek'])
            rgb_means = np.array(means)
        else:
            rgb_means = np.array(rgb_means_buffer)

        if len(rgb_means) == 0: return 98.0
        r, b = rgb_means[:, 2], rgb_means[:, 0]
        dc_r, dc_b = np.mean(r), np.mean(b)
        if dc_r == 0 or dc_b == 0: return 98.0
        ac_r, ac_b = np.std(detrend(r)), np.std(detrend(b))
        if ac_r == 0 or ac_b == 0: return 98.0
        ror = (ac_r / dc_r) / (ac_b / dc_b)
        spo2 = 115.0 - (25.0 * ror)
        return max(config.SPO2_MIN, min(config.SPO2_MAX, spo2))

    def calc_rppg_signal(self, rgb_means_buffer):
        """
        Calculates the best rPPG signal by comparing SNRs across ROIs (ROI-Switching Attention).
        """
        if not rgb_means_buffer: return np.zeros(128)
        
        # If the buffer contains simple arrays (legacy or single ROI)
        if not isinstance(rgb_means_buffer[0], dict):
            rgb_means = np.array(rgb_means_buffer)
            return self._get_rppg_from_means(rgb_means)

        # Multi-ROI ROI-Switching Logic
        rois = ['forehead', 'left_cheek', 'right_cheek']
        best_snr = -999.0
        best_signal = np.zeros(len(rgb_means_buffer))
        
        for roi_name in rois:
            means = np.array([b[roi_name] for b in rgb_means_buffer])
            if np.all(means == 0): continue
            
            sig = self._get_rppg_from_means(means)
            snr = self.calculate_sqi(sig)
            
            if snr > best_snr:
                best_snr = snr
                best_signal = sig
        
        return best_signal

    def _get_rppg_from_means(self, rgb_means):
        if self.rppg_method == 'CHROM': return self._extract_rppg_chrom(rgb_means)
        return self._extract_rppg_pos(rgb_means)

    def _extract_rppg_chrom(self, rgb_means):
        r, g, b = rgb_means[:, 2], rgb_means[:, 1], rgb_means[:, 0]
        rd, gd, bd = detrend(r), detrend(g), detrend(b)
        if np.std(rd) == 0 or np.std(gd) == 0 or np.std(bd) == 0: return np.zeros(len(r))
        rn, gn, bn = (rd-np.mean(rd))/np.std(rd), (gd-np.mean(gd))/np.std(gd), (bd-np.mean(bd))/np.std(bd)
        return (3*rn - 2*gn) - (1.5*rn + gn - 1.5*bn)

    def _extract_rppg_pos(self, rgb_means):
        rgb_means = rgb_means.T
        mean_color = np.mean(rgb_means, axis=1)
        if np.any(mean_color == 0): return np.zeros(rgb_means.shape[1])
        col_nom = rgb_means / mean_color[:, np.newaxis]
        proj = np.dot(np.array([[0, 1, -1], [-2, 1, 1]]), col_nom)
        s1, s2 = proj[0], proj[1]
        std1, std2 = np.std(s1), np.std(s2)
        if std2 == 0: return np.zeros(rgb_means.shape[1])
        return s1 + (std1/std2) * s2

    def _calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        pts = np.array([(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in eye_indices])
        return (dist.euclidean(pts[1], pts[5]) + dist.euclidean(pts[2], pts[4])) / (2.0 * dist.euclidean(pts[0], pts[3]))
