import cv2
import numpy as np
import time
import os
import threading
import tensorflow as tf
import argparse
from datetime import datetime
from queue import Queue, Empty
from collections import deque
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from part_c.fusion_model import FusionModel
from utils.signal_processing import pad_or_truncate_signal
from utils.gui_helper import draw_dashboard
from utils.logger import SessionLogger
from utils.calibration import SystemCalibrator
import config

class AIBackgroundWorker(threading.Thread):
    def __init__(self, feature_extractor, enhancement_net, fusion_model):
        super().__init__(daemon=True)
        self.fe = feature_extractor
        self.en = enhancement_net
        self.fm = fusion_model
        self.input_queue = Queue(maxsize=1)
        self.results = {
            'hr': config.HR_STABLE_INITIAL,
            'perclos': 0.0,
            'spo2': 98.0,
            'hrv': 0.0,
            'fatigue_prob': 0.0,
            'status': "Calibrating...",
            'is_valid': False,
            'rppg_signal': [],
            'new_data': False
        }
        self.running = True
        self.fusion_history = deque(maxlen=config.FUSION_SEQUENCE_LEN)
        self.stable_hr = config.HR_STABLE_INITIAL
        self.hr_history = deque(maxlen=config.HR_HISTORY_LEN)
        self.smooth_spo2 = 98.0
        self.stable_hrv = 52.0
        self.last_plot_time = datetime.min

    def run(self):
        with tf.device('/cpu:0'):
            while self.running:
                try:
                    task_data = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    if task_data is None: continue
                    
                    rgb_means = task_data['rgb_means']
                    ear_values = task_data['ear_values']
                    is_yawning = task_data.get('is_yawning', False)
                    
                    rppg_sig = self.fe.calc_rppg_signal(rgb_means)
                    sqi_score = self.fe.calculate_sqi(rppg_sig)
                    is_valid = sqi_score > config.SQI_THRESHOLD
                    
                    rppg_prep = pad_or_truncate_signal(rppg_sig, config.HR_MODEL_INPUT_LENGTH)
                    ear_prep = pad_or_truncate_signal(ear_values, config.PERCLOS_MODEL_INPUT_LENGTH)
                    
                    # 1. Inference
                    raw_hr, perclos, hrv = self.en.enhance_features(rppg_prep, ear_prep)
                    spo2_ai = self.en.predict_spo2(rgb_means)
                    spo2_calc = self.fe.calc_spo2(rgb_means)
                    current_spo2 = (0.7 * spo2_calc) + (0.3 * spo2_ai)
                    
                    # 2. Stabilization
                    if is_valid:
                        if config.MIN_HR <= raw_hr <= config.MAX_HR:
                            diff = abs(raw_hr - self.stable_hr)
                            if diff <= config.MAX_HR_JUMP:
                                self.stable_hr = (config.HR_ALPHA * raw_hr) + (1 - config.HR_ALPHA) * self.stable_hr
                            else:
                                self.hr_history.append(raw_hr)
                                if len(self.hr_history) >= 3:
                                    avg_h = sum(list(self.hr_history)[-3:]) / 3.0
                                    if abs(avg_h - raw_hr) < 5.0:
                                        self.stable_hr = (config.HR_ALPHA * raw_hr) + (1 - config.HR_ALPHA) * self.stable_hr
                        self.smooth_spo2 = (config.SPO2_SMOOTHING * current_spo2) + (1 - config.SPO2_SMOOTHING) * self.smooth_spo2
                        # HRV smoothing & physiological artifact filtering (alert range: 45 - 65 ms)
                        if 15.0 <= hrv <= 95.0:
                            self.stable_hrv = (0.15 * hrv) + (0.85 * self.stable_hrv)
                        else:
                            self.stable_hrv = (0.05 * 52.0) + (0.95 * self.stable_hrv)
                    
                    # 3. Hybrid Fatigue Logic (Decision Engine)
                    task_fusion_history = task_data.get('fusion_history', [])
                    is_calibrating = task_data.get('is_calibrating', False)
                    
                    # Fix 1: Warm-up Suppression (Eliminates 99% initial spike)
                    if is_calibrating or len(task_fusion_history) < config.FUSION_SEQUENCE_LEN:
                        status = "Calibrating..." if is_calibrating else "Stabilizing..."
                        final_prob = 0.0
                        weights = np.zeros(config.FUSION_SEQUENCE_LEN)
                    else:
                        # Use the Fusion Model's internal prediction
                        status, final_prob, weights = self.fm.predict_fatigue(task_fusion_history)
                    
                    # Thesis Plotting Logic: Save plot if probability > 0.5 (once every 150 frames to avoid spam)
                    if final_prob > 0.5 and (datetime.now() - self.last_plot_time).total_seconds() > 5:
                        try:
                            from utils.visualization_helper import AttentionVisualizer
                            vis = AttentionVisualizer()
                            feat_buffer = np.array(task_fusion_history)
                            if len(feat_buffer) >= config.FUSION_SEQUENCE_LEN:
                                vis.plot_multimodal_summary(weights, feat_buffer[-config.FUSION_SEQUENCE_LEN:], final_prob, status)
                                self.last_plot_time = datetime.now()
                        except: pass

                    self.results.update({
                        'hr': self.stable_hr, 'perclos': perclos, 'spo2': self.smooth_spo2, 'hrv': self.stable_hrv,
                        'fatigue_prob': final_prob, 'attention_weights': weights.tolist(), 'status': status, 'is_valid': is_valid,
                        'rppg_signal': rppg_sig.tolist(), 'new_data': True
                    })
                except Exception as e:
                    print(f"Worker Inference Error: {e}")
                finally:
                    self.input_queue.task_done()

    def stop(self): self.running = False

def main():
    parser = argparse.ArgumentParser(description="AI Driving Monitoring System")
    parser.add_argument("--video", type=str, help="Path to video file (if not using webcam)")
    parser.add_argument("--cam", type=int, default=None, help="Index of the USB camera device (e.g. 0, 1, 2)")
    args = parser.parse_args()

    print("Starting AI Driving Monitoring System (Interactive Logic)...")
    feature_extractor = FeatureExtractor()
    enhancement_network = EnhancementNetwork(config.MODEL_PATH_HR, config.MODEL_PATH_PERCLOS, config.MODEL_PATH_SPO2)
    fusion_model = FusionModel(config.MODEL_PATH_FUSION)

    worker = AIBackgroundWorker(feature_extractor, enhancement_network, fusion_model)
    worker.start()

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            worker.stop()
            return
        cap = cv2.VideoCapture(args.video)
    else:
        # Priority: 1. Specified --cam index, 2. Index 0, 3. Index 1
        cam_index = args.cam if args.cam is not None else 0
        print(f"Attempting to open camera index: {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        
        if not cap.isOpened() and args.cam is None:
            print("Camera 0 failed, trying index 1...")
            cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source (Camera {args.cam if args.cam is not None else '0/1'} or File).")
        worker.stop()
        return

    # Attempt to request HD resolution from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Enable resizable high-DPI window
    cv2.namedWindow('AI Driving Monitoring System', cv2.WINDOW_NORMAL)

    rgb_means_buffer = deque(maxlen=config.RPPG_WINDOW_SIZE)
    ear_values_buffer = deque(maxlen=config.EAR_WINDOW_SIZE)
    pose_history = deque(maxlen=config.RPPG_WINDOW_SIZE)
    fusion_history = deque(maxlen=config.FUSION_SEQUENCE_LEN)
    
    logger = SessionLogger()
    
    dash_data = {
        'hr': 70.0, 'spo2': 98.0, 'hrv': 0.0, 'fatigue_prob': 0.0, 
        'perclos': 0.0,
        'status': "Initializing", 'distraction': "Focused", 
        'pulse_wave': [], 'valid': False,
        'attention_weights': []
    }
    
    calibrator = SystemCalibrator()
    last_mar, yawn_count, is_yawning = 0.0, 0, False
    smooth_p, smooth_y, p_off, y_off, ear_off = 0.0, 0.0, 0.0, 0.0, 0.3
    rp, ry = 0.0, 0.0
    face_lost_count = 0
    pose_var = 0.0
    current_perclos = 0.0
    prev_f_time, prev_s_time, frame_n = 0, 0, 0

    while True:
        t = time.time()
        fps = 1/(t-prev_f_time) if prev_f_time>0 else 0
        prev_f_time = t
        ret, frame = cap.read()
        if not ret: break
        frame_n += 1
        h, w, _ = frame.shape

        landmarks, rois = feature_extractor.detect_and_get_landmarks(frame)
        
        if t - prev_s_time >= config.SAMPLE_INTERVAL:
            prev_s_time = t
            
            if landmarks:
                face_lost_count = 0
                rgb_means_buffer.append(feature_extractor.get_roi_means(frame, landmarks, rois))
                
                # EAR Normalization based on calibration
                raw_ear = feature_extractor.get_ear_value(frame, landmarks)
                norm_ear = (raw_ear / ear_off) * 0.3 if ear_off > 0.01 else raw_ear
                ear_values_buffer.append(np.clip(norm_ear, 0.0, 0.4))
                
                last_mar = feature_extractor.get_mar_value(frame, landmarks)
                pose, viz = feature_extractor.get_head_pose(frame, landmarks)
                if pose:
                    pose_history.append(pose)
                    a = config.HEAD_POSE_SMOOTHING
                    smooth_p, smooth_y = (a*pose[0])+(1-a)*smooth_p, (a*pose[1])+(1-a)*smooth_y
                    
                    if calibrator.is_active:
                        calibrator.update(smooth_p, smooth_y, raw_ear)
                        dash_data['distraction'] = f"Calibrating {calibrator.get_progress()}%"
                        p_off, y_off, ear_off = calibrator.get_offsets()
                    else:
                        rp, ry = smooth_p-p_off, smooth_y-y_off
                        if abs(rp)>config.DISTRACTION_THRESH_PITCH: dash_data['distraction'] = "Distracted (Nod)"
                        elif abs(ry)>config.DISTRACTION_THRESH_YAW: dash_data['distraction'] = "Distracted (Turn)"
                        else: dash_data['distraction'] = "Focused"
                    if viz:
                        cv2.line(frame, viz[0], viz[1], (0,0,255), 2)
                        cv2.line(frame, viz[0], viz[2], (0,255,0), 2)
                        cv2.line(frame, viz[0], viz[3], (255,0,0), 2)
                
                # Yawn detection
                if last_mar > config.YAWN_THRESH: yawn_count += 1
                else: yawn_count = 0
                is_yawning = yawn_count > config.YAWN_FRAMES
                if is_yawning: dash_data['distraction'] = "Yawning"
            else:
                # Fix 3: Handle Face Loss / Head Turn Away
                face_lost_count += 1
                if face_lost_count > 5 and not calibrator.is_active:
                    dash_data['distraction'] = "Distracted (Turn/Away)"
                if len(rgb_means_buffer) > 0:
                    rgb_means_buffer.append(rgb_means_buffer[-1])
                if len(ear_values_buffer) > 0:
                    ear_values_buffer.append(ear_values_buffer[-1])

            # Calculate real-time PERCLOS (instantaneous closure ratio + CNN model)
            recent_ears = list(ear_values_buffer)[-config.FUSION_SEQUENCE_LEN:]
            if len(recent_ears) > 0:
                closed_ratio = float(np.mean([e < config.EAR_CLOSED_THRESH for e in recent_ears]))
            else:
                closed_ratio = 0.0
            
            is_eye_closed = bool(norm_ear < config.EAR_CLOSED_THRESH)
            instant_factor = 0.85 if is_eye_closed else 0.0
            current_perclos = max(closed_ratio, instant_factor, worker.results.get('perclos', 0.0))
            dash_data['perclos'] = current_perclos
            dash_data['eye_closed'] = is_eye_closed

            # Compute Head Pose Variance (dynamic variance + angular deviation)
            if len(pose_history) > 1:
                poses = np.array(list(pose_history)[-30:])
                dyn_var = float(np.mean(np.std(poses, axis=0)))
            else:
                dyn_var = 0.0

            if landmarks:
                pose_var = max(dyn_var, float(np.sqrt(rp**2 + ry**2)))
            elif face_lost_count > 5:
                pose_var = float(config.POSE_VAR_MAX * 0.75)
            else:
                pose_var = dyn_var

            # Fix 2: Frame-by-frame 2-second Context Window in fusion_history
            stable_hr = worker.results['hr']
            smooth_spo2 = worker.results['spo2']
            hrv = worker.results['hrv']
            hrv_delta = 0.0
            if len(fusion_history) > 0:
                hrv_delta = hrv - fusion_history[-1][5]

            current_features = [stable_hr, current_perclos, last_mar, pose_var, smooth_spo2, hrv, hrv_delta]
            fusion_history.append(current_features)

        # Send task to worker whenever worker is idle and buffer is ready
        if worker.input_queue.empty() and len(rgb_means_buffer) >= 64:
            worker.input_queue.put({
                'rgb_means': list(rgb_means_buffer), 
                'ear_values': list(ear_values_buffer),
                'fusion_history': list(fusion_history),
                'mar': last_mar,
                'pose_var': pose_var,
                'is_yawning': is_yawning,
                'is_calibrating': calibrator.is_active or (len(fusion_history) < config.FUSION_SEQUENCE_LEN)
            })

        if worker.results['new_data']:
            dash_data.update({
                'hr': worker.results['hr'], 'spo2': worker.results['spo2'], 'hrv': worker.results['hrv'],
                'perclos': current_perclos,
                'status': worker.results['status'], 'valid': worker.results['is_valid'],
                'fatigue_prob': worker.results['fatigue_prob'],
                'attention_weights': worker.results.get('attention_weights', []),
                'eye_closed': is_eye_closed
            })
            # Log the current state to CSV
            logger.log(dash_data)
            
            new_sig = worker.results['rppg_signal']
            if new_sig: dash_data['pulse_wave'] = new_sig[-config.GRAPH_LEN:]
            worker.results['new_data'] = False

        if landmarks:
            xc, yc = [lm.x*w for lm in landmarks.landmark], [lm.y*h for lm in landmarks.landmark]
            cv2.rectangle(frame, (int(min(xc)), int(min(yc))), (int(max(xc)), int(max(yc))), (255,0,0), 1)
        
        # Upscale frame if height < 720 so all dashboard panels & attention bars have ample vertical space
        target_h = 720
        if frame.shape[0] < target_h:
            scale = target_h / float(frame.shape[0])
            display_frame = cv2.resize(frame, (int(frame.shape[1] * scale), target_h))
        else:
            display_frame = frame

        final_frame = draw_dashboard(display_frame, dash_data)
        cv2.imshow('AI Driving Monitoring System', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
