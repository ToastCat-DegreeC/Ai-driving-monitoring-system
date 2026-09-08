import tensorflow as tf
from tensorflow.keras import layers, models
from utils.signal_processing import pad_or_truncate_signal, calculate_sdnn
import os
import numpy as np
from scipy.signal import butter, filtfilt
import config

class EnhancementNetwork:
    def __init__(self, weights_path_hr=None, weights_path_perclos=None, weights_path_spo2=None):
        self.rppg_input_length = config.HR_MODEL_INPUT_LENGTH
        self.ear_input_length = config.PERCLOS_MODEL_INPUT_LENGTH
        
        self.hr_model = self._build_1d_cnn_hr_model()
        self.perclos_model = self._build_1d_cnn_perclos_model()
        self.spo2_model = self._build_1d_cnn_spo2_model()

        if weights_path_hr and os.path.exists(weights_path_hr):
            self.hr_model.load_weights(weights_path_hr)
            print(f"Loaded HR model weights from {weights_path_hr}")

        if weights_path_perclos and os.path.exists(weights_path_perclos):
            self.perclos_model.load_weights(weights_path_perclos)
            print(f"Loaded PERCLOS model weights from {weights_path_perclos}")
            
        if weights_path_spo2 and os.path.exists(weights_path_spo2):
            self.spo2_model.load_weights(weights_path_spo2)
            print(f"Loaded SpO2 model weights from {weights_path_spo2}")

    def _build_1d_cnn_hr_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.rppg_input_length, 1)),
            layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_1d_cnn_perclos_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.ear_input_length, 1)),
            layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def _build_1d_cnn_spo2_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.rppg_input_length, 3)),
            layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def enhance_features(self, rppg_signal, facial_features):
        rppg_signal = np.array(rppg_signal)
        facial_features = np.array(facial_features)

        rppg_signal_processed = pad_or_truncate_signal(rppg_signal, self.rppg_input_length)
        facial_features_processed = pad_or_truncate_signal(facial_features, self.ear_input_length)
        
        rppg_signal_processed = np.array(rppg_signal_processed)
        facial_features_processed = np.array(facial_features_processed)

        try:
            rppg_signal_processed = self._butter_bandpass_filter(rppg_signal_processed, config.BP_LOW, config.BP_HIGH, config.FPS, order=3)
        except Exception as e:
            pass
            
        # Calculate HRV (SDNN) from filtered signal
        hrv = calculate_sdnn(rppg_signal_processed, fs=config.FPS)

        if np.std(rppg_signal_processed) > 1e-6:
             rppg_signal_processed = (rppg_signal_processed - np.mean(rppg_signal_processed)) / np.std(rppg_signal_processed)
        else:
             rppg_signal_processed = rppg_signal_processed - np.mean(rppg_signal_processed)

        rppg_signal_processed = np.array(rppg_signal_processed)
        facial_features_processed = np.array(facial_features_processed)

        rppg_signal_reshaped = rppg_signal_processed.reshape(1, -1, 1)
        facial_features_reshaped = facial_features_processed.reshape(1, -1, 1)

        heart_rate = self.hr_model.predict(rppg_signal_reshaped, verbose=0)[0][0]
        perclos = self.perclos_model.predict(facial_features_reshaped, verbose=0)[0][0]

        return heart_rate, perclos, hrv

    def predict_spo2(self, rgb_means_buffer):
        """
        Predicts SpO2 using the SpO2 CNN model.
        Handles both legacy (arrays) and multi-ROI (dicts) buffers.
        """
        if not rgb_means_buffer: return 98.0
        
        if isinstance(rgb_means_buffer[0], dict):
            # Extract means from first available ROI
            means = []
            for item in rgb_means_buffer:
                if np.any(item['forehead'] > 0): means.append(item['forehead'])
                elif np.any(item['left_cheek'] > 0): means.append(item['left_cheek'])
                else: means.append(item['right_cheek'])
            rgb_means = np.array(means)
        else:
            rgb_means = np.array(rgb_means_buffer)
        
        if len(rgb_means) < self.rppg_input_length:
             pad_width = self.rppg_input_length - len(rgb_means)
             rgb_means_processed = np.pad(rgb_means, ((0, pad_width), (0, 0)), mode='edge')
        else:
             rgb_means_processed = rgb_means[-self.rppg_input_length:]
             
        for i in range(3):
            std = np.std(rgb_means_processed[:, i])
            if std > 1e-6:
                rgb_means_processed[:, i] = (rgb_means_processed[:, i] - np.mean(rgb_means_processed[:, i])) / std
            else:
                rgb_means_processed[:, i] = rgb_means_processed[:, i] - np.mean(rgb_means_processed[:, i])
                
        rgb_means_reshaped = rgb_means_processed.reshape(1, self.rppg_input_length, 3)
        spo2 = self.spo2_model.predict(rgb_means_reshaped, verbose=0)[0][0]
        
        return max(config.SPO2_MIN, min(config.SPO2_MAX, spo2))
