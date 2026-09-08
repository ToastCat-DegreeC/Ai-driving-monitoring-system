import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import config

class FusionModel:
    def __init__(self, weights_path=None):
        # Expecting 7 features: [HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta]
        self.num_features = 7
        self.bilstm_model = self._build_mha_fusion_model()
        if weights_path and os.path.exists(weights_path):
            try:
                self.bilstm_model.load_weights(weights_path)
                print(f"Loaded Fusion model weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load Fusion weights: {e}")

    def _build_mha_fusion_model(self):
        """
        Builds a Multi-Head Attention (MHA) based BiLSTM Fusion model.
        This provides better multi-modal feature prioritization than simple additive attention.
        """
        inputs = layers.Input(shape=(config.FUSION_SEQUENCE_LEN, self.num_features))
        
        # BiLSTM Layers for temporal feature extraction
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x) # (batch, 60, 64)
        
        # Multi-Head Attention Layer
        # num_heads=4, key_dim=16 means each head has 16 dimensions.
        mha_layer = layers.MultiHeadAttention(
            num_heads=config.MHA_HEADS, 
            key_dim=config.MHA_KEY_DIM, 
            name="mha_fusion"
        )
        
        # Self-attention: query, value, key are all from the BiLSTM sequence
        mha_output, att_scores = mha_layer(
            query=x, 
            value=x, 
            key=x, 
            return_attention_scores=True
        )
        
        # Residual Connection & Layer Normalization (Standard Transformer-style)
        x = layers.Add()([x, mha_output])
        x = layers.LayerNormalization()(x)
        
        # Context extraction: Global Average Pooling captures global sequence trends
        context = layers.GlobalAveragePooling1D()(x)
        
        # Decision Head
        d = layers.Dense(32, activation='relu')(context)
        d = layers.Dropout(0.2)(d)
        outputs = layers.Dense(1, activation='sigmoid', name="fatigue_output")(d)
        
        # For visualization, average weights across heads and query dimension
        # att_scores shape: (batch, heads, query_len, key_len)
        avg_weights = tf.reduce_mean(att_scores, axis=1) # (batch, 60, 60)
        avg_weights = tf.reduce_mean(avg_weights, axis=1, name="avg_attention_weights") # (batch, 60)
        
        model = models.Model(inputs=inputs, outputs=[outputs, avg_weights])
        
        # Compile model
        model.compile(optimizer='adam', loss=['mse', None], metrics={'fatigue_output': 'mae'})
        
        print(f"MHA Fusion Model Summary ({self.num_features} Features, {config.MHA_HEADS} Heads):")
        model.summary()
        return model

    def predict_fatigue(self, history_buffer):
        """
        Predicts fatigue and returns attention weights.
        Returns: (status, probability, attention_weights)
        """
        if not history_buffer or len(history_buffer[0]) != self.num_features:
            return "Initializing", 0.0, np.zeros(config.FUSION_SEQUENCE_LEN)

        sequence = np.array(history_buffer, dtype=np.float32)
        
        if len(sequence) < config.FUSION_SEQUENCE_LEN:
            pad_width = config.FUSION_SEQUENCE_LEN - len(sequence)
            sequence = np.pad(sequence, ((pad_width, 0), (0, 0)), mode='edge')
        else:
            sequence = sequence[-config.FUSION_SEQUENCE_LEN:]

        # Normalization (must match training)
        sequence[:, 0] = np.clip((sequence[:, 0] - config.MIN_HR) / (config.MAX_HR - config.MIN_HR), 0, 1)
        sequence[:, 2] = np.clip(sequence[:, 2] / config.MAR_MAX, 0, 1)
        sequence[:, 3] = np.clip(sequence[:, 3] / config.POSE_VAR_MAX, 0, 1)
        sequence[:, 4] = np.clip((sequence[:, 4] - config.SPO2_MIN) / (config.SPO2_MAX - config.SPO2_MIN), 0, 1)
        sequence[:, 5] = np.clip(sequence[:, 5] / config.HRV_MAX, 0, 1)
        # HRV Delta [normalized to 0-1, centered at 0.5]
        sequence[:, 6] = np.clip(sequence[:, 6], -1, 1) * 0.5 + 0.5
        
        sequence_batch = np.expand_dims(sequence, axis=0) 
        
        # Extract both probability and weights
        pred_prob, pred_weights = self.bilstm_model.predict(sequence_batch, verbose=0)
        
        fatigue_probability = float(pred_prob[0][0])
        weights = pred_weights[0]

        if fatigue_probability > config.FATIGUE_THRESHOLD_CRITICAL:
            status = "CRITICAL"
        elif fatigue_probability > config.FATIGUE_THRESHOLD_FATIGUED:
            status = "Fatigued"
        else:
            status = "Alert"
            
        return status, fatigue_probability, weights
