import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from part_a.cnn_model import build_eye_cnn
from part_b.enhancement_network import EnhancementNetwork
from part_c.fusion_model import FusionModel
import numpy as np
import config

def get_flops(model):
    # Calculate FLOPs using TensorFlow Profiler (approximate)
    # This is a bit hacky for TF 2.x Keras models but gives an estimate.
    # We use the concrete function.
    try:
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        # We need to wrap the model call in a function to trace it
        @tf.function
        def forward_pass(inputs):
            return model(inputs)

        # Create dummy input
        input_signature = [tf.TensorSpec(shape=(1, *l.shape[1:]), dtype=l.dtype) for l in model.inputs]
        concrete_func = forward_pass.get_concrete_function(*input_signature)
        
        graph = concrete_func.graph
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
        return 0

def main():
    print("Calculating FLOPs for models...")
    
    # 1. Eye CNN
    eye_cnn = build_eye_cnn()
    eye_flops = get_flops(eye_cnn)
    print(f"Eye CNN FLOPs: {eye_flops:,}")
    
    # 2. Enhancement Network (HR & PERCLOS)
    enhancement = EnhancementNetwork()
    hr_flops = get_flops(enhancement.hr_model)
    print(f"HR Model FLOPs: {hr_flops:,}")
    
    perclos_flops = get_flops(enhancement.perclos_model)
    print(f"PERCLOS Model FLOPs: {perclos_flops:,}")
    
    # 3. Fusion Model
    fusion = FusionModel(os.path.join(os.path.dirname(__file__), "../../models/fusion_model.h5"))
    # Fusion model input shape is (None, 2) but it uses LSTMs/Dense.
    # We need to build it first.
    # fusion.model is the Keras model.
    fusion_flops = get_flops(fusion.model)
    print(f"Fusion Model FLOPs: {fusion_flops:,}")
    
    # Total per second estimation
    # Eye CNN: 2 calls every 5 frames. At 30 FPS -> 6 calls/sec * 2 eyes = 12 calls/sec
    # HR Model: 1 call every 30 frames. At 30 FPS -> 1 call/sec
    # PERCLOS Model: 1 call every 30 frames. At 30 FPS -> 1 call/sec
    # Fusion Model: 1 call every 30 frames. At 30 FPS -> 1 call/sec
    
    total_flops_per_sec = (eye_flops * 12) + (hr_flops * 1) + (perclos_flops * 1) + (fusion_flops * 1)
    
    tflops = total_flops_per_sec / 1e12
    gflops = total_flops_per_sec / 1e9
    mflops = total_flops_per_sec / 1e6
    
    print("\n--- Performance Estimate (at 30 FPS) ---")
    print(f"Total FLOPs/sec: {total_flops_per_sec:,}")
    print(f"MFLOPS: {mflops:.4f}")
    print(f"GFLOPS: {gflops:.6f}")
    print(f"TFLOPS: {tflops:.12f}")

if __name__ == "__main__":
    main()
