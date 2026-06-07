# convert_h5_to_tflite.py
import tensorflow as tf
import numpy as np
import os

MODEL =  "final/BestLoss.keras" 
TFLITE_OUT = "final/BestLoss.tflite"
INPUT_SHAPE = (256, 256, 3)  # thay bằng input size mô hình của bạn

# load keras model
model = tf.keras.models.load_model(MODEL, compile=False)
print("Loaded model")

# Convert to TFLite (float16 quantization for speed & smaller size)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Float16 quantization (good balance GPU/CPU)
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)   
print("Saved TFLite model:", TFLITE_OUT)
