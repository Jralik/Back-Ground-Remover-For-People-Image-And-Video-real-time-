import cv2
import numpy as np
import tensorflow as tf
import time

# ====== Cấu hình ======
MODEL_PATH = "Deeplab\deeplabv3plus_mobilenetv2.tflite"
NUM_THREADS = 8

# ====== Load model với GPU delegate (nếu khả dụng) ======
try:
    from tensorflow.lite.experimental import load_delegate
    # Thay 'libtensorflowlite_gpu_delegate.dll' bằng 'libtensorflowlite_gpu_delegate.so' nếu bạn dùng Linux
    gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.dll')
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[gpu_delegate],
        num_threads=NUM_THREADS
    )
    print("✅ GPU delegate loaded successfully!")
except Exception as e:
    print("⚠️ GPU delegate not available, using CPU instead.")
    print("Chi tiết lỗi:", e)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lấy Height và Width từ input_shape
input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

print(f"📸 Model input shape (H, W): ({input_h}, {input_w})")
print("🎥 Nhấn Q để thoát...")

# ====== Mở webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

prev_time = 0
fps = 0
latency_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lấy kích thước khung hình gốc
    original_h, original_w = frame.shape[:2]

    # Resize về kích thước đầu vào mô hình
    # Sửa lỗi: cv2.resize yêu cầu (Width, Height)
    input_frame = cv2.resize(frame, (input_w, input_h))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)  # ✅ đổi sang RGB
    input_tensor = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

    # === BẮT ĐẦU ĐO LATENCY ===
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000  # ms
    # === KẾT THÚC ĐO LATENCY ===

    # Dự đoán segmentation mask (kích thước là input_w, input_h)
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    # Xử lý mask
    # Resize mask về kích thước khung hình gốc (original_w, original_h)
    mask = cv2.resize(mask, (original_w, original_h))
    mask = (mask > 0.6).astype(np.uint8) * 255
    mask_3ch = cv2.merge([mask, mask, mask])

    # Nền tuỳ chỉnh
    background = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)
    result = np.where(mask_3ch == 255, frame, background)

    # === Tính FPS (thời gian khung hình) ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # === Hiển thị FPS & Latency (trên cửa sổ tách nền) ===
    cv2.putText(result, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(result, f"Latency: {latency_ms:.1f} ms", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # === THAY ĐỔI CHÍNH: Hiển thị 2 cửa sổ ===
    cv2.imshow("Cửa sổ Real (Gốc)", frame)
    cv2.imshow("Cửa sổ Tách nền (Kết quả)", result)

    # Thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()