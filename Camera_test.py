import numpy as np
import cv2
import tensorflow as tf
import time
import threading
# ====== Cấu hình ====== 
# Deeplab\deeplab_fined-tune-GD2.keras
# #fine-tuned/unet_decoder_finetune2.tflite 
# #fine-tuned/unet_decoder_finetuned_best.tflite 
# #selfie_segmentation_model.tflite #Unet_MobileV2.tflite 
# #final/unetMobileNetV2_finetuned_best.tflite 
# #final/unetMobileNetV2_finetuned.tflite 
# #final/my_model.tflite 
# final/BestLoss.tflite
# ====== Cấu hình ======
MODEL_PATH = "final/BestLoss.tflite"
NUM_THREADS = 8  # hợp lý cho CPU 8-16 luồng
BACKGROUND_COLOR = (0, 255, 0)  # nền xanh

# ===== Load model =====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]['shape'][1:3]
print(f"📸 Model input shape: {input_w}x{input_h}")

# ===== Khởi tạo webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

# ===== Shared state giữa 2 thread =====
latest_frame = None
result_frame = None
running = True

# ===== Thread chạy inference =====
def inference_thread():
    global latest_frame, result_frame, running
    prev_time = 0
    fps_smooth = 0
    alpha = 0.9  # hệ số làm mượt FPS

    while running:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        frame = latest_frame.copy()
        original_h, original_w = frame.shape[:2]

        # Chuẩn bị input
        input_frame = cv2.resize(frame, (input_w, input_h))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

        # do latency
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Lấy mask và xử lý
        mask = interpreter.get_tensor(output_details[0]['index'])[0]
        mask = cv2.resize(mask, (original_w, original_h))
        mask = (mask > 0.25).astype(np.uint8) * 255
        mask_3ch = cv2.merge([mask, mask, mask])

        # Áp nền
        background = np.full(frame.shape, BACKGROUND_COLOR, dtype=np.uint8)
        result = np.where(mask_3ch == 255, frame, background)

        # FPS mượt
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        fps_smooth = fps_smooth * alpha + fps * (1 - alpha)
        prev_time = curr_time

        # Hiển thị thông tin
        cv2.putText(result, f"FPS: {fps_smooth:.1f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2) 
        cv2.putText(result, f"Latency: {latency_ms:.1f} ms", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        result_frame = result

# ===== Bắt đầu thread inference =====
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# ===== Tạo cửa sổ toàn màn hình =====
cv2.namedWindow("Tách nền (Realtime)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tách nền (Realtime)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ===== Vòng lặp camera =====
print("🎥 Nhấn Q để thoát...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    latest_frame = frame

    if result_frame is not None:
        # Không resize, hiển thị full luôn
        cv2.imshow("Tách nền (Realtime)", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()   
cv2.destroyAllWindows()

