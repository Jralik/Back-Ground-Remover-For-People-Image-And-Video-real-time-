import numpy as np
import cv2
import tensorflow as tf
import time

# ====== Cấu hình ======
MODEL_PATH = "final/BestLoss.tflite"
NUM_THREADS = 8 
BACKGROUND_COLOR = (0, 255, 0) # ## THÊM LẠI MÀU NỀN ##
# ========================

# ===== Load model =====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]['shape'][1:3]
print(f"📸 My Model input shape: {input_w}x{input_h}")

# ===== Khởi tạo webcam =====
cap = cv2.VideoCapture(0)
cv2.namedWindow("My Model Test (Single Thread)", cv2.WINDOW_NORMAL)

# ===== Biến tính FPS =====
pTime = 0
alpha = 0.9 
fps_smooth = 0
# ========================

while cap.isOpened():
    # Lấy thời gian bắt đầu khung hình
    cTime = time.time()
    
    success, frame = cap.read()
    if not success:
        break
    
    frame_for_model = cv2.flip(frame, 1)
    original_h, original_w = frame_for_model.shape[:2] # Lấy kích thước gốc
    
    # === BẮT ĐẦU ĐO THỜI GIAN ===
    start_time = time.time()
    
    # 1. Tiền xử lý (Pre-processing)
    input_frame = cv2.resize(frame_for_model, (input_w, input_h))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

    # 2. Inference (Suy luận)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # 3. Lấy kết quả (Post-processing)
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # === KẾT THÚC ĐO THỜI GIAN ===
    latency_ms = (time.time() - start_time) * 1000

    # Xử lý mask về kích thước gốc
    mask = cv2.resize(mask, (original_w, original_h))
    mask = (mask > 0.5).astype(np.uint8) * 255

    
    # ==========================================================
    # ## THÊM LẠI PHẦN TÁCH NỀN ##
    
    # 1. Tạo mask 3 kênh
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # 2. Tạo ảnh nền
    background = np.full(frame_for_model.shape, BACKGROUND_COLOR, dtype=np.uint8)
    
    # 3. Áp dụng mask
    result = np.where(mask_3ch == 255, frame_for_model, background)
    # ==========================================================
    
    # Tính FPS mượt
    fps = 1 / (cTime - pTime) if pTime > 0 else 0
    fps_smooth = fps_smooth * alpha + fps * (1 - alpha)
    pTime = cTime
    
    # Hiển thị thông tin
    # ## THAY ĐỔI: Vẽ lên 'result' thay vì 'frame_for_model' ##
    cv2.putText(result, f"Latency: {latency_ms:.1f} ms", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(result, f"FPS: {fps_smooth:.1f}", (20, 130),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # ## THAY ĐỔI: Hiển thị 'result' ##
    cv2.imshow("My Model Test (Single Thread)", result)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()