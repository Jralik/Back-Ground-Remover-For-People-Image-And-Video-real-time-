import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time


# Bật GPU growth (để tránh chiếm hết VRAM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU(s) available: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ Không phát hiện GPU — đang dùng CPU.")

# ==== 1️⃣ Load model ====
model_path = "Deeplab\deeplabv3plus_mobilenetv2.keras"  # đường dẫn model của bạn
model = load_model(model_path, compile=False)
print("✅ Model loaded successfully!")

# ==== 2️⃣ Chuẩn bị kích thước ảnh ====
IMG_SIZE = 256  # thay bằng input size của mô hình bạn train (VD: 256 hoặc 320)

# ==== 3️⃣ Hàm xử lý 1 frame ====
def segment_frame(frame):
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Dự đoán mask
    pred = model.predict(img, verbose=0)[0]
    mask = (pred[..., 0] > 0.6).astype(np.uint8)  # Áp dụng ngưỡng
    mask = cv2.resize(mask, (orig_w, orig_h))

    # Làm mượt biên
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)

    # Tạo nền mờ (hoặc đen)
    background = np.zeros_like(frame)
    # background = cv2.GaussianBlur(frame, (51, 51), 0)  # nếu muốn làm mờ nền thay vì đen

    # Áp dụng mask
    result = (frame * mask_blur[..., None] + background * (1 - mask_blur[..., None])).astype(np.uint8)
    mask_color = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return result, mask_color

# ==== 4️⃣ Mở webcam ====
# 0: camera sau (hoặc mặc định), 1: camera trước (tùy thiết bị)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("🎥 Nhấn Q để thoát...")

# ==== 5️⃣ Vòng lặp realtime ====
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình.")
        break

    # Lật ngang (để giống gương, thường dùng cho cam trước)
    frame = cv2.flip(frame, 1)

    # Phân đoạn người
    segmented, mask_color = segment_frame(frame)

    # Ghép ảnh hiển thị
    top_row = np.hstack((frame, mask_color))
    bottom_row = cv2.resize(segmented, (top_row.shape[1], frame.shape[0]))
    final_display = np.vstack((top_row, bottom_row))

    # ==== 6️⃣ Tính FPS ====
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(final_display, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ==== 7️⃣ Hiển thị ====
    display_resized = cv2.resize(final_display, (1280, 720))
    cv2.imshow("Original | Mask | Segmentation", display_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
