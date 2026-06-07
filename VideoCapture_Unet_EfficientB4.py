import cv2
import torch
import numpy as np
import time
import segmentation_models_pytorch as smp
from torchvision import transforms

# ==== 1️⃣ Load model ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load("best.pth", map_location=DEVICE))
model.eval()

print(f"✅ Model loaded on: {DEVICE}")

# ==== 2️⃣ Transform ====
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
])

# ==== 3️⃣ Hàm xử lý frame ====
def segment_frame(frame):
    orig_h, orig_w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()

    mask = cv2.resize(mask, (orig_w, orig_h))
    mask = (mask > 0.7).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    result = (frame * mask_blur[..., None]).astype(np.uint8)
    mask_color = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return result, mask_color

# ==== 4️⃣ Chạy webcam ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("🎥 Nhấn Q để thoát...")

# Dùng biến để làm mượt FPS hiển thị
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình.")
        break

    start_time = time.time()

    frame = cv2.flip(frame, 1)
    segmented, mask_color = segment_frame(frame)

    h, w = frame.shape[:2]
    mask_color = cv2.resize(mask_color, (w, h))
    segmented = cv2.resize(segmented, (w, h))

    top_row = np.hstack((frame, mask_color))
    bottom_row = cv2.resize(segmented, (top_row.shape[1], h))
    final_display = np.vstack((top_row, bottom_row))
    display_resized = cv2.resize(final_display, (1280, 720))

    # ==== 🧮 Tính FPS ====
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # ==== 📝 Vẽ FPS lên ảnh ====
    cv2.putText(display_resized, f"FPS: {fps:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Original | Mask (Top) | Segmentation (Bottom)", display_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
