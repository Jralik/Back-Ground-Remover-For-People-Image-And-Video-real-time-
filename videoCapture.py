import cv2
import torch
import torch.nn as nn
import numpy as np
import threading

# ====== Model U-Net ======
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        down = self.conv(x)
        return down, self.pool(down)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x1, x2], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)
        d4, p4 = self.down4(p3)
        bn = self.bottleneck(p4)
        u1 = self.up1(bn, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        out = self.out(u4)
        return torch.sigmoid(out)

# ====== Webcam Thread ======
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ====== Load model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load("unet_person_segmentation.pth", map_location=device))
model.eval()

# ====== Run segmentation ======
stream = VideoStream(0)
print("Press 'q' to quit...")

while True:
    frame = stream.read()
    if frame is None:
        continue

    # Resize nhỏ để nhanh hơn
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (256, 256))
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    mask = (mask > 0.5).astype(np.uint8)

    # Áp dụng mask để giữ lại người
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = np.ones_like(frame, dtype=np.uint8) * 255  # nền trắng
    final = np.where(mask[..., None] == 1, fg, bg)

    cv2.imshow("U-Net Remove Background", final)

    # Thoát khi bấm 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
