# 🎥 Real-time Person Background Remover (Tách nền người thời gian thực)

Dự án thực hành môn học **Thị giác máy tính và ứng dụng**. Ứng dụng này sử dụng các mô hình học sâu (Deep Learning) thuộc họ **U-Net** và **DeepLabV3+** để phân đoạn ảnh người (Person Image Segmentation) và thực hiện tách/thay thế nền hoặc làm mờ nền trong thời gian thực từ webcam hoặc hình ảnh tải lên.

---

## 🌟 Tính năng nổi bật

- **Tách nền thời gian thực (Real-time):** Hỗ trợ xử lý trực tiếp từ camera/webcam với độ trễ thấp.
- **Đa dạng chế độ nền:**
  - Thay thế nền bằng màu đơn sắc (ví dụ: phông xanh lá cây - Green Screen).
  - Thay thế nền bằng hình ảnh tùy chọn (người dùng tải lên).
  - Làm mờ nền (Blur background - tạo hiệu ứng xóa phông dọc).
- **Tối ưu hóa hiệu năng:**
  - Hỗ trợ mô hình **TensorFlow Lite (TFLite)** với lượng hóa `Float16` để chạy mượt mà trên CPU/GPU của các máy tính cá nhân.
  - Áp dụng **Đa luồng (Multi-threading)** trong OpenCV để tách biệt luồng đọc camera và luồng dự đoán (inference), giúp tăng đáng kể FPS hiển thị và giảm giật lag.
- **Đa dạng kiến trúc mô hình:**
  - **U-Net** với các backbone: **MobileNetV2** (nhẹ, nhanh) và **EfficientNet-B4** (độ chính xác cao).
  - **DeepLabV3+** với backbone **MobileNetV2**.
- **Giao diện Web Streamlit:** Thân thiện, trực quan, dễ dàng tương tác và cấu hình các thông số nền trực tiếp.

---

## 📂 Cấu trúc thư mục dự án

```text
├── .venv/                      # Môi trường ảo Python
├── Deeplab/                    # Thư mục chứa các mô hình và mã nguồn kiểm thử DeepLabV3+
│   ├── Camera.py               # Kiểm thử DeepLab camera đơn luồng
│   ├── Camera2.py              # Kiểm thử DeepLab camera đa luồng
│   ├── deeplab_fined-tune-GD1.keras
│   ├── deeplab_fined-tune-GD2.keras
│   ├── deeplabv3plus_mobilenetv2.keras
│   └── deeplabv3plus_mobilenetv2.tflite
├── MobileNetV3/                # Thư mục chứa mô hình phân đoạn sử dụng MobileNetV3
│   ├── best_selfie_segmentation_model.h5
│   └── best_selfie_segmentation_model.keras
├── fine-tuned/                 # Thư mục chứa các mô hình U-Net được tinh chỉnh
│   ├── unet_decoder_finetune.h5
│   ├── unet_decoder_finetune2.h5
│   ├── unet_decoder_finetune2.tflite
│   └── unet_decoder_finetuned_best.tflite
├── final/                      # Thư mục chứa các mô hình tối ưu cuối cùng
│   ├── BestLoss.keras
│   ├── BestLoss.tflite         # Mô hình tối ưu được khuyên dùng cho CPU/TFLite
│   ├── UnetMobileV2_Last.keras
│   ├── my_model.tflite
│   ├── unetMobileNetV2_finetuned.keras
│   └── unetMobileNetV2_finetuned_best.tflite
├── app.py                      # Ứng dụng giao diện Web chạy bằng Streamlit
├── Camera_test.py              # Chương trình chạy webcam thời gian thực đa luồng (TFLite - Nền xanh)
├── test.py                     # Chương trình chạy webcam thời gian thực đơn luồng (TFLite)
├── videoCapture.py             # Chương trình chạy webcam sử dụng PyTorch (U-Net)
├── Unet_MobileV2(Chuan).py     # Chương trình chạy webcam sử dụng Keras (U-Net MobileNetV2)
├── VideoCapture_Unet_EfficientB4.py # Chương trình chạy webcam PyTorch sử dụng U-Net EfficientNet-B4
├── TfLiteConvert.py            # Script chuyển đổi mô hình từ định dạng .keras sang .tflite (Float16)
└── README.md                   # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Hệ điều hành: Windows / macOS / Linux
- Phiên bản Python: **3.9 - 3.11** (khuyến nghị)
- Nên có GPU hỗ trợ CUDA nếu chạy các mô hình PyTorch hoặc Keras nặng (không bắt buộc đối với TFLite).

### 2. Cài đặt các thư viện cần thiết

Mở terminal tại thư mục dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

*Nếu chưa có file `requirements.txt`, bạn có thể cài đặt thủ công các thư viện chính bằng lệnh sau:*

```bash
pip install opencv-python numpy tensorflow streamlit torch torchvision segmentation-models-pytorch Pillow
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Khởi chạy giao diện Web (Streamlit App)
Đây là cách dễ nhất để trải nghiệm ứng dụng với giao diện đồ họa trực quan:
```bash
streamlit run app.py
```
**Tính năng trong giao diện:**
- Chọn nguồn đầu vào là **Webcam** hoặc **Upload ảnh** từ máy tính.
- Tùy chọn kiểu nền: **Màu đơn sắc** (tùy chọn bảng màu), **Ảnh upload** (thay thế bằng ảnh nền của bạn), hoặc **Làm mờ** (Blur).
- Mô hình mặc định sử dụng: `fine-tuned/unet_decoder_finetune2.tflite`.

### 2. Chạy Webcam thời gian thực đa luồng (Tối ưu nhất)
Để đạt FPS cao nhất và độ trễ thấp nhất trên CPU, chạy script sử dụng đa luồng và mô hình TFLite:
```bash
python Camera_test.py
```
- Sử dụng mô hình: `final/BestLoss.tflite`
- Hiển thị FPS và độ trễ xử lý (Latency tính bằng miligiây) trực tiếp trên màn hình.
- Nhấn phím **`q`** để thoát chương trình.

### 3. Các chương trình kiểm thử webcam khác

#### Chạy TFLite đơn luồng (Single-thread):
```bash
python test.py
```

#### Chạy mô hình Keras U-Net (MobileNetV2):
```bash
python "Unet_MobileV2(Chuan).py"
```
*Chương trình này sẽ hiển thị cùng lúc 3 khung hình ghép lại: Ảnh gốc | Mask phân đoạn | Ảnh sau khi tách nền.*

#### Chạy mô hình PyTorch U-Net (EfficientNet-B4):
```bash
python VideoCapture_Unet_EfficientB4.py
```

#### Chạy mô hình PyTorch U-Net cơ bản:
```bash
python videoCapture.py
```

---

## 🔄 Chuyển đổi mô hình (.keras sang .tflite)

Nếu bạn vừa huấn luyện xong một mô hình Keras mới (`.keras` hoặc `.h5`) và muốn tối ưu hóa nó sang định dạng TFLite để chạy mượt mà trên các thiết bị cấu hình yếu:

1. Mở file `TfLiteConvert.py` và chỉnh sửa đường dẫn mô hình tại:
   ```python
   MODEL = "final/BestLoss.keras"
   TFLITE_OUT = "final/BestLoss.tflite"
   ```
2. Chạy lệnh:
   ```bash
   python TfLiteConvert.py
   ```
Mô hình sẽ được chuyển đổi sang dạng TFLite được lượng hóa `Float16` giúp giảm 1/2 dung lượng file và tăng tốc độ suy luận nhưng vẫn giữ nguyên độ chính xác.

