import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Realtime Background Remover", page_icon="🎥", layout="wide")

# ==============================
# Load TFLite Model
# ==============================
MODEL_PATH = "fine-tuned/unet_decoder_finetune2.tflite"

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]['shape'][1:3]

# ==============================
# Segmentation Function
# ==============================
def segment_person(frame):
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (mask > 0.6).astype(np.uint8)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    return mask

# ==============================
# UI Controls
# ==============================
st.title("🎥 Ứng dụng Tách Nền Realtime (Streamlit + TFLite)")
st.sidebar.header("⚙️ Cài đặt nền")
mode = st.sidebar.selectbox("Chọn kiểu nền", ["Màu", "Ảnh upload", "Làm mờ"])
color = st.sidebar.color_picker("Chọn màu nền", "#00ff00")
uploaded_bg = st.sidebar.file_uploader("Tải ảnh nền", type=["jpg", "png"])

source = st.radio("Chọn nguồn đầu vào:", ["Webcam", "Upload ảnh"])

# ==============================
# Xử lý theo nguồn
# ==============================
if source == "Upload ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png"])
    if uploaded_file:
        frame = np.array(Image.open(uploaded_file).convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mask = segment_person(frame)

        if mode == "Màu":
            bg = np.full(frame.shape, tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), dtype=np.uint8)
        elif mode == "Ảnh upload" and uploaded_bg:
            bg = np.array(Image.open(uploaded_bg).resize((frame.shape[1], frame.shape[0])))
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:  # Làm mờ
            bg = cv2.GaussianBlur(frame, (55, 55), 0)

        result = np.where(mask[..., None] == 1, frame, bg)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Kết quả", use_container_width=True)

else:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Không thể truy cập camera.")
            break

        mask = segment_person(frame)

        if mode == "Màu":
            bg = np.full(frame.shape, tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), dtype=np.uint8)
        elif mode == "Ảnh upload" and uploaded_bg:
            bg = np.array(Image.open(uploaded_bg).resize((frame.shape[1], frame.shape[0])))
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            bg = cv2.GaussianBlur(frame, (55, 55), 0)

        result = np.where(mask[..., None] == 1, frame, bg)
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()