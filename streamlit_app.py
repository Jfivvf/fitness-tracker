import streamlit as st
import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt

KPT_NAMES = {
    11: 'left_hip',
    13: 'left_knee',
    15: 'left_ankle',
    12: 'right_hip',
    14: 'right_knee',
    16: 'right_ankle'
}

@st.cache_resource
def load_model():
    return YOLO('yolo11n-pose.pt')

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(cosine_angle))

def process_frame(frame, model):
    results = model(frame)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    return kpts, results[0].plot()

def analyze_squats(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    angles = []
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        kpts, plotted_frame = process_frame(frame, model)
        
        try:
            left_hip = kpts[11][:2]
            left_knee = kpts[13][:2]
            left_ankle = kpts[15][:2]
            angle = compute_angle(left_hip, left_knee, left_ankle)
            angles.append(angle)
        except:
            angles.append(0)
        
        processed_frames.append(plotted_frame)
    
    cap.release()
    return processed_frames, angles

def count_repetitions(angles):
    smoothed = savgol_filter(angles, window_length=21, polyorder=3)
    
    peaks, _ = find_peaks(-np.array(smoothed), height=-100, distance=20, prominence=10)
    return len(peaks), smoothed, peaks

st.title("🏋️ Squat Counter")
st.write("Загрузите видео для анализа приседаний")

uploaded_file = st.file_uploader("Выберите видео...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Анализировать"):
        with st.spinner('Обработка видео...'):
            frames, angles = analyze_squats(tfile.name)
            count, smoothed, peaks = count_repetitions(angles)
            
            st.success(f"**Количество приседаний:** {count}")
            
            fig, ax = plt.subplots()
            ax.plot(smoothed, label="Угол в колене")
            ax.plot(peaks, smoothed[peaks], "xr", label="Пики")
            ax.set_xlabel("Кадры")
            ax.set_ylabel("Угол (градусы)")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Обработанные кадры")
            for frame in frames[::len(frames)//10]:  # Показываем каждый 10-й кадр
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        caption="Обработанный кадр", use_column_width=True)
