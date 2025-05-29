from dataclasses import dataclass, field
import streamlit as st
import cv2
import numpy as np
import tempfile
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp
import os
import sys
import absl.logging
import logging
from collections import deque
import time


stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
sys.stderr = stderr

# MEDIAPIPE
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

KEYPOINT_INDICES = {
    'nose': 0, 'left_eye': 2, 'right_eye': 5,
    'left_ear': 7, 'right_ear': 8,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

EXERCISES = {
    'Squats': {
        'points': ['left_hip', 'left_knee', 'left_ankle'],
        'threshold': 160,  # Угол в колене в ВЕРХНЕЙ точке
        'min_angle': 60,   # Угол в колене в НИЖГНЙ точки
        'knee_over_toes_threshold': 0.15,  # РАЗНИЦА КОЛЕНИ - СТОПЫ
        'speed_threshold': 0.5,  # Максимальная скорость опускания-подъема
        'min_hip_knee_diff': -0.05,  # Минимальная разница между ТАЗОМ и КОЛЕНОМ
        'profile_shoulder_threshold': 0.2  # ПОРОГ определения положения в профиль
    }
}

# МОДЕЛЬ
@st.cache_resource
def load_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        enable_segmentation=False,
        smooth_landmarks=True
    )

pose = load_model()

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(cosine_angle))

def draw_landmarks(image, landmarks):
    #КЛЮЧЕВЫЕ ТОЧКИИ
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=3, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=3)
    )

def is_profile_view(kpts, frame_width):
    #ПРОВЕРКА НА ТО, СТОИТ ЛИ ЧЕЛОВЕК В ПРОФИЛЬ
    left_shoulder = kpts[KEYPOINT_INDICES['left_shoulder']]
    right_shoulder = kpts[KEYPOINT_INDICES['right_shoulder']]
   
    #НАДО ИСПРАВИТЬ ЧАСТЬ НИЖЕ - ИНОГДА ПУТАЕТ ПРОФИЛЬ И АНФАС

    shoulder_diff = abs(left_shoulder[0] - right_shoulder[0]) / frame_width
    # ЕСЛИ ПЛЕЧИ БЛИЗКО - ТО ПРОФИЛЬ
    return shoulder_diff < EXERCISES['Squats']['profile_shoulder_threshold']

def is_valid_squat(kpts, exercise_config, state):
    #УСЛОВИЯ ДЛЯ ПРОВЕРКИ НА КОРРЕКТНОСТ ПРИСЕДАНИЧ
    hip_idx = KEYPOINT_INDICES['left_hip']
    knee_idx = KEYPOINT_INDICES['left_knee']
    ankle_idx = KEYPOINT_INDICES['left_ankle']
    
    # ПРОВЕРКА НАЛИЧИЯ ВСЕХ КЛЮЧЕВЫХ ТОЧЕК
    if (kpts[hip_idx][0] == 0 and kpts[hip_idx][1] == 0 or
        kpts[knee_idx][0] == 0 and kpts[knee_idx][1] == 0 or
        kpts[ankle_idx][0] == 0 and kpts[ankle_idx][1] == 0):
        return False
    

    hip_y = kpts[hip_idx][1]
    knee_y = kpts[knee_idx][1]
    ankle_y = kpts[ankle_idx][1]
    
    a, b, c = kpts[hip_idx], kpts[knee_idx], kpts[ankle_idx]
    knee_angle = compute_angle(a, b, c)
    
    # ГЛУБИНА ПРИСЕДАНИЯ
    min_angle_ok = knee_angle < exercise_config['min_angle']
    hip_knee_diff_ok = (hip_y - knee_y) > exercise_config['min_hip_knee_diff'] * state['frame_height']
    
    is_profile = is_profile_view(kpts, state['frame_width'])
    knee_x = kpts[knee_idx][0]
    ankle_x = kpts[ankle_idx][0]
    
    if is_profile:
        knee_over_toes_ok = (knee_x-ankle_x) < exercise_config['knee_over_toes_threshold'] * state['frame_width']
    else:
        knee_over_toes_ok = True
    
    current_time = time.time()
    if state.get('last_hip_y') is not None and state.get('last_hip_time') is not None:
        time_diff = current_time - state['last_hip_time']
        if time_diff > 0:
            speed = abs(hip_y - state['last_hip_y']) / time_diff / state['frame_height']
            speed_ok = speed < exercise_config['speed_threshold']
        else:
            speed_ok = True
    else:
        speed_ok = True
    
    state['last_hip_y'] = hip_y
    state['last_hip_time'] = current_time
    
    if not is_profile and (kpts[KEYPOINT_INDICES['right_knee']][0] > 0 and 
                          kpts[KEYPOINT_INDICES['right_ankle']][0] > 0):
        right_knee_angle = compute_angle(
            kpts[KEYPOINT_INDICES['right_hip']],
            kpts[KEYPOINT_INDICES['right_knee']],
            kpts[KEYPOINT_INDICES['right_ankle']]
        )
        symmetry_ok = abs(knee_angle - right_knee_angle) < 20
    else:
        symmetry_ok = True
    
    return all([min_angle_ok, hip_knee_diff_ok, knee_over_toes_ok, speed_ok, symmetry_ok])

def process_frame(frame, exercise, state):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if not results.pose_landmarks:
        return None, None
    
    h, w = frame.shape[:2]
    state['frame_width'] = w
    state['frame_height'] = h
    
    kpts = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark])
    
    required_points = [KEYPOINT_INDICES[p] for p in EXERCISES[exercise]['points']]
    if any(kpts[i][0] == 0 and kpts[i][1] == 0 for i in required_points):
        return None, None
    
    # УГОЛ
    points = [KEYPOINT_INDICES[p] for p in EXERCISES[exercise]['points']]
    a, b, c = kpts[points[0]], kpts[points[1]], kpts[points[2]]
    angle = compute_angle(a, b, c)
    
    if exercise == 'Squats':
        is_valid = is_valid_squat(kpts, EXERCISES[exercise], state)
    else:
        is_valid = True
    
    threshold = EXERCISES[exercise]['threshold']
    if angle > threshold + 5:
        state['stage'] = "down"
    if angle < threshold - 5 and state['stage'] == "down" and is_valid:
        state['stage'] = "up"
        state['counter'] += 1
    
    annotated_frame = frame.copy()
    draw_landmarks(annotated_frame, results.pose_landmarks)
    
    # ПОЛОЖЕНИЕ ЧЕЛОВЕКА
    is_profile = is_profile_view(kpts, w)
    view_text = "Profile" if is_profile else "Front/Back"
    view_color = (0, 255, 255) if is_profile else (255, 255, 0)
    
    # ТЕКСТ
    cv2.putText(annotated_frame, f"Count: {state['counter']}", (12, 42), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(annotated_frame, f"Count: {state['counter']}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    
    cv2.putText(annotated_frame, f"Angle: {angle:.1f}", (12, 82), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.putText(annotated_frame, f"Angle: {angle:.1f}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.putText(annotated_frame, f"View: {view_text}", (12, 122), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.putText(annotated_frame, f"View: {view_text}", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, view_color, 2)
    
    if exercise == 'Squats':
        validity_text = "Valid" if is_valid else "Invalid"
        validity_color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.putText(annotated_frame, f"Squat: {validity_text}", (12, 162), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(annotated_frame, f"Squat: {validity_text}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, validity_color, 2)
    
    return kpts, annotated_frame

#ВСЯКИЕ СТИЛИ

st.markdown(
    """
    <style>
    /* Основные стили приложения */
    .stApp {
        background-color: #97a897;
    }
    
    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #6a8a6a;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        font-family: 'Arial', sans-serif; /* Изменение шрифта */
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #5a7a5a;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a6b4a;
        color: white;
        border-bottom: 3px solid #ffd700; /* Желтая полоса вместо красного */
        font-weight: bold;
    }
    
    /* Остальные ваши стили */
    .repetition-counter {
        background-color: #4a6b4a;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 2px solid #3a553a;
        margin: 1.5rem 0;
        text-align: center;
        display: inline-block;
    }
    .repetition-counter span {
        font-size: 1.8rem;
        color: #ffd700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .exercise-title {
        color: #2d4d2d;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4a6b4a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏋️Фитнес-трекер")

upload_tab, live_stream_tab = st.tabs(["📁 Анализ видео", "📷 Live режим"])

with upload_tab:
    with st.container():
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        
        exercise = st.selectbox("Выберите упражнение", list(EXERCISES.keys()))
        st.markdown(f'<p class="exercise-title">Текущее упражнение: {exercise}</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "mov"])
        
        if uploaded_file and st.button("Анализировать", type="primary"):
            state = {
                'counter': 0, 
                'stage': None,
                'last_hip_y': None,
                'last_hip_time': None,
                'frame_width': None,
                'frame_height': None
            }
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            angles = []
            processed_frames = []
            counter = 0
            stage = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                result = process_frame(frame, exercise, state)
                if result is None:
                    continue
                    
                kpts, processed = result
                points = [KEYPOINT_INDICES[p] for p in EXERCISES[exercise]['points']]
                
                try:
                    a, b, c = kpts[points]
                    angle = compute_angle(a, b, c)
                    angles.append(angle)
                    
                    threshold = EXERCISES[exercise]['threshold']
                    if angle > threshold + 5:
                        stage = "down"
                    if angle < threshold - 5 and stage == "down":
                        stage = "up"
                        counter += 1
                    
                    cv2.putText(processed, f"Count: {counter}", (12, 42), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                    cv2.putText(processed, f"Count: {counter}", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    
                    cv2.putText(processed, f"Angle: {angle:.1f}", (12, 82), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    cv2.putText(processed, f"Angle: {angle:.1f}", (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    processed_frames.append(processed)
                
                except Exception as e:
                    print(f"Ошибка обработки кадра: {e}")
            
            cap.release()
            
            st.markdown(
                f'<div class="repetition-counter">Всего повторений: <span>{counter}</span></div>',
                unsafe_allow_html=True
            )

with live_stream_tab:
    with st.container():
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        
        exercise_live = st.selectbox("Упражнение", list(EXERCISES.keys()), key="live")
        st.markdown(f'<p class="exercise-title">Текущее упражнение: {exercise_live}</p>', unsafe_allow_html=True)
        
        class LiveState:
            def __init__(self):
                self.counter = 0
                self.stage = None
                self.last_hip_y = None
                self.last_hip_time = None
                self.frame_width = None
                self.frame_height = None
                self.pose = mp_pose.Pose(
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8
                )
        
        state = LiveState()
        
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            state.frame_width = w
            state.frame_height = h
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = state.pose.process(img_rgb)
            
            if not results.pose_landmarks:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            kpts = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark])
            
            # УГОЛ
            points = [KEYPOINT_INDICES[p] for p in EXERCISES[exercise_live]['points']]
            a, b, c = kpts[points[0]], kpts[points[1]], kpts[points[2]]
            angle = compute_angle(a, b, c)
            
            if exercise_live == 'Squats':
                is_valid = is_valid_squat(kpts, EXERCISES[exercise_live], state.__dict__)
            else:
                is_valid = True
            
            threshold = EXERCISES[exercise_live]['threshold']
            if angle > threshold + 5:
                state.stage = "down"
            if angle < threshold - 5 and state.stage == "down" and is_valid:
                state.stage = "up"
                state.counter += 1
            
            annotated_image = img.copy()
            draw_landmarks(annotated_image, results.pose_landmarks)
            
            is_profile = is_profile_view(kpts, w)
            view_text = "Profile" if is_profile else "Front/Back"
            view_color = (0, 255, 255) if is_profile else (255, 255, 0)
            
            cv2.putText(annotated_image, f"Count: {state.counter}", (12, 42), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(annotated_image, f"Count: {state.counter}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            
            cv2.putText(annotated_image, f"Angle: {angle:.1f}", (12, 82), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(annotated_image, f"Angle: {angle:.1f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            cv2.putText(annotated_image, f"View: {view_text}", (12, 122), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(annotated_image, f"View: {view_text}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, view_color, 2)
            
            if exercise_live == 'Squats':
                validity_text = "Valid" if is_valid else "Invalid"
                validity_color = (0, 255, 0) if is_valid else (0, 0, 255)
                cv2.putText(annotated_image, f"Squat: {validity_text}", (12, 162), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(annotated_image, f"Squat: {validity_text}", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, validity_color, 2)
            
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
        
        webrtc_ctx = webrtc_streamer(
            key="squats-counter",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 20}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        if webrtc_ctx.state.playing:
            st.markdown(
                f'<div class="repetition-counter">Текущее количество: <span>{state.counter}</span></div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
