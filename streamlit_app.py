from collections import deque
from dataclasses import dataclass, field
import streamlit as st
import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import tempfile
import matplotlib.pyplot as plt
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from typing import List, Optional
import time


EXERCISES = {
    'Squats': {'left': ('left_hip','left_knee','left_ankle'), 'right': ('right_hip','right_knee','right_ankle')},
    'Pull-ups': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Push-ups': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Crunches': {'left': ('left_shoulder','left_hip','left_knee'), 'right': ('right_shoulder','right_hip','right_knee')},
    'Barbell Biceps Curl': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Bench Press': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Chest Fly': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Deadlift': {'left': ('left_shoulder','left_hip','left_knee'), 'right': ('right_shoulder','right_hip','right_knee')},
    'Hammer Curl': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Hip Thrust': {'left': ('left_shoulder','left_hip','left_knee'), 'right': ('right_shoulder','right_hip','right_knee')},
    'Lat Pulldown': {'left': ('left_shoulder','left_elbow','left_wrist'), 'right': ('right_shoulder','right_elbow','right_wrist')},
    'Lateral Raise': {'left': ('left_elbow','left_shoulder','left_hip'), 'right': ('right_elbow','right_shoulder','right_hip')},
    'Leg Raises': {'left': ('left_shoulder','left_hip','left_knee'), 'right': ('right_shoulder','right_hip','right_knee')},
    'Russian Twist': {'left': ('left_shoulder','left_hip','left_knee'), 'right': ('right_shoulder','right_hip','right_knee')}
}

ANGLE_LABELS = {
    'Squats': 'Knee Angle',
    'Pull-ups': 'Elbow Angle',
    'Push-ups': 'Elbow Angle',
    'Crunches': 'Hip Angle',
    'Barbell Biceps Curl': 'Elbow Angle',
    'Bench Press': 'Elbow Angle',
    'Chest Fly': 'Elbow Angle',
    'Deadlift': 'Hip Angle',
    'Hammer Curl': 'Elbow Angle',
    'Hip Thrust': 'Hip Angle',
    'Lat Pulldown': 'Elbow Angle',
    'Lateral Raise': 'Shoulder Abduction Angle',
    'Leg Raises': 'Hip Angle',
    'Russian Twist': 'Torso Rotation Angle'
}

KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Specifies whether to detect minima ('min') or maxima ('max') for each exercise
COUNT_MODES = {
    'Lateral Raise': 'max',
    'Leg Raises': 'min',
    'Russian Twist': 'max'
}

@st.cache_resource
def load_model():
    # only here do we touch ultralytics/torch
    from ultralytics import YOLO
    return YOLO('yolo11n-pose.pt')

model = load_model()

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(cosine_angle))

def compute_torso_rotation(a, b):
    """Calculate torso rotation angle between two shoulders"""
    vec = b - a
    return np.degrees(np.arctan2(vec[1], vec[0]))

def process_frame(frame):
    results = model(frame)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    confs = results[0].keypoints.conf[0].cpu().numpy()
    return kpts, confs, results[0].plot()

def get_angle_data(kpts, confs, is_russian, left_idx, right_idx):
    try:
        if is_russian:
            ls = kpts[KEYPOINT_INDICES['left_shoulder']][:2]
            rs = kpts[KEYPOINT_INDICES['right_shoulder']][:2]
            torso_angle = compute_torso_rotation(ls, rs)
            confL = confs[KEYPOINT_INDICES['left_shoulder']]
            confR = confs[KEYPOINT_INDICES['right_shoulder']]
            angleL = torso_angle
            angleR = torso_angle
            angle = torso_angle
        else:
            aL, bL, cL = [kpts[i][:2] for i in left_idx]
            aR, bR, cR = [kpts[i][:2] for i in right_idx]
            # Проверка, что точки не равны (0, 0)
            for point in [aL, bL, cL, aR, bR, cR]:
                if np.array_equal(point, np.array([0, 0])):
                    return None, None, None, confL, confR
            angleL = compute_angle(aL, bL, cL)
            angleR = compute_angle(aR, bR, cR)
            confL = np.mean([confs[i] for i in left_idx])
            confR = np.mean([confs[i] for i in right_idx])
            if np.array_equal(aL, np.array([0, 0])) or np.array_equal(bL, np.array([0, 0])) or np.array_equal(cL, np.array([0, 0])):
                angleL = None
            if np.array_equal(aR, np.array([0, 0])) or np.array_equal(bR, np.array([0, 0])) or np.array_equal(cR, np.array([0, 0])):
                angleR = None
            if confL + confR > 1e-6:
                angle = (confL * angleL + confR * angleR) / (confL + confR)
                if angle is None:
                    angle = angleL if angleL is not None else angleR
            else:
                angle = None
        return angleL, angleR, angle, confL, confR
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

def analyze_exercise(video_path, exercise):
    # Special case for torso rotation in Russian Twist
    is_russian = (exercise == 'Russian Twist')
    if is_russian:
        twist_left_idx = KEYPOINT_INDICES['left_shoulder']
        twist_right_idx = KEYPOINT_INDICES['right_shoulder']
    cap = cv2.VideoCapture(video_path)

    anglesL = []
    anglesR = []
    confsL = []
    confsR = []
    aggr_angles = []
    processed_frames = []

    sides = EXERCISES[exercise]
    left_names = sides['left']
    right_names = sides['right']
    left_idx = [KEYPOINT_INDICES[name] for name in left_names]
    right_idx = [KEYPOINT_INDICES[name] for name in right_names]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kpts, confs, plotted_frame = process_frame(frame)

        angleL, angleR, angle, confL, confR = get_angle_data(kpts, confs, is_russian, left_idx, right_idx)
        if angle is None:
            continue
        anglesL.append(angleL)
        anglesR.append(angleR)
        confsL.append(confL)
        confsR.append(confR)
        aggr_angles.append(angle)

        processed_frames.append(plotted_frame)

    cap.release()
    return processed_frames, anglesL, anglesR, confsL, confsR, aggr_angles

def count_repetitions(angles, mode='min'):
    smoothed = savgol_filter(angles, window_length=21, polyorder=3)
    arr = np.array(smoothed)
    # Choose data to detect peaks: minima (-arr) or maxima (arr)
    data = -arr if mode == 'min' else arr
    peaks, _ = find_peaks(data, distance=20, prominence=10)
    return len(peaks), smoothed, peaks

from enum import Enum, auto


WINDOW_SIZE   = 21
POLY_ORDER    = 3

EMA_ALPHA       = 0.3
VEL_THRESH      = 1.5
ANGLE_DELTA_MIN = 20.0
HIGHLIGHT_FRM   = 10


class Phase(Enum):
    UNKNOWN = auto()
    DESCENT = auto()
    ASCENT  = auto()

@dataclass
class State:
    ema: Optional[float] = None
    prev_ema: Optional[float] = None
    phase: Phase = Phase.UNKNOWN
    top_angle: float = 0.0
    bottom_angle: float = 0.0
    count: int = 0
    highlight_until: int = 0
    angles: List[float] = field(default_factory=list)

def make_callback(exercise: str):
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise: {exercise}")

    sides = EXERCISES[exercise]
    left_idx  = [KEYPOINT_INDICES[n] for n in sides["left"]]
    right_idx = [KEYPOINT_INDICES[n] for n in sides["right"]]

    state = State()
    is_russian = (exercise == "Russian Twist")
    mode = COUNT_MODES.get(exercise, "min")  # 'min'|'max'

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        nonlocal state
        idx = len(state.angles)
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # --- 1. Pose ------------------------------------------------------
        try:
            kpts, confs, plotted = process_frame(img)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        if kpts is None or confs is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # --- 2. Angle -----------------------------------------------------
        angleL, angleR, angle_raw, *_ = get_angle_data(
            kpts, confs, is_russian, left_idx, right_idx)
        if angle_raw is None:
            canvas = plotted.copy()
            cv2.putText(canvas, f"Angle: N/A", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
            cv2.putText(canvas, f"Count: {state.count}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)
            return av.VideoFrame.from_ndarray(canvas, format="bgr24")

        angle = -angle_raw if mode == "max" else angle_raw
        state.angles.append(angle)

        # --- 3. EMA + velocity -------------------------------------------
        if state.ema is None:
            state.ema = angle
            state.prev_ema = angle
        else:
            state.prev_ema = state.ema
            state.ema = EMA_ALPHA * angle + (1 - EMA_ALPHA) * state.ema
        vel = state.ema - state.prev_ema  # °/кадр

        # --- 4. 2‑фазный FSM ---------------------------------------------
        if state.phase in (Phase.UNKNOWN, Phase.ASCENT):
            if vel < -VEL_THRESH:
                state.phase = Phase.DESCENT
                state.top_angle = state.ema
        if state.phase == Phase.DESCENT and vel > VEL_THRESH:
            state.phase = Phase.ASCENT
            state.bottom_angle = state.ema
            if (state.top_angle - state.bottom_angle) >= ANGLE_DELTA_MIN:
                state.count += 1
                state.highlight_until = idx + HIGHLIGHT_FRM
        if state.phase == Phase.ASCENT and vel < -VEL_THRESH:
            # начало нового спуска без завершения подъёма – перезаписать top
            state.phase = Phase.DESCENT
            state.top_angle = state.ema

        # --- 5. Overlay ---------------------------------------------------
        canvas = plotted
        if idx <= state.highlight_until:
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 255, 0), 10)
        cv2.putText(canvas, f"Angle: {angle_raw:.0f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Count: {state.count}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)
        cv2.putText(canvas, f"Phase: {state.phase.name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

    return callback

upload_tab, live_stream_tab = st.tabs(["Upload video", "Live Stream"])

with upload_tab:
    exercises = list(EXERCISES.keys())
    exercise = st.selectbox("Select exercise to analyze", exercises)
    st.title(f"🏋️ Repetition Counter: {exercise}")
    st.write(f"Upload video for {exercise} analysis")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.flush()

        st.video(uploaded_file)
        sides = EXERCISES[exercise]
        left_names = sides['left']
        right_names = sides['right']
        left_idx = [KEYPOINT_INDICES[name] for name in left_names]
        right_idx = [KEYPOINT_INDICES[name] for name in right_names]
        st.write(f"**Used keypoints:** {', '.join(left_names)} (indices {', '.join(map(str, left_idx))}); {', '.join(right_names)} (indices {', '.join(map(str, right_idx))})")

        if st.button("Analyze"):
            with st.spinner('Processing video...'):
                frames, anglesL, anglesR, confsL, confsR, angles = analyze_exercise(tfile.name, exercise)
                # Determine whether to detect minima or maxima for this exercise
                mode = COUNT_MODES.get(exercise, 'min')
                count, smoothed, peaks = count_repetitions(angles, mode)

                st.success(f"**Number of repetitions ({exercise}):** {count}")
                # Annotated video generation with persistent highlight
                is_peak = np.zeros(len(angles), dtype=bool)
                is_peak[peaks] = True
                cum_counts = np.cumsum(is_peak)
                cap_in = cv2.VideoCapture(tfile.name)
                fps = cap_in.get(cv2.CAP_PROP_FPS) or 30
                cap_in.release()
                h, w = frames[0].shape[:2]
                vfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(vfile.name, fourcc, fps, (w, h))
                # Highlight duration in seconds
                highlight_duration = 0.5
                highlight_len = int(fps * highlight_duration)
                highlight_until = -1
                for i, frame in enumerate(frames):
                    f = frame.copy()
                    angle_val = angles[i]
                    count_val = int(cum_counts[i])
                    if is_peak[i]:
                        highlight_until = i + highlight_len
                    if i <= highlight_until:
                        cv2.rectangle(f, (0,0), (w-1,h-1), (0,255,0), 10)
                    # Overlay angle text
                    cv2.putText(f, f"Angle: {angle_val:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    # Overlay count text
                    cv2.putText(f, f"Count: {count_val}", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    out.write(f)
                out.release()
                # Display annotated video
                st.video(vfile.name)

                with st.expander("Show plots"):
                    # Angles
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
                    ax1.plot(anglesL, label='Left Angle', color='blue', alpha=0.3)
                    ax1.plot(anglesR, label='Right Angle', color='red', alpha=0.3)
                    ax1.plot(smoothed, label='Total Angle', color='black')
                    ax1.plot(peaks, smoothed[peaks], 'x', color='red', label='Peaks')
                    ax1.set_ylabel('Angle (degrees)')
                    ax1.legend(loc='upper right')
                    # Confidence
                    ax2.plot(confsL, label='Confidence Left', color='blue', linestyle='--')
                    ax2.plot(confsR, label='Confidence Right', color='red', linestyle='--')
                    ax2.set_xlabel('Frames')
                    ax2.set_ylabel('Confidence')
                    ax2.legend(loc='upper right')
                    plt.tight_layout()
                    st.pyplot(fig)

with live_stream_tab:
    exercises_live = list(EXERCISES.keys())
    exercise_live = st.selectbox("Select exercise for live analysis", exercises_live, key="live_exercise")
    st.title(f"🏋️ Live Repetition Counter: {exercise_live}")
    st.write(f"Streaming live for {exercise_live} analysis")
    webrtc_ctx = webrtc_streamer(
        key="skeleton",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=make_callback(exercise_live),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
