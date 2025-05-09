import streamlit as st
import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import tempfile
import matplotlib.pyplot as plt

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

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(cosine_angle))

def compute_torso_rotation(a, b):
    """Calculate torso rotation angle between two shoulders"""
    vec = b - a
    return np.degrees(np.arctan2(vec[1], vec[0]))

def process_frame(frame, model):
    results = model(frame)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    confs = results[0].keypoints.conf[0].cpu().numpy()
    return kpts, confs, results[0].plot()

def analyze_exercise(video_path, exercise):
    # Special case for torso rotation in Russian Twist
    is_russian = (exercise == 'Russian Twist')
    if is_russian:
        twist_left_idx = KEYPOINT_INDICES['left_shoulder']
        twist_right_idx = KEYPOINT_INDICES['right_shoulder']
    model = load_model()
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

        kpts, confs, plotted_frame = process_frame(frame, model)

        try:
            if is_russian:
                # Compute torso rotation angle
                ls = kpts[twist_left_idx][:2]
                rs = kpts[twist_right_idx][:2]
                torso_angle = compute_torso_rotation(ls, rs)
                confL = confs[twist_left_idx]
                confR = confs[twist_right_idx]
                angleL = torso_angle
                angleR = torso_angle
                angle = torso_angle
            else:
                aL, bL, cL = [kpts[i][:2] for i in left_idx]
                aR, bR, cR = [kpts[i][:2] for i in right_idx]
                angleL = compute_angle(aL, bL, cL)
                angleR = compute_angle(aR, bR, cR)
                confL = np.mean([confs[i] for i in left_idx])
                confR = np.mean([confs[i] for i in right_idx])
                if confL + confR > 1e-6:
                    angle = (confL * angleL + confR * angleR) / (confL + confR)
                else:
                    angle = 0
            anglesL.append(angleL)
            anglesR.append(angleR)
            confsL.append(confL)
            confsR.append(confR)
            aggr_angles.append(angle)
        except:
            anglesL.append(0)
            anglesR.append(0)
            confsL.append(0)
            confsR.append(0)
            aggr_angles.append(0)

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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
