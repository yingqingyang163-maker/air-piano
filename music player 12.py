import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

print("🎹 Air Piano Stable Version 启动中...")

# =========================================================
# 音频系统
# =========================================================

fs = 44100

current_freq = 440.0
target_freq = 440.0

phase = 0.0

is_playing = False
muted = False

envelope_time = 0.0


def reset_envelope():
    global envelope_time
    envelope_time = 0.0


def envelope_guitar(t):
    attack = 0.003
    decay = 0.06
    sustain = 0.35

    if t < attack:
        return t / attack

    elif t < attack + decay:
        return 1.0 - (1.0 - sustain) * ((t - attack) / decay)

    else:
        return sustain


def audio_callback(outdata, frames, time_info, status):

    global current_freq
    global target_freq
    global phase
    global envelope_time
    global is_playing

    if muted or not is_playing:
        outdata[:] = np.zeros((frames, 1))
        return

    alpha = 0.15
    current_freq = current_freq * (1 - alpha) + target_freq * alpha

    delta = 2 * np.pi * current_freq / fs

    phases = phase + np.cumsum(np.ones(frames) * delta)

    phase = phases[-1] % (2 * np.pi)

    wave = (
        0.6 * np.sin(phases)
        + 0.35 * np.sin(2 * phases)
        + 0.2 * np.sin(3 * phases)
        + 0.12 * np.sin(4 * phases)
        + 0.06 * np.sin(5 * phases)
    )

    t_arr = envelope_time + np.arange(frames) / fs

    env = np.array([envelope_guitar(t) for t in t_arr])

    envelope_time = t_arr[-1]

    outdata[:] = (0.25 * wave * env).reshape(-1, 1)


stream = sd.OutputStream(
    callback=audio_callback,
    channels=1,
    samplerate=fs,
)

stream.start()

print("✅ 音频系统已启动")

# =========================================================
# 音符映射
# =========================================================

RIGHT_FINGER_TIPS = {
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

FINGER_ORDER = ['index', 'middle', 'ring', 'pinky']

RIGHT_FINGER_NOTES = {

    'upper': {
        'index': ('G4', 392.00),
        'middle': ('A4', 440.00),
        'ring': ('B4', 493.88),
        'pinky': ('C5', 523.25),
    },

    'lower': {
        'index': ('C4', 261.63),
        'middle': ('D4', 293.66),
        'ring': ('E4', 329.63),
        'pinky': ('F4', 349.23),
    }
}

# =========================================================
# 参数（稳定版）
# =========================================================

PINCH_ON = 0.38
PINCH_OFF = 0.46

Z_THRESHOLD = 0.08

DEBOUNCE_FRAMES = 5

HAND_LOSS_FRAMES = 6

DISC_RADIUS = 22

# =========================================================
# 工具函数
# =========================================================

def landmark_distance(a, b):

    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z

    return np.sqrt(dx * dx + dy * dy + dz * dz)


# =========================================================
# 状态变量
# =========================================================

left_counter = 0
left_confirmed = False

right_counter = 0
right_confirmed = False

active_right_finger = None

right_region = None

left_loss_counter = 0
right_loss_counter = 0

current_note_name = ""

# =========================================================
# MediaPipe
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

mp_draw = mp.solutions.drawing_utils

# =========================================================
# 摄像头
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FPS, 60)

cv2.namedWindow("Air Piano", cv2.WINDOW_NORMAL)

# =========================================================
# 主循环
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    mid_y = h // 2

    # =====================================================
    # 背景区域
    # =====================================================

    overlay = np.zeros_like(frame)

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, mid_y),
        (220, 140, 40),
        -1
    )

    cv2.rectangle(
        overlay,
        (0, mid_y),
        (w, h),
        (20, 160, 30),
        -1
    )

    frame = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

    # =====================================================
    # hand detection
    # =====================================================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    left_detected = False
    right_detected = False

    left_raw_pinch = False
    right_raw_pinch = False

    candidate_finger = None
    candidate_dist = 999

    if results.multi_hand_landmarks and results.multi_handedness:

        for handLms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # =============================================
            # 修复镜像左右手
            # =============================================

            raw_label = handedness.classification[0].label

            label = 'Right' if raw_label == 'Left' else 'Left'

            # =============================================
            # 画骨架
            # =============================================

            if label == 'Left':

                left_detected = True

                color1 = (0, 255, 0)
                color2 = (0, 180, 0)

            else:

                right_detected = True

                color1 = (255, 0, 0)
                color2 = (0, 0, 180)

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=color1,
                    thickness=2
                ),
                mp_draw.DrawingSpec(
                    color=color2,
                    thickness=2
                )
            )

            # =============================================
            # 手掌尺度
            # =============================================

            palm_size = landmark_distance(
                handLms.landmark[5],
                handLms.landmark[17]
            )

            # =============================================
            # 左手 pinch
            # =============================================

            if label == 'Left':

                idx = handLms.landmark[8]
                thb = handLms.landmark[4]

                dist = landmark_distance(idx, thb)

                normalized = dist / palm_size

                z_close = abs(idx.z - thb.z) < Z_THRESHOLD

                if left_confirmed:
                    left_raw_pinch = (
                        normalized < PINCH_OFF
                        and z_close
                    )
                else:
                    left_raw_pinch = (
                        normalized < PINCH_ON
                        and z_close
                    )

                ix = int(idx.x * w)
                iy = int(idx.y * h)

                cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)

            # =============================================
            # 右手
            # =============================================

            elif label == 'Right':

                index_y = handLms.landmark[8].y * h

                right_region = (
                    'upper'
                    if index_y < mid_y
                    else 'lower'
                )

                thumb = handLms.landmark[4]

                # =========================================
                # 只选最近的一根手指
                # =========================================

                candidate_finger = None
                candidate_dist = 999

                for fname in FINGER_ORDER:

                    tip_id = RIGHT_FINGER_TIPS[fname]

                    tip = handLms.landmark[tip_id]

                    dist = landmark_distance(tip, thumb)

                    normalized = dist / palm_size

                    z_close = abs(tip.z - thumb.z) < Z_THRESHOLD

                    # 手指伸直
                    finger_open = (
                        tip.y <
                        handLms.landmark[tip_id - 2].y
                    )

                    valid = (
                        z_close
                        and finger_open
                    )

                    if valid and normalized < candidate_dist:

                        candidate_dist = normalized
                        candidate_finger = fname

                    fx = int(tip.x * w)
                    fy = int(tip.y * h)

                    cv2.circle(
                        frame,
                        (fx, fy),
                        8,
                        (255, 0, 0),
                        -1
                    )

                # =========================================
                # pinch hysteresis
                # =========================================

                if candidate_finger is not None:

                    if right_confirmed:
                        right_raw_pinch = (
                            candidate_dist < PINCH_OFF
                        )
                    else:
                        right_raw_pinch = (
                            candidate_dist < PINCH_ON
                        )

    # =====================================================
    # hand loss
    # =====================================================

    if left_detected:
        left_loss_counter = 0
    else:
        left_loss_counter += 1

        if left_loss_counter > HAND_LOSS_FRAMES:
            left_confirmed = False
            left_counter = 0

    if right_detected:
        right_loss_counter = 0
    else:
        right_loss_counter += 1

        if right_loss_counter > HAND_LOSS_FRAMES:
            right_confirmed = False
            right_counter = 0
            active_right_finger = None

    # =====================================================
    # debounce
    # =====================================================

    # left

    if left_raw_pinch:
        left_counter = min(
            left_counter + 1,
            DEBOUNCE_FRAMES
        )
    else:
        left_counter = max(
            left_counter - 1,
            0
        )

    left_confirmed = (
        left_counter >= DEBOUNCE_FRAMES
    )

    # right

    if right_raw_pinch:
        right_counter = min(
            right_counter + 1,
            DEBOUNCE_FRAMES
        )
    else:
        right_counter = max(
            right_counter - 1,
            0
        )

    right_confirmed = (
        right_counter >= DEBOUNCE_FRAMES
    )

    if right_confirmed:
        active_right_finger = candidate_finger
    else:
        active_right_finger = None

    # =====================================================
    # 发声逻辑
    # =====================================================

    new_play = False

    new_freq = target_freq

    new_note = current_note_name

    # 左手 alone → A4

    if left_confirmed and not right_confirmed:

        new_play = True

        new_freq = 440.0

        new_note = "A4"

    # 双手 → 右手选音

    elif (
        left_confirmed
        and right_confirmed
        and active_right_finger is not None
    ):

        note_name, freq = RIGHT_FINGER_NOTES[
            right_region
        ][active_right_finger]

        new_play = True

        new_freq = freq

        new_note = note_name

    # =====================================================
    # 状态切换
    # =====================================================

    if new_play and not is_playing:

        target_freq = new_freq
        current_freq = new_freq

        current_note_name = new_note

        reset_envelope()

        is_playing = True

    elif new_play and is_playing:

        target_freq = new_freq

        current_note_name = new_note

    elif not new_play and is_playing:

        is_playing = False

    # =====================================================
    # UI
    # =====================================================

    panel_h = 80

    overlay_info = frame.copy()

    cv2.rectangle(
        overlay_info,
        (0, 0),
        (w, panel_h),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay_info,
        0.5,
        frame,
        0.5,
        0
    )

    # 左手状态

    if left_confirmed:

        cv2.putText(
            frame,
            "LEFT PINCH",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # 右手状态

    if right_confirmed:

        txt = f"RIGHT: {active_right_finger}"

        cv2.putText(
            frame,
            txt,
            (240, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    # 音符

    if is_playing:

        cv2.putText(
            frame,
            f"PLAYING: {current_note_name}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "SILENT",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (120, 120, 120),
            2
        )

    # 中央圆盘

    if is_playing:

        cv2.circle(
            frame,
            (w // 2, h // 2),
            DISC_RADIUS,
            (0, 255, 255),
            -1
        )

    cv2.imshow("Air Piano", frame)

    # =====================================================
    # 键盘
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord('m'):

        muted = not muted

        print("🔇 MUTE" if muted else "🔊 UNMUTE")

# =========================================================
# 清理
# =========================================================

cap.release()

stream.stop()
stream.close()

cv2.destroyAllWindows()

print("🎹 已退出")