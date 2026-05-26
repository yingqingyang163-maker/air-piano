import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

print("🎹 空气钢琴（琴弦模式·双手交互）启动中...")
print("👉 左手捏合触发 | 右手选择音高（4指分别对应4根琴弦）")
print("   上半区(蓝): G A B C5  |  下半区(绿): C D E F")
print("   仅左手捏合 → 中音A | 仅右手捏合 → 静音 | 同时捏合 → 右手选音")

# ================= 音频设置 =================
fs = 44100
muted = False
current_freq = 261.63
target_freq = 261.63
phase = 0.0
envelope_time = 0.0
is_playing = False

def reset_envelope():
    global envelope_time
    envelope_time = 0.0

def envelope_guitar(t):
    attack = 0.003
    decay = 0.06
    sustain_level = 0.35
    if t < attack:
        return t / attack
    elif t < attack + decay:
        return 1.0 - (1.0 - sustain_level) * ((t - attack) / decay)
    else:
        return sustain_level

def audio_callback(outdata, frames, time_info, status):
    global current_freq, target_freq, muted, is_playing, phase, envelope_time
    if muted or not is_playing:
        outdata[:] = np.zeros((frames, 1))
        return

    alpha = 0.2
    current_freq = current_freq * (1 - alpha) + target_freq * alpha

    t_arr = np.arange(frames) / fs
    delta_phase = 2 * np.pi * current_freq / fs
    phases = phase + np.cumsum(delta_phase * np.ones(frames))
    phase = phases[-1] % (2 * np.pi)
    wave = (0.6 * np.sin(phases) + 0.35 * np.sin(2 * phases) +
            0.2 * np.sin(3 * phases) + 0.12 * np.sin(4 * phases) +
            0.06 * np.sin(5 * phases) + 0.03 * np.sin(6 * phases))

    env_time_arr = envelope_time + t_arr
    envelope = np.array([envelope_guitar(t) for t in env_time_arr])
    envelope_time = env_time_arr[-1]

    outdata[:] = (0.25 * wave * envelope).reshape(-1, 1)

stream = sd.OutputStream(callback=audio_callback, channels=1, samplerate=fs)
stream.start()
print("✅ 音频系统就绪")

def landmark_distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return np.sqrt(dx*dx + dy*dy + dz*dz)

# ================= 常量 & 映射表 =================
RIGHT_FINGER_TIPS = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
FINGER_ORDER = ['index', 'middle', 'ring', 'pinky']

# 每个区域 + 手指 → (音符名, 频率)
RIGHT_FINGER_NOTES = {
    'lower': {
        'index':  ('C4', 261.63),
        'middle': ('D4', 293.66),
        'ring':   ('E4', 329.63),
        'pinky':  ('F4', 349.23),
    },
    'upper': {
        'index':  ('G4', 392.00),
        'middle': ('A4', 440.00),
        'ring':   ('B4', 493.88),
        'pinky':  ('C5', 523.25),
    },
}

# 琴弦定义（从上到下，共8根）
# y_frac: 归一化y坐标（0顶→1底），zone中心位置
STRINGS = [
    {'label': 'C5', 'y_frac': 1/16,  'region': 'upper', 'finger': 'pinky'},
    {'label': 'B',  'y_frac': 3/16,  'region': 'upper', 'finger': 'ring'},
    {'label': 'A',  'y_frac': 5/16,  'region': 'upper', 'finger': 'middle'},
    {'label': 'G',  'y_frac': 7/16,  'region': 'upper', 'finger': 'index'},
    {'label': 'F',  'y_frac': 9/16,  'region': 'lower', 'finger': 'pinky'},
    {'label': 'E',  'y_frac': 11/16, 'region': 'lower', 'finger': 'ring'},
    {'label': 'D',  'y_frac': 13/16, 'region': 'lower', 'finger': 'middle'},
    {'label': 'C',  'y_frac': 15/16, 'region': 'lower', 'finger': 'index'},
]

PINCH_ON = 0.38
PINCH_OFF = 0.45
DEBOUNCE_FRAMES = 5
DISC_RADIUS = 20

# ================= 手部状态 =================
left_pinch_counter = 0
left_pinch_confirmed = False

right_raw = {f: False for f in FINGER_ORDER}
right_counter = {f: 0 for f in FINGER_ORDER}
right_confirmed = {f: False for f in FINGER_ORDER}
right_distances = {f: 999.0 for f in FINGER_ORDER}
right_region = None
active_right_finger = None

HAND_LOSS_FRAMES = 5
left_loss_counter = 0
right_loss_counter = 0

current_note_name = ""
active_string_y = None

# ================= 手部检测 =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FPS, 60)
for _ in range(5):
    cap.read()

cv2.namedWindow("Air Piano 🎹", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Air Piano 🎹", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    mid_y = h // 2

    # ---------- 1. 绘制半透明区域背景 ----------
    alpha_bg = 0.55
    overlay = np.zeros_like(frame)
    # 上半区：蓝色 (BGR)
    cv2.rectangle(overlay, (0, 0), (w, mid_y), (220, 140, 40), -1)
    # 下半区：绿色 (BGR)
    cv2.rectangle(overlay, (0, mid_y), (w, h), (20, 160, 30), -1)
    frame = cv2.addWeighted(frame, 1 - alpha_bg, overlay, alpha_bg, 0)

    # ---------- 2. 绘制琴弦 ----------
    string_y_positions = {}
    for s in STRINGS:
        y = int(s['y_frac'] * h)
        string_y_positions[s['label']] = y
        cv2.line(frame, (50, y), (w - 50, y), (128, 128, 128), 2)
        cv2.putText(frame, s['label'], (10, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

    # ---------- 3. 手部检测 ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    left_detected = False
    right_detected = False
    left_raw_pinch = False
    right_index_y_norm = None

    if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
        for handLms, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
            raw_label = handedness.classification[0].label
            # 画面做了镜像翻转，MediaPipe的Left/Right需要交换
            label = 'Right' if raw_label == 'Left' else 'Left'

            # 计算手掌尺寸（索引指MCP到小指MCP的3D距离）
            palm_size = landmark_distance(handLms.landmark[5], handLms.landmark[17])
            if palm_size < 0.01:
                palm_size = 0.01

            if label == 'Left' and not left_detected:
                left_detected = True
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                      mp_draw.DrawingSpec(color=(0, 180, 0), thickness=2))

                idx = handLms.landmark[8]
                thb = handLms.landmark[4]
                dist = landmark_distance(idx, thb)
                normalized_dist = dist / palm_size
                z_close = abs(idx.z - thb.z) < 0.08
                if left_pinch_confirmed:
                    left_raw_pinch = normalized_dist < PINCH_OFF and z_close
                else:
                    left_raw_pinch = normalized_dist < PINCH_ON and z_close

                ix, iy = int(idx.x * w), int(idx.y * h)
                cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)
                cv2.circle(frame, (ix, iy), 6, (255, 255, 255), -1)

            elif label == 'Right' and not right_detected:
                right_detected = True
                right_index_y_norm = handLms.landmark[8].y

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                                      mp_draw.DrawingSpec(color=(0, 0, 180), thickness=2))

                thb = handLms.landmark[4]
                tx, ty = int(thb.x * w), int(thb.y * h)
                cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)

                # 只追踪距离拇指最近的指尖（逐个计算后取最小）
                candidate_finger = None
                candidate_dist = 999.0
                for fname in FINGER_ORDER:
                    tip = handLms.landmark[RIGHT_FINGER_TIPS[fname]]
                    dist = landmark_distance(tip, thb)
                    normalized_dist = dist / palm_size
                    if normalized_dist < candidate_dist:
                        candidate_dist = normalized_dist
                        candidate_finger = fname

                for fname in FINGER_ORDER:
                    right_distances[fname] = 999.0
                    right_raw[fname] = False

                if candidate_finger is not None:
                    tip = handLms.landmark[RIGHT_FINGER_TIPS[candidate_finger]]
                    dist = landmark_distance(tip, thb)
                    normalized_dist = candidate_dist
                    z_close = abs(tip.z - thb.z) < 0.08
                    # 手指张开检查：指尖应在其PIP关节之上
                    tip_id = RIGHT_FINGER_TIPS[candidate_finger]
                    finger_open = tip.y < handLms.landmark[tip_id - 2].y
                    right_distances[candidate_finger] = normalized_dist
                    if right_confirmed[candidate_finger]:
                        right_raw[candidate_finger] = (normalized_dist < PINCH_OFF and z_close and finger_open)
                    else:
                        right_raw[candidate_finger] = (normalized_dist < PINCH_ON and z_close and finger_open)

                for fname in FINGER_ORDER:
                    tip = handLms.landmark[RIGHT_FINGER_TIPS[fname]]
                    fx, fy = int(tip.x * w), int(tip.y * h)
                    if right_raw[fname]:
                        color = (0, 255, 255)
                    elif fname == candidate_finger:
                        color = (0, 140, 255)
                    else:
                        color = (255, 0, 0)
                    cv2.circle(frame, (fx, fy), 9, color, -1)
                    cv2.circle(frame, (fx, fy), 5, (255, 255, 255), -1)

    # ---------- 4. 手部消失宽限期 ----------
    if left_detected:
        left_loss_counter = 0
    else:
        left_loss_counter += 1
        if left_loss_counter > HAND_LOSS_FRAMES:
            left_pinch_counter = 0
            left_pinch_confirmed = False

    if right_detected:
        right_loss_counter = 0
    else:
        right_loss_counter += 1
        if right_loss_counter > HAND_LOSS_FRAMES:
            for f in FINGER_ORDER:
                right_raw[f] = False
                right_counter[f] = 0
                right_confirmed[f] = False
                right_distances[f] = 999.0
            right_region = None
            active_right_finger = None

    # ---------- 5. 区域判定 ----------
    if right_detected and right_index_y_norm is not None:
        right_region = 'upper' if (right_index_y_norm * h) < mid_y else 'lower'
    elif not right_detected:
        right_region = None

    # ---------- 6. 捏合去抖动 ----------
    # 左手
    if left_raw_pinch:
        left_pinch_counter = min(left_pinch_counter + 1, DEBOUNCE_FRAMES)
    else:
        left_pinch_counter = max(left_pinch_counter - 1, 0)
    left_pinch_confirmed = (left_pinch_counter >= DEBOUNCE_FRAMES)

    # 右手（4指独立）
    for fname in FINGER_ORDER:
        if right_raw[fname]:
            right_counter[fname] = min(right_counter[fname] + 1, DEBOUNCE_FRAMES)
        else:
            right_counter[fname] = max(right_counter[fname] - 1, 0)
        right_confirmed[fname] = (right_counter[fname] >= DEBOUNCE_FRAMES)

    # 确定活跃右手手指（已确认捏合 + 最小距离）
    active_right_finger = None
    min_d = 999.0
    for fname in FINGER_ORDER:
        if right_confirmed[fname] and right_distances[fname] < min_d:
            min_d = right_distances[fname]
            active_right_finger = fname
    right_any_pinch = (active_right_finger is not None)

    # ---------- 7. 发声逻辑 ----------
    new_is_playing = False
    new_target_freq = target_freq
    new_note_name = current_note_name
    new_active_string_y = active_string_y

    # 规则1：仅左手捏合 → 中音A
    if left_pinch_confirmed and not right_any_pinch:
        new_is_playing = True
        new_target_freq = 440.0
        new_note_name = 'A4'
        new_active_string_y = string_y_positions.get('A', None)

    # 规则3：双手同时捏合 → 右手决定音高
    elif left_pinch_confirmed and right_any_pinch and right_region is not None:
        new_is_playing = True
        note_name, freq = RIGHT_FINGER_NOTES[right_region][active_right_finger]
        new_target_freq = freq
        new_note_name = note_name
        # 找到对应琴弦的y坐标
        for s in STRINGS:
            if s['region'] == right_region and s['finger'] == active_right_finger:
                new_active_string_y = string_y_positions.get(s['label'], None)
                break

    # 规则2：仅右手捏合 → 静音
    # 规则4：都不捏合 → 静音
    else:
        new_is_playing = False

    # 状态转换
    if new_is_playing and not is_playing:
        # 上升沿：重置包络
        target_freq = new_target_freq
        current_freq = target_freq
        current_note_name = new_note_name
        active_string_y = new_active_string_y
        reset_envelope()
        is_playing = True
    elif new_is_playing and is_playing:
        # 持续中：平滑切换频率
        if new_target_freq != target_freq:
            target_freq = new_target_freq
            current_note_name = new_note_name
            active_string_y = new_active_string_y
    elif not new_is_playing and is_playing:
        # 下降沿：停止
        is_playing = False
        active_string_y = None

    # ---------- 8. 琴弦高亮（右手捏合 → 亮紫色）----------
    if right_region is not None and active_right_finger is not None:
        for s in STRINGS:
            if s['region'] == right_region and s['finger'] == active_right_finger:
                y = string_y_positions[s['label']]
                cv2.line(frame, (50, y), (w - 50, y), (255, 0, 255), 6)
                cv2.putText(frame, s['label'], (10, y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # ---------- 9. 碟形显示（左手捏合 + 发声 → 黄色碟形）----------
    if is_playing and left_pinch_confirmed and active_string_y is not None:
        cx = w // 2
        cy = active_string_y
        cv2.circle(frame, (cx, cy), DISC_RADIUS, (0, 255, 255), -1)
        cv2.circle(frame, (cx, cy), DISC_RADIUS, (0, 200, 200), 2)

    # ---------- 10. 信息面板 ----------
    panel_h = 70
    overlay_info = frame.copy()
    cv2.rectangle(overlay_info, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay_info, 0.55, frame, 0.45, 0)

    # 左手状态
    if left_pinch_confirmed:
        cv2.putText(frame, "L: PINCH", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "L: --", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    # 右手状态
    if right_any_pinch:
        rtext = f"R: {active_right_finger} ({right_region})"
        cv2.putText(frame, rtext, (160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        cv2.putText(frame, "R: --", (160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    # 当前发声信息
    if is_playing:
        cv2.putText(frame, f"PLAY: {current_note_name} ({target_freq:.1f} Hz)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "SILENT", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    # 手部计数和静音指示
    hand_count = 0
    if left_detected: hand_count += 1
    if right_detected: hand_count += 1
    cv2.putText(frame, f"Hands: {hand_count}", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    if muted:
        cv2.putText(frame, "MUTE ON", (w - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Air Piano 🎹", frame)

    # ---------- 11. 键盘控制 ----------
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('m') or key == ord('M'):
        muted = not muted
        print("🔇 静音" if muted else "🔊 静音关闭")

# ================= 清理 =================
cap.release()
stream.stop()
stream.close()
cv2.destroyAllWindows()
print("🎹 空气钢琴已退出")
