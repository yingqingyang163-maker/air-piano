import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

print("🎹 空气钢琴（双手模式）启动中...")
print("👉 左手捏合 = 左手演奏 | 右手捏合 = 右手演奏 | 双手可同时发声")

# ================= 音频设置 =================
fs = 44100
muted = False

# 频率数组：索引0 = 高音C5（顶部），索引7 = 低音C（底部）
freqs_8 = [523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63]
note_names_internal = ['C5', 'B', 'A', 'G', 'F', 'E', 'D', 'C']
display_notes_bottom_to_top = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C5']

NOTE_HYSTERESIS = 0.15
PINCH_DEBOUNCE_FRAMES = 3
PINCH_THRESHOLD = 0.1

# ================= 手部音频状态 =================
class HandVoice:
    def __init__(self, label):
        self.label = label
        self.current_freq = 261.63
        self.target_freq = 261.63
        self.phase = 0.0
        self.envelope_time = 0.0
        self.is_playing = False
        self.last_note_idx = None
        self.pinch_counter = 0
        self.pinch_confirmed = False
        self.active_note_idx = None
        self.current_note_name = ""

    def reset_envelope(self):
        self.envelope_time = 0.0

    def y_to_note_index(self, y_norm):
        y = max(0.0, min(1.0, y_norm))
        raw_idx = int(y * 8)
        raw_idx = min(raw_idx, 7)
        if self.last_note_idx is None:
            self.last_note_idx = raw_idx
            return raw_idx
        center_prev = (self.last_note_idx + 0.5) / 8.0
        if abs(y - center_prev) > NOTE_HYSTERESIS / 8.0:
            self.last_note_idx = raw_idx
        return self.last_note_idx

    def note_idx_to_freq(self, idx):
        return freqs_8[idx], note_names_internal[idx]

voice_left = HandVoice('Left')
voice_right = HandVoice('Right')

def envelope_adsr_stable(t):
    attack = 0.01
    decay = 0.1
    if t < attack:
        return t / attack
    elif t < attack + decay:
        return 1.0 - 0.5 * ((t - attack) / decay)
    else:
        return 0.5

def audio_callback(outdata, frames, time_info, status):
    global muted
    if muted:
        outdata[:] = np.zeros((frames, 1))
        return

    mixed = np.zeros(frames)

    for voice in [voice_left, voice_right]:
        if not voice.is_playing:
            continue

        alpha = 0.2
        voice.current_freq = voice.current_freq * (1 - alpha) + voice.target_freq * alpha

        t_arr = np.arange(frames) / fs
        delta_phase = 2 * np.pi * voice.current_freq / fs
        phases = voice.phase + np.cumsum(delta_phase * np.ones(frames))
        voice.phase = phases[-1] % (2 * np.pi)
        wave = 0.5 * np.sin(phases) + 0.3 * np.sin(2 * phases) + 0.15 * np.sin(3 * phases)

        env_time_arr = voice.envelope_time + t_arr
        envelope = np.array([envelope_adsr_stable(t) for t in env_time_arr])
        voice.envelope_time = env_time_arr[-1]

        mixed += 0.15 * wave * envelope

    outdata[:] = mixed.reshape(-1, 1)

# 音频流启动
stream = sd.OutputStream(callback=audio_callback, channels=1, samplerate=fs)
stream.start()
print("✅ 音频系统就绪")

# ================= 手部检测 =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

    # ---------- 1. 绘制半透明音区 ----------
    overlay = np.zeros_like(frame)
    alpha_overlay = 0.35
    alpha_highlight = 0.6

    active_notes = set()
    if voice_left.active_note_idx is not None and voice_left.pinch_confirmed:
        active_notes.add(voice_left.active_note_idx)
    if voice_right.active_note_idx is not None and voice_right.pinch_confirmed:
        active_notes.add(voice_right.active_note_idx)

    for i in range(8):
        y0 = int(i * h / 8)
        y1 = int((i + 1) * h / 8)
        if i in active_notes:
            color = (0, 255, 255)
            cur_alpha = alpha_highlight
        else:
            color = (50, 50, 50)
            cur_alpha = alpha_overlay
        cv2.rectangle(overlay, (0, y0), (w, y1), color, -1)
        cv2.line(frame, (0, y0), (w, y0), (255, 255, 255), 2)

    result = frame.copy()
    for i in range(8):
        y0 = int(i * h / 8)
        y1 = int((i + 1) * h / 8)
        roi_result = result[y0:y1, 0:w]
        roi_overlay = overlay[y0:y1, 0:w]
        alpha = alpha_highlight if i in active_notes else alpha_overlay
        cv2.addWeighted(roi_result, 1 - alpha, roi_overlay, alpha, 0, roi_result)
    frame = result

    # ---------- 2. 双手检测 ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    hand_detections = []  # (label, raw_pinch, finger_y_norm)

    if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
        for handLms, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'

            index_tip = handLms.landmark[8]
            thumb_tip = handLms.landmark[4]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            dx = index_tip.x - thumb_tip.x
            dy = index_tip.y - thumb_tip.y
            dist = np.sqrt(dx*dx + dy*dy)
            raw_pinch = dist < PINCH_THRESHOLD
            finger_y_norm = index_tip.y

            # 左手绿色，右手红色
            if label == 'Left':
                landmark_color = (0, 255, 0)
                connection_color = (255, 0, 0)
            else:
                landmark_color = (255, 0, 0)
                connection_color = (0, 0, 255)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                  mp_draw.DrawingSpec(color=landmark_color, thickness=2),
                                  mp_draw.DrawingSpec(color=connection_color, thickness=2))
            cv2.circle(frame, (x, y), 15, landmark_color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

            hand_detections.append((label, raw_pinch, finger_y_norm))

    # ---------- 3. 双手捏合去抖动逻辑 ----------
    detected_labels = set()

    for label, raw_pinch, finger_y_norm in hand_detections:
        detected_labels.add(label)
        voice = voice_left if label == 'Left' else voice_right

        if raw_pinch:
            voice.pinch_counter = min(voice.pinch_counter + 1, PINCH_DEBOUNCE_FRAMES)
        else:
            voice.pinch_counter = max(voice.pinch_counter - 1, 0)

        new_pinch = (voice.pinch_counter >= PINCH_DEBOUNCE_FRAMES)

        # 捏合上升沿
        if new_pinch and not voice.pinch_confirmed:
            voice.reset_envelope()
            note_idx = voice.y_to_note_index(finger_y_norm)
            voice.target_freq, voice.current_note_name = voice.note_idx_to_freq(note_idx)
            voice.active_note_idx = note_idx
            voice.current_freq = voice.target_freq
            voice.is_playing = True

        # 捏合下降沿
        if not new_pinch and voice.pinch_confirmed:
            voice.is_playing = False
            voice.active_note_idx = None
            voice.last_note_idx = None

        voice.pinch_confirmed = new_pinch

        # 捏合保持中滑动切换音符
        if voice.pinch_confirmed and voice.is_playing:
            new_idx = voice.y_to_note_index(finger_y_norm)
            if new_idx != voice.active_note_idx:
                voice.active_note_idx = new_idx
                voice.target_freq, voice.current_note_name = voice.note_idx_to_freq(new_idx)

    # 手部消失时重置状态
    for label, voice in [('Left', voice_left), ('Right', voice_right)]:
        if label not in detected_labels:
            voice.is_playing = False
            voice.active_note_idx = None
            voice.last_note_idx = None
            voice.pinch_confirmed = False
            voice.pinch_counter = 0

    # ---------- 4. 绘制音符文字 ----------
    for i in range(8):
        y0 = int(i * h / 8)
        y1 = int((i + 1) * h / 8)
        display_text = display_notes_bottom_to_top[7 - i]
        cv2.putText(frame, display_text, (10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ---------- 5. 信息面板 ----------
    overlay_info = frame.copy()
    cv2.rectangle(overlay_info, (0, 0), (w, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay_info, 0.6, frame, 0.4, 0)

    # 左手状态（左侧显示）
    if voice_left.pinch_confirmed:
        cv2.putText(frame, f"L: {voice_left.current_note_name}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "L: --", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

    # 右手状态（右侧显示）
    if voice_right.pinch_confirmed:
        text = f"R: {voice_right.current_note_name}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.putText(frame, text, (w - text_size[0] - 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    else:
        text = "R: --"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        cv2.putText(frame, text, (w - text_size[0] - 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

    # 中间提示
    if not voice_left.pinch_confirmed and not voice_right.pinch_confirmed:
        cv2.putText(frame, "Pinch to play", (w//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 左右手数量提示
    hand_count_text = f"Hands: {len(detected_labels)}"
    cv2.putText(frame, hand_count_text, (w//2 - 50, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if muted:
        cv2.putText(frame, "🔇 MUTE ON", (w - 200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Air Piano 🎹", frame)

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
