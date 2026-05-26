import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time

print("🎹 空气钢琴（底部C → 顶部C5）启动中...")
print("👉 底部捏合 = 低音C，顶部捏合 = 高音C5")

# ================= 音频设置 =================
fs = 44100
current_freq = 261.63          # 初始频率（低音C）
target_freq = 261.63
muted = False
is_pinching = False            # 去抖动后的捏合状态
pinch_state = 0.0              # 原始捏合检测值（0或1）
phase = 0.0

# 频率数组：索引0 = 高音C5（顶部），索引7 = 低音C（底部）
freqs_8 = [523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63]
note_names_internal = ['C5', 'B', 'A', 'G', 'F', 'E', 'D', 'C']
display_notes_bottom_to_top = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C5']

# 音符索引滞回参数
NOTE_HYSTERESIS = 0.15          # 滞回范围（归一化y坐标）
last_note_idx = None            # 上一个有效音符索引

def y_to_note_index(y_norm):
    """
    带滞回的音符索引映射。
    y_norm=0（顶部）→ 索引0（C5），y_norm=1（底部）→ 索引7（C）
    """
    global last_note_idx
    # 边界裁剪
    y = max(0.0, min(1.0, y_norm))
    # 直接映射（连续值）
    raw_idx = int(y * 8)
    raw_idx = min(raw_idx, 7)

    # 如果没有上次索引，直接使用
    if last_note_idx is None:
        last_note_idx = raw_idx
        return raw_idx

    # 计算当前手指位置对应的“理论边界中心”
    # 每个音符占 1/8 高度，边界位置为 k/8
    # 如果当前原始索引与上次索引不同，且手指离上次索引的中心距离超过滞回，才切换
    center_prev = (last_note_idx + 0.5) / 8.0   # 上次索引的中心y坐标（归一化）
    if abs(y - center_prev) > NOTE_HYSTERESIS / 8.0:
        # 超过滞回范围，允许切换
        last_note_idx = raw_idx
    # 否则保持上次索引
    return last_note_idx

def note_idx_to_freq(idx):
    return freqs_8[idx], note_names_internal[idx]

# 包络相关（每次捏合开始时重置）
envelope_time = 0.0
is_playing = False               # 当前是否正在发声（捏合状态）

def reset_envelope():
    """重置包络时间（在捏合开始的瞬间调用）"""
    global envelope_time
    envelope_time = 0.0

def envelope_adsr_stable(t):
    """
    改进包络：Attack 0.01s, Decay 0.1s, 之后维持 0.5 幅度（不再衰减到0）。
    t: 从捏合开始经过的时间（秒）
    """
    attack = 0.01
    decay = 0.1
    if t < attack:
        return t / attack
    elif t < attack + decay:
        # Decay 阶段从 1 线性降到 0.5
        return 1.0 - 0.5 * ((t - attack) / decay)
    else:
        # Sustain 恒定 0.5
        return 0.5

def audio_callback(outdata, frames, time_info, status):
    global current_freq, target_freq, muted, is_playing, phase, envelope_time
    if muted or not is_playing:
        outdata[:] = np.zeros((frames, 1))
        return

    # 频率平滑（一阶低通，系数越小越平滑，这里取0.2）
    alpha = 0.2
    current_freq = current_freq * (1 - alpha) + target_freq * alpha

    # 生成波形（含三次谐波模拟钢琴）
    t_arr = np.arange(frames) / fs
    # 累加相位（保证相位连续）
    delta_phase = 2 * np.pi * current_freq / fs
    phases = phase + np.cumsum(delta_phase * np.ones(frames))
    phase = phases[-1] % (2 * np.pi)
    wave = 0.5 * np.sin(phases) + 0.3 * np.sin(2 * phases) + 0.15 * np.sin(3 * phases)

    # 应用包络（根据从捏合开始经过的时间）
    env_time_arr = envelope_time + t_arr
    envelope = np.array([envelope_adsr_stable(t) for t in env_time_arr])
    envelope_time = env_time_arr[-1]

    # 最终输出
    outdata[:] = (0.2 * wave * envelope).reshape(-1, 1)

# 音频流启动
stream = sd.OutputStream(callback=audio_callback, channels=1, samplerate=fs)
stream.start()
print("✅ 音频系统就绪")

# ================= 手部检测 & 捏合去抖动 =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(5):
    cap.read()

cv2.namedWindow("Air Piano 🎹", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Air Piano 🎹", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 捏合去抖动变量
PINCH_DEBOUNCE_FRAMES = 3
pinch_counter = 0
pinch_confirmed = False          # 去抖动后的捏合状态（用于触发逻辑）

last_pinch_state = False
active_note_idx = None
current_note_name = ""

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

    for i in range(8):
        y0 = int(i * h / 8)
        y1 = int((i + 1) * h / 8)
        if active_note_idx is not None and i == active_note_idx:
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
        alpha = alpha_highlight if (active_note_idx is not None and i == active_note_idx) else alpha_overlay
        cv2.addWeighted(roi_result, 1 - alpha, roi_overlay, alpha, 0, roi_result)
    frame = result

    # ---------- 2. 手部检测 ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    raw_pinch = False
    finger_y_norm = 0.5

    if result_hands.multi_hand_landmarks:
        for handLms in result_hands.multi_hand_landmarks:
            index_tip = handLms.landmark[8]
            thumb_tip = handLms.landmark[4]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            dx = index_tip.x - thumb_tip.x
            dy = index_tip.y - thumb_tip.y
            dist = np.sqrt(dx*dx + dy*dy)
            raw_pinch = dist < 0.1
            finger_y_norm = index_tip.y   # 归一化 y 坐标（0顶~1底）

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                  mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                                  mp_draw.DrawingSpec(color=(255,0,0), thickness=2))
            cv2.circle(frame, (x, y), 15, (0,255,0), -1)
            cv2.circle(frame, (x, y), 8, (255,0,0), -1)

    # ---------- 3. 捏合去抖动逻辑 ----------
    if raw_pinch:
        pinch_counter = min(pinch_counter + 1, PINCH_DEBOUNCE_FRAMES)
    else:
        pinch_counter = max(pinch_counter - 1, 0)

    new_pinch_confirmed = (pinch_counter >= PINCH_DEBOUNCE_FRAMES)

    # 检测捏合上升沿（开始捏合）
    if new_pinch_confirmed and not pinch_confirmed:
        # 开始播放：重置包络、根据当前手指位置确定音符
        reset_envelope()
        # 根据手指 y 坐标确定音符索引（带滞回）
        note_idx = y_to_note_index(finger_y_norm)
        target_freq, current_note_name = note_idx_to_freq(note_idx)
        active_note_idx = note_idx
        # 立即更新 current_freq，避免从旧频率滑过来
        current_freq = target_freq
        is_playing = True
        #print(f"🎵 触发音符: {current_note_name} ({target_freq:.1f} Hz)")

    # 检测捏合下降沿（松开捏合）
    if not new_pinch_confirmed and pinch_confirmed:
        is_playing = False
        active_note_idx = None
        #print("🔇 松开")

    pinch_confirmed = new_pinch_confirmed

    # 如果捏合保持中，但手指移动导致音符索引变化（带滞回），更新目标频率
    if pinch_confirmed and is_playing:
        new_idx = y_to_note_index(finger_y_norm)
        if new_idx != active_note_idx:
            active_note_idx = new_idx
            target_freq, current_note_name = note_idx_to_freq(new_idx)
            # 这里 target_freq 改变，音频回调会平滑过渡
            #print(f"🎶 滑至音符: {current_note_name} ({target_freq:.1f} Hz)")

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

    if pinch_confirmed:
        cv2.putText(frame, f"🎵 {current_note_name}", (w//2 - 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
    else:
        cv2.putText(frame, "Pinch to play", (w//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if muted:
        cv2.putText(frame, "🔇 MUTE ON", (w - 200, 50),
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