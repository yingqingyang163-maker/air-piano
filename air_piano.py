import cv2
import time
from constants import ZONES, MELODY_TWINKLE
from audio_engine import AudioEngine
from hand_detector import HandDetector
from renderer import (draw_background, draw_strings, draw_left_hand, draw_string_highlight,
                      draw_disc, draw_info_panel, draw_auto_highlight, is_click_in_button)

print("[Air Piano] 空气钢琴（四区·左手）启动中...")
print("   左上(深蓝): G3 A3 B3 C4  |  右上(浅蓝): G4 A4 B4 C5")
print("   左下(深绿): C3 D3 E3 F3  |  右下(浅绿): C4 D4 E4 F4")
print("   左手移至不同区域，捏合2/3/4/5指演奏对应音符")

audio = AudioEngine()
print("[OK] 音频系统就绪")

detector = HandDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(2):
    cap.read()

cv2.namedWindow("Air Piano", cv2.WINDOW_NORMAL)

# Auto-play state
auto_play_active = False
auto_play_index = 0
auto_play_next_time = 0.0
auto_play_note = None

# Mouse state
mouse_clicked = False
mouse_x = 0
mouse_y = 0


def on_mouse(event, x, y, flags, param):
    global mouse_clicked, mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
        mouse_x = x
        mouse_y = y


cv2.setMouseCallback("Air Piano", on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    left_state = detector.process(frame)

    # Auto-play logic (visual only)
    if auto_play_active:
        now = time.time()
        if now >= auto_play_next_time:
            if auto_play_index < len(MELODY_TWINKLE):
                note, duration = MELODY_TWINKLE[auto_play_index]
                auto_play_note = note
                auto_play_next_time = now + duration
                auto_play_index += 1
            else:
                auto_play_active = False
                auto_play_note = None
                auto_play_index = 0

    # Sound logic: left hand zone + finger → note
    if left_state['pinch'] and left_state['zone'] and left_state['active_finger']:
        note_name, freq = ZONES[left_state['zone']]['notes'][left_state['active_finger']]
        audio.play_note(freq)
        audio.note_name = note_name
    else:
        audio.silence()

    # Render
    draw_background(frame)
    string_ys = draw_strings(frame)
    draw_left_hand(frame, left_state)

    if auto_play_active and auto_play_note:
        draw_auto_highlight(frame, auto_play_note, string_ys)
    else:
        draw_string_highlight(frame, left_state, string_ys)

    draw_disc(frame, left_state, string_ys, auto_play_note)
    button_bounds = draw_info_panel(frame, left_state, audio, auto_play_active, auto_play_note)

    cv2.imshow("Air Piano", frame)

    if mouse_clicked:
        mouse_clicked = False
        if is_click_in_button(mouse_x, mouse_y, *button_bounds):
            if auto_play_active:
                auto_play_active = False
                auto_play_note = None
                auto_play_index = 0
            else:
                auto_play_active = True
                auto_play_index = 0
                auto_play_next_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m') or key == ord('M'):
        muted = audio.toggle_mute()
        print("[MUTE] 静音" if muted else "[UNMUTE] 取消静音")

cap.release()
detector.cleanup()
audio.cleanup()
cv2.destroyAllWindows()
print("[Air Piano] 空气钢琴已退出")
