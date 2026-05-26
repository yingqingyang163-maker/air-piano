import cv2
from constants import RIGHT_FINGER_NOTES
from audio_engine import AudioEngine
from hand_detector import HandDetector
from renderer import draw_background, draw_strings, draw_hands, draw_string_highlight, draw_disc, draw_info_panel

print("[Air Piano] 空气钢琴（琴弦·双手交互）启动中...")
print("   上半区(蓝): G A B C5  |  下半区(绿): C D E F")
print("   仅左手捏合 → 中音A | 同时捏合 → 右手选音")

audio = AudioEngine()
print("[OK] 音频系统就绪")

detector = HandDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(5):
    cap.read()

cv2.namedWindow("Air Piano", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Air Piano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    mid_y = h // 2

    # hand detection
    hand_state = detector.process(frame)

    # sound logic
    left = hand_state['left']
    right = hand_state['right']
    right_any_pinch = hand_state['right_any_pinch']

    if left['pinch'] and not right_any_pinch:
        audio.play_note(440.0)
        audio.note_name = 'A4'
    elif left['pinch'] and right_any_pinch and right['region'] is not None:
        note_name, freq = RIGHT_FINGER_NOTES[right['region']][right['active_finger']]
        audio.play_note(freq)
        audio.note_name = note_name
    else:
        audio.silence()

    # render
    draw_background(frame, mid_y)
    string_ys = draw_strings(frame)
    draw_hands(frame, hand_state)
    draw_string_highlight(frame, hand_state, string_ys)
    draw_disc(frame, hand_state, string_ys)
    draw_info_panel(frame, hand_state, audio, w)

    cv2.imshow("Air Piano", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m') or key == ord('M'):
        muted = audio.toggle_mute()
        print("[MUTE] 静音" if muted else "[UNMUTE] 静音关闭")

cap.release()
detector.cleanup()
audio.cleanup()
cv2.destroyAllWindows()
print("[Air Piano] 空气钢琴已退出")
