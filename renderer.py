import cv2
from mediapipe.tasks.python.vision import HandLandmarksConnections
from constants import STRINGS, FINGER_ORDER, RIGHT_FINGER_TIPS, COLORS, DISC_RADIUS

_HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def _draw_hand_landmarks(frame, landmarks):
    """Draw hand skeleton and joints for a single hand."""
    h, w = frame.shape[:2]
    points = {}
    for i in range(21):
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        points[i] = (x, y)
        cv2.circle(frame, (x, y), 3, COLORS['white'], -1)

    for conn in _HAND_CONNECTIONS:
        if conn.start in points and conn.end in points:
            cv2.line(frame, points[conn.start], points[conn.end], COLORS['white'], 1)


def draw_background(frame, mid_y):
    """Draw opaque upper (blue) and lower (green) zones."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, mid_y), COLORS['bg_upper'], -1)
    cv2.rectangle(frame, (0, mid_y), (w, h), COLORS['bg_lower'], -1)


def draw_strings(frame):
    """Draw 8 horizontal strings with labels. Returns {label: y_pixel} dict."""
    h = frame.shape[0]
    w = frame.shape[1]
    positions = {}
    for s in STRINGS:
        y = int(s['y_frac'] * h)
        positions[s['label']] = y
        cv2.line(frame, (50, y), (w - 50, y), COLORS['string'], 2)
        cv2.putText(frame, s['label'], (10, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['string'], 2)
    return positions


def draw_hands(frame, hand_state):
    """Draw hand skeletons, fingertips, and thumb markers."""
    left = hand_state['left']
    right = hand_state['right']
    h, w = frame.shape[:2]

    if left['detected'] and left['landmarks']:
        _draw_hand_landmarks(frame, left['landmarks'])
        ix, iy = left['index_xy']
        cv2.circle(frame, (ix, iy), 12, COLORS['left_tip'], -1)
        cv2.circle(frame, (ix, iy), 6, COLORS['white'], -1)

    if right['detected'] and right['landmarks']:
        _draw_hand_landmarks(frame, right['landmarks'])

        landmarks = right['landmarks']
        thb = landmarks[4]
        tx, ty = int(thb.x * w), int(thb.y * h)
        cv2.circle(frame, (tx, ty), 8, COLORS['thumb'], -1)

        confirmed = right['confirmed']
        for fname in FINGER_ORDER:
            tip = landmarks[RIGHT_FINGER_TIPS[fname]]
            fx, fy = int(tip.x * w), int(tip.y * h)
            color = COLORS['right_tip_active'] if confirmed.get(fname) else COLORS['right_tip_inactive']
            cv2.circle(frame, (fx, fy), 9, color, -1)
            cv2.circle(frame, (fx, fy), 5, COLORS['white'], -1)


def draw_string_highlight(frame, hand_state, string_y_positions):
    """Highlight the active string in magenta when right hand selects a note."""
    right = hand_state['right']
    if right['region'] is None or right['active_finger'] is None:
        return
    for s in STRINGS:
        if s['region'] == right['region'] and s['finger'] == right['active_finger']:
            y = string_y_positions[s['label']]
            cv2.line(frame, (50, y), (frame.shape[1] - 50, y), COLORS['string_highlight'], 6)
            cv2.putText(frame, s['label'], (10, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['string_highlight'], 2)


def draw_disc(frame, hand_state, string_y_positions):
    """Draw yellow disc at the active string when playing."""
    left = hand_state['left']
    if not left['pinch']:
        return
    right = hand_state['right']
    active_label = None
    if right['region'] and right['active_finger']:
        for s in STRINGS:
            if s['region'] == right['region'] and s['finger'] == right['active_finger']:
                active_label = s['label']
                break
    if active_label and active_label in string_y_positions:
        cx = frame.shape[1] // 2
        cy = string_y_positions[active_label]
        cv2.circle(frame, (cx, cy), DISC_RADIUS, COLORS['disc_fill'], -1)
        cv2.circle(frame, (cx, cy), DISC_RADIUS, COLORS['disc_border'], 2)


def draw_info_panel(frame, hand_state, audio, w):
    """Draw opaque top info panel with status text."""
    panel_h = 70
    cv2.rectangle(frame, (0, 0), (w, panel_h), COLORS['panel_bg'], -1)

    left = hand_state['left']
    right = hand_state['right']

    if left['pinch']:
        cv2.putText(frame, "L: PINCH", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_active'], 2)
    else:
        cv2.putText(frame, "L: --", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_dim'], 2)

    if hand_state['right_any_pinch']:
        rtext = f"R: {right['active_finger']} ({right['region']})"
        cv2.putText(frame, rtext, (160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_magenta'], 2)
    else:
        cv2.putText(frame, "R: --", (160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_dim'], 2)

    if audio.playing:
        cv2.putText(frame, f"PLAY: {audio.note_name} ({audio.target_freq:.1f} Hz)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_cyan'], 2)
    else:
        cv2.putText(frame, "SILENT", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_dim'], 2)

    cv2.putText(frame, f"Hands: {hand_state['hand_count']}", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_light'], 1)

    if audio.muted:
        cv2.putText(frame, "MUTE ON", (w - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_muted'], 2)
