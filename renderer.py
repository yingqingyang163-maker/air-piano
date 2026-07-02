import cv2
from mediapipe.tasks.python.vision import HandLandmarksConnections
from constants import STRINGS, FINGER_ORDER, LEFT_FINGER_TIPS, ZONES, COLORS, DISC_RADIUS, PANEL_WIDTH

_HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def _zone_bounds(frame):
    h, w = frame.shape[:2]
    mid_x = PANEL_WIDTH + (w - PANEL_WIDTH) // 2
    mid_y = h // 2
    return {
        'top_left':     (PANEL_WIDTH, 0, mid_x, mid_y),
        'top_right':    (mid_x, 0, w, mid_y),
        'bottom_left':  (PANEL_WIDTH, mid_y, mid_x, h),
        'bottom_right': (mid_x, mid_y, w, h),
    }


def draw_background(frame):
    h, w = frame.shape[:2]
    bounds = _zone_bounds(frame)
    for zone_key, (x1, y1, x2, y2) in bounds.items():
        color = ZONES[zone_key]['color_bgr']
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, -1)


def draw_strings(frame):
    h, w = frame.shape[:2]
    bounds = _zone_bounds(frame)
    positions = {}
    for s in STRINGS:
        x1, y1, x2, y2 = bounds[s['zone']]
        zone_h = y2 - y1
        y = int(y1 + zone_h * (s['index'] + 0.5) / 4)
        positions[s['label']] = (y, s['zone'], s['finger'], x1, x2)
        cv2.line(frame, (x1, y), (x2, y), COLORS['string'], 1)
        cv2.putText(frame, s['label'], (x1 + 5, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['string'], 1)
    return positions


def draw_left_hand(frame, left_state):
    if not left_state['detected'] or left_state['landmarks'] is None:
        return

    h, w = frame.shape[:2]
    landmarks = left_state['landmarks']
    points = {}
    for i in range(21):
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        points[i] = (x, y)
        cv2.circle(frame, (x, y), 3, COLORS['landmark'], -1)

    for conn in _HAND_CONNECTIONS:
        if conn.start in points and conn.end in points:
            cv2.line(frame, points[conn.start], points[conn.end], COLORS['connect'], 1)

    thb = landmarks[4]
    tx, ty = int(thb.x * w), int(thb.y * h)
    cv2.circle(frame, (tx, ty), 8, COLORS['thumb'], -1)

    confirmed = left_state['confirmed']
    for fname in FINGER_ORDER:
        tip = landmarks[LEFT_FINGER_TIPS[fname]]
        fx, fy = int(tip.x * w), int(tip.y * h)
        color = COLORS['finger_active'] if confirmed.get(fname) else COLORS['finger_inactive']
        cv2.circle(frame, (fx, fy), 9, color, -1)
        cv2.circle(frame, (fx, fy), 5, COLORS['white'], -1)

    ix, iy = points[8]
    cv2.circle(frame, (ix, iy), 12, COLORS['index_tip'], -1)
    cv2.circle(frame, (ix, iy), 6, COLORS['white'], -1)


def draw_string_highlight(frame, left_state, string_positions):
    zone = left_state['zone']
    finger = left_state['active_finger']
    if zone is None or finger is None:
        return

    for s in STRINGS:
        if s['zone'] == zone and s['finger'] == finger:
            label = s['label']
            if label in string_positions:
                y, _, _, x1, x2 = string_positions[label]
                cv2.line(frame, (x1, y), (x2, y), COLORS['string_highlight'], 5)
                cv2.putText(frame, label, (x1 + 5, y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['string_highlight'], 2)


def draw_auto_highlight(frame, note_label, string_positions):
    if note_label is None:
        return
    for s in STRINGS:
        if s['label'] == note_label:
            y, _, _, x1, x2 = string_positions[note_label]
            cv2.line(frame, (x1, y), (x2, y), COLORS['string_highlight'], 5)
            cv2.putText(frame, note_label, (x1 + 5, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['string_highlight'], 2)


def draw_disc(frame, left_state, string_positions, auto_note=None):
    active_label = auto_note

    if active_label is None and left_state['pinch']:
        zone = left_state['zone']
        finger = left_state['active_finger']
        if zone and finger:
            for s in STRINGS:
                if s['zone'] == zone and s['finger'] == finger:
                    active_label = s['label']
                    break

    if active_label and active_label in string_positions:
        y, _, _, x1, x2 = string_positions[active_label]
        cx = (x1 + x2) // 2
        cy = y
        cv2.circle(frame, (cx, cy), DISC_RADIUS, COLORS['disc_fill'], -1)
        cv2.circle(frame, (cx, cy), DISC_RADIUS, COLORS['disc_border'], 2)


def draw_info_panel(frame, left_state, audio, auto_play_active, auto_note=None):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, 0), (PANEL_WIDTH, h), COLORS['panel_bg'], -1)

    y = 20
    cv2.putText(frame, "Air", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_cyan'], 1)
    y += 14
    cv2.putText(frame, "Piano", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_cyan'], 1)
    y += 20

    if auto_note:
        cv2.putText(frame, "Auto:", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_magenta'], 1)
        y += 14
        cv2.putText(frame, auto_note, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text_magenta'], 2)
        y += 22
    else:
        if audio.playing:
            cv2.putText(frame, audio.note_name, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text_cyan'], 2)
            y += 18
            cv2.putText(frame, f"{audio.target_freq:.0f}Hz", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        COLORS['text_light'], 1)
        else:
            cv2.putText(frame, "SILENT", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_dim'], 1)
        y += 22

    zone = left_state['zone']
    if zone:
        zone_name = ZONES[zone]['display_name']
        cv2.putText(frame, f"Zone:", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS['text_light'], 1)
        y += 12
        cv2.putText(frame, zone_name, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS['text_active'], 1)
        y += 18
    else:
        cv2.putText(frame, "Zone: --", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS['text_dim'], 1)
        y += 18

    finger = left_state['active_finger']
    if finger:
        cv2.putText(frame, f"F:{finger[:3]}", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS['text_magenta'], 1)
    else:
        cv2.putText(frame, "F:--", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS['text_dim'], 1)
    y += 18

    if audio.muted:
        cv2.putText(frame, "MUTE", (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_muted'], 1)
        y += 16

    # PLAY / STOP button
    bx, by = 8, h - 70
    bw, bh = 69, 40
    bcolor = COLORS['text_magenta'] if auto_play_active else COLORS['text_cyan']
    btext = "STOP" if auto_play_active else "PLAY"
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bcolor, 2)
    (tw, th), _ = cv2.getTextSize(btext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, btext, (bx + (bw - tw) // 2, by + (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 1)

    return bx, by, bw, bh


def is_click_in_button(x, y, bx, by, bw, bh):
    return bx <= x <= bx + bw and by <= y <= by + bh
