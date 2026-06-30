import time
import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from constants import FINGER_ORDER, RIGHT_FINGER_TIPS, PINCH_THRESHOLD, PINCH_DEBOUNCE_FRAMES

_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
_MODEL_PATH = 'hand_landmarker.task'

_SMOOTH_ALPHA = 0.4


class _SmoothedLandmark:
    """Thin wrapper so renderer can access .x .y on EMA-smoothed landmarks."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.z = 0.0


def _hand_center(landmarks):
    """Normalized center of a hand (midpoint of wrist and middle-finger MCP)."""
    wrist = landmarks[0]
    mcp = landmarks[9]
    return ((wrist.x + mcp.x) / 2.0, (wrist.y + mcp.y) / 2.0)


class HandDetector:
    def __init__(self):
        self._ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._detector = HandLandmarker.create_from_options(options)

        # Left-hand pinch state
        self._left_pinch_counter = 0
        self._left_pinch_confirmed = False

        # Right-hand per-finger state
        self._right_raw = {f: False for f in FINGER_ORDER}
        self._right_counter = {f: 0 for f in FINGER_ORDER}
        self._right_confirmed = {f: False for f in FINGER_ORDER}
        self._right_distances = {f: 999.0 for f in FINGER_ORDER}
        self._right_region = None
        self._active_right_finger = None

        # EMA-smoothed landmark arrays for temporal stability ((21, 2) numpy, or None)
        self._prev_left_smoothed = None
        self._prev_right_smoothed = None

        self._start_time = time.time()

    def _ensure_model(self):
        if not os.path.exists(_MODEL_PATH):
            print("下载手部检测模型中...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("模型下载完成")

    # ------------------------------------------------------------------
    # Hand identity stabilisation
    # ------------------------------------------------------------------
    def _assign_hands(self, detected_hands):
        """Assign detected hands to left / right slots by x-position.

        Detection runs on the *mirrored* frame: the user's right hand
        appears on the left side (smaller x), and the user's left hand
        appears on the right side (larger x).  This is more reliable than
        MediaPipe's per-frame handedness which can swap when hands cross.
        """
        left_hand = None   # user's actual left hand
        right_hand = None  # user's actual right hand

        if len(detected_hands) == 0:
            pass
        elif len(detected_hands) == 1:
            h = detected_hands[0]
            # In the mirrored frame, right hand → smaller x, left → larger x
            if h['center'][0] < 0.5:
                right_hand = h
            else:
                left_hand = h
        else:
            sorted_by_x = sorted(detected_hands, key=lambda h: h['center'][0])
            right_hand = sorted_by_x[0]
            left_hand = sorted_by_x[1]

        return left_hand, right_hand

    # ------------------------------------------------------------------
    # Temporal landmark smoothing (EMA)
    # ------------------------------------------------------------------
    def _smooth_landmarks(self, raw_landmarks, prev_smoothed):
        """Apply exponential moving average, returning (wrapped_list, array)."""
        current = np.array([[lm.x, lm.y] for lm in raw_landmarks], dtype=np.float64)
        if prev_smoothed is not None:
            smoothed = _SMOOTH_ALPHA * current + (1.0 - _SMOOTH_ALPHA) * prev_smoothed
        else:
            smoothed = current.copy()
        wrapped = [_SmoothedLandmark(s[0], s[1]) for s in smoothed]
        return wrapped, smoothed

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def process(self, frame):
        """Take BGR frame (already mirrored), return hand_state dict."""
        h, w = frame.shape[:2]
        mid_y = h // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        # Collect all detected hands with their centre positions
        detected_hands = []
        for landmarks in result.hand_landmarks:
            detected_hands.append({
                'landmarks': landmarks,
                'center': _hand_center(landmarks),
            })

        # Assign to left / right using x-position (ignoring MediaPipe handedness)
        left_hand, right_hand = self._assign_hands(detected_hands)

        # Apply EMA smoothing
        left_landmarks = None
        right_landmarks = None
        if left_hand is not None:
            left_landmarks, self._prev_left_smoothed = self._smooth_landmarks(
                left_hand['landmarks'], self._prev_left_smoothed)
        else:
            self._prev_left_smoothed = None
        if right_hand is not None:
            right_landmarks, self._prev_right_smoothed = self._smooth_landmarks(
                right_hand['landmarks'], self._prev_right_smoothed)
        else:
            self._prev_right_smoothed = None

        left_detected = left_hand is not None
        right_detected = right_hand is not None

        left_raw_pinch = False
        right_index_y_norm = None

        left_state = {'detected': left_detected, 'landmarks': None, 'pinch': False, 'index_xy': None}
        right_state = {'detected': right_detected, 'landmarks': None, 'confirmed': {}, 'region': None, 'active_finger': None}

        if left_detected:
            left_state['landmarks'] = left_landmarks
            # Pinch detection uses RAW landmarks for responsiveness
            raw_lm = left_hand['landmarks']
            idx = raw_lm[8]
            thb = raw_lm[4]
            dx = idx.x - thb.x
            dy = idx.y - thb.y
            dist = np.sqrt(dx * dx + dy * dy)
            left_raw_pinch = dist < PINCH_THRESHOLD
            left_state['index_xy'] = (int(idx.x * w), int(idx.y * h))

        if right_detected:
            right_state['landmarks'] = right_landmarks
            raw_lm = right_hand['landmarks']
            right_index_y_norm = raw_lm[8].y
            thb = raw_lm[4]
            for fname in FINGER_ORDER:
                tip = raw_lm[RIGHT_FINGER_TIPS[fname]]
                dx = tip.x - thb.x
                dy = tip.y - thb.y
                dist = np.sqrt(dx * dx + dy * dy)
                self._right_distances[fname] = dist
                self._right_raw[fname] = dist < PINCH_THRESHOLD

        # Hand-loss reset
        if not left_detected:
            self._left_pinch_counter = 0
            self._left_pinch_confirmed = False
        if not right_detected:
            for f in FINGER_ORDER:
                self._right_raw[f] = False
                self._right_counter[f] = 0
                self._right_confirmed[f] = False
                self._right_distances[f] = 999.0
            self._right_region = None
            self._active_right_finger = None

        # Region determination
        if right_detected and right_index_y_norm is not None:
            self._right_region = 'upper' if (right_index_y_norm * h) < mid_y else 'lower'
        elif not right_detected:
            self._right_region = None

        # Pinch debounce
        if left_raw_pinch:
            self._left_pinch_counter = min(self._left_pinch_counter + 1, PINCH_DEBOUNCE_FRAMES)
        else:
            self._left_pinch_counter = max(self._left_pinch_counter - 1, 0)
        self._left_pinch_confirmed = (self._left_pinch_counter >= PINCH_DEBOUNCE_FRAMES)

        for fname in FINGER_ORDER:
            if self._right_raw[fname]:
                self._right_counter[fname] = min(self._right_counter[fname] + 1, PINCH_DEBOUNCE_FRAMES)
            else:
                self._right_counter[fname] = max(self._right_counter[fname] - 1, 0)
            self._right_confirmed[fname] = (self._right_counter[fname] >= PINCH_DEBOUNCE_FRAMES)

        # Active right finger: closest confirmed pinch to thumb
        self._active_right_finger = None
        min_d = 999.0
        for fname in FINGER_ORDER:
            if self._right_confirmed[fname] and self._right_distances[fname] < min_d:
                min_d = self._right_distances[fname]
                self._active_right_finger = fname
        right_any_pinch = (self._active_right_finger is not None)

        left_state['pinch'] = self._left_pinch_confirmed
        right_state['confirmed'] = dict(self._right_confirmed)
        right_state['region'] = self._right_region
        right_state['active_finger'] = self._active_right_finger

        hand_count = (1 if left_detected else 0) + (1 if right_detected else 0)

        return {
            'left': left_state,
            'right': right_state,
            'right_any_pinch': right_any_pinch,
            'hand_count': hand_count,
        }

    def cleanup(self):
        self._detector.close()
