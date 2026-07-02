import time
import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from constants import FINGER_ORDER, LEFT_FINGER_TIPS, PINCH_THRESHOLD, PINCH_DEBOUNCE_FRAMES, PANEL_WIDTH

_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
_MODEL_PATH = 'hand_landmarker.task'

_SMOOTH_ALPHA = 0.4


class _SmoothedLandmark:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.z = 0.0


def _hand_center(landmarks):
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

        self._raw = {f: False for f in FINGER_ORDER}
        self._counter = {f: 0 for f in FINGER_ORDER}
        self._confirmed = {f: False for f in FINGER_ORDER}
        self._distances = {f: 999.0 for f in FINGER_ORDER}
        self._zone = None
        self._active_finger = None

        self._prev_smoothed = None
        self._start_time = time.time()

    def _ensure_model(self):
        if not os.path.exists(_MODEL_PATH):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("Model downloaded.")

    def _pick_left_hand(self, detected_hands):
        """Pick the user's left hand from detected hands.
        In the mirrored frame, the user's left hand appears on the right side (larger x).
        With 1 hand, assume it's the left hand."""
        if len(detected_hands) == 0:
            return None
        if len(detected_hands) == 1:
            return detected_hands[0]
        sorted_by_x = sorted(detected_hands, key=lambda h: h['center'][0])
        return sorted_by_x[1]

    def _smooth_landmarks(self, raw_landmarks, prev_smoothed):
        current = np.array([[lm.x, lm.y] for lm in raw_landmarks], dtype=np.float64)
        if prev_smoothed is not None:
            smoothed = _SMOOTH_ALPHA * current + (1.0 - _SMOOTH_ALPHA) * prev_smoothed
        else:
            smoothed = current.copy()
        wrapped = [_SmoothedLandmark(s[0], s[1]) for s in smoothed]
        return wrapped, smoothed

    def _determine_zone(self, cx_norm, cy_norm, h, w):
        """Determine which quadrant the hand center falls in."""
        mid_x = (PANEL_WIDTH / w + 1.0) / 2.0
        mid_y = 0.5

        if cy_norm < mid_y:
            if cx_norm < mid_x:
                return 'top_left'
            else:
                return 'top_right'
        else:
            if cx_norm < mid_x:
                return 'bottom_left'
            else:
                return 'bottom_right'

    def process(self, frame):
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        detected_hands = []
        for landmarks in result.hand_landmarks:
            detected_hands.append({
                'landmarks': landmarks,
                'center': _hand_center(landmarks),
            })

        left_hand = self._pick_left_hand(detected_hands)
        detected = left_hand is not None
        landmarks_smoothed = None
        index_xy = None

        if detected:
            raw_lm = left_hand['landmarks']
            landmarks_smoothed, self._prev_smoothed = self._smooth_landmarks(
                raw_lm, self._prev_smoothed)

            cx, cy = left_hand['center']
            self._zone = self._determine_zone(cx, cy, h, w)

            thb = raw_lm[4]
            for fname in FINGER_ORDER:
                tip = raw_lm[LEFT_FINGER_TIPS[fname]]
                dx = tip.x - thb.x
                dy = tip.y - thb.y
                dist = np.sqrt(dx * dx + dy * dy)
                self._distances[fname] = dist
                self._raw[fname] = dist < PINCH_THRESHOLD

            idx = raw_lm[8]
            index_xy = (int(idx.x * w), int(idx.y * h))
        else:
            self._prev_smoothed = None
            self._zone = None
            for f in FINGER_ORDER:
                self._raw[f] = False
                self._counter[f] = 0
                self._confirmed[f] = False
                self._distances[f] = 999.0

        # Debounce per-finger pinch
        for fname in FINGER_ORDER:
            if self._raw[fname]:
                self._counter[fname] = min(self._counter[fname] + 1, PINCH_DEBOUNCE_FRAMES)
            else:
                self._counter[fname] = max(self._counter[fname] - 1, 0)
            self._confirmed[fname] = (self._counter[fname] >= PINCH_DEBOUNCE_FRAMES)

        # Active finger: closest confirmed pinch
        self._active_finger = None
        min_d = 999.0
        for fname in FINGER_ORDER:
            if self._confirmed[fname] and self._distances[fname] < min_d:
                min_d = self._distances[fname]
                self._active_finger = fname
        any_pinch = (self._active_finger is not None)

        return {
            'detected': detected,
            'landmarks': landmarks_smoothed,
            'zone': self._zone,
            'active_finger': self._active_finger,
            'pinch': any_pinch,
            'index_xy': index_xy,
            'confirmed': dict(self._confirmed),
        }

    def cleanup(self):
        self._detector.close()
