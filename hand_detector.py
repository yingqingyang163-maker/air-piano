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


class HandDetector:
    def __init__(self):
        self._ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = HandLandmarker.create_from_options(options)

        self._left_pinch_counter = 0
        self._left_pinch_confirmed = False

        self._right_raw = {f: False for f in FINGER_ORDER}
        self._right_counter = {f: 0 for f in FINGER_ORDER}
        self._right_confirmed = {f: False for f in FINGER_ORDER}
        self._right_distances = {f: 999.0 for f in FINGER_ORDER}
        self._right_region = None
        self._active_right_finger = None

    def _ensure_model(self):
        if not os.path.exists(_MODEL_PATH):
            print("下载手部检测模型中...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("模型下载完成")

    def process(self, frame):
        """Take BGR frame, return hand_state dict. Does NOT draw on frame."""
        h, w = frame.shape[:2]
        mid_y = h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        left_detected = False
        right_detected = False
        left_raw_pinch = False
        right_index_y_norm = None

        left_state = {'detected': False, 'landmarks': None, 'pinch': False, 'index_xy': None}
        right_state = {'detected': False, 'landmarks': None, 'confirmed': {}, 'region': None, 'active_finger': None}

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name

            if label == 'Left' and not left_detected:
                left_detected = True
                left_state['landmarks'] = landmarks

                idx = landmarks[8]
                thb = landmarks[4]
                dx = idx.x - thb.x; dy = idx.y - thb.y
                dist = np.sqrt(dx*dx + dy*dy)
                left_raw_pinch = dist < PINCH_THRESHOLD
                left_state['index_xy'] = (int(idx.x * w), int(idx.y * h))

            elif label == 'Right' and not right_detected:
                right_detected = True
                right_state['landmarks'] = landmarks
                right_index_y_norm = landmarks[8].y

                thb = landmarks[4]
                for fname in FINGER_ORDER:
                    tip = landmarks[RIGHT_FINGER_TIPS[fname]]
                    dx = tip.x - thb.x; dy = tip.y - thb.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    self._right_distances[fname] = dist
                    self._right_raw[fname] = dist < PINCH_THRESHOLD

        # hand loss reset
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

        # region determination
        if right_detected and right_index_y_norm is not None:
            self._right_region = 'upper' if (right_index_y_norm * h) < mid_y else 'lower'
        elif not right_detected:
            self._right_region = None

        # pinch debounce
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

        # active right finger: min-distance among confirmed
        self._active_right_finger = None
        min_d = 999.0
        for fname in FINGER_ORDER:
            if self._right_confirmed[fname] and self._right_distances[fname] < min_d:
                min_d = self._right_distances[fname]
                self._active_right_finger = fname
        right_any_pinch = (self._active_right_finger is not None)

        left_state['detected'] = left_detected
        left_state['pinch'] = self._left_pinch_confirmed

        right_state['detected'] = right_detected
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
