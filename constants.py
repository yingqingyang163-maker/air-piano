# Air Piano: constants & configuration

FINGER_ORDER = ['index', 'middle', 'ring', 'pinky']
LEFT_FINGER_TIPS = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}

# Zone: quadrant → finger → (note_label, frequency)
ZONES = {
    'bottom_left': {
        'color_bgr': (80, 140, 60),     # dark green
        'display_name': 'dark green',
        'notes': {
            'index':  ('C3', 130.81),
            'middle': ('D3', 146.83),
            'ring':   ('E3', 164.81),
            'pinky':  ('F3', 174.61),
        },
    },
    'top_left': {
        'color_bgr': (180, 100, 60),    # dark blue
        'display_name': 'dark blue',
        'notes': {
            'index':  ('G3', 196.00),
            'middle': ('A3', 220.00),
            'ring':   ('B3', 246.94),
            'pinky':  ('C4', 261.63),
        },
    },
    'bottom_right': {
        'color_bgr': (140, 230, 120),   # light green
        'display_name': 'light green',
        'notes': {
            'index':  ('C4', 261.63),
            'middle': ('D4', 293.66),
            'ring':   ('E4', 329.63),
            'pinky':  ('F4', 349.23),
        },
    },
    'top_right': {
        'color_bgr': (230, 190, 150),   # light blue
        'display_name': 'light blue',
        'notes': {
            'index':  ('G4', 392.00),
            'middle': ('A4', 440.00),
            'ring':   ('B4', 493.88),
            'pinky':  ('C5', 523.25),
        },
    },
}

# Flat list for rendering convenience
def _build_strings():
    result = []
    for zone_key in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
        zone_info = ZONES[zone_key]
        for i, finger in enumerate(FINGER_ORDER):
            label, freq = zone_info['notes'][finger]
            result.append({
                'zone': zone_key,
                'finger': finger,
                'label': label,
                'freq': freq,
                'index': i,
            })
    return result

STRINGS = _build_strings()

PINCH_THRESHOLD = 0.1
PINCH_DEBOUNCE_FRAMES = 1
DISC_RADIUS = 20
SAMPLE_RATE = 44100
DEFAULT_FREQ = 261.63

PANEL_WIDTH = 85

MELODY_TWINKLE = [
    ('C4', 0.5), ('C4', 0.5), ('G4', 0.5), ('G4', 0.5), ('A4', 0.5), ('A4', 0.5), ('G4', 1.0),
    ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5), ('D4', 0.5), ('D4', 0.5), ('C4', 1.0),
    ('G4', 0.5), ('G4', 0.5), ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5), ('D4', 1.0),
    ('G4', 0.5), ('G4', 0.5), ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5), ('D4', 1.0),
    ('C4', 0.5), ('C4', 0.5), ('G4', 0.5), ('G4', 0.5), ('A4', 0.5), ('A4', 0.5), ('G4', 1.0),
    ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5), ('D4', 0.5), ('D4', 0.5), ('C4', 1.0),
]

COLORS = {
    'string':           (200, 200, 200),
    'string_highlight': (255, 0, 255),
    'disc_fill':        (0, 255, 255),
    'disc_border':      (0, 200, 200),
    'panel_bg':         (0, 0, 0),
    'landmark':         (0, 255, 0),
    'connect':          (0, 180, 0),
    'index_tip':        (0, 255, 0),
    'thumb':            (0, 0, 255),
    'finger_active':    (0, 255, 255),
    'finger_inactive':  (255, 0, 0),
    'text_active':      (0, 255, 0),
    'text_magenta':     (255, 0, 255),
    'text_cyan':        (0, 255, 255),
    'text_dim':         (120, 120, 120),
    'text_muted':       (0, 0, 255),
    'text_light':       (180, 180, 180),
    'white':            (255, 255, 255),
}
