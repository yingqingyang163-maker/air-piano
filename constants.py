# Air Piano: constants & configuration

FINGER_ORDER = ['index', 'middle', 'ring', 'pinky']

RIGHT_FINGER_TIPS = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}

RIGHT_FINGER_NOTES = {
    'lower': {
        'index':  ('C4', 261.63),
        'middle': ('D4', 293.66),
        'ring':   ('E4', 329.63),
        'pinky':  ('F4', 349.23),
    },
    'upper': {
        'index':  ('G4', 392.00),
        'middle': ('A4', 440.00),
        'ring':   ('B4', 493.88),
        'pinky':  ('C5', 523.25),
    },
}

STRINGS = [
    {'label': 'C5', 'y_frac': 1/16,  'region': 'upper', 'finger': 'pinky'},
    {'label': 'B',  'y_frac': 3/16,  'region': 'upper', 'finger': 'ring'},
    {'label': 'A',  'y_frac': 5/16,  'region': 'upper', 'finger': 'middle'},
    {'label': 'G',  'y_frac': 7/16,  'region': 'upper', 'finger': 'index'},
    {'label': 'F',  'y_frac': 9/16,  'region': 'lower', 'finger': 'pinky'},
    {'label': 'E',  'y_frac': 11/16, 'region': 'lower', 'finger': 'ring'},
    {'label': 'D',  'y_frac': 13/16, 'region': 'lower', 'finger': 'middle'},
    {'label': 'C',  'y_frac': 15/16, 'region': 'lower', 'finger': 'index'},
]

PINCH_THRESHOLD = 0.1
PINCH_DEBOUNCE_FRAMES = 1
DISC_RADIUS = 20
SAMPLE_RATE = 44100
DEFAULT_FREQ = 261.63

COLORS = {
    'bg_upper':      (255, 200, 100),
    'bg_lower':      (100, 230, 120),
    'string':        (128, 128, 128),
    'string_highlight': (255, 0, 255),
    'disc_fill':     (0, 255, 255),
    'disc_border':   (0, 200, 200),
    'panel_bg':      (0, 0, 0),
    'left_landmark': (0, 255, 0),
    'left_connect':  (0, 180, 0),
    'right_landmark':(255, 0, 0),
    'right_connect': (0, 0, 180),
    'left_tip':      (0, 255, 0),
    'right_tip_active':   (0, 255, 255),
    'right_tip_inactive': (255, 0, 0),
    'thumb':         (0, 0, 255),
    'text_active':   (0, 255, 0),
    'text_magenta':  (255, 0, 255),
    'text_cyan':     (0, 255, 255),
    'text_dim':      (120, 120, 120),
    'text_muted':    (0, 0, 255),
    'text_light':    (180, 180, 180),
    'white':         (255, 255, 255),
}
