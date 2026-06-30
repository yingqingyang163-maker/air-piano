import numpy as np
import sounddevice as sd
from constants import SAMPLE_RATE, DEFAULT_FREQ

class AudioEngine:
    def __init__(self):
        self._muted = False
        self._current_freq = DEFAULT_FREQ
        self._target_freq = DEFAULT_FREQ
        self._phase = 0.0
        self._envelope_time = 0.0
        self._is_playing = False
        self.note_name = ""

        self._stream = sd.OutputStream(callback=self._callback, channels=1, samplerate=SAMPLE_RATE,
                                        blocksize=256, latency='low')
        self._stream.start()

    def _reset_envelope(self):
        self._envelope_time = 0.0

    def _envelope_array(self, t_arr):
        attack = 0.003
        decay = 0.06
        sustain_level = 0.35
        env = np.where(t_arr < attack,
                       t_arr / attack,
                       np.where(t_arr < attack + decay,
                                1.0 - (1.0 - sustain_level) * ((t_arr - attack) / decay),
                                sustain_level))
        return env

    def _callback(self, outdata, frames, time_info, status):
        if self._muted or not self._is_playing:
            outdata[:] = np.zeros((frames, 1))
            return

        alpha = 0.6
        self._current_freq = self._current_freq * (1 - alpha) + self._target_freq * alpha

        t_arr = np.arange(frames) / SAMPLE_RATE
        delta_phase = 2 * np.pi * self._current_freq / SAMPLE_RATE
        phases = self._phase + np.cumsum(delta_phase * np.ones(frames))
        self._phase = phases[-1] % (2 * np.pi)
        wave = (0.6 * np.sin(phases) + 0.35 * np.sin(2 * phases) +
                0.2 * np.sin(3 * phases) + 0.12 * np.sin(4 * phases) +
                0.06 * np.sin(5 * phases) + 0.03 * np.sin(6 * phases))

        env_time_arr = self._envelope_time + t_arr
        envelope = self._envelope_array(env_time_arr)
        self._envelope_time = env_time_arr[-1]

        outdata[:] = (0.25 * wave * envelope).reshape(-1, 1)

    def play_note(self, freq):
        was_silent = not self._is_playing
        self._is_playing = True
        if was_silent:
            self._target_freq = freq
            self._current_freq = freq
            self._reset_envelope()
        elif freq != self._target_freq:
            self._target_freq = freq

    def silence(self):
        self._is_playing = False

    @property
    def playing(self):
        return self._is_playing

    @property
    def freq(self):
        return self._current_freq

    @property
    def target_freq(self):
        return self._target_freq

    def toggle_mute(self):
        self._muted = not self._muted
        return self._muted

    @property
    def muted(self):
        return self._muted

    def cleanup(self):
        self._stream.stop()
        self._stream.close()
