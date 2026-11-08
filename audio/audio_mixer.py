
import threading

import numpy as np
import sounddevice as sd
import librosa


class DualDeckMixer:

    def __init__(self, sample_rate=44100, blocksize=1024):

        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self._stream = None

        self._decks = []
        self._positions = [0, 0]
        self._deck_active = [False, False]

        self._trim_gains = [1.0, 1.0]

        self._crossfader = 0.0
        self._ramp_start = 0.0
        self._ramp_target = 0.0

        self._ramp_total_frames = 0
        self._ramp_progress = 0

        self._control_lock = threading.Lock()
        self._finished = threading.Event()

        self._active = False

    # -------------------------
    # Audio Preparation
    # -------------------------

    def _prepare_audio(self, audio_data, original_sr):

        audio_data = np.asarray(audio_data, dtype=np.float32)

        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])

        elif audio_data.ndim == 2:

            if audio_data.shape[0] == 1:
                audio_data = np.repeat(audio_data, 2, axis=0)

            elif audio_data.shape[0] != 2:
                raise ValueError(
                    "Only mono and stereo audio are supported."
                )

        else:
            raise ValueError("Unsupported audio shape.")

        if original_sr != self.sample_rate:

            channels = []

            for channel in audio_data:

                resampled = librosa.resample(
                    y=channel,
                    orig_sr=original_sr,
                    target_sr=self.sample_rate
                )

                channels.append(resampled)

            audio_data = np.stack(channels)

        return np.ascontiguousarray(
            audio_data.T,
            dtype=np.float32
        )

    # -------------------------
    # Playback
    # -------------------------

    def play(self, audio_a, sr_a, audio_b, sr_b):

        self.stop()

        deck_a = self._prepare_audio(audio_a, sr_a)
        deck_b = self._prepare_audio(audio_b, sr_b)

        if len(deck_a) == 0 or len(deck_b) == 0:
            raise ValueError("Audio tracks cannot be empty.")

        self._decks = [deck_a, deck_b]

        self._positions = [0, 0]
        self._deck_active = [True, True]

        self._finished.clear()

        with self._control_lock:

            self._crossfader = 0.0
            self._ramp_start = 0.0
            self._ramp_target = 0.0

            self._ramp_total_frames = 0
            self._ramp_progress = 0

        self._active = True

        try:

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._audio_callback
            )

            self._stream.start()

        except Exception:

            self._active = False
            self._finished.set()

            if self._stream is not None:

                self._stream.close()
                self._stream = None

            raise

    # -------------------------
    # Crossfade Processing
    # -------------------------

    @staticmethod
    def _calculate_gains(crossfader):

        angle = crossfader * (np.pi / 2)

        gain_a = np.cos(angle)
        gain_b = np.sin(angle)

        return gain_a, gain_b

    def _get_gain_curves(self, frames):

        with self._control_lock:

            start = self._ramp_progress
            total = self._ramp_total_frames

            ramp_start = self._ramp_start
            ramp_target = self._ramp_target

            trims = tuple(self._trim_gains)

            if total <= 0:

                values = np.full(
                    frames,
                    self._crossfader,
                    dtype=np.float32
                )

            else:

                positions = np.minimum(
                    np.arange(start + 1, start + frames + 1),
                    total
                )

                progress = positions / total

                # Smoothstep easing
                progress = (
                    3 * progress ** 2
                    - 2 * progress ** 3
                )

                values = (
                    ramp_start
                    + (ramp_target - ramp_start) * progress
                ).astype(np.float32)

                self._ramp_progress = min(
                    start + frames,
                    total
                )

                if self._ramp_progress >= total:

                    self._crossfader = ramp_target
                    self._ramp_total_frames = 0

        angle = values * (np.pi / 2)

        gain_a = np.cos(angle) * trims[0]
        gain_b = np.sin(angle) * trims[1]

        return gain_a, gain_b

    # -------------------------
    # Audio Callback
    # -------------------------

    def _audio_callback(self, outdata, frames, time_info, status):

        outdata.fill(0)

        if not self._active:
            raise sd.CallbackStop

        gain_a, gain_b = self._get_gain_curves(frames)

        gains = (gain_a, gain_b)

        for deck_index in range(2):

            if not self._deck_active[deck_index]:
                continue

            deck = self._decks[deck_index]
            position = self._positions[deck_index]

            remaining = len(deck) - position

            if remaining <= 0:

                self._deck_active[deck_index] = False
                continue

            chunk_size = min(frames, remaining)

            chunk = deck[
                position:position + chunk_size
            ]

            outdata[:chunk_size] += (
                chunk * gains[deck_index][
                    :chunk_size, np.newaxis
                ]
            )

            self._positions[deck_index] += chunk_size

            if self._positions[deck_index] >= len(deck):
                self._deck_active[deck_index] = False

        # Prevent digital clipping
        peak = np.max(np.abs(outdata))

        if peak > 0.98:
            outdata *= 0.98 / peak

        if not any(self._deck_active):

            self._active = False
            self._finished.set()

            raise sd.CallbackStop

    # -------------------------
    # Crossfader Control
    # -------------------------

    def set_crossfader(self, value, duration=0.0):

        value = float(np.clip(value, 0.0, 1.0))

        with self._control_lock:

            current = self._crossfader

            if self._ramp_total_frames > 0:

                progress = min(
                    self._ramp_progress / self._ramp_total_frames,
                    1.0
                )

                progress = (
                    3 * progress ** 2
                    - 2 * progress ** 3
                )

                current = (
                    self._ramp_start
                    + (self._ramp_target - self._ramp_start)
                    * progress
                )

            self._ramp_start = current
            self._ramp_target = value
            self._ramp_progress = 0

            self._ramp_total_frames = max(
                0,
                int(duration * self.sample_rate)
            )

            if self._ramp_total_frames == 0:

                self._crossfader = value
                self._ramp_start = value

    def smooth_crossfade(self, target, duration=10.0):

        self.set_crossfader(
            value=target,
            duration=duration
        )

    # -------------------------
    # Deck Gain
    # -------------------------

    def set_gain(self, deck_index, gain):

        if deck_index not in (0, 1):
            raise ValueError("Deck index must be 0 or 1.")

        gain = float(max(0.0, gain))

        with self._control_lock:
            self._trim_gains[deck_index] = gain

    # -------------------------
    # Offline Tempo Adjustment
    # -------------------------

    def adjust_tempo(self, audio_data, tempo_ratio):

        if tempo_ratio <= 0:
            raise ValueError(
                "Tempo ratio must be greater than zero."
            )

        audio_data = np.asarray(
            audio_data,
            dtype=np.float32
        )

        if audio_data.ndim == 1:

            adjusted = librosa.effects.time_stretch(
                audio_data,
                rate=tempo_ratio
            )

        elif audio_data.ndim == 2:

            adjusted_channels = []

            for channel in audio_data:

                adjusted_channel = librosa.effects.time_stretch(
                    channel,
                    rate=tempo_ratio
                )

                adjusted_channels.append(adjusted_channel)

            adjusted = np.stack(adjusted_channels)

        else:
            raise ValueError("Unsupported audio shape.")

        return np.ascontiguousarray(
            adjusted,
            dtype=np.float32
        )

    # -------------------------
    # Playback Information
    # -------------------------

    def get_positions(self):

        return (
            self._positions[0] / self.sample_rate,
            self._positions[1] / self.sample_rate
        )

    def is_playing(self):

        return self._active and not self._finished.is_set()

    @property
    def active(self):

        return self.is_playing()

    # -------------------------
    # Stop
    # -------------------------

    def stop(self):

        self._active = False

        stream = self._stream

        if stream is not None:

            try:
                stream.stop()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

            self._stream = None

        self._deck_active = [False, False]
        self._finished.set()
