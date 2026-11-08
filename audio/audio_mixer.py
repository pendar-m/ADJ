import time

import numpy as np
import sounddevice as sd
import librosa
import threading


class DualDeckMixer:

    def __init__(self, sample_rate=44100, blocksize=1024):

        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self._stream = None

        self._decks = []
        self._positions = [0, 0]
        self._deck_active = [False, False]

        self._trim_gains = (1.0, 1.0)
        self._crossfader = 0.0
        self._target_gains = (1.0, 0.0)
        self._current_gains = [1.0, 0.0]

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
                raise ValueError("Only mono and stereo audio are supported.")

        else:
            raise ValueError("Unsupported audio shape.")

        if original_sr != self.sample_rate:

            resampled_channels = []

            for channel in audio_data:

                resampled = librosa.resample(
                    y=channel,
                    orig_sr=original_sr,
                    target_sr=self.sample_rate
                )

                resampled_channels.append(resampled)

            audio_data = np.stack(resampled_channels)

        # Convert from (channels, samples) to (samples, channels)
        audio_data = audio_data.T

        return np.ascontiguousarray(audio_data, dtype=np.float32)

    # -------------------------
    # Playback
    # -------------------------

    def play(self, audio_a, sr_a, audio_b, sr_b):

        self.stop()

        deck_a = self._prepare_audio(audio_a, sr_a)
        deck_b = self._prepare_audio(audio_b, sr_b)

        self._decks = [deck_a, deck_b]

        self._positions = [0, 0]
        self._deck_active = [True, True]

        self._finished.clear()

        self._current_gains = list(self._target_gains)
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
    # Audio Callback
    # -------------------------

    def _audio_callback(self, outdata, frames, time_info, status):

        outdata.fill(0)

        if not self._active:
            raise sd.CallbackStop

        mixed_audio = np.zeros((frames, 2), dtype=np.float32)

        target_gains = self._target_gains

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

            audio_chunk = deck[
                position:position + chunk_size
            ]

            start_gain = self._current_gains[deck_index]
            end_gain = target_gains[deck_index]

            gain_curve = np.linspace(
                start_gain,
                end_gain,
                frames,
                endpoint=True,
                dtype=np.float32
            )

            mixed_audio[:chunk_size] += (
                audio_chunk * gain_curve[:chunk_size, np.newaxis]
            )

            self._positions[deck_index] += chunk_size

            if self._positions[deck_index] >= len(deck):
                self._deck_active[deck_index] = False

        self._current_gains = list(target_gains)

        # Prevent digital clipping while preserving normal levels.
        peak = np.max(np.abs(mixed_audio))

        if peak > 0.98:
            mixed_audio *= 0.98 / peak

        outdata[:] = mixed_audio

        if not any(self._deck_active):

            self._active = False
            self._finished.set()

            raise sd.CallbackStop

    # -------------------------
    # Gain Control
    # -------------------------

    def _update_target_gains(self):

        angle = self._crossfader * (np.pi / 2)

        crossfade_a = np.cos(angle)
        crossfade_b = np.sin(angle)

        gain_a = crossfade_a * self._trim_gains[0]
        gain_b = crossfade_b * self._trim_gains[1]

        self._target_gains = (gain_a, gain_b)

    def set_crossfader(self, value):

        value = float(np.clip(value, 0.0, 1.0))

        self._crossfader = value

        self._update_target_gains()

    def set_gain(self, deck_index, gain):

        if deck_index not in (0, 1):
            raise ValueError("Deck index must be 0 or 1.")

        gain = float(max(0.0, gain))

        gains = list(self._trim_gains)
        gains[deck_index] = gain

        self._trim_gains = tuple(gains)

        self._update_target_gains()

    # -------------------------
    # Tempo Adjustment
    # -------------------------

    def adjust_tempo(self, audio_data, tempo_ratio):

        if tempo_ratio <= 0:
            raise ValueError("Tempo ratio must be greater than zero.")

        audio_data = np.asarray(audio_data, dtype=np.float32)

        if audio_data.ndim == 1:

            adjusted_audio = librosa.effects.time_stretch(
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

            adjusted_audio = np.stack(adjusted_channels)

        else:
            raise ValueError("Unsupported audio shape.")

        return np.ascontiguousarray(
            adjusted_audio,
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

        stream = self._stream

        self._active = False

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
    def smooth_crossfade(self, target, duration=10.0):

        target = float(np.clip(target, 0.0, 1.0))

        if duration <= 0:
            self.set_crossfader(target)
            return

        start = self._crossfader
        steps = max(1, int(duration * 30))

        for step in range(1, steps + 1):

            progress = step / steps

            # Smooth easing curve
            progress = (
                3 * progress ** 2
                - 2 * progress ** 3
            )

            value = start + (target - start) * progress

            self.set_crossfader(value)

            time.sleep(duration / steps)