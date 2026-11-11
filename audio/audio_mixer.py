
import threading

import librosa
import numpy as np
import sounddevice as sd


class DualDeckMixer:

    def __init__(self, sample_rate=44100, blocksize=1024):

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if blocksize <= 0:
            raise ValueError("Blocksize must be positive.")

        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self._stream = None

        self._decks = [None, None]
        self._positions = [0, 0]

        self._deck_started = [False, False]
        self._deck_active = [False, False]

        self._trim_gains = [1.0, 1.0]

        self._crossfader = 0.0

        self._ramp_start = 0.0
        self._ramp_target = 0.0
        self._ramp_total_frames = 0
        self._ramp_progress = 0

        self._transition_at_sample = None
        self._transition_duration = 0.0
        self._transition_triggered = False

        self._control_lock = threading.RLock()
        self._finished = threading.Event()

        self._active = False
        self._stopping = False

    # ==========================================
    # AUDIO PREPARATION
    # ==========================================

    def _prepare_audio(
        self,
        audio_data,
        original_sr,
        deck_index,
        tempo_ratio=1.0
    ):

        audio = np.asarray(audio_data, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("Audio data is empty.")

        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains invalid values.")

        if original_sr <= 0:
            raise ValueError("Invalid sample rate.")

        if not np.isfinite(tempo_ratio) or tempo_ratio <= 0:
            raise ValueError("Tempo ratio must be positive.")

        # Convert to channels-first format.
        if audio.ndim == 1:

            audio = audio[np.newaxis, :]

        elif audio.ndim == 2:

            if audio.shape[0] > 2 and audio.shape[1] <= 2:
                audio = audio.T

            if audio.shape[0] not in (1, 2):
                raise ValueError("Unsupported number of channels.")

        else:
            raise ValueError("Unsupported audio dimensions.")

        # Resample.
        if original_sr != self.sample_rate:

            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=self.sample_rate,
                axis=1
            )

        # Pitch-preserving tempo adjustment.
        if not np.isclose(tempo_ratio, 1.0):

            processed_channels = []

            for channel in audio:

                processed = librosa.effects.time_stretch(
                    channel,
                    rate=tempo_ratio
                )

                processed_channels.append(processed)

            min_length = min(
                len(channel)
                for channel in processed_channels
            )

            audio = np.stack([
                channel[:min_length]
                for channel in processed_channels
            ])

        # Convert mono to stereo.
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)

        # Convert to samples-first format.
        audio = audio.T

        return np.ascontiguousarray(
            audio,
            dtype=np.float32
        )

    # ==========================================
    # PLAY
    # ==========================================

    def play(
        self,
        audio_a,
        sr_a,
        audio_b,
        sr_b,
        ratio_a=1.0,
        ratio_b=1.0,
        start_offset_a=0.0,
        start_offset_b=0.0,
        transition_at=None,
        crossfade_duration=8.0
    ):

        self.stop()

        # Prepare both decks.
        deck_a = self._prepare_audio(
            audio_a,
            sr_a,
            0,
            ratio_a
        )

        deck_b = self._prepare_audio(
            audio_b,
            sr_b,
            1,
            ratio_b
        )

        # Validate offsets.
        offsets = [start_offset_a, start_offset_b]

        for offset in offsets:

            if not np.isfinite(offset) or offset < 0:
                raise ValueError(
                    "Start offsets must be finite and non-negative."
                )

        if (
            not np.isfinite(crossfade_duration)
            or crossfade_duration < 0
        ):
            raise ValueError(
                "Crossfade duration must be non-negative."
            )

        position_a = int(
            round(start_offset_a * self.sample_rate)
        )

        position_b = int(
            round(start_offset_b * self.sample_rate)
        )

        if position_a >= len(deck_a):
            raise ValueError("Track A offset exceeds duration.")

        if position_b >= len(deck_b):
            raise ValueError("Track B offset exceeds duration.")

        transition_sample = None

        if transition_at is not None:

            if not np.isfinite(transition_at):
                raise ValueError("Invalid transition time.")

            if transition_at <= start_offset_a:
                raise ValueError(
                    "Transition must happen after Track A starts."
                )

            transition_sample = int(
                round(transition_at * self.sample_rate)
            )

            if transition_sample >= len(deck_a):
                raise ValueError(
                    "Transition time exceeds Track A duration."
                )

        with self._control_lock:

            self._decks = [deck_a, deck_b]

            self._positions = [
                position_a,
                position_b
            ]

            # Deck A starts immediately.
            # Deck B waits for the scheduled transition.
            self._deck_started = [True, False]

            self._deck_active = [True, True]

            self._trim_gains = [1.0, 1.0]

            self._crossfader = 0.0

            self._ramp_start = 0.0
            self._ramp_target = 0.0
            self._ramp_total_frames = 0
            self._ramp_progress = 0

            self._transition_at_sample = transition_sample
            self._transition_duration = crossfade_duration
            self._transition_triggered = False

            self._finished.clear()

            self._active = True
            self._stopping = False

        try:

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=2,
                dtype="float32",
                callback=self._audio_callback
            )

            self._stream.start()

        except Exception:

            self._active = False
            self._finished.set()

            raise

    # ==========================================
    # SCHEDULED TRANSITION
    # ==========================================

    def _trigger_scheduled_transition(self):

        with self._control_lock:

            if self._transition_at_sample is None:
                return

            if self._transition_triggered:
                return

            current_position = self._positions[0]

            if current_position < self._transition_at_sample:
                return

            # Start Deck B.
            if self._positions[1] >= len(self._decks[1]):

                self._deck_active[1] = False
                self._transition_triggered = True
                self._transition_at_sample = None

                return

            self._deck_started[1] = True
            self._deck_active[1] = True

            # Automatically start crossfade.
            self._ramp_start = self._crossfader
            self._ramp_target = 1.0

            self._ramp_total_frames = int(
                self._transition_duration * self.sample_rate
            )

            self._ramp_progress = 0

            if self._transition_duration == 0:
                self._crossfader = 1.0

            self._transition_triggered = True
            self._transition_at_sample = None

    # ==========================================
    # START DECK
    # ==========================================

    def start_deck(self, deck_index):

        if deck_index not in (0, 1):
            raise ValueError("Invalid deck index.")

        with self._control_lock:

            if not self._active:
                return

            if self._decks[deck_index] is None:
                return

            if self._positions[deck_index] >= len(
                self._decks[deck_index]
            ):
                self._deck_active[deck_index] = False
                return

            self._deck_started[deck_index] = True
            self._deck_active[deck_index] = True

    # ==========================================
    # CROSSFADE GAINS
    # ==========================================

    @staticmethod
    def _calculate_gains(value):

        value = np.clip(value, 0.0, 1.0)

        gain_a = np.cos(value * np.pi / 2)
        gain_b = np.sin(value * np.pi / 2)

        return gain_a, gain_b

    # ==========================================
    # GAIN CURVES
    # ==========================================

    def _get_gain_curves(self, frames):

        with self._control_lock:

            if self._ramp_total_frames > 0:

                remaining = (
                    self._ramp_total_frames
                    - self._ramp_progress
                )

                count = min(frames, remaining)

                progress = (
                    np.arange(count, dtype=np.float32)
                    + self._ramp_progress
                ) / self._ramp_total_frames

                progress = np.clip(progress, 0.0, 1.0)

                # Smoothstep interpolation.
                smooth = progress * progress * (
                    3.0 - 2.0 * progress
                )

                values = (
                    self._ramp_start
                    + (
                        self._ramp_target - self._ramp_start
                    ) * smooth
                )

                self._ramp_progress += count

                if self._ramp_progress >= self._ramp_total_frames:

                    self._crossfader = self._ramp_target

                    self._ramp_total_frames = 0
                    self._ramp_progress = 0

                if count < frames:

                    values = np.concatenate([
                        values,
                        np.full(
                            frames - count,
                            self._crossfader,
                            dtype=np.float32
                        )
                    ])

            else:

                values = np.full(
                    frames,
                    self._crossfader,
                    dtype=np.float32
                )

            gain_a = np.cos(values * np.pi / 2)
            gain_b = np.sin(values * np.pi / 2)

            gain_a *= self._trim_gains[0]
            gain_b *= self._trim_gains[1]

            return gain_a, gain_b

    # ==========================================
    # SET CROSSFADER
    # ==========================================

    def set_crossfader(self, value, duration=0.0):

        value = float(value)
        duration = float(duration)

        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "Crossfader must be between 0 and 1."
            )

        if not np.isfinite(duration) or duration < 0:
            raise ValueError(
                "Duration must be non-negative."
            )

        if value > 0:
            self.start_deck(1)

        if value < 1:
            self.start_deck(0)

        with self._control_lock:

            self._ramp_start = self._crossfader
            self._ramp_target = value

            self._ramp_total_frames = int(
                duration * self.sample_rate
            )

            self._ramp_progress = 0

            if duration == 0:
                self._crossfader = value

    # ==========================================
    # SMOOTH CROSSFADE
    # ==========================================

    def smooth_crossfade(self, target=1.0, duration=10.0):

        self.set_crossfader(
            target,
            duration
        )

    # ==========================================
    # SET DECK GAIN
    # ==========================================

    def set_gain(self, deck_index, gain):

        if deck_index not in (0, 1):
            raise ValueError("Invalid deck index.")

        gain = float(gain)

        if not np.isfinite(gain) or not 0.0 <= gain <= 2.0:
            raise ValueError("Gain must be between 0 and 2.")

        with self._control_lock:
            self._trim_gains[deck_index] = gain

    # ==========================================
    # AUDIO CALLBACK
    # ==========================================

    def _audio_callback(
        self,
        outdata,
        frames,
        time_info,
        status
    ):

        outdata.fill(0)

        if not self._active:
            raise sd.CallbackStop

        gain_a, gain_b = self._get_gain_curves(frames)

        gains = [gain_a, gain_b]

        with self._control_lock:

            decks = list(self._decks)
            positions = list(self._positions)
            started = list(self._deck_started)
            active = list(self._deck_active)

        for index in range(2):

            if not started[index] or not active[index]:
                continue

            deck = decks[index]

            if deck is None:
                continue

            position = positions[index]

            available = len(deck) - position

            if available <= 0:

                with self._control_lock:
                    self._deck_active[index] = False

                continue

            count = min(frames, available)

            outdata[:count] += (
                deck[position:position + count]
                * gains[index][:count, np.newaxis]
            )

            new_position = position + count

            with self._control_lock:

                self._positions[index] = new_position

                if new_position >= len(deck):
                    self._deck_active[index] = False

        # Check the transition after updating Track A's position.
        # Deck B will begin on the next audio callback.
        self._trigger_scheduled_transition()

        with self._control_lock:

            remaining_active = any(
                self._deck_active[i] and self._deck_started[i]
                for i in range(2)
            )

            if not remaining_active:
                self._active = False

        if not remaining_active:

            self._finished.set()
            raise sd.CallbackStop

        # Simple peak limiter.
        peak = np.max(np.abs(outdata))

        if peak > 0.98:
            outdata *= 0.98 / peak

    # ==========================================
    # POSITIONS
    # ==========================================

    def get_positions(self):

        with self._control_lock:

            return [
                self._positions[i] / self.sample_rate
                for i in range(2)
            ]

    # ==========================================
    # STATUS
    # ==========================================

    def is_playing(self):

        with self._control_lock:
            return self._active

    @property
    def active(self):
        return self.is_playing()

    def wait(self, timeout=None):
        return self._finished.wait(timeout)

    # ==========================================
    # STOP
    # ==========================================

    def stop(self):

        with self._control_lock:

            self._stopping = True
            self._active = False

            stream = self._stream
            self._stream = None

        if stream is not None:

            try:

                if stream.active:
                    stream.stop()

            except Exception:
                pass

            try:
                stream.close()

            except Exception:
                pass

        with self._control_lock:

            self._decks = [None, None]

            self._positions = [0, 0]

            self._deck_started = [False, False]
            self._deck_active = [False, False]

            self._crossfader = 0.0

            self._ramp_start = 0.0
            self._ramp_target = 0.0
            self._ramp_total_frames = 0
            self._ramp_progress = 0

            self._transition_at_sample = None
            self._transition_duration = 0.0
            self._transition_triggered = False

            self._finished.set()
            self._stopping = False
