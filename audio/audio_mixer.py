
import numpy as np
import sounddevice as sd
import librosa


class DualDeckMixer:

    def __init__(self, target_sample_rate=44100):
        self.target_sample_rate = target_sample_rate

        self.decks = [None, None]
        self.gains = (0.5, 0.5)
        self.crossfader = 0.5
        self.active = [False, False]

        self.stream = None

    def _prepare_audio(self, audio_data, sample_rate):
        """
        Convert audio to stereo and resample if necessary.
        """

        audio_data = np.asarray(audio_data, dtype=np.float32)

        # Convert mono to stereo.
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])

        elif audio_data.shape[0] == 1:
            audio_data = np.repeat(audio_data, 2, axis=0)

        elif audio_data.shape[0] > 2:
            raise ValueError("Only mono and stereo audio are supported.")

        # Resample if necessary.
        if sample_rate != self.target_sample_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate,
                axis=1
            )

        # Convert channel-first to channel-last.
        return np.ascontiguousarray(audio_data.T, dtype=np.float32)

    def play(self, audio_a, sr_a, audio_b, sr_b):
        """
        Load two tracks and play them simultaneously.
        """

        self.stop()

        self.decks[0] = self._prepare_audio(audio_a, sr_a)
        self.decks[1] = self._prepare_audio(audio_b, sr_b)

        self.positions = [0, 0]
        self.active = [
            len(self.decks[0]) > 0,
            len(self.decks[1]) > 0
        ]

        self.stream = sd.OutputStream(
            samplerate=self.target_sample_rate,
            channels=2,
            dtype="float32",
            callback=self._audio_callback
        )

        self.stream.start()

    def _audio_callback(self, outdata, frames, time, status):
        """
        Mix audio from both decks.
        """

        mixed_audio = np.zeros((frames, 2), dtype=np.float32)

        gains = self.gains

        for deck_index in range(2):

            if not self.active[deck_index]:
                continue

            audio = self.decks[deck_index]

            start = self.positions[deck_index]
            end = min(start + frames, len(audio))

            chunk = audio[start:end]

            mixed_audio[:len(chunk)] += (
                chunk * gains[deck_index])

            self.positions[deck_index] = end

            if end >= len(audio):
                self.active[deck_index] = False

        outdata[:] = np.clip(mixed_audio, -1.0, 1.0)

        if not any(self.active):
            raise sd.CallbackStop

    def stop(self):
        """
        Stop the mixer.
        """

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        self.active = [False, False]
        self.positions = [0, 0]

    def get_positions(self):
        """
        Return playback positions in seconds.
        """

        return [
            position / self.target_sample_rate
            for position in self.positions
        ]
    def set_gain(self, deck_index, gain):
        if deck_index not in (0, 1):
            raise ValueError("Deck index must be 0 or 1.")

        gain = max(0.0, min(1.0, gain))

        gains = list(self.gains)
        gains[deck_index] = gain

        self.gains = tuple(gains)


    def set_crossfader(self, value):
        value = max(0.0, min(1.0, value))

        self.crossfader = value

        angle = value * (np.pi / 2)

        gain_a = np.cos(angle)
        gain_b = np.sin(angle)

        self.gains = (gain_a, gain_b)
    
    def adjust_tempo(self, audio_data, tempo_ratio):
        """
        Change audio playback speed while preserving pitch.
        """

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
