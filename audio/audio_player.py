import numpy as np
import sounddevice as sd


class AudioPlayer:

    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.position = 0

        self._stream = None
        self._paused = False
        self._is_playing = False

    def play(self, audio_data, sample_rate):
        """
        Start playing audio from the beginning.
        """

        self.stop()

        audio_data = np.asarray(audio_data, dtype=np.float32)

        # Convert Librosa's channel-first format
        # to SoundDevice's channel-last format.
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        elif audio_data.ndim == 2:
            audio_data = audio_data.T

        self.audio_data = np.ascontiguousarray(audio_data)
        self.sample_rate = sample_rate
        self.position = 0
        self._paused = False
        self._is_playing = True

        channels = self.audio_data.shape[1]

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._audio_callback
        )

        self._stream.start()

    def _audio_callback(self, outdata, frames, time, status):
        """
        Called automatically whenever SoundDevice
        needs another block of audio.
        """

        outdata.fill(0)

        if self._paused:
            return

        start = self.position
        end = min(start + frames, len(self.audio_data))

        audio_chunk = self.audio_data[start:end]

        outdata[:len(audio_chunk)] = audio_chunk

        self.position = end

        if end >= len(self.audio_data):
            self._is_playing = False
            raise sd.CallbackStop

    def pause(self):
        """
        Pause playback without resetting position.
        """

        if self._is_playing:
            self._paused = True

    def resume(self):
        """
        Resume playback from the paused position.
        """

        if self._is_playing and self._paused:
            self._paused = False

    def stop(self):
        """
        Stop playback and reset position.
        """

        self._paused = False
        self._is_playing = False

        if self._stream is not None:
            self._stream.close()
            self._stream = None

        self.position = 0

    def is_playing(self):
        return self._is_playing and not self._paused

    def is_paused(self):
        return self._paused

    def get_position(self):
        """
        Return current playback position in seconds.
        """

        if self.audio_data is None or self.sample_rate is None:
            return 0

        return self.position / self.sample_rate