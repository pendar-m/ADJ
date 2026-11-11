import ctypes
import platform
from pathlib import Path
import numpy as np
import librosa


class SoundTouchProcessor:

    def __init__(self, sample_rate, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.fallback_mode = False
        self._buffer = []
        self.tempo_ratio = 1.0

        system = platform.system()
        if system == "Windows":
            lib_name = "SoundTouch.dll"
        elif system == "Darwin":
            lib_name = "libSoundTouch.dylib"
        else:
            lib_name = "libSoundTouch.so"

        dll_path = Path(__file__).resolve().parent.parent / lib_name
        if not dll_path.exists():
            dll_path = Path.cwd() / lib_name

        try:
            if dll_path.exists():
                self.dll = ctypes.CDLL(str(dll_path))
                self._setup_functions()
                self.handle = self.dll.soundtouch_createInstance()
                if self.handle:
                    self.dll.soundtouch_setSampleRate(
                        self.handle, ctypes.c_uint(sample_rate)
                    )
                    self.dll.soundtouch_setChannels(
                        self.handle, ctypes.c_uint(channels)
                    )
                    self.set_tempo_ratio(1.0)
                else:
                    self.fallback_mode = True
            else:
                self.fallback_mode = True
        except Exception:
            self.fallback_mode = True

    def _setup_functions(self):
        """Configure C function signatures for SoundTouch."""
        self.dll.soundtouch_createInstance.restype = ctypes.c_void_p
        self.dll.soundtouch_destroyInstance.argtypes = [ctypes.c_void_p]

        self.dll.soundtouch_setSampleRate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

        self.dll.soundtouch_setChannels.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

        self.dll.soundtouch_setTempo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
        ]

        self.dll.soundtouch_putSamples.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
        ]

        self.dll.soundtouch_receiveSamples.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
        ]
        self.dll.soundtouch_receiveSamples.restype = ctypes.c_uint

        self.dll.soundtouch_numSamples.argtypes = [ctypes.c_void_p]
        self.dll.soundtouch_numSamples.restype = ctypes.c_uint

        self.dll.soundtouch_flush.argtypes = [ctypes.c_void_p]
        self.dll.soundtouch_clear.argtypes = [ctypes.c_void_p]

    def set_tempo_ratio(self, ratio):
        """Set tempo stretching ratio."""
        if not 0.5 <= ratio <= 2.0:
            raise ValueError("Tempo ratio must be between 0.5 and 2.0.")
        self.tempo_ratio = ratio
        if not self.fallback_mode and self.handle:
            self.dll.soundtouch_setTempo(self.handle, ctypes.c_float(ratio))

    def process_chunk(self, audio_chunk):
        """Process chunk through SoundTouch or buffer for fallback mode."""
        audio_chunk = np.ascontiguousarray(audio_chunk, dtype=np.float32)

        if audio_chunk.ndim != 2:
            raise ValueError("Audio must have shape (frames, channels).")

        if audio_chunk.shape[1] != self.channels:
            raise ValueError("Channel count mismatch.")

        if self.fallback_mode:
            self._buffer.append(audio_chunk)
            return np.empty((0, self.channels), dtype=np.float32)

        frames = len(audio_chunk)
        self.dll.soundtouch_putSamples(
            self.handle,
            audio_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint(frames),
        )

        return self._receive_available()

    def _receive_available(self):
        available = self.dll.soundtouch_numSamples(self.handle)

        if available == 0:
            return np.empty((0, self.channels), dtype=np.float32)

        output = np.empty((available, self.channels), dtype=np.float32)

        received = self.dll.soundtouch_receiveSamples(
            self.handle,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint(available),
        )

        return output[:received].copy()

    def flush(self):
        """Flush remaining audio, applying a fast fallback if C++ library is missing."""
        if self.fallback_mode:
            if not self._buffer:
                return np.empty((0, self.channels), dtype=np.float32)
            
            # Combine chunks
            full_audio = np.vstack(self._buffer)
            self._buffer = []
            
            # If tempo ratio is extremely close to 1.0, skip heavy stretching to prevent freezing!
            if abs(self.tempo_ratio - 1.0) < 0.001:
                return np.ascontiguousarray(full_audio, dtype=np.float32)

            stretched_channels = []
            for ch in range(self.channels):
                # Use a faster hop length or direct slice if needed, 
                # or fallback safely without freezing the terminal
                stretched = librosa.effects.time_stretch(
                    full_audio[:, ch], rate=self.tempo_ratio
                )
                stretched_channels.append(stretched)
            
            result = np.stack(stretched_channels, axis=-1)
            return np.ascontiguousarray(result, dtype=np.float32)

        self.dll.soundtouch_flush(self.handle)
        return self._receive_available()

    def clear(self):
        if not self.fallback_mode and self.handle:
            self.dll.soundtouch_clear(self.handle)

    def close(self):
        if not self.fallback_mode and self.handle:
            self.dll.soundtouch_destroyInstance(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()