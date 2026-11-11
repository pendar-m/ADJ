
import librosa
import numpy as np


def detect_bpm(audio_data, sample_rate):

    audio_data = np.asarray(audio_data, dtype=np.float32)

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if audio_data.size == 0:
        raise ValueError("Audio data cannot be empty.")

    if not np.all(np.isfinite(audio_data)):
        raise ValueError("Audio contains invalid values.")

    if audio_data.ndim == 2:

        # librosa.load(mono=False) returns (channels, samples)
        if audio_data.shape[0] in (1, 2):
            audio_data = np.mean(audio_data, axis=0)

        # Also support (samples, channels)
        elif audio_data.shape[1] in (1, 2):
            audio_data = np.mean(audio_data, axis=1)

        else:
            raise ValueError("Unsupported audio channel layout.")

    elif audio_data.ndim != 1:
        raise ValueError("Unsupported audio shape.")

    if len(audio_data) < sample_rate * 2:
        raise ValueError("Audio must be at least 2 seconds long.")

    if np.max(np.abs(audio_data)) < 1e-7:
        raise ValueError("Audio contains silence.")

    tempo, beat_frames = librosa.beat.beat_track(
        y=audio_data,
        sr=sample_rate
    )

    bpm = float(np.asarray(tempo).reshape(-1)[0])

    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("Could not detect a valid BPM.")

    # Half-time / double-time normalization
    if bpm < 80:
        bpm *= 2

    elif bpm > 160:
        bpm /= 2

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate
    )

    return {
        "bpm": round(bpm, 2),
        "beat_times": beat_times
    }
