import librosa
import numpy as np


def detect_bpm(audio_data, sample_rate):

    audio_data = np.asarray(audio_data, dtype=np.float32)

    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=0)

    elif audio_data.ndim != 1:
        raise ValueError("Unsupported audio shape.")

    tempo, beat_frames = librosa.beat.beat_track(
        y=audio_data,
        sr=sample_rate
    )

    bpm = float(np.asarray(tempo).reshape(-1)[0])

    # Correct half-time and double-time ambiguity
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