import librosa
import soundfile as sf

def load_audio(file_path):
    """
    Load an audio file and return its samples and sample rate using librosa.
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise


def get_duration(audio_data, sample_rate):
    """
    Calculate audio duration in seconds.
    """
    if audio_data.ndim > 1:
        total_samples = audio_data.shape[1]
    else:
        total_samples = len(audio_data)

    duration = total_samples / sample_rate
    return duration


def get_metadata(file_path):
    """
    Extract basic audio metadata. Tries soundfile first, 
    and falls back to librosa if needed.
    """
    try:
        info = sf.info(file_path)
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        frames = y.shape[1] if y.ndim > 1 else len(y)
        return {
            "sample_rate": sr,
            "channels": channels,
            "frames": frames,
            "duration": frames / sr,
            "format": "SUPPORTED (librosa)",
            "subtype": "PCM/FLOAT"
        }