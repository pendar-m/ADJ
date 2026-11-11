
import numpy as np
import librosa


class StructureAnalyzer:

    def __init__(self, window_seconds=4.0):
        if window_seconds <= 0:
            raise ValueError("Window duration must be positive.")

        self.window_seconds = window_seconds

    def _to_mono(self, audio_data):

        audio_data = np.asarray(audio_data, dtype=np.float32)

        if audio_data.size == 0:
            raise ValueError("Audio data cannot be empty.")

        if not np.all(np.isfinite(audio_data)):
            raise ValueError("Audio contains invalid values.")

        if audio_data.ndim == 1:
            return audio_data

        if audio_data.ndim != 2:
            raise ValueError("Unsupported audio shape.")

        # Support channels-first and samples-first audio
        if audio_data.shape[0] <= 8:
            return np.mean(audio_data, axis=0)

        if audio_data.shape[1] <= 8:
            return np.mean(audio_data, axis=1)

        raise ValueError("Unsupported channel layout.")

    def analyze(self, audio_data, sample_rate):

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        y = self._to_mono(audio_data)

        if len(y) < sample_rate:
            raise ValueError("Audio must be at least one second long.")

        window_samples = int(
            self.window_seconds * sample_rate
        )

        segments = []

        for start in range(0, len(y), window_samples):

            end = min(start + window_samples, len(y))

            chunk = y[start:end]

            if len(chunk) < sample_rate:
                continue

            # RMS Energy
            rms = librosa.feature.rms(
                y=chunk,
                frame_length=2048,
                hop_length=512
            )[0]

            energy = float(np.mean(rms))

            # Spectral Centroid
            centroid = librosa.feature.spectral_centroid(
                y=chunk,
                sr=sample_rate,
                hop_length=512
            )[0]

            brightness = float(np.mean(centroid))

            # Onset Strength
            onset = librosa.onset.onset_strength(
                y=chunk,
                sr=sample_rate,
                hop_length=512
            )

            rhythmic_activity = float(np.mean(onset))

            segments.append({
                "start": start / sample_rate,
                "end": end / sample_rate,
                "energy": energy,
                "brightness": brightness,
                "rhythmic_activity": rhythmic_activity
            })

        if not segments:
            raise ValueError("Could not analyze audio segments.")

        # Normalize energy for relative comparison
        energies = np.array(
            [segment["energy"] for segment in segments]
        )

        max_energy = float(np.max(energies))

        if max_energy > 0:
            normalized = energies / max_energy
        else:
            normalized = np.zeros_like(energies)

        # Estimate energy state
        for index, segment in enumerate(segments):

            current = normalized[index]

            if current >= 0.75:
                state = "HIGH"
            elif current >= 0.40:
                state = "MEDIUM"
            else:
                state = "LOW"

            segment["normalized_energy"] = round(
                float(current), 3
            )

            segment["energy_state"] = state

        return {
            "duration": len(y) / sample_rate,
            "window_seconds": self.window_seconds,
            "segments": segments
        }
