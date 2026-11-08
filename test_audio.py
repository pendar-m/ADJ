
import time

from audio.audio_loader import load_audio
from audio.audio_mixer import DualDeckMixer

from analysis.bpm_detector import detect_bpm
from analysis.beat_matcher import BeatMatcher


# 1. Load both tracks

audio_a, sr_a = load_audio(
    "Olly Alexander (Years   Years) - Desire (Gryffin Remix).mp3"
)

audio_b, sr_b = load_audio("Calvin Harris - Summer.mp3")


# 2. Detect BPM

bpm_a = detect_bpm(audio_a, sr_a)["bpm"]
bpm_b = detect_bpm(audio_b, sr_b)["bpm"]

print("Track A BPM:", bpm_a)
print("Track B BPM:", bpm_b)


# 3. Calculate tempo adjustment

matcher = BeatMatcher()

result = matcher.analyze(
    source_bpm=bpm_b,
    target_bpm=bpm_a
)

print("\nBeat Matching Analysis:")
print(result)

if not result["can_match"]:
    raise ValueError("Tempo difference is too large for matching.")


# 4. Adjust Track B tempo

mixer = DualDeckMixer()

adjusted_audio_b = mixer.adjust_tempo(
    audio_b,
    result["tempo_ratio"]
)

print("\nTempo adjustment completed.")
print("Adjusted Track B shape:", adjusted_audio_b.shape)


# 5. Play both tracks

mixer.set_crossfader(0.0)

mixer.play(
    audio_a,
    sr_a,
    adjusted_audio_b,
    sr_b
)

print("\nPlaying Track A only...")
time.sleep(5)

print("Crossfading to both tracks...")
mixer.set_crossfader(0.5)
time.sleep(10)

print("Playing Track B only...")
mixer.set_crossfader(1.0)

# Wait until both tracks finish
while any(mixer.active):
    time.sleep(0.5)

mixer.stop()

print("\nTest completed.")
