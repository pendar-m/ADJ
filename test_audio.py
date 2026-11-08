import time

from audio.audio_loader import load_audio
from audio.audio_mixer import DualDeckMixer

from analysis.bpm_detector import detect_bpm
from analysis.beat_matcher import BeatMatcher
from analysis.beat_aligner import BeatAligner


# -------------------------
# Load Tracks
# -------------------------

track_a, sr_a = load_audio(
    "Calvin Harris - Feel So Close (Radio Edit).mp3"
)

track_b, sr_b = load_audio(
    "David Guetta - Titanium (feat. Sia).mp3"
)

print("\nTracks loaded successfully.")


# -------------------------
# Detect BPM
# -------------------------

analysis_a = detect_bpm(track_a, sr_a)
analysis_b = detect_bpm(track_b, sr_b)

bpm_a = analysis_a["bpm"]
bpm_b = analysis_b["bpm"]

beats_a = analysis_a["beat_times"]
beats_b = analysis_b["beat_times"]

print("\nBPM Analysis:")
print("Track A:", bpm_a)
print("Track B:", bpm_b)


# -------------------------
# Calculate Tempo Match
# -------------------------

matcher = BeatMatcher()

match_result = matcher.analyze(
    bpm_b,
    bpm_a
)

print("\nTempo Match:")
print(match_result)

if not match_result["can_match"]:
    raise ValueError("Tempo difference is too large.")


# -------------------------
# Adjust Track B Tempo
# -------------------------

tempo_ratio = match_result["tempo_ratio"]

mixer = DualDeckMixer()

adjusted_b = mixer.adjust_tempo(
    track_b,
    tempo_ratio
)

print("\nTempo adjustment completed.")


# -------------------------
# Beat Alignment
# -------------------------

aligner = BeatAligner()

# Track B's beat times must be converted
# into the adjusted audio's time scale.

adjusted_beats_b = beats_b / tempo_ratio

alignment = aligner.align(
    beats_a,
    adjusted_beats_b
)

cue_a = alignment["cue_a"]
cue_b = alignment["cue_b"]

print("\nBeat Alignment:")
print(alignment)


# -------------------------
# Trim Audio
# -------------------------

prepared_a = aligner.trim_audio(
    track_a,
    sr_a,
    cue_a
)

prepared_b = aligner.trim_audio(
    adjusted_b,
    sr_b,
    cue_b
)

print("\nAudio prepared.")


# -------------------------
# Start Mixer
# -------------------------

mixer.set_crossfader(0.0)

mixer.play(
    prepared_a,
    sr_a,
    prepared_b,
    sr_b
)

print("\nPlayback started.")


# -------------------------
# Crossfade Test
# -------------------------

time.sleep(5)

print("Starting smooth crossfade...")

mixer.smooth_crossfade(
    target=1.0,
    duration=16.0
)


# -------------------------
# Wait Until Finished
# -------------------------

while mixer.is_playing():
    time.sleep(0.1)

print("\nPlayback finished.")

mixer.stop()