import time
import numpy as np

from audio.audio_loader import load_audio
from audio.audio_mixer import DualDeckMixer

from analysis.bpm_detector import detect_bpm
from analysis.beat_matcher import BeatMatcher
from analysis.beat_aligner import BeatAligner
from analysis.beat_verifier import BeatVerifier
from analysis.phrase_detector import PhraseDetector
from analysis.structure_analyzer import StructureAnalyzer
from analysis.section_detector import SectionDetector
from analysis.transition_selector import TransitionCandidateSelector


TRACK_A_PATH = "Calvin Harris - Summer.mp3"
TRACK_B_PATH = "Calvin Harris - Feel So Close (Radio Edit).mp3"

CROSSFADE_DURATION = 8.0


def analyze_track(name, audio, sample_rate):
    result = detect_bpm(audio, sample_rate)

    bpm = result["bpm"]
    beat_times = np.asarray(
        result["beat_times"],
        dtype=np.float64
    )

    if len(beat_times) < 2:
        raise ValueError(f"{name}: Not enough beats detected.")

    intervals = np.diff(beat_times)
    median_interval = np.median(intervals)
    expected_interval = 60.0 / bpm

    # Reconstruct potentially missed beats.
    if expected_interval * 1.5 < median_interval < expected_interval * 2.5:
        reconstructed = []

        for i in range(len(beat_times) - 1):
            reconstructed.append(beat_times[i])

            midpoint = (
                beat_times[i] + beat_times[i + 1]
            ) / 2.0

            reconstructed.append(midpoint)

        reconstructed.append(beat_times[-1])

        beat_times = np.asarray(
            reconstructed,
            dtype=np.float64
        )

        print(f"{name}: Missing beats reconstructed.")

    print(f"\n{name}")
    print(f"BPM: {bpm}")
    print(f"Beats detected: {len(beat_times)}")
    print(f"Median beat interval: {np.median(np.diff(beat_times)):.4f}s")

    return {
        "bpm": bpm,
        "beat_times": beat_times
    }


# ==========================================
# LOAD AUDIO
# ==========================================

print("Loading tracks...")

audio_a, sr_a = load_audio(TRACK_A_PATH)
audio_b, sr_b = load_audio(TRACK_B_PATH)

print("Tracks loaded.")


# ==========================================
# BPM AND BEAT ANALYSIS
# ==========================================

print("\nAnalyzing BPM and beats...")

track_a = analyze_track("TRACK A", audio_a, sr_a)
track_b = analyze_track("TRACK B", audio_b, sr_b)

bpm_a = track_a["bpm"]
bpm_b = track_b["bpm"]

beats_a = track_a["beat_times"]
beats_b = track_b["beat_times"]


# ==========================================
# TEMPO MATCHING
# ==========================================

print("\nMatching tempos...")

beat_matcher = BeatMatcher(max_tempo_change=0.10)

tempo_result = beat_matcher.analyze(
    source_bpm=bpm_b,
    target_bpm=bpm_a
)

print("Tempo ratio:", tempo_result["tempo_ratio"])
print("Tempo change:", tempo_result["tempo_change_percent"], "%")
print("Can match:", tempo_result["can_match"])

if not tempo_result["can_match"]:
    raise ValueError("Tracks cannot be tempo matched.")

ratio_b = tempo_result["tempo_ratio"]


# ==========================================
# BEAT ALIGNMENT
# ==========================================

print("\nAligning beats...")

aligner = BeatAligner()

alignment = aligner.align(
    beats_a,
    beats_b
)

cue_a = alignment["cue_a"]
cue_b = alignment["cue_b"]

print("Track A Cue:", cue_a)
print("Track B Cue:", cue_b)


# ==========================================
# BEAT VERIFICATION
# ==========================================

print("\nVerifying beat alignment...")

verifier = BeatVerifier(tolerance=0.05)

verification = verifier.verify(
    beats_a=beats_a,
    beats_b=beats_b,
    bpm_a=bpm_a,
    bpm_b=bpm_b,
    cue_a=cue_a,
    cue_b=cue_b,
    tempo_ratio=ratio_b
)

print("Track A Tempo Valid:", verification["tempo_a_valid"])
print("Track B Tempo Valid:", verification["tempo_b_valid"])
print("Phase Aligned:", verification["phase_aligned"])
print(
    "Phase Difference:",
    round(verification["phase_difference"], 4),
    "seconds"
)


# ==========================================
# STRUCTURE ANALYSIS
# ==========================================

print("\nAnalyzing track structures...")

structure_analyzer = StructureAnalyzer(
    window_seconds=4.0
)

structure_a = structure_analyzer.analyze(
    audio_a,
    sr_a
)

structure_b = structure_analyzer.analyze(
    audio_b,
    sr_b
)

print("Track A segments:", len(structure_a["segments"]))
print("Track B segments:", len(structure_b["segments"]))


# ==========================================
# SECTION DETECTION
# ==========================================

print("\nDetecting sections...")

section_detector = SectionDetector()

sections_a = section_detector.detect(structure_a)
sections_b = section_detector.detect(structure_b)

print("Track A boundaries:", sections_a["boundaries"])
print("Track B boundaries:", sections_b["boundaries"])


# ==========================================
# PHRASE DETECTION
# ==========================================

print("\nDetecting phrases...")

phrase_detector = PhraseDetector(
    beats_per_phrase=16
)

phrases_a = phrase_detector.detect(
    beat_times=beats_a,
    bpm=bpm_a,
    cue_time=cue_a
)

phrases_b = phrase_detector.detect(
    beat_times=beats_b,
    bpm=bpm_b,
    cue_time=cue_b
)

print("Track A phrases:", phrases_a["count"])
print("Track B phrases:", phrases_b["count"])


# ==========================================
# TRANSITION SELECTION
# ==========================================

print("\nSelecting transition...")

selector = TransitionCandidateSelector(
    top_k=10
)

selection = selector.select(
    phrases_a=phrases_a,
    phrases_b=phrases_b,
    structure_a=structure_a,
    structure_b=structure_b,
    sections_a=sections_a,
    sections_b=sections_b,
    cue_a=cue_a,
    cue_b=cue_b,
    ratio_b=ratio_b
)

best = selection["best_candidate"]

print("Total candidates:", selection["total_combinations"])

if best is None:
    raise ValueError("No valid transition candidates found.")

print("\n========== BEST TRANSITION ==========")

print("Track A transition:", best["source_time_a"])
print("Track B entry:", best["source_time_b"])
print("Track B processed entry:", best["processed_time_b"])

print("Energy A:", best["energy_a"])
print("Energy B:", best["energy_b"])

print("Trend A:", best["trend_a"])
print("Trend B:", best["trend_b"])

print("Score:", best["score"])

print("\n========== TOP CANDIDATES ==========")

for index, candidate in enumerate(selection["candidates"], start=1):
    print(
        f"{index}. "
        f"A: {candidate['source_time_a']:.2f}s | "
        f"B: {candidate['source_time_b']:.2f}s | "
        f"Score: {candidate['score']:.4f}"
    )


# ==========================================
# AUDIO PLAYBACK
# ==========================================

print("\nPreparing mixer...")

mixer = DualDeckMixer(sample_rate=sr_a)

try:
    print("\nStarting playback...")

    mixer.play(
        audio_a=audio_a,
        sr_a=sr_a,

        audio_b=audio_b,
        sr_b=sr_b,

        ratio_a=1.0,
        ratio_b=ratio_b,

        start_offset_a=0.0,
        start_offset_b=best["processed_time_b"],

        transition_at=best["source_time_a"],
        crossfade_duration=CROSSFADE_DURATION
    )

    print("Track A started.")
    print("Track B scheduled.")
    print("Transition at:", best["source_time_a"])

    while mixer.is_playing():
        positions = mixer.get_positions()

        print(
            f"\rDeck A: {positions[0]:.2f}s | "
            f"Deck B: {positions[1]:.2f}s",
            end="",
            flush=True
        )

        time.sleep(0.5)

    print("\n\nPlayback finished.")

finally:
    mixer.stop()
    print("Mixer stopped.")