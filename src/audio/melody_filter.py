"""
Melody cleaning / filtering.

Responsibility:
    Raw Notes  -->  Clean Notes

Only cleaning. No extraction, no normalization, no embedding.

Filters applied (in order):
    1. invalid pitches / impossible durations
    2. duplicate / overlapping notes within a short time window
    3. impossible pitch jumps
    4. harmonic (octave) false-detections
    5. light vibrato smoothing (merge near-identical adjacent notes)
"""

from typing import List, Dict


def filter_invalid_notes(
    notes: List[Dict],
    min_pitch: int = 40,
    max_pitch: int = 80,
    min_duration: float = 0.12,
) -> List[Dict]:

    filtered = []

    for note in notes:

        pitch = note["pitch"]
        duration = note["end"] - note["start"]

        # Remove unrealistic pitches
        if pitch < min_pitch or pitch > max_pitch:
            continue

        # Remove very short artifacts
        if duration < min_duration:
            continue

        filtered.append(note)

    return filtered


def remove_overlapping_duplicates(
    notes: List[Dict],
    window: float = 0.25,
) -> List[Dict]:

    if not notes:
        return []

    notes = sorted(notes, key=lambda x: x["start"])

    melody = []

    current_window = []
    window_start = notes[0]["start"]

    for note in notes:

        if note["start"] - window_start <= window:
            current_window.append(note)
        else:
            if current_window:
                best = max(current_window, key=lambda x: x["pitch"])
                melody.append(best)

            current_window = [note]
            window_start = note["start"]

    if current_window:
        best = max(current_window, key=lambda x: x["pitch"])
        melody.append(best)

    return melody


def remove_pitch_jumps(
    notes: List[Dict],
    max_jump: int = 12,
) -> List[Dict]:

    cleaned = []

    for note in notes:

        if not cleaned:
            cleaned.append(note)
            continue

        previous_pitch = cleaned[-1]["pitch"]
        current_pitch = note["pitch"]

        jump = abs(current_pitch - previous_pitch)

        # remove impossible jumps
        if jump <= max_jump:
            cleaned.append(note)

    return cleaned


def remove_harmonic_notes(
    notes: List[Dict],
    harmonic_jumps=(12, 24, 36),
) -> List[Dict]:

    cleaned = []

    previous_pitch = None

    for note in notes:

        pitch = note["pitch"]

        if previous_pitch is None:
            cleaned.append(note)
            previous_pitch = pitch
            continue

        difference = abs(pitch - previous_pitch)

        # remove octave/harmonic jumps
        if difference in harmonic_jumps:
            continue

        cleaned.append(note)
        previous_pitch = pitch

    return cleaned


def smooth_vibrato(
    notes: List[Dict],
    pitch_tolerance: int = 2,
) -> List[Dict]:

    if not notes:
        return []

    smoothed = []

    current = dict(notes[0])

    for note in notes[1:]:

        pitch_diff = abs(note["pitch"] - current["pitch"])

        # same note / vibrato -> merge
        if pitch_diff <= pitch_tolerance:
            current["end"] = note["end"]
        else:
            smoothed.append(current)
            current = dict(note)

    smoothed.append(current)

    return smoothed


def clean_notes(
    raw_notes: List[Dict],
    min_pitch: int = 40,
    max_pitch: int = 80,
    min_duration: float = 0.12,
    dedup_window: float = 0.25,
    max_jump: int = 12,
) -> List[Dict]:
    """
    Full production cleaning pipeline.

    Raw Notes -> Clean Notes

    Avoids excessive smoothing so real melody is preserved.
    """

    notes = sorted(raw_notes, key=lambda x: x["start"])

    notes = filter_invalid_notes(
        notes,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        min_duration=min_duration,
    )

    notes = remove_overlapping_duplicates(notes, window=dedup_window)

    notes = remove_pitch_jumps(notes, max_jump=max_jump)

    notes = remove_harmonic_notes(notes)

    notes = smooth_vibrato(notes)

    return notes