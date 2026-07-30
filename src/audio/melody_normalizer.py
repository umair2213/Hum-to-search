"""
Melody normalization.

Responsibility:
    Clean Notes  -->  Intervals

Absolute pitch cannot be compared across different keys/octaves, so the
melody is converted into a key-independent sequence of pitch intervals.
"""

from typing import List, Dict


def notes_to_intervals(notes: List[Dict]) -> List[int]:

    pitches = [n["pitch"] for n in notes]

    intervals = []

    for i in range(1, len(pitches)):
        intervals.append(pitches[i] - pitches[i - 1])

    return intervals
