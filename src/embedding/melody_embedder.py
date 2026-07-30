"""
Melody embedding.

Responsibility:
    Intervals  -->  Fixed-Length Vector

Deterministic, no ML model: the interval sequence is encoded as a
concatenation of

    1. an interval histogram   (distribution of semitone jumps)
    2. a resampled pitch-contour shape (overall melodic shape)
    3. summary statistics      (mean / std / min / max / etc.)

padded/truncated to a fixed size and L2-normalized.

No database logic lives here.
"""

from typing import List

import numpy as np


EMBEDDING_DIM = 768

# Interval histogram covers jumps from -24 to +24 semitones (inclusive).
HIST_MIN = -24
HIST_MAX = 24
HIST_BINS = HIST_MAX - HIST_MIN + 1  # 49

STATS_DIM = 7

CONTOUR_DIM = EMBEDDING_DIM - HIST_BINS - STATS_DIM  # 712


def _interval_histogram(intervals: List[int]) -> np.ndarray:

    hist = np.zeros(HIST_BINS, dtype=np.float64)

    if not intervals:
        return hist

    for interval in intervals:
        clipped = max(HIST_MIN, min(HIST_MAX, interval))
        hist[clipped - HIST_MIN] += 1.0

    hist /= len(intervals)

    return hist


def _pitch_contour(intervals: List[int]) -> np.ndarray:

    if not intervals:
        return np.zeros(CONTOUR_DIM, dtype=np.float64)

    # cumulative sum of intervals = relative pitch contour shape
    contour = np.concatenate(
        [[0.0], np.cumsum(np.asarray(intervals, dtype=np.float64))]
    )

    original_x = np.linspace(0.0, 1.0, num=len(contour))
    resampled_x = np.linspace(0.0, 1.0, num=CONTOUR_DIM)

    resampled = np.interp(resampled_x, original_x, contour)

    max_abs = np.max(np.abs(resampled))
    if max_abs > 0:
        resampled = resampled / max_abs

    return resampled


def _summary_stats(intervals: List[int]) -> np.ndarray:

    if not intervals:
        return np.zeros(STATS_DIM, dtype=np.float64)

    arr = np.asarray(intervals, dtype=np.float64)

    mean = arr.mean()
    std = arr.std()
    minimum = arr.min()
    maximum = arr.max()
    num_intervals = float(len(arr))
    positive_ratio = float(np.mean(arr > 0))
    negative_ratio = float(np.mean(arr < 0))

    return np.array(
        [
            mean,
            std,
            minimum,
            maximum,
            num_intervals,
            positive_ratio,
            negative_ratio,
        ],
        dtype=np.float64,
    )


def embed_intervals(intervals: List[int]) -> np.ndarray:
    """
    Convert a normalized interval sequence into a fixed-length,
    L2-normalized embedding vector.
    """

    histogram = _interval_histogram(intervals)
    contour = _pitch_contour(intervals)
    stats = _summary_stats(intervals)

    vector = np.concatenate([histogram, contour, stats]).astype(np.float32)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector
