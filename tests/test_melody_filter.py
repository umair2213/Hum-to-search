from src.audio.melody_filter import (
    filter_invalid_notes,
    remove_overlapping_duplicates,
    remove_pitch_jumps,
    remove_harmonic_notes,
    smooth_vibrato,
    clean_notes,
)


def note(pitch, start, end):
    return {"pitch": pitch, "start": start, "end": end}


def test_filter_invalid_notes_removes_out_of_range_pitch():
    notes = [note(20, 0.0, 1.0), note(60, 0.0, 1.0)]
    result = filter_invalid_notes(notes, min_pitch=40, max_pitch=80)
    assert result == [note(60, 0.0, 1.0)]


def test_filter_invalid_notes_removes_short_duration():
    notes = [note(60, 0.0, 0.05), note(60, 1.0, 1.5)]
    result = filter_invalid_notes(notes, min_duration=0.12)
    assert result == [note(60, 1.0, 1.5)]


def test_remove_overlapping_duplicates_keeps_highest_pitch_in_window():
    notes = [note(59, 0.0, 0.2), note(61, 0.05, 0.2), note(58, 0.1, 0.2)]
    result = remove_overlapping_duplicates(notes, window=0.25)
    assert len(result) == 1
    assert result[0]["pitch"] == 61


def test_remove_pitch_jumps_drops_impossible_jump():
    notes = [note(60, 0.0, 1.0), note(90, 1.0, 2.0), note(61, 2.0, 3.0)]
    result = remove_pitch_jumps(notes, max_jump=12)
    pitches = [n["pitch"] for n in result]
    assert 90 not in pitches


def test_remove_harmonic_notes_drops_octave_jump():
    notes = [note(60, 0.0, 1.0), note(72, 1.0, 2.0), note(61, 2.0, 3.0)]
    result = remove_harmonic_notes(notes)
    pitches = [n["pitch"] for n in result]
    assert 72 not in pitches


def test_smooth_vibrato_merges_close_pitches():
    notes = [note(60, 0.0, 1.0), note(61, 1.0, 2.0), note(70, 2.0, 3.0)]
    result = smooth_vibrato(notes, pitch_tolerance=2)
    assert len(result) == 2
    assert result[0]["end"] == 2.0


def test_clean_notes_full_pipeline_runs_end_to_end():
    notes = [
        note(20, 0.0, 0.05),       # invalid pitch + too short -> removed
        note(60, 0.0, 1.0),
        note(60, 0.05, 1.0),       # duplicate within window
        note(72, 1.0, 2.0),        # harmonic jump -> removed
        note(61, 2.0, 3.0),
    ]
    result = clean_notes(notes)
    assert all(40 <= n["pitch"] <= 80 for n in result)
    assert len(result) >= 1


def test_clean_notes_empty_input():
    assert clean_notes([]) == []
