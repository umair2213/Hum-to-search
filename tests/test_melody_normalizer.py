from src.audio.melody_normalizer import notes_to_intervals


def note(pitch):
    return {"pitch": pitch, "start": 0.0, "end": 1.0}


def test_notes_to_intervals_basic():
    notes = [note(60), note(62), note(64), note(62)]
    assert notes_to_intervals(notes) == [2, 2, -2]


def test_notes_to_intervals_transposition_invariant():
    song = [note(60), note(62), note(64), note(62)]
    hum = [note(48), note(50), note(52), note(50)]
    assert notes_to_intervals(song) == notes_to_intervals(hum)


def test_notes_to_intervals_empty_and_single():
    assert notes_to_intervals([]) == []
    assert notes_to_intervals([note(60)]) == []
