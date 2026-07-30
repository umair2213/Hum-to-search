"""
Vocal separation.

Responsibility:
    Song  -->  Demucs  -->  vocals.wav

No melody logic lives here.
"""

from pathlib import Path

from src.config import SEPARATED_DIR
from src.config import DEMUCS_MODEL


_separator = None


def _get_separator():

    global _separator

    if _separator is None:

        from demucs.api import Separator

        _separator = Separator(model=DEMUCS_MODEL)

    return _separator


def _vocals_path(song_path: Path) -> Path:

    return (
        SEPARATED_DIR
        / DEMUCS_MODEL
        / song_path.stem
        / "vocals.wav"
    )


def separate_vocals(song_path) -> str:
    """
    Run Demucs on ``song_path`` and return the path to vocals.wav.

    If vocals.wav already exists (cache), Demucs is not re-run.
    """

    song_path = Path(song_path)

    vocals_path = _vocals_path(song_path)

    if vocals_path.exists():
        return str(vocals_path)

    from demucs.api import save_audio

    separator = _get_separator()

    _, separated = separator.separate_audio_file(str(song_path))

    vocals_path.parent.mkdir(parents=True, exist_ok=True)

    save_audio(
        separated["vocals"],
        str(vocals_path),
        samplerate=separator.samplerate,
    )

    return str(vocals_path)
