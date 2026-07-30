"""
Melody extraction.

Responsibility:
    vocals.wav  -->  BasicPitch  -->  Raw Notes

No cleaning or normalization logic lives here.
"""

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def extract_melody(audio_path):

    model_output, midi_data, note_events = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH
    )

    melody = []

    for note in note_events:

        start_time = note[0]
        end_time = note[1]
        pitch = note[2]

        melody.append(
            {
                "pitch": int(pitch),
                "start": float(start_time),
                "end": float(end_time)
            }
        )

    # Sort notes according to time
    melody = sorted(
        melody,
        key=lambda x: x["start"]
    )

    return melody