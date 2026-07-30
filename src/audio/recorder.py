import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 24000
DURATION = 20  # seconds


def record_audio(output_file="query.wav"):
    print("Recording...")

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        output_file,
        recording,
        SAMPLE_RATE
    )

    print("Recording Saved")

    return output_file