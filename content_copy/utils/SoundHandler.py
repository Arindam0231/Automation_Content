from ffmpeg import FFmpeg
import os
from pydub import AudioSegment
from pydub.utils import make_chunks


def convert_webm_to_wav(input_file, output_file):
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
        ff = FFmpeg()
        ff.input(input_file).output(
            output_file, ar=16000, ac=1, acodec="pcm_s16le"
        ).execute()
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def segment_waves(output_file):
    SEGMENT_FOLDER = "segments"
    os.makedirs(SEGMENT_FOLDER, exist_ok=True)
    base_name = os.path.basename(output_file).split(".")[0]
    for files in os.listdir(SEGMENT_FOLDER):
        if files.startswith(base_name):
            os.remove(os.path.join(SEGMENT_FOLDER, files))
    try:
        ff = FFmpeg()
        ff.input(output_file).output(
            os.path.join(SEGMENT_FOLDER, f"{base_name}_%03d.wav"),
            ar=16000,
            ac=1,
            acodec="pcm_s16le",
            f="segment",
            segment_time=600,
            segment_start_number=1,
        ).execute()
        return True
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return False
