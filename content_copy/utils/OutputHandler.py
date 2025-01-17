import cv2
import mediapipe as mp
from gtts import gTTS
import os


SEGMENT_FOLDER = "segments"


# Generate TTS using gTTS
def generate_tts_gtts(text, output_file):
    tts = gTTS(text)
    tts.save(output_file)


def run_wav2lip(input_video, input_audio, output_video):
    os.system(
        f"python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip.pth "
        f"--face {input_video} --audio {input_audio} --outfile {output_video}"
    )
    print(f"Lip-synced video saved: {output_video}")


def process_speech(speech_file):
    assert os.path.exists(speech_file), "Speech file does not exist"
    with open(speech_file, "r") as f:
        text = f.read()
    base_name = os.path.basename(speech_file).split(".")[0]
    audio_file = os.path.join(SEGMENT_FOLDER, f"{base_name}_audio.mp3")
    video_file = os.path.join(SEGMENT_FOLDER, f"{base_name}_video.avi")

    # Step 1: Generate TTS
    generate_tts_gtts(text, audio_file)

    # Step 2: Generate Avatar Video (commented out for simplicity)
    # generate_avatar_video(audio_output, video_output)
    # run_wav2lip(input_avatar_video, tts_audio, output_synced_video)

    print(f"TTS audio generated: {audio_file}")
    # print(f"Avatar video generated: {video_file}")
