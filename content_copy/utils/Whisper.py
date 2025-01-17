import whisper
from utils.SoundHandler import convert_webm_to_wav, segment_waves
import os

SEGMENT_FOLDER = "segments"


def process_audio(input_file):
    # Convert the .webm file to .wav
    wav_file = os.path.join(
        os.path.dirname(input_file), f"{os.path.basename(input_file).split('.')[0]}.wav"
    )
    conversion_response = convert_webm_to_wav(input_file, wav_file)
    if conversion_response and os.path.exists(wav_file):
        # Transcribe the audio
        transcribed_text = transcribe_audio(wav_file)
        if transcribed_text is not None:
            # os.remove(wav_file)
            return transcribed_text
        else:
            return False
    else:
        return False


def transcribe_helper(audio_file):
    try:
        # Load the Whisper model
        model = whisper.load_model(
            "base"
        )  # You can use 'small', 'medium', 'large' models too
        result = model.transcribe(audio_file)

        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def transcribe_audio(audio_file):
    try:
        base_name = os.path.basename(audio_file).split(".")[0]
        segmented = segment_waves(audio_file)
        transcription_text = ""
        if segmented:
            for files in os.listdir(SEGMENT_FOLDER):
                if files.startswith(base_name):
                    text = transcribe_helper(os.path.join(SEGMENT_FOLDER, files))
                    if text is not None:
                        transcription_text += text

        if os.path.exists(audio_file):
            os.remove(audio_file)
        for files in os.listdir(SEGMENT_FOLDER):
            if files.startswith(base_name):
                os.remove(os.path.join(SEGMENT_FOLDER, files))
        text_path = os.path.join(SEGMENT_FOLDER, f"{base_name}.txt")
        with open(text_path, "w") as file:
            file.write(transcription_text)
        return text_path
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
