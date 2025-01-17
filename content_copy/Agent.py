import os
from langdetect import detect
from utils.TextHandler import chunk_transcription
import os
from mistralai import Mistral
from utils.OutputHandler import process_speech

# Initialize the Mistral API client
api_key = os.getenv("MISTRAL_API_KEY")
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)


def generate_langgraph_speech(transcribed_file):
    assert os.path.exists(
        transcribed_file
    ), f"File not found:x:\Downloads\pexels-anastasiya-gepp-654466-3995920.jpg {transcribed_file}"
    language = detect(transcribed_file)
    raw_speech = chunk_transcription(transcribed_file)


# process_speech("segments\\n434ha4QwU0_speech.txt")
