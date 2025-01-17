from dotenv import load_dotenv

env_loaded = load_dotenv()

from flask import Flask, render_template, request, send_file
import yt_dlp
import os
from utils.Whisper import process_audio
from Agent import generate_langgraph_speech

app = Flask(__name__)

# Configure folder for saving files
UPLOAD_FOLDER = "downloads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download", methods=["POST"])
def download():
    url = request.form["url"]
    if not url:
        return "No URL provided", 400

    # Set options for yt-dlp (downloading audio only)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{UPLOAD_FOLDER}/%(id)s.%(ext)s",
    }

    # Log to check the process
    print(f"Starting download for URL: {url}")

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file = os.path.join(
            UPLOAD_FOLDER, f"{info_dict['id']}.webm"
        )  # Extracted audio file

    if os.path.exists(audio_file):
        transcribed_file = process_audio(audio_file)
        print(transcribed_file)
        # generate_langgraph_speech(transcribed_file)
        print(f"Download complete: {audio_file}")
        return render_template("download.html", audio=audio_file)
    else:
        print(f"Failed to download audio: {audio_file}")
        return "Failed to download audio", 500


if __name__ == "__main__":
    app.run(debug=True)
