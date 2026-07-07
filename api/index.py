import os
import shutil
import base64
import tempfile
import traceback

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# Add FFmpeg to PATH
# --------------------------------------------------
FFMPEG_DIR = r"C:\Users\WIN 10\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"

os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

print("FFmpeg found:", shutil.which("ffmpeg"))

import whisper

# --------------------------------------------------
# Load Whisper model
# --------------------------------------------------
model = whisper.load_model("base")

app = FastAPI()


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.get("/")
def home():
    return {
        "message": "Audio Stats API is running"
    }


@app.post("/")
def process_audio(request: AudioRequest):
    temp_file = None

    try:
        # Decode Base64
        audio_bytes = base64.b64decode(request.audio_base64)

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_file = f.name

        # Transcribe audio
        result = model.transcribe(temp_file)

        return {
            "audio_id": request.audio_id,
            "transcription": result["text"].strip()
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)