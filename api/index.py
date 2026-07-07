import os
import base64
import tempfile
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
import whisper

app = FastAPI()

# Load Whisper model once when the server starts
model = whisper.load_model("tiny")


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
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_file = f.name

        # Transcribe using Whisper
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