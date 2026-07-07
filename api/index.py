import os
import base64
import tempfile
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

# For AIPipe/OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.get("/")
def home():
    return {"message": "Audio Stats API is running"}


@app.post("/")
def process_audio(request: AudioRequest):
    temp_file = None

    try:
        audio_bytes = base64.b64decode(request.audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_file = f.name

        with open(temp_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return {
            "audio_id": request.audio_id,
            "transcription": transcript.text
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)