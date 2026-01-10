from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import re
import tempfile

import librosa  # BPM detection

app = FastAPI(title="Shiftune API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADJECTIVES = [
    "Midnight", "Neon", "Velvet", "Electric", "Golden",
    "Shadow", "Crystal", "Liquid", "Silent", "Ghost",
    "Solar", "Urban", "Static", "Emerald",
]

NOUNS = [
    "Transit", "Echoes", "Parade", "Circuit", "Skyline",
    "Mirage", "Boulevard", "Memory", "Pulse", "Pattern",
    "Harbor", "Avenue", "Spectrum", "Signal", "Garden",
]

MOODS = [
    "chill", "dark", "uplifting", "moody", "energetic",
    "dreamy", "aggressive", "romantic", "mysterious", "playful",
]


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:60]


def generate_title() -> str:
    return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"


def detect_bpm(tmp_path: str) -> float:
    """
    Try to detect BPM using librosa.
    If anything fails, fall back to a random reasonable BPM.
    """
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        if bpm <= 0:
            raise ValueError("invalid tempo")
        return round(bpm)
    except Exception:
        return random.randint(80, 140)


@app.get("/")
def root():
    return {
        "service": "Shiftune API",
        "message": "Upload an audio file to /process-audio",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Receive one audio file and return:
    - clean, made-up track name
    - slug for filename (includes BPM)
    - detected BPM (or reasonable fallback)
    - mood tag
    """
    original_name = file.filename or "untitled.wav"

    # Save upload to a temp file for librosa
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_name)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        bpm = detect_bpm(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    track_name = generate_title()
    mood = random.choice(MOODS)
    base_slug = slugify(track_name)
    track_slug = f"{base_slug}-{bpm}"

    return {
        "status": "success",
        "track_name": track_name,
        "track_slug": track_slug,
        "bpm": float(bpm),
        "mood": mood,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
