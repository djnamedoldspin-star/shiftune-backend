from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import re

app = FastAPI(title="Shiftune API")

# ------------------------------------------------------
# CORS – allow the frontend and local testing to call us
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow any origin (frontend, localhost, etc.)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helper data
# -------------------------
ADJECTIVES = [
    "Midnight", "Neon", "Velvet", "Electric", "Golden",
    "Shadow", "Crystal", "Liquid", "Silent", "Ghost",
    "Solar", "Urban", "Velvet", "Static", "Emerald",
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
    """Turn a title into a URL/filename-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:60]


def generate_title_from_filename(filename: str) -> str:
    """
    OLD: tried to tidy the original filename.
    NEW: always invent a fresh, store-ready title.
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adjective} {noun}"


# -------------------------
# Routes
# -------------------------
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
    - fake BPM + mood (for display)
    """
    original_name = file.filename or "untitled.wav"

    # 1) generate new title (ignores messy original)
    track_name = generate_title_from_filename(original_name)

    # 2) random but reasonable BPM
    bpm = random.randint(80, 140)

    # 3) random mood tag
    mood = random.choice(MOODS)

    # 4) slug for filename, includes BPM so rename is obvious
    base_slug = slugify(track_name)
    track_slug = f"{base_slug}-{bpm}"

    return {
        "status": "success",
        "track_name": track_name,
        "track_slug": track_slug,
        "bpm": float(bpm),
        "mood": mood,
    }


# for local testing (not used by Render directly)
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
