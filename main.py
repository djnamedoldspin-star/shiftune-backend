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
# Helper functions
# -------------------------
ADJECTIVES = [
    "Midnight", "Neon", "Velvet", "Electric", "Golden",
    "Shadow", "Crystal", "Liquid", "Silent", "Ghost"
]

NOUNS = [
    "Transit", "Echoes", "Parade", "Circuit", "Skyline",
    "Mirage", "Boulevard", "Memory", "Pulse", "Pattern"
]

MOODS = [
    "chill", "dark", "uplifting", "moody", "energetic",
    "dreamy", "aggressive", "romantic", "mysterious", "playful"
]


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:60]


def generate_title_from_filename(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    # clean common junk
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"(mix|master|final|v\d+|\d{3}bpm)", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip()

    # if something readable is left, title-case it
    if len(name) > 3:
        base = name.title()
    else:
        base = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"

    return base


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
    - nice track name
    - slug for filename
    - fake BPM + mood (for display)
    """
    original_name = file.filename or "untitled.wav"

    # generate title
    track_name = generate_title_from_filename(original_name)

    # random but reasonable BPM
    bpm = random.randint(80, 140)

    # random mood tag
    mood = random.choice(MOODS)

    # slug for filename
    track_slug = slugify(track_name)

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
