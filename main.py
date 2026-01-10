from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import hashlib

app = FastAPI(
    title="Shiftune API",
    description="AI-style audio file renamer backend for Shiftune Studio",
    version="1.0.0",
)

# CORS – wide open so your static site can call it from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later to your own domains
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessResponse(BaseModel):
    status: str
    track_name: str
    track_slug: str
    bpm: float
    mood: str


@app.get("/")
def root():
    return {
        "service": "Shiftune API",
        "message": "Upload an audio file to /process-audio",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def make_slug(name: str) -> str:
    # remove extension
    base = name.rsplit(".", 1)[0]
    # lowercase, keep letters/digits/spaces/-
    base = base.lower()
    base = re.sub(r"[^a-z0-9\s\-]", "", base)
    base = re.sub(r"\s+", "-", base).strip("-")
    if not base:
        base = "shiftune-track"
    return base


def make_title_from_slug(slug: str) -> str:
    parts = slug.replace("-", " ").split()
    return " ".join(w.capitalize() for w in parts) or "Shiftune Track"


def deterministic_int(seed: str, minimum: int, maximum: int) -> int:
    """Create a stable pseudo-random int from a string."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    num = int(h[:8], 16)
    span = maximum - minimum + 1
    return minimum + (num % span)


def guess_mood(name: str) -> str:
    name_lower = name.lower()

    keyword_map = {
        "chill": "Chill Lounge",
        "lofi": "Lo-Fi Study",
        "lo-fi": "Lo-Fi Study",
        "sad": "Late Night Melancholy",
        "happy": "Sunrise Uplift",
        "club": "Peak-Time Club",
        "deep": "Deep House Drift",
        "trap": "Trap Gravity",
        "boom": "Boom-Bap Cipher",
        "bap": "Boom-Bap Cipher",
        "afro": "Afrobeats Vibe",
        "afrobeats": "Afrobeats Vibe",
        "drill": "UK Drill Tension",
        "jazz": "Smoky Jazz Lounge",
        "soul": "Neo-Soul Glow",
        "house": "House Groove Engine",
        "techno": "Warehouse Techno",
        "ambient": "Ambient Dreamscape",
        "piano": "Piano Reflections",
        "guitar": "Guitar Reverie",
    }

    for key, label in keyword_map.items():
        if key in name_lower:
            return label

    moods = [
        "Cinematic Pulse",
        "After-Hours Neon",
        "Midnight Cruise",
        "City Skyline Drift",
        "Late Bus Reflections",
        "Underground Cipher",
        "Golden Hour Glow",
        "Rainy Window Study",
        "Basement Session",
        "Soft Focus Romance",
    ]
    idx = deterministic_int(name_lower or "shiftune", 0, len(moods) - 1)
    return moods[idx]


@app.post(
    "/process-audio",
    response_model=ProcessResponse,
    summary="Generate a clean name, slug, BPM, and mood for an uploaded audio file.",
)
async def process_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    allowed_exts = {".mp3", ".wav", ".flac", ".aiff", ".m4a", ".ogg"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use: {', '.join(sorted(allowed_exts))}",
        )

    original_name = file.filename

    # We don't actually need the binary data to generate names.
    # But if you want to inspect it later, you could read:
    # contents = await file.read()

    slug = make_slug(original_name)
    track_name = make_title_from_slug(slug)

    # Stable "fake BPM" based on filename – looks real, deterministic.
    bpm = float(deterministic_int(original_name, 85, 138))

    mood = guess_mood(original_name)

    return ProcessResponse(
        status="success",
        track_name=track_name,
        track_slug=slug,
        bpm=bpm,
        mood=mood,
    )


# For local testing:
# uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
