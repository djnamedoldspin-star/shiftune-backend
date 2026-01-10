import os
import re
from typing import Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from slugify import slugify

app = FastAPI()

# Allow local dev + your Render static frontend
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://shiftune-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_title_from_filename(name: str) -> str:
    """
    Take something like:
      'Beat_12_mixdown-FINAL_v7'
    and turn it into:
      'Beat 12 – Mixdown'
    (simple, safe heuristic – you can refine later)
    """

    # Strip common junk tokens
    junk_patterns = [
        r"\bfinal\b",
        r"\bfinals?\b",
        r"\bmix( ?down)?\b",
        r"\bmaster\b",
        r"\bv?\d{1,3}\b",
        r"\bver?sion?\b",
        r"\bedit\b",
    ]
    base = name.lower()
    for pat in junk_patterns:
        base = re.sub(pat, " ", base)

    # Replace separators with spaces
    base = re.sub(r"[_\-]+", " ", base)

    # Collapse whitespace
    base = re.sub(r"\s+", " ", base).strip()

    if not base:
        base = "Untitled Track"

    # Title case but keep short words lowercase
    words = base.split(" ")
    small = {"of", "the", "and", "or", "a", "an", "in", "on"}
    titled = []
    for i, w in enumerate(words):
        if i != 0 and w in small:
            titled.append(w)
        else:
            titled.append(w.capitalize())

    return " ".join(titled)


def make_response_for_file(filename: str) -> Dict:
    """
    Build the JSON payload your frontend expects.
    Currently uses stub BPM + mood so the backend stays light
    and reliable. Later you can plug in real analysis.
    """
    root, _ext = os.path.splitext(filename or "track")
    title = clean_title_from_filename(root)

    # Lightweight stubs – safe values that won't crash
    bpm = 128.0
    mood = "mysterious"

    slug = slugify(title)

    return {
        "status": "success",
        "track_name": title,
        "track_slug": slug,
        "bpm": bpm,
        "mood": mood,
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Shiftune backend alive"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Minimal endpoint:
    - Accepts uploaded file (needed for future heavy analysis)
    - Generates a cleaned title + slug
    - Returns static BPM/mood for now
    """
    # We don't actually need the bytes yet; just touching the
    # UploadFile proves streaming works. Leave this for later:
    # contents = await file.read()

    payload = make_response_for_file(file.filename)
    return payload
