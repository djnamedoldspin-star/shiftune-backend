from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
import base64
import logging
import re
import time
from collections import defaultdict, deque
from typing import Optional, Dict, Any
import mimetypes

APP_VERSION = "2.2.1-public"
APP_NAME = "Shiftune Audio Processor"

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(APP_NAME)

# ---------------------------
# Config (env)
# ---------------------------
DEFAULT_ALLOWED_ORIGINS = [
    "https://shiftune-frontend.onrender.com",
    "https://classy-truffle-46355a.netlify.app",
]
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",") if o.strip()
]

MAX_FILE_MB = int(os.getenv("SHIFTUNE_MAX_FILE_MB", "25"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Simple, best-effort rate limiting (per instance)
RATE_WINDOW_SEC = int(os.getenv("SHIFTUNE_RATE_WINDOW_SEC", "60"))
RATE_MAX_REQ = int(os.getenv("SHIFTUNE_RATE_MAX_REQ", "40"))  # per IP per window

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "20"))

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Security headers
# ---------------------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # Adjust CSP if you embed this behind a web app; kept minimal server-side
    return response


# ---------------------------
# Rate limiter (in-memory)
# ---------------------------
_requests: Dict[str, deque] = defaultdict(deque)

def _client_ip(request: Request) -> str:
    # If you're behind Render, X-Forwarded-For is typically set.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _rate_limit_or_429(ip: str):
    now = time.time()
    q = _requests[ip]
    # prune
    while q and (now - q[0]) > RATE_WINDOW_SEC:
        q.popleft()
    if len(q) >= RATE_MAX_REQ:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")
    q.append(now)


# ---------------------------
# Helpers: sanitize title/slug + robust JSON extraction
# ---------------------------
_slug_re = re.compile(r"[^a-z0-9]+")
_ws_re = re.compile(r"\s+")

def sanitize_title(title: str, max_len: int = 80) -> str:
    title = (title or "").strip()
    title = title.replace("\u0000", "")
    title = _ws_re.sub(" ", title)
    # remove risky filesystem characters
    title = re.sub(r'[\\/:*?"<>|]+', "", title)
    return title[:max_len].strip() or "Untitled"

def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").strip().lower()
    text = text.replace("&", " and ")
    text = _ws_re.sub(" ", text)
    text = _slug_re.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return (text[:max_len].strip("-")) or "untitled"

def extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a string that might include extra text or code fences.
    This is intentionally defensive because LLM outputs can be messy.
    """
    if not s:
        return None

    s = s.strip()

    # Remove common code fences
    if "```" in s:
        # keep the largest fenced block if present
        parts = s.split("```")
        # heuristic: choose the longest non-empty part
        candidates = [p for p in parts if p.strip()]
        if candidates:
            s = max(candidates, key=lambda x: len(x)).strip()
        # remove a leading language tag like "json"
        s = re.sub(r"^\s*json\s*\n", "", s, flags=re.IGNORECASE).strip()

    # Find first balanced {...}
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start:i+1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


# ---------------------------
# Audio analysis
# ---------------------------
def analyze_audio(file_path: str) -> dict:
    """
    Best-effort analysis. If it fails, we return safe defaults.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(file_path, sr=16000, duration=15, mono=True)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        try:
            # librosa can return ndarray or scalar
            tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
        except Exception:
            tempo_val = float(tempo)

        bpm = int(round(tempo_val))
        bpm = max(60, min(200, bpm))

        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        mood = "Bright" if centroid > 2500 else "Balanced" if centroid > 1500 else "Warm"

        rms = float(librosa.feature.rms(y=y).mean())
        energy = "High" if rms > 0.08 else "Medium" if rms > 0.03 else "Low"

        return {"bpm": bpm, "mood": mood, "energy": energy}
    except Exception as e:
        logger.warning("Audio analysis failed; using defaults. Error=%s", e)
        return {"bpm": 120, "mood": "Balanced", "energy": "Medium"}


# ---------------------------
# OpenAI naming
# ---------------------------
def fallback_name(bpm: int, mood: str, energy: str) -> dict:
    base = f"{mood} {energy} {bpm}BPM"
    slug = slugify(f"{mood}-{energy}-{bpm}bpm")
    return {"trackName": base, "trackSlug": slug}

def generate_name(bpm: int, mood: str, energy: str) -> dict:
    """
    Returns: {"trackName": "...", "trackSlug": "..."}.
    If OpenAI is unavailable or output is invalid, returns fallback.
    """
    if not OPENAI_API_KEY:
        return fallback_name(bpm, mood, energy)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SEC)

        system = (
            "You are Shiftune Studio, a DJ-ready track renaming engine.\n"
            "Rules:\n"
            "1) Output MUST be valid JSON only (no markdown, no commentary).\n"
            "2) trackName: 2–5 words, memorable, not cheesy, no profanity.\n"
            "3) trackSlug: lowercase, hyphens only, derived from trackName.\n"
            "4) Avoid generic phrases like 'Balanced Medium 120BPM'.\n"
            "5) Do not include quotes inside the values.\n"
        )

        user = (
            f"Audio traits:\n"
            f"- BPM: {bpm}\n"
            f"- Mood: {mood}\n"
            f"- Energy: {energy}\n\n"
            "Return exactly:\n"
            "{\"trackName\":\"...\",\"trackSlug\":\"...\"}\n\n"
            "Examples (style only):\n"
            "{\"trackName\":\"Neon Drift\",\"trackSlug\":\"neon-drift\"}\n"
            "{\"trackName\":\"Velvet Circuit\",\"trackSlug\":\"velvet-circuit\"}\n"
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=60,
            temperature=0.9,
        )

        raw = (response.choices[0].message.content or "").strip()
        data = extract_first_json_object(raw)

        if not data or "trackName" not in data:
            logger.warning("OpenAI returned non-JSON or missing fields. Raw=%r", raw[:300])
            return fallback_name(bpm, mood, energy)

        track_name = sanitize_title(str(data.get("trackName", "")))
        track_slug = slugify(str(data.get("trackSlug", "")) or track_name)

        # Basic safety: ensure slug isn't empty, ensure name isn't empty
        if not track_slug or not track_name:
            return fallback_name(bpm, mood, energy)

        return {"trackName": track_name, "trackSlug": track_slug}

    except Exception as e:
        logger.warning("OpenAI naming failed; using fallback. Error=%s", e)
        return fallback_name(bpm, mood, energy)


# ---------------------------
# Tagging (MP3 / FLAC)
# ---------------------------
def write_id3_tags(file_path: str, track_name: str, bpm: int, mood: str, energy: str) -> bytes:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TBPM, COMM, ID3NoHeaderError

    try:
        try:
            audio = MP3(file_path, ID3=ID3)
        except ID3NoHeaderError:
            audio = MP3(file_path)
            audio.add_tags()

        audio.tags.add(TIT2(encoding=3, text=track_name))
        audio.tags.add(TPE1(encoding=3, text="Shiftune Studio"))
        audio.tags.add(TALB(encoding=3, text="Shiftune Library"))
        audio.tags.add(TBPM(encoding=3, text=str(bpm)))
        audio.tags.add(COMM(encoding=3, lang="eng", desc="Comment", text=f"{mood}, {energy} energy"))

        audio.save()

        with open(file_path, "rb") as f:
            return f.read()

    except Exception as e:
        logger.warning("ID3 write failed; returning raw bytes. Error=%s", e)
        with open(file_path, "rb") as f:
            return f.read()


def write_flac_tags(file_path: str, track_name: str, bpm: int, mood: str, energy: str) -> bytes:
    from mutagen.flac import FLAC
    try:
        audio = FLAC(file_path)
        audio["title"] = track_name
        audio["artist"] = "Shiftune Studio"
        audio["album"] = "Shiftune Library"
        audio["bpm"] = str(bpm)
        audio["comment"] = f"{mood}, {energy} energy"
        audio.save()
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("FLAC write failed; returning raw bytes. Error=%s", e)
        with open(file_path, "rb") as f:
            return f.read()


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "openai_enabled": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "max_file_mb": MAX_FILE_MB,
        "allowed_origins": ALLOWED_ORIGINS,
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": APP_VERSION}

@app.get("/ping")
def ping():
    return {"pong": True, "version": APP_VERSION}


@app.post("/process-audio")
async def process_audio(request: Request, file: UploadFile = File(...)):
    ip = _client_ip(request)
    _rate_limit_or_429(ip)

    tmp_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename.")

        ext = os.path.splitext(file.filename)[1].lower().strip()
        if not ext:
            ext = ".mp3"

        # Read bytes (enforces an upper bound)
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_MB} MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio_data = analyze_audio(tmp_path)
        name_data = generate_name(audio_data["bpm"], audio_data["mood"], audio_data["energy"])

        track_name = name_data["trackName"]
        track_slug = name_data["trackSlug"]

        new_filename = f"{track_slug}{ext}"

        # Tagging only for MP3/FLAC
        if ext == ".mp3":
            out_bytes = write_id3_tags(tmp_path, track_name, audio_data["bpm"], audio_data["mood"], audio_data["energy"])
        elif ext == ".flac":
            out_bytes = write_flac_tags(tmp_path, track_name, audio_data["bpm"], audio_data["mood"], audio_data["energy"])
        else:
            # No tagging support yet; still return bytes (with new filename)
            with open(tmp_path, "rb") as f:
                out_bytes = f.read()

        file_base64 = base64.b64encode(out_bytes).decode("utf-8")

        file_type = mimetypes.types_map.get(ext, None) or file.content_type or "application/octet-stream"

        return {
            "status": "success",
            "track_name": track_name,
            "track_slug": track_slug,
            "bpm": audio_data["bpm"],
            "mood": audio_data["mood"],
            "energy": audio_data["energy"],
            "original_filename": file.filename,
            "new_filename": new_filename,
            "file_data": file_base64,
            "file_type": file_type,
            "openai_used": bool(OPENAI_API_KEY),
            "version": APP_VERSION,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /process-audio. ip=%s filename=%s", ip, getattr(file, "filename", None))
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
