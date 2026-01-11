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

APP_VERSION = "2.2.2-public"
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
# Naming / novelty controls
SHIFTUNE_NAME_MAX_ATTEMPTS = int(os.getenv("SHIFTUNE_NAME_MAX_ATTEMPTS", "6"))
SHIFTUNE_RECENT_SLUG_CACHE = int(os.getenv("SHIFTUNE_RECENT_SLUG_CACHE", "200"))

# Recent de-dup (in-memory; per instance)
RECENT_SLUGS = deque()
RECENT_SLUGS_SET = set()

SYSTEM_PROMPT = "You generate short, marketable track titles for instrumental beats.\n\nOUTPUT RULES (hard):\n- Return ONLY valid JSON (no markdown, no commentary).\n- Keys exactly: trackName, trackSlug, oneLiner\n- trackName: 2\u20135 words, Title Case, no quotes, no emojis.\n- trackSlug: lowercase kebab-case derived from trackName; ASCII only; 3\u201345 chars.\n- oneLiner: 8\u201316 words, plain English, no hashtags, no quotes.\n\nNOVELTY RULES (hard):\n- You MUST include EXACTLY ONE of the provided VIBE words in trackName (spelled exactly).\n- You MUST include EXACTLY ONE of the provided THEME words in trackName (spelled exactly).\n- You MUST NOT reuse or closely imitate any \u201cavoidSlugs\u201d (treat as blocked).\n- Do NOT use generic filler titles: \u201cExperimental\u201d, \u201cViral\u201d, \u201cUnpredictable\u201d, \u201cHook\u201d, \u201cType Beat\u201d, \u201cBeat\u201d, \u201cInstrumental\u201d, \u201cVolume\u201d, \u201cFinal\u201d, \u201cMixdown\u201d.\n- Do NOT include BPM numbers in trackName.\n\nSTYLE:\n- Sound: modern, DJ-ready, clean, premium, slightly provocative but not explicit.\n- Prefer concrete imagery + motion verbs + sleek nouns.\n- Avoid clich\u00e9 combos (Neon/Velvet/Midnight/Glow/Night) unless forced by VIBE/THEME lists.\n"

def build_user_prompt(bpm: int, mood: str, energy: str, vibe_words: list, theme_words: list, avoid_slugs: list) -> str:
    vibe_csv = ", ".join(vibe_words) if vibe_words else ""
    theme_csv = ", ".join(theme_words) if theme_words else ""
    avoid_csv = ", ".join(avoid_slugs) if avoid_slugs else "none"

    return (
        "Create a unique track title.\n\n"
        f"Audio traits:\n- bpm: {bpm}\n- mood: {mood}\n- energy: {energy}\n\n"
        "Use exactly one VIBE word and exactly one THEME word in the trackName:\n"
        f"VIBE words: {vibe_csv}\n"
        f"THEME words: {theme_csv}\n\n"
        "Avoid these slugs (blocked — do not match, rhyme with, or resemble):\n"
        f"{avoid_csv}\n\n"
        "Return JSON only with:\n"
        "- trackName (2–5 words, Title Case)\n"
        "- trackSlug (kebab-case)\n"
        "- oneLiner (8–16 words)\n"
    ).format(bpm=bpm, mood=mood, energy=energy, vibe_csv=vibe_csv, theme_csv=theme_csv, avoid_csv=avoid_csv)


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
    """
    Randomized fallback naming (used when OpenAI is unavailable or fails).
    Keeps results DJ-ready and avoids including BPM in the title.
    """
    import random

    adjs = [
        "Crisp", "Sleek", "Lush", "Clean", "Tight", "Fluid", "Punchy", "Airy", "Warm", "Icy",
        "Hush", "Rogue", "Sharp", "Soft", "Electric", "Analog", "Chrome", "Satin", "Obsidian", "Cobalt",
        "Amber", "Prism", "Quartz", "Velour", "Ghosted", "Static", "Feral", "Vivid", "Noir", "Kinetic",
    ]
    nouns = [
        "Crossfade", "Backspin", "Slipstream", "Overdrive", "Afterglow", "Signal", "Blueprint", "Relay",
        "Waypoint", "Orbit", "Horizon", "Mirage", "Cutscene", "Undertow", "Headroom", "Tension",
        "Friction", "Catalyst", "Second Wind", "Skylines", "Voltage", "Drift", "Surge", "Glide",
    ]

    mood_t = sanitize_title(str(mood).strip()) or "Balanced"
    energy_t = sanitize_title(str(energy).strip()) or "Medium"
    adj = random.choice(adjs)
    noun = random.choice(nouns)

    templates = [
        f"{mood_t} {adj} {noun}",
        f"{adj} {noun} {energy_t}",
        f"{mood_t} {noun}",
        f"{adj} {mood_t} {noun}",
    ]

    track_name = sanitize_title(random.choice(templates)).strip()
    # Enforce 2–5 words
    words = track_name.split()
    if len(words) < 2:
        words = [adj, noun]
    track_name = " ".join(words[:5])

    track_slug = slugify(track_name)
    return {"trackName": track_name, "trackSlug": track_slug}


def generate_name(bpm: int, mood: str, energy: str) -> dict:
    """
    Returns: {"trackName": "...", "trackSlug": "..."}.
    Uses OpenAI for naming when available; otherwise uses a randomized fallback.
    Enforces local de-duplication against recently generated slugs.
    """
    import random
    import json as _json

    # --- Random vocab pools (server-side) ---
    vibe_pool = [
        "Crisp", "Sleek", "Silk", "Grit", "Lush", "Clean", "Tight", "Fluid", "Punchy", "Airy",
        "Warm", "Icy", "Hush", "Rogue", "Sharp", "Soft", "Electric", "Analog", "Chrome", "Satin",
        "Shadow", "Solar", "Lunar", "Quartz", "Ivory", "Obsidian", "Cobalt", "Amber", "Velour", "Prism",
        "Motion", "Drift", "Surge", "Glide", "Skate", "Pulse", "Snap", "Sway", "Bloom", "Vortex",
    ]
    theme_pool = [
        "Runway", "Undertow", "Afterglow", "Crossfade", "Backspin", "Switchback", "Slipstream", "Overdrive",
        "Rooftops", "Sidequest", "Signal", "Blueprint", "Arcade", "Late Train", "Glass City", "Cutscene",
        "Mirage", "Airstream", "Longform", "Shortcut", "Nightmarket", "Driveway", "Orbit", "Horizon",
        "Echo", "Relay", "Gravity", "Skylines", "Voltage", "Second Wind", "Headroom", "Tension",
        "Friction", "Catalyst", "Clutch", "Gears", "Latitude", "Ritual", "Waypoint", "Trim",
    ]

    # Sample lists (forces divergence per request)
    vibe_words = random.sample(vibe_pool, k=min(7, len(vibe_pool)))
    theme_words = random.sample(theme_pool, k=min(7, len(theme_pool)))

    # Build avoid list from recent cache (defined at module level)
    try:
        avoid_slugs = list(RECENT_SLUGS)[-25:]
    except Exception:
        avoid_slugs = []

    # If OpenAI is missing, randomized fallback (NOT deterministic)
    if not OPENAI_API_KEY:
        return fallback_name(bpm, mood, energy)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SEC)

        # Retry a few times if we collide with recent slugs
        attempts = max(1, SHIFTUNE_NAME_MAX_ATTEMPTS)

        for _ in range(attempts):
            user_prompt = build_user_prompt(
                bpm=bpm,
                mood=mood,
                energy=energy,
                vibe_words=vibe_words,
                theme_words=theme_words,
                avoid_slugs=avoid_slugs,
            )

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=140,
                temperature=1.05,
                top_p=0.92,
                presence_penalty=0.6,
                frequency_penalty=0.4,
            )

            raw = (response.choices[0].message.content or "").strip()

            # JSON mode should already be clean JSON, but keep defensive parsing
            data = None
            try:
                data = _json.loads(raw)
            except Exception:
                data = extract_first_json_object(raw)

            if not isinstance(data, dict):
                continue

            track_name = sanitize_title(str(data.get("trackName", "")).strip())
            # Prefer provided slug, else derive from track_name
            track_slug = slugify(str(data.get("trackSlug", "")).strip() or track_name)

            if not track_name or not track_slug:
                continue

            # Local de-dup guard
            if track_slug in RECENT_SLUGS_SET:
                # Add to avoid list and retry
                avoid_slugs = (avoid_slugs + [track_slug])[-50:]
                continue

            # Store in recent cache
            RECENT_SLUGS.append(track_slug)
            RECENT_SLUGS_SET.add(track_slug)
            # Trim cache
            while len(RECENT_SLUGS) > SHIFTUNE_RECENT_SLUG_CACHE:
                old = RECENT_SLUGS.popleft()
                RECENT_SLUGS_SET.discard(old)

            return {"trackName": track_name, "trackSlug": track_slug}

        # If we exhausted attempts, fall back
        return fallback_name(bpm, mood, energy)

    except Exception as e:
        logger.warning("OpenAI naming failed; using fallback. Error=%s", e)
        return fallback_name(bpm, mood, energy)
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
