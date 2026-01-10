from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
import base64
from typing import Optional

app = FastAPI(title="Shiftune Audio Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"service": "Shiftune", "status": "running", "version": "2.2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ping")
def ping():
    return {"pong": True}


def analyze_audio(file_path: str) -> dict:
    try:
        import librosa
        import numpy as np
        
        y, sr = librosa.load(file_path, sr=16000, duration=15, mono=True)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo)))
        bpm = max(60, min(200, bpm))
        
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        mood = "Bright" if centroid > 2500 else "Balanced" if centroid > 1500 else "Warm"
        
        rms = float(librosa.feature.rms(y=y).mean())
        energy = "High" if rms > 0.08 else "Medium" if rms > 0.03 else "Low"
        
        return {"bpm": bpm, "mood": mood, "energy": energy}
    except Exception as e:
        print(f"Librosa error: {e}")
        return {"bpm": 120, "mood": "Balanced", "energy": "Medium"}


def generate_name(bpm: int, mood: str, energy: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        slug = f"{mood.lower()}-{energy.lower()}-{bpm}bpm"
        return {"trackName": f"{mood} {energy} {bpm}BPM", "trackSlug": slug}
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Generate ONE creative music track name for:
BPM: {bpm}, Mood: {mood}, Energy: {energy}

Reply ONLY with JSON: {{"trackName": "Name Here", "trackSlug": "name-here"}}
trackSlug = lowercase, hyphens only, no spaces."""
            }],
            max_tokens=50,
            temperature=0.9
        )
        
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        data = json.loads(text)
        return {"trackName": data["trackName"], "trackSlug": data["trackSlug"]}
    except Exception as e:
        print(f"OpenAI error: {e}")
        slug = f"{mood.lower()}-{energy.lower()}-{bpm}bpm"
        return {"trackName": f"{mood} {energy} {bpm}BPM", "trackSlug": slug}


def write_id3_tags(file_path: str, track_name: str, bpm: int, mood: str, energy: str) -> bytes:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TBPM, COMM, ID3NoHeaderError
    
    try:
        try:
            audio = MP3(file_path, ID3=ID3)
        except ID3NoHeaderError:
            audio = MP3(file_path)
            audio.add_tags()
        
        audio.tags.delall("TIT2")
        audio.tags.delall("TPE1")
        audio.tags.delall("TALB")
        audio.tags.delall("TBPM")
        audio.tags.delall("COMM")
        
        audio.tags.add(TIT2(encoding=3, text=track_name))
        audio.tags.add(TPE1(encoding=3, text="Shiftune Studio"))
        audio.tags.add(TALB(encoding=3, text="Shiftune Library"))
        audio.tags.add(TBPM(encoding=3, text=str(bpm)))
        audio.tags.add(COMM(encoding=3, lang='eng', desc='Mood', text=f"{mood}, {energy} energy"))
        
        audio.save()
        
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"ID3 error: {e}")
        with open(file_path, 'rb') as f:
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
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"FLAC error: {e}")
        with open(file_path, 'rb') as f:
            return f.read()


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    skip_duplicate_check: Optional[str] = Form("false"),
    payment_intent_id: Optional[str] = Form(None)
):
    """Process audio: detect BPM, generate name, write ID3 tags, return JSON with base64 file."""
    tmp_path = None
    
    try:
        ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing: {file.filename}")
        
        audio_data = analyze_audio(tmp_path)
        print(f"BPM={audio_data['bpm']}, Mood={audio_data['mood']}, Energy={audio_data['energy']}")
        
        name_data = generate_name(audio_data['bpm'], audio_data['mood'], audio_data['energy'])
        track_name = name_data["trackName"]
        track_slug = name_data["trackSlug"]
        print(f"Generated: {track_name}")
        
        # Write tags
        if ext == '.mp3':
            file_bytes = write_id3_tags(tmp_path, track_name, audio_data['bpm'], audio_data['mood'], audio_data['energy'])
        elif ext == '.flac':
            file_bytes = write_flac_tags(tmp_path, track_name, audio_data['bpm'], audio_data['mood'], audio_data['energy'])
        else:
            with open(tmp_path, 'rb') as f:
                file_bytes = f.read()
        
        # Base64 encode for JSON
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')
        new_filename = f"{track_slug}{ext}"
        
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
            "file_type": "audio/mpeg" if ext == ".mp3" else "audio/flac" if ext == ".flac" else "audio/wav"
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
