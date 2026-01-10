from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
from typing import Optional

app = FastAPI(title="Shiftune Audio Processor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"service": "Shiftune", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


def analyze_audio(file_path: str) -> dict:
    """Analyze audio file for BPM and characteristics."""
    try:
        import librosa
        import numpy as np
        
        # Load audio (limit to 30 seconds for speed/memory)
        y, sr = librosa.load(file_path, sr=22050, duration=30, mono=True)
        
        # Get tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo)))
        
        # Clamp BPM to reasonable range
        bpm = max(60, min(200, bpm))
        
        # Get mood from spectral centroid
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        mood = "Bright" if centroid > 3000 else "Balanced" if centroid > 2000 else "Warm"
        
        # Get energy from RMS
        rms = float(librosa.feature.rms(y=y).mean())
        energy = "High" if rms > 0.08 else "Medium" if rms > 0.03 else "Low"
        
        return {"bpm": bpm, "mood": mood, "energy": energy}
        
    except Exception as e:
        print(f"Librosa error: {e}")
        # Fallback: return reasonable defaults
        return {"bpm": 120, "mood": "Balanced", "energy": "Medium"}


def generate_name(bpm: int, mood: str, energy: str) -> dict:
    """Generate creative name using OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("WARNING: No OPENAI_API_KEY set!")
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

Reply with ONLY JSON: {{"trackName": "Name Here", "trackSlug": "name-here"}}
trackSlug must be lowercase with hyphens, no spaces."""
            }],
            max_tokens=50,
            temperature=0.9
        )
        
        text = response.choices[0].message.content.strip()
        # Clean markdown
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        data = json.loads(text)
        return {"trackName": data["trackName"], "trackSlug": data["trackSlug"]}
        
    except Exception as e:
        print(f"OpenAI error: {e}")
        slug = f"{mood.lower()}-{energy.lower()}-{bpm}bpm"
        return {"trackName": f"{mood} {energy} {bpm}BPM", "trackSlug": slug}


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    skip_duplicate_check: Optional[str] = Form("false"),
    payment_intent_id: Optional[str] = Form(None)
):
    """Process audio file - detect BPM and generate creative name."""
    
    tmp_path = None
    
    try:
        # Get file extension
        ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing: {file.filename} ({len(content)} bytes)")
        
        # Analyze audio
        audio_data = analyze_audio(tmp_path)
        print(f"Analysis: BPM={audio_data['bpm']}, Mood={audio_data['mood']}, Energy={audio_data['energy']}")
        
        # Generate name
        name_data = generate_name(audio_data['bpm'], audio_data['mood'], audio_data['energy'])
        print(f"Generated: {name_data['trackName']}")
        
        return {
            "status": "success",
            "track_name": name_data["trackName"],
            "track_slug": name_data["trackSlug"],
            "bpm": audio_data["bpm"],
            "mood": audio_data["mood"],
            "energy": audio_data["energy"],
            "original_filename": file.filename
        }
        
    except Exception as e:
        print(f"PROCESS ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
