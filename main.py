from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import numpy as np
from typing import Optional
import json

# Initialize FastAPI
app = FastAPI(title="Shiftune Audio Processor")

# CORS - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load heavy libraries to avoid startup issues
librosa = None
openai_client = None

def get_librosa():
    global librosa
    if librosa is None:
        import librosa as lb
        librosa = lb
    return librosa

def get_openai_client():
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return openai_client

@app.get("/")
def read_root():
    return {
        "service": "Shiftune Audio Processor",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "process": "/process-audio",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


def detect_bpm(audio_file_path: str) -> dict:
    """
    Detect BPM and audio characteristics using librosa.
    Returns actual BPM, mood, and energy values.
    """
    try:
        lb = get_librosa()
        
        # Load audio file
        y, sr = lb.load(audio_file_path, sr=22050, duration=60)  # Limit to 60 seconds for speed
        
        # Detect tempo/BPM
        tempo, beat_frames = lb.beat.beat_track(y=y, sr=sr)
        
        # Handle numpy array vs scalar
        if hasattr(tempo, '__iter__'):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)
        
        # Round BPM to nearest integer
        bpm = round(bpm)
        
        # Analyze spectral characteristics for mood
        spectral_centroid = np.mean(lb.feature.spectral_centroid(y=y, sr=sr))
        
        # Analyze energy/loudness
        rms = np.mean(lb.feature.rms(y=y))
        
        # Determine mood based on spectral centroid
        if spectral_centroid > 3000:
            mood = "Bright"
        elif spectral_centroid > 2000:
            mood = "Balanced"
        else:
            mood = "Warm"
        
        # Determine energy level
        if rms > 0.08:
            energy = "High"
        elif rms > 0.03:
            energy = "Medium"
        else:
            energy = "Low"
        
        return {
            "bpm": bpm,
            "mood": mood,
            "energy": energy,
            "spectral_centroid": float(spectral_centroid),
            "rms": float(rms)
        }
        
    except Exception as e:
        print(f"BPM detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BPM detection failed: {str(e)}")


def generate_creative_name(bpm: int, mood: str, energy: str, original_filename: str = "") -> dict:
    """
    Generate creative track name using OpenAI GPT-4o.
    Returns actual creative names based on audio analysis.
    """
    try:
        client = get_openai_client()
        
        prompt = f"""You are a creative music curator generating unique track names for a music library.

Based on this audio analysis:
- BPM: {bpm}
- Mood: {mood} (Warm = bass-heavy/mellow, Balanced = neutral, Bright = treble-heavy/energetic)
- Energy: {energy}
- Original filename hint: {original_filename}

Generate a creative, evocative track name that:
1. Captures the vibe and energy of the music
2. Is unique and memorable
3. Works for commercial/background music licensing
4. Is 2-5 words long
5. Avoids generic words like "track", "beat", "music"

Respond with ONLY valid JSON in this exact format:
{{"trackName": "Your Creative Title Here", "trackSlug": "your-creative-title-here"}}

The trackSlug must be lowercase, hyphenated, URL-safe (no spaces or special characters).

Examples for different vibes:
- 120 BPM, Bright, High energy → "Neon Rush Hour"
- 85 BPM, Warm, Low energy → "Velvet Midnight Drift"  
- 100 BPM, Balanced, Medium energy → "Urban Coffee Morning"
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative music naming assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.9  # Higher temperature for more creative variety
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        return {
            "trackName": result.get("trackName", "Untitled Track"),
            "trackSlug": result.get("trackSlug", "untitled-track")
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {str(e)}, raw: {result_text}")
        # Fallback: generate from BPM and mood
        fallback_name = f"{mood} {energy} {bpm}"
        fallback_slug = f"{mood.lower()}-{energy.lower()}-{bpm}"
        return {"trackName": fallback_name, "trackSlug": fallback_slug}
        
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Name generation failed: {str(e)}")


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    skip_duplicate_check: Optional[str] = Form("false"),
    payment_intent_id: Optional[str] = Form(None)
):
    """
    Main endpoint to process audio files.
    1. Detect BPM using librosa
    2. Analyze mood/energy
    3. Generate creative name using GPT-4o
    4. Return complete metadata
    """
    
    # Validate file type
    allowed_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Step 1: Detect BPM and analyze audio
        print(f"Processing: {file.filename}")
        print("Step 1: Detecting BPM...")
        bpm_data = detect_bpm(tmp_file_path)
        print(f"  → BPM: {bpm_data['bpm']}, Mood: {bpm_data['mood']}, Energy: {bpm_data['energy']}")
        
        # Step 2: Generate creative name
        print("Step 2: Generating creative name with GPT-4o...")
        name_data = generate_creative_name(
            bpm=bpm_data['bpm'],
            mood=bpm_data['mood'],
            energy=bpm_data['energy'],
            original_filename=file.filename
        )
        print(f"  → Name: {name_data['trackName']}")
        
        # Return complete response
        return {
            "status": "success",
            "message": "Track processed successfully",
            "track_name": name_data['trackName'],
            "track_slug": name_data['trackSlug'],
            "bpm": bpm_data['bpm'],
            "mood": bpm_data['mood'],
            "energy": bpm_data['energy'],
            "original_filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
