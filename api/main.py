"""
FastAPI Backend for Guitar Tab Generator
"""

import os
import sys
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import guitar_tabs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar_tabs import (
    detect_notes_from_audio,
    notes_to_tabs,
    format_ascii_tab,
    download_youtube_audio,
    is_youtube_url,
)

app = FastAPI(
    title="Guitar Tab Generator API",
    description="Generate guitar tabs from audio files or YouTube URLs",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoteResponse(BaseModel):
    """A detected note"""
    midi: int
    name: str
    start_time: float
    duration: float
    confidence: float


class TabNoteResponse(BaseModel):
    """A note on the fretboard"""
    string: int
    string_name: str
    fret: int
    start_time: float
    duration: float


class GenerateResponse(BaseModel):
    """Response from /generate endpoint"""
    success: bool
    notes: list[NoteResponse]
    tab_notes: list[TabNoteResponse]
    ascii_tab: str
    note_count: int
    message: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "guitar-tab-generator"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_tabs(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    confidence: float = Form(0.3),
):
    """
    Generate guitar tabs from an audio file or YouTube URL.
    
    - **file**: Audio file upload (wav, mp3, etc.)
    - **youtube_url**: YouTube URL to extract audio from
    - **confidence**: Minimum confidence threshold (0-1), default 0.3
    """
    if not file and not youtube_url:
        raise HTTPException(
            status_code=400,
            detail="Either file or youtube_url must be provided"
        )
    
    if file and youtube_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either file or youtube_url, not both"
        )
    
    audio_path = None
    cleanup_file = False
    
    try:
        # Handle YouTube URL
        if youtube_url:
            if not is_youtube_url(youtube_url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid YouTube URL"
                )
            audio_path = download_youtube_audio(youtube_url)
            cleanup_file = True
        
        # Handle file upload
        elif file:
            # Save uploaded file to temp location
            suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                audio_path = tmp.name
            cleanup_file = True
        
        # Detect notes from audio
        notes = detect_notes_from_audio(
            audio_path,
            confidence_threshold=confidence
        )
        
        if not notes:
            return GenerateResponse(
                success=True,
                notes=[],
                tab_notes=[],
                ascii_tab="No notes detected! Try lowering confidence threshold.",
                note_count=0,
                message="No notes detected in audio"
            )
        
        # Convert to tab notes
        tab_notes = notes_to_tabs(notes)
        
        # Format ASCII tab
        ascii_tab = format_ascii_tab(tab_notes)
        
        # String names for response
        STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
        
        # Build response
        return GenerateResponse(
            success=True,
            notes=[
                NoteResponse(
                    midi=n.midi,
                    name=n.name,
                    start_time=n.start_time,
                    duration=n.duration,
                    confidence=n.confidence
                )
                for n in notes
            ],
            tab_notes=[
                TabNoteResponse(
                    string=tn.string,
                    string_name=STRING_NAMES[tn.string],
                    fret=tn.fret,
                    start_time=tn.start_time,
                    duration=tn.duration
                )
                for tn in tab_notes
            ],
            ascii_tab=ascii_tab,
            note_count=len(notes),
            message=f"Successfully detected {len(notes)} notes"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Cleanup temp file
        if cleanup_file and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/health")
async def health():
    """Health check for load balancers"""
    return {"status": "healthy"}
