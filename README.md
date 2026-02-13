# 🎸 Guitar Tab Transcription

Audio-to-tablature transcription using multiple approaches.

## Existing Implementation

This directory contains a custom-built guitar tab generator (`guitar_tabs.py`) with:
- **Pitch Detection**: pyin, piptrack, polyphonic NMF
- **Onset Detection**: Spectral flux + complex domain + HFC voting
- **Music Theory**: Key detection, scale filtering, chord recognition
- **Playability**: Fret optimization, impossible transition filtering
- **Export**: ASCII tabs, MusicXML, Guitar Pro

### Usage
```bash
# Basic usage
python guitar_tabs.py audio.mp3 -o output.txt

# With polyphonic detection (chords)
python guitar_tabs.py audio.mp3 --polyphonic

# Export to Guitar Pro
python guitar_tabs.py audio.mp3 -o output.gp5 --format gp5
```

---

## ML-Enhanced Approaches

## Quick Start

### Option 1: Online (No Install)
Just use the web demos:
- **Spotify Basic Pitch**: https://basicpitch.spotify.com
- **Demucs (Guitar Isolation)**: https://huggingface.co/spaces/akhaliq/demucs

### Option 2: Local Installation

```bash
# Use Python 3.10/3.11 (not 3.13)
python3.11 -m venv venv
source venv/bin/activate
pip install basic-pitch demucs pretty_midi

# Transcribe guitar audio to MIDI
python transcribe.py guitar_solo.mp3 -o output/

# For full band recordings, isolate guitar first
python transcribe.py song.mp3 --isolate -o output/
```

## Files

| File | Description |
|------|-------------|
| `RESEARCH.md` | Comprehensive research on AMT approaches |
| `transcribe.py` | Main transcription CLI tool |
| `requirements.txt` | Python dependencies |

## Best Models

| Use Case | Model | Install |
|----------|-------|---------|
| Quick MIDI output | Basic Pitch | `pip install basic-pitch` |
| Guitar isolation | Demucs htdemucs_6s | `pip install demucs` |
| Actual tabs (fret/string) | trimplexx CRNN | See RESEARCH.md |

## Pipeline

```
Full Band Mix → Demucs (isolate guitar) → Basic Pitch → MIDI → ASCII Tab
```

## Links

- [Basic Pitch (Spotify)](https://github.com/spotify/basic-pitch)
- [Demucs (Meta)](https://github.com/facebookresearch/demucs)
- [CRNN Tabs](https://github.com/trimplexx/music-transcription)
- [GuitarSet Dataset](https://guitarset.weebly.com)
