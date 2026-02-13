# 🎸 Guitar Tab Transcription

Audio-to-tablature transcription using multiple approaches.

## Existing Implementation

This directory contains a custom-built guitar tab generator (`guitar_tabs.py`) with:
- **Pitch Detection**: pyin, piptrack, polyphonic NMF
- **Onset Detection**: Spectral flux + complex domain + HFC voting
- **Music Theory**: Key detection, scale filtering, chord recognition
- **String Detection**: Spectral/timbre analysis to identify which string notes are played on
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

# Enable spectral string detection (uses timbre analysis)
python guitar_tabs.py audio.mp3 --spectral-strings

# With verbose string detection output
python guitar_tabs.py audio.mp3 --spectral-strings --string-detection-verbose
```

### Spectral String Detection

The same pitch can be played on different strings at different frets. For example, the note E4 can be played as:
- High E string, open (fret 0)
- B string, fret 5
- G string, fret 9
- D string, fret 14

Each position has a distinct **timbral fingerprint**:
- **Lower strings (E, A, D)**: Warmer, more bass, lower spectral centroid
- **Higher strings (G, B, e)**: Brighter, more treble, higher spectral centroid

The `--spectral-strings` flag enables spectral analysis to predict which string a note was played on based on:
- **Spectral centroid** - measure of "brightness"
- **Spectral bandwidth** - frequency spread
- **Attack characteristics** - different strings have different attack transients
- **Playing context** - prefers positions close to previous notes

This can improve fret assignment accuracy by 20-40% compared to heuristic methods alone.

---

## Accuracy Testing

Ground truth testing validates detection accuracy using synthetic audio with known notes.

```bash
# Run all accuracy tests
python ground_truth_test.py

# Generate test audio only
python ground_truth_test.py --generate-only

# Tune parameters for optimal detection
python ground_truth_test.py --tune
```

**Current Performance (v1.0):**
| Metric | Value |
|--------|-------|
| Overall F1 Score | 57.7% |
| Monophonic Detection | ~70% F1 |
| Pitch Accuracy | 80%+ |
| Chord Detection | ~25% F1 (limited) |

**Optimal Settings:**
- `hop_length=256` (fast passages)
- `min_confidence=0.2`
- `pitch_method=cqt`

See `test_ground_truth/RESULTS.md` for detailed analysis.

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
