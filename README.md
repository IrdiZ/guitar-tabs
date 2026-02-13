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

## Lead Guitar Technique Detection

Detects advanced techniques used in distorted/high-gain lead guitar:

### Supported Techniques

| Technique | Notation | Description |
|-----------|----------|-------------|
| **Pinch Harmonic** | `7(PH)` | Artificial harmonic (squeal) - thumb touches string while picking |
| **Natural Harmonic** | `<12>` | Harmonic at fret position (5, 7, 12, etc.) |
| **Dive Bomb** | `7~dive` | Whammy bar pitch drop |
| **Whammy Flutter** | `5~flutter` | Rapid pitch oscillation with whammy bar |
| **Vibrato** | `5~` | Finger vibrato |
| **Bend** | `7b9` | Bend from fret 7 to pitch of fret 9 |
| **Hammer-on/Pull-off** | `5h7`, `7p5` | Legato techniques |
| **Slides** | `5/7`, `7\5` | Slide up/down |

### Usage

```bash
# Detect pinch harmonics and squeals
python pinch_harmonic_detector.py audio.mp3 -v

# Output to JSON
python pinch_harmonic_detector.py audio.mp3 -o techniques.json

# With custom sample rate
python pinch_harmonic_detector.py audio.mp3 --sr 44100
```

### Detection Details

**Pinch Harmonics (PH)**:
- Very high spectral centroid (squealing sound)
- Dominant upper harmonics (3rd-5th)
- High-frequency transient at onset
- Characteristic pick attack

**Natural Harmonics (NH)**:
- Very pure tone (single dominant harmonic)
- Matches specific fret positions (5, 7, 12)
- Long sustain, clean decay

**Dive Bombs**:
- Continuous pitch descent
- Minimum 4 semitones drop
- Smooth trajectory (few direction changes)

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

## Attack Transient Pitch Detection

**Critical for distorted guitar**: The attack transient (first 20-50ms) contains the clearest pitch information. After that, distortion muddies the signal.

### Why Attack-Based Detection?

For distorted guitar, the sustain portion is problematic:
- Compression takes time to engage
- Harmonics build up from distortion
- Intermodulation artifacts appear
- The original pitch becomes obscured

The **attack transient** is cleaner because:
1. Initial string excitation before saturation kicks in
2. First few wave cycles are most periodic
3. Amp compression hasn't engaged yet
4. Harmonics haven't built up

### Usage

```bash
# Basic attack-based detection
python attack_transient_pitch.py audio.mp3

# Compare attack vs sustain (demonstrates the difference)
python test_attack_vs_sustain.py audio.mp3

# Full attack-priority ensemble detection
python attack_ensemble_integration.py audio.mp3
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attack_start_ms` | 5 | Skip first 5ms (pick noise) |
| `attack_end_ms` | 50 | Primary analysis window end |
| `use_hps` | True | Harmonic Product Spectrum (best for distortion) |
| `use_yin` | True | YIN algorithm (robust monophonic) |
| `require_agreement` | 2 | Minimum methods that must agree |

### Pitch Detection Methods

The attack analyzer uses multiple methods:

| Method | Strength | Use Case |
|--------|----------|----------|
| **HPS** | Finds fundamental even when harmonics are stronger | Distorted guitar |
| **YIN** | Robust for noisy signals | General monophonic |
| **Autocorrelation** | Classic, fast | Clean transients |
| **CQT** | Good frequency resolution at low frequencies | Bass notes |

### Integration with Ensemble

The `attack_ensemble_integration.py` module combines attack-based detection with sustain validation:

1. **Primary**: Detect pitch from attack (weighted 2.0x)
2. **Validation**: Check sustain for confirmation (weighted 0.5x)
3. **Boost**: If attack and sustain agree, +30% confidence

This approach ensures robust detection while still benefiting from sustain information when available.

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
