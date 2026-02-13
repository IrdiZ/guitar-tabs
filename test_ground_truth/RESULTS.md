# Ground Truth Testing Results

## Overview

This document summarizes the accuracy testing of `guitar_tabs.py` pitch detection against synthetic audio with known notes.

## Test Cases

| Test | Description | Notes | Type |
|------|-------------|-------|------|
| e_minor_pentatonic_scale | 6 notes, 600ms spacing | E4, G4, A4, B4, D5, E5 | Monophonic |
| simple_riff | 8 notes, 400ms spacing | E2, G2, A2 pattern | Monophonic |
| pure_tones | 4 pure sine waves | A4, B4, C5, D5 | Monophonic |
| fast_notes | 8 notes, 200ms spacing | E4 to E5 scale | Monophonic |
| wide_range | E2 to E5 octave jumps | Testing range | Monophonic |
| chromatic | 6 semitones E4-A4 | Testing pitch precision | Monophonic |
| e_major_chord | 6 simultaneous notes | E major chord | **Polyphonic** |

## Results Summary

### Monophonic Detection (Optimal Settings)

**Best Parameters:**
```json
{
  "hop_length": 256,
  "pitch_method": "cqt",
  "min_confidence": 0.2
}
```

| Test | Precision | Recall | F1 | Pitch Accuracy |
|------|-----------|--------|-----|----------------|
| wide_range | 85.7% | 100% | **92.3%** | 100% |
| e_minor_pentatonic_scale | 60% | 100% | **75.0%** | 100% |
| pure_tones | 50% | 100% | **66.7%** | 75% |
| chromatic | 50% | 100% | **66.7%** | 83% |
| fast_notes | 60% | 75% | **66.7%** | 50% |
| simple_riff | 44% | 63% | **50.0%** | 100% |

**Overall Monophonic F1: ~70%**

### Polyphonic Detection (Chords)

| Method | Precision | Recall | F1 | Notes |
|--------|-----------|--------|-----|-------|
| CQT (default) | 50% | 17% | **25%** | Only detects strongest note |
| Harmonic | 21% | 83% | **33%** | Many false positives |
| NMF | 100% | 17% | **29%** | Underfitting |

**Key Finding:** Current detection is optimized for monophonic audio. Chord detection requires `--polyphonic` flag but accuracy is limited (~30% F1).

## Observed Issues

### 1. Onset Detection
- Too many false positive onsets (detecting sustain as new notes)
- Fast notes (200ms) occasionally merge or split incorrectly
- Solution: Use `hop_length=256` for better temporal resolution

### 2. Pitch Detection  
- Chromatic passages sometimes off by semitone
- Low notes (E2) occasionally detected as E3 (octave error)
- High notes with strong harmonics can confuse detector

### 3. Polyphonic Limitations
- CQT method: single-note-per-onset architecture
- NMF method: requires parameter tuning per audio
- Harmonic method: high false positive rate

## Recommendations

### For Best Results:
1. **Use `hop_length=256`** for faster passages
2. **Use `min_confidence=0.2-0.3`** to balance precision/recall
3. **For chords:** Consider using external tools (Basic Pitch, etc.)
4. **Preprocess audio:** Use `--preprocessing` to enhance note attacks

### Parameter Guide:
```
Slow passages (>300ms notes):    hop_length=512,  confidence=0.3
Fast passages (<300ms notes):    hop_length=256,  confidence=0.2
Mixed/unknown:                   hop_length=256,  confidence=0.25
Chords:                          Use --polyphonic --poly-method harmonic
```

## Accuracy vs Real Audio

**Important:** These tests use synthetic audio with perfect timing and pitch. Real guitar recordings will have:
- String noise and fret buzz
- Slide/bend artifacts
- Sustain and decay overlaps
- Room acoustics

**Expected real-world accuracy:** ~50-60% F1 for clean guitar solos, lower for distorted or ambient recordings.

## How to Run Tests

```bash
# Generate test audio
python ground_truth_test.py --generate-only

# Run all tests
python ground_truth_test.py

# Parameter tuning
python ground_truth_test.py --tune

# Verbose output
python ground_truth_test.py --verbose
```
