# Guitar Tab Transcription Research

## 🎯 Executive Summary

After comprehensive research, the **best open-source approach** for guitar tablature transcription combines:

1. **Source Separation** (Optional): Use `demucs htdemucs_6s` to isolate guitar from full mixes
2. **Primary Transcription**: `spotify/basic-pitch` for robust polyphonic AMT → MIDI
3. **Guitar-Specific Tablature**: `trimplexx/music-transcription` CRNN (0.87 F1 on GuitarSet)

---

## 📊 State of the Art Comparison

| Model | Type | F1 Score | Outputs | Pre-trained | Best For |
|-------|------|----------|---------|-------------|----------|
| **trimplexx/music-transcription** | CRNN | 0.87 MPE | Tablature (fret/string) | ✅ | Solo guitar → tabs |
| **spotify/basic-pitch** | CNN | ~0.85 | MIDI + pitch bends | ✅ | General AMT |
| **TabCNN** | CNN | ~0.80 | Tablature | ❌ (train from scratch) | Research baseline |
| **FretNet** | CNN+Transformer | ~0.85 | Continuous pitch | ✅ | Pitch contours |
| **SynthTab-pretrained** | CNN | 0.87+ | Tablature | ✅ | Cross-dataset |

---

## 🔬 Key Projects

### 1. trimplexx/music-transcription ⭐ BEST FOR GUITAR TABS
- **GitHub**: https://github.com/trimplexx/music-transcription
- **Paper**: Master's thesis (Polish), Silesian University of Technology
- **Architecture**: CRNN (5-layer CNN → BiGRU → dual heads for onset + fret)
- **Performance**: 0.8736 MPE F1-Score on GuitarSet
- **Input**: CQT spectrogram (168 bins, 24 bins/octave, hop=512)
- **Output**: 6-string × 22-fret class predictions per frame

**Key Features:**
- Multi-task learning (onset detection + fret classification)
- Aggressive data augmentation (time stretch, noise, reverb, EQ, SpecAugment)
- Pre-trained weights available
- MIDI export capability

```bash
# Installation
git clone https://github.com/trimplexx/music-transcription
cd music-transcription/python
pip install -r requirements.txt
```

### 2. spotify/basic-pitch ⭐ EASIEST TO USE
- **GitHub**: https://github.com/spotify/basic-pitch
- **Paper**: ICASSP 2022 - "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription"
- **Demo**: https://basicpitch.spotify.com
- **License**: Apache 2.0

**Pros:**
- pip installable (`pip install basic-pitch`)
- Works on any instrument (guitar, voice, piano, etc.)
- Outputs MIDI with pitch bends
- Multiple backends: TensorFlow, CoreML, TFLite, ONNX
- Very fast inference

**Cons:**
- Outputs MIDI pitches, not guitar-specific tablature (string/fret)
- No playing technique detection

```python
from basic_pitch.inference import predict
model_output, midi_data, note_events = predict("guitar_audio.mp3")
midi_data.write("output.mid")
```

### 3. TabCNN (Original)
- **GitHub**: https://github.com/andywiggins/tab-cnn
- **Paper**: ISMIR 2019 - "Guitar Tablature Estimation with a CNN"
- **Status**: Reference implementation, Python 2.7 (dated)

Foundation work that most guitar-specific models build upon.

### 4. FretNet (Continuous Pitch)
- **GitHub**: https://github.com/cwitkowitz/guitar-transcription-continuous
- **Paper**: ICASSP 2023 - "FretNet: Continuous-Valued Pitch Contour Streaming"
- **Dependencies**: amt-tools, guitar-transcription-with-inhibition

Handles pitch bends and vibrato better than discrete fret models.

### 5. SynthTab (Synthetic Data)
- **Paper**: https://arxiv.org/html/2309.09085v3
- **Website**: www.synthtab.dev
- **Dataset**: 13,113 hours of synthesized guitar audio

Large-scale synthetic data for pre-training. Significantly improves cross-dataset generalization.

---

## 🎸 Source Separation (Pre-processing)

### Meta/Facebook Demucs
- **GitHub**: https://github.com/facebookresearch/demucs
- **Model**: `htdemucs_6s` - separates 6 sources including **guitar** and piano

```bash
# Install
pip install demucs

# Separate guitar from a full mix
demucs -n htdemucs_6s --two-stems=guitar "song.mp3"
# Output: separated/htdemucs_6s/song/guitar.wav
```

**Pipeline**: Full Mix → Demucs → Isolated Guitar → Basic Pitch/CRNN → Tabs

---

## 📁 Datasets

### GuitarSet (Standard Benchmark)
- **GitHub**: https://github.com/marl/GuitarSet
- **Paper**: ISMIR 2018
- **Content**: 360 recordings, 6 performers, acoustic guitar
- **Annotations**: JAMS format with onset/offset/pitch/string/fret
- **Download**: Via `mirdata` Python package

```python
import mirdata
guitarset = mirdata.initialize('guitarset')
guitarset.download()
```

### Other Datasets
- **IDMT-SMT-Guitar**: Electric guitar, various techniques
- **EGDB**: Electric guitar database (clean DI signals)
- **DadaGP**: Symbolic tablature (26k songs) - used for SynthTab

---

## 🔧 Implementation Approaches

### Approach A: Quick & Easy (Basic Pitch)
Best for: Getting MIDI quickly, instrument-agnostic use cases

```python
pip install basic-pitch
basic-pitch /output/dir /input/audio.mp3
```

### Approach B: Guitar Tablature (CRNN)
Best for: Actual guitar tabs with fret/string positions

1. Clone trimplexx/music-transcription
2. Download pre-trained weights
3. Run inference with CQT preprocessing

### Approach C: Full Pipeline (Isolation + Transcription)
Best for: Transcribing guitar from full band recordings

```bash
# Step 1: Isolate guitar
demucs -n htdemucs_6s --two-stems=guitar song.mp3

# Step 2: Transcribe isolated guitar
basic-pitch /output separated/htdemucs_6s/song/guitar.wav
# OR use trimplexx CRNN for tabs
```

---

## 📈 Technical Deep Dive

### Audio Preprocessing (Common)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 22,050 Hz | Standard for music |
| CQT Bins | 168-192 | 7-8 octaves × 24 bins/octave |
| Hop Length | 512 samples | ~23ms frame resolution |
| F_min | 82.4 Hz (E2) | Lowest guitar note |
| Max Frets | 19-22 | Standard guitar fretboard |

### Data Augmentation (Critical for Performance)
- **Time Stretch**: 0.8-1.2x rate (60% probability)
- **Noise Addition**: 0.001-0.01 level (70%)
- **Random Gain**: 0.6-1.4 factor (70%)
- **Reverb**: 0.1-0.45s decay (40%)
- **EQ/Bandpass**: 250-400 Hz to 3-4.5 kHz (50%)
- **Clipping**: 0.5-0.9 threshold (30%)
- **SpecAugment**: Time=40, Freq=26 masks

### The Tablature Problem
Unlike piano, the same pitch can be played at multiple fret/string combinations on guitar:
- E4 can be played at: 5th fret/1st string, 9th fret/2nd string, 14th fret/3rd string, etc.

Models must learn physical playability constraints and timbre differences between strings.

---

## 🚀 Recommended Implementation

### For This Project
I recommend a **two-stage approach**:

**Stage 1: Use Basic Pitch (immediate)**
- pip install
- Works out of the box
- Outputs MIDI with pitch detection

**Stage 2: Implement trimplexx CRNN (for proper tabs)**
- Clone repo, download GuitarSet
- Use pre-trained weights
- Outputs actual tablature notation

### Code Structure
```
/root/clawd/guitar-tabs/
├── RESEARCH.md           # This file
├── transcribe.py         # Main CLI tool
├── lib/
│   ├── basic_pitch.py    # Wrapper for Spotify model
│   ├── crnn_tabs.py      # Wrapper for CRNN model
│   └── demucs_sep.py     # Guitar isolation
└── models/
    └── crnn_weights.pth  # Pre-trained weights
```

---

## 📚 Key Papers

1. **"A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription"** (ICASSP 2022)
   - Bittner et al., Spotify
   - https://arxiv.org/abs/2203.09893

2. **"Guitar Tablature Estimation with a CNN"** (ISMIR 2019)
   - Wiggins & Kim
   - https://archives.ismir.net/ismir2019/paper/000033.pdf

3. **"FretNet: Continuous-Valued Pitch Contour Streaming"** (ICASSP 2023)
   - Cwitkowitz et al.
   - https://arxiv.org/abs/2212.03023

4. **"SynthTab: Leveraging Synthesized Data for GTT"** (2024)
   - https://arxiv.org/abs/2309.09085

5. **"Hybrid Transformers for Music Source Separation"** (ICASSP 2023)
   - Rouard et al., Meta/Facebook
   - https://arxiv.org/abs/2211.08553

6. **"GuitarSet: A Dataset for Guitar Transcription"** (ISMIR 2018)
   - Xi et al.
   - https://guitarset.weebly.com/

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| Basic Pitch Demo | https://basicpitch.spotify.com |
| trimplexx CRNN | https://github.com/trimplexx/music-transcription |
| Demucs | https://github.com/facebookresearch/demucs |
| GuitarSet | https://guitarset.weebly.com |
| Awesome AGT List | https://github.com/lucasgris/awesome-agt |
| amt-tools Framework | https://github.com/cwitkowitz/amt-tools |

---

*Last Updated: 2026-02-13*
*Research by: Claude (Subagent)*
