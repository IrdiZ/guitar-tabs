# Basic Pitch Integration

This directory contains the Docker setup for Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) library, which provides polyphonic music transcription using a neural network.

## Why Docker?

Basic Pitch requires specific versions of TensorFlow and NumPy that aren't compatible with Python 3.13 (which is what the main project uses). The Docker container uses Python 3.11 with TensorFlow 2.15 to ensure compatibility.

## Building the Docker Image

```bash
cd guitar-tabs/basicpitch
docker build -t guitar-tabs-basicpitch .
```

This will create a ~4GB Docker image with all the necessary dependencies.

## Usage

### From guitar_tabs.py

```bash
python guitar_tabs.py --pitch-method basicpitch song.mp3
```

### Standalone

```bash
docker run --rm -v /path/to/audio:/data guitar-tabs-basicpitch /data/song.mp3
```

The output is JSON with detected notes:

```json
{
  "source": "/data/song.mp3",
  "model": "basic-pitch-icassp-2022",
  "notes": [
    {
      "midi": 60,
      "start_time": 0.5,
      "duration": 0.25,
      "confidence": 0.95,
      "name": "C4",
      "pitch_bends": []
    }
  ],
  "total_notes": 42
}
```

## Options

- `--onset-threshold` (0-1, default: 0.5): Minimum confidence for note onset
- `--frame-threshold` (0-1, default: 0.3): Minimum confidence for note frames  
- `--min-note-len` (ms, default: 50): Minimum note length in milliseconds
- `--min-freq` (Hz): Minimum frequency to detect
- `--max-freq` (Hz): Maximum frequency to detect

## Why Basic Pitch?

Unlike pYIN or CREPE which are designed for monophonic (single note) pitch detection, Basic Pitch can detect multiple simultaneous notes - perfect for guitar chords. It uses a neural network trained on a large dataset of music to provide accurate polyphonic transcription.

## Model

Uses the ICASSP 2022 model included in basic-pitch, which was trained on:
- MIR-1K dataset
- MedleyDB
- Slakh2100
- And other polyphonic music datasets
