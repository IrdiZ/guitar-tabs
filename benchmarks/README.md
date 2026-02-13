# Guitar Tab Transcription Benchmarks

Testing and benchmarking framework for evaluating pitch detection methods.

## Quick Start

```bash
# Generate test audio and run full benchmark
cd /root/clawd/guitar-tabs
source venv/bin/activate
python benchmarks/run_benchmark.py --generate --run

# Run benchmark only (if test audio already exists)
python benchmarks/run_benchmark.py --run

# Run specific methods
python benchmarks/run_benchmark.py --run --methods pyin,hybrid
```

## Structure

```
benchmarks/
├── README.md              # This file
├── generate_test_audio.py # Creates synthetic guitar audio with ground truth
├── metrics.py             # Accuracy metrics (precision, recall, F1, etc.)
├── pitch_detectors.py     # Wrappers for different pitch detection methods
├── run_benchmark.py       # Main benchmark runner
├── test_audio/            # Generated test files (auto-created)
│   ├── index.json         # Test case index
│   ├── single_E2_low_e_open.wav
│   ├── single_E2_low_e_open.json  # Ground truth
│   └── ...
└── results/               # Benchmark results (auto-created)
    ├── results_latest.json
    ├── results_YYYYMMDD_HHMMSS.json
    └── summary_YYYYMMDD_HHMMSS.txt
```

## Test Cases

### Single Notes
- `single_*` - Individual guitar-like notes across the range (E2-A5)
- `pure_*` - Pure sine waves (baseline for comparison)

### Sequences
- `seq_c_major_scale` - Ascending C major scale
- `seq_chromatic` - Chromatic passage
- `seq_pentatonic_lick` - Pentatonic lick pattern
- `seq_fast_16ths` - Fast 16th notes at 120 BPM

### Chords
- `chord_E_major`, `chord_A_major`, etc. - Common guitar chords
- `chord_progression_*` - Chord progressions

### Edge Cases
- `edge_quiet_note` - Very quiet note (velocity 0.15)
- `edge_staccato` - Very short note (50ms)
- `edge_sustained` - Long held note (5s)
- `edge_repeated_note` - Same pitch re-attacked
- `edge_overlapping` - Overlapping legato notes
- `edge_with_noise` - Note with background noise
- `edge_interval_fifth` - Perfect fifth interval

## Detection Methods

| Method | Type | Best For |
|--------|------|----------|
| `pyin` | Probabilistic YIN | Monophonic, vibrato |
| `piptrack` | FFT peak tracking | Polyphonic |
| `fft_peak` | Simple FFT peak | Baseline, pure tones |
| `autocorr` | Autocorrelation | Simple, clean signals |
| `hybrid` | pYIN + onset detection | Better timing |

## Metrics

### Core Metrics
- **Precision**: % of detected notes that match ground truth
- **Recall**: % of ground truth notes that were detected
- **F1 Score**: Harmonic mean of precision and recall

### Tolerances
- **Pitch tolerance**: 0.5 semitones (quarter tone)
- **Onset tolerance**: 50ms
- **Offset tolerance**: 100ms

### Error Metrics
- **Mean pitch error**: Average deviation in semitones
- **Mean onset error**: Average timing error in milliseconds
- **Processing time**: Milliseconds to process audio

## Interpreting Results

```
================================================================================
Method               Precision     Recall         F1   Pitch Err    Onset Err   Time (ms)
================================================================================
pyin                     0.850      0.920      0.884    0.050 st      15.2 ms        120.5
hybrid                   0.880      0.890      0.885    0.045 st      10.1 ms        150.2
piptrack                 0.720      0.850      0.780    0.100 st      25.5 ms         80.3
```

- **F1 > 0.9**: Excellent - method works well for this test case
- **F1 0.7-0.9**: Good - some issues but usable
- **F1 < 0.7**: Poor - significant detection problems

## Adding New Methods

1. Create a new detector class in `pitch_detectors.py`:

```python
class MyDetector(PitchDetector):
    name = "my_method"
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Your detection logic here
        return notes
```

2. Register it in the `DETECTORS` dict:

```python
DETECTORS['my_method'] = MyDetector
```

3. Run benchmark: `python run_benchmark.py --run`

## Adding New Test Cases

Edit `generate_test_audio.py` and add to the appropriate generator method:
- `generate_single_notes()` for individual note tests
- `generate_sequences()` for melodic tests
- `generate_chords()` for polyphonic tests
- `generate_edge_cases()` for edge case tests
