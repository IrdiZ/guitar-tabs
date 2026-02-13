#!/usr/bin/env python3
"""
Ground Truth Testing for Guitar Tab Detection

Creates synthetic audio with KNOWN notes and compares detection results
to calculate accuracy metrics. Use this to tune parameters.

Usage:
    python ground_truth_test.py                    # Run all tests
    python ground_truth_test.py --generate-only    # Just generate test audio
    python ground_truth_test.py --tune             # Parameter tuning mode
"""

import numpy as np
import soundfile as sf
import librosa
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import subprocess
import tempfile

# Constants
SR = 22050  # Sample rate matching guitar_tabs.py
GUITAR_TUNING = [329.63, 246.94, 196.00, 146.83, 110.00, 82.41]  # E4 B3 G3 D3 A2 E2
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class GroundTruthNote:
    """A note in our ground truth."""
    start_time: float  # seconds
    duration: float    # seconds
    frequency: float   # Hz
    midi_note: int     # MIDI number
    note_name: str     # e.g., "E4"
    string: Optional[int] = None  # 0-5 for guitar string
    fret: Optional[int] = None


@dataclass
class DetectedNote:
    """A detected note from the system."""
    start_time: float
    duration: float
    frequency: float
    midi_note: int
    note_name: str


@dataclass
class TestCase:
    """A test case with audio and ground truth."""
    name: str
    description: str
    audio_file: str
    ground_truth: List[GroundTruthNote]
    duration: float
    
    
def freq_to_midi(freq: float) -> int:
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def midi_to_name(midi: int) -> str:
    """Convert MIDI note to name like 'E4'."""
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def name_to_midi(name: str) -> int:
    """Convert note name like 'E4' to MIDI number."""
    # Parse note name and octave
    if len(name) == 2:
        note = name[0]
        octave = int(name[1])
    elif len(name) == 3:
        note = name[:2]
        octave = int(name[2])
    else:
        raise ValueError(f"Invalid note name: {name}")
    
    note_idx = NOTE_NAMES.index(note)
    return (octave + 1) * 12 + note_idx


def generate_guitar_tone(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    """
    Generate a guitar-like tone using Karplus-Strong synthesis.
    
    This creates a more realistic test signal than pure sine waves.
    """
    n_samples = int(duration * sr)
    
    # Karplus-Strong: start with noise burst, then filter
    delay_samples = int(sr / freq)
    
    # Initial noise burst (pluck)
    noise = np.random.uniform(-1, 1, delay_samples)
    
    # Output buffer
    output = np.zeros(n_samples)
    output[:delay_samples] = noise
    
    # Karplus-Strong feedback loop with lowpass averaging
    decay = 0.996  # Controls sustain
    for i in range(delay_samples, n_samples):
        # Average two samples (simple lowpass) with decay
        output[i] = decay * 0.5 * (output[i - delay_samples] + output[i - delay_samples - 1])
    
    # Apply envelope (quick attack, slow decay)
    attack_samples = int(0.005 * sr)  # 5ms attack
    decay_samples = n_samples - attack_samples
    
    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples),
        np.exp(-np.linspace(0, 4, decay_samples))  # Exponential decay
    ])
    
    output = output * envelope[:len(output)]
    
    # Normalize
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.8
    
    return output


def generate_pure_tone(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    """Generate a pure sine wave tone (simpler test case)."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    # Add slight attack/decay to avoid clicks
    envelope = np.ones_like(t)
    attack = int(0.01 * sr)
    release = int(0.01 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    return 0.8 * np.sin(2 * np.pi * freq * t) * envelope


def generate_harmonic_tone(freq: float, duration: float, sr: int = SR, n_harmonics: int = 6) -> np.ndarray:
    """Generate a tone with harmonics (more guitar-like than pure sine)."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    # Create tone with decaying harmonics
    tone = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        harmonic_amp = 1.0 / (h ** 1.5)  # Harmonics decay
        if freq * h < sr / 2:  # Stay below Nyquist
            tone += harmonic_amp * np.sin(2 * np.pi * freq * h * t)
    
    # Envelope
    attack = int(0.01 * sr)
    decay_samples = len(t) - attack
    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.exp(-np.linspace(0, 3, decay_samples))
    ])
    
    tone = tone * envelope
    
    # Normalize
    if np.max(np.abs(tone)) > 0:
        tone = tone / np.max(np.abs(tone)) * 0.8
    
    return tone


def create_test_audio(notes: List[GroundTruthNote], duration: float, 
                      tone_type: str = 'guitar', sr: int = SR) -> np.ndarray:
    """Create audio from a list of ground truth notes."""
    audio = np.zeros(int(duration * sr))
    
    for note in notes:
        start_sample = int(note.start_time * sr)
        note_samples = int(note.duration * sr)
        
        if tone_type == 'guitar':
            tone = generate_guitar_tone(note.frequency, note.duration, sr)
        elif tone_type == 'harmonic':
            tone = generate_harmonic_tone(note.frequency, note.duration, sr)
        else:
            tone = generate_pure_tone(note.frequency, note.duration, sr)
        
        # Add to audio (handle polyphony by summing)
        end_sample = min(start_sample + len(tone), len(audio))
        actual_len = end_sample - start_sample
        audio[start_sample:end_sample] += tone[:actual_len]
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 0.95:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio


def create_test_cases(output_dir: str) -> List[TestCase]:
    """Generate all test cases with known ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    test_cases = []
    
    # Test Case 1: Simple Scale (E minor pentatonic)
    # E4, G4, A4, B4, D5, E5
    scale_notes = [
        GroundTruthNote(0.0, 0.5, 329.63, 64, "E4", 0, 0),
        GroundTruthNote(0.6, 0.5, 392.00, 67, "G4", 0, 3),
        GroundTruthNote(1.2, 0.5, 440.00, 69, "A4", 0, 5),
        GroundTruthNote(1.8, 0.5, 493.88, 71, "B4", 1, 0),
        GroundTruthNote(2.4, 0.5, 587.33, 74, "D5", 1, 3),
        GroundTruthNote(3.0, 0.5, 659.25, 76, "E5", 1, 5),
    ]
    audio = create_test_audio(scale_notes, 4.0, 'guitar')
    audio_path = os.path.join(output_dir, "test_scale.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "e_minor_pentatonic_scale",
        "E minor pentatonic scale - 6 notes, 600ms spacing",
        audio_path,
        scale_notes,
        4.0
    ))
    
    # Test Case 2: Open Chord (E major - multiple simultaneous notes)
    # E2, B2, E3, G#3, B3, E4
    chord_notes = [
        GroundTruthNote(0.0, 1.5, 82.41, 40, "E2", 5, 0),
        GroundTruthNote(0.0, 1.5, 123.47, 47, "B2", 4, 2),
        GroundTruthNote(0.0, 1.5, 164.81, 52, "E3", 3, 2),
        GroundTruthNote(0.0, 1.5, 207.65, 56, "G#3", 2, 1),
        GroundTruthNote(0.0, 1.5, 246.94, 59, "B3", 1, 0),
        GroundTruthNote(0.0, 1.5, 329.63, 64, "E4", 0, 0),
    ]
    audio = create_test_audio(chord_notes, 2.0, 'guitar')
    audio_path = os.path.join(output_dir, "test_chord_e.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "e_major_chord",
        "E major open chord - 6 simultaneous notes",
        audio_path,
        chord_notes,
        2.0
    ))
    
    # Test Case 3: Simple Riff (classic rock pattern)
    # E2 - G2 - A2 - G2 - E2 (power chord roots)
    riff_notes = [
        GroundTruthNote(0.0, 0.3, 82.41, 40, "E2", 5, 0),
        GroundTruthNote(0.4, 0.3, 98.00, 43, "G2", 5, 3),
        GroundTruthNote(0.8, 0.3, 110.00, 45, "A2", 5, 5),
        GroundTruthNote(1.2, 0.3, 98.00, 43, "G2", 5, 3),
        GroundTruthNote(1.6, 0.3, 82.41, 40, "E2", 5, 0),
        GroundTruthNote(2.0, 0.3, 98.00, 43, "G2", 5, 3),
        GroundTruthNote(2.4, 0.3, 110.00, 45, "A2", 5, 5),
        GroundTruthNote(2.8, 0.3, 98.00, 43, "G2", 5, 3),
    ]
    audio = create_test_audio(riff_notes, 3.5, 'guitar')
    audio_path = os.path.join(output_dir, "test_riff.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "simple_riff",
        "Simple rock riff - 8 notes, 400ms spacing, repeated pattern",
        audio_path,
        riff_notes,
        3.5
    ))
    
    # Test Case 4: Pure tones (easiest to detect - baseline)
    pure_notes = [
        GroundTruthNote(0.0, 0.8, 440.00, 69, "A4"),
        GroundTruthNote(1.0, 0.8, 493.88, 71, "B4"),
        GroundTruthNote(2.0, 0.8, 523.25, 72, "C5"),
        GroundTruthNote(3.0, 0.8, 587.33, 74, "D5"),
    ]
    audio = create_test_audio(pure_notes, 4.0, 'pure')
    audio_path = os.path.join(output_dir, "test_pure_tones.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "pure_tones",
        "Pure sine waves - A4, B4, C5, D5 - easiest to detect",
        audio_path,
        pure_notes,
        4.0
    ))
    
    # Test Case 5: Fast notes (testing onset detection)
    fast_notes = [
        GroundTruthNote(0.0, 0.15, 329.63, 64, "E4"),
        GroundTruthNote(0.2, 0.15, 349.23, 65, "F4"),
        GroundTruthNote(0.4, 0.15, 392.00, 67, "G4"),
        GroundTruthNote(0.6, 0.15, 440.00, 69, "A4"),
        GroundTruthNote(0.8, 0.15, 493.88, 71, "B4"),
        GroundTruthNote(1.0, 0.15, 523.25, 72, "C5"),
        GroundTruthNote(1.2, 0.15, 587.33, 74, "D5"),
        GroundTruthNote(1.4, 0.15, 659.25, 76, "E5"),
    ]
    audio = create_test_audio(fast_notes, 2.0, 'guitar')
    audio_path = os.path.join(output_dir, "test_fast_notes.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "fast_notes",
        "Fast sequence - 8 notes, 200ms spacing (testing onset detection)",
        audio_path,
        fast_notes,
        2.0
    ))
    
    # Test Case 6: Wide range (low to high)
    range_notes = [
        GroundTruthNote(0.0, 0.5, 82.41, 40, "E2"),   # Low E
        GroundTruthNote(0.7, 0.5, 164.81, 52, "E3"),  # Octave up
        GroundTruthNote(1.4, 0.5, 329.63, 64, "E4"),  # Another octave
        GroundTruthNote(2.1, 0.5, 659.25, 76, "E5"),  # High E
        GroundTruthNote(2.8, 0.5, 329.63, 64, "E4"),  # Back down
        GroundTruthNote(3.5, 0.5, 164.81, 52, "E3"),
    ]
    audio = create_test_audio(range_notes, 4.5, 'guitar')
    audio_path = os.path.join(output_dir, "test_range.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "wide_range",
        "Wide range - E2 to E5 (testing frequency detection across octaves)",
        audio_path,
        range_notes,
        4.5
    ))
    
    # Test Case 7: Chromatic passage (adjacent semitones)
    chromatic_notes = [
        GroundTruthNote(0.0, 0.3, 329.63, 64, "E4"),
        GroundTruthNote(0.4, 0.3, 349.23, 65, "F4"),
        GroundTruthNote(0.8, 0.3, 369.99, 66, "F#4"),
        GroundTruthNote(1.2, 0.3, 392.00, 67, "G4"),
        GroundTruthNote(1.6, 0.3, 415.30, 68, "G#4"),
        GroundTruthNote(2.0, 0.3, 440.00, 69, "A4"),
    ]
    audio = create_test_audio(chromatic_notes, 2.8, 'harmonic')
    audio_path = os.path.join(output_dir, "test_chromatic.wav")
    sf.write(audio_path, audio, SR)
    test_cases.append(TestCase(
        "chromatic",
        "Chromatic scale E4-A4 - testing semitone detection",
        audio_path,
        chromatic_notes,
        2.8
    ))
    
    # Save ground truth JSON
    gt_data = {}
    for tc in test_cases:
        gt_data[tc.name] = {
            "description": tc.description,
            "audio_file": tc.audio_file,
            "duration": tc.duration,
            "notes": [
                {
                    "start_time": n.start_time,
                    "duration": n.duration,
                    "frequency": n.frequency,
                    "midi_note": n.midi_note,
                    "note_name": n.note_name,
                    "string": n.string,
                    "fret": n.fret
                }
                for n in tc.ground_truth
            ]
        }
    
    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    print(f"Generated {len(test_cases)} test cases in {output_dir}")
    return test_cases


def run_detection(audio_file: str, extra_args: List[str] = None) -> List[DetectedNote]:
    """Run guitar_tabs.py on audio and parse output."""
    cmd = [
        sys.executable, "guitar_tabs.py", 
        audio_file, 
        "-o", "/dev/stdout",
        "--verbose"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__) or '.')
        
        # Parse the output - looking for JSON or notes in various formats
        notes = []
        
        # Try to find note information in output
        lines = result.stdout.split('\n') + result.stderr.split('\n')
        
        # Look for JSON output file
        for line in lines:
            if "transcribed_audio.json" in line or "Writing JSON" in line:
                json_path = os.path.join(os.path.dirname(__file__), "transcribed_audio.json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                        for n in data.get('notes', []):
                            notes.append(DetectedNote(
                                start_time=n.get('start_time', 0),
                                duration=n.get('duration', 0),
                                frequency=n.get('frequency', 0),
                                midi_note=n.get('midi_note', 0),
                                note_name=n.get('note_name', '')
                            ))
                    return notes
        
        # Parse verbose output for notes
        for line in lines:
            if "Note:" in line or "Detected:" in line:
                # Try to extract note info
                pass
        
        return notes
        
    except Exception as e:
        print(f"Error running detection: {e}")
        return []


def run_detection_direct(audio_file: str, **params) -> List[DetectedNote]:
    """Run detection directly using the module (more control over params)."""
    import importlib.util
    
    # Load the guitar_tabs module
    spec = importlib.util.spec_from_file_location("guitar_tabs", 
        os.path.join(os.path.dirname(__file__), "guitar_tabs.py"))
    guitar_tabs = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(guitar_tabs)
    except Exception as e:
        print(f"Warning loading guitar_tabs: {e}")
        return []
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=22050)
    
    # Get parameters with defaults
    hop_length = params.get('hop_length', 512)
    pitch_method = params.get('pitch_method', 'cqt')
    min_confidence = params.get('min_confidence', 0.3)
    
    notes = []
    
    try:
        if pitch_method == 'pyin':
            # Use pyin pitch detection
            f0, voiced, probs = guitar_tabs.detect_pitch_pyin(y, sr, hop_length)
            frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
            
            # Detect onsets
            onset_times, onset_details = guitar_tabs.detect_onsets_ensemble(
                y, sr, hop_length=hop_length, 
                verbose=params.get('verbose', False)
            )
            
            for onset_time in onset_times:
                # Find nearest frame
                frame_idx = np.argmin(np.abs(frame_times - onset_time))
                
                # Look for valid pitch in window after onset
                window_end = min(len(f0), frame_idx + 10)
                for i in range(frame_idx, window_end):
                    if f0[i] > 0 and voiced[i]:
                        freq = f0[i]
                        midi = freq_to_midi(freq)
                        
                        # Estimate duration
                        dur = 0.3  # Default
                        for j in range(i + 1, min(len(f0), i + 50)):
                            if f0[j] == 0 or not voiced[j]:
                                dur = frame_times[j] - onset_time
                                break
                        
                        notes.append(DetectedNote(
                            start_time=onset_time,
                            duration=dur,
                            frequency=freq,
                            midi_note=midi,
                            note_name=midi_to_name(midi)
                        ))
                        break
        
        elif pitch_method == 'cqt':
            # Use CQT-based note detection
            detected_notes = guitar_tabs.detect_notes_cqt(
                y, sr, 
                hop_length=hop_length,
                confidence_threshold=min_confidence,
            )
            
            for n in detected_notes:
                freq = midi_to_freq(n.midi)
                notes.append(DetectedNote(
                    start_time=n.start_time,
                    duration=n.duration,
                    frequency=freq,
                    midi_note=n.midi,
                    note_name=midi_to_name(n.midi)
                ))
        
        elif pitch_method == 'voting':
            # Use voting-based detection (multi-algorithm)
            detected_notes = guitar_tabs.detect_notes_with_voting(
                y, sr,
                hop_length=hop_length,
                verbose=params.get('verbose', False)
            )
            
            for n in detected_notes:
                freq = midi_to_freq(n.midi)
                notes.append(DetectedNote(
                    start_time=n.start_time,
                    duration=n.duration,
                    frequency=freq,
                    midi_note=n.midi,
                    note_name=midi_to_name(n.midi)
                ))
        
        elif pitch_method == 'polyphonic':
            # Use polyphonic NMF-based detection (for chords)
            # This function takes audio_path, not y/sr
            detected_notes = guitar_tabs.detect_notes_polyphonic(
                audio_file,
                hop_length=hop_length,
                confidence_threshold=min_confidence,
                method=params.get('poly_method', 'nmf'),
                max_simultaneous=params.get('max_simultaneous', 6)
            )
            
            for n in detected_notes:
                freq = midi_to_freq(n.midi)
                notes.append(DetectedNote(
                    start_time=n.start_time,
                    duration=n.duration,
                    frequency=freq,
                    midi_note=n.midi,
                    note_name=midi_to_name(n.midi)
                ))
    
    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
    
    return notes


@dataclass
class ComparisonResult:
    """Results of comparing detected vs ground truth."""
    test_name: str
    ground_truth_count: int
    detected_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    pitch_errors: List[Tuple[str, str, int]]  # (expected, got, cents_error)
    timing_errors: List[Tuple[float, float]]  # (expected_time, detected_time)
    
    @property
    def precision(self) -> float:
        if self.detected_count == 0:
            return 0.0
        return self.true_positives / self.detected_count
    
    @property
    def recall(self) -> float:
        if self.ground_truth_count == 0:
            return 0.0
        return self.true_positives / self.ground_truth_count
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def pitch_accuracy(self) -> float:
        """Percentage of detected notes with correct pitch (within 50 cents)."""
        if self.true_positives == 0:
            return 0.0
        correct = sum(1 for _, _, err in self.pitch_errors if abs(err) <= 50)
        return correct / self.true_positives


def compare_results(ground_truth: List[GroundTruthNote], 
                   detected: List[DetectedNote],
                   time_tolerance: float = 0.1,
                   pitch_tolerance_cents: float = 50) -> ComparisonResult:
    """
    Compare detected notes to ground truth.
    
    Args:
        ground_truth: Expected notes
        detected: Detected notes
        time_tolerance: Max time difference (seconds) to match notes
        pitch_tolerance_cents: Max pitch difference (cents) for correct detection
    """
    matched_gt = set()
    matched_det = set()
    pitch_errors = []
    timing_errors = []
    
    # Sort by time
    gt_sorted = sorted(enumerate(ground_truth), key=lambda x: x[1].start_time)
    det_sorted = sorted(enumerate(detected), key=lambda x: x[1].start_time)
    
    # Match detected notes to ground truth
    for det_idx, det_note in det_sorted:
        if det_idx in matched_det:
            continue
            
        best_match = None
        best_time_diff = float('inf')
        
        for gt_idx, gt_note in gt_sorted:
            if gt_idx in matched_gt:
                continue
            
            time_diff = abs(det_note.start_time - gt_note.start_time)
            if time_diff <= time_tolerance and time_diff < best_time_diff:
                # Check if pitch is close enough
                midi_diff = abs(det_note.midi_note - gt_note.midi_note)
                if midi_diff <= 2:  # Within a whole tone
                    best_match = gt_idx
                    best_time_diff = time_diff
        
        if best_match is not None:
            matched_gt.add(best_match)
            matched_det.add(det_idx)
            
            gt_note = ground_truth[best_match]
            
            # Calculate pitch error in cents
            if det_note.frequency > 0 and gt_note.frequency > 0:
                cents_error = 1200 * np.log2(det_note.frequency / gt_note.frequency)
            else:
                cents_error = (det_note.midi_note - gt_note.midi_note) * 100
            
            pitch_errors.append((gt_note.note_name, det_note.note_name, int(cents_error)))
            timing_errors.append((gt_note.start_time, det_note.start_time))
    
    true_positives = len(matched_gt)
    false_positives = len(detected) - len(matched_det)
    false_negatives = len(ground_truth) - len(matched_gt)
    
    return ComparisonResult(
        test_name="",
        ground_truth_count=len(ground_truth),
        detected_count=len(detected),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        pitch_errors=pitch_errors,
        timing_errors=timing_errors
    )


def run_all_tests(test_dir: str, params: Dict = None) -> Dict[str, ComparisonResult]:
    """Run all test cases and return results."""
    params = params or {}
    results = {}
    
    # Load ground truth
    gt_path = os.path.join(test_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        print("Ground truth not found, generating test cases...")
        test_cases = create_test_cases(test_dir)
    else:
        with open(gt_path) as f:
            gt_data = json.load(f)
        
        test_cases = []
        for name, data in gt_data.items():
            notes = [
                GroundTruthNote(
                    start_time=n['start_time'],
                    duration=n['duration'],
                    frequency=n['frequency'],
                    midi_note=n['midi_note'],
                    note_name=n['note_name'],
                    string=n.get('string'),
                    fret=n.get('fret')
                )
                for n in data['notes']
            ]
            test_cases.append(TestCase(
                name=name,
                description=data['description'],
                audio_file=data['audio_file'],
                ground_truth=notes,
                duration=data['duration']
            ))
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print("=" * 70)
    
    for tc in test_cases:
        print(f"\n📋 {tc.name}: {tc.description}")
        
        if not os.path.exists(tc.audio_file):
            print(f"   ⚠️  Audio file not found: {tc.audio_file}")
            continue
        
        # Run detection
        detected = run_detection_direct(tc.audio_file, **params)
        
        # Compare
        result = compare_results(tc.ground_truth, detected)
        result.test_name = tc.name
        results[tc.name] = result
        
        # Print results
        print(f"   Ground truth: {result.ground_truth_count} notes")
        print(f"   Detected: {result.detected_count} notes")
        print(f"   ✓ True positives: {result.true_positives}")
        print(f"   ✗ False positives: {result.false_positives}")
        print(f"   ○ Missed: {result.false_negatives}")
        print(f"   📊 Precision: {result.precision:.1%}")
        print(f"   📊 Recall: {result.recall:.1%}")
        print(f"   📊 F1 Score: {result.f1:.1%}")
        
        if result.pitch_errors:
            print(f"   🎵 Pitch accuracy: {result.pitch_accuracy:.1%}")
            # Show any significant pitch errors
            errors = [(e, g, c) for e, g, c in result.pitch_errors if abs(c) > 50]
            if errors:
                print(f"   ⚠️  Pitch errors (>50 cents):")
                for exp, got, cents in errors[:3]:
                    print(f"      Expected {exp}, got {got} ({cents:+d} cents)")
    
    return results


def print_summary(results: Dict[str, ComparisonResult]):
    """Print summary of all test results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_gt = sum(r.ground_truth_count for r in results.values())
    total_det = sum(r.detected_count for r in results.values())
    total_tp = sum(r.true_positives for r in results.values())
    
    overall_precision = total_tp / total_det if total_det > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\nOverall Metrics:")
    print(f"  Total ground truth notes: {total_gt}")
    print(f"  Total detected notes: {total_det}")
    print(f"  Overall Precision: {overall_precision:.1%}")
    print(f"  Overall Recall: {overall_recall:.1%}")
    print(f"  Overall F1 Score: {overall_f1:.1%}")
    
    print(f"\nPer-test F1 scores:")
    for name, result in sorted(results.items(), key=lambda x: x[1].f1, reverse=True):
        emoji = "✅" if result.f1 >= 0.8 else "⚠️" if result.f1 >= 0.5 else "❌"
        print(f"  {emoji} {name}: {result.f1:.1%}")
    
    # Identify worst performing tests
    worst = sorted(results.items(), key=lambda x: x[1].f1)[:3]
    if worst and worst[0][1].f1 < 0.8:
        print(f"\n⚠️  Tests needing improvement:")
        for name, result in worst:
            if result.f1 < 0.8:
                print(f"   - {name} (F1: {result.f1:.1%})")


def tune_parameters(test_dir: str):
    """Try different parameter combinations to find optimal settings."""
    print("\n🔧 PARAMETER TUNING MODE")
    print("=" * 70)
    
    # Parameter grid to search
    param_grid = {
        'hop_length': [256, 512, 1024],
        'pitch_method': ['pyin', 'cqt', 'voting'],
        'min_confidence': [0.2, 0.3, 0.4, 0.5],
    }
    
    best_f1 = 0
    best_params = {}
    
    # Quick test with a subset
    test_audio = os.path.join(test_dir, "test_scale.wav")
    if not os.path.exists(test_audio):
        create_test_cases(test_dir)
    
    gt_path = os.path.join(test_dir, "ground_truth.json")
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Use scale test for tuning (quickest)
    scale_data = gt_data['e_minor_pentatonic_scale']
    gt_notes = [
        GroundTruthNote(
            start_time=n['start_time'],
            duration=n['duration'],
            frequency=n['frequency'],
            midi_note=n['midi_note'],
            note_name=n['note_name']
        )
        for n in scale_data['notes']
    ]
    
    print(f"Using '{scale_data['description']}' for parameter tuning\n")
    
    tried = 0
    for hop in param_grid['hop_length']:
        for method in param_grid['pitch_method']:
            for conf in param_grid['min_confidence']:
                params = {
                    'hop_length': hop,
                    'pitch_method': method,
                    'min_confidence': conf,
                    'verbose': False
                }
                
                detected = run_detection_direct(test_audio, **params)
                result = compare_results(gt_notes, detected)
                
                tried += 1
                if result.f1 > best_f1:
                    best_f1 = result.f1
                    best_params = params.copy()
                    print(f"✨ New best F1: {result.f1:.1%} with {params}")
    
    print(f"\nTried {tried} combinations")
    print(f"\n🏆 BEST PARAMETERS:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    print(f"   F1 Score: {best_f1:.1%}")
    
    # Run full tests with best params
    print(f"\nRunning full test suite with optimal parameters...")
    results = run_all_tests(test_dir, best_params)
    print_summary(results)
    
    # Save best params
    config_path = os.path.join(test_dir, "optimal_params.json")
    with open(config_path, 'w') as f:
        json.dump({
            'best_f1': best_f1,
            'parameters': best_params
        }, f, indent=2)
    print(f"\nSaved optimal parameters to {config_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ground truth testing for guitar tab detection")
    parser.add_argument('--test-dir', default='test_ground_truth', help='Directory for test files')
    parser.add_argument('--generate-only', action='store_true', help='Only generate test audio')
    parser.add_argument('--tune', action='store_true', help='Run parameter tuning')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    test_dir = os.path.join(os.path.dirname(__file__), args.test_dir)
    
    if args.generate_only:
        create_test_cases(test_dir)
        print("\nTest audio generated. Listen to verify quality.")
        return
    
    if args.tune:
        tune_parameters(test_dir)
        return
    
    # Normal test run
    results = run_all_tests(test_dir, {'verbose': args.verbose})
    print_summary(results)
    
    # Save results
    results_path = os.path.join(test_dir, "test_results.json")
    results_data = {}
    for name, r in results.items():
        results_data[name] = {
            'ground_truth_count': r.ground_truth_count,
            'detected_count': r.detected_count,
            'true_positives': r.true_positives,
            'false_positives': r.false_positives,
            'false_negatives': r.false_negatives,
            'precision': r.precision,
            'recall': r.recall,
            'f1': r.f1,
            'pitch_accuracy': r.pitch_accuracy
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
