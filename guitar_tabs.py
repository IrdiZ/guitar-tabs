#!/usr/bin/env python3
"""
Guitar Tab Generator - Audio to Tabs using AI
Built for Albanian songs with no existing tabs 🎸

Improved pitch detection with:
- Multiple pitch detection algorithms (pyin, piptrack)
- Median filtering for jitter reduction
- Onset detection with backtracking
- Pitch confidence smoothing
- Harmonic handling for guitar
- Support for alternate tunings
"""

import librosa
import numpy as np
from scipy import ndimage
from scipy.signal import medfilt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import argparse
import sys
import os
import subprocess
import tempfile

# Guitar tunings (MIDI note numbers, low to high)
TUNINGS = {
    'standard': [40, 45, 50, 55, 59, 64],  # E A D G B E
    'drop_d': [38, 45, 50, 55, 59, 64],    # D A D G B E
    'drop_c': [36, 43, 48, 53, 57, 62],    # C G C F A D
    'half_step_down': [39, 44, 49, 54, 58, 63],  # Eb Ab Db Gb Bb Eb
    'full_step_down': [38, 43, 48, 53, 57, 62],  # D G C F A D
    'open_d': [38, 45, 50, 54, 57, 62],    # D A D F# A D
    'open_g': [38, 43, 50, 55, 59, 62],    # D G D G B D
    'dadgad': [38, 45, 50, 55, 57, 62],    # D A D G A D
}

STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
NUM_FRETS = 24

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Guitar frequency range (E2 to ~E6 with harmonics)
GUITAR_MIN_HZ = 75   # Below E2 (82 Hz) to catch drop tunings
GUITAR_MAX_HZ = 1400  # Above high frets on high E string

# Chord detection threshold (notes within this time window are considered simultaneous)
CHORD_TIME_THRESHOLD = 0.050  # 50ms

# Common chord definitions: chord_name -> intervals from root (semitones)
CHORD_INTERVALS = {
    # Major chords
    '': [0, 4, 7],  # Major (no suffix)
    'maj': [0, 4, 7],
    # Minor chords
    'm': [0, 3, 7],
    'min': [0, 3, 7],
    # 7th chords
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    # Sus chords
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    # Diminished/Augmented
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    # Add chords
    'add9': [0, 4, 7, 14],
    # Power chord
    '5': [0, 7],
}

# Common open chord shapes: chord_name -> [(string, fret), ...] where -1 = muted
OPEN_CHORD_SHAPES = {
    'C': [(0, -1), (1, 3), (2, 2), (3, 0), (4, 1), (5, 0)],
    'Am': [(0, -1), (1, 0), (2, 2), (3, 2), (4, 1), (5, 0)],
    'G': [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)],
    'D': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 3), (5, 2)],
    'Dm': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 3), (5, 1)],
    'E': [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    'Em': [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    'A': [(0, -1), (1, 0), (2, 2), (3, 2), (4, 2), (5, 0)],
    'F': [(0, 1), (1, 3), (2, 3), (3, 2), (4, 1), (5, 1)],  # Barre chord
    'Bm': [(0, -1), (1, 2), (2, 4), (3, 4), (4, 3), (5, 2)],  # Barre chord
    'B7': [(0, -1), (1, 2), (2, 1), (3, 2), (4, 0), (5, 2)],
    'G7': [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 1)],
    'C7': [(0, -1), (1, 3), (2, 2), (3, 3), (4, 1), (5, 0)],
    'D7': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 1), (5, 2)],
    'A7': [(0, -1), (1, 0), (2, 2), (3, 0), (4, 2), (5, 0)],
    'E7': [(0, 0), (1, 2), (2, 0), (3, 1), (4, 0), (5, 0)],
}


@dataclass
class Note:
    """Represents a detected note"""
    midi: int
    start_time: float
    duration: float
    confidence: float
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)
    
    @property
    def note_class(self) -> int:
        """Return pitch class (0-11)"""
        return self.midi % 12
    
    @property
    def note_name(self) -> str:
        """Return just the note name without octave"""
        return NOTE_NAMES[self.midi % 12]
    
    @property
    def frequency(self) -> float:
        return 440 * (2 ** ((self.midi - 69) / 12))


@dataclass
class TabNote:
    """A note on the guitar fretboard"""
    string: int  # 0-5 (low E to high e)
    fret: int    # 0-24
    start_time: float
    duration: float
    
    def __str__(self):
        return f"String {STRING_NAMES[self.string]}, Fret {self.fret}"


@dataclass
class Chord:
    """Represents a detected chord"""
    name: str  # e.g., "Am", "G", "C7"
    root: int  # MIDI pitch class (0-11)
    notes: List[Note]  # Notes that make up this chord
    start_time: float
    duration: float
    quality: str = ""  # "maj", "m", "7", etc.
    is_barre: bool = False
    confidence: float = 0.0
    
    @property
    def root_name(self) -> str:
        return NOTE_NAMES[self.root]
    
    def __str__(self) -> str:
        return self.name


def get_tuning(tuning_name: str) -> List[int]:
    """Get tuning by name, with helpful error message."""
    if tuning_name.lower() in TUNINGS:
        return TUNINGS[tuning_name.lower()]
    
    # Try to parse custom tuning (e.g., "38,45,50,55,59,64")
    try:
        notes = [int(n.strip()) for n in tuning_name.split(',')]
        if len(notes) == 6:
            return notes
    except ValueError:
        pass
    
    available = ', '.join(TUNINGS.keys())
    raise ValueError(f"Unknown tuning '{tuning_name}'. Available: {available}\n"
                    f"Or provide custom MIDI notes: e.g., '38,45,50,55,59,64'")


def midi_to_fret_options(midi_note: int, tuning: List[int] = None) -> List[Tuple[int, int]]:
    """
    Given a MIDI note, return all possible (string, fret) combinations.
    Returns list of (string_index, fret_number) tuples.
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    options = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= NUM_FRETS:
            options.append((string_idx, fret))
    return options


def choose_best_fret_position(
    note: Note,
    prev_position: Optional[Tuple[int, int]],
    tuning: List[int] = None
) -> Optional[TabNote]:
    """
    Choose the best fret position for a note, considering:
    - Playability (prefer lower frets for beginners)
    - Hand position continuity (stay close to previous note)
    - String preference (middle strings often easier)
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    options = midi_to_fret_options(note.midi, tuning)
    
    if not options:
        return None  # Note out of guitar range
    
    if len(options) == 1:
        string, fret = options[0]
        return TabNote(string, fret, note.start_time, note.duration)
    
    # Score each option
    scored_options = []
    for string, fret in options:
        score = 0
        
        # Prefer lower frets (easier to play) but not too aggressively
        if fret <= 5:
            score += 3  # Bonus for first position
        elif fret <= 12:
            score += 1  # Small bonus for comfortable range
        score -= fret * 0.2
        
        # Prefer middle strings (easier access)
        string_preference = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5]  # Middle strings preferred
        score += string_preference[string] * 3
        
        # If we have a previous position, prefer staying close
        if prev_position:
            prev_string, prev_fret = prev_position
            # Penalize large jumps
            fret_distance = abs(fret - prev_fret)
            string_distance = abs(string - prev_string)
            
            # Heavy penalty for large fret jumps
            if fret_distance > 4:
                score -= fret_distance * 0.5
            else:
                score -= fret_distance * 0.2
            
            # Prefer staying on same or adjacent strings
            score -= string_distance * 0.8
        
        scored_options.append((score, string, fret))
    
    # Sort by score (highest first)
    scored_options.sort(reverse=True)
    _, best_string, best_fret = scored_options[0]
    
    return TabNote(best_string, best_fret, note.start_time, note.duration)


def apply_median_filter(pitches: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to reduce pitch jitter."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    
    # Only filter valid (non-zero) pitches
    valid_mask = pitches > 0
    if not np.any(valid_mask):
        return pitches
    
    # Apply median filter
    filtered = medfilt(pitches, kernel_size=kernel_size)
    
    # Preserve zero values (no pitch detected)
    result = np.where(valid_mask, filtered, pitches)
    return result


def smooth_confidence(confidence: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooth confidence values using a moving average."""
    if len(confidence) < window_size:
        return confidence
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(confidence, kernel, mode='same')
    return smoothed


def detect_pitch_pyin(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect pitch using pYIN algorithm - better for monophonic sources.
    Returns: (f0, voiced_flag, voiced_probs)
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=GUITAR_MIN_HZ,
        fmax=GUITAR_MAX_HZ,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0
    )
    return f0, voiced_flag, voiced_probs


def detect_pitch_piptrack(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect pitch using piptrack algorithm.
    Returns: (pitches_per_frame, magnitudes_per_frame)
    """
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=GUITAR_MIN_HZ,
        fmax=GUITAR_MAX_HZ
    )
    return pitches, magnitudes


def extract_harmonic_component(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Separate harmonic component from percussive for better pitch detection.
    Guitar strings have strong harmonic content.
    """
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
    return y_harmonic


def suppress_harmonics(midi_notes: List[int], min_interval: float = 0.05) -> List[int]:
    """
    Suppress octave harmonics - guitar often shows strong 2nd/3rd harmonics.
    Keep the fundamental frequency (lowest note in close timing).
    """
    if len(midi_notes) <= 1:
        return midi_notes
    
    filtered = []
    for note in midi_notes:
        is_harmonic = False
        for existing in filtered:
            # Check if this note is an octave (12 semitones) above existing
            diff = note - existing
            if diff in [12, 19, 24]:  # Octave, octave+fifth, 2 octaves
                is_harmonic = True
                break
        if not is_harmonic:
            filtered.append(note)
    
    return filtered


def detect_notes_from_audio(
    audio_path: str,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    confidence_threshold: float = 0.5,
    pitch_method: str = 'pyin',
    use_harmonic_separation: bool = True,
    median_filter_size: int = 5,
    tuning: List[int] = None
) -> List[Note]:
    """
    Detect notes from audio file using advanced pitch detection.
    
    Args:
        audio_path: Path to audio file
        hop_length: Hop length for STFT
        min_note_duration: Minimum note duration in seconds
        confidence_threshold: Minimum confidence for note detection
        pitch_method: 'pyin' (better for monophonic) or 'piptrack'
        use_harmonic_separation: Whether to use HPSS
        median_filter_size: Size of median filter for pitch smoothing
        tuning: Guitar tuning (MIDI notes)
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Optional: Extract harmonic component for cleaner pitch detection
    if use_harmonic_separation:
        print("Separating harmonic component...")
        y_pitch = extract_harmonic_component(y, sr)
    else:
        y_pitch = y
    
    print(f"Detecting pitches using {pitch_method}...")
    
    if pitch_method == 'pyin':
        # pYIN is better for monophonic sources like guitar
        f0, voiced_flag, voiced_probs = detect_pitch_pyin(y_pitch, sr, hop_length)
        
        # Apply median filter to reduce jitter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        # Smooth confidence values
        confidence = smooth_confidence(voiced_probs, window_size=5)
        
    else:  # piptrack
        pitches, magnitudes = detect_pitch_piptrack(y_pitch, sr, hop_length)
        
        # Extract dominant pitch per frame
        f0 = np.zeros(pitches.shape[1])
        confidence = np.zeros(pitches.shape[1])
        
        for i in range(pitches.shape[1]):
            mag_slice = magnitudes[:, i]
            pitch_slice = pitches[:, i]
            
            if mag_slice.max() > 0:
                max_idx = mag_slice.argmax()
                f0[i] = pitch_slice[max_idx]
                confidence[i] = mag_slice[max_idx] / (magnitudes.max() + 1e-10)
        
        # Apply median filter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        confidence = smooth_confidence(confidence, window_size=5)
    
    print("Detecting onsets with backtracking...")
    # Detect note onsets with backtracking for better timing
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # Backtrack to find true onset
        units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    print(f"Found {len(onset_times)} onsets")
    
    notes = []
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    for i, onset_time in enumerate(onset_times):
        # Find the frame closest to this onset
        onset_frame = onset_frames[i] if i < len(onset_frames) else int(onset_time * sr / hop_length)
        
        if onset_frame >= len(f0):
            continue
        
        # Get pitch and confidence at onset
        # Look at a small window around onset for stability
        window_start = max(0, onset_frame)
        window_end = min(len(f0), onset_frame + 5)
        
        pitch_window = f0[window_start:window_end]
        conf_window = confidence[window_start:window_end]
        
        # Find valid pitches in window
        valid_mask = pitch_window > 0
        if not np.any(valid_mask):
            continue
        
        # Use the pitch with highest confidence in the window
        valid_pitches = pitch_window[valid_mask]
        valid_confs = conf_window[valid_mask]
        
        if len(valid_confs) == 0:
            continue
            
        best_idx = np.argmax(valid_confs)
        pitch_hz = valid_pitches[best_idx]
        note_confidence = valid_confs[best_idx]
        
        if pitch_hz <= 0 or note_confidence < confidence_threshold:
            continue
        
        # Convert Hz to MIDI note number
        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
        
        # Sanity check: is this within guitar range?
        # Standard guitar: E2 (40) to about E6 (88) at highest fret
        if midi_note < 36 or midi_note > 90:
            continue
        
        # Estimate duration (until next onset or end)
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - onset_time
        else:
            duration = librosa.get_duration(y=y, sr=sr) - onset_time
        
        # Filter by duration
        if duration < min_note_duration:
            continue
        
        notes.append(Note(
            midi=midi_note,
            start_time=onset_time,
            duration=duration,
            confidence=float(note_confidence)
        ))
    
    # Post-process: suppress obvious harmonics
    print("Filtering harmonics...")
    notes = filter_harmonic_notes(notes)
    
    print(f"Detected {len(notes)} notes")
    return notes


def filter_harmonic_notes(notes: List[Note], time_threshold: float = 0.05) -> List[Note]:
    """
    Filter out notes that are likely harmonics of other notes.
    If two notes start at nearly the same time and one is an octave/fifth above,
    keep only the lower (fundamental) one.
    """
    if len(notes) <= 1:
        return notes
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.midi))
    filtered = []
    
    i = 0
    while i < len(sorted_notes):
        current = sorted_notes[i]
        
        # Find all notes starting within time_threshold
        group = [current]
        j = i + 1
        while j < len(sorted_notes) and sorted_notes[j].start_time - current.start_time < time_threshold:
            group.append(sorted_notes[j])
            j += 1
        
        if len(group) == 1:
            filtered.append(current)
        else:
            # Keep notes that aren't harmonics of others in the group
            group_midis = [n.midi for n in group]
            kept = []
            
            for note in group:
                is_harmonic = False
                for other in group:
                    if other.midi < note.midi:
                        diff = note.midi - other.midi
                        # Common harmonic intervals: octave (12), octave+fifth (19), 2 octaves (24)
                        if diff in [12, 19, 24, 28, 31]:
                            is_harmonic = True
                            break
                
                if not is_harmonic:
                    kept.append(note)
            
            filtered.extend(kept if kept else [group[0]])  # Keep at least one note
        
        i = j if j > i else i + 1
    
    return filtered


def notes_to_tabs(notes: List[Note], tuning: List[int] = None) -> List[TabNote]:
    """Convert detected notes to guitar tab positions."""
    if tuning is None:
        tuning = TUNINGS['standard']
    
    tab_notes = []
    prev_position = None
    
    for note in notes:
        tab_note = choose_best_fret_position(note, prev_position, tuning)
        if tab_note:
            tab_notes.append(tab_note)
            prev_position = (tab_note.string, tab_note.fret)
    
    return tab_notes


# ============================================================================
# CHORD DETECTION
# ============================================================================

def group_simultaneous_notes(
    notes: List[Note],
    time_threshold: float = CHORD_TIME_THRESHOLD
) -> List[List[Note]]:
    """
    Group notes that play simultaneously (within time_threshold).
    Returns list of note groups.
    """
    if not notes:
        return []
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    groups = []
    current_group = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        # Check if note is within threshold of first note in current group
        if note.start_time - current_group[0].start_time <= time_threshold:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    
    # Don't forget the last group
    if current_group:
        groups.append(current_group)
    
    return groups


def identify_chord_from_notes(notes: List[Note]) -> Optional[Chord]:
    """
    Identify a chord from a group of simultaneous notes.
    Returns None if no chord pattern matches.
    """
    if len(notes) < 2:
        return None
    
    # Get unique pitch classes
    pitch_classes = sorted(set(n.note_class for n in notes))
    
    if len(pitch_classes) < 2:
        return None
    
    # Calculate confidence based on number of notes
    avg_confidence = sum(n.confidence for n in notes) / len(notes)
    
    # Try each pitch class as potential root
    best_match = None
    best_score = 0
    
    for potential_root in pitch_classes:
        # Normalize intervals relative to this root
        intervals = sorted(set((pc - potential_root) % 12 for pc in pitch_classes))
        
        # Try to match against known chord types
        for chord_suffix, chord_intervals in CHORD_INTERVALS.items():
            # Calculate match score
            matched = sum(1 for i in intervals if i in chord_intervals)
            total = len(chord_intervals)
            
            # Score considers both coverage and accuracy
            if matched >= 2:  # Need at least 2 matching intervals
                score = (matched / total) * (matched / len(intervals))
                
                if score > best_score:
                    best_score = score
                    root_name = NOTE_NAMES[potential_root]
                    chord_name = root_name + chord_suffix if chord_suffix else root_name
                    
                    best_match = {
                        'name': chord_name,
                        'root': potential_root,
                        'quality': chord_suffix,
                        'score': score
                    }
    
    if best_match and best_score >= 0.5:  # Minimum match threshold
        start_time = min(n.start_time for n in notes)
        max_end = max(n.start_time + n.duration for n in notes)
        duration = max_end - start_time
        
        return Chord(
            name=best_match['name'],
            root=best_match['root'],
            notes=notes,
            start_time=start_time,
            duration=duration,
            quality=best_match['quality'],
            confidence=avg_confidence * best_score
        )
    
    return None


def detect_chord_shape(chord: Chord, tab_notes: List[TabNote]) -> Optional[str]:
    """
    Detect if a chord matches a known open or barre chord shape.
    Returns the shape name if found.
    """
    # Find tab notes that correspond to this chord's timing
    chord_tab_notes = [
        tn for tn in tab_notes
        if abs(tn.start_time - chord.start_time) <= CHORD_TIME_THRESHOLD
    ]
    
    if len(chord_tab_notes) < 2:
        return None
    
    # Create a fret pattern from the tab notes
    fret_pattern = {tn.string: tn.fret for tn in chord_tab_notes}
    
    # Check against known shapes
    for shape_name, shape_frets in OPEN_CHORD_SHAPES.items():
        # Skip if root doesn't match
        shape_root = shape_name.replace('m', '').replace('7', '').replace('sus', '').replace('dim', '')
        if shape_root and shape_root[0] != chord.root_name[0]:
            continue
        
        # Count matching frets
        matches = 0
        total_played = 0
        
        for string, expected_fret in shape_frets:
            if expected_fret == -1:  # Muted string
                continue
            total_played += 1
            if string in fret_pattern and fret_pattern[string] == expected_fret:
                matches += 1
        
        if total_played > 0 and matches / total_played >= 0.6:
            # Check if it's a barre chord
            is_barre = any(
                sum(1 for s, f in shape_frets if f == fret and f > 0) >= 2
                for fret in range(1, 13)
            )
            chord.is_barre = is_barre
            return shape_name
    
    return None


def detect_chords(
    notes: List[Note],
    time_threshold: float = CHORD_TIME_THRESHOLD
) -> List[Chord]:
    """
    Detect chords from a list of notes.
    Groups simultaneous notes and identifies chord patterns.
    """
    groups = group_simultaneous_notes(notes, time_threshold)
    chords = []
    
    for group in groups:
        if len(group) >= 2:  # Need at least 2 notes for a chord
            chord = identify_chord_from_notes(group)
            if chord:
                chords.append(chord)
    
    return chords


def generate_chord_diagram(chord_name: str) -> str:
    """
    Generate ASCII art chord diagram for a chord.
    Returns multi-line string with the diagram.
    """
    if chord_name not in OPEN_CHORD_SHAPES:
        # Try to find similar chord
        base_name = chord_name.replace('m', '').replace('7', '')
        if base_name + 'm' in OPEN_CHORD_SHAPES and 'm' in chord_name:
            chord_name = base_name + 'm'
        elif base_name in OPEN_CHORD_SHAPES:
            chord_name = base_name
        else:
            return f"  {chord_name}\n  (no diagram)"
    
    shape = OPEN_CHORD_SHAPES[chord_name]
    
    # Find fret range
    frets_played = [f for s, f in shape if f > 0]
    if not frets_played:
        min_fret, max_fret = 0, 3
    else:
        min_fret = min(frets_played)
        max_fret = max(frets_played)
    
    # Adjust for barre chords
    start_fret = 0 if max_fret <= 4 else min_fret - 1
    
    lines = [f"  {chord_name}"]
    
    # Top indicators (open/muted strings)
    top_line = "  "
    for string in range(6):
        string_fret = shape[string][1]
        if string_fret == -1:
            top_line += "x"
        elif string_fret == 0:
            top_line += "o"
        else:
            top_line += " "
    lines.append(top_line)
    
    # Nut or fret number
    if start_fret == 0:
        lines.append("  ╔═════╗")
    else:
        lines.append(f" {start_fret}├─────┤")
    
    # Draw frets
    for fret in range(start_fret + 1, start_fret + 5):
        fret_line = "  │"
        for string in range(6):
            string_fret = shape[string][1]
            if string_fret == fret:
                fret_line += "●"
            else:
                fret_line += "│"
        fret_line += "│"
        lines.append(fret_line)
        if fret < start_fret + 4:
            lines.append("  ├─────┤")
        else:
            lines.append("  └─────┘")
    
    # String names
    lines.append("  E A D G B e")
    
    return '\n'.join(lines)


def generate_all_chord_diagrams(chords: List[Chord]) -> str:
    """Generate chord diagrams for all unique chords."""
    unique_chords = list(set(c.name for c in chords))
    diagrams = []
    
    for chord_name in sorted(unique_chords):
        diagrams.append(generate_chord_diagram(chord_name))
    
    return '\n\n'.join(diagrams)


# ============================================================================
# TAB FORMATTING
# ============================================================================

def format_ascii_tab(
    tab_notes: List[TabNote],
    beats_per_line: int = 16,
    tuning: List[int] = None,
    chords: Optional[List[Chord]] = None
) -> str:
    """Format tab notes as ASCII guitar tablature with optional chord names above."""
    if not tab_notes:
        return "No notes detected!"
    
    if tuning is None:
        tuning = TUNINGS['standard']
    
    # Generate string names based on tuning
    string_names = []
    for midi in tuning:
        note_name = NOTE_NAMES[midi % 12]
        string_names.append(note_name)
    
    # Group notes by time (quantize to grid)
    time_resolution = 0.125  # 1/8 note at 120 BPM
    
    # Find total duration
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_positions = int(max_time / time_resolution) + 1
    
    # Create grid for each string
    grid = {i: ['-'] * num_positions for i in range(6)}
    
    # Create chord label grid
    chord_grid = [' '] * num_positions
    if chords:
        for chord in chords:
            pos = int(chord.start_time / time_resolution)
            if pos < num_positions:
                # Place chord name, handling overlaps
                chord_name = chord.name
                for i, char in enumerate(chord_name):
                    if pos + i < num_positions and chord_grid[pos + i] == ' ':
                        chord_grid[pos + i] = char
    
    for note in tab_notes:
        pos = int(note.start_time / time_resolution)
        if pos < num_positions:
            fret_str = str(note.fret) if note.fret < 10 else f"({note.fret})"
            grid[note.string][pos] = fret_str
    
    # Format output
    lines = []
    for start in range(0, num_positions, beats_per_line):
        end = min(start + beats_per_line, num_positions)
        
        # Add chord labels line if we have chords
        if chords:
            chord_line = ''.join(chord_grid[start:end])
            if chord_line.strip():  # Only add if there are chords
                lines.append(f"  {chord_line}")
        
        for string in range(5, -1, -1):  # High e to low E
            name = string_names[string] if string < len(string_names) else STRING_NAMES[string]
            notes_str = ''.join(grid[string][start:end])
            lines.append(f"{name}|{notes_str}|")
        lines.append("")  # Empty line between measures
    
    return '\n'.join(lines)


def download_youtube_audio(url: str, output_dir: str = None) -> str:
    """Download audio from YouTube URL using yt-dlp."""
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--print", "after_move:filepath",  # Print final path
        url
    ]
    
    print(f"📥 Downloading audio from YouTube...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    # Get the output file path from stdout
    output_path = result.stdout.strip().split('\n')[-1]
    print(f"✅ Downloaded: {output_path}")
    
    return output_path


def is_youtube_url(s: str) -> bool:
    """Check if string is a YouTube URL."""
    return any(domain in s for domain in ['youtube.com', 'youtu.be', 'youtube-nocookie.com'])


def export_guitar_pro(
    tab_notes: List[TabNote],
    output_path: str,
    title: str = "Generated Tab",
    artist: str = "Guitar Tab Generator",
    tempo: int = 120,
    tuning: List[int] = None
) -> bool:
    """
    Export tab notes to Guitar Pro format (GP5).
    
    Args:
        tab_notes: List of TabNote objects
        output_path: Path to save the GP5 file
        title: Song title
        artist: Artist name
        tempo: Tempo in BPM
        tuning: Guitar tuning as MIDI note numbers
        
    Returns:
        True if successful, False otherwise
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    if not HAS_GUITARPRO:
        print("❌ pyguitarpro not installed. Install with: pip install pyguitarpro")
        return False
    
    if not tab_notes:
        print("❌ No notes to export")
        return False
    
    # Create song
    song = guitarpro.Song()
    song.title = title
    song.artist = artist
    song.tempo = guitarpro.models.MixTableItem(value=tempo)
    
    # Create track
    track = guitarpro.Track(song=song)
    track.name = "Guitar"
    track.fretCount = NUM_FRETS
    
    # Set up guitar strings (standard tuning)
    track.strings = []
    for i, midi_note in enumerate(reversed(tuning)):  # GP uses high to low
        string = guitarpro.GuitarString(number=i + 1, value=midi_note)
        track.strings.append(string)
    
    # Time resolution - convert time to beats
    time_per_beat = 60.0 / tempo  # seconds per beat
    time_per_measure = time_per_beat * 4  # 4/4 time
    
    # Group notes into measures
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_measures = int(max_time / time_per_measure) + 1
    
    # Create measure headers
    for i in range(num_measures):
        header = guitarpro.MeasureHeader()
        header.number = i + 1
        header.start = int(i * 960 * 4)  # 960 ticks per quarter note
        header.tempo = guitarpro.models.MixTableItem(value=tempo) if i == 0 else None
        header.timeSignature = guitarpro.TimeSignature()
        header.timeSignature.numerator = 4
        header.timeSignature.denominator = guitarpro.Duration()
        header.timeSignature.denominator.value = 4
        song.measureHeaders.append(header)
    
    # Create measures for the track
    for header in song.measureHeaders:
        measure = guitarpro.Measure(track=track, header=header)
        
        # Each measure has voices (we use voice 0)
        voice = measure.voices[0]
        
        # Find notes in this measure
        measure_start = (header.number - 1) * time_per_measure
        measure_end = measure_start + time_per_measure
        
        measure_notes = [n for n in tab_notes 
                        if measure_start <= n.start_time < measure_end]
        
        if not measure_notes:
            # Add a rest beat if no notes
            beat = guitarpro.Beat(voice)
            beat.status = guitarpro.BeatStatus.rest
            beat.duration = guitarpro.Duration()
            beat.duration.value = 1  # Whole note rest
            voice.beats.append(beat)
        else:
            # Group notes by time (for chords)
            note_groups = {}
            for tab_note in measure_notes:
                # Quantize to 16th notes
                beat_time = round((tab_note.start_time - measure_start) / (time_per_beat / 4)) * (time_per_beat / 4)
                if beat_time not in note_groups:
                    note_groups[beat_time] = []
                note_groups[beat_time].append(tab_note)
            
            # Create beats for each time position
            for beat_time in sorted(note_groups.keys()):
                beat = guitarpro.Beat(voice)
                
                # Determine duration based on note duration
                avg_duration = sum(n.duration for n in note_groups[beat_time]) / len(note_groups[beat_time])
                beat.duration = guitarpro.Duration()
                
                # Map duration to note value
                if avg_duration >= time_per_beat * 2:
                    beat.duration.value = 2  # Half note
                elif avg_duration >= time_per_beat:
                    beat.duration.value = 4  # Quarter note
                elif avg_duration >= time_per_beat / 2:
                    beat.duration.value = 8  # Eighth note
                else:
                    beat.duration.value = 16  # Sixteenth note
                
                # Add notes
                for tab_note in note_groups[beat_time]:
                    note = guitarpro.Note(beat)
                    # Guitar Pro uses 1-indexed strings, high to low
                    note.string = 6 - tab_note.string  # Convert from our 0-indexed low-to-high
                    note.value = tab_note.fret
                    note.velocity = 95  # Default velocity
                    beat.notes.append(note)
                
                voice.beats.append(beat)
        
        track.measures.append(measure)
    
    song.tracks.append(track)
    
    # Write file
    try:
        guitarpro.write(song, output_path, version=(5, 1, 0))
        print(f"✅ Exported Guitar Pro file: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write GP5 file: {e}")
        return False


def export_musicxml(
    tab_notes: List[TabNote],
    output_path: str,
    title: str = "Generated Tab",
    composer: str = "Guitar Tab Generator",
    tempo: int = 120,
    tuning: List[int] = None
) -> bool:
    """
    Export tab notes to MusicXML format.
    
    MusicXML is a universal format supported by:
    - Guitar Pro
    - MuseScore
    - Finale
    - Sibelius
    - Many other music notation apps
    
    Args:
        tab_notes: List of TabNote objects
        output_path: Path to save the MusicXML file
        title: Song title
        composer: Composer/artist name
        tempo: Tempo in BPM
        tuning: Guitar tuning as MIDI note numbers
        
    Returns:
        True if successful, False otherwise
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    if not tab_notes:
        print("❌ No notes to export")
        return False
    
    # MIDI note to pitch mapping
    def midi_to_pitch(midi: int) -> Tuple[str, int, int]:
        """Convert MIDI note to (step, alter, octave)"""
        note_steps = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
        note_alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        pitch_class = midi % 12
        octave = (midi // 12) - 1
        
        return note_steps[pitch_class], note_alters[pitch_class], octave
    
    # Create root element
    root = ET.Element('score-partwise', version='4.0')
    
    # Work info
    work = ET.SubElement(root, 'work')
    work_title = ET.SubElement(work, 'work-title')
    work_title.text = title
    
    # Identification
    identification = ET.SubElement(root, 'identification')
    creator = ET.SubElement(identification, 'creator', type='composer')
    creator.text = composer
    encoding = ET.SubElement(identification, 'encoding')
    software = ET.SubElement(encoding, 'software')
    software.text = 'Guitar Tab Generator'
    
    # Part list
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    part_name = ET.SubElement(score_part, 'part-name')
    part_name.text = 'Guitar'
    
    # Score instrument for tablature
    score_inst = ET.SubElement(score_part, 'score-instrument', id='P1-I1')
    inst_name = ET.SubElement(score_inst, 'instrument-name')
    inst_name.text = 'Acoustic Guitar'
    
    # Part content
    part = ET.SubElement(root, 'part', id='P1')
    
    # Time and beat calculations
    time_per_beat = 60.0 / tempo
    time_per_measure = time_per_beat * 4  # 4/4 time
    divisions = 4  # Divisions per quarter note (allows 16th notes)
    
    # Group notes by measure
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_measures = int(max_time / time_per_measure) + 1
    
    for measure_num in range(1, num_measures + 1):
        measure = ET.SubElement(part, 'measure', number=str(measure_num))
        
        # Attributes for first measure
        if measure_num == 1:
            attributes = ET.SubElement(measure, 'attributes')
            
            div_elem = ET.SubElement(attributes, 'divisions')
            div_elem.text = str(divisions)
            
            # Key signature (C major)
            key = ET.SubElement(attributes, 'key')
            fifths = ET.SubElement(key, 'fifths')
            fifths.text = '0'
            
            # Time signature (4/4)
            time = ET.SubElement(attributes, 'time')
            beats = ET.SubElement(time, 'beats')
            beats.text = '4'
            beat_type = ET.SubElement(time, 'beat-type')
            beat_type.text = '4'
            
            # Clef (TAB)
            clef = ET.SubElement(attributes, 'clef')
            sign = ET.SubElement(clef, 'sign')
            sign.text = 'TAB'
            line = ET.SubElement(clef, 'line')
            line.text = '5'
            
            # Staff details (6 string guitar)
            staff_details = ET.SubElement(attributes, 'staff-details')
            staff_lines = ET.SubElement(staff_details, 'staff-lines')
            staff_lines.text = '6'
            
            # Tuning
            for string_num, midi_note in enumerate(reversed(tuning), 1):
                staff_tuning = ET.SubElement(staff_details, 'staff-tuning', line=str(string_num))
                step, alter, octave = midi_to_pitch(midi_note)
                tuning_step = ET.SubElement(staff_tuning, 'tuning-step')
                tuning_step.text = step
                if alter:
                    tuning_alter = ET.SubElement(staff_tuning, 'tuning-alter')
                    tuning_alter.text = str(alter)
                tuning_octave = ET.SubElement(staff_tuning, 'tuning-octave')
                tuning_octave.text = str(octave)
            
            # Tempo direction
            direction = ET.SubElement(measure, 'direction', placement='above')
            direction_type = ET.SubElement(direction, 'direction-type')
            metronome = ET.SubElement(direction_type, 'metronome')
            beat_unit = ET.SubElement(metronome, 'beat-unit')
            beat_unit.text = 'quarter'
            per_minute = ET.SubElement(metronome, 'per-minute')
            per_minute.text = str(tempo)
            sound = ET.SubElement(direction, 'sound', tempo=str(tempo))
        
        # Find notes in this measure
        measure_start = (measure_num - 1) * time_per_measure
        measure_end = measure_start + time_per_measure
        
        measure_notes = [n for n in tab_notes 
                        if measure_start <= n.start_time < measure_end]
        
        if not measure_notes:
            # Add whole rest
            note_elem = ET.SubElement(measure, 'note')
            rest = ET.SubElement(note_elem, 'rest')
            duration = ET.SubElement(note_elem, 'duration')
            duration.text = str(divisions * 4)  # Whole note
            ntype = ET.SubElement(note_elem, 'type')
            ntype.text = 'whole'
        else:
            # Group notes by time (for chords)
            note_groups = {}
            for tab_note in measure_notes:
                # Quantize to 16th notes
                quantized = round((tab_note.start_time - measure_start) / (time_per_beat / 4))
                if quantized not in note_groups:
                    note_groups[quantized] = []
                note_groups[quantized].append(tab_note)
            
            # Track position for rests
            current_position = 0
            
            for quantized_pos in sorted(note_groups.keys()):
                # Add rest if there's a gap
                if quantized_pos > current_position:
                    gap_duration = quantized_pos - current_position
                    note_elem = ET.SubElement(measure, 'note')
                    rest = ET.SubElement(note_elem, 'rest')
                    dur = ET.SubElement(note_elem, 'duration')
                    dur.text = str(gap_duration)
                
                # Add notes at this position
                notes_at_pos = note_groups[quantized_pos]
                is_chord = len(notes_at_pos) > 1
                
                for i, tab_note in enumerate(notes_at_pos):
                    note_elem = ET.SubElement(measure, 'note')
                    
                    # Chord indication for simultaneous notes
                    if is_chord and i > 0:
                        ET.SubElement(note_elem, 'chord')
                    
                    # Calculate MIDI pitch from string and fret
                    midi_pitch = tuning[tab_note.string] + tab_note.fret
                    step, alter, octave = midi_to_pitch(midi_pitch)
                    
                    pitch = ET.SubElement(note_elem, 'pitch')
                    step_elem = ET.SubElement(pitch, 'step')
                    step_elem.text = step
                    if alter:
                        alter_elem = ET.SubElement(pitch, 'alter')
                        alter_elem.text = str(alter)
                    octave_elem = ET.SubElement(pitch, 'octave')
                    octave_elem.text = str(octave)
                    
                    # Duration
                    dur_beats = max(1, round(tab_note.duration / time_per_beat * divisions))
                    duration_elem = ET.SubElement(note_elem, 'duration')
                    duration_elem.text = str(min(dur_beats, divisions * 4))  # Cap at whole note
                    
                    # Note type
                    ntype = ET.SubElement(note_elem, 'type')
                    if dur_beats >= divisions * 4:
                        ntype.text = 'whole'
                    elif dur_beats >= divisions * 2:
                        ntype.text = 'half'
                    elif dur_beats >= divisions:
                        ntype.text = 'quarter'
                    elif dur_beats >= divisions // 2:
                        ntype.text = 'eighth'
                    else:
                        ntype.text = '16th'
                    
                    # Notations with technical (fret/string)
                    notations = ET.SubElement(note_elem, 'notations')
                    technical = ET.SubElement(notations, 'technical')
                    
                    string_elem = ET.SubElement(technical, 'string')
                    string_elem.text = str(6 - tab_note.string)  # MusicXML: 1 = high e
                    
                    fret_elem = ET.SubElement(technical, 'fret')
                    fret_elem.text = str(tab_note.fret)
                
                # Update position
                current_position = quantized_pos + dur_beats
    
    # Format XML nicely
    xml_str = ET.tostring(root, encoding='unicode')
    
    # Add XML declaration and DOCTYPE
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    
    try:
        # Pretty print
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        # Remove extra declaration added by minidom
        pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doctype)
            f.write(pretty_xml)
        
        print(f"✅ Exported MusicXML file: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write MusicXML file: {e}")
        return False


def get_export_extension(format_name: str) -> str:
    """Get the appropriate file extension for an export format."""
    extensions = {
        'gp5': '.gp5',
        'gp': '.gp5',
        'guitarpro': '.gp5',
        'musicxml': '.musicxml',
        'xml': '.musicxml',
        'ascii': '.txt',
        'txt': '.txt',
    }
    return extensions.get(format_name.lower(), '.txt')


def main():
    parser = argparse.ArgumentParser(
        description='Generate guitar tabs from audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tunings:
  standard       - E A D G B E (default)
  drop_d         - D A D G B E
  drop_c         - C G C F A D
  half_step_down - Eb Ab Db Gb Bb Eb
  full_step_down - D G C F A D
  open_d         - D A D F# A D
  open_g         - D G D G B D
  dadgad         - D A D G A D

Or specify custom tuning as comma-separated MIDI notes:
  --tuning "38,45,50,55,59,64"

Export formats:
  ascii     - Plain text tablature (default)
  gp5       - Guitar Pro 5 format (.gp5)
  musicxml  - MusicXML format (.musicxml) - universal format

Examples:
  %(prog)s song.mp3
  %(prog)s song.mp3 -o tabs.txt
  %(prog)s song.mp3 -o tabs.gp5 --format gp5
  %(prog)s "https://youtube.com/watch?v=..." --format musicxml -o song.musicxml
        """
    )
    parser.add_argument('audio_file', help='Path to audio file OR YouTube URL')
    parser.add_argument('--output', '-o', help='Output file for tabs')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                        help='Minimum confidence threshold (0-1, default: 0.3)')
    parser.add_argument('--tuning', '-t', default='standard',
                        help='Guitar tuning (see list below, default: standard)')
    parser.add_argument('--format', '-f', 
                        choices=['ascii', 'gp5', 'gp', 'guitarpro', 'musicxml', 'xml'],
                        default='ascii',
                        help='Output format: ascii (default), gp5/guitarpro, musicxml/xml')
    parser.add_argument('--title', default=None,
                        help='Song title (for GP5/MusicXML export)')
    parser.add_argument('--artist', '-a', default='Guitar Tab Generator',
                        help='Artist name (for GP5/MusicXML export)')
    parser.add_argument('--tempo', type=int, default=120,
                        help='Tempo in BPM (default: 120)')
    parser.add_argument('--pitch-method', '-p', choices=['pyin', 'piptrack'], default='pyin',
                        help='Pitch detection method (pyin=better for monophonic, default: pyin)')
    parser.add_argument('--no-harmonic-separation', action='store_true',
                        help='Disable harmonic/percussive separation')
    parser.add_argument('--median-filter', '-m', type=int, default=5,
                        help='Median filter size for pitch smoothing (0=disabled, default: 5)')
    parser.add_argument('--min-duration', type=float, default=0.05,
                        help='Minimum note duration in seconds (default: 0.05)')
    parser.add_argument('--chords', action='store_true',
                        help='Enable chord detection')
    parser.add_argument('--chord-diagrams', action='store_true',
                        help='Show ASCII chord diagrams')
    parser.add_argument('--chord-threshold', type=float, default=CHORD_TIME_THRESHOLD,
                        help=f'Time window for simultaneous notes in seconds (default: {CHORD_TIME_THRESHOLD})')
    
    args = parser.parse_args()
    
    # Parse tuning
    try:
        tuning = get_tuning(args.tuning)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Handle YouTube URLs
    audio_path = args.audio_file
    cleanup_file = False
    
    if is_youtube_url(audio_path):
        audio_path = download_youtube_audio(audio_path)
        cleanup_file = True
    
    # Determine title from filename if not specified
    title = args.title or os.path.splitext(os.path.basename(audio_path))[0]
    
    print("🎸 Guitar Tab Generator (Enhanced)")
    print("=" * 40)
    print(f"Tuning: {args.tuning} {tuning}")
    print(f"Pitch method: {args.pitch_method}")
    print(f"Harmonic separation: {'enabled' if not args.no_harmonic_separation else 'disabled'}")
    print(f"Median filter: {args.median_filter if args.median_filter > 0 else 'disabled'}")
    print()
    
    # Detect notes
    notes = detect_notes_from_audio(
        audio_path,
        confidence_threshold=args.confidence,
        pitch_method=args.pitch_method,
        use_harmonic_separation=not args.no_harmonic_separation,
        median_filter_size=args.median_filter,
        min_note_duration=args.min_duration,
        tuning=tuning
    )
    
    if not notes:
        print("No notes detected! Try lowering confidence threshold with -c 0.1")
        sys.exit(1)
    
    # Print detected notes
    print("\n📝 Detected Notes:")
    for note in notes[:20]:  # Show first 20
        print(f"  {note.name:4} at {note.start_time:.2f}s (confidence: {note.confidence:.2f})")
    if len(notes) > 20:
        print(f"  ... and {len(notes) - 20} more")
    
    # Detect chords if enabled
    chords = None
    if args.chords:
        print("\n🎵 Detecting chords...")
        chords = detect_chords(notes, time_threshold=args.chord_threshold)
        
        if chords:
            print(f"\n🎶 Detected {len(chords)} chords:")
            unique_chords = {}
            for chord in chords:
                if chord.name not in unique_chords:
                    unique_chords[chord.name] = 0
                unique_chords[chord.name] += 1
            
            for chord_name, count in sorted(unique_chords.items()):
                print(f"  {chord_name}: {count}x")
            
            # Show chord progression
            print("\n📋 Chord Progression:")
            progression = [c.name for c in chords[:16]]  # First 16 chords
            print(f"  {' → '.join(progression)}")
            if len(chords) > 16:
                print(f"  ... and {len(chords) - 16} more")
        else:
            print("  No chord patterns detected")
    
    # Convert to tabs
    tab_notes = notes_to_tabs(notes, tuning)
    
    # Format as ASCII tab (always show preview, with chords if detected)
    tab_output = format_ascii_tab(tab_notes, tuning=tuning, chords=chords)
    
    # Show chord diagrams if requested
    chord_diagrams = ""
    if args.chord_diagrams and chords:
        chord_diagrams = generate_all_chord_diagrams(chords)
        print("\n📊 Chord Diagrams:")
        print("-" * 40)
        print(chord_diagrams)
    
    print("\n🎼 Guitar Tablature:")
    print("-" * 40)
    print(tab_output)
    
    # Save to file if requested
    if args.output:
        format_name = args.format.lower()
        output_path = args.output
        
        # Add extension if needed
        if not os.path.splitext(output_path)[1]:
            output_path += get_export_extension(format_name)
        
        if format_name in ('gp5', 'gp', 'guitarpro'):
            if not HAS_GUITARPRO:
                print("\n⚠️  pyguitarpro not installed. Falling back to MusicXML...")
                format_name = 'musicxml'
                output_path = os.path.splitext(output_path)[0] + '.musicxml'
            else:
                success = export_guitar_pro(
                    tab_notes,
                    output_path,
                    title=title,
                    artist=args.artist,
                    tempo=args.tempo,
                    tuning=tuning
                )
                if not success:
                    sys.exit(1)
        
        if format_name in ('musicxml', 'xml'):
            success = export_musicxml(
                tab_notes,
                output_path,
                title=title,
                composer=args.artist,
                tempo=args.tempo,
                tuning=tuning
            )
            if not success:
                sys.exit(1)
        
        elif format_name in ('ascii', 'txt'):
            with open(output_path, 'w') as f:
                f.write(f"# Guitar Tab - {title}\n")
                f.write(f"# Generated from: {os.path.basename(audio_path)}\n")
                f.write(f"# Tuning: {args.tuning}\n")
                f.write(f"# Tempo: {args.tempo} BPM\n\n")
                
                if chords:
                    unique_chords = list(set(c.name for c in chords))
                    f.write(f"## Chords Used: {', '.join(sorted(unique_chords))}\n\n")
                
                if chord_diagrams:
                    f.write("## Chord Diagrams\n\n")
                    f.write("```\n")
                    f.write(chord_diagrams)
                    f.write("\n```\n\n")
                
                f.write("## Tablature\n\n")
                f.write("```\n")
                f.write(tab_output)
                f.write("\n```\n")
            print(f"\n✅ Saved to {output_path}")
    
    # Cleanup temp file if downloaded from YouTube
    if cleanup_file and os.path.exists(audio_path):
        os.remove(audio_path)
        print("🗑️ Cleaned up temp audio file")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
