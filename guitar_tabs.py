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


def format_ascii_tab(tab_notes: List[TabNote], beats_per_line: int = 16, tuning: List[int] = None) -> str:
    """Format tab notes as ASCII guitar tablature."""
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
    
    for note in tab_notes:
        pos = int(note.start_time / time_resolution)
        if pos < num_positions:
            fret_str = str(note.fret) if note.fret < 10 else f"({note.fret})"
            grid[note.string][pos] = fret_str
    
    # Format output
    lines = []
    for start in range(0, num_positions, beats_per_line):
        end = min(start + beats_per_line, num_positions)
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
        """
    )
    parser.add_argument('audio_file', help='Path to audio file OR YouTube URL')
    parser.add_argument('--output', '-o', help='Output file for tabs')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                        help='Minimum confidence threshold (0-1, default: 0.3)')
    parser.add_argument('--tuning', '-t', default='standard',
                        help='Guitar tuning (see list below, default: standard)')
    parser.add_argument('--pitch-method', '-p', choices=['pyin', 'piptrack'], default='pyin',
                        help='Pitch detection method (pyin=better for monophonic, default: pyin)')
    parser.add_argument('--no-harmonic-separation', action='store_true',
                        help='Disable harmonic/percussive separation')
    parser.add_argument('--median-filter', '-m', type=int, default=5,
                        help='Median filter size for pitch smoothing (0=disabled, default: 5)')
    parser.add_argument('--min-duration', type=float, default=0.05,
                        help='Minimum note duration in seconds (default: 0.05)')
    
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
    
    # Convert to tabs
    tab_notes = notes_to_tabs(notes, tuning)
    
    # Format as ASCII tab
    tab_output = format_ascii_tab(tab_notes, tuning=tuning)
    
    print("\n🎼 Guitar Tablature:")
    print("-" * 40)
    print(tab_output)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Guitar Tab - Generated from {os.path.basename(audio_path)}\n")
            f.write(f"# Tuning: {args.tuning}\n\n")
            f.write(tab_output)
        print(f"\n✅ Saved to {args.output}")
    
    # Cleanup temp file if downloaded from YouTube
    if cleanup_file and os.path.exists(audio_path):
        os.remove(audio_path)
        print("🗑️ Cleaned up temp audio file")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
