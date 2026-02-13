#!/usr/bin/env python3
"""
Guitar Tab Generator - Audio to Tabs using AI
Built for Albanian songs with no existing tabs 🎸
"""

import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import argparse
import sys
import os
import subprocess
import tempfile

# Standard guitar tuning (low to high): E2, A2, D3, G3, B3, E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # MIDI note numbers
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
NUM_FRETS = 24

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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


def midi_to_fret_options(midi_note: int, tuning: List[int] = STANDARD_TUNING) -> List[Tuple[int, int]]:
    """
    Given a MIDI note, return all possible (string, fret) combinations.
    Returns list of (string_index, fret_number) tuples.
    """
    options = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= NUM_FRETS:
            options.append((string_idx, fret))
    return options


def choose_best_fret_position(
    note: Note,
    prev_position: Optional[Tuple[int, int]],
    tuning: List[int] = STANDARD_TUNING
) -> Optional[TabNote]:
    """
    Choose the best fret position for a note, considering:
    - Playability (prefer lower frets for beginners)
    - Hand position continuity (stay close to previous note)
    - String preference (middle strings often easier)
    """
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
        
        # Prefer lower frets (easier to play)
        score -= fret * 0.5
        
        # Prefer middle strings (easier access)
        string_preference = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5]  # Middle strings preferred
        score += string_preference[string] * 5
        
        # If we have a previous position, prefer staying close
        if prev_position:
            prev_string, prev_fret = prev_position
            # Penalize large jumps
            fret_distance = abs(fret - prev_fret)
            string_distance = abs(string - prev_string)
            score -= fret_distance * 0.3
            score -= string_distance * 1.0
        
        scored_options.append((score, string, fret))
    
    # Sort by score (highest first)
    scored_options.sort(reverse=True)
    _, best_string, best_fret = scored_options[0]
    
    return TabNote(best_string, best_fret, note.start_time, note.duration)


def detect_notes_from_audio(
    audio_path: str,
    hop_length: int = 512,
    min_note_duration: float = 0.05,  # 50ms minimum
    confidence_threshold: float = 0.5
) -> List[Note]:
    """
    Detect notes from audio file using librosa.
    Uses pitch detection + onset detection.
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    print("Detecting pitches...")
    # Use piptrack for pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    
    print("Detecting onsets...")
    # Detect note onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    print(f"Found {len(onset_times)} onsets")
    
    notes = []
    
    for i, onset_time in enumerate(onset_times):
        onset_frame = onset_frames[i]
        
        # Get the most prominent pitch at this onset
        pitch_slice = pitches[:, onset_frame]
        mag_slice = magnitudes[:, onset_frame]
        
        if mag_slice.max() == 0:
            continue
            
        # Find the bin with maximum magnitude
        max_idx = mag_slice.argmax()
        pitch_hz = pitch_slice[max_idx]
        
        if pitch_hz <= 0:
            continue
        
        # Convert Hz to MIDI note number
        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
        
        # Estimate duration (until next onset or end)
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - onset_time
        else:
            duration = librosa.get_duration(y=y, sr=sr) - onset_time
        
        # Filter by duration
        if duration < min_note_duration:
            continue
        
        # Calculate confidence from magnitude
        confidence = mag_slice[max_idx] / (magnitudes.max() + 1e-10)
        
        if confidence >= confidence_threshold:
            notes.append(Note(
                midi=midi_note,
                start_time=onset_time,
                duration=duration,
                confidence=confidence
            ))
    
    print(f"Detected {len(notes)} notes")
    return notes


def notes_to_tabs(notes: List[Note]) -> List[TabNote]:
    """Convert detected notes to guitar tab positions."""
    tab_notes = []
    prev_position = None
    
    for note in notes:
        tab_note = choose_best_fret_position(note, prev_position)
        if tab_note:
            tab_notes.append(tab_note)
            prev_position = (tab_note.string, tab_note.fret)
    
    return tab_notes


def format_ascii_tab(tab_notes: List[TabNote], beats_per_line: int = 16) -> str:
    """Format tab notes as ASCII guitar tablature."""
    if not tab_notes:
        return "No notes detected!"
    
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
            string_name = STRING_NAMES[string]
            notes_str = ''.join(grid[string][start:end])
            lines.append(f"{string_name}|{notes_str}|")
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
    parser = argparse.ArgumentParser(description='Generate guitar tabs from audio')
    parser.add_argument('audio_file', help='Path to audio file OR YouTube URL')
    parser.add_argument('--output', '-o', help='Output file for tabs')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                        help='Minimum confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Handle YouTube URLs
    audio_path = args.audio_file
    cleanup_file = False
    
    if is_youtube_url(audio_path):
        audio_path = download_youtube_audio(audio_path)
        cleanup_file = True
    
    print("🎸 Guitar Tab Generator")
    print("=" * 40)
    
    # Detect notes
    notes = detect_notes_from_audio(
        audio_path,
        confidence_threshold=args.confidence
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
    tab_notes = notes_to_tabs(notes)
    
    # Format as ASCII tab
    tab_output = format_ascii_tab(tab_notes)
    
    print("\n🎼 Guitar Tablature:")
    print("-" * 40)
    print(tab_output)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Guitar Tab - Generated from {audio_path}\n\n")
            f.write(tab_output)
        print(f"\n✅ Saved to {args.output}")
    
    # Cleanup temp file if downloaded from YouTube
    if cleanup_file and os.path.exists(audio_path):
        os.remove(audio_path)
        print("🗑️ Cleaned up temp audio file")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
