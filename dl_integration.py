#!/usr/bin/env python3
"""
Deep Learning Integration Module for Guitar Tab Transcription

This module provides a unified interface for using neural network-based
transcription with the guitar_tabs.py system.

Features:
- Basic Pitch (Spotify) neural network via Docker
- MIDI to tablature conversion with guitar-specific optimizations
- Integration with existing preprocessing and music theory modules

Usage:
    python dl_integration.py audio.mp3 --output tabs.txt
    python dl_integration.py audio.mp3 --json output.json

Author: Claude (Subagent) for guitar-tabs project
Date: 2026-02-13
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Note:
    """Represents a detected note."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    pitch_bends: List[float] = field(default_factory=list)
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    @property
    def name(self) -> str:
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.midi // 12) - 1
        note = notes[self.midi % 12]
        return f"{note}{octave}"


@dataclass
class TabPosition:
    """Represents a position on the guitar fretboard."""
    string: int  # 0-5 (low E to high E)
    fret: int    # 0-22
    midi: int
    start_time: float
    duration: float
    confidence: float = 1.0


# ============================================================================
# CONSTANTS
# ============================================================================

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # Standard tuning MIDI: E2-E4
NUM_STRINGS = 6
NUM_FRETS = 22
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']


# ============================================================================
# BASIC PITCH DOCKER WRAPPER
# ============================================================================

def run_basic_pitch_docker(audio_path: str, output_json: Optional[str] = None) -> Dict:
    """
    Run Basic Pitch via Docker container.
    
    Args:
        audio_path: Path to audio file
        output_json: Optional output JSON path
    
    Returns:
        Parsed JSON result with detected notes
    """
    audio_path = Path(audio_path).resolve()
    audio_dir = audio_path.parent
    audio_name = audio_path.name
    
    # Build Docker command
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{audio_dir}:/audio',
    ]
    
    if output_json:
        output_path = Path(output_json).resolve()
        output_dir = output_path.parent
        output_name = output_path.name
        cmd.extend(['-v', f'{output_dir}:/output'])
        cmd.extend(['basic-pitch-runner', f'/audio/{audio_name}', '-o', f'/output/{output_name}'])
    else:
        cmd.extend(['basic-pitch-runner', f'/audio/{audio_name}'])
    
    # Run Docker
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Parse output
    if output_json and Path(output_json).exists():
        with open(output_json) as f:
            return json.load(f)
    
    # Parse stdout - find JSON in output (may have log lines before)
    stdout = result.stdout
    json_start = stdout.find('{')
    if json_start == -1:
        if result.returncode != 0:
            raise RuntimeError(f"Basic Pitch failed: {result.stderr}")
        raise RuntimeError("No JSON found in Basic Pitch output")
    
    try:
        return json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Basic Pitch output: {e}")


def transcribe_with_basic_pitch(audio_path: str) -> List[Note]:
    """
    Transcribe audio using Basic Pitch neural network.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        List of detected Note objects
    """
    result = run_basic_pitch_docker(audio_path)
    
    notes = []
    for note_data in result.get('notes', []):
        notes.append(Note(
            midi=note_data['midi'],
            start_time=note_data['start_time'],
            duration=note_data['duration'],
            confidence=note_data['confidence'],
            pitch_bends=note_data.get('pitch_bends', [])
        ))
    
    return notes


# ============================================================================
# MIDI TO TABLATURE CONVERSION
# ============================================================================

class MIDIToTabConverter:
    """
    Convert MIDI notes to guitar tablature positions.
    
    Uses optimization to find playable fingerings considering:
    - Physical constraints (reach, string assignments)
    - Preference for lower frets
    - Hand position continuity
    """
    
    def __init__(
        self,
        tuning: List[int] = None,
        num_frets: int = NUM_FRETS,
        max_fret_span: int = 5,
        prefer_lower: bool = True
    ):
        self.tuning = tuning or GUITAR_TUNING
        self.num_frets = num_frets
        self.max_fret_span = max_fret_span
        self.prefer_lower = prefer_lower
        
        # Build lookup table: midi -> [(string, fret), ...]
        self._build_lookup()
    
    def _build_lookup(self):
        """Build MIDI to string/fret lookup table."""
        self.midi_to_positions = {}
        
        for string_idx, open_midi in enumerate(self.tuning):
            for fret in range(self.num_frets + 1):
                midi = open_midi + fret
                if midi not in self.midi_to_positions:
                    self.midi_to_positions[midi] = []
                self.midi_to_positions[midi].append((string_idx, fret))
    
    def get_possible_positions(self, midi: int) -> List[Tuple[int, int]]:
        """Get all possible string/fret positions for a MIDI note."""
        return self.midi_to_positions.get(midi, [])
    
    def convert_note(
        self, 
        note: Note,
        prev_position: Optional[TabPosition] = None
    ) -> Optional[TabPosition]:
        """Convert a single note to tablature position."""
        positions = self.get_possible_positions(note.midi)
        
        if not positions:
            return None
        
        if len(positions) == 1:
            string, fret = positions[0]
            return TabPosition(
                string=string,
                fret=fret,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.confidence
            )
        
        # Score each position
        best_pos = None
        best_score = float('inf')
        
        for string, fret in positions:
            score = 0.0
            
            # Prefer lower frets
            if self.prefer_lower:
                score += fret * 0.1
            
            # Prefer continuity with previous position
            if prev_position is not None:
                fret_distance = abs(fret - prev_position.fret)
                string_distance = abs(string - prev_position.string)
                score += fret_distance * 0.5 + string_distance * 0.2
            
            # Penalize extreme positions
            if fret > 12:
                score += (fret - 12) * 0.3
            
            if score < best_score:
                best_score = score
                best_pos = (string, fret)
        
        if best_pos:
            string, fret = best_pos
            return TabPosition(
                string=string,
                fret=fret,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.confidence
            )
        
        return None
    
    def convert_notes(self, notes: List[Note]) -> List[TabPosition]:
        """Convert list of notes to tablature positions."""
        tablature = []
        prev_pos = None
        
        for note in sorted(notes, key=lambda n: n.start_time):
            pos = self.convert_note(note, prev_pos)
            if pos:
                tablature.append(pos)
                prev_pos = pos
        
        return tablature
    
    def resolve_chord_positions(
        self,
        notes: List[Note],
        time_threshold: float = 0.05
    ) -> List[TabPosition]:
        """
        Convert notes to tablature with proper chord handling.
        
        Groups simultaneous notes and optimizes string assignments together.
        """
        if not notes:
            return []
        
        # Sort by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Group into chords
        chords = []
        current_chord = [sorted_notes[0]]
        
        for note in sorted_notes[1:]:
            if note.start_time - current_chord[0].start_time <= time_threshold:
                current_chord.append(note)
            else:
                chords.append(current_chord)
                current_chord = [note]
        chords.append(current_chord)
        
        # Process each chord
        tablature = []
        prev_positions = []
        
        for chord in chords:
            if len(chord) == 1:
                # Single note
                prev_pos = prev_positions[-1] if prev_positions else None
                pos = self.convert_note(chord[0], prev_pos)
                if pos:
                    tablature.append(pos)
                    prev_positions = [pos]
            else:
                # Chord - find optimal string assignment
                chord_positions = self._optimize_chord(chord, prev_positions)
                tablature.extend(chord_positions)
                prev_positions = chord_positions
        
        return tablature
    
    def _optimize_chord(
        self,
        notes: List[Note],
        prev_positions: List[TabPosition]
    ) -> List[TabPosition]:
        """Optimize string assignment for a chord."""
        from itertools import product
        
        # Get all possible positions for each note
        all_positions = []
        for note in notes:
            positions = self.get_possible_positions(note.midi)
            if positions:
                all_positions.append([(note, s, f) for s, f in positions])
        
        if not all_positions:
            return []
        
        # Find combination with no string conflicts and best score
        best_combo = None
        best_score = float('inf')
        
        for combo in product(*all_positions):
            # Check for string conflicts
            strings_used = [pos[1] for pos in combo]
            if len(strings_used) != len(set(strings_used)):
                continue  # Conflict
            
            # Calculate score
            frets = [pos[2] for pos in combo]
            fret_span = max(frets) - min(frets) if frets else 0
            
            if fret_span > self.max_fret_span:
                continue  # Unplayable
            
            score = 0.0
            avg_fret = sum(frets) / len(frets)
            score += avg_fret * 0.2
            score += fret_span * 0.5
            
            if prev_positions:
                prev_avg_fret = sum(p.fret for p in prev_positions) / len(prev_positions)
                score += abs(avg_fret - prev_avg_fret) * 0.3
            
            if score < best_score:
                best_score = score
                best_combo = combo
        
        if best_combo:
            return [
                TabPosition(
                    string=string,
                    fret=fret,
                    midi=note.midi,
                    start_time=note.start_time,
                    duration=note.duration,
                    confidence=note.confidence
                )
                for note, string, fret in best_combo
            ]
        
        # Fallback
        result = []
        used_strings = set()
        for note in sorted(notes, key=lambda n: n.midi):
            positions = self.get_possible_positions(note.midi)
            for string, fret in positions:
                if string not in used_strings:
                    result.append(TabPosition(
                        string=string,
                        fret=fret,
                        midi=note.midi,
                        start_time=note.start_time,
                        duration=note.duration,
                        confidence=note.confidence
                    ))
                    used_strings.add(string)
                    break
        
        return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_ascii_tablature(
    positions: List[TabPosition],
    time_step: float = 0.1,
    max_width: int = 80
) -> str:
    """Format tablature positions as ASCII tab."""
    if not positions:
        return "No tablature generated"
    
    max_time = max(p.start_time for p in positions) + 1.0
    n_cols = int(max_time / time_step) + 1
    
    # Initialize strings (reversed for display: high E on top)
    string_names = ['e', 'B', 'G', 'D', 'A', 'E']
    lines = {i: ['-'] * n_cols for i in range(6)}
    
    for pos in positions:
        col = int(pos.start_time / time_step)
        if col < n_cols:
            display_string = 5 - pos.string  # Reverse for display
            fret_str = str(pos.fret)
            lines[display_string][col] = fret_str
    
    # Format output
    result = []
    for i in range(6):
        string_line = string_names[i] + '|' + ''.join(lines[i])
        
        # Split into chunks if too wide
        if len(string_line) > max_width:
            result.append(string_line[:max_width])
        else:
            result.append(string_line)
    
    return '\n'.join(result)


def format_json_output(
    notes: List[Note],
    tablature: List[TabPosition],
    metadata: Dict[str, Any] = None
) -> str:
    """Format output as JSON."""
    output = {
        'model': 'basic-pitch-icassp-2022',
        'notes': [
            {
                'midi': n.midi,
                'name': n.name,
                'start_time': n.start_time,
                'duration': n.duration,
                'confidence': n.confidence
            }
            for n in notes
        ],
        'tablature': [
            {
                'string': STRING_NAMES[p.string],
                'string_index': p.string,
                'fret': p.fret,
                'midi': p.midi,
                'start_time': p.start_time,
                'duration': p.duration
            }
            for p in tablature
        ],
        'summary': {
            'total_notes': len(notes),
            'total_tab_positions': len(tablature),
            'unique_frets': len(set(p.fret for p in tablature)),
            'strings_used': sorted(set(p.string for p in tablature))
        }
    }
    
    if metadata:
        output['metadata'] = metadata
    
    return json.dumps(output, indent=2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def transcribe_to_tabs(
    audio_path: str,
    tuning: List[int] = None,
    confidence_threshold: float = 0.3,
    output_format: str = 'ascii'
) -> str:
    """
    Complete pipeline: audio -> neural network -> tablature.
    
    Args:
        audio_path: Path to audio file
        tuning: Guitar tuning (MIDI notes, low to high)
        confidence_threshold: Minimum note confidence
        output_format: 'ascii', 'json', or 'both'
    
    Returns:
        Formatted tablature string
    """
    print(f"Transcribing: {audio_path}", file=sys.stderr)
    
    # Step 1: Run Basic Pitch neural network
    print("Running Basic Pitch neural network...", file=sys.stderr)
    notes = transcribe_with_basic_pitch(audio_path)
    
    # Filter by confidence
    notes = [n for n in notes if n.confidence >= confidence_threshold]
    print(f"Detected {len(notes)} notes (confidence >= {confidence_threshold})", file=sys.stderr)
    
    # Step 2: Convert to tablature
    print("Converting to tablature...", file=sys.stderr)
    converter = MIDIToTabConverter(tuning=tuning)
    tablature = converter.resolve_chord_positions(notes)
    print(f"Generated {len(tablature)} tab positions", file=sys.stderr)
    
    # Step 3: Format output
    if output_format == 'json':
        return format_json_output(notes, tablature)
    elif output_format == 'both':
        ascii_tab = format_ascii_tablature(tablature)
        json_out = format_json_output(notes, tablature)
        return f"ASCII Tablature:\n{ascii_tab}\n\nJSON:\n{json_out}"
    else:
        return format_ascii_tablature(tablature)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Learning Guitar Tab Transcription using Basic Pitch'
    )
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument(
        '-o', '--output',
        help='Output file path'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['ascii', 'json', 'both'],
        default='ascii',
        help='Output format (default: ascii)'
    )
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.3,
        help='Minimum note confidence (default: 0.3)'
    )
    parser.add_argument(
        '--tuning',
        help='Guitar tuning as comma-separated MIDI notes (default: standard)'
    )
    
    args = parser.parse_args()
    
    # Parse tuning
    tuning = None
    if args.tuning:
        tuning = [int(x) for x in args.tuning.split(',')]
    
    # Run transcription
    result = transcribe_to_tabs(
        args.audio_path,
        tuning=tuning,
        confidence_threshold=args.confidence,
        output_format=args.format
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == '__main__':
    main()
