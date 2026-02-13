#!/usr/bin/env python3
"""
Basic Pitch Runner - Docker wrapper for Spotify's Basic Pitch
Outputs detected notes as JSON for integration with guitar_tabs.py
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"


def run_basic_pitch(audio_path, output_path=None, onset_threshold=0.5, 
                    frame_threshold=0.3, min_note_len=50, min_freq=None, max_freq=None):
    """
    Run Basic Pitch on an audio file and output detected notes as JSON.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to output JSON file (default: stdout)
        onset_threshold: Minimum confidence for note onset (0-1)
        frame_threshold: Minimum confidence for note frames (0-1)
        min_note_len: Minimum note length in milliseconds
        min_freq: Minimum frequency in Hz (None for no limit)
        max_freq: Maximum frequency in Hz (None for no limit)
    """
    print(f"Processing: {audio_path}", file=sys.stderr)
    print(f"Using model: {ICASSP_2022_MODEL_PATH}", file=sys.stderr)
    
    # Run Basic Pitch prediction
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_len,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )
    
    # Convert note events to our format
    notes = []
    for start_time, end_time, midi_note, amplitude, bends in note_events:
        # Filter by guitar range if not already done
        if min_freq is None and max_freq is None:
            # Default guitar range: E2 (82 Hz) to E6 (~1300 Hz)
            # MIDI: 40 to 88
            if midi_note < 36 or midi_note > 90:
                continue
        
        notes.append({
            'midi': int(midi_note),
            'start_time': float(start_time),
            'duration': float(end_time - start_time),
            'confidence': float(amplitude),
            'name': midi_to_note_name(int(midi_note)),
            'pitch_bends': [float(b) for b in bends] if bends else []
        })
    
    # Sort by start time
    notes.sort(key=lambda n: (n['start_time'], n['midi']))
    
    result = {
        'source': str(audio_path),
        'model': 'basic-pitch-icassp-2022',
        'notes': notes,
        'total_notes': len(notes)
    }
    
    # Output
    output_json = json.dumps(result, indent=2)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_json)
        print(f"Output written to: {output_path}", file=sys.stderr)
    else:
        print(output_json)
    
    print(f"Detected {len(notes)} notes", file=sys.stderr)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run Basic Pitch on audio file')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('-o', '--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--onset-threshold', type=float, default=0.5,
                        help='Onset detection threshold (0-1, default: 0.5)')
    parser.add_argument('--frame-threshold', type=float, default=0.3,
                        help='Frame detection threshold (0-1, default: 0.3)')
    parser.add_argument('--min-note-len', type=int, default=50,
                        help='Minimum note length in ms (default: 50)')
    parser.add_argument('--min-freq', type=float, default=None,
                        help='Minimum frequency in Hz')
    parser.add_argument('--max-freq', type=float, default=None,
                        help='Maximum frequency in Hz')
    
    args = parser.parse_args()
    
    run_basic_pitch(
        args.audio_path,
        output_path=args.output,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_note_len=args.min_note_len,
        min_freq=args.min_freq,
        max_freq=args.max_freq
    )


if __name__ == '__main__':
    main()
