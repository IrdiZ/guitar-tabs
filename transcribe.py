#!/usr/bin/env python3
"""
Guitar Tab Transcription Tool

Converts guitar audio to MIDI and tablature using state-of-the-art models.

Usage:
    python transcribe.py input.mp3 -o output/
    python transcribe.py input.mp3 --isolate  # Isolate guitar first (for full mixes)
    python transcribe.py input.mp3 --format midi  # Output MIDI only
    python transcribe.py input.mp3 --format ascii  # Output ASCII tabs
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import json

# Optional imports - we check availability
BASIC_PITCH_AVAILABLE = False
DEMUCS_AVAILABLE = False
PRETTY_MIDI_AVAILABLE = False

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    pass

try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    pass

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    pass


def check_dependencies():
    """Check and report on available dependencies."""
    deps = {
        "basic-pitch": BASIC_PITCH_AVAILABLE,
        "demucs": DEMUCS_AVAILABLE,
        "pretty_midi": PRETTY_MIDI_AVAILABLE,
    }
    
    missing = [name for name, available in deps.items() if not available]
    
    if missing:
        print("Missing optional dependencies:")
        for dep in missing:
            print(f"  - {dep}: pip install {dep}")
        print()
    
    return deps


def isolate_guitar(input_path: str, output_dir: str) -> Optional[str]:
    """
    Use Demucs to isolate guitar from a full mix.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save isolated guitar
        
    Returns:
        Path to isolated guitar audio, or None if failed
    """
    if not DEMUCS_AVAILABLE:
        print("Error: demucs not installed. Run: pip install demucs")
        return None
    
    print(f"🎸 Isolating guitar from: {input_path}")
    
    # Use htdemucs_6s model which has guitar as a separate stem
    try:
        demucs.separate.main([
            "-n", "htdemucs_6s",
            "--two-stems=guitar",
            "-o", output_dir,
            input_path
        ])
        
        # Find the output file
        input_name = Path(input_path).stem
        guitar_path = Path(output_dir) / "htdemucs_6s" / input_name / "guitar.wav"
        
        if guitar_path.exists():
            print(f"✅ Guitar isolated: {guitar_path}")
            return str(guitar_path)
        else:
            print(f"❌ Guitar file not found at expected path: {guitar_path}")
            return None
            
    except Exception as e:
        print(f"❌ Demucs separation failed: {e}")
        return None


def transcribe_to_midi(input_path: str, output_dir: str) -> Tuple[Optional[str], Optional[List]]:
    """
    Transcribe audio to MIDI using Basic Pitch.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save output
        
    Returns:
        Tuple of (midi_path, note_events) or (None, None) if failed
    """
    if not BASIC_PITCH_AVAILABLE:
        print("Error: basic-pitch not installed. Run: pip install basic-pitch")
        return None, None
    
    print(f"🎵 Transcribing: {input_path}")
    
    try:
        model_output, midi_data, note_events = predict(input_path)
        
        # Save MIDI
        input_name = Path(input_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        midi_path = output_path / f"{input_name}.mid"
        midi_data.write(str(midi_path))
        
        print(f"✅ MIDI saved: {midi_path}")
        print(f"   Notes detected: {len(note_events)}")
        
        return str(midi_path), note_events
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None, None


def midi_to_ascii_tab(midi_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert MIDI to ASCII guitar tablature.
    
    This is a simplified conversion that maps MIDI notes to guitar frets.
    For proper tablature with optimal fingering, use the CRNN model.
    
    Args:
        midi_path: Path to MIDI file
        output_path: Optional path to save ASCII tab
        
    Returns:
        ASCII tablature string
    """
    if not PRETTY_MIDI_AVAILABLE:
        print("Warning: pretty_midi not installed, using simple tab format")
        return "Install pretty_midi for ASCII tab generation: pip install pretty_midi"
    
    # Standard guitar tuning (E A D G B E) in MIDI note numbers
    TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
    STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']
    MAX_FRET = 22
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Collect all notes
    all_notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            all_notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })
    
    if not all_notes:
        return "No notes found in MIDI file"
    
    # Sort by start time
    all_notes.sort(key=lambda x: x['start'])
    
    # Group notes into time buckets (quantize to ~0.1s)
    QUANTUM = 0.1
    time_buckets = {}
    for note in all_notes:
        bucket = round(note['start'] / QUANTUM) * QUANTUM
        if bucket not in time_buckets:
            time_buckets[bucket] = []
        time_buckets[bucket].append(note)
    
    # Build tab lines
    tab_lines = {i: [] for i in range(6)}
    times = sorted(time_buckets.keys())
    
    for t in times:
        notes = time_buckets[t]
        frets = ['-'] * 6  # One for each string
        
        for note in notes:
            pitch = note['pitch']
            
            # Find best string/fret combination
            best_string = None
            best_fret = None
            
            for string_idx, open_pitch in enumerate(TUNING):
                fret = pitch - open_pitch
                if 0 <= fret <= MAX_FRET:
                    if best_fret is None or fret < best_fret:
                        best_string = string_idx
                        best_fret = fret
            
            if best_string is not None:
                frets[best_string] = str(best_fret) if best_fret < 10 else f"({best_fret})"
        
        for i in range(6):
            tab_lines[i].append(frets[i])
    
    # Format output
    output_lines = []
    chars_per_line = 60
    
    for start in range(0, len(times), chars_per_line):
        end = min(start + chars_per_line, len(times))
        for string_idx in range(6):
            line_content = ''.join(
                f"{tab_lines[string_idx][i]:>3}" 
                for i in range(start, end)
            )
            output_lines.append(f"{STRING_NAMES[string_idx]}|{line_content}|")
        output_lines.append("")  # Blank line between sections
    
    tab_str = '\n'.join(output_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(tab_str)
        print(f"✅ ASCII tab saved: {output_path}")
    
    return tab_str


def main():
    parser = argparse.ArgumentParser(
        description='Guitar Tab Transcription Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3                    # Basic transcription to MIDI
  %(prog)s song.mp3 -o results/        # Specify output directory
  %(prog)s song.mp3 --isolate          # Isolate guitar first (for full mixes)
  %(prog)s song.mp3 --format ascii     # Generate ASCII tablature
  %(prog)s --check                     # Check installed dependencies
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input audio file')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--isolate', action='store_true', 
                        help='Isolate guitar from mix first (requires demucs)')
    parser.add_argument('--format', choices=['midi', 'ascii', 'both'], default='both',
                        help='Output format')
    parser.add_argument('--check', action='store_true', 
                        help='Check installed dependencies')
    
    args = parser.parse_args()
    
    # Check dependencies
    deps = check_dependencies()
    
    if args.check:
        print("Dependency status:")
        for name, available in deps.items():
            status = "✅ installed" if available else "❌ missing"
            print(f"  {name}: {status}")
        
        print("\nRecommended install:")
        print("  pip install basic-pitch demucs pretty_midi")
        return 0
    
    if not args.input:
        parser.print_help()
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Optionally isolate guitar
    audio_to_transcribe = args.input
    if args.isolate:
        isolated = isolate_guitar(args.input, args.output)
        if isolated:
            audio_to_transcribe = isolated
        else:
            print("Warning: Guitar isolation failed, using original audio")
    
    # Step 2: Transcribe to MIDI
    midi_path, note_events = transcribe_to_midi(audio_to_transcribe, args.output)
    
    if not midi_path:
        print("Transcription failed")
        return 1
    
    # Step 3: Optionally generate ASCII tab
    if args.format in ['ascii', 'both']:
        input_name = Path(args.input).stem
        tab_path = Path(args.output) / f"{input_name}.tab"
        tab = midi_to_ascii_tab(midi_path, str(tab_path))
        
        if args.format == 'ascii':
            print("\n" + "="*60)
            print("ASCII TABLATURE")
            print("="*60)
            print(tab[:2000])  # Print first 2000 chars
            if len(tab) > 2000:
                print(f"\n... (truncated, see full tab at {tab_path})")
    
    print(f"\n✅ Done! Output saved to: {args.output}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
