#!/usr/bin/env python3
"""
Guitar Technique Annotation - Integration Module

Bridges the technique detection engine with the main guitar_tabs system.
Integrates with OnsetDetector's legato detection (59+ legato events detected).

Usage:
    python annotate_techniques.py <audio_file> [--output <file>]
    
Output notation:
    5h7  - Hammer-on from fret 5 to 7
    7p5  - Pull-off from fret 7 to 5
    7b9  - Bend from fret 7 reaching pitch of fret 9
    5/7  - Slide up from fret 5 to 7
    7\\5  - Slide down from fret 7 to 5
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import librosa
from typing import List, Tuple, Optional, Any

from technique_detector import (
    TechniqueDetector,
    AnnotatedNote,
    TechniqueAnnotation,
    Technique,
    format_ascii_tab_with_techniques
)

from guitar_tabs import (
    detect_notes_from_audio,
    detect_onsets_ensemble,
    TUNINGS,
    Note,
    STRING_NAMES
)


def analyze_audio_techniques(
    audio_path: str,
    tuning_name: str = 'standard',
    confidence_threshold: float = 0.3,
    pitch_method: str = 'pyin',
    verbose: bool = True
) -> Tuple[List[AnnotatedNote], str, dict]:
    """
    Full pipeline: detect notes, analyze techniques, generate annotated tabs.
    
    Args:
        audio_path: Path to audio file
        tuning_name: Guitar tuning name
        confidence_threshold: Minimum note confidence
        pitch_method: Pitch detection method
        verbose: Print progress
        
    Returns:
        Tuple of (annotated_notes, ascii_tab, stats_dict)
    """
    if verbose:
        print(f"🎸 Guitar Technique Annotation Engine")
        print(f"   Audio: {audio_path}")
        print(f"   Tuning: {tuning_name}")
        print()
    
    # Get tuning
    tuning = TUNINGS.get(tuning_name.lower(), TUNINGS['standard'])
    
    # Load audio
    if verbose:
        print("📂 Loading audio...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 512
    
    # Detect notes
    if verbose:
        print(f"\n🎵 Detecting notes using {pitch_method}...")
    notes = detect_notes_from_audio(
        audio_path=audio_path,
        hop_length=hop_length,
        confidence_threshold=confidence_threshold,
        pitch_method=pitch_method,
        tuning=tuning
    )
    
    if not notes:
        return [], "No notes detected.", {}
    
    if verbose:
        print(f"   Found {len(notes)} notes")
    
    # Detect onsets with legato information
    if verbose:
        print("\n🥁 Running ensemble onset detection with legato analysis...")
    
    onset_times, onset_details = detect_onsets_ensemble(
        y=y,
        sr=sr,
        hop_length=hop_length,
        min_votes=2,
        attack_detection=True,
        legato_detection=True,
        legato_sensitivity=0.4,
        verbose=verbose
    )
    
    # Count legato events
    legato_count = sum(1 for o in onset_details if o.is_legato)
    if verbose:
        print(f"   Legato events detected: {legato_count}")
    
    # Run technique detection
    if verbose:
        print("\n🎸 Analyzing techniques...")
    
    detector = TechniqueDetector(sr=sr, hop_length=hop_length)
    annotated_notes = detector.analyze(
        y=y,
        notes=notes,
        onset_details=onset_details,
        tuning=tuning,
        verbose=verbose
    )
    
    # Generate ASCII tab
    if verbose:
        print("\n📝 Generating annotated tablature...")
    
    ascii_tab = format_ascii_tab_with_techniques(annotated_notes)
    
    # Compile statistics
    stats = {
        'total_notes': len(notes),
        'annotated_notes': len(annotated_notes),
        'legato_events': legato_count,
        'techniques': {
            'hammer_on': sum(1 for n in annotated_notes if n.technique.technique == Technique.HAMMER_ON),
            'pull_off': sum(1 for n in annotated_notes if n.technique.technique == Technique.PULL_OFF),
            'bend': sum(1 for n in annotated_notes if n.technique.technique == Technique.BEND),
            'slide_up': sum(1 for n in annotated_notes if n.technique.technique == Technique.SLIDE_UP),
            'slide_down': sum(1 for n in annotated_notes if n.technique.technique == Technique.SLIDE_DOWN),
        }
    }
    
    return annotated_notes, ascii_tab, stats


def format_detailed_output(
    annotated_notes: List[AnnotatedNote],
    stats: dict
) -> str:
    """Format detailed output with technique breakdown."""
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("GUITAR TECHNIQUE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Statistics
    lines.append("STATISTICS:")
    lines.append(f"  Total notes detected: {stats['total_notes']}")
    lines.append(f"  Notes with positions: {stats['annotated_notes']}")
    lines.append(f"  Legato events: {stats['legato_events']}")
    lines.append("")
    
    lines.append("TECHNIQUES DETECTED:")
    tech = stats['techniques']
    lines.append(f"  Hammer-ons (h):   {tech['hammer_on']}")
    lines.append(f"  Pull-offs (p):    {tech['pull_off']}")
    lines.append(f"  Bends (b):        {tech['bend']}")
    lines.append(f"  Slides up (/):    {tech['slide_up']}")
    lines.append(f"  Slides down (\\):  {tech['slide_down']}")
    lines.append("")
    
    # Legend
    lines.append("NOTATION LEGEND:")
    lines.append("  5h7  = Hammer-on from fret 5 to fret 7")
    lines.append("  7p5  = Pull-off from fret 7 to fret 5")
    lines.append("  7b9  = Bend at fret 7, reaching pitch of fret 9")
    lines.append("  5/7  = Slide up from fret 5 to fret 7")
    lines.append("  7\\5  = Slide down from fret 7 to fret 5")
    lines.append("")
    
    # Detailed note list
    lines.append("-" * 60)
    lines.append("DETAILED NOTE LIST:")
    lines.append("-" * 60)
    
    string_names = ['E', 'A', 'D', 'G', 'B', 'e']
    
    for i, note in enumerate(annotated_notes[:50]):  # First 50 notes
        tech_str = note.to_ascii()
        string_name = string_names[note.string] if 0 <= note.string < 6 else '?'
        
        legato_mark = " [legato]" if note.is_legato else ""
        tech_name = note.technique.technique.name if note.technique.technique != Technique.NONE else ""
        
        lines.append(
            f"  {i+1:3d}. t={note.start_time:6.3f}s | "
            f"String {string_name} | "
            f"Tab: {tech_str:6s} | "
            f"{tech_name}{legato_mark}"
        )
    
    if len(annotated_notes) > 50:
        lines.append(f"  ... and {len(annotated_notes) - 50} more notes")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Detect and annotate guitar playing techniques',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Technique notation:
  5h7  - Hammer-on from fret 5 to 7
  7p5  - Pull-off from fret 7 to 5
  7b9  - Bend at fret 7 reaching pitch of fret 9
  5/7  - Slide up from fret 5 to 7
  7\\5  - Slide down from fret 7 to 5

Examples:
  python annotate_techniques.py song.mp3
  python annotate_techniques.py song.mp3 -o tabs.txt
  python annotate_techniques.py song.mp3 --tuning drop_d --verbose
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--tuning', '-t', default='standard',
                        choices=list(TUNINGS.keys()),
                        help='Guitar tuning (default: standard)')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                        help='Minimum note confidence (default: 0.3)')
    parser.add_argument('--pitch-method', '-p', 
                        choices=['pyin', 'crepe', 'cqt'],
                        default='pyin',
                        help='Pitch detection method (default: pyin)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Include detailed note list in output')
    parser.add_argument('--tab-only', action='store_true',
                        help='Output only the ASCII tablature')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    try:
        annotated_notes, ascii_tab, stats = analyze_audio_techniques(
            audio_path=args.audio_file,
            tuning_name=args.tuning,
            confidence_threshold=args.confidence,
            pitch_method=args.pitch_method,
            verbose=args.verbose or not args.tab_only
        )
    except Exception as e:
        print(f"Error analyzing audio: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Generate output
    if args.tab_only:
        output = ascii_tab
    elif args.detailed:
        output = format_detailed_output(annotated_notes, stats)
        output += "\n\n" + "=" * 60 + "\nTABLATURE:\n" + "=" * 60 + "\n"
        output += ascii_tab
    else:
        output = f"""Guitar Technique Analysis
========================
Notes: {stats['total_notes']} | Legato: {stats['legato_events']}
Techniques: {stats['techniques']['hammer_on']}h {stats['techniques']['pull_off']}p {stats['techniques']['bend']}b {stats['techniques']['slide_up']}/ {stats['techniques']['slide_down']}\\

{ascii_tab}
"""
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Output written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
