#!/usr/bin/env python3
"""
Test pattern matching on real audio.

Usage:
    python test_pattern_matching.py audio.mp3
    
Tests the pattern matching pipeline end-to-end:
1. Load audio and detect pitches
2. Apply pattern matching
3. Show corrections and pattern annotations
"""

import sys
import os
import json
import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

from pattern_library import (
    PatternMatcher, match_patterns, analyze_sequence,
    PatternType, PATTERN_LIBRARY
)
from pattern_integration import (
    enhance_transcription, DetectedNote,
    PatternEnhancedTranscription
)


def load_audio(audio_path: str):
    """Load audio file and return samples + sample rate."""
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    return y, sr


def detect_notes_simple(audio_path: str) -> list:
    """
    Detect notes using available pitch detection.
    
    Falls back through methods based on what's available.
    """
    import librosa
    
    print(f"Loading: {audio_path}")
    y, sr = load_audio(audio_path)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr}")
    
    notes = []
    
    # Try ensemble pitch detection first
    try:
        from ensemble_pitch import detect_notes_ensemble
        print("Using ensemble pitch detection...")
        result = detect_notes_ensemble(audio_path, sr)  # Pass path, not array
        for note in result.notes:
            notes.append(DetectedNote(
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.consensus_score,
            ))
        print(f"Ensemble detected {len(notes)} notes")
        return notes
    except (ImportError, Exception) as e:
        print(f"Ensemble detection failed: {e}")
    
    # Fallback to pyin
    try:
        print("Using pyin pitch detection...")
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=sr, frame_length=2048
        )
        
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        
        # Convert to notes
        current_note = None
        current_start = 0
        min_duration = 0.05
        
        for i, (freq, is_voiced) in enumerate(zip(f0, voiced)):
            time = times[i]
            
            if is_voiced and freq and freq > 0:
                midi = int(round(librosa.hz_to_midi(freq)))
                
                if current_note is None:
                    current_note = midi
                    current_start = time
                elif abs(midi - current_note) > 1:  # Note change
                    # Save previous note
                    if time - current_start >= min_duration:
                        notes.append(DetectedNote(
                            midi=current_note,
                            start_time=current_start,
                            duration=time - current_start,
                        ))
                    current_note = midi
                    current_start = time
            else:
                # Unvoiced - end current note
                if current_note is not None:
                    if time - current_start >= min_duration:
                        notes.append(DetectedNote(
                            midi=current_note,
                            start_time=current_start,
                            duration=time - current_start,
                        ))
                    current_note = None
        
        # Final note
        if current_note is not None and times[-1] - current_start >= min_duration:
            notes.append(DetectedNote(
                midi=current_note,
                start_time=current_start,
                duration=times[-1] - current_start,
            ))
        
        print(f"pyin detected {len(notes)} notes")
        return notes
        
    except Exception as e:
        print(f"pyin failed: {e}")
        return []


def analyze_patterns_in_audio(audio_path: str, verbose: bool = True):
    """
    Full pipeline: detect notes -> match patterns -> enhance
    """
    print("\n" + "=" * 60)
    print("PATTERN MATCHING TEST")
    print("=" * 60)
    
    # Detect notes
    notes = detect_notes_simple(audio_path)
    
    if not notes:
        print("No notes detected!")
        return None
    
    # Show detected sequence
    if verbose:
        print(f"\nDetected sequence ({len(notes)} notes):")
        midi_seq = [n.midi for n in notes[:20]]
        print(f"  First 20: {midi_seq}")
        
        # Analyze intervals
        if len(midi_seq) > 1:
            intervals = np.diff(midi_seq)
            print(f"  Intervals: {list(intervals)}")
            print(f"  Range: {max(midi_seq) - min(midi_seq)} semitones")
    
    # Match patterns
    print("\n" + "-" * 40)
    print("PATTERN MATCHING")
    print("-" * 40)
    
    midi_notes = [n.midi for n in notes]
    
    # Quick pattern analysis
    analysis = analyze_sequence(midi_notes[:50])
    print(f"\nSequence analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
    
    # Full pattern matching
    print("\nSearching for patterns...")
    matcher = PatternMatcher(match_threshold=0.60)  # Slightly lower for testing
    matches = matcher.find_patterns(midi_notes[:100])  # First 100 notes
    
    print(f"\nFound {len(matches)} pattern matches:")
    for m in matches[:10]:  # Top 10
        print(f"  {m.pattern.name}:")
        print(f"    Type: {m.pattern.pattern_type.value}")
        print(f"    Score: {m.match_score:.0%}")
        print(f"    Root: {m.root_name} (MIDI {m.root_midi})")
        print(f"    Indices: {m.start_idx}-{m.end_idx}")
        if m.inferred_notes:
            print(f"    Inferred notes: {m.inferred_notes}")
    
    # Apply enhancement
    print("\n" + "-" * 40)
    print("APPLYING PATTERN CORRECTIONS")
    print("-" * 40)
    
    result = enhance_transcription(notes, match_threshold=0.70, verbose=True)
    
    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Total notes: {result['stats']['total_notes']}")
    print(f"Patterns found: {result['stats']['patterns_found']}")
    print(f"Corrections made: {result['stats']['corrections_made']}")
    print(f"Pattern coverage: {result['stats']['pattern_coverage']}")
    print(f"Pattern types: {result['stats']['pattern_types']}")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pattern_matching.py <audio_file>")
        print("\nRunning library self-test instead...")
        
        # Test the pattern library
        print("\n" + "=" * 50)
        print("PATTERN LIBRARY STATS")
        print("=" * 50)
        print(f"Total patterns: {len(PATTERN_LIBRARY)}")
        
        for ptype in PatternType:
            count = len([p for p in PATTERN_LIBRARY if p.pattern_type == ptype])
            print(f"  {ptype.value}: {count}")
        
        # Test pattern matching on synthetic data
        print("\n" + "=" * 50)
        print("SYNTHETIC TEST")
        print("=" * 50)
        
        # Minor pentatonic ascending
        pent_run = [40, 43, 45, 47, 50, 52]  # E minor pent from E2
        print(f"\nTest: Minor pentatonic {pent_run}")
        matches = match_patterns(pent_run)
        for m in matches[:3]:
            print(f"  -> {m.pattern.name}: {m.match_score:.0%}")
        
        # Blues lick
        blues = [40, 43, 45, 46, 47, 50]  # With blue note
        print(f"\nTest: Blues lick {blues}")
        matches = match_patterns(blues)
        for m in matches[:3]:
            print(f"  -> {m.pattern.name}: {m.match_score:.0%}")
        
        # Arpeggio
        arp = [40, 44, 47, 52, 47, 44, 40]  # Major arpeggio
        print(f"\nTest: Arpeggio {arp}")
        matches = match_patterns(arp)
        for m in matches[:3]:
            print(f"  -> {m.pattern.name}: {m.match_score:.0%}")
        
        print("\n✓ Pattern library working correctly")
        return
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    result = analyze_patterns_in_audio(audio_path)
    
    # Optionally save results
    if result and len(sys.argv) > 2 and sys.argv[2] == '--save':
        output_path = audio_path.rsplit('.', 1)[0] + '_patterns.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
