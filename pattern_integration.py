#!/usr/bin/env python3
"""
Pattern Integration Module

Integrates pattern matching with pitch detection pipeline:
1. Detect notes using standard pitch detection
2. Analyze sequences for pattern matches
3. If 70%+ match found, use pattern to correct/fill notes
4. Output improved tab with pattern annotations

This makes transcription more reliable by leveraging
musical knowledge (patterns) rather than pure signal processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json

from pattern_library import (
    PatternMatcher, PatternMatch, PatternType, GuitarPattern,
    PATTERN_LIBRARY, match_patterns, correct_with_patterns,
    analyze_sequence, find_similar_patterns
)


@dataclass
class DetectedNote:
    """Simplified note representation for pattern matching."""
    midi: int
    start_time: float
    duration: float
    confidence: float = 1.0
    source: str = "detected"  # detected, pattern_inferred, pattern_corrected
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    @property
    def note_name(self) -> str:
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return names[self.midi % 12] + str(self.midi // 12 - 1)


@dataclass
class PatternSegment:
    """A segment of notes identified as a pattern."""
    start_idx: int
    end_idx: int
    pattern: GuitarPattern
    match_score: float
    original_notes: List[DetectedNote]
    corrected_notes: List[DetectedNote]
    inferred_count: int = 0
    corrections_made: List[str] = field(default_factory=list)


class PatternEnhancedTranscription:
    """
    Enhances transcription by applying pattern recognition.
    
    Pipeline:
    1. Receive notes from pitch detection
    2. Segment into potential phrases/runs
    3. Match each segment against pattern library
    4. Apply corrections where match >= 70%
    5. Output enhanced note list with annotations
    """
    
    def __init__(
        self,
        match_threshold: float = 0.70,
        min_segment_size: int = 4,
        max_segment_gap_ms: float = 150,  # Gap to break phrases
        enable_inferrence: bool = True,   # Fill in missed notes
        enable_octave_fix: bool = True,   # Fix octave errors
    ):
        self.match_threshold = match_threshold
        self.min_segment_size = min_segment_size
        self.max_segment_gap_ms = max_segment_gap_ms / 1000.0
        self.enable_inferrence = enable_inferrence
        self.enable_octave_fix = enable_octave_fix
        
        self.matcher = PatternMatcher(match_threshold=match_threshold)
        self.segments_found: List[PatternSegment] = []
    
    def _segment_notes(self, notes: List[DetectedNote]) -> List[List[int]]:
        """
        Break notes into segments based on timing gaps.
        
        Returns list of (start_idx, end_idx) tuples.
        """
        if len(notes) < self.min_segment_size:
            return [(0, len(notes))]
        
        segments = []
        seg_start = 0
        
        for i in range(1, len(notes)):
            gap = notes[i].start_time - notes[i-1].end_time
            
            # Break on large gaps
            if gap > self.max_segment_gap_ms:
                if i - seg_start >= self.min_segment_size:
                    segments.append((seg_start, i))
                seg_start = i
        
        # Add final segment
        if len(notes) - seg_start >= self.min_segment_size:
            segments.append((seg_start, len(notes)))
        
        return segments
    
    def _analyze_segment(
        self,
        notes: List[DetectedNote],
        start_idx: int,
        end_idx: int
    ) -> Optional[PatternSegment]:
        """
        Analyze a segment of notes for pattern matches.
        
        Returns PatternSegment if pattern found, None otherwise.
        """
        segment_notes = notes[start_idx:end_idx]
        midi_notes = [n.midi for n in segment_notes]
        
        # Find best pattern match
        matches = self.matcher.find_patterns(midi_notes)
        
        if not matches:
            return None
        
        best_match = matches[0]
        
        if best_match.match_score < self.match_threshold:
            return None
        
        # Apply corrections
        corrected_notes = list(segment_notes)
        corrections = []
        inferred_count = 0
        
        # Octave corrections
        if self.enable_octave_fix:
            expected_midi = best_match.pattern.as_midi_notes(best_match.root_midi)
            for i, note in enumerate(corrected_notes):
                if i < len(expected_midi):
                    exp = expected_midi[i]
                    if abs(note.midi - exp) == 12:
                        # Octave error detected
                        old_midi = note.midi
                        corrected_notes[i] = DetectedNote(
                            midi=exp,
                            start_time=note.start_time,
                            duration=note.duration,
                            confidence=note.confidence * 0.9,
                            source="pattern_corrected"
                        )
                        corrections.append(
                            f"Octave fix at {note.start_time:.3f}s: {old_midi} -> {exp}"
                        )
        
        # Fill in inferred notes
        if self.enable_inferrence and best_match.inferred_notes:
            inferred_count = len(best_match.inferred_notes)
            # Note: actual insertion would require timing estimation
            # For now, just track that notes could be inferred
            corrections.append(
                f"Pattern suggests {inferred_count} potentially missed notes"
            )
        
        return PatternSegment(
            start_idx=start_idx,
            end_idx=end_idx,
            pattern=best_match.pattern,
            match_score=best_match.match_score,
            original_notes=segment_notes,
            corrected_notes=corrected_notes,
            inferred_count=inferred_count,
            corrections_made=corrections
        )
    
    def process(
        self,
        notes: List[DetectedNote]
    ) -> Tuple[List[DetectedNote], List[PatternSegment]]:
        """
        Process notes through pattern matching pipeline.
        
        Returns:
            (enhanced_notes, pattern_segments)
        """
        if len(notes) < self.min_segment_size:
            return notes, []
        
        self.segments_found = []
        enhanced_notes = list(notes)
        
        # Segment notes by timing
        segments = self._segment_notes(notes)
        
        # Analyze each segment
        pattern_indices = set()  # Track which indices were pattern-matched
        
        for seg_start, seg_end in segments:
            segment = self._analyze_segment(notes, seg_start, seg_end)
            
            if segment:
                self.segments_found.append(segment)
                
                # Apply corrections to enhanced_notes
                for i, corrected in enumerate(segment.corrected_notes):
                    global_idx = seg_start + i
                    enhanced_notes[global_idx] = corrected
                    pattern_indices.add(global_idx)
        
        return enhanced_notes, self.segments_found
    
    def get_pattern_annotations(self) -> List[Dict]:
        """
        Get pattern annotations for tab output.
        
        Returns list of annotations with pattern info.
        """
        annotations = []
        
        for seg in self.segments_found:
            annotations.append({
                'start_time': seg.original_notes[0].start_time,
                'end_time': seg.original_notes[-1].end_time,
                'pattern_name': seg.pattern.name,
                'pattern_type': seg.pattern.pattern_type.value,
                'match_score': f"{seg.match_score:.0%}",
                'corrections': seg.corrections_made,
                'description': seg.pattern.description,
            })
        
        return annotations


def convert_notes_format(notes_list: List[Any]) -> List[DetectedNote]:
    """
    Convert notes from various formats to DetectedNote.
    
    Handles:
    - Dicts with 'midi', 'start_time', 'duration'
    - Objects with .midi, .start_time, .duration attributes
    - Tuples of (midi, start_time, duration)
    """
    converted = []
    
    for note in notes_list:
        if isinstance(note, DetectedNote):
            converted.append(note)
        elif isinstance(note, dict):
            converted.append(DetectedNote(
                midi=note.get('midi', note.get('pitch', 0)),
                start_time=note.get('start_time', note.get('onset', 0)),
                duration=note.get('duration', 0.1),
                confidence=note.get('confidence', 1.0),
            ))
        elif isinstance(note, (list, tuple)) and len(note) >= 3:
            converted.append(DetectedNote(
                midi=int(note[0]),
                start_time=float(note[1]),
                duration=float(note[2]),
            ))
        elif hasattr(note, 'midi'):
            converted.append(DetectedNote(
                midi=getattr(note, 'midi'),
                start_time=getattr(note, 'start_time', 0),
                duration=getattr(note, 'duration', 0.1),
                confidence=getattr(note, 'confidence', 1.0),
            ))
    
    return converted


def enhance_transcription(
    notes: List[Any],
    match_threshold: float = 0.70,
    verbose: bool = False
) -> Dict:
    """
    Main entry point for pattern-enhanced transcription.
    
    Args:
        notes: List of notes in any supported format
        match_threshold: Minimum pattern match score (default 70%)
        verbose: Print debug info
        
    Returns:
        Dict with 'notes', 'patterns', 'stats'
    """
    # Convert to standard format
    converted = convert_notes_format(notes)
    
    if verbose:
        print(f"Processing {len(converted)} notes...")
    
    # Create enhancer and process
    enhancer = PatternEnhancedTranscription(match_threshold=match_threshold)
    enhanced_notes, segments = enhancer.process(converted)
    
    # Get annotations
    annotations = enhancer.get_pattern_annotations()
    
    # Compile stats
    total_corrections = sum(len(s.corrections_made) for s in segments)
    pattern_coverage = sum(s.end_idx - s.start_idx for s in segments) / max(len(converted), 1)
    
    stats = {
        'total_notes': len(converted),
        'patterns_found': len(segments),
        'corrections_made': total_corrections,
        'pattern_coverage': f"{pattern_coverage:.0%}",
        'pattern_types': list(set(s.pattern.pattern_type.value for s in segments)),
    }
    
    if verbose:
        print(f"Found {len(segments)} pattern matches")
        print(f"Made {total_corrections} corrections")
        print(f"Pattern coverage: {pattern_coverage:.0%}")
        for ann in annotations:
            print(f"  - {ann['pattern_name']} ({ann['match_score']}): {ann['description']}")
    
    # Convert back to list of dicts
    output_notes = []
    for note in enhanced_notes:
        output_notes.append({
            'midi': note.midi,
            'start_time': note.start_time,
            'duration': note.duration,
            'confidence': note.confidence,
            'source': note.source,
            'note_name': note.note_name,
        })
    
    return {
        'notes': output_notes,
        'patterns': annotations,
        'stats': stats,
    }


# =============================================================================
# Integration with Existing Pipeline
# =============================================================================

def add_pattern_args(parser):
    """Add pattern matching arguments to argument parser."""
    group = parser.add_argument_group('Pattern Matching')
    group.add_argument(
        '--pattern-match', action='store_true',
        help='Enable pattern-based note correction'
    )
    group.add_argument(
        '--pattern-threshold', type=float, default=0.70,
        help='Minimum pattern match score (default: 0.70 = 70%%)'
    )
    group.add_argument(
        '--pattern-verbose', action='store_true',
        help='Print pattern matching details'
    )
    group.add_argument(
        '--no-pattern-infer', action='store_true',
        help='Disable inferring missed notes from patterns'
    )
    group.add_argument(
        '--no-pattern-octave', action='store_true',
        help='Disable octave correction from patterns'
    )


def apply_pattern_enhancement_to_notes(
    notes: List,
    args=None,
    match_threshold: float = 0.70,
    verbose: bool = False
) -> Tuple[List, Dict]:
    """
    Apply pattern enhancement to notes list from guitar_tabs.py
    
    Returns:
        (enhanced_notes, pattern_info)
    """
    # Get settings from args if provided
    if args:
        match_threshold = getattr(args, 'pattern_threshold', 0.70)
        verbose = getattr(args, 'pattern_verbose', False)
        enable_infer = not getattr(args, 'no_pattern_infer', False)
        enable_octave = not getattr(args, 'no_pattern_octave', False)
    else:
        enable_infer = True
        enable_octave = True
    
    enhancer = PatternEnhancedTranscription(
        match_threshold=match_threshold,
        enable_inferrence=enable_infer,
        enable_octave_fix=enable_octave,
    )
    
    # Convert notes
    converted = convert_notes_format(notes)
    
    # Process
    enhanced, segments = enhancer.process(converted)
    
    # Build pattern info
    pattern_info = {
        'segments': enhancer.get_pattern_annotations(),
        'stats': {
            'patterns_found': len(segments),
            'corrections': sum(len(s.corrections_made) for s in segments),
        }
    }
    
    if verbose:
        print(f"\n[Pattern Matching] Found {len(segments)} patterns:")
        for seg in segments:
            print(f"  {seg.pattern.name}: {seg.match_score:.0%} match")
            for corr in seg.corrections_made:
                print(f"    - {corr}")
    
    # Convert back to original note format (preserve attributes)
    enhanced_notes = []
    for i, note in enumerate(enhanced):
        if i < len(notes):
            # Copy original note and update MIDI
            original = notes[i]
            if hasattr(original, '__dict__'):
                # Object - create copy with updated midi
                from copy import copy
                updated = copy(original)
                updated.midi = note.midi
                if hasattr(updated, 'pattern_corrected'):
                    updated.pattern_corrected = (note.source != "detected")
                enhanced_notes.append(updated)
            elif isinstance(original, dict):
                # Dict - update copy
                updated = dict(original)
                updated['midi'] = note.midi
                updated['pattern_corrected'] = (note.source != "detected")
                enhanced_notes.append(updated)
            else:
                # Other - just use converted note
                enhanced_notes.append(note)
        else:
            enhanced_notes.append(note)
    
    return enhanced_notes, pattern_info


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test pattern integration')
    parser.add_argument('--test', action='store_true', help='Run test cases')
    add_pattern_args(parser)
    args = parser.parse_args()
    
    if args.test:
        print("Testing Pattern Integration")
        print("=" * 50)
        
        # Create test notes (minor pentatonic run with one "wrong" note)
        test_notes = [
            DetectedNote(midi=64, start_time=0.0, duration=0.1),   # E4
            DetectedNote(midi=67, start_time=0.1, duration=0.1),   # G4
            DetectedNote(midi=69, start_time=0.2, duration=0.1),   # A4
            DetectedNote(midi=71, start_time=0.3, duration=0.1),   # B4
            DetectedNote(midi=86, start_time=0.4, duration=0.1),   # D5 (octave error: should be 74)
            DetectedNote(midi=76, start_time=0.5, duration=0.1),   # E5
        ]
        
        print(f"Input: {[n.midi for n in test_notes]}")
        print(f"Note names: {[n.note_name for n in test_notes]}")
        
        # Process
        result = enhance_transcription(test_notes, verbose=True)
        
        print(f"\nOutput: {[n['midi'] for n in result['notes']]}")
        print(f"Stats: {result['stats']}")
    else:
        print("Use --test to run test cases")
        print("Use --help to see options")
