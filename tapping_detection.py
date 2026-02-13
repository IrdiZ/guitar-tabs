#!/usr/bin/env python3
"""
Two-Hand Tapping Detection for Guitar Tabs

Detects two-hand tapping techniques common in lead guitar solos:
- Right hand taps on the fretboard (high notes)
- Creates distinctive attack patterns (legato-like, no pick)
- Often involves wide interval jumps (>7 frets)
- Pull-offs from tapped notes back to fretted notes

Tapping Notation:
  t12-p5-h7  - Tap at fret 12, pull-off to 5, hammer-on to 7
  t15        - Simple tap at fret 15
  
Common Tapping Patterns:
  - Van Halen style: t12-p8-h5 (octave + 5th arpeggios)
  - Sequential tapping arpeggios
  - Tapped harmonics (not yet supported)

Author: Guitar Tab Generator
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# Minimum interval (in semitones/frets) to consider as possible tap
# Normal fingering rarely jumps >7 frets in legato
MIN_TAP_INTERVAL = 7

# Maximum interval for a tap (typical guitar range limit)
MAX_TAP_INTERVAL = 19  # More than an octave + fifth

# Minimum fret for a tap (right hand taps are usually high on neck)
MIN_TAP_FRET = 10  # Frets 10+ are common tap targets


@dataclass
class TapCandidate:
    """A potential tapped note."""
    note_index: int
    fret: int
    midi: int
    start_time: float
    duration: float
    interval_from_prev: int  # Semitones from previous note
    interval_to_next: int    # Semitones to next note
    is_legato: bool
    attack_strength: float   # Lower = more like tap
    confidence: float = 0.0


@dataclass
class TappingPattern:
    """A detected tapping pattern (e.g., t12-p5-h7)."""
    start_index: int
    end_index: int
    pattern_type: str  # 'single', 'arpeggio', 'sequence'
    tap_indices: List[int]  # Indices of tapped notes
    notation: str  # e.g., "t12-p5-h7"
    confidence: float


class TappingDetector:
    """
    Detects two-hand tapping in guitar audio.
    
    Two-hand tapping characteristics:
    1. Wide interval jumps (>7 frets) executed with legato attack
    2. Tapping notes typically on high frets (12+)
    3. Often followed by pull-offs to lower fretted notes
    4. Attack strength is lower than picked notes (no pick)
    5. Patterns repeat (arpeggios, sequences)
    
    Detection strategy:
    1. Find notes with wide interval jumps + legato onset
    2. Check if destination note is in typical tap range (fret 12+)
    3. Analyze attack characteristics (should be soft)
    4. Look for tap-pull-hammer patterns
    5. Detect repeating tap arpeggios
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        # Tap detection thresholds
        min_tap_interval: int = MIN_TAP_INTERVAL,
        max_tap_interval: int = MAX_TAP_INTERVAL,
        min_tap_fret: int = MIN_TAP_FRET,
        # Attack analysis
        tap_attack_threshold: float = 0.4,  # Max attack strength for tap
        # Pattern detection
        min_pattern_length: int = 3,  # Minimum notes in a tapping pattern
        pattern_time_window: float = 0.5,  # Max time for pattern notes
    ):
        self.sr = sr
        self.hop_length = hop_length
        
        self.min_tap_interval = min_tap_interval
        self.max_tap_interval = max_tap_interval
        self.min_tap_fret = min_tap_fret
        
        self.tap_attack_threshold = tap_attack_threshold
        
        self.min_pattern_length = min_pattern_length
        self.pattern_time_window = pattern_time_window
    
    def detect_taps(
        self,
        y: np.ndarray,
        notes: List[Any],  # Note objects with midi, start_time, duration
        onset_details: Optional[List[Any]] = None,  # EnsembleOnset objects
        tuning: List[int] = None,
        verbose: bool = True
    ) -> Tuple[List[int], List[TappingPattern]]:
        """
        Detect tapped notes and tapping patterns.
        
        Args:
            y: Audio signal
            notes: List of Note objects
            onset_details: EnsembleOnset objects with attack/legato info
            tuning: Guitar tuning as MIDI notes
            verbose: Print detection info
            
        Returns:
            Tuple of:
            - tap_indices: List of indices of tapped notes
            - patterns: List of TappingPattern objects
        """
        if tuning is None:
            tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
        
        if len(notes) < 2:
            return [], []
        
        # Sort notes by time
        sorted_notes = sorted(enumerate(notes), key=lambda x: x[1].start_time)
        original_indices = [idx for idx, _ in sorted_notes]
        notes_sorted = [n for _, n in sorted_notes]
        
        # Build onset lookup
        onset_map = {}
        if onset_details:
            for onset in onset_details:
                onset_map[round(onset.time * 1000)] = onset
        
        # Compute attack strength for each note
        attack_strengths = self._compute_attack_strengths(y, notes_sorted)
        
        # Phase 1: Find tap candidates
        tap_candidates = self._find_tap_candidates(
            notes_sorted, original_indices, onset_map, 
            attack_strengths, tuning
        )
        
        if verbose and tap_candidates:
            print(f"  Tapping: Found {len(tap_candidates)} tap candidates")
        
        # Phase 2: Validate candidates
        validated_taps = self._validate_tap_candidates(
            tap_candidates, notes_sorted, y
        )
        
        if verbose and validated_taps:
            print(f"  Tapping: Validated {len(validated_taps)} taps")
        
        # Phase 3: Detect tapping patterns
        patterns = self._detect_tapping_patterns(
            validated_taps, notes_sorted, tuning
        )
        
        if verbose and patterns:
            print(f"  Tapping: Found {len(patterns)} tapping patterns")
            for p in patterns[:3]:  # Show first 3
                print(f"    Pattern: {p.notation} ({p.pattern_type})")
        
        # Return original indices of tapped notes
        tap_indices = [c.note_index for c in validated_taps]
        
        return tap_indices, patterns
    
    def _compute_attack_strengths(
        self,
        y: np.ndarray,
        notes: List[Any]
    ) -> Dict[int, float]:
        """
        Compute attack strength for each note.
        
        Attack strength is based on:
        - High frequency transient content at onset
        - Rate of energy rise
        
        Tapped notes have lower attack strength than picked notes.
        """
        attack_strengths = {}
        
        # High-pass filter for transient detection
        from scipy.signal import butter, filtfilt
        nyq = self.sr / 2
        high_cutoff = min(2000 / nyq, 0.95)
        
        try:
            b, a = butter(4, high_cutoff, btype='high')
            y_hp = filtfilt(b, a, y)
        except Exception:
            y_hp = y
        
        for i, note in enumerate(notes):
            start_sample = int(note.start_time * self.sr)
            # Analyze just the attack portion (first 20ms)
            attack_window = int(0.02 * self.sr)
            end_sample = min(start_sample + attack_window, len(y_hp))
            
            if start_sample >= len(y_hp):
                attack_strengths[i] = 0.5
                continue
            
            segment = y_hp[start_sample:end_sample]
            
            if len(segment) == 0:
                attack_strengths[i] = 0.5
                continue
            
            # Attack strength = peak of HF transient
            peak = np.max(np.abs(segment))
            
            # Also check energy rise rate
            envelope = np.abs(segment)
            if len(envelope) > 1:
                rise_rate = np.max(np.diff(envelope))
            else:
                rise_rate = 0
            
            # Combine: higher transient + faster rise = stronger attack
            strength = (peak + rise_rate) / 2
            
            # Normalize to 0-1 range (rough normalization)
            strength = min(1.0, strength * 10)  # Scale factor
            
            attack_strengths[i] = strength
        
        # Normalize across all notes
        if attack_strengths:
            max_strength = max(attack_strengths.values())
            if max_strength > 0:
                for i in attack_strengths:
                    attack_strengths[i] /= max_strength
        
        return attack_strengths
    
    def _find_tap_candidates(
        self,
        notes: List[Any],
        original_indices: List[int],
        onset_map: Dict[int, Any],
        attack_strengths: Dict[int, float],
        tuning: List[int]
    ) -> List[TapCandidate]:
        """
        Find notes that could be taps based on interval and onset characteristics.
        """
        candidates = []
        
        for i, note in enumerate(notes):
            # Calculate intervals
            interval_from_prev = 0
            interval_to_next = 0
            
            if i > 0:
                interval_from_prev = abs(note.midi - notes[i-1].midi)
            if i < len(notes) - 1:
                interval_to_next = abs(notes[i+1].midi - note.midi)
            
            # Get fret position
            fret = self._get_fret(note.midi, tuning)
            if fret is None:
                continue
            
            # Check if this could be a tap
            is_tap_candidate = False
            
            # Criterion 1: Wide interval from previous note
            if interval_from_prev >= self.min_tap_interval:
                is_tap_candidate = True
            
            # Criterion 2: High fret + followed by wide jump down
            if fret >= self.min_tap_fret and interval_to_next >= self.min_tap_interval:
                is_tap_candidate = True
            
            if not is_tap_candidate:
                continue
            
            # Check onset characteristics
            is_legato = False
            onset_key = round(note.start_time * 1000)
            for key in range(onset_key - 50, onset_key + 51):
                if key in onset_map:
                    onset = onset_map[key]
                    if hasattr(onset, 'is_legato') and onset.is_legato:
                        is_legato = True
                    break
            
            attack_strength = attack_strengths.get(i, 0.5)
            
            # Calculate confidence
            confidence = 0.0
            
            # Wider interval = more likely tap
            max_interval = max(interval_from_prev, interval_to_next)
            if max_interval >= 12:  # Octave or more
                confidence += 0.4
            elif max_interval >= 7:
                confidence += 0.2
            
            # High fret = more likely tap
            if fret >= 15:
                confidence += 0.3
            elif fret >= 12:
                confidence += 0.2
            elif fret >= 10:
                confidence += 0.1
            
            # Legato onset = more likely tap
            if is_legato:
                confidence += 0.2
            
            # Weak attack = more likely tap
            if attack_strength < self.tap_attack_threshold:
                confidence += 0.2
            
            candidates.append(TapCandidate(
                note_index=original_indices[i],
                fret=fret,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                interval_from_prev=interval_from_prev,
                interval_to_next=interval_to_next,
                is_legato=is_legato,
                attack_strength=attack_strength,
                confidence=confidence
            ))
        
        return candidates
    
    def _validate_tap_candidates(
        self,
        candidates: List[TapCandidate],
        notes: List[Any],
        y: np.ndarray
    ) -> List[TapCandidate]:
        """
        Validate tap candidates based on additional analysis.
        
        Validation checks:
        1. Confidence threshold
        2. Context: surrounded by legato notes
        3. Not a simple position shift
        """
        validated = []
        
        for cand in candidates:
            # Minimum confidence threshold
            if cand.confidence < 0.4:
                continue
            
            # Wide interval with weak attack is strong indicator
            if (cand.interval_from_prev >= self.min_tap_interval and 
                cand.attack_strength < self.tap_attack_threshold):
                validated.append(cand)
                continue
            
            # High fret with wide interval following is also good
            if (cand.fret >= self.min_tap_fret and 
                cand.interval_to_next >= self.min_tap_interval):
                validated.append(cand)
                continue
            
            # Legato onset with very wide interval
            if cand.is_legato and cand.interval_from_prev >= 10:
                validated.append(cand)
                continue
        
        return validated
    
    def _detect_tapping_patterns(
        self,
        tap_candidates: List[TapCandidate],
        notes: List[Any],
        tuning: List[int]
    ) -> List[TappingPattern]:
        """
        Detect tapping patterns (sequences of tap-pull-hammer notes).
        
        Common patterns:
        - t12-p5-h7: Tap, pull-off, hammer-on (Van Halen style)
        - t12-p8-p5: Tap, double pull-off
        - Sequential tapping arpeggios
        """
        patterns = []
        
        if not tap_candidates:
            return patterns
        
        # Sort by time
        sorted_cands = sorted(tap_candidates, key=lambda c: c.start_time)
        
        # Find sequences of notes around taps
        used_indices = set()
        
        for tap in sorted_cands:
            if tap.note_index in used_indices:
                continue
            
            # Look for notes immediately after the tap (pull-offs)
            pattern_notes = [tap]
            current_time = tap.start_time
            
            # Find subsequent notes (likely pull-offs and hammer-ons)
            for note in notes:
                if note.start_time <= tap.start_time:
                    continue
                
                time_gap = note.start_time - current_time
                if time_gap > self.pattern_time_window:
                    break
                
                # Check if this continues the legato sequence
                fret = self._get_fret(note.midi, tuning)
                if fret is None:
                    break
                
                # Create a pseudo-candidate for pattern tracking
                cand = TapCandidate(
                    note_index=-1,  # Will be identified as non-tap
                    fret=fret,
                    midi=note.midi,
                    start_time=note.start_time,
                    duration=note.duration,
                    interval_from_prev=abs(note.midi - pattern_notes[-1].midi),
                    interval_to_next=0,
                    is_legato=True,  # Assume legato in pattern
                    attack_strength=0.5,
                    confidence=0.0
                )
                
                pattern_notes.append(cand)
                current_time = note.start_time
                
                # Typical patterns are 3-4 notes
                if len(pattern_notes) >= 4:
                    break
            
            # Validate pattern
            if len(pattern_notes) >= self.min_pattern_length:
                # Determine pattern type
                pattern_type = self._classify_pattern(pattern_notes)
                
                # Generate notation
                notation = self._generate_tap_notation(pattern_notes)
                
                # Mark indices as used
                for pn in pattern_notes:
                    if pn.note_index >= 0:
                        used_indices.add(pn.note_index)
                
                patterns.append(TappingPattern(
                    start_index=tap.note_index,
                    end_index=pattern_notes[-1].note_index if pattern_notes[-1].note_index >= 0 else -1,
                    pattern_type=pattern_type,
                    tap_indices=[pn.note_index for pn in pattern_notes if pn.note_index >= 0],
                    notation=notation,
                    confidence=tap.confidence
                ))
        
        return patterns
    
    def _classify_pattern(self, pattern_notes: List[TapCandidate]) -> str:
        """Classify the type of tapping pattern."""
        if len(pattern_notes) < 2:
            return 'single'
        
        # Check if it's an arpeggio (notes form a chord pattern)
        intervals = [pattern_notes[i+1].midi - pattern_notes[i].midi 
                     for i in range(len(pattern_notes)-1)]
        
        # Arpeggio: alternating direction or consistent chord tones
        if len(set(abs(i) for i in intervals if i != 0)) <= 2:
            return 'arpeggio'
        
        # Sequential: mostly descending (typical tap-pulloff-pulloff)
        if sum(1 for i in intervals if i < 0) >= len(intervals) - 1:
            return 'sequence'
        
        return 'mixed'
    
    def _generate_tap_notation(self, pattern_notes: List[TapCandidate]) -> str:
        """
        Generate ASCII notation for a tapping pattern.
        
        Format: t12-p5-h7
        - t = tap
        - p = pull-off
        - h = hammer-on
        """
        if not pattern_notes:
            return ""
        
        parts = []
        
        for i, note in enumerate(pattern_notes):
            if i == 0:
                # First note is the tap
                parts.append(f"t{note.fret}")
            else:
                # Determine technique from interval
                interval = note.midi - pattern_notes[i-1].midi
                if interval < 0:
                    # Descending = pull-off
                    parts.append(f"p{note.fret}")
                else:
                    # Ascending = hammer-on
                    parts.append(f"h{note.fret}")
        
        return "-".join(parts)
    
    def _get_fret(self, midi: int, tuning: List[int]) -> Optional[int]:
        """Get the best fret position for a MIDI note."""
        best_fret = None
        best_score = -999
        
        for string_idx, open_note in enumerate(tuning):
            fret = midi - open_note
            if 0 <= fret <= 24:
                # Prefer higher frets for tap detection context
                score = fret * 0.1
                if score > best_score:
                    best_score = score
                    best_fret = fret
        
        return best_fret


def detect_tapping_in_audio(
    audio_path: str,
    notes: List[Any],
    onset_details: Optional[List[Any]] = None,
    tuning: List[int] = None,
    verbose: bool = True
) -> Tuple[List[int], List[TappingPattern]]:
    """
    High-level function to detect tapping in an audio file.
    
    Args:
        audio_path: Path to audio file
        notes: List of Note objects
        onset_details: EnsembleOnset objects (optional)
        tuning: Guitar tuning
        verbose: Print progress
        
    Returns:
        Tuple of (tap_indices, patterns)
    """
    import librosa
    
    if verbose:
        print(f"\n🎸 Detecting tapping in: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Create detector
    detector = TappingDetector(sr=sr)
    
    # Detect
    tap_indices, patterns = detector.detect_taps(
        y=y,
        notes=notes,
        onset_details=onset_details,
        tuning=tuning,
        verbose=verbose
    )
    
    return tap_indices, patterns


def add_tapping_to_technique_enum():
    """
    Returns the tap technique constant for integration with technique_detector.py.
    
    To integrate with the main Technique enum in technique_detector.py:
    
    ```python
    class Technique(Enum):
        ...
        TAP = "t"  # Add this line
    ```
    """
    return "t"


def format_tap_notation(fret: int, following_notes: List[Tuple[str, int]] = None) -> str:
    """
    Format tapping notation.
    
    Args:
        fret: The tapped fret
        following_notes: List of (technique, fret) tuples for the sequence
        
    Returns:
        Formatted notation string (e.g., "t12-p5-h7")
    """
    parts = [f"t{fret}"]
    
    if following_notes:
        for technique, f in following_notes:
            parts.append(f"{technique}{f}")
    
    return "-".join(parts)


# ============================================================================
# Integration with technique_detector.py
# ============================================================================

def integrate_tapping_detection(
    notes: List[Any],
    y: np.ndarray,
    sr: int,
    onset_details: Optional[List[Any]] = None,
    tuning: List[int] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Wrapper for integrating tapping detection into the main technique detector.
    
    Returns a dict mapping note indices to tapping info:
    {
        note_index: {
            'is_tap': True,
            'notation': 't12',
            'pattern': 't12-p5-h7' (if part of pattern),
            'confidence': 0.8
        }
    }
    """
    detector = TappingDetector(sr=sr)
    
    tap_indices, patterns = detector.detect_taps(
        y=y,
        notes=notes,
        onset_details=onset_details,
        tuning=tuning,
        verbose=False
    )
    
    result = {}
    
    # Add tap info for each tapped note
    for idx in tap_indices:
        note = notes[idx]
        fret = detector._get_fret(note.midi, tuning or [40, 45, 50, 55, 59, 64])
        
        result[idx] = {
            'is_tap': True,
            'notation': f"t{fret}",
            'pattern': None,
            'confidence': 0.7
        }
    
    # Update with pattern info
    for pattern in patterns:
        for idx in pattern.tap_indices:
            if idx in result:
                result[idx]['pattern'] = pattern.notation
                result[idx]['confidence'] = pattern.confidence
    
    return result


# ============================================================================
# Main / Test
# ============================================================================

def main():
    """Test tapping detection on a sample file."""
    import sys
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Detect two-hand tapping in guitar audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tapping notation:
  t12     - Tap at fret 12
  t12-p5  - Tap at 12, pull-off to 5
  t12-p5-h7 - Tap at 12, pull-off to 5, hammer-on to 7

Examples:
  python tapping_detection.py song.mp3
  python tapping_detection.py song.mp3 --verbose
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--tuning', default='standard',
                        help='Guitar tuning (standard, drop_d, etc.)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        return 1
    
    import librosa
    
    print(f"🎸 Analyzing: {args.audio_file}")
    
    # Load audio
    y, sr = librosa.load(args.audio_file, sr=22050, mono=True)
    
    # First detect notes using pYIN
    print("  Detecting pitches...")
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=75, fmax=1400, sr=sr, hop_length=512
    )
    
    # Detect onsets
    print("  Detecting onsets...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    # Create Note objects
    @dataclass
    class SimpleNote:
        midi: int
        start_time: float
        duration: float
        confidence: float
    
    notes = []
    hop_length = 512
    
    for i, onset_time in enumerate(onset_times):
        frame = int(onset_time * sr / hop_length)
        if frame < len(f0) and f0[frame] > 0:
            midi = int(round(librosa.hz_to_midi(f0[frame])))
            
            # Estimate duration
            if i < len(onset_times) - 1:
                duration = min(0.5, onset_times[i + 1] - onset_time)
            else:
                duration = 0.3
            
            notes.append(SimpleNote(
                midi=midi,
                start_time=onset_time,
                duration=max(0.05, duration),
                confidence=voiced_prob[frame] if frame < len(voiced_prob) else 0.5
            ))
    
    print(f"  Found {len(notes)} notes")
    
    # Get tuning
    TUNINGS = {
        'standard': [40, 45, 50, 55, 59, 64],
        'drop_d': [38, 45, 50, 55, 59, 64],
        'drop_c': [36, 43, 48, 53, 57, 62],
    }
    tuning = TUNINGS.get(args.tuning.lower(), TUNINGS['standard'])
    
    # Detect tapping
    detector = TappingDetector(sr=sr)
    tap_indices, patterns = detector.detect_taps(
        y=y,
        notes=notes,
        onset_details=None,
        tuning=tuning,
        verbose=True
    )
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Total notes: {len(notes)}")
    print(f"  Tapped notes: {len(tap_indices)}")
    print(f"  Tapping patterns: {len(patterns)}")
    
    if tap_indices:
        print(f"\n🎯 Tapped Notes:")
        for idx in tap_indices[:10]:  # Show first 10
            note = notes[idx]
            fret = detector._get_fret(note.midi, tuning)
            print(f"  {note.start_time:.2f}s: Fret {fret} (MIDI {note.midi})")
    
    if patterns:
        print(f"\n🎼 Tapping Patterns:")
        for p in patterns[:5]:  # Show first 5
            print(f"  {p.notation} ({p.pattern_type}, confidence: {p.confidence:.2f})")
    
    return 0


if __name__ == '__main__':
    import sys
    from dataclasses import dataclass
    sys.exit(main())
