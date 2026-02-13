"""
Scale-Constrained Pitch Detection Module

Uses music theory to constrain detected pitches to valid scale notes.
This eliminates random wrong notes that aren't in the detected key.

The approach:
1. Run a quick first pass to detect key (using note histogram)
2. Build a set of valid MIDI notes for that key/scale
3. Snap all detected pitches to nearest valid scale note
4. This enforces musical coherence and eliminates false positives
"""

import numpy as np
from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass
import librosa

# Import from music_theory module
from music_theory import (
    detect_key, Key, SCALE_INTERVALS, NOTE_NAMES,
    snap_to_scale as snap_note_to_scale
)


@dataclass
class ScaleConstraintConfig:
    """Configuration for scale-constrained detection."""
    enabled: bool = True
    scale_type: Optional[str] = None  # None = auto from key, or 'pentatonic_minor', etc.
    detection_method: str = 'krumhansl'  # 'krumhansl' or 'simple'
    snap_direction: str = 'nearest'  # 'nearest', 'down', or 'up'
    allow_chromatic_passing: bool = False  # Allow non-scale notes as passing tones
    passing_max_duration: float = 0.1  # Max duration (sec) for passing tones
    min_notes_for_key: int = 8  # Minimum notes needed to detect key
    key_override: Optional[str] = None  # Manual key override (e.g., 'Am', 'C')
    verbose: bool = True


class ScaleConstrainer:
    """
    Constrains detected pitches to a musical scale.
    
    This class provides:
    - Key detection from a set of notes
    - MIDI note validation against scale
    - Pitch snapping to nearest scale note
    - Support for passing tones (chromatic notes)
    """
    
    def __init__(self, config: Optional[ScaleConstraintConfig] = None):
        self.config = config or ScaleConstraintConfig()
        self.detected_key: Optional[Key] = None
        self.valid_pitch_classes: Set[int] = set()
        self.scale_type: Optional[str] = None
    
    def detect_key_from_notes(self, notes: List) -> Key:
        """
        Detect the musical key from a list of notes.
        
        Args:
            notes: List of Note objects with .midi attribute
            
        Returns:
            Detected Key object
        """
        if self.config.key_override:
            from music_theory import parse_key_string
            key = parse_key_string(self.config.key_override)
            if self.config.verbose:
                print(f"🔑 Using specified key: {key.name}")
            return key
        
        if len(notes) < self.config.min_notes_for_key:
            # Not enough notes - default to most common (C major or A minor based on first note)
            if notes:
                first_pc = notes[0].midi % 12
                # Check if first note suggests minor
                if first_pc in [9, 4, 2]:  # A, E, D (common minor roots)
                    key = Key(first_pc, 'minor', 0.5)
                else:
                    key = Key(0, 'major', 0.5)  # Default C major
            else:
                key = Key(0, 'major', 0.0)
            
            if self.config.verbose:
                print(f"⚠️  Not enough notes for key detection ({len(notes)} < {self.config.min_notes_for_key})")
                print(f"    Using default: {key.name}")
            return key
        
        key = detect_key(notes, method=self.config.detection_method)
        
        if self.config.verbose:
            print(f"🔑 Auto-detected key: {key.name} (confidence: {key.confidence:.2f})")
        
        return key
    
    def setup_scale(self, key: Key, scale_type: Optional[str] = None) -> None:
        """
        Set up the valid pitch classes for the given key and scale.
        
        Args:
            key: Key object
            scale_type: Override scale type (e.g., 'pentatonic_minor')
        """
        self.detected_key = key
        self.scale_type = scale_type or self.config.scale_type
        
        # If no scale type specified, infer from key mode
        if self.scale_type is None:
            if key.mode == 'major':
                self.scale_type = 'major'
            else:
                self.scale_type = 'natural_minor'
        
        # Get scale intervals
        intervals = SCALE_INTERVALS.get(self.scale_type, SCALE_INTERVALS['major'])
        
        # Build set of valid pitch classes
        self.valid_pitch_classes = set()
        for interval in intervals:
            pc = (key.root + interval) % 12
            self.valid_pitch_classes.add(pc)
        
        if self.config.verbose:
            valid_names = [NOTE_NAMES[pc] for pc in sorted(self.valid_pitch_classes)]
            print(f"🎼 Scale: {key.short_name} {self.scale_type}")
            print(f"   Valid notes: {', '.join(valid_names)}")
    
    def is_valid_note(self, midi_note: int) -> bool:
        """Check if a MIDI note is in the current scale."""
        if not self.valid_pitch_classes:
            return True  # No scale set, accept all
        return (midi_note % 12) in self.valid_pitch_classes
    
    def snap_to_scale(self, midi_note: int) -> int:
        """
        Snap a MIDI note to the nearest note in the scale.
        
        Args:
            midi_note: Input MIDI note number
            
        Returns:
            Adjusted MIDI note number (constrained to scale)
        """
        if not self.detected_key:
            return midi_note
        
        if self.is_valid_note(midi_note):
            return midi_note
        
        # Use music_theory's snap function
        return snap_note_to_scale(
            midi_note,
            self.detected_key,
            self.scale_type,
            direction=self.config.snap_direction
        )
    
    def snap_frequency_to_scale(self, freq_hz: float) -> float:
        """
        Snap a frequency to the nearest scale note frequency.
        
        Args:
            freq_hz: Input frequency in Hz
            
        Returns:
            Adjusted frequency in Hz (constrained to scale)
        """
        if freq_hz <= 0 or not self.detected_key:
            return freq_hz
        
        # Convert to MIDI, snap, convert back
        midi = librosa.hz_to_midi(freq_hz)
        snapped_midi = self.snap_to_scale(int(round(midi)))
        return librosa.midi_to_hz(snapped_midi)
    
    def constrain_notes(
        self,
        notes: List,
        allow_passing: bool = None
    ) -> Tuple[List, Dict]:
        """
        Constrain a list of notes to the scale.
        
        Args:
            notes: List of Note objects
            allow_passing: Allow short chromatic passing tones
            
        Returns:
            Tuple of (constrained_notes, statistics)
        """
        from copy import deepcopy
        
        if allow_passing is None:
            allow_passing = self.config.allow_chromatic_passing
        
        constrained = []
        stats = {
            'total': len(notes),
            'in_scale': 0,
            'snapped': 0,
            'passing_tones': 0,
            'snap_distances': []
        }
        
        for note in notes:
            new_note = deepcopy(note)
            original_midi = note.midi
            
            # Check if already in scale
            if self.is_valid_note(original_midi):
                stats['in_scale'] += 1
                constrained.append(new_note)
                continue
            
            # Check if it could be a passing tone
            if allow_passing and note.duration <= self.config.passing_max_duration:
                stats['passing_tones'] += 1
                constrained.append(new_note)
                continue
            
            # Snap to scale
            new_note.midi = self.snap_to_scale(original_midi)
            snap_distance = abs(new_note.midi - original_midi)
            stats['snapped'] += 1
            stats['snap_distances'].append(snap_distance)
            constrained.append(new_note)
        
        if self.config.verbose and stats['snapped'] > 0:
            avg_dist = np.mean(stats['snap_distances']) if stats['snap_distances'] else 0
            print(f"🎯 Scale constraint results:")
            print(f"   In scale: {stats['in_scale']}/{stats['total']}")
            print(f"   Snapped: {stats['snapped']} (avg distance: {avg_dist:.1f} semitones)")
            if stats['passing_tones'] > 0:
                print(f"   Passing tones (kept): {stats['passing_tones']}")
        
        return constrained, stats
    
    def constrain_pitch_array(
        self,
        f0: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        min_confidence: float = 0.3
    ) -> np.ndarray:
        """
        Constrain an array of pitch values (in Hz) to the scale.
        
        This is useful for constraining pitches during detection,
        before they're converted to note events.
        
        Args:
            f0: Array of fundamental frequencies in Hz (0 = unvoiced)
            confidence: Optional confidence array
            min_confidence: Minimum confidence to process
            
        Returns:
            Constrained f0 array
        """
        if not self.detected_key:
            return f0
        
        constrained = f0.copy()
        
        for i in range(len(f0)):
            if f0[i] <= 0:
                continue
            if confidence is not None and confidence[i] < min_confidence:
                continue
            
            constrained[i] = self.snap_frequency_to_scale(f0[i])
        
        return constrained


def detect_key_quick(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_samples: int = 100,
    verbose: bool = True
) -> Key:
    """
    Quick key detection using chroma features directly from audio.
    
    This is faster than full note detection for initial key estimation.
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for chroma
        n_samples: Number of frames to sample
        verbose: Print info
        
    Returns:
        Detected Key object
    """
    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Average chroma (pitch class distribution)
    chroma_avg = np.mean(chroma, axis=1)
    
    # Normalize
    if chroma_avg.sum() > 0:
        chroma_avg = chroma_avg / chroma_avg.sum()
    
    # Use Krumhansl-Schmuckler profiles
    from music_theory import MAJOR_PROFILE, MINOR_PROFILE
    
    best_key = None
    best_corr = -2
    
    for root in range(12):
        # Rotate chroma to test this root
        rotated = np.roll(chroma_avg, -root)
        
        # Test major
        major_corr = np.corrcoef(rotated, MAJOR_PROFILE / MAJOR_PROFILE.sum())[0, 1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = Key(root, 'major', float(major_corr))
        
        # Test minor
        minor_corr = np.corrcoef(rotated, MINOR_PROFILE / MINOR_PROFILE.sum())[0, 1]
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = Key(root, 'minor', float(minor_corr))
    
    if verbose and best_key:
        print(f"🔑 Quick key detection: {best_key.name} (confidence: {best_key.confidence:.2f})")
    
    return best_key if best_key else Key(0, 'major', 0.0)


def apply_scale_constraint_to_detection(
    audio_path: str,
    notes: List,
    config: Optional[ScaleConstraintConfig] = None,
    sr: int = 22050
) -> Tuple[List, Key, Dict]:
    """
    Apply scale constraints to detected notes.
    
    This is the main entry point for scale-constrained detection.
    It detects the key, sets up the scale, and constrains all notes.
    
    Args:
        audio_path: Path to audio file (for quick key detection if needed)
        notes: List of detected Note objects
        config: Scale constraint configuration
        sr: Sample rate
        
    Returns:
        Tuple of (constrained_notes, detected_key, statistics)
    """
    config = config or ScaleConstraintConfig()
    constrainer = ScaleConstrainer(config)
    
    # Detect key from notes (or use override)
    key = constrainer.detect_key_from_notes(notes)
    
    # If key confidence is low, try quick audio-based detection
    if key.confidence < 0.5 and not config.key_override:
        try:
            y, sr_loaded = librosa.load(audio_path, sr=sr, duration=30)  # First 30 sec
            audio_key = detect_key_quick(y, sr_loaded, verbose=config.verbose)
            
            # Use audio key if more confident
            if audio_key.confidence > key.confidence:
                if config.verbose:
                    print(f"   Using audio-based key: {audio_key.name} (higher confidence)")
                key = audio_key
        except Exception as e:
            if config.verbose:
                print(f"⚠️  Audio key detection failed: {e}")
    
    # Set up scale
    constrainer.setup_scale(key, config.scale_type)
    
    # Constrain notes
    constrained, stats = constrainer.constrain_notes(notes)
    
    return constrained, key, stats


def add_scale_constrain_args(parser) -> None:
    """Add scale constraint arguments to argument parser."""
    group = parser.add_argument_group('Scale Constraint Options')
    
    group.add_argument('--scale-constrain', action='store_true',
                       help='Enable scale-constrained detection: snap detected pitches to '
                            'nearest note in the detected key/scale. Eliminates wrong notes '
                            'that are out of key.')
    
    group.add_argument('--scale-constrain-type', type=str, default=None,
                       choices=['major', 'natural_minor', 'harmonic_minor', 'melodic_minor',
                                'pentatonic_major', 'pentatonic_minor', 'blues',
                                'dorian', 'phrygian', 'lydian', 'mixolydian'],
                       help='Scale type for constraint (default: auto from key mode)')
    
    group.add_argument('--scale-snap-direction', type=str, default='nearest',
                       choices=['nearest', 'down', 'up'],
                       help='Direction to snap out-of-scale notes (default: nearest)')
    
    group.add_argument('--allow-passing-tones', action='store_true',
                       help='Allow short chromatic passing tones (notes < 100ms)')
    
    group.add_argument('--passing-tone-max-ms', type=float, default=100,
                       help='Maximum duration (ms) for passing tones (default: 100)')


def config_from_args(args) -> ScaleConstraintConfig:
    """Create ScaleConstraintConfig from parsed arguments."""
    return ScaleConstraintConfig(
        enabled=getattr(args, 'scale_constrain', False),
        scale_type=getattr(args, 'scale_constrain_type', None),
        snap_direction=getattr(args, 'scale_snap_direction', 'nearest'),
        allow_chromatic_passing=getattr(args, 'allow_passing_tones', False),
        passing_max_duration=getattr(args, 'passing_tone_max_ms', 100) / 1000.0,
        key_override=getattr(args, 'key', None),  # Use existing --key arg
        verbose=True
    )
