#!/usr/bin/env python3
"""
Genre-Specific Optimization for Guitar Tab Detection

Provides genre-aware priors and biases for improved detection accuracy.
Currently supports:
- metal: Heavy distortion, fast alternate picking, palm muting, harmonic minor scales
- rock: Classic rock techniques, pentatonic/blues scales, power chords

The genre optimizer adjusts:
1. Scale priors (probability weights for pitch detection)
2. Technique detection sensitivity
3. Tempo and note density expectations
4. Pitch detection parameters
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class Genre(Enum):
    """Supported genres for optimization."""
    METAL = "metal"
    ROCK = "rock"
    BLUES = "blues"
    JAZZ = "jazz"
    ACOUSTIC = "acoustic"
    DEFAULT = "default"


# Note names for reference
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale intervals (semitones from root)
SCALE_INTERVALS = {
    'minor_pentatonic': [0, 3, 5, 7, 10],
    'major_pentatonic': [0, 2, 4, 7, 9],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'blues': [0, 3, 5, 6, 7, 10],
    'major': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10],  # Common in metal
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'chromatic': list(range(12)),
}


@dataclass
class GenreProfile:
    """
    Profile containing genre-specific parameters for detection optimization.
    """
    name: str
    display_name: str
    
    # Scale preferences (ordered by likelihood)
    preferred_scales: List[str] = field(default_factory=list)
    scale_weights: Dict[str, float] = field(default_factory=dict)
    
    # Common roots for the genre (pitch classes 0-11)
    common_roots: List[int] = field(default_factory=list)
    root_weights: Dict[int, float] = field(default_factory=dict)
    
    # Technique sensitivity adjustments (multipliers)
    technique_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # Expected tempo range (BPM)
    tempo_range: Tuple[int, int] = (80, 180)
    typical_tempo: int = 120
    
    # Note density expectations (notes per second)
    note_density_range: Tuple[float, float] = (2.0, 10.0)
    typical_density: float = 5.0
    
    # Pitch detection adjustments
    min_note_duration: float = 0.05  # Minimum note duration in seconds
    confidence_threshold: float = 0.3  # Minimum pitch confidence
    
    # Octave preferences (0 = prefer low, 1 = prefer high)
    octave_preference: float = 0.5
    
    # Harmonic handling
    harmonic_emphasis: float = 1.0  # Higher = expect more harmonics
    distortion_expected: bool = False
    
    # String/fret preferences
    prefer_lower_frets: bool = True
    power_chord_weight: float = 1.0  # Weight for power chord detection
    
    def get_scale_prior(self, scale_name: str) -> float:
        """Get prior probability weight for a scale."""
        return self.scale_weights.get(scale_name, 0.1)
    
    def get_root_prior(self, root: int) -> float:
        """Get prior probability weight for a root note."""
        return self.root_weights.get(root % 12, 1.0)
    
    def get_technique_sensitivity(self, technique: str) -> float:
        """Get sensitivity multiplier for a technique."""
        return self.technique_sensitivity.get(technique, 1.0)
    
    def get_pitch_class_weights(self, key_root: int = 0) -> np.ndarray:
        """
        Get pitch class weights for the detected/assumed key.
        
        Returns array of 12 weights (C to B) where higher values
        indicate more likely pitches for this genre.
        """
        weights = np.ones(12) * 0.5  # Base weight
        
        # Add weights from all preferred scales
        for scale_name in self.preferred_scales:
            scale_weight = self.scale_weights.get(scale_name, 0.5)
            intervals = SCALE_INTERVALS.get(scale_name, [])
            
            for interval in intervals:
                pitch_class = (key_root + interval) % 12
                weights[pitch_class] += scale_weight
        
        # Normalize
        weights = weights / np.max(weights)
        
        return weights


# ============================================================================
# GENRE PROFILES
# ============================================================================

METAL_PROFILE = GenreProfile(
    name="metal",
    display_name="Metal/Heavy Rock",
    
    # Scales commonly used in metal (ordered by frequency)
    preferred_scales=[
        'minor_pentatonic',    # Most common for solos
        'natural_minor',       # Aeolian - foundation of metal
        'harmonic_minor',      # Neoclassical metal, exotic flavor
        'phrygian',           # Thrash metal, aggressive sound
        'phrygian_dominant',  # Exotic/middle-eastern metal
        'blues',              # Classic rock-influenced metal
        'locrian',            # Extreme metal dissonance
        'diminished',         # Technical metal
    ],
    
    # Scale weights (probability priors)
    scale_weights={
        'minor_pentatonic': 1.5,
        'natural_minor': 1.4,
        'harmonic_minor': 1.2,
        'phrygian': 1.0,
        'phrygian_dominant': 0.9,
        'blues': 0.8,
        'locrian': 0.6,
        'diminished': 0.5,
        'major': 0.3,  # Rare in metal
    },
    
    # Common root notes for metal (E, A, D, B are most common due to tuning)
    common_roots=[4, 9, 2, 11, 7],  # E, A, D, B, G
    root_weights={
        4: 1.5,   # E - most common metal key
        9: 1.3,   # A
        2: 1.2,   # D
        11: 1.1,  # B
        7: 1.0,   # G
        0: 0.8,   # C
        5: 0.7,   # F
    },
    
    # Technique sensitivity (higher = detect more)
    technique_sensitivity={
        'bend': 1.3,          # Bends common in metal solos
        'vibrato': 1.4,       # Wide vibrato is signature metal
        'hammer_on': 1.5,     # Legato very common
        'pull_off': 1.5,
        'tapping': 1.8,       # Tapping more common in metal
        'tremolo': 1.6,       # Tremolo picking
        'slide': 1.2,
        'palm_mute': 2.0,     # Very metal-specific
        'pinch_harmonic': 1.5,
    },
    
    # Metal tempo range (can be very fast)
    tempo_range=(80, 240),
    typical_tempo=140,
    
    # Metal often has dense, fast passages
    note_density_range=(3.0, 20.0),
    typical_density=8.0,
    
    # Allow shorter notes for fast passages
    min_note_duration=0.03,
    confidence_threshold=0.25,
    
    # Metal often uses mid-to-low octaves
    octave_preference=0.35,
    
    # Heavy distortion expected
    harmonic_emphasis=1.5,
    distortion_expected=True,
    
    # Metal often stays in first 12 frets
    prefer_lower_frets=True,
    power_chord_weight=2.0,
)


ROCK_PROFILE = GenreProfile(
    name="rock",
    display_name="Rock/Classic Rock",
    
    # Scales commonly used in rock
    preferred_scales=[
        'minor_pentatonic',    # The rock scale
        'blues',               # Blues-rock foundation
        'major_pentatonic',    # Country-rock, classic rock
        'natural_minor',       # Hard rock
        'mixolydian',         # Classic rock (Hendrix, etc.)
        'dorian',             # Funk-rock
        'major',              # Pop-rock
    ],
    
    scale_weights={
        'minor_pentatonic': 1.6,
        'blues': 1.4,
        'major_pentatonic': 1.2,
        'natural_minor': 1.0,
        'mixolydian': 1.0,
        'dorian': 0.9,
        'major': 0.8,
        'harmonic_minor': 0.5,
    },
    
    # Common root notes for rock
    common_roots=[4, 9, 7, 2, 0],  # E, A, G, D, C
    root_weights={
        4: 1.5,   # E
        9: 1.4,   # A
        7: 1.3,   # G
        2: 1.2,   # D
        0: 1.1,   # C
        5: 0.9,   # F
        11: 0.8,  # B
    },
    
    # Technique sensitivity for rock
    technique_sensitivity={
        'bend': 1.5,          # Bends are signature rock
        'vibrato': 1.3,       # Expressive vibrato
        'hammer_on': 1.2,
        'pull_off': 1.2,
        'slide': 1.3,         # Slides common in rock
        'tapping': 0.8,       # Less common than metal
        'tremolo': 0.9,
        'double_stop': 1.4,   # Common in rock
    },
    
    # Rock tempo range
    tempo_range=(70, 180),
    typical_tempo=120,
    
    # Moderate note density
    note_density_range=(2.0, 12.0),
    typical_density=5.0,
    
    min_note_duration=0.04,
    confidence_threshold=0.28,
    
    # Rock uses full octave range
    octave_preference=0.5,
    
    # Moderate distortion
    harmonic_emphasis=1.2,
    distortion_expected=True,
    
    prefer_lower_frets=True,
    power_chord_weight=1.5,
)


BLUES_PROFILE = GenreProfile(
    name="blues",
    display_name="Blues",
    
    preferred_scales=[
        'blues',
        'minor_pentatonic',
        'major_pentatonic',
        'mixolydian',
        'dorian',
    ],
    
    scale_weights={
        'blues': 1.8,
        'minor_pentatonic': 1.5,
        'major_pentatonic': 1.2,
        'mixolydian': 1.0,
        'dorian': 0.9,
    },
    
    common_roots=[4, 9, 7, 0, 2],  # E, A, G, C, D
    root_weights={4: 1.5, 9: 1.4, 7: 1.2, 0: 1.1, 2: 1.0},
    
    technique_sensitivity={
        'bend': 2.0,          # Bends are essential to blues
        'vibrato': 1.8,       # Expressive vibrato
        'slide': 1.5,
        'hammer_on': 1.0,
        'pull_off': 1.0,
    },
    
    tempo_range=(60, 150),
    typical_tempo=90,
    
    note_density_range=(1.0, 8.0),
    typical_density=3.0,
    
    min_note_duration=0.06,
    confidence_threshold=0.30,
    
    octave_preference=0.4,
    harmonic_emphasis=1.0,
    distortion_expected=False,
    
    prefer_lower_frets=True,
    power_chord_weight=0.8,
)


DEFAULT_PROFILE = GenreProfile(
    name="default",
    display_name="General",
    
    preferred_scales=[
        'major',
        'natural_minor',
        'minor_pentatonic',
        'major_pentatonic',
    ],
    
    scale_weights={
        'major': 1.0,
        'natural_minor': 1.0,
        'minor_pentatonic': 1.0,
        'major_pentatonic': 1.0,
    },
    
    common_roots=list(range(12)),  # All roots equally likely
    root_weights={i: 1.0 for i in range(12)},
    
    technique_sensitivity={
        'bend': 1.0,
        'vibrato': 1.0,
        'hammer_on': 1.0,
        'pull_off': 1.0,
        'slide': 1.0,
        'tapping': 1.0,
    },
    
    tempo_range=(60, 180),
    typical_tempo=120,
    
    note_density_range=(2.0, 10.0),
    typical_density=5.0,
    
    min_note_duration=0.05,
    confidence_threshold=0.30,
    
    octave_preference=0.5,
    harmonic_emphasis=1.0,
    distortion_expected=False,
    
    prefer_lower_frets=True,
    power_chord_weight=1.0,
)


# Profile registry
GENRE_PROFILES: Dict[str, GenreProfile] = {
    'metal': METAL_PROFILE,
    'rock': ROCK_PROFILE,
    'blues': BLUES_PROFILE,
    'default': DEFAULT_PROFILE,
}


def get_genre_profile(genre: str) -> GenreProfile:
    """
    Get the genre profile by name.
    
    Args:
        genre: Genre name ('metal', 'rock', 'blues', 'default')
        
    Returns:
        GenreProfile for the specified genre
    """
    genre_lower = genre.lower().strip()
    
    # Handle aliases
    aliases = {
        'heavy': 'metal',
        'thrash': 'metal',
        'death': 'metal',
        'black': 'metal',
        'power': 'metal',
        'prog': 'metal',
        'progressive': 'metal',
        'hard rock': 'rock',
        'classic rock': 'rock',
        'alternative': 'rock',
        'grunge': 'rock',
        'punk': 'rock',
    }
    
    if genre_lower in aliases:
        genre_lower = aliases[genre_lower]
    
    return GENRE_PROFILES.get(genre_lower, DEFAULT_PROFILE)


@dataclass
class GenreOptimizer:
    """
    Applies genre-specific optimizations to pitch detection and tab generation.
    """
    
    profile: GenreProfile
    detected_key: Optional[int] = None  # Detected key root (0-11)
    
    def __post_init__(self):
        """Initialize pitch class weights."""
        self._pitch_weights = None
    
    @property
    def pitch_weights(self) -> np.ndarray:
        """Get cached pitch class weights."""
        if self._pitch_weights is None:
            key_root = self.detected_key if self.detected_key is not None else 0
            self._pitch_weights = self.profile.get_pitch_class_weights(key_root)
        return self._pitch_weights
    
    def set_key(self, key_root: int):
        """Set the detected key root and update pitch weights."""
        self.detected_key = key_root % 12
        self._pitch_weights = self.profile.get_pitch_class_weights(self.detected_key)
    
    def adjust_pitch_confidence(self, midi_note: int, confidence: float) -> float:
        """
        Adjust pitch confidence based on genre priors.
        
        Notes that are more likely in the genre's scales get a boost.
        
        Args:
            midi_note: MIDI note number
            confidence: Original detection confidence
            
        Returns:
            Adjusted confidence value
        """
        pitch_class = midi_note % 12
        weight = self.pitch_weights[pitch_class]
        
        # Apply weight as a multiplier (capped at 1.0)
        adjusted = confidence * (0.7 + 0.3 * weight)
        return min(1.0, adjusted)
    
    def score_pitch_candidates(
        self,
        candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Score and re-rank pitch candidates based on genre priors.
        
        Args:
            candidates: List of (midi_note, confidence) tuples
            
        Returns:
            Re-scored list sorted by adjusted confidence
        """
        scored = []
        for midi, conf in candidates:
            adjusted_conf = self.adjust_pitch_confidence(midi, conf)
            scored.append((midi, adjusted_conf))
        
        # Sort by adjusted confidence
        scored.sort(key=lambda x: -x[1])
        return scored
    
    def filter_by_tempo(
        self,
        notes: List[Any],
        detected_tempo: float
    ) -> List[Any]:
        """
        Filter notes based on tempo plausibility.
        
        Very fast notes at slow tempos or vice versa may be detection errors.
        
        Args:
            notes: List of detected notes
            detected_tempo: Detected tempo in BPM
            
        Returns:
            Filtered note list
        """
        min_tempo, max_tempo = self.profile.tempo_range
        
        if detected_tempo < min_tempo * 0.7 or detected_tempo > max_tempo * 1.3:
            # Tempo is outside expected range - be more conservative
            min_duration = self.profile.min_note_duration * 1.5
        else:
            min_duration = self.profile.min_note_duration
        
        return [n for n in notes if n.duration >= min_duration]
    
    def adjust_technique_detection(
        self,
        technique: str,
        confidence: float
    ) -> float:
        """
        Adjust technique detection confidence based on genre expectations.
        
        Args:
            technique: Technique name ('bend', 'vibrato', etc.)
            confidence: Original detection confidence
            
        Returns:
            Adjusted confidence
        """
        sensitivity = self.profile.get_technique_sensitivity(technique)
        return min(1.0, confidence * sensitivity)
    
    def get_detection_params(self) -> Dict[str, Any]:
        """
        Get genre-optimized detection parameters.
        
        Returns dict of parameters to pass to detection functions.
        """
        return {
            'min_note_duration': self.profile.min_note_duration,
            'confidence_threshold': self.profile.confidence_threshold,
            'distortion_mode': self.profile.distortion_expected,
            'prefer_lower_frets': self.profile.prefer_lower_frets,
            'typical_tempo': self.profile.typical_tempo,
        }
    
    def suggest_scale(self, pitch_histogram: np.ndarray) -> Tuple[str, float]:
        """
        Suggest the most likely scale based on pitch histogram and genre priors.
        
        Args:
            pitch_histogram: Array of 12 values (pitch class counts)
            
        Returns:
            Tuple of (scale_name, confidence)
        """
        best_scale = None
        best_score = -1
        
        for scale_name in self.profile.preferred_scales:
            intervals = SCALE_INTERVALS.get(scale_name, [])
            if not intervals:
                continue
            
            # Score based on how well histogram matches scale
            scale_prior = self.profile.get_scale_prior(scale_name)
            
            # Try all possible roots
            for root in range(12):
                score = scale_prior  # Start with prior
                
                # Add root weight
                score *= self.profile.get_root_prior(root)
                
                # Count matches
                total_weight = 0
                matched_weight = 0
                
                for i in range(12):
                    pitch_class = (root + i) % 12
                    weight = pitch_histogram[pitch_class]
                    total_weight += weight
                    
                    if i in intervals:
                        matched_weight += weight
                
                if total_weight > 0:
                    match_ratio = matched_weight / total_weight
                    score *= match_ratio
                
                if score > best_score:
                    best_score = score
                    best_scale = scale_name
        
        # Normalize score to 0-1 range
        confidence = min(1.0, best_score / 2.0)
        
        return best_scale or 'chromatic', confidence
    
    def apply_octave_preference(
        self,
        candidates: List[Tuple[int, int, float]]  # (string, fret, score)
    ) -> List[Tuple[int, int, float]]:
        """
        Apply octave preference to fret position candidates.
        
        Args:
            candidates: List of (string, fret, score) tuples
            
        Returns:
            Re-scored candidates with octave preference applied
        """
        if not candidates:
            return candidates
        
        result = []
        pref = self.profile.octave_preference
        
        for string, fret, score in candidates:
            # Estimate octave from string and fret
            # Lower strings + lower frets = lower octave
            octave_factor = (string / 6.0 + fret / 24.0) / 2.0
            
            # Apply preference
            if pref < 0.5:
                # Prefer lower octaves - boost low positions
                adjustment = 1.0 + (0.5 - pref) * (1.0 - octave_factor)
            else:
                # Prefer higher octaves - boost high positions
                adjustment = 1.0 + (pref - 0.5) * octave_factor
            
            result.append((string, fret, score * adjustment))
        
        return result


def apply_genre_optimization(
    notes: List[Any],
    genre: str = 'default',
    detected_key: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[Any], GenreOptimizer]:
    """
    Apply genre-specific optimization to detected notes.
    
    This is the main entry point for genre optimization.
    
    Args:
        notes: List of detected Note objects
        genre: Genre name ('metal', 'rock', 'blues', 'default')
        detected_key: Detected key root (0-11), or None for auto-detect
        verbose: Print optimization info
        
    Returns:
        Tuple of (optimized_notes, optimizer)
    """
    profile = get_genre_profile(genre)
    optimizer = GenreOptimizer(profile, detected_key)
    
    if verbose:
        print(f"\n🎸 Genre Optimization: {profile.display_name}")
        print(f"   Preferred scales: {', '.join(profile.preferred_scales[:3])}")
        print(f"   Expected tempo: {profile.tempo_range[0]}-{profile.tempo_range[1]} BPM")
        print(f"   Distortion mode: {'enabled' if profile.distortion_expected else 'disabled'}")
    
    # Build pitch histogram from notes
    pitch_histogram = np.zeros(12)
    for note in notes:
        pitch_class = note.midi % 12
        weight = getattr(note, 'duration', 1.0) * getattr(note, 'confidence', 1.0)
        pitch_histogram[pitch_class] += weight
    
    # Suggest scale if key not provided
    if detected_key is None and len(notes) > 0:
        suggested_scale, scale_conf = optimizer.suggest_scale(pitch_histogram)
        if verbose:
            print(f"   Suggested scale: {suggested_scale} (confidence: {scale_conf:.2f})")
    
    # Re-score notes based on genre priors
    optimized = []
    for note in notes:
        adjusted_conf = optimizer.adjust_pitch_confidence(note.midi, note.confidence)
        
        # Create a copy with adjusted confidence
        from copy import deepcopy
        new_note = deepcopy(note)
        new_note.confidence = adjusted_conf
        optimized.append(new_note)
    
    if verbose:
        original_avg_conf = np.mean([n.confidence for n in notes]) if notes else 0
        optimized_avg_conf = np.mean([n.confidence for n in optimized]) if optimized else 0
        print(f"   Confidence adjustment: {original_avg_conf:.3f} → {optimized_avg_conf:.3f}")
    
    return optimized, optimizer


def get_genre_choices() -> List[str]:
    """Get list of available genre choices for argparse."""
    return list(GENRE_PROFILES.keys())


def add_genre_args(parser):
    """Add genre-related arguments to an argument parser."""
    parser.add_argument(
        '--genre', '-g',
        type=str,
        choices=get_genre_choices(),
        default=None,
        help='Genre for optimized detection (metal, rock, blues). '
             'Adjusts scale priors, technique sensitivity, and tempo expectations.'
    )


# ============================================================================
# Testing
# ============================================================================

def test_genre_optimizer():
    """Test the genre optimizer with mock data."""
    print("Testing GenreOptimizer...")
    
    # Test profile retrieval
    metal = get_genre_profile('metal')
    assert metal.name == 'metal'
    assert 'minor_pentatonic' in metal.preferred_scales
    print(f"  ✅ Metal profile loaded: {metal.display_name}")
    
    rock = get_genre_profile('rock')
    assert rock.name == 'rock'
    print(f"  ✅ Rock profile loaded: {rock.display_name}")
    
    # Test alias handling
    thrash = get_genre_profile('thrash')
    assert thrash.name == 'metal'
    print(f"  ✅ Alias 'thrash' → metal")
    
    # Test optimizer
    optimizer = GenreOptimizer(metal)
    
    # E minor pentatonic should get high weight in metal
    e_conf = optimizer.adjust_pitch_confidence(40, 0.5)  # E2
    fsharp_conf = optimizer.adjust_pitch_confidence(42, 0.5)  # F#2 (not in E minor pent)
    
    print(f"  E confidence: 0.5 → {e_conf:.3f}")
    print(f"  F# confidence: 0.5 → {fsharp_conf:.3f}")
    
    # Test pitch weights
    weights = optimizer.pitch_weights
    print(f"  Pitch weights: {weights}")
    
    # Test technique sensitivity
    bend_sens = metal.get_technique_sensitivity('bend')
    tap_sens = metal.get_technique_sensitivity('tapping')
    print(f"  Bend sensitivity: {bend_sens}")
    print(f"  Tapping sensitivity: {tap_sens}")
    
    # Test detection params
    params = optimizer.get_detection_params()
    print(f"  Detection params: {params}")
    
    print("\n✅ All genre optimizer tests passed!")


if __name__ == "__main__":
    test_genre_optimizer()
