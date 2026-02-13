#!/usr/bin/env python3
"""
Solo Range Detection for Guitar Tab Transcription

Automatically detects solo vs rhythm sections and applies appropriate
pitch detection constraints for each section type.

Lead guitar solos are typically characterized by:
1. Higher pitch range - played on B and high E strings, upper frets (7th-22nd)
2. Monophonic playing - single note lines vs. chord strumming
3. Higher note density - faster passages with more notes per second
4. Specific frequency range - 300Hz-1500Hz for fundamentals
5. More melodic movement - stepwise motion and scale patterns

Usage:
    from solo_range_detection import (
        SoloDetector,
        SoloDetectionConfig,
        SectionType,
        detect_solo_sections,
        apply_solo_range_constraints
    )
    
    # Detect sections
    detector = SoloDetector(sr=22050)
    sections = detector.detect(y)
    
    # Apply constraints to notes based on section type
    constrained_notes = apply_solo_range_constraints(notes, sections, tuning)
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum, auto


class SectionType(Enum):
    """Type of musical section."""
    SOLO = auto()      # Lead guitar solo
    RHYTHM = auto()    # Rhythm guitar / chords
    MIXED = auto()     # Combination or transition
    SILENCE = auto()   # No guitar playing


@dataclass
class Section:
    """Represents a detected musical section."""
    start_time: float
    end_time: float
    section_type: SectionType
    confidence: float
    
    # Analysis details
    avg_frequency: float = 0.0        # Average fundamental frequency
    note_density: float = 0.0          # Notes per second
    polyphony_ratio: float = 0.0       # Ratio of polyphonic frames
    spectral_centroid: float = 0.0     # Average spectral centroid
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return (f"{self.section_type.name}: {self.start_time:.2f}s - {self.end_time:.2f}s "
                f"({self.confidence:.0%} conf)")


@dataclass
class SoloDetectionConfig:
    """Configuration for solo detection."""
    
    # =========================================================================
    # FREQUENCY THRESHOLDS
    # =========================================================================
    # Lead solos typically above this frequency (Hz)
    solo_freq_min: float = 250.0
    # Lead solos typically below this frequency (fundamentals)
    solo_freq_max: float = 1500.0
    # Rhythm guitar typically has lower fundamentals
    rhythm_freq_max: float = 400.0
    
    # =========================================================================
    # LEAD GUITAR MIDI RANGE (for pitch constraints)
    # =========================================================================
    # Standard tuning high E string open = E4 = MIDI 64
    # 22nd fret on high E = D6 = MIDI 86
    lead_midi_min: int = 55    # G3 - 3rd fret on low E, or open G string
    lead_midi_max: int = 88    # E6 - 24th fret on high E
    
    # Solo detection prefers these strings (0-5 = E A D G B e)
    lead_strings: List[int] = field(default_factory=lambda: [3, 4, 5])  # G, B, e
    
    # Preferred fret range for solos
    lead_fret_min: int = 5
    lead_fret_max: int = 22
    
    # =========================================================================
    # NOTE DENSITY THRESHOLDS
    # =========================================================================
    # Notes per second threshold for solo classification
    solo_note_density_min: float = 3.0
    # Below this is definitely rhythm
    rhythm_note_density_max: float = 2.0
    
    # =========================================================================
    # POLYPHONY THRESHOLDS
    # =========================================================================
    # Ratio of frames with multiple simultaneous notes
    # Solos are mostly monophonic
    solo_polyphony_max: float = 0.2
    # Rhythm has more polyphony (chords)
    rhythm_polyphony_min: float = 0.4
    
    # =========================================================================
    # SPECTRAL THRESHOLDS
    # =========================================================================
    # Spectral centroid threshold (Hz) - solos tend to be brighter
    solo_centroid_min: float = 1500.0
    
    # =========================================================================
    # SECTION DETECTION
    # =========================================================================
    # Minimum section duration (seconds)
    min_section_duration: float = 2.0
    # Analysis window size (seconds)
    analysis_window: float = 1.0
    # Window hop (seconds)
    analysis_hop: float = 0.5
    
    # =========================================================================
    # CONFIDENCE WEIGHTS
    # =========================================================================
    freq_weight: float = 0.3
    density_weight: float = 0.25
    polyphony_weight: float = 0.25
    centroid_weight: float = 0.2
    
    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class SoloRangeConstraints:
    """Pitch detection constraints for solo sections."""
    # MIDI note range
    midi_min: int = 55   # G3
    midi_max: int = 88   # E6
    
    # Frequency range (Hz)
    freq_min: float = 196.0   # G3
    freq_max: float = 1318.5  # E6
    
    # Preferred strings (indices)
    preferred_strings: List[int] = field(default_factory=lambda: [3, 4, 5])
    
    # Preferred fret range
    fret_min: int = 5
    fret_max: int = 22
    
    # Higher fret preference strength (0-1)
    # Higher values push towards upper frets for same pitch
    upper_fret_preference: float = 0.7


class SoloDetector:
    """
    Detects solo vs rhythm sections in guitar audio.
    
    Uses multiple features to classify sections:
    - Frequency/pitch range
    - Note density
    - Polyphony
    - Spectral characteristics
    """
    
    def __init__(
        self,
        sr: int = 22050,
        config: Optional[SoloDetectionConfig] = None
    ):
        self.sr = sr
        self.config = config or SoloDetectionConfig()
    
    def detect(
        self,
        y: np.ndarray,
        verbose: bool = True
    ) -> List[Section]:
        """
        Detect solo and rhythm sections in audio.
        
        Args:
            y: Audio signal (mono)
            verbose: Print detection progress
            
        Returns:
            List of detected Section objects
        """
        if verbose:
            print("\n🎸 Solo Section Detection")
            print("-" * 40)
        
        cfg = self.config
        
        # Compute features
        features = self._compute_features(y, verbose)
        
        # Classify each analysis window
        window_classifications = self._classify_windows(features, verbose)
        
        # Merge into contiguous sections
        sections = self._merge_into_sections(window_classifications, verbose)
        
        if verbose:
            print(f"\n📊 Detected {len(sections)} sections:")
            for section in sections:
                print(f"   {section}")
        
        return sections
    
    def _compute_features(
        self,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute analysis features for each window.
        
        Returns dict with arrays of features for each window.
        """
        cfg = self.config
        
        # Compute spectrogram
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=cfg.n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=cfg.hop_length)
        
        # Window parameters in frames
        window_frames = int(cfg.analysis_window * self.sr / cfg.hop_length)
        hop_frames = int(cfg.analysis_hop * self.sr / cfg.hop_length)
        n_windows = max(1, (S.shape[1] - window_frames) // hop_frames + 1)
        
        if verbose:
            print(f"  Analysis: {n_windows} windows of {cfg.analysis_window}s each")
        
        # Initialize feature arrays
        window_times = np.zeros(n_windows)
        avg_frequencies = np.zeros(n_windows)
        note_densities = np.zeros(n_windows)
        polyphony_ratios = np.zeros(n_windows)
        spectral_centroids = np.zeros(n_windows)
        energy_levels = np.zeros(n_windows)
        
        # Compute pitch for frequency analysis
        if verbose:
            print("  Computing pitch contour...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=60,  # Below guitar range for detection
            fmax=2000,  # Above typical fundamentals
            sr=self.sr,
            hop_length=cfg.hop_length,
            fill_na=0.0
        )
        
        # Onset detection for note density
        if verbose:
            print("  Detecting onsets...")
        onset_strength = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=cfg.hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_strength,
            sr=self.sr,
            hop_length=cfg.hop_length,
            units='frames'
        )
        onset_mask = np.zeros(len(onset_strength), dtype=bool)
        onset_mask[onsets] = True
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=cfg.hop_length
        )[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=cfg.hop_length)[0]
        
        # Polyphony detection via spectral peaks
        if verbose:
            print("  Analyzing polyphony...")
        polyphony = self._detect_polyphony_per_frame(S, freqs)
        
        # Compute features per window
        for i in range(n_windows):
            start_frame = i * hop_frames
            end_frame = min(start_frame + window_frames, S.shape[1])
            
            window_times[i] = times[start_frame]
            
            # Average frequency (from f0)
            window_f0 = f0[start_frame:end_frame]
            valid_f0 = window_f0[window_f0 > 0]
            avg_frequencies[i] = np.median(valid_f0) if len(valid_f0) > 0 else 0
            
            # Note density (onsets per second)
            window_onsets = np.sum(onset_mask[start_frame:end_frame])
            window_duration = (end_frame - start_frame) * cfg.hop_length / self.sr
            note_densities[i] = window_onsets / window_duration if window_duration > 0 else 0
            
            # Polyphony ratio
            window_poly = polyphony[start_frame:end_frame]
            polyphony_ratios[i] = np.mean(window_poly > 1)
            
            # Spectral centroid
            window_centroid = centroid[start_frame:end_frame]
            spectral_centroids[i] = np.mean(window_centroid)
            
            # Energy level
            window_rms = rms[start_frame:end_frame]
            energy_levels[i] = np.mean(window_rms)
        
        return {
            'times': window_times,
            'avg_frequency': avg_frequencies,
            'note_density': note_densities,
            'polyphony_ratio': polyphony_ratios,
            'spectral_centroid': spectral_centroids,
            'energy': energy_levels
        }
    
    def _detect_polyphony_per_frame(
        self,
        S: np.ndarray,
        freqs: np.ndarray
    ) -> np.ndarray:
        """
        Detect number of simultaneous notes per frame.
        
        Uses peak counting in the guitar frequency range.
        """
        cfg = self.config
        n_frames = S.shape[1]
        polyphony = np.zeros(n_frames)
        
        # Guitar fundamental frequency range
        fmin_bin = np.searchsorted(freqs, 75)
        fmax_bin = np.searchsorted(freqs, 1500)
        
        for i in range(n_frames):
            frame = S[fmin_bin:fmax_bin, i]
            if np.max(frame) < 1e-10:
                continue
            
            # Normalize
            frame_norm = frame / np.max(frame)
            
            # Find peaks
            peaks, _ = find_peaks(
                frame_norm,
                height=0.2,
                distance=5,  # ~50Hz at this resolution
                prominence=0.1
            )
            
            # Count significant peaks as notes
            polyphony[i] = min(len(peaks), 6)  # Cap at 6 (guitar strings)
        
        return polyphony
    
    def _classify_windows(
        self,
        features: Dict[str, np.ndarray],
        verbose: bool = True
    ) -> List[Tuple[float, SectionType, float]]:
        """
        Classify each window as SOLO, RHYTHM, or MIXED.
        
        Returns list of (time, section_type, confidence) tuples.
        """
        cfg = self.config
        n_windows = len(features['times'])
        
        classifications = []
        
        for i in range(n_windows):
            time = features['times'][i]
            avg_freq = features['avg_frequency'][i]
            note_density = features['note_density'][i]
            poly_ratio = features['polyphony_ratio'][i]
            centroid = features['spectral_centroid'][i]
            energy = features['energy'][i]
            
            # Skip silent sections
            if energy < 0.01:
                classifications.append((time, SectionType.SILENCE, 1.0))
                continue
            
            # Compute solo score (0-1) from each feature
            scores = {}
            
            # Frequency score: higher avg frequency = more solo-like
            if avg_freq > 0:
                freq_score = np.clip(
                    (avg_freq - cfg.rhythm_freq_max) / (cfg.solo_freq_min - cfg.rhythm_freq_max + 1),
                    0, 1
                )
            else:
                freq_score = 0.5  # Unknown
            scores['freq'] = freq_score
            
            # Density score: higher density = more solo-like
            density_score = np.clip(
                (note_density - cfg.rhythm_note_density_max) / 
                (cfg.solo_note_density_min - cfg.rhythm_note_density_max + 1),
                0, 1
            )
            scores['density'] = density_score
            
            # Polyphony score: lower polyphony = more solo-like
            poly_score = 1.0 - np.clip(
                (poly_ratio - cfg.solo_polyphony_max) /
                (cfg.rhythm_polyphony_min - cfg.solo_polyphony_max + 0.01),
                0, 1
            )
            scores['polyphony'] = poly_score
            
            # Centroid score: higher centroid = more solo-like
            centroid_score = np.clip(
                (centroid - 1000) / (cfg.solo_centroid_min - 1000 + 1),
                0, 1
            )
            scores['centroid'] = centroid_score
            
            # Weighted combination
            solo_score = (
                cfg.freq_weight * scores['freq'] +
                cfg.density_weight * scores['density'] +
                cfg.polyphony_weight * scores['polyphony'] +
                cfg.centroid_weight * scores['centroid']
            )
            
            # Classify based on score
            if solo_score >= 0.6:
                section_type = SectionType.SOLO
                confidence = solo_score
            elif solo_score <= 0.4:
                section_type = SectionType.RHYTHM
                confidence = 1.0 - solo_score
            else:
                section_type = SectionType.MIXED
                confidence = 0.5
            
            classifications.append((time, section_type, confidence))
        
        return classifications
    
    def _merge_into_sections(
        self,
        classifications: List[Tuple[float, SectionType, float]],
        verbose: bool = True
    ) -> List[Section]:
        """
        Merge consecutive windows of the same type into sections.
        """
        cfg = self.config
        
        if not classifications:
            return []
        
        sections = []
        current_type = classifications[0][1]
        current_start = classifications[0][0]
        current_confidences = [classifications[0][2]]
        
        for i in range(1, len(classifications)):
            time, section_type, confidence = classifications[i]
            
            if section_type != current_type:
                # End current section
                section_end = time
                if section_end - current_start >= cfg.min_section_duration:
                    sections.append(Section(
                        start_time=current_start,
                        end_time=section_end,
                        section_type=current_type,
                        confidence=np.mean(current_confidences)
                    ))
                
                # Start new section
                current_type = section_type
                current_start = time
                current_confidences = [confidence]
            else:
                current_confidences.append(confidence)
        
        # Add final section
        if classifications:
            final_time = classifications[-1][0] + cfg.analysis_window
            if final_time - current_start >= cfg.min_section_duration:
                sections.append(Section(
                    start_time=current_start,
                    end_time=final_time,
                    section_type=current_type,
                    confidence=np.mean(current_confidences)
                ))
        
        return sections


def detect_solo_sections(
    audio_path: str = None,
    y: np.ndarray = None,
    sr: int = 22050,
    config: Optional[SoloDetectionConfig] = None,
    verbose: bool = True
) -> List[Section]:
    """
    Convenience function to detect solo sections.
    
    Args:
        audio_path: Path to audio file (will be loaded if y not provided)
        y: Pre-loaded audio signal (optional)
        sr: Sample rate
        config: Detection configuration
        verbose: Print progress
        
    Returns:
        List of detected Section objects
    """
    if y is None:
        if audio_path is None:
            raise ValueError("Must provide either audio_path or y")
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    detector = SoloDetector(sr=sr, config=config)
    return detector.detect(y, verbose=verbose)


def get_solo_range_constraints(
    section_type: SectionType,
    config: Optional[SoloDetectionConfig] = None
) -> SoloRangeConstraints:
    """
    Get pitch detection constraints for a section type.
    
    For SOLO sections, returns constraints that:
    - Focus on high strings (G, B, e)
    - Prefer upper frets (7th-22nd)
    - Constrain to lead guitar range
    
    For RHYTHM sections, returns looser constraints.
    """
    cfg = config or SoloDetectionConfig()
    
    if section_type == SectionType.SOLO:
        return SoloRangeConstraints(
            midi_min=cfg.lead_midi_min,
            midi_max=cfg.lead_midi_max,
            freq_min=librosa.midi_to_hz(cfg.lead_midi_min),
            freq_max=librosa.midi_to_hz(cfg.lead_midi_max),
            preferred_strings=cfg.lead_strings,
            fret_min=cfg.lead_fret_min,
            fret_max=cfg.lead_fret_max,
            upper_fret_preference=0.7
        )
    elif section_type == SectionType.RHYTHM:
        return SoloRangeConstraints(
            midi_min=40,   # E2 (low E open)
            midi_max=76,   # E5 (12th fret high E)
            freq_min=82.4,
            freq_max=659.3,
            preferred_strings=[0, 1, 2, 3, 4, 5],  # All strings
            fret_min=0,
            fret_max=12,
            upper_fret_preference=0.0  # No upper fret preference
        )
    else:  # MIXED or SILENCE
        return SoloRangeConstraints(
            midi_min=40,
            midi_max=88,
            freq_min=82.4,
            freq_max=1318.5,
            preferred_strings=[0, 1, 2, 3, 4, 5],
            fret_min=0,
            fret_max=22,
            upper_fret_preference=0.3  # Mild preference
        )


def constrain_pitch_to_range(
    midi_note: int,
    constraints: SoloRangeConstraints
) -> Tuple[int, bool]:
    """
    Constrain a MIDI note to the given range.
    
    Returns:
        (constrained_midi, was_modified)
    """
    original = midi_note
    
    # Octave shift if out of range
    while midi_note < constraints.midi_min:
        midi_note += 12
    while midi_note > constraints.midi_max:
        midi_note -= 12
    
    # Final bounds check
    midi_note = max(constraints.midi_min, min(constraints.midi_max, midi_note))
    
    return midi_note, midi_note != original


def choose_fret_position_solo(
    midi_note: int,
    tuning: List[int],
    constraints: SoloRangeConstraints,
    prev_position: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int]]:
    """
    Choose the best fret position for a note during a solo section.
    
    Prefers:
    1. Higher frets (upper_fret_preference)
    2. High strings (G, B, e)
    3. Continuity with previous position
    
    Args:
        midi_note: MIDI note number
        tuning: Guitar tuning as MIDI notes
        constraints: Solo range constraints
        prev_position: Previous (string, fret) if any
        
    Returns:
        (string, fret) tuple or None if note can't be played
    """
    # Find all possible positions
    options = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= 24:
            options.append((string_idx, fret))
    
    if not options:
        return None
    
    if len(options) == 1:
        return options[0]
    
    # Score each option
    scored = []
    for string, fret in options:
        score = 0.0
        
        # Preferred strings bonus
        if string in constraints.preferred_strings:
            score += 5.0
        
        # Upper fret preference
        if constraints.upper_fret_preference > 0:
            if fret >= constraints.fret_min:
                # Bonus proportional to how high the fret is
                fret_score = (fret - constraints.fret_min) / (constraints.fret_max - constraints.fret_min + 1)
                score += fret_score * constraints.upper_fret_preference * 4.0
        else:
            # Prefer lower frets for rhythm
            score += (12 - fret) * 0.1 if fret <= 12 else 0
        
        # Fret range bonus
        if constraints.fret_min <= fret <= constraints.fret_max:
            score += 2.0
        
        # Continuity with previous position
        if prev_position:
            prev_string, prev_fret = prev_position
            fret_distance = abs(fret - prev_fret)
            string_distance = abs(string - prev_string)
            
            # Small fret jumps preferred
            if fret_distance <= 3:
                score += 2.0 - fret_distance * 0.3
            else:
                score -= fret_distance * 0.2
            
            # Adjacent strings preferred
            if string_distance <= 1:
                score += 1.0
            else:
                score -= string_distance * 0.5
        
        scored.append((score, string, fret))
    
    # Sort by score (highest first)
    scored.sort(reverse=True)
    _, best_string, best_fret = scored[0]
    
    return (best_string, best_fret)


def apply_solo_range_constraints(
    notes: List,  # List of Note objects
    sections: List[Section],
    tuning: List[int],
    verbose: bool = False
) -> List:
    """
    Apply section-specific constraints to detected notes.
    
    For notes in SOLO sections:
    - Constrain to lead guitar MIDI range
    - Recompute tab positions with upper fret preference
    
    Args:
        notes: List of Note objects (with midi, start_time, duration)
        sections: Detected sections from SoloDetector
        tuning: Guitar tuning as MIDI notes
        verbose: Print modification info
        
    Returns:
        Modified notes list with constrained MIDI values
    """
    if not sections:
        return notes
    
    modified_count = 0
    
    for note in notes:
        # Find which section this note belongs to
        section = None
        for s in sections:
            if s.start_time <= note.start_time < s.end_time:
                section = s
                break
        
        if section is None:
            continue
        
        # Get constraints for this section type
        constraints = get_solo_range_constraints(section.section_type)
        
        # Apply MIDI range constraint
        new_midi, was_modified = constrain_pitch_to_range(note.midi, constraints)
        
        if was_modified:
            if verbose:
                old_name = librosa.midi_to_note(note.midi)
                new_name = librosa.midi_to_note(new_midi)
                print(f"  Constrained {old_name} -> {new_name} ({section.section_type.name})")
            note.midi = new_midi
            modified_count += 1
    
    if verbose and modified_count > 0:
        print(f"\n  Total notes constrained: {modified_count}")
    
    return notes


def notes_to_tabs_with_solo_detection(
    notes: List,  # List of Note objects
    sections: List[Section],
    tuning: List[int],
    verbose: bool = False
) -> List:
    """
    Convert notes to tab positions with solo-aware fret placement.
    
    Uses higher fret positions for solo sections.
    
    Args:
        notes: List of Note objects
        sections: Detected sections
        tuning: Guitar tuning
        verbose: Print details
        
    Returns:
        List of TabNote objects (string, fret, start_time, duration)
    """
    from dataclasses import dataclass as _dataclass
    
    @_dataclass
    class TabNote:
        string: int
        fret: int
        start_time: float
        duration: float
    
    tab_notes = []
    prev_position = None
    
    for note in notes:
        # Find section for this note
        section = None
        for s in sections:
            if s.start_time <= note.start_time < s.end_time:
                section = s
                break
        
        # Get appropriate constraints
        if section:
            constraints = get_solo_range_constraints(section.section_type)
        else:
            constraints = get_solo_range_constraints(SectionType.MIXED)
        
        # Choose fret position
        position = choose_fret_position_solo(
            note.midi,
            tuning,
            constraints,
            prev_position
        )
        
        if position:
            string, fret = position
            tab_notes.append(TabNote(
                string=string,
                fret=fret,
                start_time=note.start_time,
                duration=note.duration
            ))
            prev_position = position
            
            if verbose:
                note_name = librosa.midi_to_note(note.midi)
                section_name = section.section_type.name if section else "UNKNOWN"
                print(f"  {note_name} -> string {string} fret {fret} ({section_name})")
    
    return tab_notes


# =============================================================================
# ARGPARSE INTEGRATION
# =============================================================================

def add_solo_detection_args(parser):
    """Add solo detection arguments to an argument parser."""
    solo_group = parser.add_argument_group('Solo Range Detection')
    
    solo_group.add_argument(
        '--detect-solos',
        action='store_true',
        help='Enable automatic solo vs rhythm section detection'
    )
    
    solo_group.add_argument(
        '--solo-fret-min',
        type=int,
        default=5,
        metavar='FRET',
        help='Minimum fret for solo sections (default: 5)'
    )
    
    solo_group.add_argument(
        '--solo-fret-max',
        type=int,
        default=22,
        metavar='FRET',
        help='Maximum fret for solo sections (default: 22)'
    )
    
    solo_group.add_argument(
        '--solo-upper-fret-preference',
        type=float,
        default=0.7,
        metavar='WEIGHT',
        help='Preference for upper frets in solos, 0-1 (default: 0.7)'
    )
    
    solo_group.add_argument(
        '--solo-min-duration',
        type=float,
        default=2.0,
        metavar='SECONDS',
        help='Minimum solo section duration (default: 2.0s)'
    )
    
    return solo_group


def config_from_args(args) -> SoloDetectionConfig:
    """Create SoloDetectionConfig from argparse namespace."""
    config = SoloDetectionConfig()
    
    if hasattr(args, 'solo_fret_min') and args.solo_fret_min:
        config.lead_fret_min = args.solo_fret_min
    if hasattr(args, 'solo_fret_max') and args.solo_fret_max:
        config.lead_fret_max = args.solo_fret_max
    if hasattr(args, 'solo_min_duration') and args.solo_min_duration:
        config.min_section_duration = args.solo_min_duration
    
    return config


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect solo vs rhythm sections in guitar audio"
    )
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-v', '--verbose', action='store_true')
    add_solo_detection_args(parser)
    
    args = parser.parse_args()
    config = config_from_args(args)
    
    # Load audio
    y, sr = librosa.load(args.input, sr=22050, mono=True)
    
    # Detect sections
    detector = SoloDetector(sr=sr, config=config)
    sections = detector.detect(y, verbose=args.verbose)
    
    # Print results
    print("\n" + "=" * 50)
    print("DETECTED SECTIONS")
    print("=" * 50)
    for section in sections:
        print(f"\n{section.section_type.name}:")
        print(f"  Time: {section.start_time:.2f}s - {section.end_time:.2f}s")
        print(f"  Duration: {section.duration:.2f}s")
        print(f"  Confidence: {section.confidence:.0%}")
