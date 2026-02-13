#!/usr/bin/env python3
"""
Guitar Tab Generator - Audio to Tabs using AI
Built for Albanian songs with no existing tabs 🎸

Improved pitch detection with:
- Multiple pitch detection algorithms (pyin, piptrack)
- Median filtering for jitter reduction
- Onset detection with backtracking
- Pitch confidence smoothing
- Harmonic handling for guitar
- Support for alternate tunings

Advanced onset detection with:
- Multiple onset detection methods (spectral flux, complex domain, HFC)
- Ensemble/voting across methods
- Guitar-specific attack transient detection
- Hammer-on/pull-off (legato) detection for softer onsets

Polyphonic pitch detection (--polyphonic):
- NMF: Non-negative Matrix Factorization on Constant-Q Transform
- Harmonic: Harmonic/percussive separation with multi-peak picking
- Detects when multiple strings are played simultaneously
- Ideal for chords and fingerstyle playing
"""

import librosa
import numpy as np
from scipy import ndimage
from scipy.signal import medfilt, find_peaks, butter, filtfilt
from scipy.ndimage import maximum_filter1d
from sklearn.decomposition import NMF
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import argparse
import sys
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
import soundfile as sf

# Import preprocessing module
from preprocessing import (
    PreprocessingConfig,
    DistortionPreprocessingConfig,
    preprocess_audio,
    preprocess_distortion,
    add_preprocessing_args
)

# Frequency domain cleanup module (applied before pitch detection)
try:
    from freq_cleanup import (
        FreqCleanupConfig,
        freq_domain_cleanup,
        add_freq_cleanup_args,
        config_from_args as freq_config_from_args
    )
    HAS_FREQ_CLEANUP = True
except ImportError:
    HAS_FREQ_CLEANUP = False

# Sync export module for web player integration
try:
    from sync_export import export_sync_json, generate_sync_data
    HAS_SYNC_EXPORT = True
except ImportError:
    HAS_SYNC_EXPORT = False

# String detection module for spectral-based string identification
try:
    from string_detection import (
        StringDetector,
        choose_string_position_spectral,
        compute_string_brightness_profile,
        STRING_NAMES as STRING_DETECTION_NAMES
    )
    HAS_STRING_DETECTION = True
except ImportError:
    HAS_STRING_DETECTION = False

# Guitar Pro support
try:
    import guitarpro
    HAS_GUITARPRO = True
except ImportError:
    HAS_GUITARPRO = False

# CREPE pitch detection (deep learning model for monophonic pitch)
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

# Accurate pitch detection with multi-detector voting
try:
    from pitch_accuracy import (
        AccuratePitchDetector,
        detect_pitches_accurate,
        compare_detectors,
        GUITAR_MIN_MIDI,
        GUITAR_MAX_MIDI
    )
    HAS_PITCH_ACCURACY = True
except ImportError:
    HAS_PITCH_ACCURACY = False

# Octave correction using harmonic analysis
try:
    from octave_correction import (
        OctaveCorrector,
        OctaveCorrectionResult,
        OctaveCorrection,
        apply_octave_correction,
        validate_guitar_range
    )
    HAS_OCTAVE_CORRECTION = True
except ImportError:
    HAS_OCTAVE_CORRECTION = False

# Multi-model ensemble pitch detection (MOST ACCURATE)
try:
    from ensemble_pitch import (
        EnsemblePitchDetector,
        EnsembleConfig,
        ConsensusNote,
        detect_notes_ensemble,
    )
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

# Distortion-aware pitch detection (for heavy/distorted guitar)
try:
    from distortion_pitch import (
        DistortionPitchDetector,
        DistortionConfig,
        DistortionPitchResult,
        detect_pitches_distortion,
        convert_to_notes as distortion_convert_to_notes,
    )
    HAS_DISTORTION_PITCH = True
except ImportError:
    HAS_DISTORTION_PITCH = False

# Precise note segmentation (IMPROVED NOTE BOUNDARIES)
try:
    from note_segmentation import (
        NoteSegmenter,
        NoteSegment,
        segment_notes,
        convert_to_notes,
    )
    HAS_NOTE_SEGMENTATION = True
except ImportError:
    HAS_NOTE_SEGMENTATION = False

# Music theory module for post-processing
try:
    from music_theory import (
        detect_key, post_process_notes, filter_impossible_transitions,
        prefer_lower_frets, detect_patterns, format_pattern_info,
        parse_key_string, Key, Pattern, SCALE_INTERVALS
    )
    HAS_MUSIC_THEORY = True
except ImportError:
    HAS_MUSIC_THEORY = False

# Scale-constrained pitch detection (enforces musical coherence)
try:
    from scale_constrain import (
        ScaleConstrainer,
        ScaleConstraintConfig,
        apply_scale_constraint_to_detection,
        add_scale_constrain_args,
        config_from_args as scale_config_from_args
    )
    HAS_SCALE_CONSTRAIN = True
except ImportError:
    HAS_SCALE_CONSTRAIN = False

# Fingering optimizer module
try:
    from fingering_optimizer import (
        FingeringOptimizer, FingeringConfig, optimize_fingering,
        compare_fingerings
    )
    HAS_FINGERING_OPTIMIZER = True
except ImportError:
    HAS_FINGERING_OPTIMIZER = False

# Rhythm notation module
try:
    from rhythm_notation import (
        detect_tempo, analyze_audio_rhythm, format_tab_with_rhythm,
        format_rhythm_line, analyze_rhythm, TempoInfo
    )
    HAS_RHYTHM_NOTATION = True
except ImportError:
    HAS_RHYTHM_NOTATION = False

# Difficulty analyzer module
try:
    from difficulty_analyzer import (
        DifficultyAnalyzer, DifficultyReport, DifficultyLevel,
        analyze_difficulty
    )
    HAS_DIFFICULTY_ANALYZER = True
except ImportError:
    HAS_DIFFICULTY_ANALYZER = False

# Debug spectrogram visualization module
try:
    from debug_spectrogram import (
        generate_debug_spectrogram,
        generate_zoomed_spectrograms,
        DetectedNote as DebugNote
    )
    HAS_DEBUG_SPECTROGRAM = True
except ImportError:
    HAS_DEBUG_SPECTROGRAM = False

# Lead guitar isolation module
try:
    from lead_guitar_isolation import (
        LeadIsolationConfig,
        isolate_lead_guitar,
        isolate_lead_from_array,
        add_lead_isolation_args,
        config_from_args as lead_config_from_args
    )
    HAS_LEAD_ISOLATION = True
except ImportError:
    HAS_LEAD_ISOLATION = False

# Genre-specific optimization (metal/rock priors)
try:
    from genre_optimizer import (
        GenreOptimizer,
        GenreProfile,
        get_genre_profile,
        apply_genre_optimization,
        add_genre_args,
        get_genre_choices,
    )
    HAS_GENRE_OPTIMIZER = True
except ImportError:
    HAS_GENRE_OPTIMIZER = False

# Solo range detection (auto-detect solo vs rhythm sections)
try:
    from solo_range_detection import (
        SoloDetector,
        SoloDetectionConfig,
        SectionType,
        Section,
        detect_solo_sections,
        get_solo_range_constraints,
        apply_solo_range_constraints,
        notes_to_tabs_with_solo_detection,
        choose_fret_position_solo,
        add_solo_detection_args,
        config_from_args as solo_config_from_args
    )
    HAS_SOLO_DETECTION = True
except ImportError:
    HAS_SOLO_DETECTION = False

# Fast legato run detection (3-note-per-string, scale runs)
try:
    from fast_legato_runs import (
        FastLegatoDetector,
        LegatoNote,
        LegatoRun,
        detect_fast_legato_runs,
        SCALE_PATTERNS,
        THREE_NPS_PATTERNS,
    )
    HAS_FAST_LEGATO = True
except ImportError:
    HAS_FAST_LEGATO = False

# Fretboard position tracking (physical constraints for accuracy)
try:
    from fretboard_position import (
        PositionTracker,
        PositionTrackerConfig,
        Position,
        PositionType,
        create_position_tracker,
        apply_position_constraints,
        add_position_tracking_args,
        config_from_args as position_config_from_args
    )
    HAS_POSITION_TRACKING = True
except ImportError:
    HAS_POSITION_TRACKING = False

# Basic Pitch support via Docker
import json
import shutil

def check_basicpitch_available() -> bool:
    """Check if basic-pitch Docker container is available."""
    return shutil.which('docker') is not None

HAS_BASICPITCH = check_basicpitch_available()

# Guitar tunings (MIDI note numbers, low to high)
TUNINGS = {
    'standard': [40, 45, 50, 55, 59, 64],  # E A D G B E
    'drop_d': [38, 45, 50, 55, 59, 64],    # D A D G B E
    'drop_c': [36, 43, 48, 53, 57, 62],    # C G C F A D
    'half_step_down': [39, 44, 49, 54, 58, 63],  # Eb Ab Db Gb Bb Eb
    'full_step_down': [38, 43, 48, 53, 57, 62],  # D G C F A D
    'open_d': [38, 45, 50, 54, 57, 62],    # D A D F# A D
    'open_g': [38, 43, 50, 55, 59, 62],    # D G D G B D
    'dadgad': [38, 45, 50, 55, 57, 62],    # D A D G A D
}

STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
NUM_FRETS = 24

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Guitar frequency range (E2 to ~E6 with harmonics)
GUITAR_MIN_HZ = 75   # Below E2 (82 Hz) to catch drop tunings
GUITAR_MAX_HZ = 1400  # Above high frets on high E string

# Chord detection threshold (notes within this time window are considered simultaneous)
CHORD_TIME_THRESHOLD = 0.050  # 50ms


# ============================================================================
# ADVANCED ONSET DETECTION
# ============================================================================

@dataclass
class OnsetCandidate:
    """Represents a candidate onset from one detection method."""
    time: float
    strength: float
    method: str
    is_legato: bool = False  # True for hammer-ons/pull-offs


@dataclass
class EnsembleOnset:
    """Final onset after ensemble voting."""
    time: float
    confidence: float  # Higher = more methods agreed
    is_legato: bool = False
    methods: List[str] = field(default_factory=list)
    attack_strength: float = 0.0  # Guitar transient strength


class OnsetDetector:
    """
    Advanced onset detection with multiple methods and ensemble voting.
    Specifically tuned for guitar audio.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        # Method weights for voting
        method_weights: Optional[Dict[str, float]] = None,
        # Detection thresholds
        spectral_flux_threshold: float = 0.3,
        complex_domain_threshold: float = 0.3,
        hfc_threshold: float = 0.3,
        energy_threshold: float = 0.3,
        # Ensemble parameters
        time_tolerance: float = 0.03,  # 30ms tolerance for grouping onsets
        min_votes: int = 2,  # Minimum methods that must agree
        # Guitar-specific parameters
        attack_detection: bool = True,
        legato_detection: bool = True,
        legato_sensitivity: float = 0.4,  # Lower = detect more legatos
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Default weights prioritize methods that work well for guitar
        self.method_weights = method_weights or {
            'spectral_flux': 1.0,
            'complex_domain': 0.9,
            'hfc': 0.8,
            'energy': 0.6,
            'attack': 1.2,  # Guitar attack detection gets extra weight
        }
        
        self.spectral_flux_threshold = spectral_flux_threshold
        self.complex_domain_threshold = complex_domain_threshold
        self.hfc_threshold = hfc_threshold
        self.energy_threshold = energy_threshold
        
        self.time_tolerance = time_tolerance
        self.min_votes = min_votes
        
        self.attack_detection = attack_detection
        self.legato_detection = legato_detection
        self.legato_sensitivity = legato_sensitivity
    
    def detect_all(self, y: np.ndarray) -> List[EnsembleOnset]:
        """
        Run all onset detection methods and combine with ensemble voting.
        
        Returns list of EnsembleOnset with confidence scores.
        """
        # Compute spectrogram once for efficiency
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        S_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Collect candidates from all methods
        all_candidates: List[OnsetCandidate] = []
        
        # 1. Spectral Flux
        sf_onsets = self._detect_spectral_flux(S)
        all_candidates.extend(sf_onsets)
        
        # 2. Complex Domain
        cd_onsets = self._detect_complex_domain(S_complex)
        all_candidates.extend(cd_onsets)
        
        # 3. High Frequency Content
        hfc_onsets = self._detect_hfc(S)
        all_candidates.extend(hfc_onsets)
        
        # 4. Energy-based
        energy_onsets = self._detect_energy(y)
        all_candidates.extend(energy_onsets)
        
        # 5. Guitar Attack Detection (transient-focused)
        if self.attack_detection:
            attack_onsets = self._detect_guitar_attacks(y)
            all_candidates.extend(attack_onsets)
        
        # 6. Legato detection (hammer-ons/pull-offs)
        if self.legato_detection:
            legato_onsets = self._detect_legato(y, S)
            all_candidates.extend(legato_onsets)
        
        # Ensemble voting
        final_onsets = self._ensemble_vote(all_candidates)
        
        return final_onsets
    
    def _detect_spectral_flux(self, S: np.ndarray) -> List[OnsetCandidate]:
        """
        Spectral Flux: measures change in spectral energy over time.
        Good for detecting percussive onsets and note attacks.
        """
        # Compute spectral flux (only positive changes = onsets)
        diff = np.diff(S, axis=1)
        diff = np.maximum(0, diff)  # Half-wave rectification
        flux = np.sum(diff, axis=0)
        
        # Normalize
        flux = flux / (np.max(flux) + 1e-10)
        
        # Pick peaks
        peaks, properties = find_peaks(
            flux,
            height=self.spectral_flux_threshold,
            distance=int(0.05 * self.sr / self.hop_length),  # Min 50ms between onsets
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak + 1, sr=self.sr, hop_length=self.hop_length)
            strength = flux[peak]
            candidates.append(OnsetCandidate(time, strength, 'spectral_flux'))
        
        return candidates
    
    def _detect_complex_domain(self, S_complex: np.ndarray) -> List[OnsetCandidate]:
        """
        Complex Domain: uses phase and magnitude changes.
        Better at detecting pitched note onsets than spectral flux.
        """
        # Compute complex domain onset function
        # Uses prediction error based on phase/magnitude
        n_frames = S_complex.shape[1]
        
        if n_frames < 3:
            return []
        
        odf = np.zeros(n_frames)
        
        for i in range(2, n_frames):
            # Predicted frame based on previous phase progression
            prev_phase = np.angle(S_complex[:, i-1])
            prev_prev_phase = np.angle(S_complex[:, i-2])
            pred_phase = 2 * prev_phase - prev_prev_phase
            
            prev_mag = np.abs(S_complex[:, i-1])
            pred_mag = prev_mag  # Simple prediction
            
            # Target value (in polar form, convert to rectangular for comparison)
            target = pred_mag * np.exp(1j * pred_phase)
            actual = S_complex[:, i]
            
            # Complex domain difference
            odf[i] = np.sum(np.abs(actual - target))
        
        # Normalize
        odf = odf / (np.max(odf) + 1e-10)
        
        # Pick peaks
        peaks, _ = find_peaks(
            odf,
            height=self.complex_domain_threshold,
            distance=int(0.05 * self.sr / self.hop_length),
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak, sr=self.sr, hop_length=self.hop_length)
            strength = odf[peak]
            candidates.append(OnsetCandidate(time, strength, 'complex_domain'))
        
        return candidates
    
    def _detect_hfc(self, S: np.ndarray) -> List[OnsetCandidate]:
        """
        High Frequency Content: weighted sum emphasizing high frequencies.
        Guitar pick attacks have strong high-frequency transients.
        """
        n_bins = S.shape[0]
        weights = np.arange(1, n_bins + 1)  # Linear frequency weighting
        
        # Weighted sum of spectral magnitudes
        hfc = np.sum(S * weights[:, np.newaxis], axis=0)
        
        # Compute onset strength (first derivative, half-wave rectified)
        hfc_diff = np.diff(hfc)
        hfc_diff = np.maximum(0, hfc_diff)
        
        # Normalize
        hfc_diff = hfc_diff / (np.max(hfc_diff) + 1e-10)
        
        # Pick peaks
        peaks, _ = find_peaks(
            hfc_diff,
            height=self.hfc_threshold,
            distance=int(0.05 * self.sr / self.hop_length),
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak + 1, sr=self.sr, hop_length=self.hop_length)
            strength = hfc_diff[peak]
            candidates.append(OnsetCandidate(time, strength, 'hfc'))
        
        return candidates
    
    def _detect_energy(self, y: np.ndarray) -> List[OnsetCandidate]:
        """
        Energy-based onset detection.
        Uses RMS energy envelope changes.
        """
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Compute onset strength (log-domain for perceptual relevance)
        log_rms = np.log1p(rms * 100)
        
        # First derivative, half-wave rectified
        rms_diff = np.diff(log_rms)
        rms_diff = np.maximum(0, rms_diff)
        
        # Normalize
        rms_diff = rms_diff / (np.max(rms_diff) + 1e-10)
        
        # Pick peaks
        peaks, _ = find_peaks(
            rms_diff,
            height=self.energy_threshold,
            distance=int(0.05 * self.sr / self.hop_length),
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak + 1, sr=self.sr, hop_length=self.hop_length)
            strength = rms_diff[peak]
            candidates.append(OnsetCandidate(time, strength, 'energy'))
        
        return candidates
    
    def _detect_guitar_attacks(self, y: np.ndarray) -> List[OnsetCandidate]:
        """
        Guitar-specific attack detection.
        Focuses on the sharp transients from pick/finger attacks.
        
        Guitar attacks have:
        - Very fast rise time (< 10ms)
        - High-frequency content in the initial transient
        - Quick decay of high frequencies after attack
        """
        # High-pass filter to isolate transients
        # Guitar attack transients are typically 2-6 kHz
        nyq = self.sr / 2
        high_cutoff = min(2000 / nyq, 0.95)  # 2kHz high-pass
        
        try:
            b, a = butter(4, high_cutoff, btype='high')
            y_hp = filtfilt(b, a, y)
        except Exception:
            # Fallback if filter design fails
            y_hp = y
        
        # Compute envelope using Hilbert transform alternative (peak following)
        # Fast attack, slow decay envelope follower
        envelope = np.abs(y_hp)
        
        # Smooth envelope
        win_size = int(0.003 * self.sr)  # 3ms window
        envelope = maximum_filter1d(envelope, size=win_size)
        
        # Downsample to hop_length resolution
        n_frames = len(y) // self.hop_length
        envelope_ds = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.hop_length, len(envelope))
            envelope_ds[i] = np.max(envelope[start:end])
        
        # Compute attack strength (rapid rises)
        attack = np.diff(envelope_ds)
        attack = np.maximum(0, attack)
        
        # Normalize
        attack = attack / (np.max(attack) + 1e-10)
        
        # Pick peaks with stricter threshold
        peaks, _ = find_peaks(
            attack,
            height=0.4,  # Higher threshold for transients
            distance=int(0.03 * self.sr / self.hop_length),  # 30ms min distance
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak + 1, sr=self.sr, hop_length=self.hop_length)
            strength = attack[peak]
            candidates.append(OnsetCandidate(time, strength, 'attack'))
        
        return candidates
    
    def _detect_legato(self, y: np.ndarray, S: np.ndarray) -> List[OnsetCandidate]:
        """
        Detect hammer-ons and pull-offs (legato notes).
        
        These have:
        - Softer attack than picked notes
        - Pitch change without strong transient
        - Maintained energy level
        
        Strategy: look for pitch changes without corresponding energy attacks.
        """
        # Detect pitch changes using spectral centroid movement
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        
        # Normalize centroid
        centroid_norm = (centroid - np.mean(centroid)) / (np.std(centroid) + 1e-10)
        
        # Compute centroid change (pitch movement)
        centroid_diff = np.abs(np.diff(centroid_norm))
        
        # Compute RMS energy for comparison
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_diff = np.abs(np.diff(rms))
        rms_diff = rms_diff / (np.max(rms_diff) + 1e-10)
        
        # Legato = significant pitch change without significant attack
        # Look for frames where centroid changes but energy doesn't spike
        legato_score = np.zeros(len(centroid_diff))
        
        for i in range(len(centroid_diff)):
            pitch_change = centroid_diff[i]
            energy_attack = rms_diff[i] if i < len(rms_diff) else 0
            
            # High pitch change with low energy attack = possible legato
            if pitch_change > 0.5 and energy_attack < self.legato_sensitivity:
                legato_score[i] = pitch_change * (1 - energy_attack)
        
        # Normalize
        if np.max(legato_score) > 0:
            legato_score = legato_score / np.max(legato_score)
        
        # Pick peaks
        peaks, _ = find_peaks(
            legato_score,
            height=0.3,
            distance=int(0.05 * self.sr / self.hop_length),
        )
        
        candidates = []
        for peak in peaks:
            time = librosa.frames_to_time(peak + 1, sr=self.sr, hop_length=self.hop_length)
            strength = legato_score[peak] * 0.7  # Lower weight for legatos
            candidates.append(OnsetCandidate(time, strength, 'legato', is_legato=True))
        
        return candidates
    
    def _ensemble_vote(self, candidates: List[OnsetCandidate]) -> List[EnsembleOnset]:
        """
        Combine onset candidates using weighted voting.
        
        Groups nearby candidates and scores based on:
        - Number of methods that agree
        - Weighted sum of detection strengths
        - Method reliability weights
        """
        if not candidates:
            return []
        
        # Sort by time
        candidates.sort(key=lambda c: c.time)
        
        # Group candidates within time tolerance
        groups: List[List[OnsetCandidate]] = []
        current_group: List[OnsetCandidate] = [candidates[0]]
        
        for cand in candidates[1:]:
            if cand.time - current_group[0].time <= self.time_tolerance:
                current_group.append(cand)
            else:
                groups.append(current_group)
                current_group = [cand]
        groups.append(current_group)
        
        # Vote on each group
        final_onsets: List[EnsembleOnset] = []
        
        for group in groups:
            # Get unique methods in this group
            methods = set(c.method for c in group)
            
            # Count votes (weighted)
            total_weight = sum(self.method_weights.get(c.method, 1.0) for c in group)
            unique_methods = len(methods)
            
            # Check if enough methods agree
            # Special case: if attack detector fires, count as extra vote
            has_attack = 'attack' in methods
            effective_votes = unique_methods + (0.5 if has_attack else 0)
            
            if effective_votes >= self.min_votes or (has_attack and unique_methods >= 1):
                # Calculate weighted average time
                weighted_time = sum(
                    c.time * self.method_weights.get(c.method, 1.0) 
                    for c in group
                ) / total_weight
                
                # Calculate confidence
                max_votes = len(self.method_weights)
                confidence = min(1.0, effective_votes / max_votes)
                
                # Check if this is a legato onset
                is_legato = any(c.is_legato for c in group) and not has_attack
                
                # Calculate attack strength
                attack_strength = max(
                    (c.strength for c in group if c.method == 'attack'),
                    default=0.0
                )
                
                final_onsets.append(EnsembleOnset(
                    time=weighted_time,
                    confidence=confidence,
                    is_legato=is_legato,
                    methods=list(methods),
                    attack_strength=attack_strength
                ))
        
        return final_onsets


def detect_onsets_ensemble(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_votes: int = 2,
    attack_detection: bool = True,
    legato_detection: bool = True,
    legato_sensitivity: float = 0.4,
    verbose: bool = True
) -> Tuple[np.ndarray, List[EnsembleOnset]]:
    """
    Convenience function for ensemble onset detection.
    
    Returns:
        onset_times: numpy array of onset times in seconds
        onset_details: list of EnsembleOnset with full details
    """
    detector = OnsetDetector(
        sr=sr,
        hop_length=hop_length,
        min_votes=min_votes,
        attack_detection=attack_detection,
        legato_detection=legato_detection,
        legato_sensitivity=legato_sensitivity,
    )
    
    onsets = detector.detect_all(y)
    
    if verbose:
        print(f"  Ensemble onset detection found {len(onsets)} onsets")
        n_legato = sum(1 for o in onsets if o.is_legato)
        if n_legato > 0:
            print(f"  ({n_legato} detected as legato/hammer-on/pull-off)")
        
        # Method contribution stats
        method_counts: Dict[str, int] = {}
        for onset in onsets:
            for method in onset.methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            print("  Method contributions:")
            for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
                print(f"    {method}: {count}")
    
    onset_times = np.array([o.time for o in onsets])
    return onset_times, onsets


def apply_onset_backtracking(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    hop_length: int = 512,
    window_ms: float = 50.0
) -> np.ndarray:
    """
    Backtrack onsets to find true attack starts.
    
    Looks backward from detected onset to find where energy
    first starts rising (the actual attack start).
    """
    # Compute RMS energy at high resolution
    frame_length = 256
    rms = librosa.feature.rms(
        y=y, 
        frame_length=frame_length, 
        hop_length=frame_length // 2
    )[0]
    
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), 
        sr=sr, 
        hop_length=frame_length // 2
    )
    
    backtracked = []
    window_sec = window_ms / 1000.0
    
    for onset_time in onset_times:
        # Find RMS frame closest to onset
        onset_idx = np.searchsorted(rms_times, onset_time)
        
        # Look backwards within window
        window_frames = int(window_sec * sr / (frame_length // 2))
        start_idx = max(0, onset_idx - window_frames)
        
        if start_idx >= onset_idx or onset_idx >= len(rms):
            backtracked.append(onset_time)
            continue
        
        # Find where RMS starts rising
        window_rms = rms[start_idx:onset_idx + 1]
        
        if len(window_rms) < 2:
            backtracked.append(onset_time)
            continue
        
        # Find the minimum in the window (start of rise)
        min_idx = np.argmin(window_rms)
        backtrack_idx = start_idx + min_idx
        
        # Convert back to time
        new_onset = rms_times[backtrack_idx]
        backtracked.append(new_onset)
    
    return np.array(backtracked)


# Common chord definitions: chord_name -> intervals from root (semitones)
CHORD_INTERVALS = {
    # Major chords
    '': [0, 4, 7],  # Major (no suffix)
    'maj': [0, 4, 7],
    # Minor chords
    'm': [0, 3, 7],
    'min': [0, 3, 7],
    # 7th chords
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    # Sus chords
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    # Diminished/Augmented
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    # Add chords
    'add9': [0, 4, 7, 14],
    # Power chord
    '5': [0, 7],
}

# Common open chord shapes: chord_name -> [(string, fret), ...] where -1 = muted
OPEN_CHORD_SHAPES = {
    'C': [(0, -1), (1, 3), (2, 2), (3, 0), (4, 1), (5, 0)],
    'Am': [(0, -1), (1, 0), (2, 2), (3, 2), (4, 1), (5, 0)],
    'G': [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)],
    'D': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 3), (5, 2)],
    'Dm': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 3), (5, 1)],
    'E': [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    'Em': [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    'A': [(0, -1), (1, 0), (2, 2), (3, 2), (4, 2), (5, 0)],
    'F': [(0, 1), (1, 3), (2, 3), (3, 2), (4, 1), (5, 1)],  # Barre chord
    'Bm': [(0, -1), (1, 2), (2, 4), (3, 4), (4, 3), (5, 2)],  # Barre chord
    'B7': [(0, -1), (1, 2), (2, 1), (3, 2), (4, 0), (5, 2)],
    'G7': [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 1)],
    'C7': [(0, -1), (1, 3), (2, 2), (3, 3), (4, 1), (5, 0)],
    'D7': [(0, -1), (1, -1), (2, 0), (3, 2), (4, 1), (5, 2)],
    'A7': [(0, -1), (1, 0), (2, 2), (3, 0), (4, 2), (5, 0)],
    'E7': [(0, 0), (1, 2), (2, 0), (3, 1), (4, 0), (5, 0)],
}


@dataclass
class Note:
    """Represents a detected note"""
    midi: int
    start_time: float
    duration: float
    confidence: float
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)
    
    @property
    def note_class(self) -> int:
        """Return pitch class (0-11)"""
        return self.midi % 12
    
    @property
    def note_name(self) -> str:
        """Return just the note name without octave"""
        return NOTE_NAMES[self.midi % 12]
    
    @property
    def frequency(self) -> float:
        return 440 * (2 ** ((self.midi - 69) / 12))


@dataclass
class TabNote:
    """A note on the guitar fretboard"""
    string: int  # 0-5 (low E to high e)
    fret: int    # 0-24
    start_time: float
    duration: float
    
    def __str__(self):
        return f"String {STRING_NAMES[self.string]}, Fret {self.fret}"


@dataclass
class Chord:
    """Represents a detected chord"""
    name: str  # e.g., "Am", "G", "C7"
    root: int  # MIDI pitch class (0-11)
    notes: List[Note]  # Notes that make up this chord
    start_time: float
    duration: float
    quality: str = ""  # "maj", "m", "7", etc.
    is_barre: bool = False
    confidence: float = 0.0
    
    @property
    def root_name(self) -> str:
        return NOTE_NAMES[self.root]
    
    def __str__(self) -> str:
        return self.name


def get_tuning(tuning_name: str) -> List[int]:
    """Get tuning by name, with helpful error message."""
    if tuning_name.lower() in TUNINGS:
        return TUNINGS[tuning_name.lower()]
    
    # Try to parse custom tuning (e.g., "38,45,50,55,59,64")
    try:
        notes = [int(n.strip()) for n in tuning_name.split(',')]
        if len(notes) == 6:
            return notes
    except ValueError:
        pass
    
    available = ', '.join(TUNINGS.keys())
    raise ValueError(f"Unknown tuning '{tuning_name}'. Available: {available}\n"
                    f"Or provide custom MIDI notes: e.g., '38,45,50,55,59,64'")


def midi_to_fret_options(midi_note: int, tuning: List[int] = None) -> List[Tuple[int, int]]:
    """
    Given a MIDI note, return all possible (string, fret) combinations.
    Returns list of (string_index, fret_number) tuples.
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    options = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= NUM_FRETS:
            options.append((string_idx, fret))
    return options


def choose_best_fret_position(
    note: Note,
    prev_position: Optional[Tuple[int, int]],
    tuning: List[int] = None
) -> Optional[TabNote]:
    """
    Choose the best fret position for a note, considering:
    - Playability (prefer lower frets for beginners)
    - Hand position continuity (stay close to previous note)
    - String preference (middle strings often easier)
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    options = midi_to_fret_options(note.midi, tuning)
    
    if not options:
        return None  # Note out of guitar range
    
    if len(options) == 1:
        string, fret = options[0]
        return TabNote(string, fret, note.start_time, note.duration)
    
    # Score each option
    scored_options = []
    for string, fret in options:
        score = 0
        
        # Prefer lower frets (easier to play) but not too aggressively
        if fret <= 5:
            score += 3  # Bonus for first position
        elif fret <= 12:
            score += 1  # Small bonus for comfortable range
        score -= fret * 0.2
        
        # Prefer middle strings (easier access)
        string_preference = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5]  # Middle strings preferred
        score += string_preference[string] * 3
        
        # If we have a previous position, prefer staying close
        if prev_position:
            prev_string, prev_fret = prev_position
            # Penalize large jumps
            fret_distance = abs(fret - prev_fret)
            string_distance = abs(string - prev_string)
            
            # Heavy penalty for large fret jumps
            if fret_distance > 4:
                score -= fret_distance * 0.5
            else:
                score -= fret_distance * 0.2
            
            # Prefer staying on same or adjacent strings
            score -= string_distance * 0.8
        
        scored_options.append((score, string, fret))
    
    # Sort by score (highest first)
    scored_options.sort(reverse=True)
    _, best_string, best_fret = scored_options[0]
    
    return TabNote(best_string, best_fret, note.start_time, note.duration)


def apply_median_filter(pitches: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to reduce pitch jitter."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    
    # Only filter valid (non-zero) pitches
    valid_mask = pitches > 0
    if not np.any(valid_mask):
        return pitches
    
    # Apply median filter
    filtered = medfilt(pitches, kernel_size=kernel_size)
    
    # Preserve zero values (no pitch detected)
    result = np.where(valid_mask, filtered, pitches)
    return result


def smooth_confidence(confidence: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooth confidence values using a moving average."""
    if len(confidence) < window_size:
        return confidence
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(confidence, kernel, mode='same')
    return smoothed


def detect_pitch_pyin(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect pitch using pYIN algorithm - better for monophonic sources.
    Returns: (f0, voiced_flag, voiced_probs)
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=GUITAR_MIN_HZ,
        fmax=GUITAR_MAX_HZ,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0
    )
    return f0, voiced_flag, voiced_probs


def detect_pitch_piptrack(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect pitch using piptrack algorithm.
    Returns: (pitches_per_frame, magnitudes_per_frame)
    """
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=GUITAR_MIN_HZ,
        fmax=GUITAR_MAX_HZ
    )
    return pitches, magnitudes


def detect_pitch_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    model_capacity: str = 'full',
    viterbi: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect pitch using CREPE (Convolutional Representation for Pitch Estimation).
    
    CREPE is a deep learning model specifically trained for monophonic pitch detection.
    It is more accurate than traditional DSP methods like pYIN, especially for:
    - Noisy audio
    - Guitar recordings with string squeaks
    - Fast passages
    - Expressive playing with vibrato/bends
    
    Args:
        y: Audio signal (will be resampled to 16kHz internally)
        sr: Sample rate of input audio
        hop_length: Hop length for output alignment (CREPE uses step_size in ms)
        model_capacity: 'tiny', 'small', 'medium', 'large', or 'full'
                       Larger models are more accurate but slower.
        viterbi: Apply Viterbi smoothing to pitch curve (recommended)
    
    Returns:
        f0: Pitch values in Hz (0 for unvoiced frames)
        voiced_flag: Boolean array of voiced frame indicators
        confidence: Confidence values (0-1)
    
    Requires:
        pip install crepe tensorflow
    """
    if not HAS_CREPE:
        raise RuntimeError(
            "CREPE not available. Install with: pip install crepe tensorflow\n"
            "Or use --pitch-method pyin instead."
        )
    
    # Convert hop_length to step_size in milliseconds
    # CREPE default step_size is 10ms
    step_size_ms = int((hop_length / sr) * 1000)
    step_size_ms = max(10, step_size_ms)  # Minimum 10ms
    
    print(f"  CREPE: Using {model_capacity} model, step_size={step_size_ms}ms, viterbi={viterbi}")
    
    # Run CREPE prediction
    time, frequency, confidence, activation = crepe.predict(
        y,
        sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        step_size=step_size_ms,
        verbose=0
    )
    
    # Create voiced flag based on confidence threshold
    # CREPE confidence is voice activity probability
    voiced_threshold = 0.5
    voiced_flag = confidence >= voiced_threshold
    
    # Zero out unvoiced frames (consistent with pyin behavior)
    f0 = np.where(voiced_flag, frequency, 0.0)
    
    # Filter to guitar range
    f0 = np.where((f0 >= GUITAR_MIN_HZ) & (f0 <= GUITAR_MAX_HZ), f0, 0.0)
    
    # Update voiced flag based on filtered f0
    voiced_flag = f0 > 0
    
    print(f"  CREPE: Detected {np.sum(voiced_flag)} voiced frames out of {len(f0)}")
    
    return f0, voiced_flag, confidence


def detect_pitch_hps(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 4096,
    num_harmonics: int = 5,
    fmin: float = GUITAR_MIN_HZ,
    fmax: float = GUITAR_MAX_HZ
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect pitch using Harmonic Product Spectrum (HPS).
    
    HPS is CRITICAL for distorted guitar where harmonics dominate the fundamental.
    
    The algorithm:
    1. Compute FFT magnitude spectrum
    2. Downsample spectrum by factors of 2, 3, 4, 5
    3. Multiply all downsampled versions together
    4. The fundamental frequency "pops out" as the peak
    
    This works because:
    - Fundamental at f0 appears at f0
    - 2nd harmonic at 2*f0, when downsampled 2x, aligns with f0
    - 3rd harmonic at 3*f0, when downsampled 3x, aligns with f0
    - etc.
    
    So even when harmonics are STRONGER than the fundamental (common in 
    distorted guitar), they all vote for the true fundamental frequency.
    
    Args:
        y: Audio signal
        sr: Sample rate  
        hop_length: Hop length between frames
        n_fft: FFT size (larger = better frequency resolution)
        num_harmonics: Number of harmonic products (2-5 typical)
        fmin: Minimum frequency to consider
        fmax: Maximum frequency to consider
        
    Returns:
        f0: Fundamental frequency per frame (Hz), 0 for unvoiced
        voiced_flag: Boolean array of voiced frame indicators
        confidence: Confidence values (0-1) based on peak prominence
    """
    print(f"  HPS: Using {num_harmonics} harmonic products, n_fft={n_fft}")
    
    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    n_bins, n_frames = S.shape
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find bin indices for frequency range
    min_bin = np.searchsorted(freqs, fmin)
    max_bin = np.searchsorted(freqs, fmax)
    
    # Ensure we have enough bins for downsampling
    # Need at least max_bin * num_harmonics bins
    max_valid_bin = n_bins // num_harmonics
    max_bin = min(max_bin, max_valid_bin)
    
    if min_bin >= max_bin:
        print(f"  HPS: Warning - frequency range too narrow, adjusting")
        min_bin = max(1, max_bin - 100)
    
    f0 = np.zeros(n_frames)
    confidence = np.zeros(n_frames)
    voiced_flag = np.zeros(n_frames, dtype=bool)
    
    for frame_idx in range(n_frames):
        spectrum = S[:, frame_idx]
        
        # Skip silent frames
        if np.max(spectrum) < 1e-10:
            continue
        
        # Normalize spectrum
        spectrum_norm = spectrum / (np.max(spectrum) + 1e-10)
        
        # Initialize HPS with original spectrum (limited to valid range)
        # We need spectrum up to max_bin * num_harmonics for downsampling
        hps_len = max_bin
        hps = spectrum_norm[:hps_len].copy()
        
        # Multiply by downsampled versions
        for h in range(2, num_harmonics + 1):
            # Downsample by factor h (take every h-th sample)
            # This aligns harmonic h with the fundamental
            downsampled_len = hps_len
            downsampled = np.zeros(downsampled_len)
            
            for i in range(downsampled_len):
                src_idx = i * h
                if src_idx < len(spectrum_norm):
                    downsampled[i] = spectrum_norm[src_idx]
            
            # Multiply into HPS
            hps *= downsampled
        
        # Find the peak in the valid frequency range
        search_start = min_bin
        search_end = max_bin
        
        if search_start >= search_end or search_end > len(hps):
            continue
        
        hps_search = hps[search_start:search_end]
        
        if len(hps_search) == 0 or np.max(hps_search) < 1e-20:
            continue
        
        # Find peak
        peak_idx_local = np.argmax(hps_search)
        peak_idx = peak_idx_local + search_start
        peak_val = hps[peak_idx]
        
        # Calculate confidence based on peak prominence
        # Compare peak to median of surrounding bins
        window = 10
        start = max(search_start, peak_idx - window)
        end = min(search_end, peak_idx + window)
        local_median = np.median(hps[start:end])
        
        if local_median > 0:
            prominence = peak_val / local_median
            conf = min(1.0, prominence / 10.0)  # Normalize: 10x prominence = full confidence
        else:
            conf = 0.5 if peak_val > 0 else 0.0
        
        # Only accept if confidence is reasonable
        if conf > 0.2:
            f0[frame_idx] = freqs[peak_idx]
            confidence[frame_idx] = conf
            voiced_flag[frame_idx] = True
    
    # Apply median filter to reduce octave jumps
    if np.sum(voiced_flag) > 5:
        valid_f0 = f0[voiced_flag]
        filtered = medfilt(valid_f0, kernel_size=min(5, len(valid_f0) | 1))
        f0[voiced_flag] = filtered
    
    n_voiced = np.sum(voiced_flag)
    print(f"  HPS: Detected {n_voiced} voiced frames out of {n_frames}")
    
    if n_voiced > 0:
        avg_f0 = np.mean(f0[voiced_flag])
        print(f"  HPS: Average pitch = {avg_f0:.1f} Hz ({librosa.hz_to_note(avg_f0)})")
    
    return f0, voiced_flag, confidence


def detect_notes_basicpitch(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_len: int = 50,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    confidence_threshold: float = 0.3
) -> List['Note']:
    """
    Detect notes using Spotify's Basic Pitch model via Docker.
    
    Basic Pitch is a neural network specifically designed for polyphonic music 
    transcription. Unlike pYIN or CREPE which are monophonic, Basic Pitch can
    detect multiple simultaneous notes - perfect for guitar chords.
    
    The model runs in a Docker container with TensorFlow 2.15 and Python 3.11
    to ensure compatibility (basic-pitch requires numpy <1.24 which doesn't 
    work with Python 3.13).
    
    Args:
        audio_path: Path to audio file
        onset_threshold: Minimum confidence for note onset (0-1)
        frame_threshold: Minimum confidence for note frames (0-1)
        min_note_len: Minimum note length in milliseconds
        min_freq: Minimum frequency in Hz (None for guitar default: 75 Hz)
        max_freq: Maximum frequency in Hz (None for guitar default: 1400 Hz)
        confidence_threshold: Minimum confidence to include note
        
    Returns:
        List of Note objects with detected pitches
        
    Requires:
        Docker with the guitar-tabs-basicpitch image:
        cd guitar-tabs/basicpitch && docker build -t guitar-tabs-basicpitch .
    """
    if not HAS_BASICPITCH:
        raise RuntimeError(
            "Basic Pitch requires Docker. Install Docker and build the image:\n"
            "  cd guitar-tabs/basicpitch && docker build -t guitar-tabs-basicpitch .\n"
            "Or use --pitch-method pyin instead."
        )
    
    # Check if Docker image exists
    import subprocess
    result = subprocess.run(
        ['docker', 'images', '-q', 'guitar-tabs-basicpitch'],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        raise RuntimeError(
            "Basic Pitch Docker image not found. Build it with:\n"
            "  cd guitar-tabs/basicpitch && docker build -t guitar-tabs-basicpitch ."
        )
    
    # Get absolute path for Docker mount
    abs_path = os.path.abspath(audio_path)
    dir_path = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    
    # Build Docker command
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{dir_path}:/data',
        'guitar-tabs-basicpitch',
        f'/data/{file_name}',
        '--onset-threshold', str(onset_threshold),
        '--frame-threshold', str(frame_threshold),
        '--min-note-len', str(min_note_len),
    ]
    
    if min_freq is not None:
        cmd.extend(['--min-freq', str(min_freq)])
    if max_freq is not None:
        cmd.extend(['--max-freq', str(max_freq)])
    
    print(f"🎵 Running Basic Pitch in Docker container...")
    print(f"  File: {audio_path}")
    print(f"  Thresholds: onset={onset_threshold}, frame={frame_threshold}")
    
    # Run Basic Pitch in Docker
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Basic Pitch failed:\n{result.stderr}")
    
    # Parse JSON output (Basic Pitch writes JSON to stdout)
    try:
        # Find the JSON output (skip any TensorFlow warnings)
        stdout_lines = result.stdout.strip().split('\n')
        json_start = None
        for i, line in enumerate(stdout_lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start is None:
            raise ValueError("No JSON output found from Basic Pitch")
        
        json_str = '\n'.join(stdout_lines[json_start:])
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Basic Pitch output: {e}\n{result.stdout}")
    
    # Convert to Note objects
    notes = []
    for note_data in data.get('notes', []):
        if note_data['confidence'] >= confidence_threshold:
            notes.append(Note(
                midi=note_data['midi'],
                start_time=note_data['start_time'],
                duration=note_data['duration'],
                confidence=note_data['confidence']
            ))
    
    print(f"  ✅ Basic Pitch detected {len(notes)} notes (filtered from {data.get('total_notes', '?')} total)")
    
    return notes


def detect_pitch_cqt(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,  # 7 octaves * 12 semitones
    bins_per_octave: int = 12,
    fmin: float = None,
    threshold_ratio: float = 0.3,
    n_peaks: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect pitch using Constant-Q Transform (CQT).
    
    CQT is ideal for music analysis because:
    - Logarithmic frequency resolution matches musical notes
    - Each bin corresponds directly to a semitone
    - Better captures harmonic structure of guitar
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for CQT
        n_bins: Number of CQT bins (84 = 7 octaves)
        bins_per_octave: Bins per octave (12 = semitones)
        fmin: Minimum frequency (default: C1 ~32.7 Hz)
        threshold_ratio: Minimum magnitude ratio to consider a peak
        n_peaks: Maximum number of peaks to detect per frame
        
    Returns:
        (f0, confidence, all_peaks_per_frame)
        - f0: Dominant frequency per frame
        - confidence: Confidence for each frame
        - all_peaks_per_frame: List of (freq, magnitude) tuples for multi-pitch
    """
    if fmin is None:
        fmin = librosa.note_to_hz('C1')  # ~32.7 Hz
    
    # Compute CQT
    C = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin
    ))
    
    # Get frequency for each bin
    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    n_frames = C.shape[1]
    f0 = np.zeros(n_frames)
    confidence = np.zeros(n_frames)
    all_peaks = []
    
    for i in range(n_frames):
        frame = C[:, i]
        max_mag = frame.max()
        
        if max_mag < 1e-10:
            all_peaks.append([])
            continue
        
        # Normalize frame
        frame_norm = frame / max_mag
        
        # Find peaks above threshold
        threshold = threshold_ratio
        peak_mask = frame_norm >= threshold
        
        # Also check for local maxima
        local_max = np.zeros_like(frame_norm, dtype=bool)
        for j in range(1, len(frame_norm) - 1):
            if frame_norm[j] > frame_norm[j-1] and frame_norm[j] > frame_norm[j+1]:
                local_max[j] = True
        # Handle edges
        if len(frame_norm) > 1:
            if frame_norm[0] > frame_norm[1]:
                local_max[0] = True
            if frame_norm[-1] > frame_norm[-2]:
                local_max[-1] = True
        
        # Combine: must be above threshold AND local maximum
        valid_peaks = peak_mask & local_max
        peak_indices = np.where(valid_peaks)[0]
        
        if len(peak_indices) == 0:
            # Fall back to just max if no peaks
            peak_indices = [np.argmax(frame_norm)]
        
        # Sort by magnitude and take top n_peaks
        peak_mags = frame_norm[peak_indices]
        sorted_idx = np.argsort(peak_mags)[::-1][:n_peaks]
        top_peaks = peak_indices[sorted_idx]
        
        # Store all peaks for this frame
        frame_peaks = [(cqt_freqs[idx], frame[idx]) for idx in top_peaks]
        all_peaks.append(frame_peaks)
        
        # Primary pitch is the strongest peak within guitar range
        for idx in top_peaks:
            freq = cqt_freqs[idx]
            if GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                f0[i] = freq
                confidence[i] = frame_norm[idx]
                break
        
        # If no peak in guitar range, take the strongest overall
        if f0[i] == 0 and len(top_peaks) > 0:
            idx = top_peaks[0]
            f0[i] = cqt_freqs[idx]
            confidence[i] = frame_norm[idx]
    
    return f0, confidence, all_peaks


def detect_notes_cqt(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    onset_frames: np.ndarray = None,
    confidence_threshold: float = 0.3,
    min_note_duration: float = 0.05,
    median_filter_size: int = 5,
    tuning: List[int] = None
) -> List[Note]:
    """
    Detect notes using CQT-based pitch detection.
    
    This approach uses peak detection on CQT magnitude bins,
    where each bin corresponds directly to a musical semitone.
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length
        onset_frames: Pre-detected onset frames (optional)
        confidence_threshold: Minimum confidence for note detection
        min_note_duration: Minimum note duration in seconds
        median_filter_size: Size of median filter for smoothing
        tuning: Guitar tuning as MIDI notes
        
    Returns:
        List of detected Note objects
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    print("Computing CQT (Constant-Q Transform)...")
    
    # Detect pitches using CQT
    f0, confidence, all_peaks = detect_pitch_cqt(
        y, sr, hop_length,
        threshold_ratio=confidence_threshold
    )
    
    # Apply median filter to reduce jitter
    if median_filter_size > 1:
        f0 = apply_median_filter(f0, median_filter_size)
    
    # Smooth confidence
    confidence = smooth_confidence(confidence, window_size=5)
    
    # Detect onsets if not provided
    if onset_frames is None:
        print("Detecting onsets with backtracking...")
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='frames'
        )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    print(f"Found {len(onset_times)} onsets")
    
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    audio_duration = librosa.get_duration(y=y, sr=sr)
    
    notes = []
    
    for i, onset_time in enumerate(onset_times):
        onset_frame = onset_frames[i] if i < len(onset_frames) else int(onset_time * sr / hop_length)
        
        if onset_frame >= len(f0):
            continue
        
        # Look at a small window around onset for stability
        window_start = max(0, onset_frame)
        window_end = min(len(f0), onset_frame + 5)
        
        # Get pitches and confidences in window
        pitch_window = f0[window_start:window_end]
        conf_window = confidence[window_start:window_end]
        
        # Find valid pitches
        valid_mask = pitch_window > 0
        if not np.any(valid_mask):
            continue
        
        valid_pitches = pitch_window[valid_mask]
        valid_confs = conf_window[valid_mask]
        
        if len(valid_confs) == 0:
            continue
        
        # Use pitch with highest confidence
        best_idx = np.argmax(valid_confs)
        pitch_hz = valid_pitches[best_idx]
        note_confidence = valid_confs[best_idx]
        
        if pitch_hz <= 0 or note_confidence < confidence_threshold:
            continue
        
        # Convert Hz to MIDI
        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
        
        # Sanity check: guitar range
        if midi_note < 36 or midi_note > 90:
            continue
        
        # Estimate duration
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - onset_time
        else:
            duration = audio_duration - onset_time
        
        if duration < min_note_duration:
            continue
        
        notes.append(Note(
            midi=midi_note,
            start_time=onset_time,
            duration=duration,
            confidence=float(note_confidence)
        ))
    
    # Also try to detect notes from CQT peaks between onsets
    # This helps catch notes that don't have clear onsets
    notes = detect_sustained_notes_cqt(
        notes, f0, confidence, frame_times, 
        confidence_threshold, min_note_duration
    )
    
    return notes


def detect_sustained_notes_cqt(
    existing_notes: List[Note],
    f0: np.ndarray,
    confidence: np.ndarray,
    frame_times: np.ndarray,
    confidence_threshold: float,
    min_note_duration: float
) -> List[Note]:
    """
    Detect sustained notes from CQT data that may not have clear onsets.
    
    This is a secondary pass that looks for stable pitch regions
    not already covered by onset-based detection.
    """
    # Create a mask of frames already covered by detected notes
    covered = np.zeros(len(f0), dtype=bool)
    
    for note in existing_notes:
        start_idx = np.searchsorted(frame_times, note.start_time)
        end_idx = np.searchsorted(frame_times, note.start_time + note.duration)
        covered[start_idx:end_idx] = True
    
    # Find continuous regions of uncovered, high-confidence pitch
    in_note = False
    note_start = 0
    current_midi = 0
    midi_votes = []
    
    for i in range(len(f0)):
        if covered[i]:
            # End any ongoing note
            if in_note and i - note_start >= 3:  # At least 3 frames
                duration = frame_times[i] - frame_times[note_start]
                if duration >= min_note_duration and midi_votes:
                    # Use mode of MIDI votes
                    from collections import Counter
                    midi_mode = Counter(midi_votes).most_common(1)[0][0]
                    avg_conf = sum(confidence[note_start:i]) / (i - note_start)
                    
                    existing_notes.append(Note(
                        midi=midi_mode,
                        start_time=frame_times[note_start],
                        duration=duration,
                        confidence=float(avg_conf)
                    ))
            in_note = False
            midi_votes = []
            continue
        
        has_pitch = f0[i] > 0 and confidence[i] >= confidence_threshold
        
        if has_pitch:
            midi = int(round(librosa.hz_to_midi(f0[i])))
            
            if not in_note:
                # Start new note
                in_note = True
                note_start = i
                current_midi = midi
                midi_votes = [midi]
            else:
                # Continue note if pitch is close (within 2 semitones)
                if abs(midi - current_midi) <= 2:
                    midi_votes.append(midi)
                else:
                    # Pitch changed significantly - end current note, start new
                    duration = frame_times[i] - frame_times[note_start]
                    if duration >= min_note_duration and len(midi_votes) >= 3:
                        from collections import Counter
                        midi_mode = Counter(midi_votes).most_common(1)[0][0]
                        avg_conf = sum(confidence[note_start:i]) / (i - note_start)
                        
                        existing_notes.append(Note(
                            midi=midi_mode,
                            start_time=frame_times[note_start],
                            duration=duration,
                            confidence=float(avg_conf)
                        ))
                    
                    # Start new note
                    note_start = i
                    current_midi = midi
                    midi_votes = [midi]
        else:
            # No pitch - end any ongoing note
            if in_note and i - note_start >= 3:
                duration = frame_times[i] - frame_times[note_start]
                if duration >= min_note_duration and midi_votes:
                    from collections import Counter
                    midi_mode = Counter(midi_votes).most_common(1)[0][0]
                    avg_conf = sum(confidence[note_start:i]) / (i - note_start)
                    
                    existing_notes.append(Note(
                        midi=midi_mode,
                        start_time=frame_times[note_start],
                        duration=duration,
                        confidence=float(avg_conf)
                    ))
            in_note = False
            midi_votes = []
    
    # Handle note at end of audio
    if in_note and len(f0) - note_start >= 3:
        duration = frame_times[-1] - frame_times[note_start]
        if duration >= min_note_duration and midi_votes:
            from collections import Counter
            midi_mode = Counter(midi_votes).most_common(1)[0][0]
            avg_conf = sum(confidence[note_start:]) / (len(f0) - note_start)
            
            existing_notes.append(Note(
                midi=midi_mode,
                start_time=frame_times[note_start],
                duration=duration,
                confidence=float(avg_conf)
            ))
    
    # Sort by start time
    existing_notes.sort(key=lambda n: n.start_time)
    
    return existing_notes


def extract_harmonic_component(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Separate harmonic component from percussive for better pitch detection.
    Guitar strings have strong harmonic content.
    """
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
    return y_harmonic


def suppress_harmonics(midi_notes: List[int], min_interval: float = 0.05) -> List[int]:
    """
    Suppress octave harmonics - guitar often shows strong 2nd/3rd harmonics.
    Keep the fundamental frequency (lowest note in close timing).
    """
    if len(midi_notes) <= 1:
        return midi_notes
    
    filtered = []
    for note in midi_notes:
        is_harmonic = False
        for existing in filtered:
            # Check if this note is an octave (12 semitones) above existing
            diff = note - existing
            if diff in [12, 19, 24]:  # Octave, octave+fifth, 2 octaves
                is_harmonic = True
                break
        if not is_harmonic:
            filtered.append(note)
    
    return filtered


def detect_notes_with_voting(
    audio_path: str,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    confidence_threshold: float = 0.3,
    use_harmonic_separation: bool = True,
    tuning: List[int] = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    freq_cleanup_config: Optional['FreqCleanupConfig'] = None,
    save_preprocessed: Optional[str] = None,
    min_votes: int = 2,
    verbose: bool = True,
    distortion_preprocess_config: Optional['DistortionPreprocessingConfig'] = None
) -> List[Note]:
    """
    Detect notes using multi-detector voting for maximum accuracy.
    
    This method runs multiple pitch detectors (pYIN, CQT, piptrack, YIN)
    and uses consensus voting to determine the correct pitch. This approach:
    
    1. Filters out detector-specific errors
    2. Resolves octave ambiguities
    3. Enforces guitar range constraints (E2-E6)
    4. Tracks pitch continuity to prevent random jumps
    
    Args:
        audio_path: Path to audio file
        hop_length: Hop length for analysis
        min_note_duration: Minimum note duration in seconds
        confidence_threshold: Minimum confidence for note detection
        use_harmonic_separation: Whether to use HPSS preprocessing
        tuning: Guitar tuning (MIDI notes)
        preprocess_config: Optional preprocessing configuration
        save_preprocessed: Optional path to save preprocessed audio
        min_votes: Minimum detector votes for consensus
        verbose: Print diagnostic info
        distortion_preprocess_config: Optional distortion-specific preprocessing config
        
    Returns:
        List of detected Note objects
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    print(f"🎯 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Apply frequency domain cleanup if configured (before pitch detection)
    if HAS_FREQ_CLEANUP and freq_cleanup_config and freq_cleanup_config.enabled:
        print("\n🎸 Applying frequency domain cleanup...")
        y = freq_domain_cleanup(y, sr, freq_cleanup_config, verbose=True)
        print()
    
    # Apply audio preprocessing if configured
    if preprocess_config and preprocess_config.enabled:
        print("\n🔧 Applying audio preprocessing pipeline...")
        y = preprocess_audio(y, sr, preprocess_config, verbose=True)
        print()
        
        if save_preprocessed:
            print(f"💾 Saving preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    # Apply distortion-specific preprocessing if configured
    if distortion_preprocess_config and distortion_preprocess_config.enabled:
        print("\n⚡ Applying distortion preprocessing pipeline...")
        y = preprocess_distortion(y, sr, distortion_preprocess_config, verbose=True)
        print()
        
        if save_preprocessed and not (preprocess_config and preprocess_config.enabled):
            print(f"💾 Saving distortion-preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    # Optional: Extract harmonic component for cleaner pitch detection
    if use_harmonic_separation:
        print("Separating harmonic component...")
        y_pitch = extract_harmonic_component(y, sr)
    else:
        y_pitch = y
    
    # Run accurate pitch detection with voting
    print(f"\n🎯 Running multi-detector voting (min_votes={min_votes})...")
    
    f0, confidence, pitch_details = detect_pitches_accurate(
        y_pitch, sr, hop_length,
        min_votes=min_votes,
        min_confidence=confidence_threshold,
        verbose=verbose
    )
    
    # Detect onsets using ensemble method
    print("\nDetecting onsets with ensemble method...")
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
    
    # Apply backtracking to refine onset times
    if len(onset_times) > 0:
        onset_times = apply_onset_backtracking(y, sr, onset_times, hop_length)
    
    print(f"Found {len(onset_times)} onsets (ensemble)")
    
    # Convert frame-based pitches to note events at onsets
    notes = []
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    audio_duration = librosa.get_duration(y=y, sr=sr)
    
    for i, onset_time in enumerate(onset_times):
        # Find the frame closest to this onset
        onset_frame = int(onset_time * sr / hop_length)
        
        if onset_frame >= len(f0):
            continue
        
        # Look at a window around onset for stability
        window_start = max(0, onset_frame)
        window_end = min(len(f0), onset_frame + 5)
        
        # Get valid pitches in window
        valid_pitches = []
        valid_confs = []
        for frame in range(window_start, window_end):
            if frame < len(pitch_details) and pitch_details[frame].is_valid and pitch_details[frame].midi > 0:
                valid_pitches.append(pitch_details[frame].midi)
                valid_confs.append(pitch_details[frame].confidence)
        
        if not valid_pitches:
            continue
        
        # Use most common pitch (mode) with highest confidence
        from collections import Counter
        pitch_counts = Counter(valid_pitches)
        most_common = pitch_counts.most_common(1)[0]
        midi_note = most_common[0]
        
        # Calculate confidence as average of matching pitches
        matching_confs = [c for p, c in zip(valid_pitches, valid_confs) if p == midi_note]
        note_confidence = np.mean(matching_confs) if matching_confs else 0.5
        
        # Get onset details for legato detection
        onset_detail = None
        for detail in onset_details:
            if abs(detail.time - onset_time) < 0.05:
                onset_detail = detail
                break
        
        # Legato notes get lower confidence threshold
        effective_threshold = confidence_threshold
        if onset_detail and onset_detail.is_legato:
            effective_threshold = confidence_threshold * 0.6
        
        if note_confidence < effective_threshold:
            continue
        
        # Verify guitar range (E2=40 to E6=88)
        if midi_note < 40 or midi_note > 88:
            continue
        
        # Estimate duration
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - onset_time
        else:
            duration = audio_duration - onset_time
        
        # Minimum duration check
        effective_min_duration = min_note_duration
        if onset_detail and onset_detail.is_legato:
            effective_min_duration = min_note_duration * 0.7
        
        if duration < effective_min_duration:
            continue
        
        notes.append(Note(
            midi=midi_note,
            start_time=onset_time,
            duration=duration,
            confidence=float(note_confidence)
        ))
    
    # Post-process: suppress obvious harmonics
    print("Filtering harmonics...")
    notes = filter_harmonic_notes(notes)
    
    print(f"\n✅ Detected {len(notes)} notes with voting consensus")
    
    # Print note distribution
    if notes and verbose:
        midi_notes = [n.midi for n in notes]
        note_names = [NOTE_NAMES[m % 12] for m in midi_notes]
        print(f"   Range: {NOTE_NAMES[min(midi_notes) % 12]}{min(midi_notes)//12-1} to {NOTE_NAMES[max(midi_notes) % 12]}{max(midi_notes)//12-1}")
        print(f"   Most common: {Counter(note_names).most_common(3)}")
    
    return notes


def detect_notes_from_audio(
    audio_path: str,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    confidence_threshold: float = 0.5,
    pitch_method: str = 'voting',
    use_harmonic_separation: bool = True,
    median_filter_size: int = 5,
    tuning: List[int] = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    freq_cleanup_config: Optional['FreqCleanupConfig'] = None,
    save_preprocessed: Optional[str] = None,
    crepe_model: str = 'small',
    octave_correction: bool = True,
    lead_isolation_config: Optional['LeadIsolationConfig'] = None,
    distortion_mode: bool = False,
    distortion_harmonics: int = 5,
    distortion_preprocess_config: Optional['DistortionPreprocessingConfig'] = None
) -> List[Note]:
    """
    Detect notes from audio file using advanced pitch detection.
    
    Args:
        audio_path: Path to audio file
        hop_length: Hop length for STFT
        min_note_duration: Minimum note duration in seconds
        confidence_threshold: Minimum confidence for note detection
        pitch_method: 'voting', 'pyin', 'piptrack', 'cqt', 'crepe', 'basicpitch', or 'hps'
            - voting: Multi-detector consensus (default, MOST ACCURATE)
            - pyin: Fast, good for monophonic
            - piptrack: Alternative monophonic
            - cqt: Constant-Q transform based
            - crepe: Deep learning (requires tensorflow)
            - basicpitch: Polyphonic - best for chords (requires Docker)
            - hps: Harmonic Product Spectrum - BEST FOR DISTORTED GUITAR
        use_harmonic_separation: Whether to use HPSS
        median_filter_size: Size of median filter for pitch smoothing
        tuning: Guitar tuning (MIDI notes)
        preprocess_config: Optional preprocessing configuration
        save_preprocessed: Optional path to save preprocessed audio
        crepe_model: CREPE model capacity ('tiny', 'small', 'medium', 'large', 'full')
        distortion_mode: Enable distortion-aware pitch detection (HPS-based)
        distortion_harmonics: Number of harmonics for HPS in distortion mode
        distortion_preprocess_config: Optional distortion-specific preprocessing config
    """
    # Distortion-aware pitch detection using Harmonic Product Spectrum
    # Best for heavily distorted electric guitar
    if distortion_mode and HAS_DISTORTION_PITCH:
        print("\n🎸 Distortion-Aware Pitch Detection Mode")
        print(f"   Using Harmonic Product Spectrum with {distortion_harmonics} harmonics")
        print("   Optimized for: heavy distortion, power chords, high-gain tones")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Apply frequency domain cleanup if configured
        if HAS_FREQ_CLEANUP and freq_cleanup_config and freq_cleanup_config.enabled:
            print("\n🎸 Applying frequency domain cleanup...")
            y = freq_domain_cleanup(y, sr, freq_cleanup_config, verbose=True)
        
        # Apply preprocessing if configured
        if preprocess_config and preprocess_config.enabled:
            print("\n🔧 Applying audio preprocessing...")
            y = preprocess_audio(y, sr, preprocess_config, verbose=True)
        
        # Apply distortion-specific preprocessing if configured
        if distortion_preprocess_config and distortion_preprocess_config.enabled:
            print("\n⚡ Applying distortion preprocessing...")
            y = preprocess_distortion(y, sr, distortion_preprocess_config, verbose=True)
        
        # Configure distortion detector
        dist_config = DistortionConfig(
            hps_harmonics=distortion_harmonics,
            min_confidence=confidence_threshold * 0.8,  # Lower threshold for HPS
            n_fft=4096,  # Higher resolution for low frequencies
            hop_length=hop_length,
            verbose=True,
            detect_intermod=True,
            apply_compression_compensation=True
        )
        
        # Run distortion-aware detection
        detector = DistortionPitchDetector(sr=sr, config=dist_config)
        dist_results = detector.detect(y)
        
        # Convert to Note objects
        audio_duration = librosa.get_duration(y=y, sr=sr)
        notes = distortion_convert_to_notes(
            dist_results,
            min_duration=min_note_duration,
            audio_duration=audio_duration
        )
        
        # Filter harmonics
        print("Filtering harmonics...")
        notes = filter_harmonic_notes(notes)
        
        # Apply octave correction if enabled
        if octave_correction and HAS_OCTAVE_CORRECTION and len(notes) > 0:
            print("Applying octave correction...")
            notes = _apply_octave_correction_to_notes(y, sr, notes, tuning)
        
        # Summary statistics
        power_chord_count = sum(1 for r in dist_results if r.is_power_chord)
        print(f"\n✅ Detected {len(notes)} notes (distortion-aware)")
        print(f"   Power chords detected: {power_chord_count}")
        
        return notes
    elif distortion_mode and not HAS_DISTORTION_PITCH:
        print("⚠️  distortion_pitch module not available, falling back to standard detection")
    
    # Multi-model ensemble - MOST ACCURATE (runs pyin, cqt, crepe, basic_pitch, piptrack)
    if pitch_method == 'ensemble':
        if not HAS_ENSEMBLE:
            print("⚠️  ensemble_pitch module not available, falling back to voting")
            pitch_method = 'voting'
        else:
            from ensemble_pitch import EnsembleConfig, detect_notes_ensemble
            
            config = EnsembleConfig(
                min_votes=2,
                min_confidence=confidence_threshold,
                min_duration=min_note_duration,
                crepe_model=crepe_model
            )
            
            consensus_notes = detect_notes_ensemble(
                audio_path,
                config=config,
                sr=22050,
                use_harmonic_separation=use_harmonic_separation
            )
            
            # Convert ConsensusNote to Note objects
            notes = [
                Note(
                    midi=cn.midi_note,
                    start_time=cn.start_time,
                    duration=cn.duration,
                    confidence=cn.confidence
                )
                for cn in consensus_notes
            ]
            
            print(f"Ensemble detected {len(notes)} notes")
            return notes
    
    # Precise note segmentation (BEST FOR NOTE BOUNDARIES)
    if pitch_method == 'segment':
        if not HAS_NOTE_SEGMENTATION:
            print("⚠️  note_segmentation module not available, falling back to voting")
            pitch_method = 'voting'
        else:
            print("🎯 Using precise note segmentation...")
            segments = segment_notes(
                audio_path,
                sr=22050,
                hop_length=256,  # Smaller hop for precision
                onset_threshold=0.3,
                min_confidence=confidence_threshold * 0.6,  # Lower for better detection
                min_note_duration_ms=min_note_duration * 1000,
                use_harmonic_separation=use_harmonic_separation,
                verbose=True
            )
            
            # Convert to Note objects
            notes = convert_to_notes(segments)
            
            # Post-process: suppress obvious harmonics
            print("Filtering harmonics...")
            notes = filter_harmonic_notes(notes)
            
            # Apply octave correction
            if octave_correction and HAS_OCTAVE_CORRECTION and len(notes) > 0:
                print("Applying octave correction...")
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                notes = _apply_octave_correction_to_notes(y, sr, notes, tuning)
            
            print(f"Final note count: {len(notes)}")
            return notes
    
    # Multi-detector voting - legacy method
    if pitch_method == 'voting':
        if not HAS_PITCH_ACCURACY:
            print("⚠️  pitch_accuracy module not available, falling back to pYIN")
            pitch_method = 'pyin'
        else:
            return detect_notes_with_voting(
                audio_path=audio_path,
                hop_length=hop_length,
                min_note_duration=min_note_duration,
                confidence_threshold=confidence_threshold,
                use_harmonic_separation=use_harmonic_separation,
                tuning=tuning,
                preprocess_config=preprocess_config,
                freq_cleanup_config=freq_cleanup_config,
                save_preprocessed=save_preprocessed,
                distortion_preprocess_config=distortion_preprocess_config
            )
    
    # Basic Pitch is a special case - it does its own audio loading
    # and returns notes directly (polyphonic detection via Docker)
    if pitch_method == 'basicpitch':
        if not HAS_BASICPITCH:
            print("⚠️  Basic Pitch requires Docker, falling back to pYIN")
            pitch_method = 'pyin'
        else:
            notes = detect_notes_basicpitch(
                audio_path,
                onset_threshold=0.5,
                frame_threshold=0.3,
                min_note_len=int(min_note_duration * 1000),
                confidence_threshold=confidence_threshold
            )
            # Filter by minimum duration
            notes = [n for n in notes if n.duration >= min_note_duration]
            # Post-process: suppress obvious harmonics
            print("Filtering harmonics...")
            notes = filter_harmonic_notes(notes)
            
            # Apply octave correction
            if octave_correction and HAS_OCTAVE_CORRECTION and len(notes) > 0:
                print("Applying octave correction...")
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                notes = _apply_octave_correction_to_notes(y, sr, notes, tuning)
            
            print(f"Final note count: {len(notes)}")
            return notes
    
    print(f"Loading audio: {audio_path}")
    
    # Apply lead guitar isolation if configured (does its own loading with demucs)
    if lead_isolation_config and lead_isolation_config.enabled and HAS_LEAD_ISOLATION:
        print("\n🎸 Isolating lead guitar...")
        save_lead = getattr(lead_isolation_config, '_save_path', None)
        y, sr = isolate_lead_guitar(audio_path, config=lead_isolation_config, 
                                     verbose=True, save_path=save_lead)
        # Resample to 22050 if needed
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        print()
    else:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Apply audio preprocessing if configured
    if preprocess_config and preprocess_config.enabled:
        print("\n🔧 Applying audio preprocessing pipeline...")
        y = preprocess_audio(y, sr, preprocess_config, verbose=True)
        print()
        
        # Optionally save preprocessed audio for debugging
        if save_preprocessed:
            print(f"💾 Saving preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    # Apply distortion-specific preprocessing if configured
    if distortion_preprocess_config and distortion_preprocess_config.enabled:
        print("\n⚡ Applying distortion preprocessing pipeline...")
        y = preprocess_distortion(y, sr, distortion_preprocess_config, verbose=True)
        print()
        
        # Save preprocessed audio for debugging if requested
        if save_preprocessed and not (preprocess_config and preprocess_config.enabled):
            print(f"💾 Saving distortion-preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    # Optional: Extract harmonic component for cleaner pitch detection
    if use_harmonic_separation:
        print("Separating harmonic component...")
        y_pitch = extract_harmonic_component(y, sr)
    else:
        y_pitch = y
    
    print(f"Detecting pitches using {pitch_method}...")
    
    # CQT method - use dedicated function
    if pitch_method == 'cqt':
        # Detect onsets first (needed by CQT method)
        print("Detecting onsets with backtracking...")
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='frames'
        )
        
        notes = detect_notes_cqt(
            y_pitch, sr, hop_length,
            onset_frames=onset_frames,
            confidence_threshold=confidence_threshold,
            min_note_duration=min_note_duration,
            median_filter_size=median_filter_size,
            tuning=tuning
        )
        
        # Post-process: suppress obvious harmonics
        print("Filtering harmonics...")
        notes = filter_harmonic_notes(notes)
        
        # Apply octave correction
        if octave_correction and HAS_OCTAVE_CORRECTION and len(notes) > 0:
            print("Applying octave correction...")
            notes = _apply_octave_correction_to_notes(y, sr, notes, tuning)
        
        print(f"Detected {len(notes)} notes")
        return notes
    
    # pYIN, CREPE, or piptrack methods
    # Handle CREPE fallback first
    if pitch_method == 'crepe' and not HAS_CREPE:
        print("⚠️  CREPE not available, falling back to pYIN")
        print("   Install with: pip install crepe tensorflow")
        pitch_method = 'pyin'
    
    if pitch_method == 'crepe':
        # CREPE is a deep learning model for monophonic pitch - most accurate
        f0, voiced_flag, confidence = detect_pitch_crepe(
            y_pitch, sr, hop_length,
            model_capacity=crepe_model,
            viterbi=True  # Smooth pitch curve
        )
        
        # Apply median filter to reduce jitter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        # Smooth confidence values
        confidence = smooth_confidence(confidence, window_size=5)
    
    elif pitch_method == 'pyin':
        # pYIN is better for monophonic sources like guitar
        f0, voiced_flag, voiced_probs = detect_pitch_pyin(y_pitch, sr, hop_length)
        
        # Apply median filter to reduce jitter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        # Smooth confidence values
        confidence = smooth_confidence(voiced_probs, window_size=5)
        
    elif pitch_method == 'piptrack':
        pitches, magnitudes = detect_pitch_piptrack(y_pitch, sr, hop_length)
        
        # Extract dominant pitch per frame
        f0 = np.zeros(pitches.shape[1])
        confidence = np.zeros(pitches.shape[1])
        
        for i in range(pitches.shape[1]):
            mag_slice = magnitudes[:, i]
            pitch_slice = pitches[:, i]
            
            if mag_slice.max() > 0:
                max_idx = mag_slice.argmax()
                f0[i] = pitch_slice[max_idx]
                confidence[i] = mag_slice[max_idx] / (magnitudes.max() + 1e-10)
        
        # Apply median filter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        confidence = smooth_confidence(confidence, window_size=5)
    
    elif pitch_method == 'hps':
        # Harmonic Product Spectrum - CRITICAL for distorted guitar
        # where harmonics are often stronger than the fundamental
        f0, voiced_flag, confidence = detect_pitch_hps(
            y_pitch, sr, hop_length,
            n_fft=4096,  # Larger FFT for better frequency resolution
            num_harmonics=5,  # Use 5 harmonic products
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ
        )
        
        # Apply median filter to reduce jitter
        if median_filter_size > 1:
            f0 = apply_median_filter(f0, median_filter_size)
        
        # Smooth confidence values
        confidence = smooth_confidence(confidence, window_size=5)
    
    print("Detecting onsets with ensemble method...")
    # Use advanced ensemble onset detection
    onset_times, onset_details = detect_onsets_ensemble(
        y=y,
        sr=sr,
        hop_length=hop_length,
        min_votes=2,  # At least 2 methods must agree
        attack_detection=True,  # Use guitar-specific attack detection
        legato_detection=True,  # Detect hammer-ons/pull-offs
        legato_sensitivity=0.4,
        verbose=True
    )
    
    # Apply backtracking to refine onset times
    if len(onset_times) > 0:
        onset_times = apply_onset_backtracking(y, sr, onset_times, hop_length)
    
    # Create a mapping of onset time to details (for legato info)
    onset_info_map = {o.time: o for o in onset_details}
    
    print(f"Found {len(onset_times)} onsets (ensemble)")
    
    notes = []
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    for i, onset_time in enumerate(onset_times):
        # Find the frame closest to this onset
        onset_frame = int(onset_time * sr / hop_length)
        
        if onset_frame >= len(f0):
            continue
        
        # Get pitch and confidence at onset
        # Look at a small window around onset for stability
        window_start = max(0, onset_frame)
        window_end = min(len(f0), onset_frame + 5)
        
        pitch_window = f0[window_start:window_end]
        conf_window = confidence[window_start:window_end]
        
        # Find valid pitches in window
        valid_mask = pitch_window > 0
        if not np.any(valid_mask):
            continue
        
        # Use the pitch with highest confidence in the window
        valid_pitches = pitch_window[valid_mask]
        valid_confs = conf_window[valid_mask]
        
        if len(valid_confs) == 0:
            continue
            
        best_idx = np.argmax(valid_confs)
        pitch_hz = valid_pitches[best_idx]
        note_confidence = valid_confs[best_idx]
        
        # Get onset details (for legato info)
        onset_detail = None
        for detail in onset_details:
            if abs(detail.time - onset_time) < 0.05:  # 50ms tolerance
                onset_detail = detail
                break
        
        # Legato notes (hammer-ons/pull-offs) get lower confidence threshold
        effective_threshold = confidence_threshold
        if onset_detail and onset_detail.is_legato:
            effective_threshold = confidence_threshold * 0.6  # 40% lower threshold
        
        if pitch_hz <= 0 or note_confidence < effective_threshold:
            continue
        
        # Convert Hz to MIDI note number
        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
        
        # Sanity check: is this within guitar range?
        # Standard guitar: E2 (40) to about E6 (88) at highest fret
        if midi_note < 36 or midi_note > 90:
            continue
        
        # Estimate duration (until next onset or end)
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - onset_time
        else:
            duration = librosa.get_duration(y=y, sr=sr) - onset_time
        
        # Legato notes often have slightly shorter minimum duration requirement
        effective_min_duration = min_note_duration
        if onset_detail and onset_detail.is_legato:
            effective_min_duration = min_note_duration * 0.7
        
        # Filter by duration
        if duration < effective_min_duration:
            continue
        
        # Boost confidence if multiple onset methods agreed
        if onset_detail and onset_detail.confidence > 0.5:
            note_confidence = min(1.0, note_confidence * (1 + onset_detail.confidence * 0.3))
        
        notes.append(Note(
            midi=midi_note,
            start_time=onset_time,
            duration=duration,
            confidence=float(note_confidence)
        ))
    
    # Post-process: suppress obvious harmonics
    print("Filtering harmonics...")
    notes = filter_harmonic_notes(notes)
    
    # Apply octave correction using harmonic analysis
    if octave_correction and HAS_OCTAVE_CORRECTION and len(notes) > 0:
        print("Applying octave correction...")
        notes = _apply_octave_correction_to_notes(y, sr, notes, tuning)
    
    print(f"Detected {len(notes)} notes")
    return notes


def _apply_octave_correction_to_notes(
    y: np.ndarray,
    sr: int,
    notes: List[Note],
    tuning: List[int] = None
) -> List[Note]:
    """
    Apply octave correction to detected notes using harmonic analysis.
    
    This corrects common octave errors where pitch detectors lock onto
    the wrong octave (e.g., detecting E2 instead of E3).
    """
    if not HAS_OCTAVE_CORRECTION:
        return notes
    
    # Determine tuning name for octave correction constraints
    tuning_name = 'standard'
    if tuning:
        # Find matching tuning
        for name, midi_notes in TUNINGS.items():
            if midi_notes == tuning:
                tuning_name = name
                break
    
    corrector = OctaveCorrector(sr=sr, tuning=tuning_name)
    
    # Convert Note objects to dict format
    note_dicts = []
    for note in notes:
        note_dicts.append({
            'midi': note.midi,
            'time': note.start_time,
            'duration': note.duration,
            'confidence': note.confidence
        })
    
    # Apply correction
    corrected_dicts, results = corrector.correct_notes(y, note_dicts, verbose=False)
    
    # Count corrections for logging
    corrections = [r for r in results if r.correction != OctaveCorrection.NONE]
    if corrections:
        print(f"  Corrected {len(corrections)} octave errors")
    
    # Convert back to Note objects
    corrected_notes = []
    for d in corrected_dicts:
        corrected_notes.append(Note(
            midi=d['midi'],
            start_time=d['time'],
            duration=d['duration'],
            confidence=d['confidence']
        ))
    
    return corrected_notes


def filter_harmonic_notes(notes: List[Note], time_threshold: float = 0.05) -> List[Note]:
    """
    Filter out notes that are likely harmonics of other notes.
    If two notes start at nearly the same time and one is an octave/fifth above,
    keep only the lower (fundamental) one.
    """
    if len(notes) <= 1:
        return notes
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.midi))
    filtered = []
    
    i = 0
    while i < len(sorted_notes):
        current = sorted_notes[i]
        
        # Find all notes starting within time_threshold
        group = [current]
        j = i + 1
        while j < len(sorted_notes) and sorted_notes[j].start_time - current.start_time < time_threshold:
            group.append(sorted_notes[j])
            j += 1
        
        if len(group) == 1:
            filtered.append(current)
        else:
            # Keep notes that aren't harmonics of others in the group
            group_midis = [n.midi for n in group]
            kept = []
            
            for note in group:
                is_harmonic = False
                for other in group:
                    if other.midi < note.midi:
                        diff = note.midi - other.midi
                        # Common harmonic intervals: octave (12), octave+fifth (19), 2 octaves (24)
                        if diff in [12, 19, 24, 28, 31]:
                            is_harmonic = True
                            break
                
                if not is_harmonic:
                    kept.append(note)
            
            filtered.extend(kept if kept else [group[0]])  # Keep at least one note
        
        i = j if j > i else i + 1
    
    return filtered


# ============================================================================
# POLYPHONIC PITCH DETECTION (Multiple simultaneous notes)
# ============================================================================

def create_harmonic_templates(n_bins: int, n_notes: int = 88, n_harmonics: int = 6,
                               fmin: float = 27.5, bins_per_octave: int = 36) -> np.ndarray:
    """
    Create harmonic templates for NMF-based pitch detection.
    Each template represents the harmonic series of a musical note.
    
    Args:
        n_bins: Number of frequency bins in the spectrogram
        n_notes: Number of notes to model (88 for full piano range)
        n_harmonics: Number of harmonics to include per note
        fmin: Minimum frequency (A0 = 27.5 Hz)
        bins_per_octave: CQT bins per octave
        
    Returns:
        Template matrix of shape (n_bins, n_notes)
    """
    templates = np.zeros((n_bins, n_notes))
    
    for note_idx in range(n_notes):
        # Fundamental frequency for this note (A0 = 27.5 Hz)
        f0 = fmin * (2 ** (note_idx / 12))
        
        # Add harmonics with decreasing amplitude
        for h in range(1, n_harmonics + 1):
            freq = f0 * h
            
            # Convert frequency to bin index (CQT-style mapping)
            if freq > 0:
                bin_idx = int(round(bins_per_octave * np.log2(freq / fmin)))
                
                # Add Gaussian-shaped peak around the harmonic frequency
                if 0 <= bin_idx < n_bins:
                    # Harmonic amplitude decay (1/h weighting)
                    amplitude = 1.0 / h
                    
                    # Spread over nearby bins with Gaussian
                    spread = 2  # bins
                    for offset in range(-spread, spread + 1):
                        idx = bin_idx + offset
                        if 0 <= idx < n_bins:
                            weight = np.exp(-0.5 * (offset / spread) ** 2)
                            templates[idx, note_idx] += amplitude * weight
    
    # Normalize templates
    norms = np.linalg.norm(templates, axis=0, keepdims=True)
    norms[norms == 0] = 1
    templates = templates / norms
    
    return templates


def detect_pitches_nmf(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_components: int = 6,  # Max simultaneous notes (guitar has 6 strings)
    confidence_threshold: float = 0.3
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Detect multiple pitches per frame using NMF on spectrograms.
    
    Uses Constant-Q Transform (CQT) for better frequency resolution at low frequencies,
    then applies NMF to decompose into pitch activations.
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for CQT
        n_components: Maximum simultaneous notes to detect
        confidence_threshold: Minimum activation to count as a note
        
    Returns:
        pitches_per_frame: List of MIDI note lists for each frame
        activations: Raw activation matrix (n_notes x n_frames)
    """
    print("  Computing Constant-Q Transform for NMF...")
    
    # Compute CQT (better for music than STFT)
    fmin = librosa.note_to_hz('E2')  # Guitar low E
    n_bins = 72  # 6 octaves * 12 bins/octave
    
    C = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=12
    ))
    
    # Apply power scaling for better separation
    C = C ** 2
    
    # Normalize
    C_norm = C / (C.max() + 1e-10)
    
    print(f"  CQT shape: {C.shape}")
    
    # Apply NMF to find pitch components
    print("  Applying Non-negative Matrix Factorization...")
    
    # Initialize NMF
    nmf = NMF(
        n_components=n_components,
        init='nndsvd',
        max_iter=200,
        random_state=42
    )
    
    # W: frequency templates (n_bins x n_components)
    # H: activations over time (n_components x n_frames)
    W = nmf.fit_transform(C_norm)
    H = nmf.components_
    
    # Find fundamental frequency for each component
    component_pitches = []
    for comp_idx in range(n_components):
        template = W[:, comp_idx]
        
        # Find peaks in the template
        peaks, properties = find_peaks(template, height=0.1, distance=3)
        
        if len(peaks) > 0:
            # Get the fundamental (lowest significant peak)
            peak_heights = template[peaks]
            # Sort by height and take strongest
            sorted_idx = np.argsort(peak_heights)[::-1]
            
            # Use the lowest strong peak as fundamental
            strong_peaks = peaks[sorted_idx[:3]]  # Top 3 peaks
            fundamental_bin = strong_peaks.min()
            
            # Convert bin to MIDI note
            # CQT bin 0 = fmin (E2 = MIDI 40)
            midi_note = 40 + fundamental_bin
            component_pitches.append(midi_note)
        else:
            component_pitches.append(None)
    
    # Build pitch activations per frame
    n_frames = H.shape[1]
    pitches_per_frame = []
    
    # Normalize activations
    H_norm = H / (H.max() + 1e-10)
    
    for frame_idx in range(n_frames):
        frame_pitches = []
        frame_activations = H_norm[:, frame_idx]
        
        for comp_idx, midi_note in enumerate(component_pitches):
            if midi_note is not None:
                activation = frame_activations[comp_idx]
                if activation > confidence_threshold:
                    frame_pitches.append((midi_note, activation))
        
        # Sort by activation strength
        frame_pitches.sort(key=lambda x: x[1], reverse=True)
        pitches_per_frame.append([p[0] for p in frame_pitches])
    
    return pitches_per_frame, H


def detect_pitches_harmonic(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    max_pitches: int = 6,
    confidence_threshold: float = 0.3,
    fmin: float = 75.0,
    fmax: float = 1400.0
) -> List[List[Tuple[int, float]]]:
    """
    Detect multiple pitches using harmonic product spectrum + peak picking.
    
    This method:
    1. Separates harmonic content from percussive
    2. Computes STFT and applies harmonic product spectrum
    3. Picks multiple peaks per frame
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length
        max_pitches: Maximum pitches per frame
        confidence_threshold: Minimum confidence for a pitch
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        List of (midi_note, confidence) tuples per frame
    """
    print("  Separating harmonic/percussive components...")
    
    # Harmonic-percussive separation
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
    
    print("  Computing spectrogram for harmonic detection...")
    
    # Compute STFT
    D = librosa.stft(y_harmonic, hop_length=hop_length)
    S = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr)
    
    n_frames = S.shape[1]
    pitches_per_frame = []
    
    # Frequency bin indices for guitar range
    fmin_bin = np.searchsorted(freqs, fmin)
    fmax_bin = np.searchsorted(freqs, fmax)
    
    print(f"  Processing {n_frames} frames for harmonic peaks...")
    
    for frame_idx in range(n_frames):
        frame_spectrum = S[fmin_bin:fmax_bin, frame_idx]
        frame_freqs = freqs[fmin_bin:fmax_bin]
        
        if frame_spectrum.max() == 0:
            pitches_per_frame.append([])
            continue
        
        # Normalize spectrum
        frame_spectrum = frame_spectrum / frame_spectrum.max()
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(
            frame_spectrum,
            height=confidence_threshold,
            distance=3,  # Min distance between peaks
            prominence=0.1
        )
        
        if len(peaks) == 0:
            pitches_per_frame.append([])
            continue
        
        # Convert peaks to frequencies and MIDI notes
        frame_pitches = []
        peak_heights = properties['peak_heights']
        
        # Sort peaks by height
        sorted_idx = np.argsort(peak_heights)[::-1]
        
        detected_midis = set()
        
        for idx in sorted_idx[:max_pitches * 2]:  # Check more peaks than needed
            peak_bin = peaks[idx]
            freq = frame_freqs[peak_bin]
            confidence = peak_heights[idx]
            
            # Convert to MIDI
            midi_note = int(round(librosa.hz_to_midi(freq)))
            
            # Check if valid guitar range
            if midi_note < 36 or midi_note > 90:
                continue
            
            # Skip if we already have this note (avoid duplicates)
            if midi_note in detected_midis:
                continue
            
            # Check if this is a harmonic of an already-detected note
            is_harmonic = False
            for existing_midi in detected_midis:
                diff = midi_note - existing_midi
                if diff in [12, 19, 24]:  # Octave, octave+fifth, 2 octaves
                    is_harmonic = True
                    break
            
            if not is_harmonic:
                frame_pitches.append((midi_note, confidence))
                detected_midis.add(midi_note)
            
            if len(frame_pitches) >= max_pitches:
                break
        
        pitches_per_frame.append(frame_pitches)
    
    return pitches_per_frame


def detect_notes_polyphonic(
    audio_path: str,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    confidence_threshold: float = 0.3,
    method: str = 'nmf',  # 'nmf' or 'harmonic'
    max_simultaneous: int = 6,
    tuning: List[int] = None,
    preprocess_config: Optional[PreprocessingConfig] = None,
    save_preprocessed: Optional[str] = None,
    lead_isolation_config: Optional['LeadIsolationConfig'] = None,
    distortion_preprocess_config: Optional['DistortionPreprocessingConfig'] = None
) -> List[Note]:
    """
    Detect multiple simultaneous notes from audio (polyphonic pitch detection).
    
    Uses either:
    - NMF: Non-negative Matrix Factorization on CQT spectrogram
    - Harmonic: Harmonic/percussive separation + peak picking
    
    Args:
        audio_path: Path to audio file
        hop_length: Hop length for analysis
        min_note_duration: Minimum note duration in seconds
        confidence_threshold: Minimum confidence for note detection
        method: 'nmf' or 'harmonic'
        max_simultaneous: Maximum simultaneous notes (default 6 for guitar)
        tuning: Guitar tuning (MIDI notes)
        preprocess_config: Optional preprocessing configuration
        save_preprocessed: Optional path to save preprocessed audio
        distortion_preprocess_config: Optional distortion-specific preprocessing config
        
    Returns:
        List of detected Notes (may have multiple notes at same timestamp)
    """
    print(f"🎵 Polyphonic pitch detection using {method.upper()} method...")
    print(f"Loading audio: {audio_path}")
    
    # Apply lead guitar isolation if configured
    if lead_isolation_config and lead_isolation_config.enabled and HAS_LEAD_ISOLATION:
        print("\n🎸 Isolating lead guitar...")
        save_lead = getattr(lead_isolation_config, '_save_path', None)
        y, sr = isolate_lead_guitar(audio_path, config=lead_isolation_config,
                                     verbose=True, save_path=save_lead)
        # Resample to 22050 if needed
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        print()
    else:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Apply audio preprocessing if configured
    if preprocess_config and preprocess_config.enabled:
        print("\n🔧 Applying audio preprocessing pipeline...")
        y = preprocess_audio(y, sr, preprocess_config, verbose=True)
        print()
        
        # Optionally save preprocessed audio for debugging
        if save_preprocessed:
            print(f"💾 Saving preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    # Apply distortion-specific preprocessing if configured
    if distortion_preprocess_config and distortion_preprocess_config.enabled:
        print("\n⚡ Applying distortion preprocessing pipeline...")
        y = preprocess_distortion(y, sr, distortion_preprocess_config, verbose=True)
        print()
        
        # Save preprocessed audio for debugging if requested
        if save_preprocessed and not (preprocess_config and preprocess_config.enabled):
            print(f"💾 Saving distortion-preprocessed audio to: {save_preprocessed}")
            sf.write(save_preprocessed, y, sr)
    
    if method == 'nmf':
        print("Using NMF-based polyphonic detection...")
        pitches_per_frame, activations = detect_pitches_nmf(
            y, sr, hop_length,
            n_components=max_simultaneous,
            confidence_threshold=confidence_threshold
        )
        
        # Convert to time-stamped notes
        frame_times = librosa.frames_to_time(
            np.arange(len(pitches_per_frame)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Track active notes to determine durations
        active_notes: Dict[int, dict] = {}  # midi -> {start_time, start_frame, confidence}
        all_notes = []
        
        for frame_idx, (frame_pitches, frame_time) in enumerate(zip(pitches_per_frame, frame_times)):
            current_midis = set(frame_pitches)
            
            # End notes that are no longer active
            ended_notes = [m for m in active_notes if m not in current_midis]
            for midi in ended_notes:
                note_info = active_notes.pop(midi)
                duration = frame_time - note_info['start_time']
                if duration >= min_note_duration:
                    all_notes.append(Note(
                        midi=midi,
                        start_time=note_info['start_time'],
                        duration=duration,
                        confidence=note_info['confidence']
                    ))
            
            # Start new notes
            for midi in current_midis:
                if midi not in active_notes:
                    active_notes[midi] = {
                        'start_time': frame_time,
                        'start_frame': frame_idx,
                        'confidence': 0.7  # NMF doesn't give per-note confidence
                    }
        
        # Close any remaining active notes
        final_time = librosa.get_duration(y=y, sr=sr)
        for midi, note_info in active_notes.items():
            duration = final_time - note_info['start_time']
            if duration >= min_note_duration:
                all_notes.append(Note(
                    midi=midi,
                    start_time=note_info['start_time'],
                    duration=duration,
                    confidence=note_info['confidence']
                ))
        
    else:  # harmonic method
        print("Using harmonic peak detection...")
        pitches_per_frame = detect_pitches_harmonic(
            y, sr, hop_length,
            max_pitches=max_simultaneous,
            confidence_threshold=confidence_threshold
        )
        
        frame_times = librosa.frames_to_time(
            np.arange(len(pitches_per_frame)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Track active notes
        active_notes: Dict[int, dict] = {}
        all_notes = []
        
        for frame_idx, (frame_pitches, frame_time) in enumerate(zip(pitches_per_frame, frame_times)):
            current_midis = {p[0] for p in frame_pitches}
            pitch_confidences = {p[0]: p[1] for p in frame_pitches}
            
            # End notes that are no longer active
            ended_notes = [m for m in active_notes if m not in current_midis]
            for midi in ended_notes:
                note_info = active_notes.pop(midi)
                duration = frame_time - note_info['start_time']
                if duration >= min_note_duration:
                    all_notes.append(Note(
                        midi=midi,
                        start_time=note_info['start_time'],
                        duration=duration,
                        confidence=note_info['confidence']
                    ))
            
            # Start new notes or update confidence
            for midi in current_midis:
                if midi not in active_notes:
                    active_notes[midi] = {
                        'start_time': frame_time,
                        'start_frame': frame_idx,
                        'confidence': pitch_confidences.get(midi, 0.5)
                    }
        
        # Close remaining notes
        final_time = librosa.get_duration(y=y, sr=sr)
        for midi, note_info in active_notes.items():
            duration = final_time - note_info['start_time']
            if duration >= min_note_duration:
                all_notes.append(Note(
                    midi=midi,
                    start_time=note_info['start_time'],
                    duration=duration,
                    confidence=note_info['confidence']
                ))
    
    # Sort by start time
    all_notes.sort(key=lambda n: (n.start_time, n.midi))
    
    print(f"  Detected {len(all_notes)} notes (polyphonic)")
    
    # Analyze polyphony statistics
    from collections import Counter
    time_groups = {}
    for note in all_notes:
        # Quantize to 50ms buckets
        bucket = round(note.start_time / 0.05) * 0.05
        if bucket not in time_groups:
            time_groups[bucket] = []
        time_groups[bucket].append(note)
    
    polyphony_counts = Counter(len(notes) for notes in time_groups.values())
    print(f"  Polyphony distribution:")
    for count in sorted(polyphony_counts.keys()):
        if count > 1:
            print(f"    {count} simultaneous notes: {polyphony_counts[count]} times")
    
    # Apply octave correction
    if HAS_OCTAVE_CORRECTION and len(all_notes) > 0:
        print("Applying octave correction...")
        all_notes = _apply_octave_correction_to_notes(y, sr, all_notes, tuning)
    
    return all_notes


def notes_to_tabs(notes: List[Note], tuning: List[int] = None) -> List[TabNote]:
    """Convert detected notes to guitar tab positions."""
    if tuning is None:
        tuning = TUNINGS['standard']
    
    tab_notes = []
    prev_position = None
    
    for note in notes:
        tab_note = choose_best_fret_position(note, prev_position, tuning)
        if tab_note:
            tab_notes.append(tab_note)
            prev_position = (tab_note.string, tab_note.fret)
    
    return tab_notes


def notes_to_tabs_spectral(
    notes: List[Note], 
    audio_path: str = None,
    y: np.ndarray = None,
    sr: int = 22050,
    tuning: List[int] = None,
    verbose: bool = False
) -> List[TabNote]:
    """
    Convert detected notes to guitar tab positions using spectral string detection.
    
    This enhanced version uses timbre/spectral analysis to determine which string
    a note was most likely played on, rather than just using heuristics.
    
    Different guitar strings have characteristic timbral qualities:
    - Lower strings (E, A, D): warmer, more bass, lower spectral centroid
    - Higher strings (G, B, e): brighter, more treble, higher spectral centroid
    
    Args:
        notes: List of Note objects with midi, start_time, duration
        audio_path: Path to audio file (will be loaded if y not provided)
        y: Pre-loaded audio signal (optional)
        sr: Sample rate for y
        tuning: Guitar tuning as MIDI notes
        verbose: Print detection details
        
    Returns:
        List of TabNote objects with string/fret assignments
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    # Check if string detection is available
    if not HAS_STRING_DETECTION:
        print("⚠️  String detection module not available, using basic heuristics")
        return notes_to_tabs(notes, tuning)
    
    # Load audio if not provided
    if y is None:
        if audio_path is None:
            print("⚠️  No audio data provided for spectral analysis, using basic heuristics")
            return notes_to_tabs(notes, tuning)
        
        if verbose:
            print(f"Loading audio for spectral string detection: {audio_path}")
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Analyze audio brightness profile
    if verbose:
        profile = compute_string_brightness_profile(y, sr)
        print(f"📊 Audio brightness: {profile['brightness_category']} "
              f"(centroid: {profile['mean_centroid']:.0f} Hz)")
    
    # Create string detector
    detector = StringDetector(sr=sr, tuning=tuning)
    
    tab_notes = []
    prev_position = None
    
    confidence_sum = 0.0
    spectral_count = 0
    fallback_count = 0
    
    for i, note in enumerate(notes):
        # Try spectral detection
        pred = detector.predict_string(
            y,
            note.midi,
            note.start_time,
            note.duration,
            prev_position=prev_position,
            verbose=verbose
        )
        
        if pred.string >= 0 and pred.confidence >= 0.3:
            # Use spectral prediction
            tab_note = TabNote(
                string=pred.string,
                fret=pred.fret,
                start_time=note.start_time,
                duration=note.duration
            )
            confidence_sum += pred.confidence
            spectral_count += 1
            
            if verbose:
                alt_info = ""
                if pred.alternative_positions:
                    alts = [(STRING_NAMES[s], f) for s, f, _ in pred.alternative_positions[:2]]
                    alt_info = f" (alts: {alts})"
                print(f"  Note {i+1}: {note.name} -> {STRING_NAMES[pred.string]} fret {pred.fret} "
                      f"({pred.confidence:.0%}){alt_info}")
        else:
            # Fall back to heuristic method
            tab_note = choose_best_fret_position(note, prev_position, tuning)
            fallback_count += 1
            
            if verbose and tab_note:
                print(f"  Note {i+1}: {note.name} -> {STRING_NAMES[tab_note.string]} fret {tab_note.fret} "
                      f"(heuristic fallback)")
        
        if tab_note:
            tab_notes.append(tab_note)
            prev_position = (tab_note.string, tab_note.fret)
    
    # Print summary
    total = spectral_count + fallback_count
    if total > 0 and (verbose or spectral_count > 0):
        avg_conf = confidence_sum / spectral_count if spectral_count > 0 else 0
        print(f"\n🎸 String detection summary:")
        print(f"   Spectral detection: {spectral_count}/{total} notes ({100*spectral_count/total:.0f}%)")
        print(f"   Average confidence: {avg_conf:.0%}")
        if fallback_count > 0:
            print(f"   Heuristic fallback: {fallback_count} notes")
    
    return tab_notes


# ============================================================================
# CHORD DETECTION
# ============================================================================

def group_simultaneous_notes(
    notes: List[Note],
    time_threshold: float = CHORD_TIME_THRESHOLD
) -> List[List[Note]]:
    """
    Group notes that play simultaneously (within time_threshold).
    Returns list of note groups.
    """
    if not notes:
        return []
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    groups = []
    current_group = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        # Check if note is within threshold of first note in current group
        if note.start_time - current_group[0].start_time <= time_threshold:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    
    # Don't forget the last group
    if current_group:
        groups.append(current_group)
    
    return groups


def identify_chord_from_notes(notes: List[Note]) -> Optional[Chord]:
    """
    Identify a chord from a group of simultaneous notes.
    Returns None if no chord pattern matches.
    """
    if len(notes) < 2:
        return None
    
    # Get unique pitch classes
    pitch_classes = sorted(set(n.note_class for n in notes))
    
    if len(pitch_classes) < 2:
        return None
    
    # Calculate confidence based on number of notes
    avg_confidence = sum(n.confidence for n in notes) / len(notes)
    
    # Try each pitch class as potential root
    best_match = None
    best_score = 0
    
    for potential_root in pitch_classes:
        # Normalize intervals relative to this root
        intervals = sorted(set((pc - potential_root) % 12 for pc in pitch_classes))
        
        # Try to match against known chord types
        for chord_suffix, chord_intervals in CHORD_INTERVALS.items():
            # Calculate match score
            matched = sum(1 for i in intervals if i in chord_intervals)
            total = len(chord_intervals)
            
            # Score considers both coverage and accuracy
            if matched >= 2:  # Need at least 2 matching intervals
                score = (matched / total) * (matched / len(intervals))
                
                if score > best_score:
                    best_score = score
                    root_name = NOTE_NAMES[potential_root]
                    chord_name = root_name + chord_suffix if chord_suffix else root_name
                    
                    best_match = {
                        'name': chord_name,
                        'root': potential_root,
                        'quality': chord_suffix,
                        'score': score
                    }
    
    if best_match and best_score >= 0.5:  # Minimum match threshold
        start_time = min(n.start_time for n in notes)
        max_end = max(n.start_time + n.duration for n in notes)
        duration = max_end - start_time
        
        return Chord(
            name=best_match['name'],
            root=best_match['root'],
            notes=notes,
            start_time=start_time,
            duration=duration,
            quality=best_match['quality'],
            confidence=avg_confidence * best_score
        )
    
    return None


def detect_chord_shape(chord: Chord, tab_notes: List[TabNote]) -> Optional[str]:
    """
    Detect if a chord matches a known open or barre chord shape.
    Returns the shape name if found.
    """
    # Find tab notes that correspond to this chord's timing
    chord_tab_notes = [
        tn for tn in tab_notes
        if abs(tn.start_time - chord.start_time) <= CHORD_TIME_THRESHOLD
    ]
    
    if len(chord_tab_notes) < 2:
        return None
    
    # Create a fret pattern from the tab notes
    fret_pattern = {tn.string: tn.fret for tn in chord_tab_notes}
    
    # Check against known shapes
    for shape_name, shape_frets in OPEN_CHORD_SHAPES.items():
        # Skip if root doesn't match
        shape_root = shape_name.replace('m', '').replace('7', '').replace('sus', '').replace('dim', '')
        if shape_root and shape_root[0] != chord.root_name[0]:
            continue
        
        # Count matching frets
        matches = 0
        total_played = 0
        
        for string, expected_fret in shape_frets:
            if expected_fret == -1:  # Muted string
                continue
            total_played += 1
            if string in fret_pattern and fret_pattern[string] == expected_fret:
                matches += 1
        
        if total_played > 0 and matches / total_played >= 0.6:
            # Check if it's a barre chord
            is_barre = any(
                sum(1 for s, f in shape_frets if f == fret and f > 0) >= 2
                for fret in range(1, 13)
            )
            chord.is_barre = is_barre
            return shape_name
    
    return None


def detect_chords(
    notes: List[Note],
    time_threshold: float = CHORD_TIME_THRESHOLD
) -> List[Chord]:
    """
    Detect chords from a list of notes.
    Groups simultaneous notes and identifies chord patterns.
    """
    groups = group_simultaneous_notes(notes, time_threshold)
    chords = []
    
    for group in groups:
        if len(group) >= 2:  # Need at least 2 notes for a chord
            chord = identify_chord_from_notes(group)
            if chord:
                chords.append(chord)
    
    return chords


def generate_chord_diagram(chord_name: str) -> str:
    """
    Generate ASCII art chord diagram for a chord.
    Returns multi-line string with the diagram.
    """
    if chord_name not in OPEN_CHORD_SHAPES:
        # Try to find similar chord
        base_name = chord_name.replace('m', '').replace('7', '')
        if base_name + 'm' in OPEN_CHORD_SHAPES and 'm' in chord_name:
            chord_name = base_name + 'm'
        elif base_name in OPEN_CHORD_SHAPES:
            chord_name = base_name
        else:
            return f"  {chord_name}\n  (no diagram)"
    
    shape = OPEN_CHORD_SHAPES[chord_name]
    
    # Find fret range
    frets_played = [f for s, f in shape if f > 0]
    if not frets_played:
        min_fret, max_fret = 0, 3
    else:
        min_fret = min(frets_played)
        max_fret = max(frets_played)
    
    # Adjust for barre chords
    start_fret = 0 if max_fret <= 4 else min_fret - 1
    
    lines = [f"  {chord_name}"]
    
    # Top indicators (open/muted strings)
    top_line = "  "
    for string in range(6):
        string_fret = shape[string][1]
        if string_fret == -1:
            top_line += "x"
        elif string_fret == 0:
            top_line += "o"
        else:
            top_line += " "
    lines.append(top_line)
    
    # Nut or fret number
    if start_fret == 0:
        lines.append("  ╔═════╗")
    else:
        lines.append(f" {start_fret}├─────┤")
    
    # Draw frets
    for fret in range(start_fret + 1, start_fret + 5):
        fret_line = "  │"
        for string in range(6):
            string_fret = shape[string][1]
            if string_fret == fret:
                fret_line += "●"
            else:
                fret_line += "│"
        fret_line += "│"
        lines.append(fret_line)
        if fret < start_fret + 4:
            lines.append("  ├─────┤")
        else:
            lines.append("  └─────┘")
    
    # String names
    lines.append("  E A D G B e")
    
    return '\n'.join(lines)


def generate_all_chord_diagrams(chords: List[Chord]) -> str:
    """Generate chord diagrams for all unique chords."""
    unique_chords = list(set(c.name for c in chords))
    diagrams = []
    
    for chord_name in sorted(unique_chords):
        diagrams.append(generate_chord_diagram(chord_name))
    
    return '\n\n'.join(diagrams)


# ============================================================================
# TAB FORMATTING
# ============================================================================

def format_ascii_tab(
    tab_notes: List[TabNote],
    beats_per_line: int = 16,
    tuning: List[int] = None,
    chords: Optional[List[Chord]] = None
) -> str:
    """Format tab notes as ASCII guitar tablature with optional chord names above."""
    if not tab_notes:
        return "No notes detected!"
    
    if tuning is None:
        tuning = TUNINGS['standard']
    
    # Generate string names based on tuning
    string_names = []
    for midi in tuning:
        note_name = NOTE_NAMES[midi % 12]
        string_names.append(note_name)
    
    # Group notes by time (quantize to grid)
    time_resolution = 0.125  # 1/8 note at 120 BPM
    
    # Find total duration
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_positions = int(max_time / time_resolution) + 1
    
    # Create grid for each string
    grid = {i: ['-'] * num_positions for i in range(6)}
    
    # Create chord label grid
    chord_grid = [' '] * num_positions
    if chords:
        for chord in chords:
            pos = int(chord.start_time / time_resolution)
            if pos < num_positions:
                # Place chord name, handling overlaps
                chord_name = chord.name
                for i, char in enumerate(chord_name):
                    if pos + i < num_positions and chord_grid[pos + i] == ' ':
                        chord_grid[pos + i] = char
    
    for note in tab_notes:
        pos = int(note.start_time / time_resolution)
        if pos < num_positions:
            fret_str = str(note.fret) if note.fret < 10 else f"({note.fret})"
            grid[note.string][pos] = fret_str
    
    # Format output
    lines = []
    for start in range(0, num_positions, beats_per_line):
        end = min(start + beats_per_line, num_positions)
        
        # Add chord labels line if we have chords
        if chords:
            chord_line = ''.join(chord_grid[start:end])
            if chord_line.strip():  # Only add if there are chords
                lines.append(f"  {chord_line}")
        
        for string in range(5, -1, -1):  # High e to low E
            name = string_names[string] if string < len(string_names) else STRING_NAMES[string]
            notes_str = ''.join(grid[string][start:end])
            lines.append(f"{name}|{notes_str}|")
        lines.append("")  # Empty line between measures
    
    return '\n'.join(lines)


def download_youtube_audio(url: str, output_dir: str = None) -> str:
    """Download audio from YouTube URL using yt-dlp."""
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--print", "after_move:filepath",  # Print final path
        url
    ]
    
    print(f"📥 Downloading audio from YouTube...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    # Get the output file path from stdout
    output_path = result.stdout.strip().split('\n')[-1]
    print(f"✅ Downloaded: {output_path}")
    
    return output_path


def is_youtube_url(s: str) -> bool:
    """Check if string is a YouTube URL."""
    return any(domain in s for domain in ['youtube.com', 'youtu.be', 'youtube-nocookie.com'])


def export_guitar_pro(
    tab_notes: List[TabNote],
    output_path: str,
    title: str = "Generated Tab",
    artist: str = "Guitar Tab Generator",
    tempo: int = 120,
    tuning: List[int] = None
) -> bool:
    """
    Export tab notes to Guitar Pro format (GP5).
    
    Args:
        tab_notes: List of TabNote objects
        output_path: Path to save the GP5 file
        title: Song title
        artist: Artist name
        tempo: Tempo in BPM
        tuning: Guitar tuning as MIDI note numbers
        
    Returns:
        True if successful, False otherwise
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    if not HAS_GUITARPRO:
        print("❌ pyguitarpro not installed. Install with: pip install pyguitarpro")
        return False
    
    if not tab_notes:
        print("❌ No notes to export")
        return False
    
    # Create song
    song = guitarpro.Song()
    song.title = title
    song.artist = artist
    song.tempo = tempo  # Simple integer for BPM
    
    # Create track
    track = guitarpro.Track(song=song)
    track.name = "Guitar"
    track.fretCount = NUM_FRETS
    
    # Set up guitar strings (standard tuning)
    track.strings = []
    for i, midi_note in enumerate(reversed(tuning)):  # GP uses high to low
        string = guitarpro.GuitarString(number=i + 1, value=midi_note)
        track.strings.append(string)
    
    # Time resolution - convert time to beats
    time_per_beat = 60.0 / tempo  # seconds per beat
    time_per_measure = time_per_beat * 4  # 4/4 time
    
    # Group notes into measures
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_measures = int(max_time / time_per_measure) + 1
    
    # Create measure headers
    for i in range(num_measures):
        header = guitarpro.MeasureHeader()
        header.number = i + 1
        header.start = int(i * 960 * 4)  # 960 ticks per quarter note
        header.timeSignature = guitarpro.TimeSignature()
        header.timeSignature.numerator = 4
        header.timeSignature.denominator = guitarpro.Duration()
        header.timeSignature.denominator.value = 4
        song.measureHeaders.append(header)
    
    # Create measures for the track
    for header in song.measureHeaders:
        measure = guitarpro.Measure(track=track, header=header)
        
        # Each measure has voices (we use voice 0)
        voice = measure.voices[0]
        
        # Find notes in this measure
        measure_start = (header.number - 1) * time_per_measure
        measure_end = measure_start + time_per_measure
        
        measure_notes = [n for n in tab_notes 
                        if measure_start <= n.start_time < measure_end]
        
        if not measure_notes:
            # Add a rest beat if no notes
            beat = guitarpro.Beat(voice)
            beat.status = guitarpro.BeatStatus.rest
            beat.duration = guitarpro.Duration()
            beat.duration.value = 1  # Whole note rest
            voice.beats.append(beat)
        else:
            # Group notes by time (for chords)
            note_groups = {}
            for tab_note in measure_notes:
                # Quantize to 16th notes
                beat_time = round((tab_note.start_time - measure_start) / (time_per_beat / 4)) * (time_per_beat / 4)
                if beat_time not in note_groups:
                    note_groups[beat_time] = []
                note_groups[beat_time].append(tab_note)
            
            # Create beats for each time position
            for beat_time in sorted(note_groups.keys()):
                beat = guitarpro.Beat(voice)
                
                # Determine duration based on note duration
                avg_duration = sum(n.duration for n in note_groups[beat_time]) / len(note_groups[beat_time])
                beat.duration = guitarpro.Duration()
                
                # Map duration to note value
                if avg_duration >= time_per_beat * 2:
                    beat.duration.value = 2  # Half note
                elif avg_duration >= time_per_beat:
                    beat.duration.value = 4  # Quarter note
                elif avg_duration >= time_per_beat / 2:
                    beat.duration.value = 8  # Eighth note
                else:
                    beat.duration.value = 16  # Sixteenth note
                
                # Add notes
                for tab_note in note_groups[beat_time]:
                    note = guitarpro.Note(beat)
                    # Guitar Pro uses 1-indexed strings, high to low
                    note.string = 6 - tab_note.string  # Convert from our 0-indexed low-to-high
                    note.value = tab_note.fret
                    note.velocity = 95  # Default velocity
                    beat.notes.append(note)
                
                voice.beats.append(beat)
        
        track.measures.append(measure)
    
    song.tracks.append(track)
    
    # Write file
    try:
        guitarpro.write(song, output_path, version=(5, 1, 0))
        print(f"✅ Exported Guitar Pro file: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write GP5 file: {e}")
        return False


def export_musicxml(
    tab_notes: List[TabNote],
    output_path: str,
    title: str = "Generated Tab",
    composer: str = "Guitar Tab Generator",
    tempo: int = 120,
    tuning: List[int] = None
) -> bool:
    """
    Export tab notes to MusicXML format.
    
    MusicXML is a universal format supported by:
    - Guitar Pro
    - MuseScore
    - Finale
    - Sibelius
    - Many other music notation apps
    
    Args:
        tab_notes: List of TabNote objects
        output_path: Path to save the MusicXML file
        title: Song title
        composer: Composer/artist name
        tempo: Tempo in BPM
        tuning: Guitar tuning as MIDI note numbers
        
    Returns:
        True if successful, False otherwise
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    if not tab_notes:
        print("❌ No notes to export")
        return False
    
    # MIDI note to pitch mapping
    def midi_to_pitch(midi: int) -> Tuple[str, int, int]:
        """Convert MIDI note to (step, alter, octave)"""
        note_steps = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
        note_alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        pitch_class = midi % 12
        octave = (midi // 12) - 1
        
        return note_steps[pitch_class], note_alters[pitch_class], octave
    
    # Create root element
    root = ET.Element('score-partwise', version='4.0')
    
    # Work info
    work = ET.SubElement(root, 'work')
    work_title = ET.SubElement(work, 'work-title')
    work_title.text = title
    
    # Identification
    identification = ET.SubElement(root, 'identification')
    creator = ET.SubElement(identification, 'creator', type='composer')
    creator.text = composer
    encoding = ET.SubElement(identification, 'encoding')
    software = ET.SubElement(encoding, 'software')
    software.text = 'Guitar Tab Generator'
    
    # Part list
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    part_name = ET.SubElement(score_part, 'part-name')
    part_name.text = 'Guitar'
    
    # Score instrument for tablature
    score_inst = ET.SubElement(score_part, 'score-instrument', id='P1-I1')
    inst_name = ET.SubElement(score_inst, 'instrument-name')
    inst_name.text = 'Acoustic Guitar'
    
    # Part content
    part = ET.SubElement(root, 'part', id='P1')
    
    # Time and beat calculations
    time_per_beat = 60.0 / tempo
    time_per_measure = time_per_beat * 4  # 4/4 time
    divisions = 4  # Divisions per quarter note (allows 16th notes)
    
    # Group notes by measure
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_measures = int(max_time / time_per_measure) + 1
    
    for measure_num in range(1, num_measures + 1):
        measure = ET.SubElement(part, 'measure', number=str(measure_num))
        
        # Attributes for first measure
        if measure_num == 1:
            attributes = ET.SubElement(measure, 'attributes')
            
            div_elem = ET.SubElement(attributes, 'divisions')
            div_elem.text = str(divisions)
            
            # Key signature (C major)
            key = ET.SubElement(attributes, 'key')
            fifths = ET.SubElement(key, 'fifths')
            fifths.text = '0'
            
            # Time signature (4/4)
            time = ET.SubElement(attributes, 'time')
            beats = ET.SubElement(time, 'beats')
            beats.text = '4'
            beat_type = ET.SubElement(time, 'beat-type')
            beat_type.text = '4'
            
            # Clef (TAB)
            clef = ET.SubElement(attributes, 'clef')
            sign = ET.SubElement(clef, 'sign')
            sign.text = 'TAB'
            line = ET.SubElement(clef, 'line')
            line.text = '5'
            
            # Staff details (6 string guitar)
            staff_details = ET.SubElement(attributes, 'staff-details')
            staff_lines = ET.SubElement(staff_details, 'staff-lines')
            staff_lines.text = '6'
            
            # Tuning
            for string_num, midi_note in enumerate(reversed(tuning), 1):
                staff_tuning = ET.SubElement(staff_details, 'staff-tuning', line=str(string_num))
                step, alter, octave = midi_to_pitch(midi_note)
                tuning_step = ET.SubElement(staff_tuning, 'tuning-step')
                tuning_step.text = step
                if alter:
                    tuning_alter = ET.SubElement(staff_tuning, 'tuning-alter')
                    tuning_alter.text = str(alter)
                tuning_octave = ET.SubElement(staff_tuning, 'tuning-octave')
                tuning_octave.text = str(octave)
            
            # Tempo direction
            direction = ET.SubElement(measure, 'direction', placement='above')
            direction_type = ET.SubElement(direction, 'direction-type')
            metronome = ET.SubElement(direction_type, 'metronome')
            beat_unit = ET.SubElement(metronome, 'beat-unit')
            beat_unit.text = 'quarter'
            per_minute = ET.SubElement(metronome, 'per-minute')
            per_minute.text = str(tempo)
            sound = ET.SubElement(direction, 'sound', tempo=str(tempo))
        
        # Find notes in this measure
        measure_start = (measure_num - 1) * time_per_measure
        measure_end = measure_start + time_per_measure
        
        measure_notes = [n for n in tab_notes 
                        if measure_start <= n.start_time < measure_end]
        
        if not measure_notes:
            # Add whole rest
            note_elem = ET.SubElement(measure, 'note')
            rest = ET.SubElement(note_elem, 'rest')
            duration = ET.SubElement(note_elem, 'duration')
            duration.text = str(divisions * 4)  # Whole note
            ntype = ET.SubElement(note_elem, 'type')
            ntype.text = 'whole'
        else:
            # Group notes by time (for chords)
            note_groups = {}
            for tab_note in measure_notes:
                # Quantize to 16th notes
                quantized = round((tab_note.start_time - measure_start) / (time_per_beat / 4))
                if quantized not in note_groups:
                    note_groups[quantized] = []
                note_groups[quantized].append(tab_note)
            
            # Track position for rests
            current_position = 0
            
            for quantized_pos in sorted(note_groups.keys()):
                # Add rest if there's a gap
                if quantized_pos > current_position:
                    gap_duration = quantized_pos - current_position
                    note_elem = ET.SubElement(measure, 'note')
                    rest = ET.SubElement(note_elem, 'rest')
                    dur = ET.SubElement(note_elem, 'duration')
                    dur.text = str(gap_duration)
                
                # Add notes at this position
                notes_at_pos = note_groups[quantized_pos]
                is_chord = len(notes_at_pos) > 1
                
                for i, tab_note in enumerate(notes_at_pos):
                    note_elem = ET.SubElement(measure, 'note')
                    
                    # Chord indication for simultaneous notes
                    if is_chord and i > 0:
                        ET.SubElement(note_elem, 'chord')
                    
                    # Calculate MIDI pitch from string and fret
                    midi_pitch = tuning[tab_note.string] + tab_note.fret
                    step, alter, octave = midi_to_pitch(midi_pitch)
                    
                    pitch = ET.SubElement(note_elem, 'pitch')
                    step_elem = ET.SubElement(pitch, 'step')
                    step_elem.text = step
                    if alter:
                        alter_elem = ET.SubElement(pitch, 'alter')
                        alter_elem.text = str(alter)
                    octave_elem = ET.SubElement(pitch, 'octave')
                    octave_elem.text = str(octave)
                    
                    # Duration
                    dur_beats = max(1, round(tab_note.duration / time_per_beat * divisions))
                    duration_elem = ET.SubElement(note_elem, 'duration')
                    duration_elem.text = str(min(dur_beats, divisions * 4))  # Cap at whole note
                    
                    # Note type
                    ntype = ET.SubElement(note_elem, 'type')
                    if dur_beats >= divisions * 4:
                        ntype.text = 'whole'
                    elif dur_beats >= divisions * 2:
                        ntype.text = 'half'
                    elif dur_beats >= divisions:
                        ntype.text = 'quarter'
                    elif dur_beats >= divisions // 2:
                        ntype.text = 'eighth'
                    else:
                        ntype.text = '16th'
                    
                    # Notations with technical (fret/string)
                    notations = ET.SubElement(note_elem, 'notations')
                    technical = ET.SubElement(notations, 'technical')
                    
                    string_elem = ET.SubElement(technical, 'string')
                    string_elem.text = str(6 - tab_note.string)  # MusicXML: 1 = high e
                    
                    fret_elem = ET.SubElement(technical, 'fret')
                    fret_elem.text = str(tab_note.fret)
                
                # Update position
                current_position = quantized_pos + dur_beats
    
    # Format XML nicely
    xml_str = ET.tostring(root, encoding='unicode')
    
    # Add XML declaration and DOCTYPE
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    
    try:
        # Pretty print
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        # Remove extra declaration added by minidom
        pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doctype)
            f.write(pretty_xml)
        
        print(f"✅ Exported MusicXML file: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write MusicXML file: {e}")
        return False


def export_lilypond(
    tab_notes: List[TabNote],
    notes: List[Note],
    output_path: str,
    title: str = "Generated Tab",
    composer: str = "Guitar Tab Generator",
    tempo: int = 120,
    tuning: List[int] = None,
    chords: Optional[List['Chord']] = None,
    include_standard_notation: bool = True
) -> bool:
    """
    Export tab notes to LilyPond format (.ly) for beautiful PDF generation.
    
    LilyPond is a free music engraving program that produces publication-quality
    sheet music. After exporting, run: lilypond filename.ly
    
    Features:
    - Guitar tablature with fret numbers
    - Standard notation (optional)
    - Technique markings (hammer-ons, pull-offs, slides, bends)
    - Chord symbols above the staff
    - Custom tuning support
    - Proper rhythmic notation
    
    Args:
        tab_notes: List of TabNote objects
        notes: List of Note objects (for MIDI pitch info)
        output_path: Path to save the .ly file
        title: Song title
        composer: Composer/artist name
        tempo: Tempo in BPM
        tuning: Guitar tuning as MIDI note numbers
        chords: Optional list of detected chords
        include_standard_notation: Include standard notation staff above tab
        
    Returns:
        True if successful, False otherwise
    """
    if tuning is None:
        tuning = TUNINGS['standard']
    
    if not tab_notes:
        print("❌ No notes to export")
        return False
    
    # LilyPond note name mapping (MIDI pitch class to LilyPond name)
    LILYPOND_NOTES = ['c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b']
    
    def midi_to_lilypond_pitch(midi: int) -> str:
        """Convert MIDI note number to LilyPond pitch notation."""
        pitch_class = midi % 12
        octave = (midi // 12) - 1  # MIDI octave to LilyPond octave
        
        note_name = LILYPOND_NOTES[pitch_class]
        
        # LilyPond uses ' for octaves above middle C (c' = middle C = MIDI 60)
        # and , for octaves below
        # c' = MIDI 60, c'' = MIDI 72, c = MIDI 48, c, = MIDI 36
        lily_octave = octave - 3  # Adjust: MIDI octave 4 (middle C) = LilyPond c'
        
        if lily_octave > 0:
            note_name += "'" * lily_octave
        elif lily_octave < 0:
            note_name += "," * abs(lily_octave)
        
        return note_name
    
    def duration_to_lilypond(duration: float, tempo: int) -> str:
        """Convert duration in seconds to LilyPond duration notation."""
        # Calculate duration in beats
        beat_duration = 60.0 / tempo
        beats = duration / beat_duration
        
        # Map to LilyPond durations (1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth)
        if beats >= 3.0:
            return "1"  # Whole note
        elif beats >= 1.5:
            return "2"  # Half note
        elif beats >= 0.75:
            return "4"  # Quarter note
        elif beats >= 0.375:
            return "8"  # Eighth note
        elif beats >= 0.1875:
            return "16"  # Sixteenth note
        else:
            return "32"  # Thirty-second note
    
    def rest_duration_to_lilypond(gap_seconds: float, tempo: int) -> List[str]:
        """Convert a gap to one or more LilyPond rest durations."""
        beat_duration = 60.0 / tempo
        beats = gap_seconds / beat_duration
        
        rests = []
        # Break down into standard note values
        while beats >= 0.125:  # 32nd note minimum
            if beats >= 4.0:
                rests.append("r1")
                beats -= 4.0
            elif beats >= 2.0:
                rests.append("r2")
                beats -= 2.0
            elif beats >= 1.0:
                rests.append("r4")
                beats -= 1.0
            elif beats >= 0.5:
                rests.append("r8")
                beats -= 0.5
            elif beats >= 0.25:
                rests.append("r16")
                beats -= 0.25
            else:
                rests.append("r32")
                beats -= 0.125
        
        return rests if rests else ["r16"]
    
    def detect_technique(idx: int, tab_notes: List[TabNote], notes: List[Note]) -> str:
        """Detect technique between consecutive notes."""
        if idx == 0 or idx >= len(notes):
            return ""
        
        tab_note = tab_notes[idx]
        prev_tab = tab_notes[idx - 1]
        note = notes[idx]
        prev_note = notes[idx - 1]
        
        time_gap = tab_note.start_time - (prev_tab.start_time + prev_tab.duration)
        pitch_diff = note.midi - prev_note.midi
        same_string = tab_note.string == prev_tab.string
        
        # Hammer-on: ascending pitch on same string with minimal gap
        if same_string and pitch_diff > 0 and time_gap < 0.1:
            return "^\\markup{H}"
        
        # Pull-off: descending pitch on same string with minimal gap
        if same_string and pitch_diff < 0 and time_gap < 0.1:
            return "^\\markup{P}"
        
        # Slide: larger pitch change on same string
        if same_string and abs(pitch_diff) >= 3 and time_gap < 0.08:
            if pitch_diff > 0:
                return "\\glissando"  # Slide up
            else:
                return "\\glissando"  # Slide down
        
        return ""
    
    # Build tuning string for LilyPond (standard guitar tuning format)
    tuning_notes = []
    for midi in reversed(tuning):  # LilyPond: high to low (string 1 to 6)
        tuning_notes.append(midi_to_lilypond_pitch(midi))
    tuning_str = " ".join(tuning_notes)
    
    # Group notes by time for chord handling (50ms tolerance)
    CHORD_TOLERANCE = 0.05  # 50ms
    note_groups = []
    current_group = []
    
    for i, tab_note in enumerate(tab_notes):
        if i < len(notes):
            if not current_group:
                current_group = [(tab_note, notes[i], i)]
            else:
                # Check if this note is simultaneous with the group
                group_time = current_group[0][0].start_time
                if abs(tab_note.start_time - group_time) <= CHORD_TOLERANCE:
                    current_group.append((tab_note, notes[i], i))
                else:
                    note_groups.append(current_group)
                    current_group = [(tab_note, notes[i], i)]
    
    if current_group:
        note_groups.append(current_group)
    
    # Build music content
    music_content = []
    prev_end_time = 0
    
    for group in note_groups:
        group_time = group[0][0].start_time
        
        # Add rests for gaps
        gap = group_time - prev_end_time
        if gap > 0.1:  # Significant gap (>100ms)
            rests = rest_duration_to_lilypond(gap, tempo)
            music_content.extend(rests)
        
        # Remove duplicate strings in chord (keep lowest fret per string)
        seen_strings = {}
        unique_notes = []
        for tab_note, orig_note, idx in group:
            if tab_note.string not in seen_strings:
                seen_strings[tab_note.string] = (tab_note, orig_note, idx)
                unique_notes.append((tab_note, orig_note, idx))
        
        if len(unique_notes) == 1:
            # Single note
            tab_note, orig_note, idx = unique_notes[0]
            midi_pitch = tuning[tab_note.string] + tab_note.fret
            lily_pitch = midi_to_lilypond_pitch(midi_pitch)
            duration = duration_to_lilypond(tab_note.duration, tempo)
            
            # Guitar string number (1=high e, 6=low E)
            string_num = 6 - tab_note.string
            
            # Technique marking
            technique = detect_technique(idx, tab_notes, notes)
            
            note_str = f"{lily_pitch}{duration}\\{string_num}"
            if technique:
                note_str += f" {technique}"
            
            music_content.append(note_str)
        else:
            # Chord (multiple notes)
            chord_notes = []
            for tab_note, orig_note, idx in sorted(unique_notes, key=lambda x: x[0].string):
                midi_pitch = tuning[tab_note.string] + tab_note.fret
                lily_pitch = midi_to_lilypond_pitch(midi_pitch)
                string_num = 6 - tab_note.string
                chord_notes.append(f"{lily_pitch}\\{string_num}")
            
            duration = duration_to_lilypond(unique_notes[0][0].duration, tempo)
            chord_str = f"<{' '.join(chord_notes)}>{duration}"
            music_content.append(chord_str)
        
        # Update end time
        max_duration = max(tn[0].duration for tn in unique_notes)
        prev_end_time = group_time + max_duration
    
    # Build chord symbols if available
    chord_content = ""
    if chords:
        chord_symbols = []
        prev_chord_end = 0
        
        for chord in chords:
            # Add spacing rests if needed
            gap = chord.start_time - prev_chord_end
            if gap > 0.1:
                rest_dur = duration_to_lilypond(gap, tempo)
                chord_symbols.append(f"s{rest_dur}")  # spacer rest
            
            # LilyPond chord naming
            root = NOTE_NAMES[chord.root].lower().replace('#', 'is').replace('b', 'es')
            quality = chord.quality
            
            lily_chord = root
            if quality == 'm' or quality == 'min':
                lily_chord += ":m"
            elif quality == '7':
                lily_chord += ":7"
            elif quality == 'maj7':
                lily_chord += ":maj7"
            elif quality == 'm7':
                lily_chord += ":m7"
            elif quality == 'dim':
                lily_chord += ":dim"
            elif quality == 'aug':
                lily_chord += ":aug"
            elif quality == 'sus4':
                lily_chord += ":sus4"
            elif quality == 'sus2':
                lily_chord += ":sus2"
            
            duration = duration_to_lilypond(chord.duration, tempo)
            chord_symbols.append(f"{lily_chord}{duration}")
            prev_chord_end = chord.start_time + chord.duration
        
        if chord_symbols:
            chord_content = " ".join(chord_symbols)
    
    # Escape title for LilyPond
    safe_title = title.replace('"', '\\"').replace('\\', '\\\\')
    safe_composer = composer.replace('"', '\\"').replace('\\', '\\\\')
    
    # Generate the complete LilyPond file
    lily_content = f'''\\version "2.24.0"

\\header {{
  title = "{safe_title}"
  composer = "{safe_composer}"
  tagline = "Generated by Guitar Tab Generator"
}}

% Guitar tuning (high to low: e' b g d a, e,)
#(define guitar-tuning '({tuning_str}))

% The music
music = \\relative {{
  \\tempo 4 = {tempo}
  \\time 4/4
  {" ".join(music_content)}
  \\bar "|."
}}

'''

    # Add chord symbols section if chords exist
    if chord_content:
        lily_content += f'''% Chord symbols
chordNames = \\chordmode {{
  {chord_content}
}}

'''

    # Build the score
    if include_standard_notation:
        lily_content += '''% Score with standard notation and tablature
\\score {
  <<
'''
        if chord_content:
            lily_content += '''    \\new ChordNames { \\chordNames }
'''
        lily_content += f'''    \\new Staff {{
      \\clef "treble_8"
      \\key c \\major
      \\music
    }}
    \\new TabStaff {{
      \\set TabStaff.stringTunings = #guitar-tuning
      \\music
    }}
  >>
  \\layout {{
    \\context {{
      \\TabStaff
      \\override TabNoteHead.font-size = #-2
      \\override Stem.transparent = ##t
      \\override Beam.transparent = ##t
    }}
  }}
  \\midi {{ \\tempo 4 = {tempo} }}
}}
'''
    else:
        # Tab only
        lily_content += f'''% Score with tablature only
\\score {{
  <<
'''
        if chord_content:
            lily_content += '''    \\new ChordNames { \\chordNames }
'''
        lily_content += f'''    \\new TabStaff {{
      \\set TabStaff.stringTunings = #guitar-tuning
      \\music
    }}
  >>
  \\layout {{
    \\context {{
      \\TabStaff
      \\override TabNoteHead.font-size = #-2
    }}
  }}
  \\midi {{ \\tempo 4 = {tempo} }}
}}
'''
    
    # Write file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(lily_content)
        
        print(f"✅ Exported LilyPond file: {output_path}")
        print(f"   To generate PDF: lilypond {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write LilyPond file: {e}")
        return False


def get_export_extension(format_name: str) -> str:
    """Get the appropriate file extension for an export format."""
    extensions = {
        'gp5': '.gp5',
        'gp': '.gp5',
        'guitarpro': '.gp5',
        'musicxml': '.musicxml',
        'xml': '.musicxml',
        'ascii': '.txt',
        'txt': '.txt',
        'lilypond': '.ly',
        'ly': '.ly',
    }
    return extensions.get(format_name.lower(), '.txt')


def main():
    parser = argparse.ArgumentParser(
        description='Generate guitar tabs from audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tunings:
  standard       - E A D G B E (default)
  drop_d         - D A D G B E
  drop_c         - C G C F A D
  half_step_down - Eb Ab Db Gb Bb Eb
  full_step_down - D G C F A D
  open_d         - D A D F# A D
  open_g         - D G D G B D
  dadgad         - D A D G A D

Or specify custom tuning as comma-separated MIDI notes:
  --tuning "38,45,50,55,59,64"

Export formats:
  ascii     - Plain text tablature (default)
  gp5       - Guitar Pro 5 format (.gp5)
  musicxml  - MusicXML format (.musicxml) - universal format
  lilypond  - LilyPond format (.ly) - run 'lilypond file.ly' for beautiful PDFs

Polyphonic detection (--polyphonic):
  nmf       - Non-negative Matrix Factorization on CQT (default)
              Best for clean recordings with multiple simultaneous notes
  harmonic  - Harmonic/percussive separation + peak picking
              Good for noisy recordings or complex arrangements

Musical post-processing:
  --key Am       - Specify key (e.g., Am, C, F#m, "Bb major")
  --quantize 16  - Snap timing to 1/16 notes (4, 8, 16, or 32)
  --scale blues  - Use blues scale for pitch snapping
  --no-snap      - Disable pitch snapping to scale
  --detect-patterns - Find repeated riffs/patterns

Fingering optimization (--optimize-fingering):
  Uses dynamic programming to find optimal fret/string assignments:
  - Minimizes hand position changes and fret jumps
  - Prefers open strings and lower frets where appropriate
  - Considers technique requirements (bends easier on certain strings)
  - Handles chords with playable finger spans
  - Look-ahead optimization for upcoming notes

Examples:
  %(prog)s song.mp3
  %(prog)s song.mp3 -o tabs.txt
  %(prog)s song.mp3 -o tabs.gp5 --format gp5
  %(prog)s "https://youtube.com/watch?v=..." --format musicxml -o song.musicxml
  
  # Polyphonic detection for chords and fingerstyle
  %(prog)s song.mp3 --polyphonic
  %(prog)s song.mp3 --polyphonic --poly-method harmonic
  %(prog)s song.mp3 --polyphonic --max-simultaneous 4  # Max 4 notes at once
  
  # Fingering optimization
  %(prog)s song.mp3 --optimize-fingering                     # Enable optimization
  %(prog)s song.mp3 --optimize-fingering --fingering-max-span 5  # Allow wider stretch
  %(prog)s song.mp3 --optimize-fingering --fingering-open-string-bonus 2.0  # Prefer open strings more
  %(prog)s song.mp3 -t drop_d --optimize-fingering           # Works with alternate tunings
  
  # Musical post-processing
  %(prog)s song.mp3 --key Am --quantize 16            # Snap to Am scale, quantize to 16ths
  %(prog)s song.mp3 --scale pentatonic_minor          # Auto-detect key, use pentatonic
  %(prog)s song.mp3 --quantize 8 --tempo 140          # Quantize to 8ths at 140 BPM
  %(prog)s song.mp3 --detect-patterns --chords        # Find riffs and chords
  
  # Web player sync export (follow-along playback)
  %(prog)s song.mp3 --export-sync song.sync.json      # Generate timeline for web player
  %(prog)s song.mp3 --export-sync sync.json --chords  # Include chord events in sync data
        """
    )
    parser.add_argument('audio_file', help='Path to audio file OR YouTube URL')
    parser.add_argument('--output', '-o', help='Output file for tabs')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                        help='Minimum confidence threshold (0-1, default: 0.3)')
    parser.add_argument('--tuning', '-t', default='standard',
                        help='Guitar tuning (see list below, default: standard)')
    parser.add_argument('--format', '-f', 
                        choices=['ascii', 'gp5', 'gp', 'guitarpro', 'musicxml', 'xml', 'lilypond', 'ly'],
                        default='ascii',
                        help='Output format: ascii (default), gp5/guitarpro, musicxml/xml, lilypond/ly')
    parser.add_argument('--title', default=None,
                        help='Song title (for GP5/MusicXML export)')
    parser.add_argument('--artist', '-a', default='Guitar Tab Generator',
                        help='Artist name (for GP5/MusicXML export)')
    parser.add_argument('--tempo', type=int, default=120,
                        help='Tempo in BPM (default: 120)')
    parser.add_argument('--pitch-method', '-p', choices=['segment', 'ensemble', 'voting', 'pyin', 'piptrack', 'cqt', 'crepe', 'basicpitch', 'hps'], default='segment',
                        help='Pitch detection: segment (BEST - precise note boundaries), ensemble (5 detector consensus), voting (legacy consensus), pyin (fast), piptrack, cqt, crepe (deep learning), basicpitch (polyphonic), hps (Harmonic Product Spectrum - BEST FOR DISTORTED GUITAR). Default: segment')
    parser.add_argument('--crepe-model', choices=['tiny', 'small', 'medium', 'large', 'full'], default='small',
                        help='CREPE model capacity: tiny/small (fast), medium, large/full (accurate). Default: small')
    parser.add_argument('--distortion', action='store_true',
                        help='Enable distortion-aware pitch detection (for heavy/distorted electric guitar). '
                             'Uses Harmonic Product Spectrum to find true fundamental when harmonics are strong.')
    parser.add_argument('--distortion-harmonics', type=int, default=5,
                        help='Number of harmonics for HPS in distortion mode (default: 5)')
    parser.add_argument('--min-votes', type=int, default=2,
                        help='Minimum detector votes for consensus (voting method). Default: 2')
    parser.add_argument('--no-harmonic-separation', action='store_true',
                        help='Disable harmonic/percussive separation')
    parser.add_argument('--no-octave-correction', action='store_true',
                        help='Disable octave correction using harmonic analysis')
    parser.add_argument('--median-filter', '-m', type=int, default=5,
                        help='Median filter size for pitch smoothing (0=disabled, default: 5)')
    parser.add_argument('--min-duration', type=float, default=0.05,
                        help='Minimum note duration in seconds (default: 0.05)')
    parser.add_argument('--chords', action='store_true',
                        help='Enable chord detection')
    parser.add_argument('--chord-diagrams', action='store_true',
                        help='Show ASCII chord diagrams')
    parser.add_argument('--chord-threshold', type=float, default=CHORD_TIME_THRESHOLD,
                        help=f'Time window for simultaneous notes in seconds (default: {CHORD_TIME_THRESHOLD})')
    parser.add_argument('--polyphonic', action='store_true',
                        help='Enable polyphonic detection (multiple simultaneous notes using NMF)')
    parser.add_argument('--poly-method', choices=['nmf', 'harmonic'], default='nmf',
                        help='Polyphonic detection method: nmf (NMF on CQT) or harmonic (HPSS + peaks)')
    parser.add_argument('--max-simultaneous', type=int, default=6,
                        help='Maximum simultaneous notes for polyphonic detection (default: 6)')
    
    # Musical post-processing arguments
    parser.add_argument('--quantize', '-q', type=int, choices=[4, 8, 16, 32], default=None,
                        help='Quantize timing to note subdivision (4=quarter, 8=eighth, 16=sixteenth, 32=32nd)')
    parser.add_argument('--key', '-k', type=str, default=None,
                        help='Musical key for pitch snapping (e.g., Am, C, F#m, "Bb major"). Auto-detects if not specified.')
    parser.add_argument('--scale', type=str, default=None,
                        choices=['major', 'natural_minor', 'harmonic_minor', 'pentatonic_major', 
                                'pentatonic_minor', 'blues', 'dorian', 'mixolydian', 'chromatic'],
                        help='Scale type for pitch snapping (default: based on key mode)')
    parser.add_argument('--no-snap', action='store_true',
                        help='Disable pitch snapping to scale')
    parser.add_argument('--no-playability-filter', action='store_true',
                        help='Disable physical playability filtering')
    
    # Scale-constrained detection arguments
    parser.add_argument('--scale-constrain', action='store_true',
                        help='Enable scale-constrained detection: auto-detect key and snap ALL '
                             'detected pitches to the nearest valid scale note. This eliminates '
                             'random wrong notes that are not in the key. More aggressive than --no-snap.')
    parser.add_argument('--scale-constrain-type', type=str, default=None,
                        choices=['major', 'natural_minor', 'harmonic_minor', 'melodic_minor',
                                 'pentatonic_major', 'pentatonic_minor', 'blues',
                                 'dorian', 'phrygian', 'lydian', 'mixolydian'],
                        help='Scale type for constraint (default: auto from detected key mode)')
    parser.add_argument('--scale-snap-direction', type=str, default='nearest',
                        choices=['nearest', 'down', 'up'],
                        help='Direction to snap out-of-scale notes (default: nearest)')
    parser.add_argument('--allow-passing-tones', action='store_true',
                        help='Allow short chromatic passing tones (notes < 100ms) when using --scale-constrain')
    parser.add_argument('--passing-tone-max-ms', type=float, default=100,
                        help='Maximum duration (ms) for passing tones (default: 100)')
    parser.add_argument('--detect-patterns', action='store_true',
                        help='Detect repeated patterns/riffs in the transcription')
    parser.add_argument('--swing', type=float, default=0.0,
                        help='Swing amount for quantization (0.0=straight, 0.33=triplet, 0.5=heavy)')
    
    # Fingering optimization arguments
    parser.add_argument('--optimize-fingering', action='store_true',
                        help='Enable advanced fingering optimization (considers hand position, stretches, technique)')
    parser.add_argument('--fingering-low-fret-bonus', type=float, default=0.5,
                        help='Bonus for lower frets in fingering optimization (default: 0.5)')
    parser.add_argument('--fingering-open-string-bonus', type=float, default=1.5,
                        help='Bonus for open strings in fingering optimization (default: 1.5)')
    parser.add_argument('--fingering-max-span', type=int, default=4,
                        help='Maximum fret span for fingering (default: 4)')
    
    # String detection arguments
    parser.add_argument('--spectral-strings', action='store_true',
                        help='Use spectral/timbre analysis to detect which string notes are played on')
    parser.add_argument('--string-detection-verbose', action='store_true',
                        help='Show detailed string detection analysis')
    
    # Rhythm notation arguments
    parser.add_argument('--rhythm', '-r', action='store_true',
                        help='Enable rhythm notation (shows note durations above tab)')
    parser.add_argument('--detect-tempo', action='store_true',
                        help='Auto-detect tempo from audio (overrides --tempo if faster/more accurate)')
    parser.add_argument('--show-rests', action='store_true',
                        help='Show rest symbols in rhythm notation')
    parser.add_argument('--unicode-rhythm', action='store_true',
                        help='Use Unicode music symbols for rhythm (♩♪♬ instead of q e s)')
    
    # Sync export for web player integration
    parser.add_argument('--export-sync', type=str, default=None, metavar='PATH',
                        help='Export JSON sync data for web player "follow along" playback (e.g., output.sync.json)')
    
    # Difficulty analysis arguments
    parser.add_argument('--analyze-difficulty', action='store_true',
                        help='Analyze and report tab difficulty (rating, techniques, stretches, speed)')
    parser.add_argument('--difficulty-json', type=str, default=None, metavar='PATH',
                        help='Export difficulty analysis as JSON (e.g., difficulty.json)')
    
    # Debug spectrogram visualization
    parser.add_argument('--debug-spectrogram', type=str, default=None, metavar='PATH',
                        help='Generate debug spectrogram PNG with pitch overlays (e.g., debug.png)')
    parser.add_argument('--debug-spectrogram-sections', type=int, default=0,
                        help='Generate N zoomed section spectrograms (0 = single full view)')
    parser.add_argument('--debug-time-range', nargs=2, type=float, metavar=('START', 'END'),
                        help='Time range for spectrogram (start end) in seconds')
    
    # Add audio preprocessing arguments
    add_preprocessing_args(parser)
    
    # Add frequency domain cleanup arguments
    if HAS_FREQ_CLEANUP:
        add_freq_cleanup_args(parser)
    
    # Add lead guitar isolation arguments
    if HAS_LEAD_ISOLATION:
        add_lead_isolation_args(parser)
    
    # Add genre optimization arguments
    if HAS_GENRE_OPTIMIZER:
        add_genre_args(parser)
    else:
        # Fallback if module not loaded
        parser.add_argument('--genre', '-g', type=str, default=None,
                            choices=['metal', 'rock', 'blues', 'default'],
                            help='Genre for optimized detection (metal, rock, blues). '
                                 'Adjusts scale priors, technique sensitivity, and tempo expectations.')
    
    # Add solo range detection arguments
    if HAS_SOLO_DETECTION:
        add_solo_detection_args(parser)
    else:
        # Fallback if module not loaded
        parser.add_argument('--detect-solos', action='store_true',
                            help='Enable automatic solo vs rhythm section detection')
    
    # Fast legato run detection arguments
    parser.add_argument('--fast-legato', action='store_true',
                        help='Enable fast legato run detection (hammer-ons, pull-offs, 3nps runs)')
    parser.add_argument('--legato-min-duration', type=float, default=30.0,
                        help='Minimum note duration in fast legato runs (ms, default: 30)')
    parser.add_argument('--legato-threshold', type=float, default=125.0,
                        help='IOI threshold for fast passages (ms, default: 125)')
    parser.add_argument('--legato-min-notes', type=int, default=4,
                        help='Minimum notes to form a legato run (default: 4)')
    parser.add_argument('--legato-scale-match', action='store_true', default=True,
                        help='Use scale pattern matching to fill gaps (default: on)')
    parser.add_argument('--no-legato-scale-match', action='store_true',
                        help='Disable scale pattern matching for legato runs')
    
    args = parser.parse_args()
    
    # Parse tuning
    try:
        tuning = get_tuning(args.tuning)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Handle YouTube URLs
    audio_path = args.audio_file
    cleanup_file = False
    
    if is_youtube_url(audio_path):
        audio_path = download_youtube_audio(audio_path)
        cleanup_file = True
    
    # Determine title from filename if not specified
    title = args.title or os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create preprocessing config
    preprocess_config = PreprocessingConfig.from_args(args)
    
    # Create distortion preprocessing config
    distortion_preprocess_config = DistortionPreprocessingConfig.from_args(args)
    
    # Create frequency cleanup config
    freq_cleanup_config = None
    if HAS_FREQ_CLEANUP:
        freq_cleanup_config = freq_config_from_args(args)
    
    # Create lead guitar isolation config
    lead_isolation_config = None
    if HAS_LEAD_ISOLATION:
        lead_isolation_config = lead_config_from_args(args)
        # Store save path if specified
        if lead_isolation_config.enabled and hasattr(args, 'save_lead_audio') and args.save_lead_audio:
            lead_isolation_config._save_path = args.save_lead_audio
    
    # Create solo detection config
    solo_detection_config = None
    detect_solos = getattr(args, 'detect_solos', False)
    if HAS_SOLO_DETECTION and detect_solos:
        solo_detection_config = solo_config_from_args(args)
    
    print("🎸 Guitar Tab Generator (Enhanced)")
    print("=" * 40)
    print(f"Tuning: {args.tuning} {tuning}")
    if args.polyphonic:
        print(f"Mode: POLYPHONIC ({args.poly_method.upper()})")
        print(f"Max simultaneous notes: {args.max_simultaneous}")
    else:
        print(f"Mode: Monophonic")
        print(f"Pitch method: {args.pitch_method}")
        print(f"Harmonic separation: {'enabled' if not args.no_harmonic_separation else 'disabled'}")
        print(f"Octave correction: {'enabled' if not getattr(args, 'no_octave_correction', False) else 'disabled'}")
        print(f"Median filter: {args.median_filter if args.median_filter > 0 else 'disabled'}")
    if preprocess_config.enabled:
        print(f"Preprocessing: ENABLED")
    if distortion_preprocess_config.enabled:
        print(f"Distortion Preprocessing: ENABLED")
    if getattr(args, 'distortion', False):
        print(f"Distortion Mode: ENABLED (HPS with {getattr(args, 'distortion_harmonics', 5)} harmonics)")
    if freq_cleanup_config and freq_cleanup_config.enabled:
        print(f"Frequency Cleanup: ENABLED")
    if lead_isolation_config and lead_isolation_config.enabled:
        print(f"Lead Isolation: ENABLED (Demucs + M/S + frequency)")
    if args.genre:
        print(f"Genre Optimization: {args.genre.upper()}")
    if detect_solos and HAS_SOLO_DETECTION:
        print(f"Solo Detection: ENABLED (auto-detect solo vs rhythm)")
    if getattr(args, 'scale_constrain', False):
        scale_type_info = getattr(args, 'scale_constrain_type', None) or 'auto'
        print(f"Scale Constraint: ENABLED (scale: {scale_type_info})")
    print()
    
    # Detect notes - use polyphonic or monophonic detection
    if args.polyphonic:
        notes = detect_notes_polyphonic(
            audio_path,
            min_note_duration=args.min_duration,
            confidence_threshold=args.confidence,
            method=args.poly_method,
            max_simultaneous=args.max_simultaneous,
            tuning=tuning,
            preprocess_config=preprocess_config,
            save_preprocessed=getattr(args, 'save_preprocessed', None),
            lead_isolation_config=lead_isolation_config,
            distortion_preprocess_config=distortion_preprocess_config
        )
    else:
        notes = detect_notes_from_audio(
            audio_path,
            confidence_threshold=args.confidence,
            pitch_method=args.pitch_method,
            use_harmonic_separation=not args.no_harmonic_separation,
            median_filter_size=args.median_filter,
            min_note_duration=args.min_duration,
            tuning=tuning,
            preprocess_config=preprocess_config,
            freq_cleanup_config=freq_cleanup_config,
            save_preprocessed=getattr(args, 'save_preprocessed', None),
            crepe_model=getattr(args, 'crepe_model', 'small'),
            octave_correction=not getattr(args, 'no_octave_correction', False),
            lead_isolation_config=lead_isolation_config,
            distortion_mode=getattr(args, 'distortion', False),
            distortion_harmonics=getattr(args, 'distortion_harmonics', 5),
            distortion_preprocess_config=distortion_preprocess_config
        )
    
    if not notes:
        print("No notes detected! Try lowering confidence threshold with -c 0.1")
        sys.exit(1)
    
    # =========================================================================
    # FAST LEGATO RUN DETECTION
    # =========================================================================
    legato_runs = []
    if HAS_FAST_LEGATO and args.fast_legato:
        print("\n🎸 Fast Legato Run Detection:")
        print("-" * 40)
        
        # Load audio for legato detection
        y_legato, sr_legato = librosa.load(audio_path, sr=22050, mono=True)
        
        # Create detector
        enable_scale_matching = not getattr(args, 'no_legato_scale_match', False)
        
        detector = FastLegatoDetector(
            sr=sr_legato,
            hop_length=256,  # Small hop for fast passages
            min_note_duration_ms=getattr(args, 'legato_min_duration', 30.0),
            fast_run_threshold_ms=getattr(args, 'legato_threshold', 125.0),
            min_run_notes=getattr(args, 'legato_min_notes', 4),
            enable_scale_matching=enable_scale_matching,
        )
        
        # Detect legato runs
        legato_notes, legato_runs = detector.detect(y_legato, existing_notes=notes, verbose=True)
        
        # Convert legato notes to Note objects and merge
        if legato_notes:
            new_notes = []
            for ln in legato_notes:
                new_note = Note(
                    midi=ln.midi,
                    start_time=ln.start_time,
                    duration=ln.end_time - ln.start_time,
                    confidence=ln.confidence * 0.9  # Slightly lower weight for legato
                )
                new_notes.append(new_note)
            
            # Merge with existing notes, avoiding duplicates
            existing_times = {(round(n.start_time, 3), n.midi) for n in notes}
            for nn in new_notes:
                key = (round(nn.start_time, 3), nn.midi)
                # Check for nearby duplicates (within 50ms)
                is_dup = any(
                    abs(nn.start_time - t) < 0.05 and nn.midi == m 
                    for t, m in existing_times
                )
                if not is_dup:
                    notes.append(nn)
            
            # Re-sort by time
            notes.sort(key=lambda n: n.start_time)
            
            print(f"   Added {len(new_notes)} additional legato notes")
            print(f"   Total notes after legato detection: {len(notes)}")
            
            # Print run summary
            if legato_runs:
                print(f"\n   🎵 Legato Runs Detected:")
                for i, run in enumerate(legato_runs[:5]):  # Show first 5
                    scale_info = f" ({run.scale_type})" if run.scale_type else ""
                    nps_info = " [3nps]" if run.is_3nps else ""
                    print(f"      {i+1}. {run.num_notes} notes, {run.direction}{scale_info}{nps_info}, "
                          f"{run.notes_per_second:.1f} nps @ {run.start_time:.2f}s")
                if len(legato_runs) > 5:
                    print(f"      ... and {len(legato_runs) - 5} more runs")
    
    # =========================================================================
    # GENRE-SPECIFIC OPTIMIZATION
    # =========================================================================
    genre_optimizer = None
    if HAS_GENRE_OPTIMIZER and args.genre:
        print(f"\n🎸 Applying {args.genre.upper()} genre optimization...")
        print("-" * 40)
        
        # Apply genre-specific priors to adjust confidence scores
        notes, genre_optimizer = apply_genre_optimization(
            notes,
            genre=args.genre,
            detected_key=None,  # Will be detected in post-processing
            verbose=True
        )
        
        # Get genre-specific detection hints
        profile = get_genre_profile(args.genre)
        print(f"   Min note duration: {profile.min_note_duration:.3f}s")
        print(f"   Technique boosts: bend={profile.technique_sensitivity.get('bend', 1.0):.1f}x, "
              f"vibrato={profile.technique_sensitivity.get('vibrato', 1.0):.1f}x, "
              f"tapping={profile.technique_sensitivity.get('tapping', 1.0):.1f}x")
    
    # Print detected notes (before post-processing)
    print("\n📝 Detected Notes (raw):")
    for note in notes[:20]:  # Show first 20
        print(f"  {note.name:4} at {note.start_time:.2f}s (confidence: {note.confidence:.2f})")
    if len(notes) > 20:
        print(f"  ... and {len(notes) - 20} more")
    
    # =========================================================================
    # SCALE-CONSTRAINED DETECTION
    # =========================================================================
    scale_constrain_key = None
    if HAS_SCALE_CONSTRAIN and getattr(args, 'scale_constrain', False):
        print("\n🎯 Scale-Constrained Detection:")
        print("-" * 40)
        
        # Create config from args
        scale_config = ScaleConstraintConfig(
            enabled=True,
            scale_type=getattr(args, 'scale_constrain_type', None),
            snap_direction=getattr(args, 'scale_snap_direction', 'nearest'),
            allow_chromatic_passing=getattr(args, 'allow_passing_tones', False),
            passing_max_duration=getattr(args, 'passing_tone_max_ms', 100) / 1000.0,
            key_override=args.key,  # Use --key if specified
            verbose=True
        )
        
        # Apply scale constraint
        notes, scale_constrain_key, scale_stats = apply_scale_constraint_to_detection(
            audio_path,
            notes,
            config=scale_config
        )
        
        print(f"   Notes processed: {scale_stats['total']}")
        print(f"   Already in scale: {scale_stats['in_scale']}")
        print(f"   Snapped to scale: {scale_stats['snapped']}")
        if scale_stats['passing_tones'] > 0:
            print(f"   Passing tones kept: {scale_stats['passing_tones']}")
        
        # Show note distribution after constraint
        if notes:
            from collections import Counter
            note_names = [NOTE_NAMES[n.midi % 12] for n in notes]
            dist = Counter(note_names).most_common(7)
            print(f"   Scale notes used: {', '.join(f'{name}({count})' for name, count in dist)}")
    elif getattr(args, 'scale_constrain', False) and not HAS_SCALE_CONSTRAIN:
        print("\n⚠️  Scale constraint module not available.")
        print("   Make sure scale_constrain.py is in the same directory.")
    
    # =========================================================================
    # DEBUG SPECTROGRAM VISUALIZATION
    # =========================================================================
    if args.debug_spectrogram:
        if HAS_DEBUG_SPECTROGRAM:
            print("\n🔍 Generating Debug Spectrogram...")
            print("-" * 40)
            
            # Convert notes to DebugNote format
            debug_notes = [
                DebugNote(
                    midi_note=n.midi,
                    start_time=n.start_time,
                    end_time=n.start_time + n.duration,
                    confidence=n.confidence,
                    name=n.name
                )
                for n in notes
            ]
            
            # Time range (optional)
            time_range = None
            if args.debug_time_range:
                time_range = tuple(args.debug_time_range)
            
            # Generate spectrogram(s)
            if args.debug_spectrogram_sections > 0:
                # Generate multiple zoomed sections
                output_dir = os.path.splitext(args.debug_spectrogram)[0] + "_sections"
                paths = generate_zoomed_spectrograms(
                    audio_path,
                    debug_notes,
                    output_dir=output_dir,
                    num_sections=args.debug_spectrogram_sections,
                    section_duration=5.0
                )
                print(f"\n📊 Generated {len(paths)} section spectrograms in: {output_dir}/")
            else:
                # Single spectrogram
                generate_debug_spectrogram(
                    audio_path,
                    debug_notes,
                    output_path=args.debug_spectrogram,
                    time_range=time_range,
                    title=title
                )
        else:
            print("⚠️  Debug spectrogram module not available (debug_spectrogram.py missing)")
    
    # =========================================================================
    # MUSICAL POST-PROCESSING
    # =========================================================================
    detected_key = None
    patterns = []
    
    if HAS_MUSIC_THEORY:
        print("\n🎵 Musical Post-Processing:")
        print("-" * 40)
        
        # Parse user-specified key or use scale-constrain detected key
        user_key = None
        if scale_constrain_key:
            # Use key from scale constraint (already detected)
            user_key = scale_constrain_key
            print(f"🔑 Using key from scale constraint: {user_key.name}")
        elif args.key:
            try:
                user_key = parse_key_string(args.key)
                print(f"🔑 Using specified key: {user_key.name}")
            except Exception as e:
                print(f"⚠️  Could not parse key '{args.key}': {e}")
                print("   Will auto-detect instead.")
        
        # Determine scale type - use genre-suggested scale if no scale specified
        scale_type = args.scale
        if scale_type is None and args.genre and HAS_GENRE_OPTIMIZER:
            profile = get_genre_profile(args.genre)
            if profile.preferred_scales:
                # Use pentatonic minor for metal/rock by default (most common for lead)
                if args.genre in ('metal', 'rock'):
                    scale_type = 'pentatonic_minor'
                else:
                    scale_type = profile.preferred_scales[0]
                print(f"🎸 Using {args.genre} genre scale: {scale_type}")
        
        # Apply post-processing (key detection, pitch snapping, quantization)
        notes, detected_key, patterns = post_process_notes(
            notes,
            key=user_key,
            scale_type=scale_type,
            snap_to_scale_enabled=not args.no_snap,
            quantize_enabled=args.quantize is not None,
            tempo=args.tempo,
            subdivision=args.quantize if args.quantize else 16,
            detect_patterns_enabled=args.detect_patterns
        )
        
        # Show detected patterns
        if patterns:
            print(format_pattern_info(patterns, notes))
    else:
        if args.key or args.quantize or not args.no_snap:
            print("⚠️  Music theory module not available. Skipping post-processing.")
            print("   Make sure music_theory.py is in the same directory.")
    
    # Detect chords if enabled
    chords = None
    if args.chords:
        print("\n🎵 Detecting chords...")
        chords = detect_chords(notes, time_threshold=args.chord_threshold)
        
        if chords:
            print(f"\n🎶 Detected {len(chords)} chords:")
            unique_chords = {}
            for chord in chords:
                if chord.name not in unique_chords:
                    unique_chords[chord.name] = 0
                unique_chords[chord.name] += 1
            
            for chord_name, count in sorted(unique_chords.items()):
                print(f"  {chord_name}: {count}x")
            
            # Show chord progression
            print("\n📋 Chord Progression:")
            progression = [c.name for c in chords[:16]]  # First 16 chords
            print(f"  {' → '.join(progression)}")
            if len(chords) > 16:
                print(f"  ... and {len(chords) - 16} more")
        else:
            print("  No chord patterns detected")
    
    # =========================================================================
    # SOLO SECTION DETECTION
    # =========================================================================
    detected_sections = None
    if detect_solos and HAS_SOLO_DETECTION:
        print("\n🎸 Detecting Solo vs Rhythm Sections...")
        print("-" * 40)
        
        # Load audio for solo detection
        y_solo, sr_solo = librosa.load(audio_path, sr=22050, mono=True)
        
        # Detect sections
        detected_sections = detect_solo_sections(
            y=y_solo,
            sr=sr_solo,
            config=solo_detection_config,
            verbose=True
        )
        
        # Apply range constraints to notes based on section type
        if detected_sections:
            print("\n  Applying section-specific pitch constraints...")
            notes = apply_solo_range_constraints(notes, detected_sections, tuning, verbose=True)
    
    # Convert to tabs - use spectral string detection if requested
    use_spectral = args.spectral_strings and HAS_STRING_DETECTION
    
    if use_spectral:
        print("\n🎸 Using spectral string detection...")
        if not HAS_STRING_DETECTION:
            print("⚠️  String detection module not available, falling back to heuristics")
            use_spectral = False
    
    # Convert to tabs - use optimizer if requested
    if args.optimize_fingering:
        if HAS_FINGERING_OPTIMIZER:
            print("\n🎯 Optimizing fingering...")
            
            # Create custom config from args
            from fingering_optimizer import FingeringConfig
            config = FingeringConfig(
                max_fret_span=args.fingering_max_span,
                low_fret_bonus=args.fingering_low_fret_bonus,
                open_string_bonus=args.fingering_open_string_bonus,
            )
            
            # Get unoptimized tabs for comparison
            if use_spectral:
                original_tabs = notes_to_tabs_spectral(
                    notes, audio_path=audio_path, tuning=tuning,
                    verbose=args.string_detection_verbose
                )
            else:
                original_tabs = notes_to_tabs(notes, tuning)
            
            # Run optimization
            tab_notes = optimize_fingering(notes, tuning=tuning, config=config, verbose=True)
            
            # Show comparison metrics
            if original_tabs and tab_notes:
                comparison = compare_fingerings(notes, original_tabs, tab_notes, tuning)
                print("\n📊 Fingering Comparison:")
                print(f"   Original avg fret: {comparison['original'].get('avg_fret', 0):.1f}")
                print(f"   Optimized avg fret: {comparison['optimized'].get('avg_fret', 0):.1f}")
                print(f"   Original avg fret jump: {comparison['original'].get('avg_fret_jump', 0):.1f}")
                print(f"   Optimized avg fret jump: {comparison['optimized'].get('avg_fret_jump', 0):.1f}")
                if comparison['improvement'].get('avg_fret_jump', 0) > 0:
                    print(f"   ✅ Reduced fret jumps by {comparison['improvement']['avg_fret_jump']:.1f} on average")
        else:
            print("⚠️  Fingering optimizer not available. Using default tab assignment.")
            if use_spectral:
                tab_notes = notes_to_tabs_spectral(
                    notes, audio_path=audio_path, tuning=tuning,
                    verbose=args.string_detection_verbose
                )
            else:
                tab_notes = notes_to_tabs(notes, tuning)
    else:
        # Use solo-aware tab conversion if sections were detected
        if detected_sections and HAS_SOLO_DETECTION:
            print("\n🎸 Using solo-aware fret placement...")
            tab_notes = notes_to_tabs_with_solo_detection(
                notes, detected_sections, tuning, verbose=args.string_detection_verbose
            )
        elif use_spectral:
            tab_notes = notes_to_tabs_spectral(
                notes, audio_path=audio_path, tuning=tuning,
                verbose=args.string_detection_verbose
            )
        else:
            tab_notes = notes_to_tabs(notes, tuning)
    
    # Apply additional physical playability optimizations (if not using optimizer)
    if not args.optimize_fingering and HAS_MUSIC_THEORY and tab_notes:
        # Filter physically impossible transitions
        if not args.no_playability_filter:
            tab_notes = filter_impossible_transitions(
                tab_notes, notes, 
                max_fret_jump=5,
                min_time_for_jump=0.05,
                tuning=tuning
            )
        
        # Optimize for lower fret positions
        tab_notes = prefer_lower_frets(tab_notes, notes, tuning=tuning)
    
    # Rhythm analysis
    tempo_info = None
    rhythm_output = None
    detected_bpm = args.tempo
    
    if (args.rhythm or args.detect_tempo) and HAS_RHYTHM_NOTATION:
        print("\n🎵 Rhythm Analysis:")
        print("-" * 40)
        
        # Load audio for tempo detection
        y_rhythm, sr_rhythm = librosa.load(audio_path, sr=22050)
        
        # Analyze rhythm
        rhythm_result = analyze_audio_rhythm(
            y=y_rhythm,
            sr=sr_rhythm,
            notes=notes
        )
        
        tempo_info = rhythm_result['tempo_info']
        detected_bpm = rhythm_result['tempo']['bpm']
        
        print(f"  Detected tempo: {detected_bpm:.1f} BPM")
        print(f"  Confidence: {rhythm_result['tempo']['confidence']:.2f}")
        print(f"  Time signature: {rhythm_result['time_signature']['numerator']}/{rhythm_result['time_signature']['denominator']}")
        
        if 'swing' in rhythm_result and rhythm_result['swing']['detected']:
            print(f"  Swing detected: ratio {rhythm_result['swing']['ratio']:.2f}")
        
        if 'triplets' in rhythm_result and rhythm_result['triplets']:
            print(f"  Triplet groups: {len(rhythm_result['triplets'])}")
        
        # Use detected tempo if requested
        if args.detect_tempo:
            args.tempo = int(round(detected_bpm))
            print(f"  Using detected tempo: {args.tempo} BPM")
    elif (args.rhythm or args.detect_tempo) and not HAS_RHYTHM_NOTATION:
        print("\n⚠️  Rhythm notation module not available.")
        print("   Make sure rhythm_notation.py is in the same directory.")
    
    # Format as ASCII tab (always show preview, with chords if detected)
    if args.rhythm and HAS_RHYTHM_NOTATION and tempo_info:
        tab_output = format_tab_with_rhythm(
            tab_notes,
            notes,
            tempo_info,
            beats_per_line=8,
            tuning=tuning,
            use_unicode=args.unicode_rhythm,
            show_tempo=True,
            show_time_sig=True
        )
    else:
        tab_output = format_ascii_tab(tab_notes, tuning=tuning, chords=chords)
    
    # Show chord diagrams if requested
    chord_diagrams = ""
    if args.chord_diagrams and chords:
        chord_diagrams = generate_all_chord_diagrams(chords)
        print("\n📊 Chord Diagrams:")
        print("-" * 40)
        print(chord_diagrams)
    
    print("\n🎼 Guitar Tablature:")
    print("-" * 40)
    print(tab_output)
    
    # Difficulty analysis
    difficulty_report = None
    if args.analyze_difficulty or args.difficulty_json:
        if HAS_DIFFICULTY_ANALYZER:
            # Try to get annotated notes with techniques for better analysis
            annotated_notes = None
            try:
                from technique_detector import TechniqueDetector
                y_tech, sr_tech = librosa.load(audio_path, sr=22050)
                detector = TechniqueDetector(sr=sr_tech)
                annotated_notes = detector.analyze(
                    y=y_tech,
                    notes=notes,
                    tuning=tuning,
                    verbose=False
                )
            except Exception:
                pass  # Continue without technique annotations
            
            difficulty_report = analyze_difficulty(
                tab_notes=tab_notes,
                notes=notes,
                annotated_notes=annotated_notes,
                tempo=args.tempo,
                verbose=True
            )
            
            if args.analyze_difficulty:
                print(difficulty_report.format_report())
            
            if args.difficulty_json:
                import json
                with open(args.difficulty_json, 'w') as f:
                    json.dump(difficulty_report.to_dict(), f, indent=2)
                print(f"\n✅ Difficulty analysis saved to {args.difficulty_json}")
        else:
            print("\n⚠️  Difficulty analyzer module not available.")
            print("   Make sure difficulty_analyzer.py is in the same directory.")
    
    # Save to file if requested
    if args.output:
        format_name = args.format.lower()
        output_path = args.output
        
        # Add extension if needed
        if not os.path.splitext(output_path)[1]:
            output_path += get_export_extension(format_name)
        
        if format_name in ('gp5', 'gp', 'guitarpro'):
            if not HAS_GUITARPRO:
                print("\n⚠️  pyguitarpro not installed. Falling back to MusicXML...")
                format_name = 'musicxml'
                output_path = os.path.splitext(output_path)[0] + '.musicxml'
            else:
                success = export_guitar_pro(
                    tab_notes,
                    output_path,
                    title=title,
                    artist=args.artist,
                    tempo=args.tempo,
                    tuning=tuning
                )
                if not success:
                    sys.exit(1)
        
        if format_name in ('musicxml', 'xml'):
            success = export_musicxml(
                tab_notes,
                output_path,
                title=title,
                composer=args.artist,
                tempo=args.tempo,
                tuning=tuning
            )
            if not success:
                sys.exit(1)
        
        elif format_name in ('lilypond', 'ly'):
            success = export_lilypond(
                tab_notes,
                notes,
                output_path,
                title=title,
                composer=args.artist,
                tempo=args.tempo,
                tuning=tuning,
                chords=chords,
                include_standard_notation=True
            )
            if not success:
                sys.exit(1)
        
        elif format_name in ('ascii', 'txt'):
            with open(output_path, 'w') as f:
                f.write(f"# Guitar Tab - {title}\n")
                f.write(f"# Generated from: {os.path.basename(audio_path)}\n")
                f.write(f"# Tuning: {args.tuning}\n")
                f.write(f"# Tempo: {args.tempo} BPM\n")
                
                # Add key information
                if detected_key:
                    f.write(f"# Key: {detected_key.name} (confidence: {detected_key.confidence:.2f})\n")
                if args.quantize:
                    f.write(f"# Quantized: 1/{args.quantize} notes\n")
                f.write("\n")
                
                if chords:
                    unique_chords = list(set(c.name for c in chords))
                    f.write(f"## Chords Used: {', '.join(sorted(unique_chords))}\n\n")
                
                # Add pattern info
                if patterns:
                    f.write("## Detected Patterns/Riffs\n\n")
                    for i, pattern in enumerate(patterns[:5], 1):
                        times = [f"{t:.2f}s" for t in pattern.occurrences[:3]]
                        f.write(f"- Pattern {i}: {pattern.length} notes, {pattern.count}x occurrences at {', '.join(times)}{'...' if len(pattern.occurrences) > 3 else ''}\n")
                    f.write("\n")
                
                if chord_diagrams:
                    f.write("## Chord Diagrams\n\n")
                    f.write("```\n")
                    f.write(chord_diagrams)
                    f.write("\n```\n\n")
                
                f.write("## Tablature\n\n")
                f.write("```\n")
                f.write(tab_output)
                f.write("\n```\n")
            print(f"\n✅ Saved to {output_path}")
    
    # Export sync data for web player if requested
    if args.export_sync:
        if not HAS_SYNC_EXPORT:
            print("\n⚠️  Sync export module not available.")
            print("   Make sure sync_export.py is in the same directory.")
        else:
            # Determine key name for sync data
            key_name = None
            if detected_key:
                key_name = detected_key.name
            elif args.key:
                key_name = args.key
            
            # Export sync JSON
            success = export_sync_json(
                tab_notes=tab_notes,
                notes=notes,
                output_path=args.export_sync,
                audio_path=audio_path,
                tempo=args.tempo,
                title=title,
                artist=args.artist,
                tuning=tuning,
                tuning_name=args.tuning,
                chords=chords,
                key=key_name,
                time_signature=(4, 4),
                verbose=True
            )
            if not success:
                print("❌ Failed to export sync data")
    
    # Cleanup temp file if downloaded from YouTube
    if cleanup_file and os.path.exists(audio_path):
        os.remove(audio_path)
        print("🗑️ Cleaned up temp audio file")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
