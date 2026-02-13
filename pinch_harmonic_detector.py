#!/usr/bin/env python3
"""
Pinch Harmonic & Squeal Detection for Distorted Lead Guitar

Detects characteristic techniques used in high-gain lead guitar:

1. PINCH HARMONICS (PH) - "Squeals"
   - Created by lightly touching string with thumb while picking
   - Produces high-pitched squeal (typically 3rd-5th harmonic dominant)
   - Spectral signature: sudden high-frequency spike, very high centroid
   - Often followed by vibrato or dive bomb

2. NATURAL HARMONICS (NH)
   - Produced at specific fret positions (5, 7, 12, etc.)
   - Very pure tone with single dominant harmonic
   - Long sustain, low spectral flatness
   - Clearly defined harmonic relationship to fundamental

3. DIVE BOMBS (dive)
   - Whammy bar pushed down causing continuous pitch descent
   - Characterized by rapid, continuous pitch drop
   - Often starts from a pinch harmonic
   - Can span 1-2+ octaves

4. WHAMMY BAR EFFECTS (wham)
   - Flutter: rapid pitch oscillation
   - Dip: brief pitch drop and return
   - Scream: upward pitch bend with vibrato

ASCII Tab Notation:
   7(PH)  - Pinch harmonic at fret 7
   <12>   - Natural harmonic at fret 12 (or [12] for harmonics)
   7~dive - Pinch harmonic with dive bomb
   3^wham - Whammy bar flutter/scream
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, medfilt, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class SquealTechnique(Enum):
    """Lead guitar squeal/harmonic techniques."""
    NORMAL = ""
    PINCH_HARMONIC = "PH"
    NATURAL_HARMONIC = "NH"
    ARTIFICIAL_HARMONIC = "AH"  # Tapped harmonics
    DIVE_BOMB = "dive"
    WHAMMY_DIP = "dip"
    WHAMMY_FLUTTER = "flutter"
    WHAMMY_SCREAM = "scream"


@dataclass
class SquealFeatures:
    """Features extracted for squeal/harmonic detection."""
    time: float
    duration: float
    
    # Frequency analysis
    fundamental_freq: float
    perceived_freq: float  # What we actually hear (may be harmonic)
    freq_ratio: float      # perceived / fundamental ratio
    
    # Spectral features
    spectral_centroid: float
    spectral_centroid_ratio: float  # centroid / fundamental
    spectral_flatness: float
    high_freq_energy_ratio: float   # Energy above 2kHz vs total
    
    # Harmonic analysis
    harmonic_number: int           # Which harmonic is dominant (1=fund, 2=oct, etc)
    harmonic_dominance: float      # How much stronger is the dominant harmonic
    harmonic_purity: float         # How pure is the tone (single harmonic vs spread)
    
    # Envelope features
    attack_sharpness: float        # How sudden is the onset
    sustain_stability: float       # How stable is the pitch during sustain
    
    # Pitch trajectory
    pitch_slope: float             # Positive=rising, negative=falling
    pitch_variance: float          # How much pitch moves during note
    pitch_direction_changes: int   # Number of direction reversals
    
    # High-frequency transient
    has_squeal_transient: bool     # Sudden HF spike at onset


@dataclass
class SquealDetection:
    """Result of squeal/technique detection."""
    time: float
    duration: float
    technique: SquealTechnique
    confidence: float
    harmonic_fret: Optional[int] = None  # For NH: which fret position
    dive_semitones: float = 0.0          # For dive: how far it dropped
    details: Dict = field(default_factory=dict)
    
    def to_tab_notation(self, fret: int) -> str:
        """Generate tab notation for this technique."""
        if self.technique == SquealTechnique.NORMAL:
            return str(fret)
        elif self.technique == SquealTechnique.PINCH_HARMONIC:
            return f"{fret}(PH)"
        elif self.technique == SquealTechnique.NATURAL_HARMONIC:
            return f"<{self.harmonic_fret or fret}>"
        elif self.technique == SquealTechnique.ARTIFICIAL_HARMONIC:
            return f"[{fret}](AH)"
        elif self.technique == SquealTechnique.DIVE_BOMB:
            return f"{fret}~dive"
        elif self.technique == SquealTechnique.WHAMMY_DIP:
            return f"{fret}^dip"
        elif self.technique == SquealTechnique.WHAMMY_FLUTTER:
            return f"{fret}~flutter"
        elif self.technique == SquealTechnique.WHAMMY_SCREAM:
            return f"{fret}^scream"
        return str(fret)


class PinchHarmonicDetector:
    """
    Detects pinch harmonics, squeals, dive bombs and whammy techniques.
    
    Optimized for high-gain distorted lead guitar where these techniques
    are commonly used.
    
    Detection Strategy:
    - Pinch harmonics: High spectral centroid, strong upper harmonics
    - Natural harmonics: Very pure tone, specific harmonic ratios
    - Dive bombs: Continuous pitch descent over time
    - Whammy: Pitch modulation patterns
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,  # Shorter for finer time resolution
        n_fft: int = 2048,
        # Pinch harmonic thresholds (stricter for high-gain)
        ph_centroid_ratio: float = 4.0,     # Centroid > 4x fundamental (stricter)
        ph_high_energy_ratio: float = 0.5,  # 50%+ energy above 2kHz (stricter)
        ph_transient_threshold: float = 2.5, # HF transient > 2.5x average
        ph_freq_ratio_min: float = 3.0,     # Perceived freq > 3x fundamental
        # Natural harmonic thresholds (much stricter for high-gain)
        nh_purity_threshold: float = 0.85,  # Single harmonic must be very dominant
        nh_sustain_threshold: float = 0.7,  # Very stable pitch required
        nh_flatness_max: float = 0.015,     # Very pure tone (stricter for distorted)
        nh_harmonic_match_required: bool = True,  # Must match known harmonic fret
        # Dive bomb thresholds
        dive_min_drop: float = 5.0,         # Min 5 semitones drop (stricter)
        dive_max_time: float = 2.0,         # Max 2s for dive
        dive_slope_threshold: float = -4.0, # Semitones per second (steeper)
        dive_continuity_min: float = 0.6,   # Min continuity score
        # General
        min_note_duration: float = 0.05,    # Min 50ms note
        # High-gain mode
        high_gain_mode: bool = True,        # Stricter thresholds for distorted guitar
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.high_gain_mode = high_gain_mode
        
        # Thresholds (adjusted for high-gain if enabled)
        self.ph_centroid_ratio = ph_centroid_ratio
        self.ph_high_energy_ratio = ph_high_energy_ratio
        self.ph_transient_threshold = ph_transient_threshold
        self.ph_freq_ratio_min = ph_freq_ratio_min
        
        self.nh_purity_threshold = nh_purity_threshold
        self.nh_sustain_threshold = nh_sustain_threshold
        self.nh_flatness_max = nh_flatness_max
        self.nh_harmonic_match_required = nh_harmonic_match_required
        
        self.dive_min_drop = dive_min_drop
        self.dive_max_time = dive_max_time
        self.dive_slope_threshold = dive_slope_threshold
        self.dive_continuity_min = dive_continuity_min
        
        self.min_note_duration = min_note_duration
        
        # Natural harmonic fret positions and their harmonic ratios
        self.natural_harmonic_frets = {
            12: 2.0,   # Octave (2nd harmonic)
            7: 3.0,    # 5th above octave (3rd harmonic)
            5: 4.0,    # 2nd octave (4th harmonic)
            4: 5.0,    # Major 3rd above 2nd octave
            3: 6.0,    # 5th above 2nd octave
            2.7: 7.0,  # Minor 7th (approximate fret)
            2.4: 8.0,  # 3rd octave
        }
    
    def extract_features(
        self,
        y: np.ndarray,
        start_time: float,
        duration: float,
        expected_freq: float
    ) -> Optional[SquealFeatures]:
        """Extract features for squeal/harmonic detection."""
        
        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + duration) * self.sr)
        
        if start_sample >= len(y) or end_sample <= start_sample:
            return None
        
        segment = y[start_sample:min(end_sample, len(y))]
        
        if len(segment) < self.n_fft:
            segment = np.pad(segment, (0, self.n_fft - len(segment)))
        
        # STFT
        D = librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Time-averaged spectrum
        S_mean = np.mean(S, axis=1)
        
        # 1. Find perceived frequency (strongest peak)
        perceived_freq = self._find_perceived_frequency(S_mean, freqs, expected_freq)
        freq_ratio = perceived_freq / (expected_freq + 1e-10)
        
        # 2. Spectral centroid
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        avg_centroid = np.mean(centroid)
        centroid_ratio = avg_centroid / (expected_freq + 1e-10)
        
        # 3. Spectral flatness
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        avg_flatness = np.mean(flatness)
        
        # 4. High frequency energy ratio (> 2kHz)
        high_freq_mask = freqs > 2000
        high_energy = np.sum(S_mean[high_freq_mask])
        total_energy = np.sum(S_mean) + 1e-10
        high_freq_ratio = high_energy / total_energy
        
        # 5. Harmonic analysis
        harmonic_num, harmonic_dom, harmonic_purity = self._analyze_harmonics(
            S_mean, freqs, expected_freq
        )
        
        # 6. Attack analysis
        attack_sharpness = self._analyze_attack(segment)
        
        # 7. Pitch trajectory
        pitch_slope, pitch_var, direction_changes, sustain_stability = (
            self._analyze_pitch_trajectory(segment)
        )
        
        # 8. High-frequency transient detection
        has_hf_transient = self._detect_hf_transient(S, freqs)
        
        return SquealFeatures(
            time=start_time,
            duration=duration,
            fundamental_freq=expected_freq,
            perceived_freq=perceived_freq,
            freq_ratio=freq_ratio,
            spectral_centroid=avg_centroid,
            spectral_centroid_ratio=centroid_ratio,
            spectral_flatness=avg_flatness,
            high_freq_energy_ratio=high_freq_ratio,
            harmonic_number=harmonic_num,
            harmonic_dominance=harmonic_dom,
            harmonic_purity=harmonic_purity,
            attack_sharpness=attack_sharpness,
            sustain_stability=sustain_stability,
            pitch_slope=pitch_slope,
            pitch_variance=pitch_var,
            pitch_direction_changes=direction_changes,
            has_squeal_transient=has_hf_transient
        )
    
    def _find_perceived_frequency(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray,
        expected_freq: float
    ) -> float:
        """Find the actual perceived frequency (may be a harmonic)."""
        # Look for peaks in reasonable range
        min_freq = expected_freq * 0.8
        max_freq = expected_freq * 10  # Up to 10th harmonic
        
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(mask):
            return expected_freq
        
        # Find the strongest peak
        masked_spectrum = S_mean.copy()
        masked_spectrum[~mask] = 0
        
        peak_idx = np.argmax(masked_spectrum)
        return freqs[peak_idx]
    
    def _analyze_harmonics(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray,
        fundamental: float
    ) -> Tuple[int, float, float]:
        """
        Analyze harmonic content.
        
        Returns:
            harmonic_number: Which harmonic is dominant (1=fundamental)
            harmonic_dominance: How much stronger is the dominant harmonic
            harmonic_purity: How pure is the tone (single harmonic vs spread)
        """
        if fundamental < 50:
            return 1, 0.0, 0.0
        
        # Measure energy at each harmonic (1-8)
        harmonic_energies = []
        
        for h in range(1, 9):
            h_freq = fundamental * h
            h_bin = int(h_freq * self.n_fft / self.sr)
            
            if h_bin >= len(S_mean):
                harmonic_energies.append(0)
                continue
            
            # Search around expected position
            search_start = max(0, h_bin - 3)
            search_end = min(len(S_mean), h_bin + 4)
            h_energy = np.max(S_mean[search_start:search_end])
            harmonic_energies.append(h_energy)
        
        harmonic_energies = np.array(harmonic_energies)
        
        if np.max(harmonic_energies) < 1e-10:
            return 1, 0.0, 0.0
        
        # Find dominant harmonic
        dominant_idx = np.argmax(harmonic_energies)
        dominant_energy = harmonic_energies[dominant_idx]
        
        # Calculate dominance (ratio to second strongest)
        sorted_energies = np.sort(harmonic_energies)[::-1]
        if sorted_energies[1] > 1e-10:
            dominance = sorted_energies[0] / sorted_energies[1]
        else:
            dominance = 10.0  # Very dominant
        
        # Calculate purity (how concentrated energy is)
        total_energy = np.sum(harmonic_energies) + 1e-10
        purity = dominant_energy / total_energy
        
        return dominant_idx + 1, dominance, purity
    
    def _analyze_attack(self, segment: np.ndarray) -> float:
        """Measure attack sharpness (how sudden the onset is)."""
        envelope = np.abs(segment)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.002 * self.sr))
        
        if len(envelope_smooth) < 10:
            return 0.0
        
        # Find peak in first 50ms
        attack_samples = int(0.05 * self.sr)
        attack_region = envelope_smooth[:min(attack_samples, len(envelope_smooth))]
        
        peak_idx = np.argmax(attack_region)
        if peak_idx == 0:
            return 0.0
        
        # Attack sharpness = rise rate to peak
        rise = attack_region[peak_idx] - attack_region[0]
        rise_time = peak_idx / self.sr
        
        sharpness = rise / (rise_time + 1e-10)
        
        # Normalize to 0-1 range
        return min(1.0, sharpness / 100)
    
    def _analyze_pitch_trajectory(
        self,
        segment: np.ndarray
    ) -> Tuple[float, float, int, float]:
        """
        Analyze pitch movement over the note duration.
        
        Returns:
            slope: Pitch change rate (semitones per second)
            variance: Pitch variance (semitones)
            direction_changes: Number of direction reversals
            sustain_stability: How stable pitch is during sustain
        """
        # Quick pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            segment,
            fmin=60,
            fmax=2000,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0
        )
        
        # Get valid pitches
        valid_mask = (f0 > 0) & (~np.isnan(f0))
        if np.sum(valid_mask) < 3:
            return 0.0, 0.0, 0, 1.0
        
        valid_f0 = f0[valid_mask]
        
        # Convert to MIDI for semitone calculations
        midi_pitches = librosa.hz_to_midi(valid_f0)
        
        # Slope (linear fit)
        x = np.arange(len(midi_pitches))
        if len(x) > 1:
            coeffs = np.polyfit(x, midi_pitches, 1)
            # Convert to semitones per second
            frames_per_sec = self.sr / self.hop_length
            slope = coeffs[0] * frames_per_sec
        else:
            slope = 0.0
        
        # Variance
        variance = np.std(midi_pitches)
        
        # Direction changes
        diff = np.diff(midi_pitches)
        direction_changes = np.sum(np.diff(np.sign(diff)) != 0)
        
        # Sustain stability (ignore first 10% for attack)
        sustain_start = int(len(midi_pitches) * 0.1)
        if sustain_start < len(midi_pitches) - 2:
            sustain_pitches = midi_pitches[sustain_start:]
            sustain_stability = 1.0 - min(1.0, np.std(sustain_pitches) / 2.0)
        else:
            sustain_stability = 1.0
        
        return slope, variance, direction_changes, sustain_stability
    
    def _detect_hf_transient(
        self,
        S: np.ndarray,
        freqs: np.ndarray
    ) -> bool:
        """
        Detect high-frequency transient at note onset.
        
        Pinch harmonics often have a distinctive "squeal" onset
        with sudden HF energy.
        """
        # High frequency region (> 3kHz)
        hf_mask = freqs > 3000
        
        if S.shape[1] < 3:
            return False
        
        # Energy in HF region over time
        hf_energy = np.sum(S[hf_mask, :], axis=0)
        
        if len(hf_energy) < 3:
            return False
        
        # Compare first frames to average
        onset_frames = min(3, len(hf_energy) // 3)
        onset_hf = np.mean(hf_energy[:onset_frames])
        avg_hf = np.mean(hf_energy)
        
        return onset_hf > avg_hf * self.ph_transient_threshold
    
    def detect_pinch_harmonic(
        self,
        features: SquealFeatures
    ) -> Tuple[bool, float, Dict]:
        """
        Detect pinch harmonic (artificial harmonic / squeal).
        
        Characteristics:
        - Very high spectral centroid (squealing)
        - Dominant upper harmonics (3rd-5th typically)
        - High-frequency transient at onset
        - Still has some harmonic structure (not noise)
        """
        scores = []
        details = {}
        
        # 1. Spectral centroid ratio (key indicator)
        details['centroid_ratio'] = features.spectral_centroid_ratio
        if features.spectral_centroid_ratio > self.ph_centroid_ratio:
            # Strong squeal indicator
            score = min(1.0, (features.spectral_centroid_ratio - self.ph_centroid_ratio) / 
                        self.ph_centroid_ratio)
            scores.append(0.8 + 0.2 * score)
        elif features.spectral_centroid_ratio > self.ph_centroid_ratio * 0.7:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # 2. High frequency energy
        details['hf_energy_ratio'] = features.high_freq_energy_ratio
        if features.high_freq_energy_ratio > self.ph_high_energy_ratio:
            scores.append(1.0)
        elif features.high_freq_energy_ratio > self.ph_high_energy_ratio * 0.6:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # 3. HF transient
        details['has_hf_transient'] = features.has_squeal_transient
        if features.has_squeal_transient:
            scores.append(1.0)
        else:
            scores.append(0.3)  # Not required but helps
        
        # 4. Upper harmonic dominance (3rd or higher)
        details['dominant_harmonic'] = features.harmonic_number
        if features.harmonic_number >= 3:
            scores.append(0.9)
        elif features.harmonic_number == 2:
            scores.append(0.4)
        else:
            scores.append(0.0)
        
        # 5. Not noise (still harmonic)
        details['flatness'] = features.spectral_flatness
        if features.spectral_flatness < 0.15:  # Still tonal
            scores.append(0.8)
        elif features.spectral_flatness < 0.25:
            scores.append(0.4)
        else:
            scores.append(0.0)  # Too noisy
        
        # 6. Perceived frequency much higher than fundamental
        details['freq_ratio'] = features.freq_ratio
        if features.freq_ratio > 2.5:
            scores.append(0.9)
        elif features.freq_ratio > 1.8:
            scores.append(0.5)
        else:
            scores.append(0.1)
        
        # Calculate weighted confidence
        weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.15]
        confidence = sum(s * w for s, w in zip(scores, weights))
        
        is_pinch = confidence > 0.55
        
        return is_pinch, confidence, details
    
    def detect_natural_harmonic(
        self,
        features: SquealFeatures
    ) -> Tuple[bool, float, int, Dict]:
        """
        Detect natural harmonic (stricter for high-gain guitar).
        
        Characteristics:
        - Very pure tone (single dominant harmonic)
        - Long sustain  
        - Very low spectral flatness
        - Frequency matches specific harmonic ratios (REQUIRED in high-gain mode)
        - Much higher perceived pitch than fundamental
        """
        scores = []
        details = {}
        harmonic_fret = None
        
        # 0. First check: Must match a known harmonic fret in high-gain mode
        details['freq_ratio'] = features.freq_ratio
        matched_fret = None
        best_match = 0.0
        
        for fret, ratio in self.natural_harmonic_frets.items():
            error = abs(features.freq_ratio - ratio) / ratio
            if error < 0.08:  # Stricter: within 8%
                match_score = 1.0 - error
                if match_score > best_match:
                    best_match = match_score
                    matched_fret = fret
        
        details['matched_fret'] = matched_fret
        harmonic_fret = int(matched_fret) if matched_fret else None
        
        # In high-gain mode, require harmonic match
        if self.high_gain_mode and self.nh_harmonic_match_required and matched_fret is None:
            return False, 0.0, None, details
        
        scores.append(best_match)
        
        # 1. Harmonic purity (stricter threshold)
        details['purity'] = features.harmonic_purity
        if features.harmonic_purity > self.nh_purity_threshold:
            scores.append(1.0)
        elif features.harmonic_purity > self.nh_purity_threshold * 0.85:
            scores.append(0.4)
        else:
            scores.append(0.0)
        
        # 2. Spectral flatness (should be very low - much stricter for distorted)
        details['flatness'] = features.spectral_flatness
        if features.spectral_flatness < self.nh_flatness_max:
            scores.append(1.0)
        elif features.spectral_flatness < self.nh_flatness_max * 1.5:
            scores.append(0.3)
        else:
            scores.append(0.0)
        
        # 3. Sustain stability (stricter)
        details['sustain_stability'] = features.sustain_stability
        if features.sustain_stability > self.nh_sustain_threshold:
            scores.append(1.0)
        elif features.sustain_stability > self.nh_sustain_threshold * 0.8:
            scores.append(0.4)
        else:
            scores.append(0.0)
        
        # 4. Not a pinch harmonic (natural harmonics have cleaner attack)
        details['attack'] = features.attack_sharpness
        if features.attack_sharpness < 0.2:
            scores.append(0.9)
        elif features.attack_sharpness < 0.35:
            scores.append(0.4)
        else:
            scores.append(0.0)  # Too sharp attack - probably pinch harmonic
        
        # 5. Very high frequency ratio (must be 2x+ for natural harmonic)
        if features.freq_ratio >= 2.0:
            scores.append(0.8)
        elif features.freq_ratio >= 1.8:
            scores.append(0.3)
        else:
            scores.append(0.0)  # Not sounding like a harmonic
        
        confidence = np.mean(scores)
        
        # Higher threshold in high-gain mode
        threshold = 0.65 if self.high_gain_mode else 0.5
        is_natural = confidence > threshold
        
        return is_natural, confidence, harmonic_fret, details
    
    def detect_dive_bomb(
        self,
        features: SquealFeatures
    ) -> Tuple[bool, float, float, Dict]:
        """
        Detect dive bomb (whammy bar pitch drop).
        
        Characteristics:
        - Continuous pitch descent
        - Significant pitch drop (3+ semitones)
        - Relatively rapid descent
        """
        details = {}
        
        # 1. Check pitch slope (must be negative / descending)
        details['pitch_slope'] = features.pitch_slope
        
        if features.pitch_slope > self.dive_slope_threshold:
            # Not descending enough
            return False, 0.0, 0.0, details
        
        # 2. Calculate total pitch drop
        # Estimate from slope and duration
        total_drop = abs(features.pitch_slope) * features.duration
        details['total_drop'] = total_drop
        
        if total_drop < self.dive_min_drop:
            return False, 0.0, 0.0, details
        
        # 3. Check that it's continuous descent (few direction changes)
        details['direction_changes'] = features.pitch_direction_changes
        
        continuity_score = 1.0 - (features.pitch_direction_changes / 10.0)
        continuity_score = max(0.0, continuity_score)
        
        # 4. Calculate confidence
        slope_score = min(1.0, abs(features.pitch_slope) / 10.0)
        drop_score = min(1.0, total_drop / 12.0)  # Full octave = perfect score
        
        confidence = 0.3 * slope_score + 0.4 * drop_score + 0.3 * continuity_score
        
        is_dive = confidence > 0.5
        
        return is_dive, confidence, total_drop, details
    
    def detect_whammy_flutter(
        self,
        features: SquealFeatures
    ) -> Tuple[bool, float, Dict]:
        """
        Detect whammy bar flutter (rapid pitch oscillation).
        
        Characteristics:
        - Multiple direction changes
        - Moderate pitch variance
        - Rapid oscillation
        """
        details = {}
        
        # Flutter has many direction changes
        details['direction_changes'] = features.pitch_direction_changes
        details['pitch_variance'] = features.pitch_variance
        
        # Need significant movement
        if features.pitch_variance < 0.3:
            return False, 0.0, details
        
        # Calculate oscillation rate
        changes_per_second = features.pitch_direction_changes / features.duration
        details['changes_per_sec'] = changes_per_second
        
        # Flutter typically 3-8 changes per second
        if 3 <= changes_per_second <= 15:
            rate_score = 1.0
        elif 2 <= changes_per_second < 3 or 15 < changes_per_second <= 20:
            rate_score = 0.5
        else:
            rate_score = 0.0
        
        # Variance should be moderate (0.5-2 semitones)
        if 0.5 <= features.pitch_variance <= 2.0:
            var_score = 1.0
        elif 0.3 <= features.pitch_variance < 0.5 or 2.0 < features.pitch_variance <= 3.0:
            var_score = 0.5
        else:
            var_score = 0.0
        
        confidence = 0.6 * rate_score + 0.4 * var_score
        is_flutter = confidence > 0.5
        
        return is_flutter, confidence, details
    
    def detect(
        self,
        y: np.ndarray,
        time: float,
        duration: float,
        expected_freq: float
    ) -> SquealDetection:
        """
        Detect squeal techniques for a single note.
        
        Returns the most likely technique with confidence.
        """
        features = self.extract_features(y, time, duration, expected_freq)
        
        if features is None:
            return SquealDetection(
                time=time,
                duration=duration,
                technique=SquealTechnique.NORMAL,
                confidence=0.0,
                details={'error': 'Could not extract features'}
            )
        
        # Check each technique
        results = []
        
        # 1. Pinch harmonic
        is_ph, ph_conf, ph_details = self.detect_pinch_harmonic(features)
        if is_ph:
            results.append((SquealTechnique.PINCH_HARMONIC, ph_conf, None, 0.0, ph_details))
        
        # 2. Natural harmonic
        is_nh, nh_conf, nh_fret, nh_details = self.detect_natural_harmonic(features)
        if is_nh:
            results.append((SquealTechnique.NATURAL_HARMONIC, nh_conf, nh_fret, 0.0, nh_details))
        
        # 3. Dive bomb
        is_dive, dive_conf, dive_drop, dive_details = self.detect_dive_bomb(features)
        if is_dive:
            results.append((SquealTechnique.DIVE_BOMB, dive_conf, None, dive_drop, dive_details))
        
        # 4. Whammy flutter
        is_flutter, flutter_conf, flutter_details = self.detect_whammy_flutter(features)
        if is_flutter:
            results.append((SquealTechnique.WHAMMY_FLUTTER, flutter_conf, None, 0.0, flutter_details))
        
        # Find best match
        if not results:
            return SquealDetection(
                time=time,
                duration=duration,
                technique=SquealTechnique.NORMAL,
                confidence=1.0 - max(ph_conf, nh_conf, dive_conf if is_dive else 0, 
                                     flutter_conf if is_flutter else 0),
                details={
                    'features': {
                        'centroid_ratio': features.spectral_centroid_ratio,
                        'freq_ratio': features.freq_ratio,
                        'hf_energy': features.high_freq_energy_ratio,
                        'purity': features.harmonic_purity,
                        'pitch_slope': features.pitch_slope
                    }
                }
            )
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]
        
        return SquealDetection(
            time=time,
            duration=duration,
            technique=best[0],
            confidence=best[1],
            harmonic_fret=best[2],
            dive_semitones=best[3],
            details=best[4]
        )
    
    def analyze_audio(
        self,
        y: np.ndarray,
        notes: List[Dict],
        verbose: bool = True
    ) -> List[SquealDetection]:
        """
        Analyze audio and detect techniques for all notes.
        
        Args:
            y: Audio signal
            notes: List with {'time', 'duration', 'pitch' (MIDI or Hz)}
            verbose: Print progress
            
        Returns:
            List of SquealDetection for each note
        """
        if verbose:
            print("\n🎸 PINCH HARMONIC & SQUEAL DETECTION")
            print("=" * 45)
        
        results = []
        
        technique_counts = {t: 0 for t in SquealTechnique}
        
        for i, note in enumerate(notes):
            time = note.get('time', 0)
            duration = note.get('duration', 0.2)
            duration = max(duration, self.min_note_duration)
            
            pitch = note.get('pitch', 0)
            if pitch > 127:  # Hz
                expected_freq = pitch
            else:  # MIDI
                expected_freq = librosa.midi_to_hz(pitch)
            
            detection = self.detect(y, time, duration, expected_freq)
            results.append(detection)
            
            technique_counts[detection.technique] += 1
            
            if verbose and detection.technique != SquealTechnique.NORMAL:
                emoji = {
                    SquealTechnique.PINCH_HARMONIC: "🔊",
                    SquealTechnique.NATURAL_HARMONIC: "✨",
                    SquealTechnique.DIVE_BOMB: "📉",
                    SquealTechnique.WHAMMY_FLUTTER: "〰️",
                    SquealTechnique.WHAMMY_SCREAM: "📈",
                }.get(detection.technique, "🎵")
                
                print(f"  {emoji} Note {i} @ {time:.2f}s: {detection.technique.value} "
                      f"(conf: {detection.confidence:.0%})")
        
        if verbose:
            print("\n📊 DETECTION SUMMARY:")
            print(f"   Total notes:        {len(results)}")
            print(f"   Normal:             {technique_counts[SquealTechnique.NORMAL]}")
            print(f"   Pinch harmonics:    {technique_counts[SquealTechnique.PINCH_HARMONIC]}")
            print(f"   Natural harmonics:  {technique_counts[SquealTechnique.NATURAL_HARMONIC]}")
            print(f"   Dive bombs:         {technique_counts[SquealTechnique.DIVE_BOMB]}")
            print(f"   Whammy flutter:     {technique_counts[SquealTechnique.WHAMMY_FLUTTER]}")
        
        return results


def integrate_squeal_detection(
    notes: List[Dict],
    detections: List[SquealDetection]
) -> List[Dict]:
    """
    Integrate squeal detections back into note list.
    
    Adds 'technique' and 'technique_details' to each note.
    """
    for note, detection in zip(notes, detections):
        note['technique'] = detection.technique.value
        note['technique_confidence'] = detection.confidence
        note['harmonic_fret'] = detection.harmonic_fret
        note['dive_semitones'] = detection.dive_semitones
        note['technique_details'] = detection.details
    
    return notes


def format_tabs_with_squeals(
    notes: List[Dict],
    detections: List[SquealDetection],
    strings: int = 6
) -> str:
    """
    Format notes with squeal notations as ASCII tabs.
    """
    string_names = ['e', 'B', 'G', 'D', 'A', 'E'][:strings]
    
    lines = ["# Guitar Tab with Technique Annotations", ""]
    
    # Group by time for display
    for note, detection in zip(notes, detections):
        fret = note.get('fret', '?')
        string = note.get('string', 0)
        time = note.get('time', 0)
        
        notation = detection.to_tab_notation(fret)
        
        if detection.technique != SquealTechnique.NORMAL:
            lines.append(f"# {time:.2f}s: {notation} [{detection.technique.value}]")
    
    return '\n'.join(lines)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect pinch harmonics and squeals in guitar audio'
    )
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--notes-json', help='JSON file with detected notes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--output', '-o', help='Output JSON with detections')
    
    args = parser.parse_args()
    
    import json
    
    print(f"🎸 Loading audio: {args.audio_file}")
    y, sr = librosa.load(args.audio_file, sr=args.sr, mono=True)
    print(f"   Duration: {len(y)/sr:.2f}s")
    
    # Load or detect notes
    if args.notes_json:
        with open(args.notes_json) as f:
            notes = json.load(f)
    else:
        print("🎵 Detecting notes...")
        # Basic onset + pitch detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'),
            sr=sr
        )
        f0_times = librosa.times_like(f0, sr=sr)
        
        notes = []
        for i, onset in enumerate(onset_times):
            idx = np.argmin(np.abs(f0_times - onset))
            if f0[idx] and not np.isnan(f0[idx]):
                duration = (onset_times[i+1] - onset) if i + 1 < len(onset_times) else 0.3
                notes.append({
                    'time': float(onset),
                    'duration': float(min(duration, 0.5)),
                    'pitch': float(f0[idx])
                })
        
        print(f"   Found {len(notes)} notes")
    
    # Run detection
    detector = PinchHarmonicDetector(sr=sr)
    detections = detector.analyze_audio(y, notes, verbose=args.verbose)
    
    # Output results
    if args.output:
        output_data = []
        for note, det in zip(notes, detections):
            output_data.append({
                'time': det.time,
                'duration': det.duration,
                'pitch': note.get('pitch', 0),
                'technique': det.technique.value,
                'confidence': det.confidence,
                'harmonic_fret': det.harmonic_fret,
                'dive_semitones': det.dive_semitones
            })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Saved results to: {args.output}")
    
    return detections


if __name__ == "__main__":
    main()
