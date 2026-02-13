#!/usr/bin/env python3
"""
High-Gain Guitar Frequency Analysis and Distortion Auto-Detection

High-gain distorted guitar has distinctive spectral characteristics:
1. Boosted midrange (800Hz-3kHz) - the "crunch" zone
2. Reduced bass clarity (muddy low end, intermodulation)
3. Fizzy highs (4kHz+) - harsh harmonic content
4. Specific cabinet resonances (80-250Hz bump, 2-4kHz presence)

This module:
- Analyzes the spectral profile of input audio
- Auto-detects if it's distorted/high-gain
- Returns distortion confidence and type
- Applies appropriate detection strategy based on analysis
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import Enum


class DistortionType(Enum):
    """Classification of distortion types by spectral characteristics."""
    CLEAN = "clean"                    # No significant distortion
    LIGHT_OVERDRIVE = "light_od"       # Subtle breakup, tube warmth
    MEDIUM_DRIVE = "medium_drive"      # Classic rock crunch
    HIGH_GAIN = "high_gain"            # Modern metal, heavy distortion
    FUZZ = "fuzz"                      # Extreme clipping, fuzzy tone


@dataclass
class FrequencyBands:
    """Standard frequency bands for guitar analysis."""
    # Sub-bass (felt, not heard clearly)
    sub_bass: Tuple[float, float] = (20, 80)
    
    # Bass (guitar fundamental low notes)
    bass: Tuple[float, float] = (80, 250)
    
    # Low-mids (body, warmth)
    low_mids: Tuple[float, float] = (250, 500)
    
    # Mids (guitar fundamental upper notes, presence)
    mids: Tuple[float, float] = (500, 2000)
    
    # Upper-mids / presence (attack, bite, crunch zone)
    upper_mids: Tuple[float, float] = (2000, 4000)
    
    # Highs (sparkle, fizz in distortion)
    highs: Tuple[float, float] = (4000, 8000)
    
    # Air (extreme highs, usually rolled off)
    air: Tuple[float, float] = (8000, 20000)


@dataclass
class DistortionProfile:
    """Result of distortion analysis."""
    # Overall classification
    distortion_type: DistortionType = DistortionType.CLEAN
    distortion_confidence: float = 0.0
    
    # Band energy ratios (normalized to total energy)
    band_energies: Dict[str, float] = field(default_factory=dict)
    
    # Specific distortion indicators
    mid_boost_ratio: float = 0.0      # Ratio of 800Hz-3kHz to overall
    bass_clarity: float = 0.0          # How defined the bass is (0=muddy, 1=clear)
    fizz_level: float = 0.0            # Amount of harsh highs (4kHz+)
    harmonic_density: float = 0.0      # How many harmonics present
    crest_factor: float = 0.0          # Peak to RMS ratio (low = compressed)
    spectral_flatness: float = 0.0     # 0=tonal, 1=noise-like
    
    # Cabinet detection
    cabinet_resonance_detected: bool = False
    cabinet_resonance_freq: float = 0.0
    
    # Intermodulation artifacts
    intermod_level: float = 0.0
    
    # Recommendations
    recommended_preprocessing: str = "none"
    recommended_pitch_method: str = "pyin"
    
    def __str__(self) -> str:
        return (
            f"DistortionProfile(\n"
            f"  type={self.distortion_type.value}, confidence={self.distortion_confidence:.2%}\n"
            f"  mid_boost={self.mid_boost_ratio:.2f}, bass_clarity={self.bass_clarity:.2f}\n"
            f"  fizz={self.fizz_level:.2f}, harmonic_density={self.harmonic_density:.2f}\n"
            f"  crest_factor={self.crest_factor:.1f}dB, spectral_flatness={self.spectral_flatness:.3f}\n"
            f"  cabinet={self.cabinet_resonance_detected}, intermod={self.intermod_level:.2f}\n"
            f"  → preprocessing: {self.recommended_preprocessing}\n"
            f"  → pitch_method: {self.recommended_pitch_method}\n"
            f")"
        )


class HighGainFrequencyAnalyzer:
    """
    Analyzes audio to detect and characterize high-gain guitar distortion.
    
    Uses spectral analysis to identify:
    - Compressed dynamics (low crest factor)
    - Dense harmonic content
    - Characteristic EQ curves (mid-boost, fizzy highs)
    - Cabinet resonances
    - Intermodulation artifacts
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.bands = FrequencyBands()
        self.n_fft = 4096  # High resolution for frequency analysis
        self.hop_length = 512
        
    def analyze(self, y: np.ndarray, verbose: bool = True) -> DistortionProfile:
        """
        Analyze audio and detect distortion characteristics.
        
        Args:
            y: Audio signal (mono)
            verbose: Print analysis progress
            
        Returns:
            DistortionProfile with analysis results
        """
        if verbose:
            print("\n🔍 HIGH-GAIN FREQUENCY ANALYSIS")
            print("=" * 50)
        
        profile = DistortionProfile()
        
        # 1. Compute spectrum
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Time-averaged spectrum (power)
        S_mean = np.mean(S ** 2, axis=1)
        
        if verbose:
            print(f"📊 Analyzing {len(y)/self.sr:.2f}s of audio at {self.sr}Hz")
        
        # 2. Band energy analysis
        profile.band_energies = self._compute_band_energies(S_mean, freqs)
        
        # 3. Mid-boost ratio (key indicator of distortion)
        profile.mid_boost_ratio = self._compute_mid_boost_ratio(S_mean, freqs)
        
        # 4. Bass clarity analysis
        profile.bass_clarity = self._compute_bass_clarity(y, S_mean, freqs)
        
        # 5. Fizz level (harsh highs)
        profile.fizz_level = self._compute_fizz_level(S_mean, freqs)
        
        # 6. Harmonic density
        profile.harmonic_density = self._compute_harmonic_density(S_mean, freqs)
        
        # 7. Crest factor (dynamics compression indicator)
        profile.crest_factor = self._compute_crest_factor(y)
        
        # 8. Spectral flatness
        profile.spectral_flatness = float(np.mean(
            librosa.feature.spectral_flatness(S=S)
        ))
        
        # 9. Cabinet resonance detection
        cab_detected, cab_freq = self._detect_cabinet_resonance(S_mean, freqs)
        profile.cabinet_resonance_detected = cab_detected
        profile.cabinet_resonance_freq = cab_freq
        
        # 10. Intermodulation level
        profile.intermod_level = self._compute_intermod_level(S, freqs)
        
        # 11. Classify distortion type
        profile.distortion_type, profile.distortion_confidence = (
            self._classify_distortion(profile)
        )
        
        # 12. Set recommendations
        self._set_recommendations(profile)
        
        if verbose:
            self._print_analysis(profile)
        
        return profile
    
    def _compute_band_energies(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> Dict[str, float]:
        """Compute energy in each frequency band."""
        total_energy = np.sum(S_mean) + 1e-10
        energies = {}
        
        for band_name in ['sub_bass', 'bass', 'low_mids', 'mids', 'upper_mids', 'highs', 'air']:
            band_range = getattr(self.bands, band_name)
            mask = (freqs >= band_range[0]) & (freqs < band_range[1])
            band_energy = np.sum(S_mean[mask])
            energies[band_name] = band_energy / total_energy
        
        return energies
    
    def _compute_mid_boost_ratio(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """
        Compute ratio of mid-range energy (800Hz-3kHz) to overall.
        
        High-gain distortion typically has strong mid presence.
        Values > 0.35 indicate significant mid boost.
        """
        # Crunch/presence zone
        mid_mask = (freqs >= 800) & (freqs <= 3000)
        
        # Reference: everything below mids (80-800Hz)
        low_mask = (freqs >= 80) & (freqs < 800)
        
        mid_energy = np.sum(S_mean[mid_mask])
        low_energy = np.sum(S_mean[low_mask])
        
        if low_energy < 1e-10:
            return 0.0
        
        # Ratio normalized to expected clean guitar ratio (~0.5)
        ratio = mid_energy / (low_energy + mid_energy + 1e-10)
        
        # Scale so 0.5 = balanced, >0.5 = mid-boosted
        return ratio
    
    def _compute_bass_clarity(
        self,
        y: np.ndarray,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """
        Measure how defined/clear the bass frequencies are.
        
        Distortion causes intermodulation that muddies the bass.
        Returns 0-1 where 1 = crystal clear, 0 = very muddy.
        """
        # Isolate bass region
        bass_mask = (freqs >= 80) & (freqs <= 250)
        bass_spectrum = S_mean[bass_mask]
        
        if len(bass_spectrum) == 0 or np.max(bass_spectrum) < 1e-10:
            return 0.5
        
        # Clear bass has distinct peaks; muddy bass is flat/smeared
        bass_normalized = bass_spectrum / np.max(bass_spectrum)
        
        # Find peaks
        peaks, properties = find_peaks(bass_normalized, height=0.3, prominence=0.1)
        
        # Calculate spectral entropy in bass region (lower = clearer)
        bass_normalized = bass_normalized / (np.sum(bass_normalized) + 1e-10)
        entropy = -np.sum(bass_normalized * np.log(bass_normalized + 1e-10))
        max_entropy = np.log(len(bass_normalized))
        
        clarity_from_entropy = 1.0 - (entropy / (max_entropy + 1e-10))
        clarity_from_peaks = min(1.0, len(peaks) * 0.3)
        
        # Combine metrics
        clarity = 0.6 * clarity_from_entropy + 0.4 * clarity_from_peaks
        
        return float(np.clip(clarity, 0, 1))
    
    def _compute_fizz_level(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """
        Measure the amount of harsh high-frequency content (4kHz+).
        
        Distortion adds "fizz" - dense high-frequency harmonics.
        Returns 0-1 where 1 = extremely fizzy.
        """
        # Fizz zone: 4kHz to 8kHz
        fizz_mask = (freqs >= 4000) & (freqs <= 8000)
        
        # Reference: fundamental/low-mid zone
        reference_mask = (freqs >= 200) & (freqs <= 1000)
        
        fizz_energy = np.sum(S_mean[fizz_mask])
        reference_energy = np.sum(S_mean[reference_mask])
        
        if reference_energy < 1e-10:
            return 0.0
        
        # Ratio - clean guitar should have very little fizz
        ratio = fizz_energy / reference_energy
        
        # Scale: 0.1 ratio = 0.5 fizz level (moderate)
        # Clean guitar typically has ratio < 0.05
        # Heavy distortion can have ratio > 0.3
        fizz_level = np.clip(ratio / 0.2, 0, 1)
        
        return float(fizz_level)
    
    def _compute_harmonic_density(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """
        Measure how many harmonics are present in the spectrum.
        
        Distortion creates many additional harmonics.
        Returns 0-1 where 1 = extremely dense harmonics.
        """
        # Focus on region where harmonics are audible
        harm_mask = (freqs >= 100) & (freqs <= 5000)
        harm_spectrum = S_mean[harm_mask]
        harm_freqs = freqs[harm_mask]
        
        if len(harm_spectrum) == 0:
            return 0.0
        
        # Normalize
        harm_normalized = harm_spectrum / (np.max(harm_spectrum) + 1e-10)
        
        # Count significant peaks (harmonics appear as peaks)
        peaks, _ = find_peaks(harm_normalized, height=0.05, distance=5)
        
        # Density based on number of peaks per octave
        # Guitar range ~80Hz to 1200Hz is about 4 octaves
        # Typical harmonic series: 8-15 harmonics visible
        # Heavy distortion: 30+ harmonics visible
        
        expected_clean = 10  # Expected harmonics for clean guitar
        expected_distorted = 40  # Expected for heavy distortion
        
        density = (len(peaks) - expected_clean) / (expected_distorted - expected_clean)
        
        return float(np.clip(density, 0, 1))
    
    def _compute_crest_factor(self, y: np.ndarray) -> float:
        """
        Compute crest factor (peak to RMS ratio) in dB.
        
        Clean guitar: ~15-20 dB crest factor
        Compressed/distorted: ~6-12 dB crest factor
        Heavily limited: <6 dB
        """
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y ** 2))
        
        if rms < 1e-10:
            return 20.0  # Default to "clean" value
        
        crest_db = 20 * np.log10(peak / rms)
        
        return float(crest_db)
    
    def _detect_cabinet_resonance(
        self,
        S_mean: np.ndarray,
        freqs: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Detect guitar cabinet resonance.
        
        Typical cabinet resonances:
        - Closed back: 80-150Hz bump
        - Open back: 100-200Hz
        - Presence peak: 2-4kHz (speaker cone breakup)
        """
        # Look for prominent peak in cabinet resonance zone
        cab_mask = (freqs >= 80) & (freqs <= 250)
        cab_spectrum = S_mean[cab_mask]
        cab_freqs = freqs[cab_mask]
        
        if len(cab_spectrum) == 0:
            return False, 0.0
        
        # Smooth to find main resonance
        cab_smooth = gaussian_filter1d(cab_spectrum, sigma=2)
        
        # Find peaks
        peaks, properties = find_peaks(
            cab_smooth,
            height=np.max(cab_smooth) * 0.5,
            prominence=np.max(cab_smooth) * 0.2
        )
        
        if len(peaks) > 0:
            # Take the most prominent peak
            main_peak_idx = peaks[np.argmax(cab_smooth[peaks])]
            resonance_freq = cab_freqs[main_peak_idx]
            
            # Check if it's a true cabinet resonance (narrow, prominent)
            peak_prominence = properties.get('prominences', [0])[0]
            is_cabinet = peak_prominence > np.mean(cab_smooth) * 0.3
            
            return is_cabinet, float(resonance_freq)
        
        return False, 0.0
    
    def _compute_intermod_level(
        self,
        S: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """
        Estimate intermodulation distortion level.
        
        IMD creates sum/difference frequencies that fill in spectral gaps.
        High IMD = more "filled in" spectrum.
        """
        # Use spectral flatness variation over time
        # IMD causes spectrum to become more noise-like
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        
        # Also look at spectral contrast (IMD reduces contrast)
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr)
        mean_contrast = np.mean(contrast)
        
        # High flatness + low contrast = high IMD
        flatness_contribution = np.mean(flatness)
        contrast_contribution = 1.0 - (mean_contrast / 50.0)  # Normalize
        
        intermod = 0.5 * flatness_contribution + 0.5 * np.clip(contrast_contribution, 0, 1)
        
        return float(np.clip(intermod, 0, 1))
    
    def _classify_distortion(
        self,
        profile: DistortionProfile
    ) -> Tuple[DistortionType, float]:
        """
        Classify distortion type based on analyzed features.
        
        Uses a weighted scoring system based on spectral characteristics.
        """
        scores = {
            DistortionType.CLEAN: 0.0,
            DistortionType.LIGHT_OVERDRIVE: 0.0,
            DistortionType.MEDIUM_DRIVE: 0.0,
            DistortionType.HIGH_GAIN: 0.0,
            DistortionType.FUZZ: 0.0,
        }
        
        # --- Feature contributions ---
        
        # Crest factor (dynamics)
        crest = profile.crest_factor
        if crest > 15:
            scores[DistortionType.CLEAN] += 0.4
        elif crest > 12:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.3
            scores[DistortionType.CLEAN] += 0.1
        elif crest > 9:
            scores[DistortionType.MEDIUM_DRIVE] += 0.3
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.1
        elif crest > 6:
            scores[DistortionType.HIGH_GAIN] += 0.3
            scores[DistortionType.MEDIUM_DRIVE] += 0.1
        else:
            scores[DistortionType.FUZZ] += 0.3
            scores[DistortionType.HIGH_GAIN] += 0.1
        
        # Mid boost ratio
        mid_boost = profile.mid_boost_ratio
        if mid_boost < 0.35:
            scores[DistortionType.CLEAN] += 0.2
        elif mid_boost < 0.45:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.2
        elif mid_boost < 0.55:
            scores[DistortionType.MEDIUM_DRIVE] += 0.2
        else:
            scores[DistortionType.HIGH_GAIN] += 0.2
            scores[DistortionType.FUZZ] += 0.1
        
        # Fizz level
        fizz = profile.fizz_level
        if fizz < 0.15:
            scores[DistortionType.CLEAN] += 0.2
        elif fizz < 0.3:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.2
        elif fizz < 0.5:
            scores[DistortionType.MEDIUM_DRIVE] += 0.2
        elif fizz < 0.7:
            scores[DistortionType.HIGH_GAIN] += 0.2
        else:
            scores[DistortionType.FUZZ] += 0.3
        
        # Harmonic density
        density = profile.harmonic_density
        if density < 0.2:
            scores[DistortionType.CLEAN] += 0.15
        elif density < 0.4:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.15
        elif density < 0.6:
            scores[DistortionType.MEDIUM_DRIVE] += 0.15
        else:
            scores[DistortionType.HIGH_GAIN] += 0.15
            scores[DistortionType.FUZZ] += 0.1
        
        # Bass clarity (inverse - muddy = more distortion)
        bass_clarity = profile.bass_clarity
        if bass_clarity > 0.7:
            scores[DistortionType.CLEAN] += 0.1
        elif bass_clarity > 0.5:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.1
        elif bass_clarity > 0.3:
            scores[DistortionType.MEDIUM_DRIVE] += 0.1
        else:
            scores[DistortionType.HIGH_GAIN] += 0.1
            scores[DistortionType.FUZZ] += 0.05
        
        # Intermodulation level
        intermod = profile.intermod_level
        if intermod < 0.1:
            scores[DistortionType.CLEAN] += 0.1
        elif intermod < 0.25:
            scores[DistortionType.LIGHT_OVERDRIVE] += 0.1
        elif intermod < 0.4:
            scores[DistortionType.MEDIUM_DRIVE] += 0.1
        else:
            scores[DistortionType.HIGH_GAIN] += 0.1
            scores[DistortionType.FUZZ] += 0.1
        
        # Find winner
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Normalize confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0
        
        # Boost confidence if multiple indicators agree
        if best_score > 0.6:
            confidence = min(1.0, confidence * 1.2)
        
        return best_type, float(confidence)
    
    def _set_recommendations(self, profile: DistortionProfile) -> None:
        """Set preprocessing and pitch detection recommendations."""
        
        dist_type = profile.distortion_type
        
        if dist_type == DistortionType.CLEAN:
            profile.recommended_preprocessing = "none"
            profile.recommended_pitch_method = "pyin"
        
        elif dist_type == DistortionType.LIGHT_OVERDRIVE:
            profile.recommended_preprocessing = "light"
            profile.recommended_pitch_method = "pyin"
        
        elif dist_type == DistortionType.MEDIUM_DRIVE:
            profile.recommended_preprocessing = "standard"
            profile.recommended_pitch_method = "hps"  # Harmonic Product Spectrum
        
        elif dist_type == DistortionType.HIGH_GAIN:
            profile.recommended_preprocessing = "distortion"
            profile.recommended_pitch_method = "hps"
        
        elif dist_type == DistortionType.FUZZ:
            profile.recommended_preprocessing = "aggressive"
            profile.recommended_pitch_method = "autocorrelation"
    
    def _print_analysis(self, profile: DistortionProfile) -> None:
        """Print formatted analysis results."""
        
        type_emoji = {
            DistortionType.CLEAN: "🎸",
            DistortionType.LIGHT_OVERDRIVE: "🔥",
            DistortionType.MEDIUM_DRIVE: "⚡",
            DistortionType.HIGH_GAIN: "🤘",
            DistortionType.FUZZ: "💀",
        }
        
        emoji = type_emoji.get(profile.distortion_type, "🎵")
        
        print(f"\n{emoji} DETECTED: {profile.distortion_type.value.upper()}")
        print(f"   Confidence: {profile.distortion_confidence:.1%}")
        print()
        print("📊 FREQUENCY PROFILE:")
        print(f"   Mid boost (800-3kHz):  {profile.mid_boost_ratio:.2%}")
        print(f"   Bass clarity:          {profile.bass_clarity:.2%}")
        print(f"   Fizz level (4kHz+):    {profile.fizz_level:.2%}")
        print(f"   Harmonic density:      {profile.harmonic_density:.2%}")
        print()
        print("📈 DYNAMICS:")
        print(f"   Crest factor:          {profile.crest_factor:.1f} dB")
        print(f"   Spectral flatness:     {profile.spectral_flatness:.4f}")
        print(f"   Intermod level:        {profile.intermod_level:.2%}")
        print()
        
        if profile.cabinet_resonance_detected:
            print(f"🔊 Cabinet resonance detected @ {profile.cabinet_resonance_freq:.0f} Hz")
        
        print()
        print("📋 BAND ENERGIES:")
        for band, energy in profile.band_energies.items():
            bar = "█" * int(energy * 30)
            print(f"   {band:12}: {bar} {energy:.1%}")
        
        print()
        print("💡 RECOMMENDATIONS:")
        print(f"   Preprocessing: {profile.recommended_preprocessing}")
        print(f"   Pitch method:  {profile.recommended_pitch_method}")
        print("=" * 50)


def analyze_and_detect(
    audio_path: str,
    sr: int = 22050,
    verbose: bool = True
) -> DistortionProfile:
    """
    Convenience function to analyze audio file and detect distortion.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate to use
        verbose: Print analysis
        
    Returns:
        DistortionProfile with analysis results
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    analyzer = HighGainFrequencyAnalyzer(sr=sr)
    return analyzer.analyze(y, verbose=verbose)


def apply_adaptive_preprocessing(
    y: np.ndarray,
    sr: int,
    profile: DistortionProfile,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply adaptive preprocessing based on detected distortion profile.
    
    Args:
        y: Audio signal
        sr: Sample rate
        profile: Distortion profile from analyzer
        verbose: Print progress
        
    Returns:
        Preprocessed audio
    """
    from preprocessing import (
        PreprocessingConfig,
        DistortionPreprocessingConfig,
        preprocess_audio,
        preprocess_distortion
    )
    
    rec = profile.recommended_preprocessing
    
    if verbose:
        print(f"\n🔧 Applying '{rec}' preprocessing strategy")
    
    if rec == "none":
        return y
    
    elif rec == "light":
        # Light preprocessing for subtle overdrive
        config = PreprocessingConfig(
            enabled=True,
            noise_reduce=True,
            noise_reduce_strength=0.5,
            compress=False,  # Already compressed by tube
            highpass=True,
            highpass_freq=60,
            harmonic_enhance=True,
            harmonic_strength=0.2,
        )
        return preprocess_audio(y, sr, config, verbose=verbose)
    
    elif rec == "standard":
        # Standard preprocessing for medium crunch
        config = PreprocessingConfig(
            enabled=True,
            noise_reduce=True,
            noise_reduce_strength=0.8,
            compress=True,
            compress_threshold_db=-18,
            compress_ratio=3.0,
            highpass=True,
            highpass_freq=70,
            deess=True,
            deess_freq=4500,
            harmonic_enhance=True,
            harmonic_strength=0.3,
        )
        return preprocess_audio(y, sr, config, verbose=verbose)
    
    elif rec == "distortion":
        # Full distortion preprocessing
        config = DistortionPreprocessingConfig(
            enabled=True,
            compress=True,
            de_distort=True,
            de_distort_strength=0.5,
            lowpass=True,
            lowpass_freq=4000,
            fundamental_enhance=True,
            fundamental_strength=0.4,
            remove_intermod=True,
        )
        return preprocess_distortion(y, sr, config, verbose=verbose)
    
    elif rec == "aggressive":
        # Aggressive preprocessing for extreme fuzz
        config = DistortionPreprocessingConfig(
            enabled=True,
            compress=True,
            compress_threshold_db=-12,
            compress_ratio=8.0,
            de_distort=True,
            de_distort_strength=0.7,
            lowpass=True,
            lowpass_freq=3000,  # More aggressive
            fundamental_enhance=True,
            fundamental_strength=0.6,
            remove_intermod=True,
            intermod_threshold=0.1,  # More aggressive
        )
        return preprocess_distortion(y, sr, config, verbose=verbose)
    
    return y


def select_pitch_method(profile: DistortionProfile):
    """
    Select appropriate pitch detection method based on profile.
    
    Returns a function that can be called with (y, sr) to detect pitch.
    """
    method = profile.recommended_pitch_method
    
    if method == "pyin":
        def detect_pyin(y, sr):
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('E2'),
                fmax=librosa.note_to_hz('E6'),
                sr=sr
            )
            return f0, voiced_probs
        return detect_pyin
    
    elif method == "hps":
        def detect_hps(y, sr):
            from distortion_pitch import DistortionPitchDetector, DistortionConfig
            config = DistortionConfig(
                hps_harmonics=5,
                detect_intermod=True,
                verbose=False
            )
            detector = DistortionPitchDetector(sr=sr, config=config)
            f0, confidence, times = detector.detect_continuous(y)
            return f0, confidence
        return detect_hps
    
    elif method == "autocorrelation":
        def detect_autocorr(y, sr):
            # Use librosa's pitch estimation with modified parameters
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('D2'),  # Lower for fuzz
                fmax=librosa.note_to_hz('E5'),  # Not too high
                sr=sr,
                frame_length=4096,  # Longer for stability
                win_length=2048
            )
            return f0, voiced_probs
        return detect_autocorr
    
    # Default to pyin
    return lambda y, sr: librosa.pyin(y, fmin=80, fmax=1200, sr=sr)[:2]


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="High-Gain Guitar Frequency Analyzer"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--preprocess", "-p", action="store_true",
                       help="Apply recommended preprocessing")
    parser.add_argument("--detect-pitch", "-d", action="store_true",
                       help="Run pitch detection with recommended method")
    parser.add_argument("--output", "-o", help="Output preprocessed audio to file")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    # Analyze
    profile = analyze_and_detect(args.audio_path, sr=args.sr, verbose=not args.quiet)
    
    if args.preprocess or args.output or args.detect_pitch:
        import librosa
        import soundfile as sf
        
        y, sr = librosa.load(args.audio_path, sr=args.sr, mono=True)
        
        if args.preprocess or args.output:
            y_processed = apply_adaptive_preprocessing(y, sr, profile, verbose=not args.quiet)
            
            if args.output:
                sf.write(args.output, y_processed, sr)
                print(f"\n✅ Saved preprocessed audio to: {args.output}")
        else:
            y_processed = y
        
        if args.detect_pitch:
            print("\n🎵 PITCH DETECTION:")
            detect_fn = select_pitch_method(profile)
            f0, confidence = detect_fn(y_processed, sr)
            
            # Print summary
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                print(f"   Detected pitches: {len(valid_f0)}")
                print(f"   Frequency range: {np.min(valid_f0):.1f} - {np.max(valid_f0):.1f} Hz")
                print(f"   Mean confidence: {np.nanmean(confidence):.2%}")
