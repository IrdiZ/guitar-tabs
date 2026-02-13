#!/usr/bin/env python3
"""
Distortion-Aware Pitch Detection for Electric Guitar

Distorted guitar has DIFFERENT characteristics than clean:
1. Heavy harmonic content (overtones from clipping/saturation)
2. Compressed dynamics (less attack variation)
3. Fundamental often WEAKER than harmonics (especially 2nd and 3rd)
4. Intermodulation distortion (sum and difference frequencies)
5. Specific frequency emphasis (mids scooped or boosted depending on amp)

This module implements specialized pitch detection for distorted guitar using:
- Harmonic Product Spectrum (HPS) to find true fundamental
- Weighted harmonic analysis favoring lower partials
- Intermodulation distortion detection and filtering
- Distortion-specific preprocessing
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt, medfilt
from scipy.ndimage import maximum_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings


# Constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Guitar frequency range (extended low for drop tunings with distortion)
DISTORTED_GUITAR_MIN_HZ = 65   # Below drop tunings
DISTORTED_GUITAR_MAX_HZ = 1200  # Distortion adds harmonics, fundamentals rarely this high

# Power chord typical range (very common in distorted playing)
POWER_CHORD_MIN_HZ = 65
POWER_CHORD_MAX_HZ = 400


@dataclass
class DistortionConfig:
    """Configuration for distortion-aware pitch detection."""
    
    # Harmonic Product Spectrum settings
    hps_harmonics: int = 5          # Number of harmonics for HPS (5 is good for heavy distortion)
    hps_downsample_method: str = 'decimate'  # 'decimate' or 'interpolate'
    
    # Harmonic weighting (favor lower harmonics for distorted guitar)
    harmonic_weights: List[float] = field(default_factory=lambda: [
        1.0,    # Fundamental (H1) - often weak in distortion
        0.9,    # 2nd harmonic (H2) - often STRONGER than fundamental
        0.85,   # 3rd harmonic (H3) - also strong
        0.7,    # 4th harmonic (H4)
        0.5,    # 5th harmonic (H5)
    ])
    
    # Intermodulation detection
    detect_intermod: bool = True
    intermod_threshold: float = 0.3  # Threshold for IMD detection
    
    # Frequency domain settings
    n_fft: int = 4096               # Higher resolution for low frequencies
    hop_length: int = 512
    
    # Confidence thresholds
    min_confidence: float = 0.3
    fundamental_boost: float = 1.5   # Boost confidence when fundamental found via HPS
    
    # Preprocessing
    apply_compression_compensation: bool = True
    apply_mid_boost: bool = False    # Some amps scoop mids
    
    # Output
    verbose: bool = True


@dataclass 
class DistortionPitchResult:
    """Result from distortion-aware pitch detection."""
    time: float
    frequency: float
    midi_note: int
    confidence: float
    method: str
    
    # Distortion-specific info
    harmonic_structure: List[Tuple[float, float]] = field(default_factory=list)  # [(freq, magnitude), ...]
    detected_fundamental: float = 0.0
    hps_score: float = 0.0
    intermod_detected: List[float] = field(default_factory=list)
    is_power_chord: bool = False
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)


class HarmonicProductSpectrum:
    """
    Harmonic Product Spectrum for robust fundamental frequency detection.
    
    HPS works by:
    1. Taking the magnitude spectrum
    2. Creating downsampled versions (by factors of 2, 3, 4, ...)
    3. Multiplying all versions together
    4. The true fundamental will have peaks that align across all harmonics
    
    This is particularly effective for distorted guitar where the fundamental
    is often weaker than the 2nd or 3rd harmonic.
    """
    
    def __init__(self, n_harmonics: int = 5, sr: int = 22050, n_fft: int = 4096):
        self.n_harmonics = n_harmonics
        self.sr = sr
        self.n_fft = n_fft
        self.freq_resolution = sr / n_fft
        
    def compute(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute HPS from magnitude spectrum.
        
        Args:
            spectrum: Magnitude spectrum (single frame)
            
        Returns:
            (hps_spectrum, fundamental_frequency)
        """
        n_bins = len(spectrum)
        
        # Initialize HPS with original spectrum
        hps = spectrum.copy()
        
        # Downsample and multiply
        for h in range(2, self.n_harmonics + 1):
            # Downsample by factor h
            downsampled_len = n_bins // h
            if downsampled_len < 10:
                break
                
            # Decimate spectrum (take every h-th sample)
            downsampled = spectrum[::h][:downsampled_len]
            
            # Multiply into HPS (only up to length of downsampled)
            hps[:downsampled_len] *= downsampled
        
        # Normalize
        max_val = np.max(hps)
        if max_val > 0:
            hps = hps / max_val
        
        # Find fundamental (peak in HPS within guitar range)
        min_bin = int(DISTORTED_GUITAR_MIN_HZ / self.freq_resolution)
        max_bin = int(DISTORTED_GUITAR_MAX_HZ / self.freq_resolution)
        
        # Ensure valid range
        min_bin = max(1, min_bin)
        max_bin = min(len(hps) - 1, max_bin)
        
        if min_bin >= max_bin:
            return hps, 0.0
        
        search_region = hps[min_bin:max_bin]
        if len(search_region) == 0:
            return hps, 0.0
            
        peak_idx = np.argmax(search_region) + min_bin
        fundamental_freq = peak_idx * self.freq_resolution
        
        return hps, fundamental_freq
    
    def detect_frame(self, spectrum: np.ndarray, original_spectrum: np.ndarray = None) -> dict:
        """
        Detect fundamental frequency with confidence and harmonic analysis.
        
        Args:
            spectrum: Magnitude spectrum to analyze
            original_spectrum: Original (non-HPS) spectrum for harmonic extraction
            
        Returns:
            dict with frequency, confidence, harmonics, etc.
        """
        hps, fundamental = self.compute(spectrum)
        
        if fundamental <= 0:
            return {
                'frequency': 0,
                'confidence': 0,
                'harmonics': [],
                'hps_score': 0
            }
        
        # Calculate HPS score (how peaked is the result)
        fund_bin = int(fundamental / self.freq_resolution)
        if fund_bin > 0 and fund_bin < len(hps):
            # Score based on peak prominence
            local_region = hps[max(0, fund_bin-5):min(len(hps), fund_bin+6)]
            if len(local_region) > 0:
                hps_score = hps[fund_bin] / (np.mean(local_region) + 1e-10)
            else:
                hps_score = 0
        else:
            hps_score = 0
        
        # Extract harmonic structure from original spectrum
        harmonics = []
        if original_spectrum is not None:
            for h in range(1, self.n_harmonics + 1):
                harm_freq = fundamental * h
                harm_bin = int(harm_freq / self.freq_resolution)
                
                if 0 < harm_bin < len(original_spectrum):
                    # Get magnitude around expected harmonic
                    window = 3
                    start = max(0, harm_bin - window)
                    end = min(len(original_spectrum), harm_bin + window + 1)
                    harm_mag = np.max(original_spectrum[start:end])
                    harmonics.append((harm_freq, float(harm_mag)))
        
        # Confidence based on HPS score and harmonic coherence
        confidence = min(1.0, hps_score / 5.0)  # Normalize HPS score
        
        return {
            'frequency': fundamental,
            'confidence': confidence,
            'harmonics': harmonics,
            'hps_score': hps_score
        }


class DistortionPitchDetector:
    """
    Pitch detection optimized for distorted electric guitar.
    
    Combines:
    - Harmonic Product Spectrum for fundamental detection
    - Weighted harmonic analysis
    - Intermodulation distortion handling
    - Distortion-specific preprocessing
    """
    
    def __init__(self, sr: int = 22050, config: Optional[DistortionConfig] = None):
        self.sr = sr
        self.config = config or DistortionConfig()
        self.hps = HarmonicProductSpectrum(
            n_harmonics=self.config.hps_harmonics,
            sr=sr,
            n_fft=self.config.n_fft
        )
        
    def preprocess(self, y: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for distortion-aware detection.
        
        Distorted guitar has:
        - Compressed dynamics (apply expansion to recover transients)
        - Boosted harmonics (apply mild low-pass to reduce harmonic confusion)
        - Scooped mids on some amps (optionally boost mids)
        """
        processed = y.copy()
        
        # Mild low-pass to reduce high-frequency hash from heavy distortion
        # This helps HPS focus on musically relevant harmonics
        nyq = self.sr / 2
        cutoff = min(4000 / nyq, 0.95)  # 4kHz low-pass
        try:
            b, a = butter(2, cutoff, btype='low')
            processed = filtfilt(b, a, processed)
        except Exception:
            pass  # Skip if filter fails
        
        if self.config.apply_compression_compensation:
            # Apply mild expansion to help detect transients
            # Distortion compresses dynamics heavily
            envelope = np.abs(processed)
            envelope = maximum_filter1d(envelope, size=int(0.01 * self.sr))  # 10ms window
            
            # Gentle expansion (raise peaks, lower troughs)
            expansion_factor = 1.3
            envelope_normalized = envelope / (np.max(envelope) + 1e-10)
            expansion_curve = np.power(envelope_normalized, expansion_factor - 1)
            
            # Apply expansion
            processed = processed * (0.7 + 0.3 * expansion_curve)
        
        if self.config.apply_mid_boost:
            # Boost 400-2000 Hz range (guitar fundamental range)
            try:
                low_freq = 400 / nyq
                high_freq = min(2000 / nyq, 0.95)
                b, a = butter(2, [low_freq, high_freq], btype='band')
                mid_boost = filtfilt(b, a, y)
                processed = processed + 0.3 * mid_boost
            except Exception:
                pass
        
        return processed
    
    def detect_intermodulation(
        self, 
        spectrum: np.ndarray,
        fundamental: float,
        harmonics: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Detect intermodulation distortion products.
        
        IMD creates sum and difference frequencies:
        - If f1 and f2 are present, distortion creates f1+f2, f1-f2, 2f1-f2, etc.
        
        For power chords (root + fifth), IMD products can be confused with the root.
        """
        if not self.config.detect_intermod or fundamental <= 0:
            return []
        
        intermod_products = []
        freq_resolution = self.sr / self.config.n_fft
        
        # Get strong harmonic frequencies
        strong_harmonics = [h[0] for h in harmonics if h[1] > self.config.intermod_threshold]
        
        if len(strong_harmonics) < 2:
            return []
        
        # Check for common IMD products
        for i, f1 in enumerate(strong_harmonics):
            for f2 in strong_harmonics[i+1:]:
                # Sum and difference
                sum_freq = f1 + f2
                diff_freq = abs(f1 - f2)
                
                # Check if these frequencies have significant energy
                for imd_freq in [sum_freq, diff_freq]:
                    if DISTORTED_GUITAR_MIN_HZ <= imd_freq <= DISTORTED_GUITAR_MAX_HZ:
                        imd_bin = int(imd_freq / freq_resolution)
                        if 0 < imd_bin < len(spectrum):
                            if spectrum[imd_bin] > self.config.intermod_threshold * np.max(spectrum):
                                intermod_products.append(imd_freq)
        
        return list(set(intermod_products))
    
    def detect_power_chord(
        self,
        fundamental: float,
        harmonics: List[Tuple[float, float]],
        spectrum: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Detect if this is a power chord (root + fifth).
        
        Power chords are extremely common in distorted guitar.
        The fifth (ratio 1.5x) will be prominent alongside the root.
        
        Returns:
            (is_power_chord, root_frequency)
        """
        if fundamental <= 0:
            return False, fundamental
        
        freq_resolution = self.sr / self.config.n_fft
        fifth_ratio = 1.5  # Perfect fifth is 3:2 ratio
        
        # Expected fifth frequency
        fifth_freq = fundamental * fifth_ratio
        fifth_bin = int(fifth_freq / freq_resolution)
        
        if 0 < fifth_bin < len(spectrum):
            # Check if there's significant energy at the fifth
            window = 3
            fifth_region = spectrum[max(0, fifth_bin-window):min(len(spectrum), fifth_bin+window+1)]
            fund_bin = int(fundamental / freq_resolution)
            fund_region = spectrum[max(0, fund_bin-window):min(len(spectrum), fund_bin+window+1)]
            
            if len(fifth_region) > 0 and len(fund_region) > 0:
                fifth_mag = np.max(fifth_region)
                fund_mag = np.max(fund_region)
                
                # Power chord: fifth is prominent (at least 40% of fundamental energy)
                if fifth_mag > 0.4 * fund_mag:
                    return True, fundamental
        
        # Also check if what we think is fundamental might actually be the fifth
        # (HPS can sometimes lock onto the fifth in heavily distorted power chords)
        potential_root = fundamental / fifth_ratio
        root_bin = int(potential_root / freq_resolution)
        
        if root_bin > 0 and DISTORTED_GUITAR_MIN_HZ <= potential_root:
            root_region = spectrum[max(0, root_bin-3):min(len(spectrum), root_bin+4)]
            if len(root_region) > 0:
                root_mag = np.max(root_region)
                fund_bin = int(fundamental / freq_resolution)
                current_mag = spectrum[fund_bin] if 0 < fund_bin < len(spectrum) else 0
                
                # If the lower note has significant energy, it's probably the root
                if root_mag > 0.3 * current_mag:
                    return True, potential_root
        
        return False, fundamental
    
    def detect_frame(self, frame_spectrum: np.ndarray) -> DistortionPitchResult:
        """
        Detect pitch in a single frame using distortion-aware methods.
        """
        # Run HPS detection
        hps_result = self.hps.detect_frame(frame_spectrum, frame_spectrum)
        
        freq = hps_result['frequency']
        confidence = hps_result['confidence']
        harmonics = hps_result['harmonics']
        hps_score = hps_result['hps_score']
        
        if freq <= 0:
            return DistortionPitchResult(
                time=0,
                frequency=0,
                midi_note=0,
                confidence=0,
                method='distortion_hps',
                hps_score=0
            )
        
        # Check for power chord and correct fundamental if needed
        is_power_chord, corrected_freq = self.detect_power_chord(freq, harmonics, frame_spectrum)
        if is_power_chord and corrected_freq != freq:
            freq = corrected_freq
            # Recompute harmonics for corrected frequency
            harmonics = []
            for h in range(1, self.config.hps_harmonics + 1):
                harm_freq = freq * h
                harm_bin = int(harm_freq * self.config.n_fft / self.sr)
                if 0 < harm_bin < len(frame_spectrum):
                    harmonics.append((harm_freq, float(frame_spectrum[harm_bin])))
        
        # Detect intermodulation products
        intermod = self.detect_intermodulation(frame_spectrum, freq, harmonics)
        
        # Apply weighted harmonic scoring
        harmonic_score = self._compute_weighted_harmonic_score(harmonics)
        
        # Combine confidence scores
        final_confidence = (
            0.4 * confidence +           # HPS confidence
            0.3 * harmonic_score +        # Harmonic structure match
            0.3 * (hps_score / 10.0)      # HPS peakiness
        )
        final_confidence = min(1.0, max(0.0, final_confidence))
        
        # Boost confidence if good harmonic structure found
        if harmonic_score > 0.6:
            final_confidence *= self.config.fundamental_boost
            final_confidence = min(1.0, final_confidence)
        
        # Convert to MIDI
        midi_note = int(round(librosa.hz_to_midi(freq))) if freq > 0 else 0
        
        return DistortionPitchResult(
            time=0,  # Will be set by caller
            frequency=freq,
            midi_note=midi_note,
            confidence=final_confidence,
            method='distortion_hps',
            harmonic_structure=harmonics,
            detected_fundamental=freq,
            hps_score=hps_score,
            intermod_detected=intermod,
            is_power_chord=is_power_chord
        )
    
    def _compute_weighted_harmonic_score(
        self,
        harmonics: List[Tuple[float, float]]
    ) -> float:
        """
        Compute a score based on expected harmonic structure.
        
        For distorted guitar, we expect:
        - Strong 2nd and 3rd harmonics (from clipping)
        - Gradually decreasing higher harmonics
        - Odd harmonics often stronger than even (tube distortion)
        """
        if not harmonics or len(harmonics) < 2:
            return 0.0
        
        weights = self.config.harmonic_weights
        
        # Normalize harmonic magnitudes
        max_mag = max(h[1] for h in harmonics)
        if max_mag <= 0:
            return 0.0
        
        normalized_mags = [h[1] / max_mag for h in harmonics]
        
        # Compute weighted score
        score = 0.0
        total_weight = 0.0
        
        for i, mag in enumerate(normalized_mags):
            if i < len(weights):
                w = weights[i]
            else:
                w = 0.3  # Default weight for higher harmonics
            
            # Expect harmonics to be present (penalize missing harmonics)
            expected = 1.0 / (i + 1)  # Ideal: harmonics decrease
            match_score = 1.0 - abs(mag - expected) / (expected + 0.1)
            match_score = max(0, match_score)
            
            score += w * match_score
            total_weight += w
        
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def detect(
        self,
        y: np.ndarray,
        onset_frames: Optional[np.ndarray] = None
    ) -> List[DistortionPitchResult]:
        """
        Detect pitches in audio using distortion-aware methods.
        
        Args:
            y: Audio signal (mono)
            onset_frames: Optional pre-detected onset frames
            
        Returns:
            List of DistortionPitchResult
        """
        if self.config.verbose:
            print("\n🎸 Distortion-Aware Pitch Detection")
            print(f"   Using Harmonic Product Spectrum with {self.config.hps_harmonics} harmonics")
        
        # Preprocess for distortion
        y_processed = self.preprocess(y)
        
        # Compute STFT
        S = np.abs(librosa.stft(
            y_processed,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        ))
        
        n_frames = S.shape[1]
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sr,
            hop_length=self.config.hop_length
        )
        
        # Detect onset frames if not provided
        if onset_frames is None:
            onset_frames = librosa.onset.onset_detect(
                y=y_processed,
                sr=self.sr,
                hop_length=self.config.hop_length,
                backtrack=True,
                units='frames'
            )
        
        results = []
        
        # Process each onset
        for onset_frame in onset_frames:
            if onset_frame >= n_frames:
                continue
            
            # Average spectrum around onset for stability
            window_start = onset_frame
            window_end = min(onset_frame + 5, n_frames)
            frame_avg = np.mean(S[:, window_start:window_end], axis=1)
            
            # Detect pitch
            result = self.detect_frame(frame_avg)
            result.time = times[onset_frame]
            
            if result.confidence >= self.config.min_confidence:
                results.append(result)
        
        if self.config.verbose:
            n_power_chords = sum(1 for r in results if r.is_power_chord)
            n_intermod = sum(1 for r in results if r.intermod_detected)
            print(f"   Detected {len(results)} pitches")
            print(f"   Power chords detected: {n_power_chords}")
            print(f"   Frames with IMD artifacts: {n_intermod}")
        
        return results
    
    def detect_continuous(
        self,
        y: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuous frame-by-frame pitch detection.
        
        Returns:
            (f0, confidence, times) arrays like pYIN
        """
        y_processed = self.preprocess(y)
        
        S = np.abs(librosa.stft(
            y_processed,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        ))
        
        n_frames = S.shape[1]
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sr,
            hop_length=self.config.hop_length
        )
        
        f0 = np.zeros(n_frames)
        confidence = np.zeros(n_frames)
        
        for i in range(n_frames):
            result = self.detect_frame(S[:, i])
            if result.confidence >= confidence_threshold:
                f0[i] = result.frequency
                confidence[i] = result.confidence
        
        # Apply median filter for stability
        f0_filtered = self._median_filter_pitch(f0)
        
        return f0_filtered, confidence, times
    
    def _median_filter_pitch(
        self,
        f0: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """Apply median filter to pitch curve, preserving zeros."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Only filter valid pitches
        valid_mask = f0 > 0
        if not np.any(valid_mask):
            return f0
        
        filtered = medfilt(f0, kernel_size=kernel_size)
        
        # Preserve zero values
        result = np.where(valid_mask, filtered, f0)
        return result


def detect_pitches_distortion(
    y: np.ndarray,
    sr: int = 22050,
    config: Optional[DistortionConfig] = None,
    onset_frames: Optional[np.ndarray] = None
) -> List[DistortionPitchResult]:
    """
    Convenience function for distortion-aware pitch detection.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        config: DistortionConfig (optional)
        onset_frames: Pre-detected onset frames (optional)
        
    Returns:
        List of DistortionPitchResult
    """
    detector = DistortionPitchDetector(sr=sr, config=config)
    return detector.detect(y, onset_frames=onset_frames)


def convert_to_notes(
    results: List[DistortionPitchResult],
    min_duration: float = 0.05,
    audio_duration: Optional[float] = None
) -> List:
    """
    Convert DistortionPitchResult list to Note objects.
    
    This bridges the distortion detector with the main guitar_tabs pipeline.
    """
    from guitar_tabs import Note
    
    notes = []
    
    for i, result in enumerate(results):
        if result.frequency <= 0 or result.confidence <= 0:
            continue
        
        # Estimate duration
        if i < len(results) - 1:
            duration = results[i + 1].time - result.time
        elif audio_duration is not None:
            duration = audio_duration - result.time
        else:
            duration = 0.5  # Default duration
        
        if duration < min_duration:
            duration = min_duration
        
        notes.append(Note(
            midi=result.midi_note,
            start_time=result.time,
            duration=duration,
            confidence=result.confidence
        ))
    
    return notes


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Distortion-Aware Guitar Pitch Detection"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--hps-harmonics", type=int, default=5,
                       help="Number of harmonics for HPS")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                       help="Minimum confidence threshold")
    parser.add_argument("--no-intermod", action="store_true",
                       help="Disable intermodulation detection")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load audio
    print(f"Loading: {args.audio_path}")
    y, sr = librosa.load(args.audio_path, sr=22050, mono=True)
    print(f"Duration: {len(y)/sr:.2f}s")
    
    # Configure
    config = DistortionConfig(
        hps_harmonics=args.hps_harmonics,
        min_confidence=args.min_confidence,
        detect_intermod=not args.no_intermod,
        verbose=args.verbose
    )
    
    # Detect
    results = detect_pitches_distortion(y, sr, config=config)
    
    # Print results
    print(f"\n🎵 Detected {len(results)} notes:")
    print("-" * 70)
    print(f"{'Time':>8}  {'Note':>6}  {'Freq':>8}  {'Conf':>6}  {'HPS':>6}  {'Power':>6}")
    print("-" * 70)
    
    for r in results:
        power_str = "✓" if r.is_power_chord else ""
        print(
            f"{r.time:>8.3f}  "
            f"{r.name:>6}  "
            f"{r.frequency:>8.1f}  "
            f"{r.confidence:>6.3f}  "
            f"{r.hps_score:>6.2f}  "
            f"{power_str:>6}"
        )
    
    print("-" * 70)
