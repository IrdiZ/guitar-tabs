#!/usr/bin/env python3
"""
Attack Transient Pitch Detection for Distorted Guitar

The key insight: For distorted guitar, the ATTACK transient (first 20-50ms)
contains the clearest pitch information. After that:
- Distortion creates strong harmonics
- Compression muddies the fundamental
- Intermodulation artifacts appear
- The original pitch becomes obscured

This module implements attack-only pitch detection:
1. Detect note onsets with high precision
2. Extract ONLY the first 20-50ms after each onset
3. Run pitch detection on attack windows only
4. Ignore the sustained portion completely for pitch detection

Theory:
- Attack transient has the "pick click" which is broadband noise
- But also has the initial string excitation before saturation kicks in
- First few wave cycles are cleanest
- Compression/limiting in amp hasn't engaged yet
- Harmonics haven't built up from distortion
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt, correlate, medfilt
from scipy.ndimage import maximum_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings

# Constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GUITAR_MIN_HZ = 70    # Extended for drop tunings
GUITAR_MAX_HZ = 1200  # Upper limit


@dataclass
class AttackConfig:
    """Configuration for attack transient analysis."""
    
    # Attack window parameters
    attack_start_ms: float = 5      # Skip first 5ms (pick noise)
    attack_end_ms: float = 50       # Analyze up to 50ms
    optimal_window_ms: float = 30   # Ideal window for pitch detection
    
    # Onset detection
    onset_threshold: float = 0.35
    min_onset_gap_ms: float = 40
    
    # Pitch detection settings  
    use_autocorrelation: bool = True
    use_yin: bool = True
    use_cqt: bool = True
    use_hps: bool = True            # Harmonic Product Spectrum
    
    # Filtering
    highpass_hz: float = 70         # Remove low-freq rumble
    lowpass_hz: float = 1500        # Remove high-freq noise
    
    # Confidence
    min_confidence: float = 0.3
    require_agreement: int = 2      # Minimum methods that must agree
    
    # Output
    sr: int = 22050
    hop_length: int = 256
    n_fft: int = 2048
    verbose: bool = True


@dataclass
class AttackPitchResult:
    """Result from attack-based pitch detection."""
    onset_time: float
    pitch_hz: float
    midi_note: int
    confidence: float
    attack_duration_ms: float
    methods_agreed: List[str]
    
    # Per-method results
    autocorr_hz: float = 0.0
    yin_hz: float = 0.0
    cqt_hz: float = 0.0
    hps_hz: float = 0.0
    
    # Quality metrics
    attack_clarity: float = 0.0     # How clean the attack is
    transient_strength: float = 0.0 # Strength of the attack
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)


class AttackTransientPitchDetector:
    """
    Pitch detection focused ONLY on attack transients.
    
    The attack is the cleanest part of a distorted note because:
    1. Distortion hasn't fully engaged (compression takes time)
    2. Harmonics haven't built up yet
    3. The initial string vibration is most periodic
    4. Before any intermodulation artifacts
    
    Process:
    1. Detect precise onsets using energy + HFC
    2. Extract attack window (skip pick noise, take 20-50ms)
    3. Run multiple pitch detectors on attack only
    4. Consensus voting for final pitch
    """
    
    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
        self.sr = self.config.sr
        
    def detect(self, y: np.ndarray, sr: Optional[int] = None) -> List[AttackPitchResult]:
        """
        Main entry point: detect pitches from attack transients only.
        
        Args:
            y: Audio signal
            sr: Sample rate (uses config if not provided)
            
        Returns:
            List of AttackPitchResult with pitch from attack only
        """
        if sr is not None and sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        
        if self.config.verbose:
            print("🎸 Attack Transient Pitch Detection")
            print(f"   Window: {self.config.attack_start_ms}-{self.config.attack_end_ms}ms")
        
        # Step 1: Bandpass filter for guitar range
        y_filtered = self._bandpass_filter(y)
        
        # Step 2: Detect onsets with high precision
        onsets = self._detect_onsets(y)
        if self.config.verbose:
            print(f"   Found {len(onsets)} onsets")
        
        # Step 3: Extract and analyze attack for each onset
        results = []
        for i, onset_time in enumerate(onsets):
            result = self._analyze_attack(y_filtered, onset_time, i)
            if result is not None:
                results.append(result)
        
        if self.config.verbose:
            print(f"   Detected {len(results)} pitches from attacks")
            
        return results
    
    def _bandpass_filter(self, y: np.ndarray) -> np.ndarray:
        """Apply bandpass filter for guitar frequency range."""
        nyq = self.sr / 2
        low = self.config.highpass_hz / nyq
        high = min(self.config.lowpass_hz / nyq, 0.99)
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, y)
    
    def _detect_onsets(self, y: np.ndarray) -> List[float]:
        """
        Detect precise onset times using multiple methods.
        
        Combines:
        - Spectral flux (overall energy change)
        - HFC (High Frequency Content - emphasizes transients)
        - Energy envelope attack detection
        """
        hop = self.config.hop_length
        
        # Spectral flux
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, 
            hop_length=hop,
            aggregate=np.median
        )
        
        # HFC - emphasizes attack transients
        S = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.config.n_fft)
        weights = freqs / freqs.max() if freqs.max() > 0 else np.ones_like(freqs)
        hfc = np.sum(S * weights[:, np.newaxis], axis=0)
        hfc_diff = np.diff(hfc, prepend=0)
        hfc_diff = np.maximum(0, hfc_diff)
        if hfc_diff.max() > 0:
            hfc_diff = hfc_diff / hfc_diff.max()
        
        # Combine onset functions
        n_frames = min(len(onset_env), len(hfc_diff))
        combined = 0.5 * onset_env[:n_frames] + 0.5 * hfc_diff[:n_frames]
        if combined.max() > 0:
            combined = combined / combined.max()
        
        # Peak picking
        min_frames = int(self.config.min_onset_gap_ms / 1000 * self.sr / hop)
        peaks, _ = find_peaks(
            combined,
            height=self.config.onset_threshold,
            distance=max(1, min_frames)
        )
        
        # Convert to times and backtrack to true onset
        onset_times = []
        for peak in peaks:
            time = librosa.frames_to_time(peak, sr=self.sr, hop_length=hop)
            # Backtrack to find true onset start (before the peak)
            backtrack_frames = 3
            true_frame = max(0, peak - backtrack_frames)
            true_time = librosa.frames_to_time(true_frame, sr=self.sr, hop_length=hop)
            onset_times.append(true_time)
        
        return onset_times
    
    def _analyze_attack(
        self, 
        y: np.ndarray, 
        onset_time: float,
        note_idx: int
    ) -> Optional[AttackPitchResult]:
        """
        Analyze the attack transient of a single note.
        
        Extracts just the attack window and runs multiple pitch detectors.
        """
        # Calculate sample boundaries
        start_sample = int((onset_time + self.config.attack_start_ms / 1000) * self.sr)
        end_sample = int((onset_time + self.config.attack_end_ms / 1000) * self.sr)
        
        # Ensure bounds
        if start_sample >= len(y):
            return None
        end_sample = min(end_sample, len(y))
        
        if end_sample - start_sample < int(0.010 * self.sr):  # Need at least 10ms
            return None
        
        # Extract attack window
        attack = y[start_sample:end_sample]
        attack_duration_ms = (end_sample - start_sample) / self.sr * 1000
        
        # Measure attack quality
        transient_strength = self._measure_transient_strength(y, onset_time)
        attack_clarity = self._measure_clarity(attack)
        
        # Run pitch detectors on attack only
        pitch_estimates = {}
        
        if self.config.use_autocorrelation:
            f0 = self._pitch_autocorrelation(attack)
            if f0 is not None:
                pitch_estimates['autocorr'] = f0
        
        if self.config.use_yin:
            f0 = self._pitch_yin(attack)
            if f0 is not None:
                pitch_estimates['yin'] = f0
        
        if self.config.use_cqt:
            f0 = self._pitch_cqt(attack)
            if f0 is not None:
                pitch_estimates['cqt'] = f0
        
        if self.config.use_hps:
            f0 = self._pitch_hps(attack)
            if f0 is not None:
                pitch_estimates['hps'] = f0
        
        if not pitch_estimates:
            return None
        
        # Consensus voting - find agreement
        final_hz, agreed_methods, confidence = self._consensus_vote(pitch_estimates)
        
        if final_hz is None or len(agreed_methods) < self.config.require_agreement:
            # Not enough agreement - try with just HPS + YIN (best for distortion)
            if 'hps' in pitch_estimates and 'yin' in pitch_estimates:
                hps_midi = int(round(librosa.hz_to_midi(pitch_estimates['hps'])))
                yin_midi = int(round(librosa.hz_to_midi(pitch_estimates['yin'])))
                if abs(hps_midi - yin_midi) <= 1:
                    # Close enough - use HPS (better for distortion)
                    final_hz = pitch_estimates['hps']
                    agreed_methods = ['hps', 'yin']
                    confidence = 0.6
            elif 'hps' in pitch_estimates:
                final_hz = pitch_estimates['hps']
                agreed_methods = ['hps']
                confidence = 0.4
        
        if final_hz is None:
            return None
        
        midi_note = int(round(librosa.hz_to_midi(final_hz)))
        
        # Clamp to guitar range
        if midi_note < 28 or midi_note > 88:
            return None
        
        return AttackPitchResult(
            onset_time=onset_time,
            pitch_hz=float(final_hz),
            midi_note=midi_note,
            confidence=confidence,
            attack_duration_ms=attack_duration_ms,
            methods_agreed=agreed_methods,
            autocorr_hz=pitch_estimates.get('autocorr', 0.0),
            yin_hz=pitch_estimates.get('yin', 0.0),
            cqt_hz=pitch_estimates.get('cqt', 0.0),
            hps_hz=pitch_estimates.get('hps', 0.0),
            attack_clarity=attack_clarity,
            transient_strength=transient_strength
        )
    
    def _measure_transient_strength(self, y: np.ndarray, onset_time: float) -> float:
        """Measure how strong the attack transient is."""
        onset_sample = int(onset_time * self.sr)
        
        # Compare attack energy to average energy
        attack_window = 512  # ~23ms at 22050
        pre_window = max(0, onset_sample - attack_window)
        post_window = min(len(y), onset_sample + attack_window)
        
        pre_energy = np.sqrt(np.mean(y[pre_window:onset_sample]**2)) if onset_sample > pre_window else 0
        attack_energy = np.sqrt(np.mean(y[onset_sample:post_window]**2))
        
        if pre_energy > 0:
            return min(attack_energy / pre_energy, 10.0) / 10.0
        else:
            return min(attack_energy * 100, 1.0)
    
    def _measure_clarity(self, attack: np.ndarray) -> float:
        """
        Measure how clear/periodic the attack is.
        
        Higher clarity = better pitch estimation.
        Uses autocorrelation peak strength.
        """
        if len(attack) < 100:
            return 0.0
        
        # Normalize
        attack = attack - np.mean(attack)
        if np.std(attack) > 0:
            attack = attack / np.std(attack)
        
        # Autocorrelation
        corr = correlate(attack, attack, mode='full')
        corr = corr[len(corr)//2:]  # Take positive lags only
        
        # Find first significant peak after lag 0
        min_lag = int(self.sr / GUITAR_MAX_HZ)
        max_lag = int(self.sr / GUITAR_MIN_HZ)
        
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0
        
        peak_val = np.max(search)
        zero_lag = corr[0] if corr[0] > 0 else 1.0
        
        clarity = peak_val / zero_lag
        return float(np.clip(clarity, 0, 1))
    
    def _pitch_autocorrelation(self, attack: np.ndarray) -> Optional[float]:
        """
        Pitch detection via autocorrelation.
        
        Classic method - works well for clean transients.
        """
        if len(attack) < 200:
            return None
        
        # Center and normalize
        attack = attack - np.mean(attack)
        if np.std(attack) < 1e-6:
            return None
        attack = attack / np.std(attack)
        
        # Autocorrelation
        corr = correlate(attack, attack, mode='full')
        corr = corr[len(corr)//2:]
        
        # Search range for guitar
        min_lag = int(self.sr / GUITAR_MAX_HZ)
        max_lag = min(int(self.sr / GUITAR_MIN_HZ), len(corr) - 1)
        
        if max_lag <= min_lag:
            return None
        
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return None
        
        # Find peak
        peak_idx = np.argmax(search)
        peak_lag = min_lag + peak_idx
        
        # Parabolic interpolation for sub-sample accuracy
        if peak_idx > 0 and peak_idx < len(search) - 1:
            y0, y1, y2 = search[peak_idx-1], search[peak_idx], search[peak_idx+1]
            if (2*y1 - y0 - y2) != 0:
                delta = (y0 - y2) / (2 * (2*y1 - y0 - y2))
                peak_lag = min_lag + peak_idx + delta
        
        if peak_lag > 0:
            f0 = self.sr / peak_lag
            if GUITAR_MIN_HZ <= f0 <= GUITAR_MAX_HZ:
                return float(f0)
        
        return None
    
    def _pitch_yin(self, attack: np.ndarray) -> Optional[float]:
        """
        YIN pitch detection algorithm.
        
        Very robust for monophonic signals, especially with some noise.
        """
        if len(attack) < 256:
            return None
        
        # YIN parameters
        threshold = 0.15
        
        # Difference function
        W = len(attack) // 2
        if W < 50:
            return None
            
        diff = np.zeros(W)
        
        for tau in range(1, W):
            diff[tau] = np.sum((attack[:W] - attack[tau:tau+W])**2)
        
        # Cumulative mean normalized difference
        cumsum = np.cumsum(diff)
        cmndf = np.zeros(W)
        cmndf[0] = 1
        for tau in range(1, W):
            if cumsum[tau] != 0:
                cmndf[tau] = diff[tau] * tau / cumsum[tau]
            else:
                cmndf[tau] = 1
        
        # Find first minimum below threshold
        min_lag = int(self.sr / GUITAR_MAX_HZ)
        max_lag = min(int(self.sr / GUITAR_MIN_HZ), W - 1)
        
        for tau in range(min_lag, max_lag):
            if cmndf[tau] < threshold:
                # Check if it's a local minimum
                if tau > 0 and tau < W - 1:
                    if cmndf[tau] <= cmndf[tau-1] and cmndf[tau] <= cmndf[tau+1]:
                        # Parabolic interpolation
                        y0, y1, y2 = cmndf[tau-1], cmndf[tau], cmndf[tau+1]
                        if (y0 + y2 - 2*y1) != 0:
                            delta = (y0 - y2) / (2 * (y0 + y2 - 2*y1))
                            tau = tau + delta
                        
                        f0 = self.sr / tau
                        if GUITAR_MIN_HZ <= f0 <= GUITAR_MAX_HZ:
                            return float(f0)
        
        # Fallback: find global minimum in range
        search_range = cmndf[min_lag:max_lag]
        if len(search_range) > 0:
            min_idx = np.argmin(search_range)
            if search_range[min_idx] < 0.5:  # Reasonable threshold
                tau = min_lag + min_idx
                f0 = self.sr / tau
                if GUITAR_MIN_HZ <= f0 <= GUITAR_MAX_HZ:
                    return float(f0)
        
        return None
    
    def _pitch_cqt(self, attack: np.ndarray) -> Optional[float]:
        """
        CQT-based pitch detection.
        
        Good for capturing harmonic structure even in short windows.
        """
        if len(attack) < 512:
            return None
        
        # Use CQT for better frequency resolution at low frequencies
        try:
            C = np.abs(librosa.cqt(
                attack, 
                sr=self.sr,
                hop_length=len(attack),  # Single frame
                fmin=GUITAR_MIN_HZ,
                n_bins=60,
                bins_per_octave=12
            ))
        except:
            return None
        
        if C.shape[1] == 0:
            return None
        
        # Get magnitudes for the single frame
        mags = C[:, 0]
        
        # Find peak
        peak_bin = np.argmax(mags)
        
        # Convert bin to frequency
        fmin = GUITAR_MIN_HZ
        f0 = fmin * (2 ** (peak_bin / 12))
        
        if GUITAR_MIN_HZ <= f0 <= GUITAR_MAX_HZ:
            return float(f0)
        
        return None
    
    def _pitch_hps(self, attack: np.ndarray) -> Optional[float]:
        """
        Harmonic Product Spectrum pitch detection.
        
        BEST for distorted guitar - finds true fundamental even when
        harmonics are stronger than fundamental (common in distortion).
        
        Works by multiplying downsampled spectra, so all harmonics 
        "vote" for the fundamental.
        """
        if len(attack) < 512:
            return None
        
        # Zero-pad for better resolution
        n_fft = 4096
        padded = np.zeros(n_fft)
        padded[:len(attack)] = attack * np.hanning(len(attack))
        
        # FFT
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(n_fft, 1/self.sr)
        
        # Limit to guitar range
        min_bin = int(GUITAR_MIN_HZ * n_fft / self.sr)
        max_bin = int(GUITAR_MAX_HZ * n_fft / self.sr)
        max_bin = min(max_bin, len(spectrum) - 1)
        
        if max_bin <= min_bin:
            return None
        
        # Number of harmonics for HPS
        n_harmonics = 5
        
        # Harmonic Product Spectrum
        hps = spectrum[:max_bin].copy()
        
        for h in range(2, n_harmonics + 1):
            # Downsample by factor h
            decimated_len = max_bin // h
            if decimated_len < min_bin:
                break
            
            # Take every h-th sample (decimation)
            decimated = spectrum[:decimated_len * h:h][:decimated_len]
            
            # Multiply into HPS
            hps[:decimated_len] *= decimated
        
        # Find peak in guitar range
        search_range = hps[min_bin:max_bin]
        if len(search_range) == 0:
            return None
        
        peak_idx = np.argmax(search_range)
        peak_bin = min_bin + peak_idx
        
        # Parabolic interpolation
        if peak_idx > 0 and peak_idx < len(search_range) - 1:
            y0, y1, y2 = search_range[peak_idx-1], search_range[peak_idx], search_range[peak_idx+1]
            if (2*y1 - y0 - y2) != 0:
                delta = (y0 - y2) / (2 * (2*y1 - y0 - y2))
                peak_bin = min_bin + peak_idx + delta
        
        f0 = freqs[int(peak_bin)] if int(peak_bin) < len(freqs) else freqs[-1]
        
        # Fallback to linear interpolation if we have decimal bin
        if isinstance(peak_bin, float):
            lower_bin = int(peak_bin)
            upper_bin = min(lower_bin + 1, len(freqs) - 1)
            frac = peak_bin - lower_bin
            f0 = freqs[lower_bin] * (1 - frac) + freqs[upper_bin] * frac
        
        if GUITAR_MIN_HZ <= f0 <= GUITAR_MAX_HZ:
            return float(f0)
        
        return None
    
    def _consensus_vote(
        self, 
        pitch_estimates: Dict[str, float]
    ) -> Tuple[Optional[float], List[str], float]:
        """
        Find consensus among multiple pitch estimates.
        
        Groups estimates by MIDI note and picks the most agreed-upon pitch.
        """
        if not pitch_estimates:
            return None, [], 0.0
        
        # Convert to MIDI for comparison (more tolerant than Hz)
        midi_estimates = {}
        for method, hz in pitch_estimates.items():
            midi = int(round(librosa.hz_to_midi(hz)))
            midi_estimates[method] = (midi, hz)
        
        # Group by MIDI note
        groups: Dict[int, List[Tuple[str, float]]] = {}
        for method, (midi, hz) in midi_estimates.items():
            if midi not in groups:
                groups[midi] = []
            groups[midi].append((method, hz))
        
        # Find largest group
        best_midi = max(groups.keys(), key=lambda m: len(groups[m]))
        best_group = groups[best_midi]
        
        # Check for octave errors - if HPS says one thing but others say 
        # something an octave apart, prefer HPS (it handles harmonics better)
        if 'hps' in pitch_estimates:
            hps_midi, hps_hz = midi_estimates['hps']
            for other_midi in groups.keys():
                # Check for octave relationship
                diff = abs(hps_midi - other_midi)
                if diff == 12 and len(groups[other_midi]) >= len(groups[hps_midi]):
                    # Octave confusion - HPS likely has true fundamental
                    # Prefer the lower note (more likely fundamental)
                    if hps_midi < other_midi:
                        best_midi = hps_midi
                        best_group = groups[hps_midi]
        
        methods = [m for m, _ in best_group]
        hz_values = [hz for _, hz in best_group]
        avg_hz = np.mean(hz_values)
        
        # Confidence based on agreement
        agreement_ratio = len(best_group) / len(pitch_estimates)
        confidence = 0.3 + 0.7 * agreement_ratio
        
        return avg_hz, methods, confidence


def detect_from_attacks(
    audio_path: str,
    config: Optional[AttackConfig] = None
) -> List[AttackPitchResult]:
    """
    Convenience function to detect pitches from attack transients.
    
    Args:
        audio_path: Path to audio file
        config: Optional configuration
        
    Returns:
        List of detected notes from attack transients
    """
    config = config or AttackConfig()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sr, mono=True)
    
    # Run detector
    detector = AttackTransientPitchDetector(config)
    results = detector.detect(y)
    
    return results


def format_as_tabs(results: List[AttackPitchResult]) -> str:
    """Format results as simple guitar tablature."""
    if not results:
        return "No notes detected"
    
    lines = []
    lines.append("Attack-Based Guitar Tabs")
    lines.append("=" * 40)
    
    for r in results:
        lines.append(
            f"{r.onset_time:6.2f}s: {r.note_name:4s} "
            f"({r.pitch_hz:6.1f} Hz) "
            f"conf={r.confidence:.2f} "
            f"methods={','.join(r.methods_agreed)}"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python attack_transient_pitch.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"\n{'='*60}")
    print("ATTACK TRANSIENT PITCH DETECTION")
    print(f"{'='*60}\n")
    
    config = AttackConfig(
        verbose=True,
        attack_start_ms=5,
        attack_end_ms=50
    )
    
    results = detect_from_attacks(audio_file, config)
    
    print("\n" + format_as_tabs(results))
    
    # Summary statistics
    if results:
        print(f"\n{'='*40}")
        print("Summary:")
        print(f"  Total notes: {len(results)}")
        avg_conf = np.mean([r.confidence for r in results])
        print(f"  Avg confidence: {avg_conf:.2f}")
        
        # Method usage
        method_counts = {}
        for r in results:
            for m in r.methods_agreed:
                method_counts[m] = method_counts.get(m, 0) + 1
        print(f"  Method contributions:")
        for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {m}: {c}")
