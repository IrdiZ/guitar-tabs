#!/usr/bin/env python3
"""
YIN Pitch Detection Algorithm

Implementation of the YIN algorithm (de Cheveigné & Kawahara, 2002) with enhancements:

1. Difference function instead of autocorrelation
2. Cumulative Mean Normalized Difference Function (CMNDF) 
3. Absolute threshold for aperiodicity detection
4. Parabolic interpolation for sub-sample accuracy
5. Best local estimate to handle octave errors

YIN is specifically designed for monophonic pitch detection and is excellent
for instruments like guitar where autocorrelation-based methods suffer from
octave errors due to strong harmonics.

Key advantages over basic autocorrelation:
- CMNDF normalizes by the cumulative mean, preventing false peaks at low lags
- The threshold mechanism rejects aperiodic frames
- Parabolic interpolation gives sub-sample accuracy for precise pitch
- Local minimum search avoids octave errors

Reference:
    de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency
    estimator for speech and music. The Journal of the Acoustical Society
    of America, 111(4), 1917-1930.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import librosa


@dataclass
class YinConfig:
    """Configuration for YIN pitch detection."""
    # Frequency range
    fmin: float = 75.0      # Minimum frequency (Hz) - guitar low E is ~82Hz
    fmax: float = 1400.0    # Maximum frequency (Hz) - guitar high frets
    
    # YIN parameters
    threshold: float = 0.1  # Aperiodicity threshold (lower = stricter)
    
    # Frame parameters  
    frame_length: int = 2048    # Analysis window size
    hop_length: int = 512       # Hop between frames
    
    # Post-processing
    confidence_threshold: float = 0.3  # Minimum confidence to report pitch
    
    # Octave error handling
    check_octave_errors: bool = True   # Enable octave error correction
    octave_error_threshold: float = 0.9  # Threshold for accepting octave
    

@dataclass
class YinFrame:
    """Result for a single frame of YIN analysis."""
    frequency: float        # Detected frequency in Hz (0 if unvoiced)
    confidence: float       # Confidence (1 - CMNDF value at detected lag)
    period_samples: float   # Detected period in samples (with sub-sample accuracy)
    is_voiced: bool         # Whether frame is considered voiced


def difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the difference function d(τ).
    
    d(τ) = Σ (x[j] - x[j + τ])²
    
    This is equivalent to r(0) + r'(0) - 2*r(τ) where r is autocorrelation,
    but computed directly. The key insight is that for periodic signals,
    d(τ) has minima at multiples of the period.
    
    Args:
        frame: Audio frame
        max_lag: Maximum lag to compute
        
    Returns:
        Difference function values for lags 0 to max_lag-1
    """
    n = len(frame)
    # Use FFT-based method for efficiency (much faster for large frames)
    # d(τ) = r(0) + r'(0) - 2*r(τ)
    # where r(τ) is the autocorrelation and r'(τ) is autocorr of shifted signal
    
    # Pad to power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < n + max_lag:
        fft_size *= 2
    
    # Compute cumulative sum of squares for r(0) contribution
    cumsum_sq = np.zeros(max_lag)
    sq = frame ** 2
    cumsum = np.cumsum(sq)
    
    # r(0) part: sum of x[j]² for j in [0, n-τ)
    # This equals cumsum[n-τ-1] for τ > 0
    for tau in range(1, max_lag):
        cumsum_sq[tau] = cumsum[n - tau - 1] if n - tau - 1 >= 0 else 0
    
    # r'(0) part: sum of x[j+τ]² for j in [0, n-τ)  
    # This equals cumsum[n-1] - cumsum[τ-1] for τ > 0
    cumsum_sq_shifted = np.zeros(max_lag)
    for tau in range(1, max_lag):
        if tau - 1 >= 0:
            cumsum_sq_shifted[tau] = cumsum[n - 1] - cumsum[tau - 1]
        else:
            cumsum_sq_shifted[tau] = cumsum[n - 1]
    
    # Compute cross-correlation using FFT
    padded = np.zeros(fft_size)
    padded[:n] = frame
    fft_frame = np.fft.rfft(padded)
    
    # Autocorrelation via FFT
    power_spectrum = fft_frame * np.conj(fft_frame)
    autocorr_full = np.fft.irfft(power_spectrum)
    autocorr = autocorr_full[:max_lag]
    
    # Difference function: d(τ) = r(0) + r'(0) - 2*r(τ)
    diff = np.zeros(max_lag)
    diff[0] = 0  # d(0) = 0 by definition
    
    for tau in range(1, max_lag):
        diff[tau] = cumsum_sq[tau] + cumsum_sq_shifted[tau] - 2 * autocorr[tau]
    
    return diff


def cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
    """
    Compute the Cumulative Mean Normalized Difference Function (CMNDF).
    
    d'(τ) = d(τ) / ((1/τ) * Σ d(j)) for τ > 0
    d'(0) = 1
    
    This normalization is crucial for avoiding false pitch detection at low lags.
    Without it, the difference function tends to have its lowest values at small
    lags, causing octave errors.
    
    Args:
        diff: Difference function values
        
    Returns:
        CMNDF values (same length as diff)
    """
    n = len(diff)
    cmndf = np.ones(n)  # d'(0) = 1
    
    running_sum = 0.0
    for tau in range(1, n):
        running_sum += diff[tau]
        if running_sum > 0:
            cmndf[tau] = diff[tau] * tau / running_sum
        else:
            cmndf[tau] = 1.0
    
    return cmndf


def parabolic_interpolation(cmndf: np.ndarray, tau: int) -> Tuple[float, float]:
    """
    Refine lag estimate using parabolic interpolation.
    
    Fits a parabola through the three points around the minimum and finds
    the true minimum, giving sub-sample accuracy.
    
    Args:
        cmndf: CMNDF values
        tau: Initial lag estimate (integer)
        
    Returns:
        (refined_tau, refined_value): Sub-sample lag and interpolated CMNDF value
    """
    if tau <= 0 or tau >= len(cmndf) - 1:
        return float(tau), cmndf[tau]
    
    # Get three points
    y0 = cmndf[tau - 1]
    y1 = cmndf[tau]
    y2 = cmndf[tau + 1]
    
    # Parabolic interpolation
    # The minimum of the parabola through (-1, y0), (0, y1), (1, y2) is at:
    # x = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
    denom = 2 * (y0 - 2 * y1 + y2)
    
    if abs(denom) < 1e-10:
        return float(tau), y1
    
    delta = (y0 - y2) / denom
    
    # Clamp delta to [-1, 1] for safety
    delta = max(-1.0, min(1.0, delta))
    
    refined_tau = tau + delta
    
    # Compute interpolated value
    refined_value = y1 - (y0 - y2) * delta / 4
    
    return refined_tau, max(0.0, refined_value)


def find_best_local_estimate(
    cmndf: np.ndarray,
    min_lag: int,
    max_lag: int,
    threshold: float,
    check_octave_errors: bool = True,
    octave_error_threshold: float = 0.9
) -> Tuple[Optional[int], float]:
    """
    Find the best pitch estimate, handling octave errors.
    
    The YIN algorithm looks for the first minimum below the threshold.
    However, if there's a much better minimum at double the period (half
    the frequency), we might be detecting an octave error.
    
    This function implements "best local estimate" to avoid octave errors:
    1. Find first minimum below threshold
    2. Check if there's a significantly better minimum at 2x the lag
    3. If so, use the lower frequency (longer period) estimate
    
    Args:
        cmndf: CMNDF values
        min_lag: Minimum lag to search
        max_lag: Maximum lag to search  
        threshold: Aperiodicity threshold
        check_octave_errors: Whether to check for octave errors
        octave_error_threshold: Threshold for accepting octave
        
    Returns:
        (best_lag, cmndf_value): Best lag estimate and its CMNDF value
    """
    best_lag = None
    best_value = float('inf')
    
    # Search for first minimum below threshold
    for tau in range(min_lag, min(max_lag, len(cmndf))):
        if cmndf[tau] < threshold:
            # Check if this is a local minimum
            if tau > 0 and tau < len(cmndf) - 1:
                if cmndf[tau] <= cmndf[tau - 1] and cmndf[tau] <= cmndf[tau + 1]:
                    best_lag = tau
                    best_value = cmndf[tau]
                    break
            else:
                best_lag = tau
                best_value = cmndf[tau]
                break
    
    # If no minimum below threshold, find global minimum
    if best_lag is None:
        valid_range = cmndf[min_lag:max_lag]
        if len(valid_range) > 0:
            rel_idx = np.argmin(valid_range)
            best_lag = min_lag + rel_idx
            best_value = cmndf[best_lag]
    
    if best_lag is None:
        return None, 1.0
    
    # Octave error check: look for better minimum at 2x lag
    if check_octave_errors and best_lag * 2 < max_lag:
        # Search around 2x the current lag
        search_start = max(min_lag, int(best_lag * 1.8))
        search_end = min(max_lag, int(best_lag * 2.2))
        
        for tau in range(search_start, min(search_end, len(cmndf))):
            # Check if this is a local minimum
            if tau > 0 and tau < len(cmndf) - 1:
                if cmndf[tau] <= cmndf[tau - 1] and cmndf[tau] <= cmndf[tau + 1]:
                    # Is this minimum significantly better?
                    if cmndf[tau] < best_value * octave_error_threshold:
                        best_lag = tau
                        best_value = cmndf[tau]
                        break
    
    return best_lag, best_value


def yin_frame(
    frame: np.ndarray,
    sr: int,
    config: YinConfig
) -> YinFrame:
    """
    Apply YIN algorithm to a single frame.
    
    Args:
        frame: Audio frame (should be config.frame_length samples)
        sr: Sample rate
        config: YIN configuration
        
    Returns:
        YinFrame with detected pitch information
    """
    n = len(frame)
    
    # Compute lag bounds from frequency bounds
    min_lag = max(2, int(sr / config.fmax))
    max_lag = min(n // 2, int(sr / config.fmin))
    
    if max_lag <= min_lag:
        return YinFrame(frequency=0, confidence=0, period_samples=0, is_voiced=False)
    
    # Step 1: Compute difference function
    diff = difference_function(frame, max_lag + 1)
    
    # Step 2: Compute CMNDF
    cmndf = cumulative_mean_normalized_difference(diff)
    
    # Step 3: Find best local estimate (handles octave errors)
    best_lag, cmndf_value = find_best_local_estimate(
        cmndf, min_lag, max_lag,
        config.threshold,
        config.check_octave_errors,
        config.octave_error_threshold
    )
    
    if best_lag is None:
        return YinFrame(frequency=0, confidence=0, period_samples=0, is_voiced=False)
    
    # Step 4: Parabolic interpolation for sub-sample accuracy
    refined_lag, refined_value = parabolic_interpolation(cmndf, best_lag)
    
    # Compute confidence (1 - CMNDF value)
    confidence = 1.0 - refined_value
    
    # Check if voiced
    is_voiced = confidence >= config.confidence_threshold
    
    if not is_voiced:
        return YinFrame(frequency=0, confidence=confidence, period_samples=0, is_voiced=False)
    
    # Compute frequency from refined lag
    frequency = sr / refined_lag if refined_lag > 0 else 0
    
    # Sanity check frequency bounds
    if frequency < config.fmin or frequency > config.fmax:
        return YinFrame(frequency=0, confidence=confidence, period_samples=refined_lag, is_voiced=False)
    
    return YinFrame(
        frequency=frequency,
        confidence=confidence,
        period_samples=refined_lag,
        is_voiced=True
    )


def yin_pitch_detection(
    audio: np.ndarray,
    sr: int,
    config: Optional[YinConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full YIN pitch detection on an audio signal.
    
    Args:
        audio: Audio signal (1D array)
        sr: Sample rate
        config: YIN configuration (uses defaults if None)
        
    Returns:
        (f0, voiced_flags, confidences): Arrays of length n_frames
            - f0: Fundamental frequency in Hz (0 for unvoiced)
            - voiced_flags: Boolean array indicating voiced frames
            - confidences: Confidence values (1 - CMNDF at detected lag)
    """
    if config is None:
        config = YinConfig()
    
    n_samples = len(audio)
    n_frames = 1 + (n_samples - config.frame_length) // config.hop_length
    
    f0 = np.zeros(n_frames)
    voiced_flags = np.zeros(n_frames, dtype=bool)
    confidences = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * config.hop_length
        end = start + config.frame_length
        
        if end > n_samples:
            break
        
        frame = audio[start:end]
        result = yin_frame(frame, sr, config)
        
        f0[i] = result.frequency
        voiced_flags[i] = result.is_voiced
        confidences[i] = result.confidence
    
    return f0, voiced_flags, confidences


# =============================================================================
# Integration with existing guitar-tabs codebase
# =============================================================================

# Import for DetectedNote compatibility
try:
    from benchmarks.metrics import DetectedNote
except ImportError:
    from dataclasses import dataclass as dc
    @dc
    class DetectedNote:
        midi_note: int
        start_time: float
        end_time: float
        confidence: float
        frequency: float


def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2 ** ((midi - 69) / 12))


class YinDetector:
    """
    YIN-based pitch detector for the guitar-tabs benchmark suite.
    
    Compatible with the PitchDetector interface in benchmarks/pitch_detectors.py
    """
    
    name = "yin"
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 75.0,
        fmax: float = 1400.0,
        threshold: float = 0.1,
        confidence_threshold: float = 0.3
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.config = YinConfig(
            fmin=fmin,
            fmax=fmax,
            threshold=threshold,
            hop_length=hop_length,
            confidence_threshold=confidence_threshold
        )
    
    def detect(self, audio: np.ndarray):
        """Detect pitches using YIN algorithm."""
        import time
        
        start_time = time.time()
        notes = self._detect_impl(audio)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Return in DetectorResult-compatible format
        from dataclasses import dataclass
        
        @dataclass
        class DetectorResult:
            notes: list
            processing_time_ms: float
            method_name: str
            raw_data: dict = None
        
        return DetectorResult(
            notes=notes,
            processing_time_ms=elapsed_ms,
            method_name=self.name
        )
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        """Core detection implementation."""
        # Run YIN
        f0, voiced_flags, confidences = yin_pitch_detection(
            audio, self.sr, self.config
        )
        
        # Convert to notes
        return self._f0_to_notes(f0, voiced_flags, confidences)
    
    def _f0_to_notes(
        self,
        f0: np.ndarray,
        voiced: np.ndarray,
        confidences: np.ndarray
    ) -> List[DetectedNote]:
        """Convert frame-by-frame F0 to note events."""
        notes = []
        
        if len(f0) == 0:
            return notes
        
        # Convert frequencies to MIDI
        midi_f0 = np.array([freq_to_midi(f) if f > 0 else 0 for f in f0])
        
        # Quantize to nearest semitone
        midi_quantized = np.round(midi_f0).astype(int)
        
        # Find note regions
        in_note = False
        current_midi = 0
        note_start = 0
        note_confidences = []
        
        for i, (midi, voiced_frame, conf) in enumerate(zip(midi_quantized, voiced, confidences)):
            if voiced_frame and midi > 0:
                if not in_note:
                    in_note = True
                    current_midi = midi
                    note_start = i
                    note_confidences = [conf]
                elif midi == current_midi:
                    note_confidences.append(conf)
                else:
                    # Pitch changed
                    notes.append(self._create_note(
                        current_midi, note_start, i, note_confidences
                    ))
                    current_midi = midi
                    note_start = i
                    note_confidences = [conf]
            else:
                if in_note:
                    notes.append(self._create_note(
                        current_midi, note_start, i, note_confidences
                    ))
                    in_note = False
        
        # Handle note at end
        if in_note:
            notes.append(self._create_note(
                current_midi, note_start, len(f0), note_confidences
            ))
        
        return notes
    
    def _create_note(
        self,
        midi: int,
        start_frame: int,
        end_frame: int,
        confidences: List[float]
    ) -> DetectedNote:
        """Create a DetectedNote from frame indices."""
        start_time = librosa.frames_to_time(
            start_frame, sr=self.sr, hop_length=self.hop_length
        )
        end_time = librosa.frames_to_time(
            end_frame, sr=self.sr, hop_length=self.hop_length
        )
        
        return DetectedNote(
            midi_note=midi,
            start_time=float(start_time),
            end_time=float(end_time),
            confidence=float(np.mean(confidences)) if confidences else 0.5,
            frequency=midi_to_freq(midi)
        )


# =============================================================================
# Integration with ensemble_pitch.py
# =============================================================================

def detect_yin_for_ensemble(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 75.0,
    fmax: float = 1400.0
) -> List[dict]:
    """
    YIN detection formatted for ensemble pitch detector.
    
    Returns list of PitchCandidate-compatible dicts.
    """
    config = YinConfig(
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        threshold=0.1,
        confidence_threshold=0.3
    )
    
    f0, voiced, confidences = yin_pitch_detection(y, sr, config)
    
    # Convert to candidates
    candidates = []
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )
    
    for i, (freq, is_voiced, conf) in enumerate(zip(f0, voiced, confidences)):
        if is_voiced and freq > 0 and conf > 0.3:
            midi = int(round(freq_to_midi(freq)))
            if 30 <= midi <= 96:  # Guitar range
                candidates.append({
                    'time': float(times[i]),
                    'midi_note': midi,
                    'frequency': float(freq),
                    'confidence': float(conf),
                    'method': 'yin',
                    'raw_pitch': float(freq)
                })
    
    return candidates


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="YIN Pitch Detection")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--fmin", type=float, default=75.0, help="Minimum frequency")
    parser.add_argument("--fmax", type=float, default=1400.0, help="Maximum frequency")
    parser.add_argument("--threshold", type=float, default=0.1, help="YIN threshold")
    parser.add_argument("--hop", type=int, default=512, help="Hop length")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    
    args = parser.parse_args()
    
    print(f"Loading {args.audio_path}...")
    y, sr = librosa.load(args.audio_path, sr=args.sr, mono=True)
    
    print(f"Running YIN pitch detection...")
    print(f"  fmin={args.fmin} Hz, fmax={args.fmax} Hz")
    print(f"  threshold={args.threshold}")
    
    detector = YinDetector(
        sr=sr,
        hop_length=args.hop,
        fmin=args.fmin,
        fmax=args.fmax,
        threshold=args.threshold
    )
    
    result = detector.detect(y)
    
    print(f"\nDetected {len(result.notes)} notes in {result.processing_time_ms:.1f}ms:")
    print("-" * 70)
    print(f"{'Note':<8} {'Start':>10} {'End':>10} {'Duration':>10} {'Confidence':>12}")
    print("-" * 70)
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for note in result.notes:
        name = NOTE_NAMES[note.midi_note % 12] + str(note.midi_note // 12 - 1)
        duration = note.end_time - note.start_time
        print(f"{name:<8} {note.start_time:>10.3f} {note.end_time:>10.3f} {duration:>10.3f} {note.confidence:>12.3f}")
    
    print("-" * 70)
