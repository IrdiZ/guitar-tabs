#!/usr/bin/env python3
"""
Cepstral Analysis for Pitch Detection

The cepstrum is the INVERSE FFT of the LOG SPECTRUM. It's powerful because:
1. Harmonics in the spectrum become periodic -> peak in cepstrum
2. The peak location = fundamental period (in samples)
3. Extremely robust to harmonic content - MORE harmonics = STRONGER peak
4. Works brilliantly on distorted guitar where harmonics dominate

Cepstrum derivation:
- Take FFT of signal -> get spectrum with harmonic peaks at f0, 2*f0, 3*f0...
- Take log magnitude -> converts multiplicative (convolved) components to additive
- Take IFFT -> "quefrency" domain - periodic structures in spectrum become peaks

For a note at 100Hz with harmonics at 200Hz, 300Hz, 400Hz...
The spacing in the spectrum is uniform (100Hz apart).
The IFFT of uniformly spaced peaks = single peak at the period (1/100Hz = 10ms).

This is DIFFERENT from:
- YIN/pYIN: Time-domain autocorrelation
- HPS: Multiply downsampled spectra
- CREPE: Neural network pattern matching

Cepstrum excels when:
- Signal has many harmonics (guitar, especially distorted)
- Fundamental is weak compared to harmonics
- Signal is noisy but harmonics are clear
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt, medfilt
from scipy.fft import fft, ifft, rfft, irfft
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings


# Constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Guitar frequency range
GUITAR_MIN_HZ = 70    # Low E ~82Hz, drop D ~73Hz
GUITAR_MAX_HZ = 1400  # High frets on high E string


@dataclass
class CepstralConfig:
    """Configuration for cepstral pitch detection."""
    
    # FFT settings
    n_fft: int = 4096           # High resolution for accurate period detection
    hop_length: int = 512       # ~23ms at 22050Hz
    
    # Cepstrum settings
    lifter_cutoff_low: int = 20     # Ignore very short periods (high freq noise)
    lifter_cutoff_high: int = 1000  # Ignore very long periods (below guitar range)
    
    # Peak detection
    min_peak_prominence: float = 0.1  # Minimum prominence for cepstral peak
    min_peak_height: float = 0.05     # Minimum absolute height
    
    # Frequency constraints
    min_freq: float = GUITAR_MIN_HZ
    max_freq: float = GUITAR_MAX_HZ
    
    # Power spectrum enhancement
    use_power_cepstrum: bool = True   # Use |X|^2 instead of |X|
    apply_preemphasis: bool = True    # High-pass pre-emphasis
    preemphasis_coef: float = 0.97
    
    # Confidence thresholds
    min_confidence: float = 0.25
    
    # Multi-peak analysis (for polyphonic detection)
    detect_multiple_peaks: bool = True
    max_peaks: int = 4
    
    # Harmonic verification
    verify_harmonics: bool = True     # Cross-check with spectrum
    harmonic_tolerance_cents: float = 50  # ±50 cents tolerance
    
    # Output
    verbose: bool = False


@dataclass
class CepstralPitchResult:
    """Result from cepstral pitch detection at one frame."""
    time: float
    frequency: float
    midi_note: int
    confidence: float
    method: str = "cepstrum"
    
    # Cepstrum-specific data
    cepstral_peak_quefrency: int = 0     # Peak location in samples
    cepstral_peak_height: float = 0.0    # Peak magnitude
    secondary_peaks: List[Tuple[float, float]] = field(default_factory=list)  # [(freq, height), ...]
    harmonic_score: float = 0.0          # How well harmonics match
    
    @property
    def name(self) -> str:
        if self.midi_note <= 0:
            return "?"
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)


class CepstralPitchDetector:
    """
    Cepstral analysis-based pitch detection.
    
    Particularly effective for:
    - Distorted guitar (rich harmonics)
    - Signals where fundamental is weak
    - Noisy recordings with clear harmonic structure
    """
    
    def __init__(self, sr: int = 22050, config: Optional[CepstralConfig] = None):
        self.sr = sr
        self.config = config or CepstralConfig()
        
        # Precompute quefrency range for guitar frequencies
        # quefrency (samples) = sr / frequency
        self.min_quefrency = int(sr / self.config.max_freq)  # High freq = low quefrency
        self.max_quefrency = int(sr / self.config.min_freq)  # Low freq = high quefrency
        
        if self.config.verbose:
            print(f"Cepstral detector initialized: sr={sr}")
            print(f"  Quefrency range: {self.min_quefrency} - {self.max_quefrency} samples")
            print(f"  Frequency range: {self.config.min_freq} - {self.config.max_freq} Hz")
    
    def compute_cepstrum(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the real cepstrum of a signal frame.
        
        Real cepstrum = IFFT(log(|FFT(x)|))
        
        Args:
            frame: Audio frame (typically 2048-4096 samples)
            
        Returns:
            Real cepstrum (same length as frame)
        """
        # Apply pre-emphasis if configured (boosts high frequencies)
        if self.config.apply_preemphasis:
            frame = np.append(frame[0], frame[1:] - self.config.preemphasis_coef * frame[:-1])
        
        # Window the frame
        windowed = frame * np.hanning(len(frame))
        
        # Compute FFT
        X = rfft(windowed, n=self.config.n_fft)
        
        # Compute magnitude (or power) spectrum
        if self.config.use_power_cepstrum:
            mag = np.abs(X) ** 2 + 1e-10  # Power spectrum with floor
        else:
            mag = np.abs(X) + 1e-10  # Magnitude spectrum with floor
        
        # Log spectrum
        log_spectrum = np.log(mag)
        
        # Inverse FFT to get cepstrum
        cepstrum = irfft(log_spectrum)
        
        return np.real(cepstrum)
    
    def find_fundamental_from_cepstrum(self, cepstrum: np.ndarray) -> Tuple[float, float, List[Tuple[int, float]]]:
        """
        Find the fundamental frequency from cepstrum peaks.
        
        Args:
            cepstrum: The computed cepstrum
            
        Returns:
            Tuple of (frequency, confidence, list of (quefrency, height) peaks)
        """
        # Focus on the relevant quefrency range for guitar
        search_region = cepstrum[self.min_quefrency:self.max_quefrency]
        
        if len(search_region) == 0:
            return 0.0, 0.0, []
        
        # Normalize for consistent thresholding
        max_val = np.max(np.abs(search_region))
        if max_val > 0:
            normalized = search_region / max_val
        else:
            return 0.0, 0.0, []
        
        # Find peaks in the cepstrum
        peaks, properties = find_peaks(
            normalized,
            height=self.config.min_peak_height,
            prominence=self.config.min_peak_prominence,
            distance=5  # Minimum distance between peaks in samples
        )
        
        if len(peaks) == 0:
            return 0.0, 0.0, []
        
        # Get peak heights
        heights = properties['peak_heights'] if 'peak_heights' in properties else normalized[peaks]
        
        # Convert peaks back to absolute quefrency
        peaks_absolute = peaks + self.min_quefrency
        
        # Sort by height (strongest first)
        sorted_indices = np.argsort(heights)[::-1]
        peaks_absolute = peaks_absolute[sorted_indices]
        heights = heights[sorted_indices]
        
        # The fundamental is typically the strongest peak
        # But we need to check for subharmonic errors
        best_quefrency = peaks_absolute[0]
        best_height = heights[0]
        
        # Check if there's a peak at 2x the quefrency (octave error check)
        # If so, the higher quefrency (lower frequency) might be the true fundamental
        for i, (q, h) in enumerate(zip(peaks_absolute[1:], heights[1:]), 1):
            # Is this peak at approximately 2x the first peak's quefrency?
            ratio = q / best_quefrency
            if 1.9 < ratio < 2.1 and h > best_height * 0.5:
                # This might be the true fundamental (octave lower)
                best_quefrency = q
                best_height = h
                break
        
        # Convert quefrency to frequency
        if best_quefrency > 0:
            fundamental_freq = self.sr / best_quefrency
        else:
            fundamental_freq = 0.0
        
        # Confidence based on peak prominence and how much it stands out
        confidence = min(1.0, best_height * 1.5)
        
        # Collect secondary peaks for multi-pitch detection
        secondary = []
        if self.config.detect_multiple_peaks:
            for q, h in zip(peaks_absolute[:self.config.max_peaks], heights[:self.config.max_peaks]):
                freq = self.sr / q if q > 0 else 0
                if freq != fundamental_freq:
                    secondary.append((freq, float(h)))
        
        return fundamental_freq, confidence, secondary
    
    def verify_with_harmonics(self, frame: np.ndarray, fundamental_freq: float) -> float:
        """
        Verify the detected fundamental by checking for expected harmonics in the spectrum.
        
        Returns a harmonic score (0-1) indicating how well harmonics match.
        """
        if fundamental_freq <= 0:
            return 0.0
        
        # Compute spectrum
        windowed = frame * np.hanning(len(frame))
        X = np.abs(rfft(windowed, n=self.config.n_fft))
        
        # Frequency bins
        freqs = np.fft.rfftfreq(self.config.n_fft, 1/self.sr)
        
        # Check for harmonics 2, 3, 4, 5
        harmonic_scores = []
        tolerance_hz = fundamental_freq * (self.config.harmonic_tolerance_cents / 1200)  # cents to Hz ratio
        
        for harmonic in range(2, 6):
            expected_freq = fundamental_freq * harmonic
            if expected_freq > self.sr / 2:
                break
            
            # Find if there's a peak near the expected harmonic
            mask = np.abs(freqs - expected_freq) < tolerance_hz * harmonic
            if np.any(mask):
                harmonic_magnitude = np.max(X[mask])
                noise_floor = np.median(X)
                if noise_floor > 0:
                    score = min(1.0, harmonic_magnitude / (noise_floor * 5))
                else:
                    score = 0.0
                harmonic_scores.append(score)
        
        if harmonic_scores:
            return np.mean(harmonic_scores)
        return 0.0
    
    def detect_frame(self, frame: np.ndarray, time: float) -> CepstralPitchResult:
        """
        Detect pitch for a single frame.
        
        Args:
            frame: Audio frame
            time: Time in seconds
            
        Returns:
            CepstralPitchResult
        """
        # Compute cepstrum
        cepstrum = self.compute_cepstrum(frame)
        
        # Find fundamental
        freq, confidence, secondary = self.find_fundamental_from_cepstrum(cepstrum)
        
        # Verify with harmonics if enabled
        harmonic_score = 0.0
        if self.config.verify_harmonics and freq > 0:
            harmonic_score = self.verify_with_harmonics(frame, freq)
            # Boost confidence if harmonics verify well
            confidence = confidence * 0.7 + harmonic_score * 0.3
        
        # Convert to MIDI
        midi_note = 0
        if freq > 0 and self.config.min_freq <= freq <= self.config.max_freq:
            midi_note = int(round(librosa.hz_to_midi(freq)))
        else:
            freq = 0
            confidence = 0
        
        return CepstralPitchResult(
            time=time,
            frequency=freq,
            midi_note=midi_note,
            confidence=confidence,
            cepstral_peak_quefrency=int(self.sr / freq) if freq > 0 else 0,
            cepstral_peak_height=confidence,
            secondary_peaks=[(f, h) for f, h in secondary if self.config.min_freq <= f <= self.config.max_freq],
            harmonic_score=harmonic_score
        )
    
    def detect(self, audio: np.ndarray) -> List[CepstralPitchResult]:
        """
        Detect pitch across entire audio signal.
        
        Args:
            audio: Audio signal (mono)
            
        Returns:
            List of CepstralPitchResult for each frame
        """
        results = []
        n_frames = 1 + (len(audio) - self.config.n_fft) // self.config.hop_length
        
        for i in range(n_frames):
            start = i * self.config.hop_length
            end = start + self.config.n_fft
            
            if end > len(audio):
                break
            
            frame = audio[start:end]
            time = start / self.sr
            
            result = self.detect_frame(frame, time)
            results.append(result)
        
        return results
    
    def detect_notes(self, audio: np.ndarray, min_duration: float = 0.05) -> List[Dict]:
        """
        Detect notes (grouped pitch detections) from audio.
        
        Args:
            audio: Audio signal (mono)
            min_duration: Minimum note duration in seconds
            
        Returns:
            List of note dictionaries with start_time, end_time, midi_note, frequency, confidence
        """
        frames = self.detect(audio)
        
        if not frames:
            return []
        
        # Group consecutive frames with same MIDI note
        notes = []
        current_note = None
        current_start = 0
        confidences = []
        frequencies = []
        
        for frame in frames:
            if frame.midi_note == 0 or frame.confidence < self.config.min_confidence:
                # No valid pitch - end current note if any
                if current_note is not None and len(confidences) > 0:
                    duration = frame.time - current_start
                    if duration >= min_duration:
                        notes.append({
                            'midi_note': current_note,
                            'start_time': current_start,
                            'end_time': frame.time,
                            'frequency': np.mean(frequencies),
                            'confidence': np.mean(confidences),
                            'method': 'cepstrum'
                        })
                current_note = None
                confidences = []
                frequencies = []
            elif current_note is None:
                # Start new note
                current_note = frame.midi_note
                current_start = frame.time
                confidences = [frame.confidence]
                frequencies = [frame.frequency]
            elif frame.midi_note == current_note:
                # Continue current note
                confidences.append(frame.confidence)
                frequencies.append(frame.frequency)
            else:
                # Different note - save current and start new
                duration = frame.time - current_start
                if duration >= min_duration:
                    notes.append({
                        'midi_note': current_note,
                        'start_time': current_start,
                        'end_time': frame.time,
                        'frequency': np.mean(frequencies),
                        'confidence': np.mean(confidences),
                        'method': 'cepstrum'
                    })
                current_note = frame.midi_note
                current_start = frame.time
                confidences = [frame.confidence]
                frequencies = [frame.frequency]
        
        # Handle last note
        if current_note is not None and len(confidences) > 0:
            end_time = (len(audio) / self.sr)
            duration = end_time - current_start
            if duration >= min_duration:
                notes.append({
                    'midi_note': current_note,
                    'start_time': current_start,
                    'end_time': end_time,
                    'frequency': np.mean(frequencies),
                    'confidence': np.mean(confidences),
                    'method': 'cepstrum'
                })
        
        return notes


def detect_cepstral_for_ensemble(audio: np.ndarray, sr: int, config: Optional[CepstralConfig] = None) -> List[Dict]:
    """
    Cepstral pitch detection formatted for ensemble integration.
    
    Returns pitch candidates in the format expected by ensemble_pitch.py
    
    Args:
        audio: Mono audio signal
        sr: Sample rate
        config: Optional CepstralConfig
        
    Returns:
        List of dicts with: time, midi_note, frequency, confidence, method
    """
    detector = CepstralPitchDetector(sr=sr, config=config)
    frames = detector.detect(audio)
    
    candidates = []
    for frame in frames:
        if frame.midi_note > 0 and frame.confidence >= (config.min_confidence if config else 0.25):
            candidates.append({
                'time': frame.time,
                'midi_note': frame.midi_note,
                'frequency': frame.frequency,
                'confidence': frame.confidence,
                'method': 'cepstrum'
            })
            
            # Add secondary peaks as lower-confidence candidates
            for freq, height in frame.secondary_peaks:
                if height > 0.3:  # Only strong secondary peaks
                    midi = int(round(librosa.hz_to_midi(freq)))
                    candidates.append({
                        'time': frame.time,
                        'midi_note': midi,
                        'frequency': freq,
                        'confidence': height * 0.7,  # Reduce confidence for secondaries
                        'method': 'cepstrum_secondary'
                    })
    
    return candidates


def demo_cepstrum(audio_path: str):
    """
    Demonstrate cepstral pitch detection on an audio file.
    """
    print(f"\n{'='*60}")
    print("CEPSTRAL PITCH DETECTION DEMO")
    print(f"{'='*60}\n")
    
    # Load audio
    print(f"Loading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr}Hz")
    
    # Create detector with verbose config
    config = CepstralConfig(verbose=True, verify_harmonics=True)
    detector = CepstralPitchDetector(sr=sr, config=config)
    
    # Detect notes
    print("\nDetecting notes...")
    notes = detector.detect_notes(audio)
    
    print(f"\nDetected {len(notes)} notes:\n")
    print(f"{'Time':>8} | {'Note':>6} | {'Freq':>8} | {'Conf':>6} | {'Duration':>8}")
    print("-" * 50)
    
    for note in notes:
        midi = note['midi_note']
        name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
        print(f"{note['start_time']:>8.3f} | {name:>6} | {note['frequency']:>8.1f} | {note['confidence']:>6.2f} | {note['end_time'] - note['start_time']:>8.3f}")
    
    return notes


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_cepstrum(sys.argv[1])
    else:
        print("Usage: python cepstral_pitch.py <audio_file>")
        print("\nCepstral analysis finds pitch by:")
        print("  1. Computing FFT of the signal")
        print("  2. Taking log of the magnitude spectrum")
        print("  3. Computing IFFT -> 'quefrency' domain")
        print("  4. Finding peaks = fundamental period")
        print("\nExcellent for distorted guitar with rich harmonics!")
