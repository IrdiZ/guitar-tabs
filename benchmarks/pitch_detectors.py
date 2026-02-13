#!/usr/bin/env python3
"""
Pitch Detection Methods for Benchmarking

Implements various pitch detection algorithms to compare:
1. pYIN (librosa) - Probabilistic YIN, good for monophonic
2. CREPE - Deep learning based, high accuracy
3. Basic Pitch - Spotify's model for polyphonic transcription
4. FFT Peak - Simple FFT-based peak detection
5. Autocorrelation - Classic time-domain method
6. piptrack (librosa) - Multi-pitch tracking
7. NMF - Non-negative Matrix Factorization for polyphonic

Each detector returns a list of DetectedNote objects.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.metrics import DetectedNote


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note (can be fractional)."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


@dataclass
class DetectorResult:
    """Result from a pitch detector including timing info."""
    notes: List[DetectedNote]
    processing_time_ms: float
    method_name: str
    raw_data: dict = None  # Optional raw data for debugging


class PitchDetector:
    """Base class for pitch detectors."""
    
    name: str = "base"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect(self, audio: np.ndarray) -> DetectorResult:
        """
        Detect notes in audio.
        
        Args:
            audio: Audio signal array
            
        Returns:
            DetectorResult with detected notes and timing
        """
        start_time = time.time()
        notes = self._detect_impl(audio)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return DetectorResult(
            notes=notes,
            processing_time_ms=elapsed_ms,
            method_name=self.name
        )
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        """Override in subclasses."""
        raise NotImplementedError


class PyinDetector(PitchDetector):
    """
    pYIN (Probabilistic YIN) pitch detection.
    
    Good for monophonic signals with vibrato.
    Uses Viterbi decoding for smooth pitch tracking.
    """
    
    name = "pyin"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512,
                 fmin: float = 65.0, fmax: float = 2100.0,
                 threshold: float = 0.3):
        super().__init__(sr, hop_length)
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Run pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        return self._f0_to_notes(f0, voiced_flag, voiced_probs)
    
    def _f0_to_notes(
        self,
        f0: np.ndarray,
        voiced: np.ndarray,
        confidence: np.ndarray
    ) -> List[DetectedNote]:
        """Convert frame-by-frame F0 to note events."""
        notes = []
        
        if len(f0) == 0:
            return notes
        
        # Convert frequencies to MIDI
        midi_f0 = np.array([freq_to_midi(f) if f > 0 else 0 for f in f0])
        
        # Quantize to nearest semitone for note detection
        midi_quantized = np.round(midi_f0).astype(int)
        
        # Find note regions (consecutive frames with same pitch)
        in_note = False
        current_midi = 0
        note_start = 0
        note_confidences = []
        
        for i, (midi, voiced_frame, conf) in enumerate(zip(midi_quantized, voiced, confidence)):
            if voiced_frame and midi > 0:
                if not in_note:
                    # Start new note
                    in_note = True
                    current_midi = midi
                    note_start = i
                    note_confidences = [conf]
                elif midi == current_midi:
                    # Continue current note
                    note_confidences.append(conf)
                else:
                    # Pitch changed - end current note, start new
                    notes.append(self._create_note(
                        current_midi, note_start, i, note_confidences
                    ))
                    current_midi = midi
                    note_start = i
                    note_confidences = [conf]
            else:
                if in_note:
                    # End current note
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


class PiptrackDetector(PitchDetector):
    """
    librosa.piptrack - Multi-pitch tracking using parabolic interpolation.
    
    Can detect multiple simultaneous pitches (polyphonic).
    """
    
    name = "piptrack"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512,
                 fmin: float = 65.0, fmax: float = 2100.0,
                 threshold: float = 0.1, n_fft: int = 2048):
        super().__init__(sr, hop_length)
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.n_fft = n_fft
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Compute STFT
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Run piptrack
        pitches, magnitudes = librosa.piptrack(
            S=S,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            threshold=self.threshold
        )
        
        return self._pitches_to_notes(pitches, magnitudes)
    
    def _pitches_to_notes(
        self,
        pitches: np.ndarray,
        magnitudes: np.ndarray
    ) -> List[DetectedNote]:
        """Convert pitch/magnitude matrices to note events."""
        notes = []
        n_frames = pitches.shape[1]
        
        # Track active notes: midi -> (start_frame, magnitudes)
        active: dict = {}
        
        for frame in range(n_frames):
            # Get pitches in this frame (non-zero)
            frame_pitches = pitches[:, frame]
            frame_mags = magnitudes[:, frame]
            
            # Find peaks (strongest pitches)
            valid_idx = np.where(frame_pitches > 0)[0]
            
            if len(valid_idx) == 0:
                # No pitches - end all active notes
                for midi, (start, mags) in active.items():
                    notes.append(self._create_note(midi, start, frame, mags))
                active.clear()
                continue
            
            # Get strongest pitch(es)
            sorted_idx = valid_idx[np.argsort(frame_mags[valid_idx])[::-1]]
            top_idx = sorted_idx[:3]  # Keep top 3 pitches
            
            frame_midis = set()
            for idx in top_idx:
                freq = frame_pitches[idx]
                midi = int(round(freq_to_midi(freq)))
                if 30 <= midi <= 96:  # Guitar range
                    frame_midis.add(midi)
            
            # Check which active notes are still present
            ended_midis = set(active.keys()) - frame_midis
            new_midis = frame_midis - set(active.keys())
            continuing_midis = frame_midis & set(active.keys())
            
            # End notes that stopped
            for midi in ended_midis:
                start, mags = active.pop(midi)
                notes.append(self._create_note(midi, start, frame, mags))
            
            # Start new notes
            for midi in new_midis:
                active[midi] = (frame, [frame_mags[np.argmax(frame_pitches)]])
            
            # Update continuing notes
            for midi in continuing_midis:
                start, mags = active[midi]
                mags.append(frame_mags[np.argmax(frame_pitches)])
        
        # End remaining notes
        for midi, (start, mags) in active.items():
            notes.append(self._create_note(midi, start, n_frames, mags))
        
        return notes
    
    def _create_note(
        self,
        midi: int,
        start_frame: int,
        end_frame: int,
        magnitudes: List[float]
    ) -> DetectedNote:
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
            confidence=float(np.mean(magnitudes)) if magnitudes else 0.5,
            frequency=midi_to_freq(midi)
        )


class FFTPeakDetector(PitchDetector):
    """
    Simple FFT-based peak detection.
    
    Baseline method: find dominant frequency in each frame.
    """
    
    name = "fft_peak"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512,
                 n_fft: int = 4096, fmin: float = 65.0, fmax: float = 2100.0):
        super().__init__(sr, hop_length)
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Filter to guitar range
        valid_bins = np.where((freqs >= self.fmin) & (freqs <= self.fmax))[0]
        
        # Find peak in each frame
        n_frames = magnitude.shape[1]
        frame_pitches = np.zeros(n_frames)
        frame_mags = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame_mag = magnitude[valid_bins, i]
            if np.max(frame_mag) > np.mean(magnitude) * 0.5:  # Threshold
                peak_idx = np.argmax(frame_mag)
                frame_pitches[i] = freqs[valid_bins[peak_idx]]
                frame_mags[i] = frame_mag[peak_idx]
        
        # Convert to notes
        notes = []
        in_note = False
        current_midi = 0
        note_start = 0
        note_mags = []
        
        for i, (freq, mag) in enumerate(zip(frame_pitches, frame_mags)):
            if freq > 0:
                midi = int(round(freq_to_midi(freq)))
                if not in_note:
                    in_note = True
                    current_midi = midi
                    note_start = i
                    note_mags = [mag]
                elif midi == current_midi:
                    note_mags.append(mag)
                else:
                    # End current, start new
                    notes.append(self._create_note(current_midi, note_start, i, note_mags))
                    current_midi = midi
                    note_start = i
                    note_mags = [mag]
            else:
                if in_note:
                    notes.append(self._create_note(current_midi, note_start, i, note_mags))
                    in_note = False
        
        if in_note:
            notes.append(self._create_note(current_midi, note_start, n_frames, note_mags))
        
        return notes
    
    def _create_note(self, midi, start_frame, end_frame, mags):
        start_time = librosa.frames_to_time(start_frame, sr=self.sr, hop_length=self.hop_length)
        end_time = librosa.frames_to_time(end_frame, sr=self.sr, hop_length=self.hop_length)
        return DetectedNote(
            midi_note=midi,
            start_time=float(start_time),
            end_time=float(end_time),
            confidence=float(np.mean(mags) / (np.max(mags) + 1e-10)) if mags else 0.5,
            frequency=midi_to_freq(midi)
        )


class AutocorrelationDetector(PitchDetector):
    """
    Time-domain autocorrelation pitch detection.
    
    Classic method, computationally simple but less accurate for harmonics.
    """
    
    name = "autocorr"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512,
                 frame_length: int = 2048, fmin: float = 65.0, fmax: float = 2100.0):
        super().__init__(sr, hop_length)
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Compute pitch for each frame using autocorrelation
        n_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        
        min_lag = int(self.sr / self.fmax)
        max_lag = int(self.sr / self.fmin)
        
        frame_pitches = np.zeros(n_frames)
        frame_confs = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_length]
            
            if len(frame) < self.frame_length:
                continue
            
            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            # Find peak in valid lag range
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region) + min_lag
                
                if autocorr[peak_idx] > 0.5:  # Confidence threshold
                    frame_pitches[i] = self.sr / peak_idx
                    frame_confs[i] = autocorr[peak_idx]
        
        # Convert to notes
        return self._pitches_to_notes(frame_pitches, frame_confs)
    
    def _pitches_to_notes(self, pitches, confidences):
        notes = []
        in_note = False
        current_midi = 0
        note_start = 0
        note_confs = []
        
        for i, (freq, conf) in enumerate(zip(pitches, confidences)):
            if freq > 0:
                midi = int(round(freq_to_midi(freq)))
                if not in_note:
                    in_note = True
                    current_midi = midi
                    note_start = i
                    note_confs = [conf]
                elif midi == current_midi:
                    note_confs.append(conf)
                else:
                    notes.append(self._create_note(current_midi, note_start, i, note_confs))
                    current_midi = midi
                    note_start = i
                    note_confs = [conf]
            else:
                if in_note:
                    notes.append(self._create_note(current_midi, note_start, i, note_confs))
                    in_note = False
        
        if in_note:
            notes.append(self._create_note(current_midi, note_start, len(pitches), note_confs))
        
        return notes
    
    def _create_note(self, midi, start_frame, end_frame, confs):
        start_time = librosa.frames_to_time(start_frame, sr=self.sr, hop_length=self.hop_length)
        end_time = librosa.frames_to_time(end_frame, sr=self.sr, hop_length=self.hop_length)
        return DetectedNote(
            midi_note=midi,
            start_time=float(start_time),
            end_time=float(end_time),
            confidence=float(np.mean(confs)) if confs else 0.5,
            frequency=midi_to_freq(midi)
        )


class HybridDetector(PitchDetector):
    """
    Hybrid detector combining pYIN with onset detection.
    
    Uses onset detection to segment audio, then pYIN for pitch.
    Should be more accurate for timing.
    """
    
    name = "hybrid_pyin_onset"
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        super().__init__(sr, hop_length)
        self.pyin = PyinDetector(sr, hop_length)
    
    def _detect_impl(self, audio: np.ndarray) -> List[DetectedNote]:
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sr, hop_length=self.hop_length,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        if len(onset_times) == 0:
            # Fall back to regular pYIN
            return self.pyin._detect_impl(audio)
        
        # Add end time
        duration = len(audio) / self.sr
        boundaries = list(onset_times) + [duration]
        
        # Run pYIN
        f0, voiced, probs = librosa.pyin(
            audio, sr=self.sr, hop_length=self.hop_length,
            fmin=65.0, fmax=2100.0, fill_na=0.0
        )
        
        # Convert f0 to MIDI per frame
        midi_frames = np.array([
            int(round(freq_to_midi(f))) if f > 0 else 0 for f in f0
        ])
        
        # Create notes at onset boundaries
        notes = []
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            
            # Find dominant pitch in this segment
            start_frame = int(start_time * self.sr / self.hop_length)
            end_frame = int(end_time * self.sr / self.hop_length)
            
            segment_midi = midi_frames[start_frame:end_frame]
            segment_probs = probs[start_frame:end_frame] if start_frame < len(probs) else []
            
            # Get most common non-zero MIDI in segment
            valid_midi = segment_midi[segment_midi > 0]
            if len(valid_midi) > 0:
                midi_note = int(np.median(valid_midi))
                conf = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.5
                
                notes.append(DetectedNote(
                    midi_note=midi_note,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    confidence=conf,
                    frequency=midi_to_freq(midi_note)
                ))
        
        return notes


# Import custom YIN detector
try:
    from yin_pitch import YinDetector
    HAS_YIN = True
except ImportError:
    HAS_YIN = False


# Registry of available detectors
DETECTORS = {
    'pyin': PyinDetector,
    'piptrack': PiptrackDetector,
    'fft_peak': FFTPeakDetector,
    'autocorr': AutocorrelationDetector,
    'hybrid': HybridDetector,
}

# Add YIN if available
if HAS_YIN:
    DETECTORS['yin'] = YinDetector


def get_all_detectors(sr: int = 22050) -> List[PitchDetector]:
    """Get instances of all available detectors."""
    return [cls(sr=sr) for cls in DETECTORS.values()]


def get_detector(name: str, sr: int = 22050) -> Optional[PitchDetector]:
    """Get a specific detector by name."""
    cls = DETECTORS.get(name)
    return cls(sr=sr) if cls else None
