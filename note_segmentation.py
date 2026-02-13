#!/usr/bin/env python3
"""
Note Segmentation Module - Precise note boundary detection for guitar transcription.

This module solves the key problems in guitar tab generation:
1. Precise onset detection - find exact note start times
2. Note-off detection - find when notes actually end (not just next onset)
3. Attack-time pitch sampling - sample pitch at the right moment (attack, not decay)
4. Overlapping note handling - detect when notes ring together
5. Timing alignment - ensure onset and pitch detection are synchronized

The goal: ACCURATE NOTE BOUNDARIES = ACCURATE TABS
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import Counter
import warnings

# Guitar constants
GUITAR_MIN_HZ = 75   # Below E2 (82 Hz) for drop tunings
GUITAR_MAX_HZ = 1400  # Above high frets on high E
GUITAR_MIN_MIDI = 36  # C2 (for drop tunings)
GUITAR_MAX_MIDI = 90  # D6 (highest practical)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class NoteSegment:
    """A precisely segmented note with accurate boundaries."""
    midi: int
    hz: float
    start_time: float
    end_time: float
    confidence: float
    
    # Detection metadata
    onset_method: str = ""       # How onset was detected
    offset_method: str = ""      # How offset was detected
    pitch_sample_time: float = 0.0  # When pitch was sampled
    attack_strength: float = 0.0    # Strength of attack transient
    is_legato: bool = False         # Hammer-on/pull-off
    overlaps_with: List[int] = field(default_factory=list)  # Indices of overlapping notes
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)
    
    def __repr__(self):
        return f"Note({self.note_name}, {self.start_time:.3f}-{self.end_time:.3f}s, conf={self.confidence:.2f})"


@dataclass 
class OnsetInfo:
    """Detailed onset information."""
    time: float
    frame: int
    confidence: float
    attack_strength: float
    methods: List[str]
    is_legato: bool = False
    backtracked_time: float = 0.0  # Refined onset time after backtracking


@dataclass
class OffsetInfo:
    """Note offset (end) information."""
    time: float
    frame: int
    method: str  # 'energy_decay', 'pitch_change', 'next_onset', 'silence'
    confidence: float


class PreciseOnsetDetector:
    """
    High-precision onset detector optimized for guitar.
    
    Key improvements over basic onset detection:
    1. Sub-frame precision using energy curve analysis
    2. Attack transient detection for guitar picks/fingers
    3. Backtracking to find true attack start
    4. Legato detection (hammer-ons/pull-offs)
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,  # Smaller hop for better precision
        n_fft: int = 2048,
        # Onset thresholds
        onset_threshold: float = 0.3,
        attack_threshold: float = 0.4,
        # Timing parameters
        min_onset_gap_ms: float = 30,  # Minimum gap between onsets
        backtrack_ms: float = 20,       # How far to backtrack for true onset
        # Legato detection
        legato_enabled: bool = True,
        legato_sensitivity: float = 0.4,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.onset_threshold = onset_threshold
        self.attack_threshold = attack_threshold
        self.min_onset_gap_frames = int(min_onset_gap_ms / 1000 * sr / hop_length)
        self.backtrack_samples = int(backtrack_ms / 1000 * sr)
        self.legato_enabled = legato_enabled
        self.legato_sensitivity = legato_sensitivity
    
    def detect(self, y: np.ndarray, verbose: bool = True) -> List[OnsetInfo]:
        """
        Detect onsets with high precision.
        
        Returns list of OnsetInfo with detailed timing information.
        """
        if verbose:
            print("  🎯 Precise onset detection...")
        
        # Compute multiple onset detection functions
        odf_spectral = self._compute_spectral_flux(y)
        odf_hfc = self._compute_hfc(y)
        odf_attack = self._compute_attack_transient(y)
        odf_energy = self._compute_energy_onset(y)
        
        # Combine onset functions with weights
        n_frames = min(len(odf_spectral), len(odf_hfc), len(odf_attack), len(odf_energy))
        odf_spectral = odf_spectral[:n_frames]
        odf_hfc = odf_hfc[:n_frames]
        odf_attack = odf_attack[:n_frames]
        odf_energy = odf_energy[:n_frames]
        
        # Weighted combination (attack transient gets highest weight for guitar)
        odf_combined = (
            0.3 * odf_spectral + 
            0.2 * odf_hfc + 
            0.35 * odf_attack +  # Guitar attack is very important
            0.15 * odf_energy
        )
        
        # Normalize
        odf_combined = odf_combined / (np.max(odf_combined) + 1e-10)
        
        # Peak picking with minimum gap enforcement
        peaks, properties = find_peaks(
            odf_combined,
            height=self.onset_threshold,
            distance=self.min_onset_gap_frames,
            prominence=0.1
        )
        
        if verbose:
            print(f"    Found {len(peaks)} initial peaks")
        
        # Build onset info with detailed analysis
        onsets = []
        for peak in peaks:
            peak_time = librosa.frames_to_time(peak, sr=self.sr, hop_length=self.hop_length)
            
            # Determine which methods contributed
            methods = []
            if peak < len(odf_spectral) and odf_spectral[peak] > self.onset_threshold * 0.8:
                methods.append('spectral_flux')
            if peak < len(odf_hfc) and odf_hfc[peak] > self.onset_threshold * 0.8:
                methods.append('hfc')
            if peak < len(odf_attack) and odf_attack[peak] > self.attack_threshold * 0.8:
                methods.append('attack')
            if peak < len(odf_energy) and odf_energy[peak] > self.onset_threshold * 0.8:
                methods.append('energy')
            
            # Attack strength
            attack_strength = odf_attack[peak] if peak < len(odf_attack) else 0.0
            
            # Backtrack to find true onset start
            backtracked_time = self._backtrack_onset(y, peak_time)
            
            onsets.append(OnsetInfo(
                time=peak_time,
                frame=peak,
                confidence=float(odf_combined[peak]),
                attack_strength=float(attack_strength),
                methods=methods,
                backtracked_time=backtracked_time
            ))
        
        # Detect legato notes (hammer-ons/pull-offs)
        if self.legato_enabled:
            legato_onsets = self._detect_legato(y, odf_energy, onsets)
            onsets.extend(legato_onsets)
            # Re-sort by time
            onsets.sort(key=lambda o: o.backtracked_time)
        
        if verbose:
            n_legato = sum(1 for o in onsets if o.is_legato)
            print(f"    Final: {len(onsets)} onsets ({n_legato} legato)")
        
        return onsets
    
    def _compute_spectral_flux(self, y: np.ndarray) -> np.ndarray:
        """Spectral flux onset detection function."""
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Half-wave rectified difference
        diff = np.diff(S, axis=1)
        diff = np.maximum(0, diff)
        flux = np.sum(diff, axis=0)
        
        # Normalize
        if flux.max() > 0:
            flux = flux / flux.max()
        
        # Pad to match original frame count
        flux = np.concatenate([[0], flux])
        
        return flux
    
    def _compute_hfc(self, y: np.ndarray) -> np.ndarray:
        """High Frequency Content - emphasizes attack transients."""
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Weight by frequency (high frequencies indicate attacks)
        weights = np.arange(1, S.shape[0] + 1)
        hfc = np.sum(S * weights[:, np.newaxis], axis=0)
        
        # First difference, half-wave rectified
        hfc_diff = np.diff(hfc)
        hfc_diff = np.maximum(0, hfc_diff)
        
        if hfc_diff.max() > 0:
            hfc_diff = hfc_diff / hfc_diff.max()
        
        # Pad
        hfc_diff = np.concatenate([[0], hfc_diff])
        
        return hfc_diff
    
    def _compute_attack_transient(self, y: np.ndarray) -> np.ndarray:
        """
        Guitar-specific attack transient detector.
        
        Guitar picks/fingers create fast transients (< 10ms rise time)
        with high-frequency content that quickly decays.
        """
        # High-pass filter to isolate transients (2-8 kHz)
        nyq = self.sr / 2
        high_cutoff = min(2000 / nyq, 0.95)
        
        try:
            b, a = butter(4, high_cutoff, btype='high')
            y_hp = filtfilt(b, a, y)
        except Exception:
            y_hp = y
        
        # Compute envelope using rectification + smoothing
        envelope = np.abs(y_hp)
        
        # Fast attack, slow release envelope follower
        attack_samples = int(0.001 * self.sr)  # 1ms attack
        release_samples = int(0.050 * self.sr)  # 50ms release
        
        # Apply attack smoothing
        envelope_smooth = np.zeros_like(envelope)
        envelope_smooth[0] = envelope[0]
        
        for i in range(1, len(envelope)):
            if envelope[i] > envelope_smooth[i-1]:
                # Attack - fast response
                alpha = 1.0 - np.exp(-1.0 / attack_samples)
                envelope_smooth[i] = alpha * envelope[i] + (1 - alpha) * envelope_smooth[i-1]
            else:
                # Release - slow decay
                alpha = 1.0 - np.exp(-1.0 / release_samples)
                envelope_smooth[i] = alpha * envelope[i] + (1 - alpha) * envelope_smooth[i-1]
        
        # Downsample to frame rate
        n_frames = 1 + len(y) // self.hop_length
        attack_odf = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.hop_length, len(envelope_smooth))
            if start < len(envelope_smooth):
                attack_odf[i] = np.max(envelope_smooth[start:end])
        
        # Compute onset function (first derivative, half-wave rectified)
        attack_diff = np.diff(attack_odf)
        attack_diff = np.maximum(0, attack_diff)
        
        if attack_diff.max() > 0:
            attack_diff = attack_diff / attack_diff.max()
        
        # Pad
        attack_diff = np.concatenate([[0], attack_diff])
        
        return attack_diff
    
    def _compute_energy_onset(self, y: np.ndarray) -> np.ndarray:
        """RMS energy-based onset detection."""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length, frame_length=self.n_fft)[0]
        
        # Log-domain for perceptual scaling
        log_rms = np.log1p(rms * 100)
        
        # First difference
        rms_diff = np.diff(log_rms)
        rms_diff = np.maximum(0, rms_diff)
        
        if rms_diff.max() > 0:
            rms_diff = rms_diff / rms_diff.max()
        
        # Pad
        rms_diff = np.concatenate([[0], rms_diff])
        
        return rms_diff
    
    def _backtrack_onset(self, y: np.ndarray, onset_time: float) -> float:
        """
        Backtrack from detected onset to find true attack start.
        
        Looks backwards in the energy curve to find where the
        attack actually begins (not just where it peaks).
        """
        # Convert time to samples
        onset_sample = int(onset_time * self.sr)
        start_sample = max(0, onset_sample - self.backtrack_samples)
        
        if start_sample >= onset_sample:
            return onset_time
        
        # Get audio segment
        segment = np.abs(y[start_sample:onset_sample])
        
        if len(segment) < 10:
            return onset_time
        
        # Smooth slightly
        segment_smooth = uniform_filter1d(segment, size=32)
        
        # Find minimum (start of rise) in this window
        min_idx = np.argmin(segment_smooth)
        
        # Convert back to time
        new_onset_sample = start_sample + min_idx
        new_onset_time = new_onset_sample / self.sr
        
        # Don't backtrack too much (sanity check)
        max_backtrack = 0.015  # 15ms max
        if onset_time - new_onset_time > max_backtrack:
            new_onset_time = onset_time - max_backtrack
        
        return new_onset_time
    
    def _detect_legato(
        self, 
        y: np.ndarray, 
        energy_odf: np.ndarray,
        existing_onsets: List[OnsetInfo]
    ) -> List[OnsetInfo]:
        """
        Detect legato notes (hammer-ons/pull-offs).
        
        These have:
        - Softer attack than picked notes
        - Pitch change without strong energy onset
        - Usually occur between existing picked onsets
        """
        legato_onsets = []
        
        # Compute pitch contour for legato detection
        f0, voiced, voiced_probs = librosa.pyin(
            y,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        if len(f0) == 0:
            return legato_onsets
        
        # Convert to MIDI for easier comparison
        midi_contour = np.zeros_like(f0)
        valid_f0 = f0 > 0
        midi_contour[valid_f0] = librosa.hz_to_midi(f0[valid_f0])
        
        # Compute pitch changes
        pitch_diff = np.abs(np.diff(midi_contour))
        
        # Pad
        pitch_diff = np.concatenate([[0], pitch_diff])
        
        # Build set of existing onset frames
        existing_frames = set(o.frame for o in existing_onsets)
        
        # Look for pitch changes without corresponding energy onset
        min_pitch_change = 1.5  # At least 1.5 semitones
        
        for i in range(1, len(pitch_diff) - 1):
            # Skip if too close to existing onset
            near_existing = any(abs(i - f) < 5 for f in existing_frames)
            if near_existing:
                continue
            
            # Check for significant pitch change with low energy onset
            pitch_change = pitch_diff[i]
            energy_onset = energy_odf[i] if i < len(energy_odf) else 0
            
            if pitch_change >= min_pitch_change and energy_onset < self.legato_sensitivity:
                # Found potential legato
                time = librosa.frames_to_time(i, sr=self.sr, hop_length=self.hop_length)
                
                legato_onsets.append(OnsetInfo(
                    time=time,
                    frame=i,
                    confidence=0.6,  # Lower confidence for legato
                    attack_strength=energy_onset,
                    methods=['legato'],
                    is_legato=True,
                    backtracked_time=time  # No backtracking for legato
                ))
        
        return legato_onsets


class NoteOffsetDetector:
    """
    Detects when notes end (note-off events).
    
    Key improvements over using next onset time:
    1. Energy decay tracking - note ends when energy drops below threshold
    2. Pitch stability - note ends when pitch becomes unstable
    3. Harmonic decay - track when harmonics fade
    4. Ring-out detection - some notes ring beyond next onset
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,
        # Energy thresholds
        energy_decay_threshold: float = 0.1,  # Relative to peak
        energy_floor: float = 0.01,           # Absolute minimum
        # Timing parameters
        min_note_duration_ms: float = 50,
        max_ring_time_ms: float = 2000,  # Max time a note can ring
        # Pitch stability
        pitch_stability_window: int = 5,
        pitch_stability_threshold: float = 2.0,  # semitones
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.energy_decay_threshold = energy_decay_threshold
        self.energy_floor = energy_floor
        self.min_note_frames = int(min_note_duration_ms / 1000 * sr / hop_length)
        self.max_ring_frames = int(max_ring_time_ms / 1000 * sr / hop_length)
        self.pitch_stability_window = pitch_stability_window
        self.pitch_stability_threshold = pitch_stability_threshold
    
    def detect(
        self,
        y: np.ndarray,
        onset: OnsetInfo,
        next_onset: Optional[OnsetInfo],
        f0: np.ndarray,
        rms: np.ndarray,
        verbose: bool = False
    ) -> OffsetInfo:
        """
        Detect when a note ends given its onset.
        
        Uses multiple methods:
        1. Energy decay below threshold
        2. Pitch becomes unstable/undefined
        3. Next onset starts (but may overlap)
        4. Maximum ring time exceeded
        
        Returns OffsetInfo with timing and method used.
        """
        onset_frame = onset.frame
        
        # Determine search range
        if next_onset:
            # Look up to next onset + some ring time
            max_frame = min(
                next_onset.frame + self.max_ring_frames // 2,
                len(rms) - 1,
                onset_frame + self.max_ring_frames
            )
        else:
            # Look until end of audio or max ring time
            max_frame = min(len(rms) - 1, onset_frame + self.max_ring_frames)
        
        if onset_frame >= len(rms):
            return OffsetInfo(
                time=onset.backtracked_time + 0.1,
                frame=onset_frame + 5,
                method='fallback',
                confidence=0.3
            )
        
        # Get energy and pitch in search range
        search_start = onset_frame
        search_end = max_frame + 1
        
        energy_segment = rms[search_start:search_end]
        f0_segment = f0[search_start:search_end] if f0 is not None else None
        
        if len(energy_segment) == 0:
            return OffsetInfo(
                time=onset.backtracked_time + 0.1,
                frame=onset_frame + 5,
                method='fallback',
                confidence=0.3
            )
        
        # Find peak energy (should be near onset)
        peak_energy = energy_segment[:min(10, len(energy_segment))].max()
        
        if peak_energy < self.energy_floor:
            # Very quiet - use minimum duration
            offset_frame = onset_frame + self.min_note_frames
            return OffsetInfo(
                time=librosa.frames_to_time(offset_frame, sr=self.sr, hop_length=self.hop_length),
                frame=offset_frame,
                method='silence',
                confidence=0.7
            )
        
        # Method 1: Energy decay
        decay_threshold = peak_energy * self.energy_decay_threshold
        energy_offset = None
        
        for i in range(self.min_note_frames, len(energy_segment)):
            if energy_segment[i] < decay_threshold or energy_segment[i] < self.energy_floor:
                energy_offset = search_start + i
                break
        
        # Method 2: Pitch stability
        pitch_offset = None
        if f0_segment is not None and len(f0_segment) > self.pitch_stability_window:
            # Convert to MIDI for comparison
            midi_segment = np.zeros_like(f0_segment)
            valid = f0_segment > 0
            if np.any(valid):
                midi_segment[valid] = librosa.hz_to_midi(f0_segment[valid])
                
                # Track pitch stability
                for i in range(self.min_note_frames, len(midi_segment) - self.pitch_stability_window):
                    window = midi_segment[i:i + self.pitch_stability_window]
                    valid_window = window > 0
                    
                    if np.sum(valid_window) < self.pitch_stability_window // 2:
                        # Pitch undefined - note ending
                        pitch_offset = search_start + i
                        break
                    
                    if np.any(valid_window):
                        window_std = np.std(window[valid_window])
                        if window_std > self.pitch_stability_threshold:
                            # Pitch unstable - note ending
                            pitch_offset = search_start + i
                            break
        
        # Method 3: Next onset boundary
        next_onset_offset = None
        if next_onset:
            next_onset_offset = next_onset.frame
        
        # Choose best offset
        candidates = []
        
        if energy_offset is not None:
            candidates.append(('energy_decay', energy_offset, 0.8))
        
        if pitch_offset is not None:
            candidates.append(('pitch_change', pitch_offset, 0.7))
        
        if next_onset_offset is not None:
            candidates.append(('next_onset', next_onset_offset, 0.6))
        
        # Fallback: maximum ring time
        max_offset = onset_frame + self.max_ring_frames
        candidates.append(('max_ring', min(max_offset, len(rms) - 1), 0.4))
        
        # Sort by frame (earliest first)
        candidates.sort(key=lambda x: x[1])
        
        # Choose earliest reasonable offset
        for method, frame, confidence in candidates:
            # Ensure minimum duration
            if frame - onset_frame >= self.min_note_frames:
                return OffsetInfo(
                    time=librosa.frames_to_time(frame, sr=self.sr, hop_length=self.hop_length),
                    frame=frame,
                    method=method,
                    confidence=confidence
                )
        
        # Last resort
        offset_frame = onset_frame + self.min_note_frames
        return OffsetInfo(
            time=librosa.frames_to_time(offset_frame, sr=self.sr, hop_length=self.hop_length),
            frame=offset_frame,
            method='min_duration',
            confidence=0.5
        )


class AttackPitchSampler:
    """
    Sample pitch at the optimal moment during note attack.
    
    Key insight: Pitch should be sampled DURING the attack, not in the decay.
    
    Problems with naive pitch sampling:
    1. Sampling too early → attack transient, pitch undefined
    2. Sampling too late → pitch may have drifted (vibrato, bend)
    3. Sampling wrong octave → octave errors due to harmonic confusion
    
    Solution:
    1. Wait for attack transient to settle (3-10ms)
    2. Sample pitch during stable portion of attack (10-50ms after onset)
    3. Use multiple pitch estimators and vote
    4. Verify octave using energy distribution
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,
        # Attack timing
        attack_settle_ms: float = 5,   # Wait for transient to settle
        sample_window_ms: float = 40,  # Window to sample pitch
        # Pitch detection
        min_confidence: float = 0.4,
        # Octave verification
        verify_octave: bool = True,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.attack_settle_frames = int(attack_settle_ms / 1000 * sr / hop_length)
        self.sample_window_frames = int(sample_window_ms / 1000 * sr / hop_length)
        self.min_confidence = min_confidence
        self.verify_octave = verify_octave
    
    def sample(
        self,
        onset: OnsetInfo,
        f0: np.ndarray,
        confidence: np.ndarray,
        S: Optional[np.ndarray] = None,  # Spectrogram for octave verification
        verbose: bool = False
    ) -> Tuple[Optional[int], float, float]:
        """
        Sample pitch at optimal time during attack.
        
        Returns:
            (midi_note, confidence, sample_time) or (None, 0, 0) if no valid pitch
        """
        onset_frame = onset.frame
        
        # Define sampling window (after attack settles)
        window_start = onset_frame + self.attack_settle_frames
        window_end = window_start + self.sample_window_frames
        
        if window_start >= len(f0):
            return None, 0.0, 0.0
        
        window_end = min(window_end, len(f0))
        
        # Get pitches and confidences in window
        f0_window = f0[window_start:window_end]
        conf_window = confidence[window_start:window_end]
        
        # Find valid pitches
        valid_mask = (f0_window > 0) & (conf_window >= self.min_confidence)
        
        if not np.any(valid_mask):
            # Try with lower threshold - still need SOME pitch
            valid_mask = f0_window > 0
            if not np.any(valid_mask):
                return None, 0.0, 0.0
        
        valid_f0 = f0_window[valid_mask]
        valid_conf = conf_window[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Convert to MIDI
        valid_midi = np.round(librosa.hz_to_midi(valid_f0)).astype(int)
        
        # Filter to guitar range
        guitar_mask = (valid_midi >= GUITAR_MIN_MIDI) & (valid_midi <= GUITAR_MAX_MIDI)
        if not np.any(guitar_mask):
            return None, 0.0, 0.0
        
        valid_midi = valid_midi[guitar_mask]
        valid_conf = valid_conf[guitar_mask]
        valid_f0 = valid_f0[guitar_mask]
        valid_indices = valid_indices[guitar_mask]
        
        # Vote on MIDI note (weighted by confidence)
        midi_votes = Counter()
        midi_max_conf = {}  # Track max confidence per MIDI note
        for midi, conf in zip(valid_midi, valid_conf):
            midi_votes[midi] += conf
            midi_max_conf[midi] = max(midi_max_conf.get(midi, 0), conf)
        
        if not midi_votes:
            return None, 0.0, 0.0
        
        # Get most confident pitch
        best_midi = midi_votes.most_common(1)[0][0]
        
        # Calculate confidence: use max confidence for this pitch (not average)
        # This gives more meaningful confidence values
        best_confidence = midi_max_conf[best_midi]
        
        # Boost confidence based on vote count (consistency)
        matching_count = np.sum(valid_midi == best_midi)
        consistency_boost = min(0.2, matching_count * 0.05)
        best_confidence = min(1.0, best_confidence + consistency_boost)
        
        # Calculate sample time (median of valid samples)
        matching_indices = valid_indices[valid_midi == best_midi]
        median_idx = int(np.median(matching_indices))
        sample_frame = window_start + median_idx
        sample_time = librosa.frames_to_time(sample_frame, sr=self.sr, hop_length=self.hop_length)
        
        # Octave verification using spectral energy
        if self.verify_octave and S is not None:
            best_midi = self._verify_octave(best_midi, sample_frame, S)
        
        return best_midi, float(best_confidence), sample_time
    
    def _verify_octave(self, midi: int, frame: int, S: np.ndarray) -> int:
        """
        Verify octave by checking spectral energy distribution.
        
        If most energy is in harmonics rather than fundamental,
        we may have detected a harmonic instead of fundamental.
        """
        if frame >= S.shape[1]:
            return midi
        
        frame_spectrum = S[:, frame]
        
        # Get frequency bins
        n_fft = (S.shape[0] - 1) * 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Expected fundamental frequency
        f0_expected = librosa.midi_to_hz(midi)
        
        # Check if there's more energy at f0/2 (we detected octave harmonic)
        f0_half = f0_expected / 2
        
        if f0_half < GUITAR_MIN_HZ:
            return midi  # Can't go lower
        
        # Find bins for f0 and f0/2
        def get_energy_around_freq(freq, width_hz=20):
            low = freq - width_hz
            high = freq + width_hz
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                return np.sum(frame_spectrum[mask])
            return 0
        
        energy_f0 = get_energy_around_freq(f0_expected)
        energy_f0_half = get_energy_around_freq(f0_half)
        
        # If lower octave has significant energy, might be the fundamental
        if energy_f0_half > energy_f0 * 0.5:
            # Check if lower note is playable on guitar
            lower_midi = midi - 12
            if lower_midi >= GUITAR_MIN_MIDI:
                return lower_midi
        
        return midi


class NoteSegmenter:
    """
    Main class for precise note segmentation.
    
    Combines:
    - Precise onset detection
    - Note-off detection
    - Attack-time pitch sampling
    - Overlap handling
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,  # Smaller hop for precision
        # Onset parameters
        onset_threshold: float = 0.3,
        min_onset_gap_ms: float = 30,
        # Offset parameters
        energy_decay_threshold: float = 0.1,
        # Pitch parameters
        min_confidence: float = 0.3,  # Lower default for more notes
        # Note filtering
        min_note_duration_ms: float = 50,
        # Deduplication
        merge_same_pitch_ms: float = 100,  # Merge notes of same pitch within this window
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.min_note_duration = min_note_duration_ms / 1000
        self.merge_same_pitch_window = merge_same_pitch_ms / 1000
        
        # Initialize detectors
        self.onset_detector = PreciseOnsetDetector(
            sr=sr,
            hop_length=hop_length,
            onset_threshold=onset_threshold,
            min_onset_gap_ms=min_onset_gap_ms
        )
        
        self.offset_detector = NoteOffsetDetector(
            sr=sr,
            hop_length=hop_length,
            energy_decay_threshold=energy_decay_threshold,
            min_note_duration_ms=min_note_duration_ms
        )
        
        self.pitch_sampler = AttackPitchSampler(
            sr=sr,
            hop_length=hop_length,
            min_confidence=min_confidence
        )
    
    def segment(
        self,
        y: np.ndarray,
        use_harmonic_separation: bool = True,
        verbose: bool = True
    ) -> List[NoteSegment]:
        """
        Segment audio into precise note boundaries.
        
        Args:
            y: Audio signal
            use_harmonic_separation: Use HPSS for cleaner pitch detection
            verbose: Print diagnostic info
            
        Returns:
            List of NoteSegment with accurate boundaries
        """
        if verbose:
            print("🎸 Precise Note Segmentation")
            duration = len(y) / self.sr
            print(f"   Audio: {duration:.2f}s @ {self.sr}Hz")
        
        # Harmonic separation for pitch (optional)
        if use_harmonic_separation:
            if verbose:
                print("   Separating harmonic component...")
            y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)
        else:
            y_harmonic = y
        
        # Step 1: Detect onsets
        if verbose:
            print("\n   Step 1: Onset Detection")
        onsets = self.onset_detector.detect(y, verbose=verbose)
        
        if len(onsets) == 0:
            if verbose:
                print("   ⚠️  No onsets detected")
            return []
        
        # Step 2: Compute features for offset detection and pitch sampling
        if verbose:
            print("\n   Step 2: Computing features...")
        
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Use pYIN as primary pitch detector
        f0_pyin, voiced_flag, voiced_probs_pyin = librosa.pyin(
            y_harmonic,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        # Also use YIN for comparison (more stable but less accurate)
        f0_yin = librosa.yin(
            y_harmonic,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Combine: use pYIN where confident, fall back to YIN
        n_frames = min(len(f0_pyin), len(f0_yin))
        f0 = np.zeros(n_frames)
        voiced_probs = np.zeros(n_frames)
        
        for i in range(n_frames):
            if f0_pyin[i] > 0 and voiced_probs_pyin[i] > 0.5:
                # pYIN is confident
                f0[i] = f0_pyin[i]
                voiced_probs[i] = voiced_probs_pyin[i]
            elif f0_yin[i] > 0 and GUITAR_MIN_HZ <= f0_yin[i] <= GUITAR_MAX_HZ:
                # Fall back to YIN
                f0[i] = f0_yin[i]
                voiced_probs[i] = 0.5  # Default confidence
            elif f0_pyin[i] > 0:
                # Use pYIN even with lower confidence
                f0[i] = f0_pyin[i]
                voiced_probs[i] = max(0.3, voiced_probs_pyin[i])
        
        # Apply median filter for stability
        f0_filtered = self._median_filter_f0(f0, kernel_size=5)
        
        # Compute spectrogram for octave verification
        S = np.abs(librosa.stft(y_harmonic, hop_length=self.hop_length))
        
        # Step 3: For each onset, detect offset and sample pitch
        if verbose:
            print("\n   Step 3: Segmenting notes...")
        
        notes = []
        
        for i, onset in enumerate(onsets):
            # Get next onset (if any)
            next_onset = onsets[i + 1] if i + 1 < len(onsets) else None
            
            # Sample pitch during attack
            midi, pitch_conf, sample_time = self.pitch_sampler.sample(
                onset, f0_filtered, voiced_probs, S, verbose=False
            )
            
            if midi is None:
                continue
            
            # Detect note offset
            offset = self.offset_detector.detect(
                y, onset, next_onset, f0_filtered, rms, verbose=False
            )
            
            # Verify duration
            duration = offset.time - onset.backtracked_time
            if duration < self.min_note_duration:
                continue
            
            # Create note segment
            note = NoteSegment(
                midi=midi,
                hz=librosa.midi_to_hz(midi),
                start_time=onset.backtracked_time,
                end_time=offset.time,
                confidence=pitch_conf,
                onset_method=','.join(onset.methods),
                offset_method=offset.method,
                pitch_sample_time=sample_time,
                attack_strength=onset.attack_strength,
                is_legato=onset.is_legato
            )
            
            notes.append(note)
        
        # Step 4: Merge consecutive same-pitch notes
        if verbose:
            print("\n   Step 4: Merging consecutive same-pitch notes...")
        notes = self._merge_same_pitch(notes)
        
        # Step 5: Detect overlapping notes
        if verbose:
            print("\n   Step 5: Detecting overlaps...")
        notes = self._detect_overlaps(notes)
        
        # Step 6: Filter harmonic duplicates
        if verbose:
            print("\n   Step 6: Filtering harmonics...")
        notes = self._filter_harmonics(notes)
        
        # Step 7: Filter low-confidence notes (after merging)
        if verbose:
            print("\n   Step 7: Final confidence filter...")
        notes = [n for n in notes if n.confidence >= 0.1]  # Very low floor
        
        if verbose:
            print(f"\n   ✅ Found {len(notes)} notes")
            n_legato = sum(1 for n in notes if n.is_legato)
            n_overlaps = sum(1 for n in notes if len(n.overlaps_with) > 0)
            if n_legato > 0:
                print(f"      {n_legato} legato (hammer-ons/pull-offs)")
            if n_overlaps > 0:
                print(f"      {n_overlaps} with overlaps (ringing notes)")
        
        return notes
    
    def _median_filter_f0(self, f0: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter to f0, only on valid values."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        valid_mask = f0 > 0
        if not np.any(valid_mask):
            return f0
        
        filtered = medfilt(f0, kernel_size=kernel_size)
        result = np.where(valid_mask, filtered, f0)
        
        return result
    
    def _merge_same_pitch(self, notes: List[NoteSegment]) -> List[NoteSegment]:
        """
        Merge consecutive notes of the same pitch that are close together.
        
        This handles cases where the same note is detected multiple times
        due to onset detection sensitivity or pitch instability.
        """
        if len(notes) <= 1:
            return notes
        
        # Sort by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        merged = []
        
        i = 0
        while i < len(sorted_notes):
            current = sorted_notes[i]
            
            # Look for notes to merge
            j = i + 1
            while j < len(sorted_notes):
                next_note = sorted_notes[j]
                
                # Check if same pitch and close enough
                same_pitch = current.midi == next_note.midi
                close_enough = next_note.start_time - current.end_time < self.merge_same_pitch_window
                overlapping = next_note.start_time < current.end_time
                
                if same_pitch and (close_enough or overlapping):
                    # Merge: extend current note, take max confidence
                    current = NoteSegment(
                        midi=current.midi,
                        hz=current.hz,
                        start_time=current.start_time,
                        end_time=max(current.end_time, next_note.end_time),
                        confidence=max(current.confidence, next_note.confidence),
                        onset_method=current.onset_method,
                        offset_method=next_note.offset_method,  # Use later note's offset
                        pitch_sample_time=current.pitch_sample_time,
                        attack_strength=current.attack_strength,
                        is_legato=current.is_legato,
                        overlaps_with=[]
                    )
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        return merged
    
    def _detect_overlaps(self, notes: List[NoteSegment]) -> List[NoteSegment]:
        """
        Detect when notes overlap (ring together).
        
        On guitar, notes can ring beyond the next onset, especially
        on different strings.
        """
        n = len(notes)
        
        for i in range(n):
            note_i = notes[i]
            
            for j in range(i + 1, n):
                note_j = notes[j]
                
                # Check if j starts before i ends
                if note_j.start_time < note_i.end_time:
                    # Check if they're on different strings (different pitches)
                    # Allow overlaps if pitches are > 2 semitones apart
                    if abs(note_i.midi - note_j.midi) > 2:
                        note_i.overlaps_with.append(j)
                        note_j.overlaps_with.append(i)
        
        return notes
    
    def _filter_harmonics(
        self, 
        notes: List[NoteSegment], 
        time_threshold: float = 0.05
    ) -> List[NoteSegment]:
        """
        Filter out notes that are likely harmonics of other notes.
        
        If two notes start at nearly the same time and one is an
        octave/fifth above, keep only the lower (fundamental).
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
                # Keep notes that aren't harmonics of others
                kept = []
                
                for note in group:
                    is_harmonic = False
                    for other in group:
                        if other.midi < note.midi:
                            diff = note.midi - other.midi
                            # Harmonic intervals: octave (12), octave+fifth (19), 2 octaves (24)
                            if diff in [12, 19, 24, 28, 31]:
                                is_harmonic = True
                                break
                    
                    if not is_harmonic:
                        kept.append(note)
                
                filtered.extend(kept if kept else [group[0]])
            
            i = j if j > i else i + 1
        
        return filtered


def segment_notes(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 256,
    onset_threshold: float = 0.3,
    min_confidence: float = 0.4,
    min_note_duration_ms: float = 50,
    use_harmonic_separation: bool = True,
    verbose: bool = True
) -> List[NoteSegment]:
    """
    Convenience function for precise note segmentation.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length: Hop length for analysis
        onset_threshold: Onset detection threshold
        min_confidence: Minimum pitch confidence
        min_note_duration_ms: Minimum note duration in ms
        use_harmonic_separation: Use HPSS for cleaner pitch
        verbose: Print diagnostic info
        
    Returns:
        List of NoteSegment with precise boundaries
    """
    # Load audio
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    
    # Create segmenter
    segmenter = NoteSegmenter(
        sr=sr,
        hop_length=hop_length,
        onset_threshold=onset_threshold,
        min_confidence=min_confidence,
        min_note_duration_ms=min_note_duration_ms
    )
    
    # Segment
    notes = segmenter.segment(y, use_harmonic_separation=use_harmonic_separation, verbose=verbose)
    
    return notes


def convert_to_notes(segments: List[NoteSegment]) -> List['Note']:
    """
    Convert NoteSegments to guitar_tabs.Note objects for compatibility.
    """
    # Import here to avoid circular imports
    from guitar_tabs import Note
    
    notes = []
    for seg in segments:
        notes.append(Note(
            midi=seg.midi,
            start_time=seg.start_time,
            duration=seg.duration,
            confidence=seg.confidence
        ))
    
    return notes


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python note_segmentation.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"\n🎵 Processing: {audio_path}\n")
    
    notes = segment_notes(audio_path, verbose=True)
    
    print("\n📋 Detected Notes:")
    print("-" * 60)
    
    for i, note in enumerate(notes):
        overlap_str = f" [overlaps: {note.overlaps_with}]" if note.overlaps_with else ""
        legato_str = " (legato)" if note.is_legato else ""
        print(f"  {i+1:3d}. {note.note_name:4s}  {note.start_time:6.3f}s - {note.end_time:6.3f}s  "
              f"(dur: {note.duration:.3f}s, conf: {note.confidence:.2f}){legato_str}{overlap_str}")
    
    print(f"\n   Total: {len(notes)} notes")
