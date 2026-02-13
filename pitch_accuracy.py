#!/usr/bin/env python3
"""
Pitch Accuracy Module - Multi-detector voting system for accurate pitch detection.

This module implements:
1. Multiple pitch detection methods (pYIN, piptrack, CQT, YIN)
2. Consensus voting across detectors
3. Confidence-weighted voting
4. Guitar range filtering (E2=40 to E6=88 MIDI)
5. Pitch continuity tracking (prevent random jumps)
6. Octave disambiguation

The goal is ACCURATE pitches, not just more notes.
"""

import numpy as np
import librosa
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import warnings

# Guitar MIDI range
GUITAR_MIN_MIDI = 40  # E2
GUITAR_MAX_MIDI = 88  # E6 (highest practical guitar note)
GUITAR_MIN_HZ = 82.4   # E2
GUITAR_MAX_HZ = 1320   # E6

# Pitch class names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class PitchCandidate:
    """A pitch candidate from one detector."""
    midi: int
    hz: float
    confidence: float
    detector: str
    frame: int


@dataclass
class VotedPitch:
    """A pitch after consensus voting."""
    midi: int
    hz: float
    confidence: float
    votes: int
    detectors: List[str]
    frame: int
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)


@dataclass
class AccuratePitch:
    """Final pitch after continuity filtering."""
    midi: int
    hz: float
    confidence: float
    frame: int
    start_time: float
    is_valid: bool = True
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)


class PitchDetectorPYIN:
    """pYIN pitch detector - probabilistic YIN with HMM smoothing."""
    
    name = "pyin"
    
    def __init__(self, sr: int, hop_length: int, fmin: float = GUITAR_MIN_HZ, fmax: float = GUITAR_MAX_HZ):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def detect(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (f0, confidence) arrays."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        # Use voiced probability as confidence
        confidence = np.where(f0 > 0, voiced_probs, 0.0)
        return f0, confidence


class PitchDetectorPiptrack:
    """piptrack pitch detector - parabolic interpolation on STFT peaks."""
    
    name = "piptrack"
    
    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048,
                 fmin: float = GUITAR_MIN_HZ, fmax: float = GUITAR_MAX_HZ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
    
    def detect(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (f0, confidence) arrays."""
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            n_fft=self.n_fft
        )
        
        n_frames = pitches.shape[1]
        f0 = np.zeros(n_frames)
        confidence = np.zeros(n_frames)
        
        # Get dominant pitch per frame
        max_mag = magnitudes.max() + 1e-10
        for i in range(n_frames):
            mag_frame = magnitudes[:, i]
            pitch_frame = pitches[:, i]
            
            # Find strongest pitch
            if mag_frame.max() > 0:
                idx = mag_frame.argmax()
                f0[i] = pitch_frame[idx]
                confidence[i] = mag_frame[idx] / max_mag
        
        return f0, confidence


class PitchDetectorCQT:
    """CQT-based pitch detector - constant-Q transform peak picking."""
    
    name = "cqt"
    
    def __init__(self, sr: int, hop_length: int, n_bins: int = 84,
                 bins_per_octave: int = 12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = librosa.note_to_hz('C2')  # ~65 Hz
    
    def detect(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (f0, confidence) arrays."""
        C = np.abs(librosa.cqt(
            y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin
        ))
        
        # Get CQT frequencies
        cqt_freqs = librosa.cqt_frequencies(
            n_bins=self.n_bins,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave
        )
        
        n_frames = C.shape[1]
        f0 = np.zeros(n_frames)
        confidence = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = C[:, i]
            max_mag = frame.max()
            
            if max_mag > 1e-10:
                # Find peak
                peak_idx = frame.argmax()
                peak_freq = cqt_freqs[peak_idx]
                
                # Only accept if within guitar range
                if GUITAR_MIN_HZ <= peak_freq <= GUITAR_MAX_HZ:
                    f0[i] = peak_freq
                    confidence[i] = frame[peak_idx] / max_mag
        
        return f0, confidence


class PitchDetectorYIN:
    """YIN pitch detector - classic autocorrelation-based method."""
    
    name = "yin"
    
    def __init__(self, sr: int, hop_length: int, frame_length: int = 2048,
                 fmin: float = GUITAR_MIN_HZ, fmax: float = GUITAR_MAX_HZ):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
    
    def detect(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (f0, confidence) arrays."""
        f0 = librosa.yin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=self.frame_length
        )
        
        # YIN doesn't give confidence directly, estimate from stability
        # Use local variance as inverse confidence proxy
        confidence = np.ones_like(f0) * 0.7
        
        # Zero confidence for zero pitch
        confidence[f0 <= 0] = 0.0
        
        # Filter to guitar range
        out_of_range = (f0 < self.fmin) | (f0 > self.fmax)
        f0[out_of_range] = 0.0
        confidence[out_of_range] = 0.0
        
        return f0, confidence


class AccuratePitchDetector:
    """
    Multi-detector pitch system with voting and continuity tracking.
    
    This is the main class that combines multiple detectors for accurate pitch detection.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        # Voting parameters
        min_votes: int = 2,
        semitone_tolerance: int = 1,  # Allow ±1 semitone for consensus
        # Confidence parameters
        min_confidence: float = 0.3,
        confidence_weights: Dict[str, float] = None,
        # Continuity parameters
        max_jump_semitones: int = 12,  # Max pitch jump in semitones
        continuity_window: int = 5,    # Frames to consider for continuity
        # Guitar range
        min_midi: int = GUITAR_MIN_MIDI,
        max_midi: int = GUITAR_MAX_MIDI,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.min_votes = min_votes
        self.semitone_tolerance = semitone_tolerance
        self.min_confidence = min_confidence
        self.max_jump_semitones = max_jump_semitones
        self.continuity_window = continuity_window
        self.min_midi = min_midi
        self.max_midi = max_midi
        
        # Default confidence weights - pYIN and CQT are generally more reliable
        self.confidence_weights = confidence_weights or {
            'pyin': 1.0,
            'cqt': 0.9,
            'piptrack': 0.7,
            'yin': 0.6,
        }
        
        # Initialize detectors
        self.detectors = [
            PitchDetectorPYIN(sr, hop_length),
            PitchDetectorCQT(sr, hop_length),
            PitchDetectorPiptrack(sr, hop_length),
            PitchDetectorYIN(sr, hop_length),
        ]
    
    def detect(self, y: np.ndarray, verbose: bool = True) -> List[AccuratePitch]:
        """
        Detect pitches using multi-detector voting with continuity tracking.
        
        Args:
            y: Audio signal
            verbose: Print diagnostic info
            
        Returns:
            List of AccuratePitch objects for each frame
        """
        if verbose:
            print("🎯 Accurate Pitch Detection with Multi-Detector Voting")
            print(f"   Min votes: {self.min_votes}, Semitone tolerance: ±{self.semitone_tolerance}")
        
        # Step 1: Run all detectors
        detector_results = {}
        for detector in self.detectors:
            try:
                f0, conf = detector.detect(y)
                detector_results[detector.name] = (f0, conf)
                if verbose:
                    valid = f0 > 0
                    print(f"   {detector.name}: {sum(valid)}/{len(f0)} voiced frames")
            except Exception as e:
                if verbose:
                    print(f"   {detector.name}: FAILED - {e}")
        
        if len(detector_results) < 2:
            raise RuntimeError("Not enough detectors succeeded for voting")
        
        # Determine number of frames
        n_frames = min(len(f0) for f0, _ in detector_results.values())
        
        # Step 2: Collect candidates per frame
        frame_candidates = self._collect_candidates(detector_results, n_frames)
        
        # Step 3: Vote on each frame
        voted_pitches = self._vote_on_frames(frame_candidates, verbose)
        
        # Step 4: Apply continuity filtering
        accurate_pitches = self._apply_continuity(voted_pitches, verbose)
        
        # Step 5: Filter by confidence
        final_pitches = [p for p in accurate_pitches if p.confidence >= self.min_confidence]
        
        if verbose:
            valid_count = sum(1 for p in final_pitches if p.is_valid and p.midi > 0)
            print(f"   Final: {valid_count} valid pitched frames")
        
        return final_pitches
    
    def _collect_candidates(
        self,
        detector_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        n_frames: int
    ) -> List[List[PitchCandidate]]:
        """Collect pitch candidates from all detectors for each frame."""
        frame_candidates = [[] for _ in range(n_frames)]
        
        for detector_name, (f0, conf) in detector_results.items():
            for frame in range(min(n_frames, len(f0))):
                if f0[frame] > 0:
                    midi = int(round(librosa.hz_to_midi(f0[frame])))
                    
                    # Filter to guitar range
                    if self.min_midi <= midi <= self.max_midi:
                        frame_candidates[frame].append(PitchCandidate(
                            midi=midi,
                            hz=f0[frame],
                            confidence=conf[frame],
                            detector=detector_name,
                            frame=frame
                        ))
        
        return frame_candidates
    
    def _vote_on_frames(
        self,
        frame_candidates: List[List[PitchCandidate]],
        verbose: bool
    ) -> List[Optional[VotedPitch]]:
        """Apply voting to select best pitch per frame."""
        voted = []
        
        for frame, candidates in enumerate(frame_candidates):
            if not candidates:
                voted.append(None)
                continue
            
            # Group candidates by MIDI note (with tolerance)
            groups = self._group_by_pitch(candidates)
            
            # Find best group
            best_group = None
            best_score = 0
            
            for midi, group_candidates in groups.items():
                # Score = weighted votes + weighted confidence
                votes = len(group_candidates)
                weighted_conf = sum(
                    c.confidence * self.confidence_weights.get(c.detector, 1.0)
                    for c in group_candidates
                )
                
                score = votes + weighted_conf * 0.5
                
                if score > best_score and votes >= self.min_votes:
                    best_score = score
                    best_group = (midi, group_candidates)
            
            if best_group is None:
                # No consensus - try with lower vote threshold if high confidence
                for midi, group_candidates in groups.items():
                    max_conf = max(c.confidence for c in group_candidates)
                    if max_conf >= 0.7:  # High confidence single detector
                        best_group = (midi, group_candidates)
                        break
            
            if best_group:
                midi, group_candidates = best_group
                # Weighted average frequency
                total_weight = sum(
                    c.confidence * self.confidence_weights.get(c.detector, 1.0)
                    for c in group_candidates
                )
                weighted_hz = sum(
                    c.hz * c.confidence * self.confidence_weights.get(c.detector, 1.0)
                    for c in group_candidates
                ) / (total_weight + 1e-10)
                
                # Average confidence
                avg_conf = np.mean([c.confidence for c in group_candidates])
                
                voted.append(VotedPitch(
                    midi=midi,
                    hz=weighted_hz,
                    confidence=avg_conf,
                    votes=len(group_candidates),
                    detectors=[c.detector for c in group_candidates],
                    frame=frame
                ))
            else:
                voted.append(None)
        
        return voted
    
    def _group_by_pitch(
        self,
        candidates: List[PitchCandidate]
    ) -> Dict[int, List[PitchCandidate]]:
        """Group candidates by MIDI note with tolerance."""
        groups: Dict[int, List[PitchCandidate]] = {}
        
        for candidate in candidates:
            # Find existing group within tolerance
            matched = False
            for group_midi in list(groups.keys()):
                if abs(candidate.midi - group_midi) <= self.semitone_tolerance:
                    groups[group_midi].append(candidate)
                    matched = True
                    break
            
            if not matched:
                groups[candidate.midi] = [candidate]
        
        return groups
    
    def _apply_continuity(
        self,
        voted: List[Optional[VotedPitch]],
        verbose: bool
    ) -> List[AccuratePitch]:
        """Apply pitch continuity tracking to remove outliers."""
        n_frames = len(voted)
        frame_duration = self.hop_length / self.sr
        
        accurate = []
        
        for i, pitch in enumerate(voted):
            if pitch is None:
                accurate.append(AccuratePitch(
                    midi=0,
                    hz=0.0,
                    confidence=0.0,
                    frame=i,
                    start_time=i * frame_duration,
                    is_valid=False
                ))
                continue
            
            # Check continuity with neighbors
            is_valid = True
            
            # Look at previous frames
            prev_pitches = []
            for j in range(max(0, i - self.continuity_window), i):
                if voted[j] is not None:
                    prev_pitches.append(voted[j].midi)
            
            # Look at next frames
            next_pitches = []
            for j in range(i + 1, min(n_frames, i + self.continuity_window + 1)):
                if voted[j] is not None:
                    next_pitches.append(voted[j].midi)
            
            # Check if this pitch is an outlier
            if prev_pitches or next_pitches:
                neighbors = prev_pitches + next_pitches
                
                if neighbors:
                    # Check if current pitch is too different from median of neighbors
                    median_neighbor = int(np.median(neighbors))
                    jump = abs(pitch.midi - median_neighbor)
                    
                    if jump > self.max_jump_semitones:
                        # Likely an octave error or glitch
                        is_valid = False
                        
                        # Try to correct octave errors
                        octave_corrections = [pitch.midi - 12, pitch.midi + 12]
                        for corrected in octave_corrections:
                            if abs(corrected - median_neighbor) <= 3:
                                # Accept the correction
                                pitch = VotedPitch(
                                    midi=corrected,
                                    hz=pitch.hz * (2 if corrected > pitch.midi else 0.5),
                                    confidence=pitch.confidence * 0.8,
                                    votes=pitch.votes,
                                    detectors=pitch.detectors,
                                    frame=pitch.frame
                                )
                                is_valid = True
                                break
            
            # Verify guitar range one more time
            if pitch.midi < self.min_midi or pitch.midi > self.max_midi:
                is_valid = False
            
            accurate.append(AccuratePitch(
                midi=pitch.midi,
                hz=pitch.hz,
                confidence=pitch.confidence,
                frame=i,
                start_time=i * frame_duration,
                is_valid=is_valid
            ))
        
        # Second pass: fill gaps with interpolation if short
        accurate = self._fill_short_gaps(accurate)
        
        return accurate
    
    def _fill_short_gaps(self, pitches: List[AccuratePitch], max_gap: int = 3) -> List[AccuratePitch]:
        """Fill short gaps in pitch track with interpolation."""
        n = len(pitches)
        result = pitches.copy()
        
        i = 0
        while i < n:
            if not pitches[i].is_valid or pitches[i].midi == 0:
                # Find gap length
                gap_start = i
                while i < n and (not pitches[i].is_valid or pitches[i].midi == 0):
                    i += 1
                gap_end = i
                gap_len = gap_end - gap_start
                
                # Fill short gaps
                if gap_len <= max_gap and gap_start > 0 and gap_end < n:
                    prev_midi = pitches[gap_start - 1].midi
                    next_midi = pitches[gap_end].midi
                    
                    # Only fill if pitches are close
                    if prev_midi > 0 and next_midi > 0 and abs(prev_midi - next_midi) <= 2:
                        fill_midi = int(round((prev_midi + next_midi) / 2))
                        fill_hz = librosa.midi_to_hz(fill_midi)
                        
                        for j in range(gap_start, gap_end):
                            result[j] = AccuratePitch(
                                midi=fill_midi,
                                hz=fill_hz,
                                confidence=0.5,
                                frame=j,
                                start_time=pitches[j].start_time,
                                is_valid=True
                            )
            else:
                i += 1
        
        return result


def detect_pitches_accurate(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_votes: int = 2,
    min_confidence: float = 0.3,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[AccuratePitch]]:
    """
    Convenience function for accurate pitch detection.
    
    Returns:
        f0: Frequency array (Hz)
        confidence: Confidence array
        pitch_details: List of AccuratePitch with full details
    """
    detector = AccuratePitchDetector(
        sr=sr,
        hop_length=hop_length,
        min_votes=min_votes,
        min_confidence=min_confidence
    )
    
    pitches = detector.detect(y, verbose=verbose)
    
    # Convert to arrays
    f0 = np.array([p.hz if p.is_valid else 0.0 for p in pitches])
    confidence = np.array([p.confidence if p.is_valid else 0.0 for p in pitches])
    
    return f0, confidence, pitches


def compare_detectors(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare all available pitch detectors on an audio file.
    
    Returns statistics for each detector.
    """
    import librosa
    
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    
    if verbose:
        print(f"\n📊 Pitch Detector Comparison")
        print(f"   Audio: {audio_path}")
        print(f"   Duration: {duration:.2f}s, Sample rate: {sr}Hz")
        print()
    
    results = {}
    
    detectors = [
        ("pYIN", PitchDetectorPYIN(sr, hop_length)),
        ("CQT", PitchDetectorCQT(sr, hop_length)),
        ("piptrack", PitchDetectorPiptrack(sr, hop_length)),
        ("YIN", PitchDetectorYIN(sr, hop_length)),
    ]
    
    for name, detector in detectors:
        try:
            f0, conf = detector.detect(y)
            
            valid = f0 > 0
            n_valid = sum(valid)
            
            if n_valid > 0:
                valid_f0 = f0[valid]
                valid_conf = conf[valid]
                valid_midi = librosa.hz_to_midi(valid_f0)
                
                results[name] = {
                    'voiced_frames': n_valid,
                    'total_frames': len(f0),
                    'voiced_pct': 100 * n_valid / len(f0),
                    'min_hz': float(valid_f0.min()),
                    'max_hz': float(valid_f0.max()),
                    'mean_hz': float(valid_f0.mean()),
                    'min_midi': int(round(valid_midi.min())),
                    'max_midi': int(round(valid_midi.max())),
                    'mean_conf': float(valid_conf.mean()),
                }
                
                if verbose:
                    r = results[name]
                    print(f"   {name}:")
                    print(f"      Voiced: {r['voiced_frames']}/{r['total_frames']} ({r['voiced_pct']:.1f}%)")
                    print(f"      Range: {r['min_hz']:.1f}Hz - {r['max_hz']:.1f}Hz (MIDI {r['min_midi']}-{r['max_midi']})")
                    print(f"      Mean confidence: {r['mean_conf']:.2f}")
            else:
                results[name] = {'error': 'No pitches detected'}
                if verbose:
                    print(f"   {name}: No pitches detected")
                    
        except Exception as e:
            results[name] = {'error': str(e)}
            if verbose:
                print(f"   {name}: ERROR - {e}")
    
    # Run voting detector
    if verbose:
        print()
        print("   Running consensus voting...")
    
    f0, conf, pitches = detect_pitches_accurate(y, sr, hop_length, verbose=verbose)
    
    valid = f0 > 0
    n_valid = sum(valid)
    
    if n_valid > 0:
        valid_f0 = f0[valid]
        valid_midi = librosa.hz_to_midi(valid_f0)
        
        results['VOTING'] = {
            'voiced_frames': n_valid,
            'total_frames': len(f0),
            'voiced_pct': 100 * n_valid / len(f0),
            'min_hz': float(valid_f0.min()),
            'max_hz': float(valid_f0.max()),
            'min_midi': int(round(valid_midi.min())),
            'max_midi': int(round(valid_midi.max())),
            'mean_conf': float(conf[valid].mean()),
        }
        
        if verbose:
            r = results['VOTING']
            print()
            print(f"   🎯 VOTING CONSENSUS:")
            print(f"      Voiced: {r['voiced_frames']}/{r['total_frames']} ({r['voiced_pct']:.1f}%)")
            print(f"      Range: {r['min_hz']:.1f}Hz - {r['max_hz']:.1f}Hz (MIDI {r['min_midi']}-{r['max_midi']})")
            print(f"      Mean confidence: {r['mean_conf']:.2f}")
    
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        compare_detectors(sys.argv[1])
    else:
        print("Usage: python pitch_accuracy.py <audio_file>")
