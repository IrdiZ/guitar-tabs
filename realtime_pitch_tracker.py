#!/usr/bin/env python3
"""
Real-Time Frame-by-Frame Pitch Tracker with Viterbi Smoothing

Implements continuous pitch tracking optimized for guitar:
1. Process audio in small frames (10-20ms)
2. Track pitch continuity across frames using Viterbi algorithm
3. Penalize unrealistic pitch jumps
4. Model guitar physics (pitch can't jump 2 octaves instantly)

The key insight: pitch detection is a sequence problem. Individual frames
may have ambiguous or noisy pitch estimates, but smooth continuous tracking
dramatically reduces errors by finding the globally optimal path.

This module provides:
- ViterbiPitchTracker: Main pitch tracker using HMM/Viterbi
- GuitarPhysicsModel: Models realistic guitar pitch transitions
- FramePitchExtractor: Extracts pitch candidates per frame
- PitchPath: Final smoothed pitch trajectory
"""

import numpy as np
import librosa
from scipy.signal import medfilt, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings


# Guitar frequency/MIDI range
GUITAR_MIN_HZ = 75    # Below E2 to catch drop tunings
GUITAR_MAX_HZ = 1400  # High frets on high E
GUITAR_MIN_MIDI = 38  # D2 (drop D low)
GUITAR_MAX_MIDI = 88  # E6 (highest practical)

# Pitch resolution
CENTS_PER_SEMITONE = 100
SEMITONES_PER_OCTAVE = 12

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class FramePitch:
    """Pitch candidate for a single frame."""
    frame: int
    time: float
    frequency: float      # Hz (0 if unvoiced)
    midi: float           # MIDI note (fractional for pitch bends)
    confidence: float     # 0-1
    is_voiced: bool
    method: str = "pyin"


@dataclass
class PitchPathNode:
    """Node in the Viterbi path."""
    frame: int
    midi: int             # Quantized MIDI note (state)
    log_prob: float       # Log probability at this node
    prev_midi: Optional[int] = None  # For backtracking
    raw_frequency: float = 0.0
    confidence: float = 0.0


@dataclass
class PitchPath:
    """Final smoothed pitch trajectory."""
    frames: np.ndarray            # Frame indices
    times: np.ndarray             # Time in seconds
    frequencies: np.ndarray       # Hz values (0 = unvoiced)
    midi_notes: np.ndarray        # MIDI values (0 = unvoiced)
    confidences: np.ndarray       # Per-frame confidence
    is_voiced: np.ndarray         # Boolean voiced flags
    
    @property
    def n_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return self.times[-1] - self.times[0]
    
    def get_notes(self, min_duration: float = 0.03) -> List[Dict]:
        """Extract note events from pitch path."""
        notes = []
        in_note = False
        note_start = 0
        note_midi = 0
        note_freqs = []
        note_confs = []
        
        for i in range(len(self.frames)):
            is_voiced = self.is_voiced[i] and self.midi_notes[i] > 0
            midi = int(round(self.midi_notes[i])) if is_voiced else 0
            
            if is_voiced:
                if not in_note:
                    # Start new note
                    in_note = True
                    note_start = i
                    note_midi = midi
                    note_freqs = [self.frequencies[i]]
                    note_confs = [self.confidences[i]]
                elif abs(midi - note_midi) <= 1:
                    # Continue same note (allow ±1 semitone jitter)
                    note_freqs.append(self.frequencies[i])
                    note_confs.append(self.confidences[i])
                else:
                    # New note - save previous
                    duration = self.times[i] - self.times[note_start]
                    if duration >= min_duration:
                        notes.append({
                            'midi': note_midi,
                            'start_time': self.times[note_start],
                            'duration': duration,
                            'confidence': float(np.mean(note_confs)),
                            'frequency': float(np.mean(note_freqs)),
                            'n_frames': len(note_freqs)
                        })
                    # Start new
                    note_start = i
                    note_midi = midi
                    note_freqs = [self.frequencies[i]]
                    note_confs = [self.confidences[i]]
            else:
                if in_note:
                    # End current note
                    duration = self.times[i] - self.times[note_start]
                    if duration >= min_duration:
                        notes.append({
                            'midi': note_midi,
                            'start_time': self.times[note_start],
                            'duration': duration,
                            'confidence': float(np.mean(note_confs)),
                            'frequency': float(np.mean(note_freqs)),
                            'n_frames': len(note_freqs)
                        })
                    in_note = False
        
        # Don't forget last note
        if in_note:
            duration = self.times[-1] - self.times[note_start]
            if duration >= min_duration:
                notes.append({
                    'midi': note_midi,
                    'start_time': self.times[note_start],
                    'duration': duration,
                    'confidence': float(np.mean(note_confs)),
                    'frequency': float(np.mean(note_freqs)),
                    'n_frames': len(note_freqs)
                })
        
        return notes


class GuitarPhysicsModel:
    """
    Model realistic guitar pitch transitions.
    
    Guitar physics constraints:
    - Pitch cannot jump more than ~12 semitones instantly (fret hand can reach)
    - Most transitions are small (1-3 semitones)
    - String changes can cause larger jumps but still limited
    - Bends are typically max 3 semitones (1.5 tones)
    - Vibrato is ±50 cents typically
    
    This model penalizes unrealistic transitions in the Viterbi search.
    """
    
    def __init__(
        self,
        max_jump_semitones: int = 12,       # Max instant jump
        typical_jump_semitones: float = 2,  # Typical melodic motion
        bend_max_semitones: float = 3,      # Max bend
        vibrato_cents: float = 50,          # Typical vibrato width
        frame_duration_ms: float = 15,      # Frame size
    ):
        self.max_jump = max_jump_semitones
        self.typical_jump = typical_jump_semitones
        self.bend_max = bend_max_semitones
        self.vibrato_cents = vibrato_cents
        self.frame_duration_ms = frame_duration_ms
        
        # Compute max rate of change (semitones per frame)
        # Fast passages: ~16 notes/sec = 62.5ms per note
        # So 15ms frame could see 0-1 semitone change typically
        self.max_rate = max_jump_semitones / (100 / frame_duration_ms)
        
        # Pre-compute transition matrix
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """Build log-probability transition matrix."""
        # States are MIDI notes from GUITAR_MIN_MIDI to GUITAR_MAX_MIDI
        n_states = GUITAR_MAX_MIDI - GUITAR_MIN_MIDI + 1
        
        # Transition probabilities based on interval
        # Use exponential decay for larger intervals
        self.trans_log_prob = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(n_states):
                interval = abs(i - j)
                self.trans_log_prob[i, j] = self._interval_log_prob(interval)
    
    def _interval_log_prob(self, semitones: int) -> float:
        """
        Log probability of a pitch transition.
        
        Based on guitar physics:
        - 0-2 semitones: Very common (same position, adjacent frets)
        - 3-5 semitones: Common (position shift, string change)
        - 6-7 semitones: Less common (large shift)
        - 8-12 semitones: Rare (cross-position jump)
        - >12 semitones: Very rare (octave leap) but possible
        - >24 semitones: Essentially impossible in one frame
        """
        if semitones == 0:
            return 0.0  # log(1) - staying same pitch most likely
        elif semitones <= 2:
            return -0.5  # Adjacent semitones very common
        elif semitones <= 5:
            return -1.0 - 0.2 * (semitones - 2)
        elif semitones <= 7:
            return -2.0 - 0.3 * (semitones - 5)
        elif semitones <= 12:
            return -3.0 - 0.5 * (semitones - 7)
        elif semitones <= 24:
            return -6.0 - 1.0 * (semitones - 12)
        else:
            return -30.0  # Essentially impossible
    
    def transition_cost(self, from_midi: int, to_midi: int) -> float:
        """
        Cost of transitioning from one pitch to another.
        Lower is better. Returns negative log probability.
        """
        if from_midi == 0 or to_midi == 0:
            # Transition involving silence
            return 0.5  # Small penalty for voice/unvoice transitions
        
        interval = abs(from_midi - to_midi)
        return -self._interval_log_prob(interval)
    
    def get_transition_log_prob(self, from_midi: int, to_midi: int) -> float:
        """Get log probability of transition."""
        if from_midi == 0 or to_midi == 0:
            return -0.5
        
        # Check range
        if not (GUITAR_MIN_MIDI <= from_midi <= GUITAR_MAX_MIDI):
            return -10.0
        if not (GUITAR_MIN_MIDI <= to_midi <= GUITAR_MAX_MIDI):
            return -10.0
        
        i = from_midi - GUITAR_MIN_MIDI
        j = to_midi - GUITAR_MIN_MIDI
        return self.trans_log_prob[i, j]


class FramePitchExtractor:
    """
    Extract pitch candidates from individual frames.
    
    Uses multiple methods and returns top candidates per frame
    for the Viterbi search to evaluate.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        frame_length_ms: float = 50,   # 50ms frames for pitch detection
        hop_length_ms: float = 10,     # 10ms hop for fine resolution
        n_candidates: int = 3,         # Top candidates per frame
        fmin: float = GUITAR_MIN_HZ,
        fmax: float = GUITAR_MAX_HZ,
    ):
        self.sr = sr
        self.frame_length = int(sr * frame_length_ms / 1000)
        self.hop_length = int(sr * hop_length_ms / 1000)
        self.n_candidates = n_candidates
        self.fmin = fmin
        self.fmax = fmax
        
    def extract_all_frames(self, y: np.ndarray) -> List[List[FramePitch]]:
        """
        Extract pitch candidates for all frames.
        
        Returns: List of frame candidates, each frame has up to n_candidates
        """
        # Run multiple detectors
        pyin_f0, pyin_conf = self._detect_pyin(y)
        cqt_f0, cqt_conf = self._detect_cqt(y)
        yin_f0, yin_conf = self._detect_yin(y)
        
        n_frames = len(pyin_f0)
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=self.sr, hop_length=self.hop_length
        )
        
        all_frame_candidates = []
        
        for i in range(n_frames):
            frame_candidates = []
            
            # Collect candidates from each detector
            if pyin_f0[i] > 0 and pyin_conf[i] > 0.3:
                frame_candidates.append(FramePitch(
                    frame=i, time=times[i],
                    frequency=pyin_f0[i],
                    midi=librosa.hz_to_midi(pyin_f0[i]),
                    confidence=pyin_conf[i],
                    is_voiced=True,
                    method='pyin'
                ))
            
            if cqt_f0[i] > 0 and cqt_conf[i] > 0.3:
                frame_candidates.append(FramePitch(
                    frame=i, time=times[i],
                    frequency=cqt_f0[i],
                    midi=librosa.hz_to_midi(cqt_f0[i]),
                    confidence=cqt_conf[i],
                    is_voiced=True,
                    method='cqt'
                ))
            
            if yin_f0[i] > 0 and yin_conf[i] > 0.3:
                frame_candidates.append(FramePitch(
                    frame=i, time=times[i],
                    frequency=yin_f0[i],
                    midi=librosa.hz_to_midi(yin_f0[i]),
                    confidence=yin_conf[i],
                    is_voiced=True,
                    method='yin'
                ))
            
            # Also add unvoiced candidate
            frame_candidates.append(FramePitch(
                frame=i, time=times[i],
                frequency=0.0,
                midi=0.0,
                confidence=0.5,  # Default confidence for silence
                is_voiced=False,
                method='silence'
            ))
            
            # Merge similar candidates (within 1 semitone)
            merged = self._merge_candidates(frame_candidates)
            
            # Sort by confidence and take top
            merged.sort(key=lambda c: c.confidence, reverse=True)
            top_candidates = merged[:self.n_candidates + 1]  # +1 for silence
            
            all_frame_candidates.append(top_candidates)
        
        return all_frame_candidates
    
    def _merge_candidates(self, candidates: List[FramePitch]) -> List[FramePitch]:
        """Merge candidates within 1 semitone, averaging their values."""
        if len(candidates) <= 1:
            return candidates
        
        # Separate voiced and unvoiced
        voiced = [c for c in candidates if c.is_voiced]
        unvoiced = [c for c in candidates if not c.is_voiced]
        
        if not voiced:
            return unvoiced
        
        # Sort voiced by MIDI
        voiced.sort(key=lambda c: c.midi)
        
        merged = []
        current_group = [voiced[0]]
        
        for c in voiced[1:]:
            if abs(c.midi - current_group[0].midi) <= 1.0:
                current_group.append(c)
            else:
                # Merge current group
                merged.append(self._merge_group(current_group))
                current_group = [c]
        
        merged.append(self._merge_group(current_group))
        
        return merged + unvoiced
    
    def _merge_group(self, group: List[FramePitch]) -> FramePitch:
        """Merge a group of similar candidates."""
        if len(group) == 1:
            return group[0]
        
        # Weighted average by confidence
        total_conf = sum(c.confidence for c in group)
        avg_freq = sum(c.frequency * c.confidence for c in group) / total_conf
        avg_midi = sum(c.midi * c.confidence for c in group) / total_conf
        max_conf = max(c.confidence for c in group)
        methods = '+'.join(set(c.method for c in group))
        
        return FramePitch(
            frame=group[0].frame,
            time=group[0].time,
            frequency=avg_freq,
            midi=avg_midi,
            confidence=max_conf,
            is_voiced=True,
            method=methods
        )
    
    def _detect_pyin(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pitch using pYIN."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        confidence = np.where(f0 > 0, voiced_probs, 0.0)
        return f0, confidence
    
    def _detect_cqt(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pitch using CQT peak picking."""
        fmin_cqt = librosa.note_to_hz('C2')
        n_bins = 84
        
        C = np.abs(librosa.cqt(
            y, sr=self.sr, hop_length=self.hop_length,
            fmin=fmin_cqt, n_bins=n_bins, bins_per_octave=12
        ))
        
        cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin_cqt, bins_per_octave=12)
        
        n_frames = C.shape[1]
        f0 = np.zeros(n_frames)
        confidence = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = C[:, i]
            max_mag = frame.max()
            
            if max_mag > 1e-10:
                peak_idx = frame.argmax()
                peak_freq = cqt_freqs[peak_idx]
                
                if self.fmin <= peak_freq <= self.fmax:
                    f0[i] = peak_freq
                    confidence[i] = frame[peak_idx] / max_mag
        
        return f0, confidence
    
    def _detect_yin(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pitch using YIN algorithm."""
        # Use librosa's yin implementation
        f0 = librosa.yin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # YIN doesn't directly give confidence, estimate from f0 stability
        # Use local variance as inverse confidence
        f0_smooth = medfilt(f0, kernel_size=5)
        variance = np.abs(f0 - f0_smooth) / (f0 + 1e-10)
        confidence = np.clip(1.0 - variance * 10, 0, 1)
        
        # Filter to valid range
        valid_mask = (f0 >= self.fmin) & (f0 <= self.fmax)
        f0 = np.where(valid_mask, f0, 0.0)
        confidence = np.where(valid_mask, confidence, 0.0)
        
        return f0, confidence


class ViterbiPitchTracker:
    """
    Main pitch tracker using Viterbi algorithm.
    
    The Viterbi algorithm finds the optimal sequence of pitch states
    that maximizes the joint probability of observations and transitions.
    
    State space: MIDI notes from GUITAR_MIN_MIDI to GUITAR_MAX_MIDI + 1 silence state
    Observations: Frame pitch candidates with confidences
    Transitions: Governed by GuitarPhysicsModel
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length_ms: float = 10,    # Fine resolution for tracking
        frame_length_ms: float = 50,  # Pitch detection window
        physics_model: Optional[GuitarPhysicsModel] = None,
        use_log_space: bool = True,
        min_voiced_confidence: float = 0.4,
        emission_weight: float = 1.0,
        transition_weight: float = 1.0,
    ):
        self.sr = sr
        self.hop_length_ms = hop_length_ms
        self.hop_length = int(sr * hop_length_ms / 1000)
        self.frame_length_ms = frame_length_ms
        
        self.physics = physics_model or GuitarPhysicsModel(
            frame_duration_ms=hop_length_ms
        )
        
        self.use_log_space = use_log_space
        self.min_voiced_confidence = min_voiced_confidence
        self.emission_weight = emission_weight
        self.transition_weight = transition_weight
        
        # State space: silence (0) + MIDI notes
        self.n_midi_states = GUITAR_MAX_MIDI - GUITAR_MIN_MIDI + 1
        self.n_states = self.n_midi_states + 1  # +1 for silence
        
        # State 0 = silence, states 1+ = MIDI notes offset by GUITAR_MIN_MIDI
        self.silence_state = 0
        
        # Extractor
        self.extractor = FramePitchExtractor(
            sr=sr,
            hop_length_ms=hop_length_ms,
            frame_length_ms=frame_length_ms,
        )
    
    def track(self, y: np.ndarray, verbose: bool = True) -> PitchPath:
        """
        Track pitch through audio using Viterbi decoding.
        
        Args:
            y: Audio signal (mono)
            verbose: Print progress info
            
        Returns:
            PitchPath with optimal pitch trajectory
        """
        if verbose:
            print(f"\n🎸 Viterbi Pitch Tracker")
            print(f"   Hop: {self.hop_length_ms}ms, Frame: {self.frame_length_ms}ms")
        
        # Extract candidates per frame
        if verbose:
            print("   Extracting pitch candidates...")
        
        frame_candidates = self.extractor.extract_all_frames(y)
        n_frames = len(frame_candidates)
        
        if verbose:
            n_with_pitched = sum(1 for fc in frame_candidates 
                                if any(c.is_voiced for c in fc))
            print(f"   {n_frames} frames, {n_with_pitched} with pitched content")
        
        # Run Viterbi
        if verbose:
            print("   Running Viterbi decoding...")
        
        best_path = self._viterbi_decode(frame_candidates)
        
        # Convert to PitchPath
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=self.sr, hop_length=self.hop_length
        )
        
        frequencies = np.zeros(n_frames)
        midi_notes = np.zeros(n_frames)
        confidences = np.zeros(n_frames)
        is_voiced = np.zeros(n_frames, dtype=bool)
        
        for i, state in enumerate(best_path):
            if state == self.silence_state:
                is_voiced[i] = False
            else:
                midi = state + GUITAR_MIN_MIDI - 1
                midi_notes[i] = midi
                frequencies[i] = librosa.midi_to_hz(midi)
                is_voiced[i] = True
                
                # Get confidence from original candidates
                for cand in frame_candidates[i]:
                    if cand.is_voiced and abs(cand.midi - midi) <= 0.5:
                        confidences[i] = cand.confidence
                        frequencies[i] = cand.frequency  # Use original freq
                        break
                else:
                    confidences[i] = 0.5  # Default if no match
        
        # Apply post-smoothing
        path = PitchPath(
            frames=np.arange(n_frames),
            times=times,
            frequencies=frequencies,
            midi_notes=midi_notes,
            confidences=confidences,
            is_voiced=is_voiced
        )
        
        path = self._post_smooth(path)
        
        if verbose:
            n_voiced = np.sum(path.is_voiced)
            print(f"   Tracked {n_voiced} voiced frames ({100*n_voiced/n_frames:.1f}%)")
        
        return path
    
    def _viterbi_decode(self, frame_candidates: List[List[FramePitch]]) -> List[int]:
        """
        Viterbi decoding to find optimal pitch sequence.
        
        Uses log probabilities for numerical stability.
        """
        n_frames = len(frame_candidates)
        
        if n_frames == 0:
            return []
        
        # Viterbi tables
        # V[t][s] = log prob of best path ending at state s at time t
        V = np.full((n_frames, self.n_states), -np.inf)
        backtrack = np.zeros((n_frames, self.n_states), dtype=int)
        
        # Initialize with first frame
        for cand in frame_candidates[0]:
            state = self._get_state(cand)
            emission = self._emission_log_prob(cand)
            V[0][state] = max(V[0][state], emission)
        
        # Forward pass
        for t in range(1, n_frames):
            candidates = frame_candidates[t]
            
            for cand in candidates:
                state = self._get_state(cand)
                emission = self._emission_log_prob(cand)
                
                # Find best previous state
                best_prob = -np.inf
                best_prev = 0
                
                for prev_state in range(self.n_states):
                    if V[t-1][prev_state] == -np.inf:
                        continue
                    
                    trans = self._transition_log_prob(prev_state, state)
                    prob = V[t-1][prev_state] + trans + emission
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_state
                
                if best_prob > V[t][state]:
                    V[t][state] = best_prob
                    backtrack[t][state] = best_prev
        
        # Backtrack to find best path
        best_final_state = np.argmax(V[n_frames - 1])
        
        path = [best_final_state]
        for t in range(n_frames - 1, 0, -1):
            path.append(backtrack[t][path[-1]])
        
        path.reverse()
        return path
    
    def _get_state(self, cand: FramePitch) -> int:
        """Convert candidate to state index."""
        if not cand.is_voiced or cand.midi <= 0:
            return self.silence_state
        
        midi = int(round(cand.midi))
        midi = max(GUITAR_MIN_MIDI, min(GUITAR_MAX_MIDI, midi))
        return midi - GUITAR_MIN_MIDI + 1
    
    def _emission_log_prob(self, cand: FramePitch) -> float:
        """Log probability of observing this candidate."""
        if not cand.is_voiced:
            # Silence emission - slight penalty to prefer voiced when confident
            return -1.0 * self.emission_weight
        
        # Use confidence as probability
        conf = max(0.01, min(0.99, cand.confidence))
        return np.log(conf) * self.emission_weight
    
    def _transition_log_prob(self, from_state: int, to_state: int) -> float:
        """Log probability of transition between states."""
        # Handle silence
        if from_state == self.silence_state:
            if to_state == self.silence_state:
                return 0.0  # Staying silent is fine
            else:
                return -0.5 * self.transition_weight  # Starting to voice
        
        if to_state == self.silence_state:
            return -0.5 * self.transition_weight  # Stopping voicing
        
        # Both voiced - use physics model
        from_midi = from_state + GUITAR_MIN_MIDI - 1
        to_midi = to_state + GUITAR_MIN_MIDI - 1
        
        return self.physics.get_transition_log_prob(from_midi, to_midi) * self.transition_weight
    
    def _post_smooth(self, path: PitchPath) -> PitchPath:
        """Apply post-processing smoothing."""
        # Fill short gaps (< 30ms)
        gap_frames = int(0.030 * self.sr / self.hop_length)
        
        is_voiced = path.is_voiced.copy()
        midi_notes = path.midi_notes.copy()
        frequencies = path.frequencies.copy()
        
        # Fill gaps
        in_gap = False
        gap_start = 0
        
        for i in range(len(is_voiced)):
            if is_voiced[i]:
                if in_gap and i - gap_start <= gap_frames:
                    # Fill gap with interpolation
                    if gap_start > 0 and i < len(is_voiced):
                        prev_midi = midi_notes[gap_start - 1]
                        next_midi = midi_notes[i]
                        
                        for j in range(gap_start, i):
                            alpha = (j - gap_start + 1) / (i - gap_start + 1)
                            midi_notes[j] = prev_midi + alpha * (next_midi - prev_midi)
                            frequencies[j] = librosa.midi_to_hz(midi_notes[j])
                            is_voiced[j] = True
                
                in_gap = False
            else:
                if not in_gap:
                    in_gap = True
                    gap_start = i
        
        # Apply mild Gaussian smoothing to MIDI
        voiced_mask = is_voiced
        if np.sum(voiced_mask) > 3:
            midi_smoothed = midi_notes.copy()
            midi_smoothed[voiced_mask] = gaussian_filter1d(
                midi_notes[voiced_mask], sigma=1
            )
            # Quantize back to nearest semitone
            midi_notes[voiced_mask] = np.round(midi_smoothed[voiced_mask])
            frequencies[voiced_mask] = librosa.midi_to_hz(midi_notes[voiced_mask])
        
        return PitchPath(
            frames=path.frames,
            times=path.times,
            frequencies=frequencies,
            midi_notes=midi_notes,
            confidences=path.confidences,
            is_voiced=is_voiced
        )


def track_pitch_realtime(
    y: np.ndarray,
    sr: int = 22050,
    hop_length_ms: float = 10,
    verbose: bool = True
) -> PitchPath:
    """
    Convenience function for real-time pitch tracking.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        hop_length_ms: Frame hop in milliseconds
        verbose: Print progress
        
    Returns:
        PitchPath with tracked pitches
    """
    tracker = ViterbiPitchTracker(
        sr=sr,
        hop_length_ms=hop_length_ms
    )
    return tracker.track(y, verbose=verbose)


def track_pitch_from_file(
    audio_path: str,
    sr: int = 22050,
    hop_length_ms: float = 10,
    use_harmonic: bool = True,
    verbose: bool = True
) -> PitchPath:
    """
    Track pitch from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length_ms: Frame hop in milliseconds
        use_harmonic: Apply harmonic separation first
        verbose: Print progress
        
    Returns:
        PitchPath with tracked pitches
    """
    if verbose:
        print(f"\n📂 Loading: {audio_path}")
    
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    if use_harmonic:
        if verbose:
            print("🎵 Separating harmonic component...")
        y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
        y = y_harmonic
    
    return track_pitch_realtime(y, sr=sr, hop_length_ms=hop_length_ms, verbose=verbose)


# =============================================================================
# CREPE-Enhanced Tracker
# =============================================================================

# Check for CREPE
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False


class CREPEFrameExtractor(FramePitchExtractor):
    """
    Frame extractor that uses CREPE neural network as primary detector.
    
    CREPE provides very accurate pitch but is computationally expensive.
    Use this for best accuracy when speed is not critical.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length_ms: float = 10,
        frame_length_ms: float = 50,
        n_candidates: int = 3,
        fmin: float = GUITAR_MIN_HZ,
        fmax: float = GUITAR_MAX_HZ,
        crepe_model: str = 'small',  # 'tiny', 'small', 'medium', 'large', 'full'
    ):
        super().__init__(sr, frame_length_ms, hop_length_ms, n_candidates, fmin, fmax)
        self.crepe_model = crepe_model
        
        if not HAS_CREPE:
            raise RuntimeError("CREPE not available. Install: pip install crepe tensorflow")
    
    def extract_all_frames(self, y: np.ndarray) -> List[List[FramePitch]]:
        """Extract using CREPE as primary, with fallback detectors."""
        # Get CREPE predictions
        step_size_ms = int(self.hop_length * 1000 / self.sr)
        step_size_ms = max(10, step_size_ms)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time, frequency, confidence, activation = crepe.predict(
                y, self.sr,
                model_capacity=self.crepe_model,
                viterbi=True,  # CREPE's built-in Viterbi
                step_size=step_size_ms,
                verbose=0
            )
        
        # Also run pYIN for comparison
        pyin_f0, pyin_conf = self._detect_pyin(y)
        
        # Align to our frame resolution
        n_frames = min(len(frequency), len(pyin_f0))
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=self.sr, hop_length=self.hop_length
        )
        
        all_frame_candidates = []
        
        for i in range(n_frames):
            frame_candidates = []
            t = times[i] if i < len(times) else i * self.hop_length / self.sr
            
            # CREPE candidate
            if i < len(frequency) and confidence[i] > 0.3:
                freq = frequency[i]
                if self.fmin <= freq <= self.fmax:
                    frame_candidates.append(FramePitch(
                        frame=i, time=t,
                        frequency=float(freq),
                        midi=float(librosa.hz_to_midi(freq)),
                        confidence=float(confidence[i]),
                        is_voiced=True,
                        method='crepe'
                    ))
            
            # pYIN candidate
            if i < len(pyin_f0) and pyin_f0[i] > 0 and pyin_conf[i] > 0.3:
                frame_candidates.append(FramePitch(
                    frame=i, time=t,
                    frequency=pyin_f0[i],
                    midi=librosa.hz_to_midi(pyin_f0[i]),
                    confidence=pyin_conf[i],
                    is_voiced=True,
                    method='pyin'
                ))
            
            # Silence candidate
            frame_candidates.append(FramePitch(
                frame=i, time=t,
                frequency=0.0,
                midi=0.0,
                confidence=0.3,
                is_voiced=False,
                method='silence'
            ))
            
            # Merge similar
            merged = self._merge_candidates(frame_candidates)
            merged.sort(key=lambda c: c.confidence, reverse=True)
            
            all_frame_candidates.append(merged[:self.n_candidates + 1])
        
        return all_frame_candidates


class HybridViterbiTracker(ViterbiPitchTracker):
    """
    Hybrid tracker that can use CREPE when available.
    
    Falls back to standard multi-detector approach if CREPE not installed.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length_ms: float = 10,
        frame_length_ms: float = 50,
        use_crepe: bool = True,
        crepe_model: str = 'small',
        physics_model: Optional[GuitarPhysicsModel] = None,
        **kwargs
    ):
        super().__init__(sr, hop_length_ms, frame_length_ms, physics_model, **kwargs)
        
        self.use_crepe = use_crepe and HAS_CREPE
        self.crepe_model = crepe_model
        
        if self.use_crepe:
            self.extractor = CREPEFrameExtractor(
                sr=sr,
                hop_length_ms=hop_length_ms,
                frame_length_ms=frame_length_ms,
                crepe_model=crepe_model,
            )


def track_pitch_hybrid(
    y: np.ndarray,
    sr: int = 22050,
    hop_length_ms: float = 10,
    use_crepe: bool = True,
    crepe_model: str = 'small',
    verbose: bool = True
) -> PitchPath:
    """
    Track pitch using hybrid CREPE + Viterbi approach.
    
    This combines CREPE's neural network accuracy with our
    guitar-physics-aware Viterbi smoothing.
    """
    tracker = HybridViterbiTracker(
        sr=sr,
        hop_length_ms=hop_length_ms,
        use_crepe=use_crepe,
        crepe_model=crepe_model,
    )
    return tracker.track(y, verbose=verbose)


# =============================================================================
# Integration with guitar_tabs.py
# =============================================================================

def convert_path_to_notes(path: PitchPath, min_duration: float = 0.03):
    """
    Convert PitchPath to Note objects compatible with guitar_tabs.py.
    
    Returns list of dicts with midi, start_time, duration, confidence.
    """
    return path.get_notes(min_duration=min_duration)


def detect_notes_viterbi(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 220,  # ~10ms at 22050
    use_crepe: bool = False,
    use_harmonic: bool = True,
    min_note_duration: float = 0.03,
    verbose: bool = True
) -> List[Dict]:
    """
    Main entry point for guitar_tabs.py integration.
    
    Returns notes in the standard format used by the rest of the system.
    """
    hop_length_ms = hop_length * 1000 / sr
    
    if use_harmonic:
        y_detect, _ = librosa.effects.hpss(y, margin=3.0)
    else:
        y_detect = y
    
    if use_crepe and HAS_CREPE:
        tracker = HybridViterbiTracker(
            sr=sr,
            hop_length_ms=hop_length_ms,
            use_crepe=True,
        )
    else:
        tracker = ViterbiPitchTracker(
            sr=sr,
            hop_length_ms=hop_length_ms,
        )
    
    path = tracker.track(y_detect, verbose=verbose)
    return path.get_notes(min_duration=min_note_duration)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Real-time Viterbi Pitch Tracker")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--hop-ms", type=float, default=10, help="Hop length in ms")
    parser.add_argument("--no-harmonic", action="store_true", help="Skip harmonic separation")
    parser.add_argument("--use-crepe", action="store_true", help="Use CREPE neural network")
    parser.add_argument("--crepe-model", default="small", choices=['tiny', 'small', 'medium', 'large', 'full'])
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less output")
    parser.add_argument("--compare", action="store_true", help="Compare with pYIN baseline")
    
    args = parser.parse_args()
    
    # Load audio
    print(f"📂 Loading: {args.audio_path}")
    y, sr = librosa.load(args.audio_path, sr=22050, mono=True)
    
    if not args.no_harmonic:
        print("🎵 Separating harmonic component...")
        y_detect, _ = librosa.effects.hpss(y, margin=3.0)
    else:
        y_detect = y
    
    # Track with Viterbi
    if args.use_crepe:
        if not HAS_CREPE:
            print("⚠️  CREPE not available, using standard tracker")
            path = track_pitch_realtime(y_detect, sr=sr, hop_length_ms=args.hop_ms, 
                                        verbose=not args.quiet)
        else:
            print(f"🧠 Using CREPE ({args.crepe_model}) + Viterbi")
            path = track_pitch_hybrid(y_detect, sr=sr, hop_length_ms=args.hop_ms,
                                      use_crepe=True, crepe_model=args.crepe_model,
                                      verbose=not args.quiet)
    else:
        path = track_pitch_realtime(y_detect, sr=sr, hop_length_ms=args.hop_ms, 
                                    verbose=not args.quiet)
    
    # Get notes
    notes = path.get_notes(min_duration=0.03)
    
    print(f"\n🎵 Extracted {len(notes)} notes:")
    print("-" * 70)
    
    for i, note in enumerate(notes[:30]):
        midi = note['midi']
        name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
        print(f"  {i+1:3d}. {name:4s} (MIDI {midi:2d}) "
              f"@ {note['start_time']:.3f}s, "
              f"dur={note['duration']:.3f}s, "
              f"conf={note['confidence']:.2f}")
    
    if len(notes) > 30:
        print(f"  ... and {len(notes) - 30} more")
    
    # Optional comparison
    if args.compare:
        print("\n" + "=" * 70)
        print("📊 Comparison with pYIN baseline:")
        print("=" * 70)
        
        # Run baseline pYIN
        f0_pyin, voiced, probs = librosa.pyin(
            y_detect, fmin=GUITAR_MIN_HZ, fmax=GUITAR_MAX_HZ, sr=sr
        )
        
        # Count detected frames
        viterbi_voiced = int(np.sum(path.is_voiced))
        pyin_voiced = int(np.sum(voiced))
        
        print(f"   Viterbi voiced frames: {viterbi_voiced}")
        print(f"   pYIN voiced frames: {pyin_voiced}")
        
        # Count big jumps (>12 semitones)
        viterbi_jumps = 0
        pyin_jumps = 0
        
        for i in range(1, len(path.midi_notes)):
            if path.is_voiced[i] and path.is_voiced[i-1]:
                jump = abs(path.midi_notes[i] - path.midi_notes[i-1])
                if jump > 12:
                    viterbi_jumps += 1
        
        pyin_midi = librosa.hz_to_midi(np.where(f0_pyin > 0, f0_pyin, 1))
        for i in range(1, len(pyin_midi)):
            if voiced[i] and voiced[i-1]:
                jump = abs(pyin_midi[i] - pyin_midi[i-1])
                if jump > 12:
                    pyin_jumps += 1
        
        print(f"   Viterbi large jumps (>12 semitones): {viterbi_jumps}")
        print(f"   pYIN large jumps (>12 semitones): {pyin_jumps}")
        print(f"   Jump reduction: {100*(pyin_jumps - viterbi_jumps)/(pyin_jumps+1):.1f}%")
    
    if args.output:
        output_data = {
            "n_frames": path.n_frames,
            "duration": path.duration,
            "n_notes": len(notes),
            "notes": notes,
            "config": {
                "hop_ms": args.hop_ms,
                "use_crepe": args.use_crepe,
                "use_harmonic": not args.no_harmonic,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Saved to {args.output}")
