#!/usr/bin/env python3
"""
Fast Legato Run Detection for Guitar Transcription

Lead guitar solos often feature fast legato runs:
1. Rapid sequences of hammer-ons and pull-offs
2. 3-note-per-string patterns following scale shapes
3. Very fast (16th notes at 120+ BPM = notes < 125ms apart)
4. Often ascending or descending through scale positions

This module improves detection of these fast passages:
- Lower onset threshold for closely spaced notes
- Scale pattern matching to fill in missed notes
- Marking entire runs as legato phrases
- Velocity-aware detection (fast runs have consistent energy)
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, medfilt, butter, filtfilt
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter

# Guitar constants
GUITAR_MIN_HZ = 75
GUITAR_MAX_HZ = 1400
GUITAR_MIN_MIDI = 36
GUITAR_MAX_MIDI = 90

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale patterns (intervals from root)
SCALE_PATTERNS = {
    'major': [0, 2, 4, 5, 7, 9, 11],  # W-W-H-W-W-W-H
    'minor': [0, 2, 3, 5, 7, 8, 10],  # W-H-W-W-H-W-W (natural minor)
    'minor_pentatonic': [0, 3, 5, 7, 10],  # Minor pentatonic
    'major_pentatonic': [0, 2, 4, 7, 9],   # Major pentatonic
    'blues': [0, 3, 5, 6, 7, 10],    # Minor pentatonic + blue note
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
}

# 3-note-per-string patterns (common for shred/legato runs)
# Format: intervals within each string position
THREE_NPS_PATTERNS = {
    'major_3nps': [
        [0, 2, 4],   # String 6 (low E) - root, 2nd, 3rd
        [5, 7, 9],   # String 5 (A) - 4th, 5th, 6th
        [11, 12, 14], # String 4 (D) - 7th, root+8va, 2nd+8va
        [16, 17, 19], # String 3 (G) - 3rd+8va, 4th+8va, 5th+8va
        [21, 23, 24], # String 2 (B) - 6th+8va, 7th+8va, root+2*8va
        [26, 28, 29], # String 1 (high e) - 2nd+2*8va, 3rd+2*8va, 4th+2*8va
    ],
    'minor_3nps': [
        [0, 2, 3],   # Root, 2nd, b3rd
        [5, 7, 8],   # 4th, 5th, b6th
        [10, 12, 14], # b7th, root+8va, 2nd+8va
        [15, 17, 19], # b3rd+8va, 4th+8va, 5th+8va
        [20, 22, 24], # b6th+8va, b7th+8va, root+2*8va
        [26, 27, 29], # 2nd+2*8va, b3rd+2*8va, 4th+2*8va
    ],
    'pentatonic_3nps': [
        [0, 3, 5],   # Root, b3rd, 4th
        [7, 10, 12], # 5th, b7th, root+8va
        [15, 17, 19], # b3rd+8va, 4th+8va, 5th+8va
        [22, 24, 27], # b7th+8va, root+2*8va, b3rd+2*8va
    ],
}


@dataclass
class LegatoNote:
    """A note detected as part of a legato run."""
    midi: int
    start_time: float
    end_time: float
    confidence: float
    is_detected: bool = True  # False if inferred from pattern
    pitch_source: str = ""    # How pitch was determined
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)


@dataclass
class LegatoRun:
    """A sequence of notes played as a legato phrase."""
    notes: List[LegatoNote]
    start_time: float
    end_time: float
    direction: str  # 'ascending', 'descending', or 'mixed'
    scale_type: str = ""  # Detected scale pattern
    root_note: int = -1   # Root note (MIDI) if scale detected
    is_3nps: bool = False  # Is this a 3-note-per-string pattern?
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def num_notes(self) -> int:
        return len(self.notes)
    
    @property
    def notes_per_second(self) -> float:
        if self.duration > 0:
            return self.num_notes / self.duration
        return 0
    
    def __repr__(self):
        scale_info = f" ({self.scale_type})" if self.scale_type else ""
        return f"LegatoRun({self.num_notes} notes, {self.direction}{scale_info}, {self.notes_per_second:.1f} nps)"


class FastLegatoDetector:
    """
    Detects fast legato runs in guitar audio.
    
    Key improvements:
    1. Adaptive threshold that lowers for fast passages
    2. Pitch tracking that handles rapid changes
    3. Scale pattern matching to fill gaps
    4. Run grouping and phrase detection
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 256,  # Smaller hop for fast notes
        n_fft: int = 2048,
        # Fast note detection
        min_note_duration_ms: float = 30,   # Min note length in fast runs
        max_note_duration_ms: float = 200,  # Max for fast legato notes
        fast_run_threshold_ms: float = 125, # If IOI < this, consider fast run
        # Onset thresholds
        base_onset_threshold: float = 0.3,
        fast_onset_threshold: float = 0.15,  # Lower for fast passages
        # Legato detection
        legato_energy_ratio: float = 0.3,   # Max attack ratio for legato
        min_run_notes: int = 4,             # Minimum notes to form a run
        # Scale matching
        enable_scale_matching: bool = True,
        scale_match_threshold: float = 0.7,  # Min match ratio
        # Pitch tracking
        pitch_confidence_threshold: float = 0.3,
        use_harmonic_separation: bool = True,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        self.min_note_frames = int(min_note_duration_ms / 1000 * sr / hop_length)
        self.max_note_frames = int(max_note_duration_ms / 1000 * sr / hop_length)
        self.fast_run_frames = int(fast_run_threshold_ms / 1000 * sr / hop_length)
        
        self.base_onset_threshold = base_onset_threshold
        self.fast_onset_threshold = fast_onset_threshold
        self.legato_energy_ratio = legato_energy_ratio
        self.min_run_notes = min_run_notes
        
        self.enable_scale_matching = enable_scale_matching
        self.scale_match_threshold = scale_match_threshold
        
        self.pitch_confidence_threshold = pitch_confidence_threshold
        self.use_harmonic_separation = use_harmonic_separation
    
    def detect(
        self,
        y: np.ndarray,
        existing_notes: Optional[List] = None,
        verbose: bool = True
    ) -> Tuple[List[LegatoNote], List[LegatoRun]]:
        """
        Detect fast legato runs in audio.
        
        Args:
            y: Audio signal
            existing_notes: Already detected notes (to avoid duplicates)
            verbose: Print diagnostic info
            
        Returns:
            (all_legato_notes, legato_runs)
        """
        if verbose:
            print("🎸 Fast Legato Run Detection")
            duration = len(y) / self.sr
            print(f"   Audio: {duration:.2f}s @ {self.sr}Hz")
        
        # Step 1: Harmonic separation for cleaner pitch
        if self.use_harmonic_separation:
            if verbose:
                print("   Separating harmonic component...")
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
        else:
            y_harmonic = y
            y_percussive = None
        
        # Step 2: Compute features
        if verbose:
            print("   Computing pitch contour...")
        f0, voiced_flag, voiced_prob = self._compute_pitch(y_harmonic)
        
        if verbose:
            print("   Computing onset features...")
        onset_env, attack_env = self._compute_onset_features(y)
        
        # Step 3: Detect fast passages (where we need lower threshold)
        if verbose:
            print("   Detecting fast passages...")
        fast_regions = self._detect_fast_regions(f0, voiced_flag)
        
        # Step 4: Detect legato onsets with adaptive threshold
        if verbose:
            print("   Detecting legato onsets...")
        onsets = self._detect_legato_onsets(
            f0, voiced_flag, voiced_prob, onset_env, attack_env, fast_regions
        )
        
        if verbose:
            print(f"   Found {len(onsets)} potential legato note onsets")
        
        # Step 5: Extract notes from onsets
        if verbose:
            print("   Extracting notes...")
        notes = self._extract_notes(f0, voiced_prob, onsets)
        
        if verbose:
            print(f"   Extracted {len(notes)} legato notes")
        
        # Step 6: Group into runs
        if verbose:
            print("   Grouping into runs...")
        runs = self._group_runs(notes)
        
        if verbose:
            print(f"   Found {len(runs)} legato runs")
        
        # Step 7: Fill gaps using scale patterns
        if self.enable_scale_matching and len(runs) > 0:
            if verbose:
                print("   Applying scale pattern matching...")
            runs, additional_notes = self._apply_scale_matching(runs, f0, voiced_prob)
            notes.extend(additional_notes)
            notes.sort(key=lambda n: n.start_time)
            
            if verbose:
                print(f"   Added {len(additional_notes)} notes from pattern matching")
        
        # Step 8: Merge with existing notes (avoid duplicates)
        if existing_notes:
            notes = self._merge_with_existing(notes, existing_notes)
        
        if verbose:
            self._print_summary(notes, runs)
        
        return notes, runs
    
    def _compute_pitch(
        self,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pitch contour using multiple methods.
        
        For fast passages, we need very stable pitch tracking that
        can handle rapid pitch changes.
        """
        # Use pYIN as primary (probabilistic, handles uncertainty)
        f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        # Also compute YIN (more stable, less accurate on pitch)
        f0_yin = librosa.yin(
            y,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Combine: prefer pYIN when confident, fall back to YIN
        n_frames = min(len(f0_pyin), len(f0_yin))
        f0 = np.zeros(n_frames)
        
        for i in range(n_frames):
            if f0_pyin[i] > 0 and voiced_prob[i] >= self.pitch_confidence_threshold:
                f0[i] = f0_pyin[i]
            elif f0_yin[i] > 0 and GUITAR_MIN_HZ <= f0_yin[i] <= GUITAR_MAX_HZ:
                f0[i] = f0_yin[i]
                # Assign lower confidence
                if voiced_prob[i] < 0.3:
                    voiced_prob[i] = 0.3
        
        # Apply light median filter for stability
        f0 = self._median_filter_f0(f0, kernel_size=3)
        
        return f0, voiced_flag[:n_frames], voiced_prob[:n_frames]
    
    def _median_filter_f0(self, f0: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply median filter only to valid pitch values."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        valid = f0 > 0
        if not np.any(valid):
            return f0
        
        filtered = medfilt(f0, kernel_size=kernel_size)
        return np.where(valid, filtered, f0)
    
    def _compute_onset_features(
        self,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute onset detection features.
        
        Returns:
            onset_env: Standard onset envelope
            attack_env: Attack transient envelope (for legato detection)
        """
        # Standard onset envelope (spectral flux based)
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            aggregate=np.median
        )
        
        # Normalize
        onset_env = onset_env / (np.max(onset_env) + 1e-10)
        
        # Attack transient envelope (high-frequency energy)
        # This helps distinguish picked notes from legato
        nyq = self.sr / 2
        high_cutoff = min(2000 / nyq, 0.95)
        
        try:
            b, a = butter(4, high_cutoff, btype='high')
            y_hp = filtfilt(b, a, y)
        except Exception:
            y_hp = y
        
        # Compute envelope
        envelope = np.abs(y_hp)
        
        # Downsample to hop_length resolution
        n_frames = len(onset_env)
        attack_env = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.hop_length, len(envelope))
            if start < len(envelope):
                attack_env[i] = np.max(envelope[start:end])
        
        # Normalize
        attack_env = attack_env / (np.max(attack_env) + 1e-10)
        
        return onset_env, attack_env
    
    def _detect_fast_regions(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray
    ) -> np.ndarray:
        """
        Detect regions with rapid pitch changes (fast passages).
        
        Returns boolean array marking fast regions where we should
        use a lower onset threshold.
        """
        n_frames = len(f0)
        fast_regions = np.zeros(n_frames, dtype=bool)
        
        # Convert to MIDI for easier comparison
        midi = np.zeros_like(f0)
        valid = f0 > 0
        midi[valid] = librosa.hz_to_midi(f0[valid])
        
        # Compute pitch change rate
        pitch_diff = np.abs(np.diff(midi))
        pitch_diff = np.concatenate([[0], pitch_diff])
        
        # Find regions with consistent pitch changes (scale runs)
        # Use a sliding window
        window_size = self.fast_run_frames
        
        for i in range(n_frames - window_size):
            window = pitch_diff[i:i + window_size]
            voiced_window = voiced_flag[i:i + window_size] if i + window_size <= len(voiced_flag) else voiced_flag[i:]
            
            # Count significant pitch changes in window
            changes = np.sum((window >= 1) & (window <= 5))  # 1-5 semitones
            voiced_ratio = np.mean(voiced_window) if len(voiced_window) > 0 else 0
            
            # Mark as fast if many pitch changes and mostly voiced
            if changes >= 3 and voiced_ratio > 0.7:
                fast_regions[i:i + window_size] = True
        
        return fast_regions
    
    def _detect_legato_onsets(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        voiced_prob: np.ndarray,
        onset_env: np.ndarray,
        attack_env: np.ndarray,
        fast_regions: np.ndarray
    ) -> List[Tuple[int, float, bool]]:
        """
        Detect onsets suitable for legato notes.
        
        Returns list of (frame, confidence, is_detected_vs_inferred)
        """
        n_frames = len(f0)
        onsets = []
        
        # Convert to MIDI
        midi = np.zeros_like(f0)
        valid = f0 > 0
        midi[valid] = np.round(librosa.hz_to_midi(f0[valid]))
        
        # Compute pitch changes
        pitch_change = np.abs(np.diff(midi))
        pitch_change = np.concatenate([[0], pitch_change])
        
        # Detect onset candidates
        prev_onset_frame = -self.min_note_frames
        
        for i in range(1, n_frames - 1):
            if not voiced_flag[i]:
                continue
            
            # Check minimum gap from previous onset
            if i - prev_onset_frame < self.min_note_frames:
                continue
            
            # Adaptive threshold based on context
            if fast_regions[i]:
                onset_threshold = self.fast_onset_threshold
            else:
                onset_threshold = self.base_onset_threshold
            
            # Check for legato characteristics:
            # 1. Pitch change (note transition)
            # 2. Low attack transient (not picked)
            # 3. Continuous energy (no gap)
            
            has_pitch_change = pitch_change[i] >= 0.5  # At least half semitone
            is_low_attack = attack_env[i] < self.legato_energy_ratio
            is_continuous = onset_env[i] < onset_threshold or is_low_attack
            
            # For legato: need pitch change with low attack
            is_legato = has_pitch_change and is_low_attack
            
            # For regular note in fast passage: need onset or pitch change
            is_fast_note = fast_regions[i] and (
                onset_env[i] > onset_threshold or 
                has_pitch_change
            )
            
            if is_legato or is_fast_note:
                confidence = voiced_prob[i]
                if is_legato:
                    # Boost confidence for clear legato
                    confidence = min(1.0, confidence + 0.2)
                
                onsets.append((i, confidence, True))  # True = detected
                prev_onset_frame = i
        
        return onsets
    
    def _extract_notes(
        self,
        f0: np.ndarray,
        voiced_prob: np.ndarray,
        onsets: List[Tuple[int, float, bool]]
    ) -> List[LegatoNote]:
        """
        Extract notes from onset positions.
        """
        notes = []
        n_frames = len(f0)
        
        for idx, (frame, onset_conf, is_detected) in enumerate(onsets):
            # Determine note end (next onset or max duration)
            if idx + 1 < len(onsets):
                end_frame = onsets[idx + 1][0]
            else:
                end_frame = min(frame + self.max_note_frames, n_frames - 1)
            
            # Ensure minimum duration
            if end_frame - frame < self.min_note_frames:
                end_frame = frame + self.min_note_frames
                if end_frame >= n_frames:
                    continue
            
            # Sample pitch (use median in attack region)
            sample_start = frame
            sample_end = min(frame + 10, end_frame)  # First ~40ms
            
            f0_window = f0[sample_start:sample_end]
            valid_window = f0_window > 0
            
            if not np.any(valid_window):
                continue
            
            # Use median pitch
            median_f0 = np.median(f0_window[valid_window])
            midi = int(np.round(librosa.hz_to_midi(median_f0)))
            
            # Validate guitar range
            if midi < GUITAR_MIN_MIDI or midi > GUITAR_MAX_MIDI:
                continue
            
            # Calculate confidence
            conf_window = voiced_prob[sample_start:sample_end]
            confidence = float(np.mean(conf_window[valid_window]) * onset_conf)
            
            # Convert frames to time
            start_time = librosa.frames_to_time(frame, sr=self.sr, hop_length=self.hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=self.sr, hop_length=self.hop_length)
            
            notes.append(LegatoNote(
                midi=midi,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                is_detected=is_detected,
                pitch_source='median_f0'
            ))
        
        return notes
    
    def _group_runs(self, notes: List[LegatoNote]) -> List[LegatoRun]:
        """
        Group consecutive notes into legato runs.
        """
        if len(notes) < self.min_run_notes:
            return []
        
        runs = []
        current_run: List[LegatoNote] = [notes[0]]
        
        for i in range(1, len(notes)):
            note = notes[i]
            prev_note = notes[i - 1]
            
            # Check if this note continues the run
            gap = note.start_time - prev_note.end_time
            pitch_diff = abs(note.midi - prev_note.midi)
            
            # Continue run if:
            # - Small time gap (< 100ms)
            # - Reasonable pitch interval (1-5 semitones typically)
            continues_run = (
                gap < 0.1 and
                1 <= pitch_diff <= 7
            )
            
            if continues_run:
                current_run.append(note)
            else:
                # End current run
                if len(current_run) >= self.min_run_notes:
                    runs.append(self._create_run(current_run))
                # Start new run
                current_run = [note]
        
        # Don't forget last run
        if len(current_run) >= self.min_run_notes:
            runs.append(self._create_run(current_run))
        
        return runs
    
    def _create_run(self, notes: List[LegatoNote]) -> LegatoRun:
        """
        Create a LegatoRun from a list of notes.
        """
        # Determine direction
        if len(notes) >= 2:
            pitch_changes = [notes[i+1].midi - notes[i].midi for i in range(len(notes)-1)]
            ascending = sum(1 for pc in pitch_changes if pc > 0)
            descending = sum(1 for pc in pitch_changes if pc < 0)
            
            if ascending > descending * 2:
                direction = 'ascending'
            elif descending > ascending * 2:
                direction = 'descending'
            else:
                direction = 'mixed'
        else:
            direction = 'mixed'
        
        # Check for 3-note-per-string pattern
        is_3nps = self._check_3nps_pattern(notes)
        
        # Calculate average confidence
        avg_confidence = np.mean([n.confidence for n in notes])
        
        return LegatoRun(
            notes=notes,
            start_time=notes[0].start_time,
            end_time=notes[-1].end_time,
            direction=direction,
            is_3nps=is_3nps,
            confidence=avg_confidence
        )
    
    def _check_3nps_pattern(self, notes: List[LegatoNote]) -> bool:
        """
        Check if notes follow a 3-note-per-string pattern.
        """
        if len(notes) < 6:  # Need at least 2 strings worth
            return False
        
        # Look for groups of 3 with consistent intervals
        intervals = [notes[i+1].midi - notes[i].midi for i in range(len(notes)-1)]
        
        # In 3nps, you often see patterns like: 2,2,3,2,2,3 or similar
        # (whole steps within string, larger jump between strings)
        
        groups = []
        current_group = [0]  # Start with first note
        
        for i, interval in enumerate(intervals):
            if abs(interval) <= 3:  # Same string
                current_group.append(i + 1)
            else:  # String change
                groups.append(len(current_group))
                current_group = [i + 1]
        
        groups.append(len(current_group))
        
        # Check if most groups have 3 notes
        threes = sum(1 for g in groups if g == 3)
        return threes >= len(groups) // 2
    
    def _apply_scale_matching(
        self,
        runs: List[LegatoRun],
        f0: np.ndarray,
        voiced_prob: np.ndarray
    ) -> Tuple[List[LegatoRun], List[LegatoNote]]:
        """
        Use scale patterns to fill in gaps in runs.
        """
        updated_runs = []
        additional_notes = []
        
        for run in runs:
            # Try to match to a scale
            scale_type, root, match_score = self._match_scale(run)
            
            if scale_type and match_score >= self.scale_match_threshold:
                run.scale_type = scale_type
                run.root_note = root
                
                # Find missing notes in the scale
                missing = self._find_missing_scale_notes(run, scale_type, root, f0, voiced_prob)
                
                if missing:
                    # Add missing notes to run
                    run.notes.extend(missing)
                    run.notes.sort(key=lambda n: n.start_time)
                    additional_notes.extend(missing)
                    
                    # Recalculate confidence
                    run.confidence = np.mean([n.confidence for n in run.notes])
            
            updated_runs.append(run)
        
        return updated_runs, additional_notes
    
    def _match_scale(
        self,
        run: LegatoRun
    ) -> Tuple[Optional[str], int, float]:
        """
        Try to match a run to a known scale pattern.
        
        Returns (scale_name, root_midi, match_score) or (None, -1, 0)
        """
        if len(run.notes) < 4:
            return None, -1, 0.0
        
        # Get pitch classes used in run
        pitch_classes = set(n.midi % 12 for n in run.notes)
        
        best_match = None
        best_root = -1
        best_score = 0.0
        
        # Try each scale type and each possible root
        for scale_name, intervals in SCALE_PATTERNS.items():
            for root in range(12):
                scale_pcs = set((root + i) % 12 for i in intervals)
                
                # Calculate match score
                matching = len(pitch_classes & scale_pcs)
                non_matching = len(pitch_classes - scale_pcs)
                
                if len(pitch_classes) > 0:
                    score = (matching - non_matching * 0.5) / len(pitch_classes)
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_match = scale_name
                    best_root = root
        
        return best_match, best_root, best_score
    
    def _find_missing_scale_notes(
        self,
        run: LegatoRun,
        scale_type: str,
        root: int,
        f0: np.ndarray,
        voiced_prob: np.ndarray
    ) -> List[LegatoNote]:
        """
        Find notes that should be in the run based on scale pattern.
        """
        missing_notes = []
        
        if scale_type not in SCALE_PATTERNS:
            return missing_notes
        
        scale_intervals = SCALE_PATTERNS[scale_type]
        
        # Get existing MIDI notes and their times
        existing_midi = sorted([(n.midi, n.start_time, n.end_time) for n in run.notes])
        
        # Look for gaps in the scale
        for i in range(len(existing_midi) - 1):
            midi1, start1, end1 = existing_midi[i]
            midi2, start2, end2 = existing_midi[i + 1]
            
            # Check if there's a gap (missing note in scale)
            pitch_gap = midi2 - midi1
            
            if pitch_gap > 2:  # More than a whole step - might be missing notes
                # Find what scale degrees should be between
                pc1 = midi1 % 12
                pc2 = midi2 % 12
                
                # Get scale degrees
                for offset in range(1, pitch_gap):
                    candidate_midi = midi1 + offset
                    candidate_pc = candidate_midi % 12
                    
                    # Check if this pitch class is in the scale
                    relative_pc = (candidate_pc - root) % 12
                    if relative_pc in scale_intervals:
                        # Calculate expected time for this note
                        # (linear interpolation)
                        time_ratio = offset / pitch_gap
                        expected_time = end1 + (start2 - end1) * time_ratio
                        expected_end = expected_time + (end2 - start2) / pitch_gap
                        
                        # Check if there's voiced audio at this time
                        frame = librosa.time_to_frames(expected_time, sr=self.sr, hop_length=self.hop_length)
                        
                        if frame < len(f0) and f0[frame] > 0:
                            # Verify pitch is close to expected
                            detected_midi = int(np.round(librosa.hz_to_midi(f0[frame])))
                            if abs(detected_midi - candidate_midi) <= 1:
                                missing_notes.append(LegatoNote(
                                    midi=candidate_midi,
                                    start_time=expected_time,
                                    end_time=expected_end,
                                    confidence=voiced_prob[frame] * 0.7,  # Lower confidence for inferred
                                    is_detected=False,
                                    pitch_source='scale_inference'
                                ))
        
        return missing_notes
    
    def _merge_with_existing(
        self,
        new_notes: List[LegatoNote],
        existing_notes: List
    ) -> List[LegatoNote]:
        """
        Merge new legato notes with existing notes, avoiding duplicates.
        """
        # Create a set of (start_time, midi) tuples from existing notes
        existing_set = set()
        for note in existing_notes:
            # Round time to avoid floating point issues
            t = round(note.start_time, 3)
            m = note.midi if hasattr(note, 'midi') else 0
            existing_set.add((t, m))
        
        # Filter out duplicates
        unique_notes = []
        for note in new_notes:
            t = round(note.start_time, 3)
            # Check if any existing note is within 50ms and same pitch
            is_duplicate = False
            for et, em in existing_set:
                if abs(t - et) < 0.05 and note.midi == em:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_notes.append(note)
        
        return unique_notes
    
    def _print_summary(
        self,
        notes: List[LegatoNote],
        runs: List[LegatoRun]
    ):
        """Print a summary of detected legato runs."""
        print(f"\n   📊 Summary:")
        print(f"      Total legato notes: {len(notes)}")
        print(f"      Legato runs: {len(runs)}")
        
        if runs:
            # Stats
            avg_nps = np.mean([r.notes_per_second for r in runs])
            max_nps = max(r.notes_per_second for r in runs)
            
            print(f"      Avg notes/sec: {avg_nps:.1f}")
            print(f"      Max notes/sec: {max_nps:.1f}")
            
            # Direction breakdown
            asc = sum(1 for r in runs if r.direction == 'ascending')
            desc = sum(1 for r in runs if r.direction == 'descending')
            mixed = sum(1 for r in runs if r.direction == 'mixed')
            print(f"      Directions: {asc} ascending, {desc} descending, {mixed} mixed")
            
            # Scale detection
            with_scale = sum(1 for r in runs if r.scale_type)
            if with_scale > 0:
                print(f"      Scale-matched runs: {with_scale}")
                scales = Counter(r.scale_type for r in runs if r.scale_type)
                for scale, count in scales.most_common(3):
                    print(f"         {scale}: {count}")
            
            # 3nps
            nps_3 = sum(1 for r in runs if r.is_3nps)
            if nps_3 > 0:
                print(f"      3-note-per-string runs: {nps_3}")


def detect_fast_legato_runs(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 256,
    min_note_duration_ms: float = 30,
    fast_run_threshold_ms: float = 125,
    min_run_notes: int = 4,
    enable_scale_matching: bool = True,
    verbose: bool = True
) -> Tuple[List[LegatoNote], List[LegatoRun]]:
    """
    Convenience function for fast legato run detection.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length: Hop length
        min_note_duration_ms: Minimum note duration
        fast_run_threshold_ms: Threshold for fast passage detection
        min_run_notes: Minimum notes to form a run
        enable_scale_matching: Use scale patterns to fill gaps
        verbose: Print diagnostic info
        
    Returns:
        (legato_notes, legato_runs)
    """
    # Load audio
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    
    # Create detector
    detector = FastLegatoDetector(
        sr=sr,
        hop_length=hop_length,
        min_note_duration_ms=min_note_duration_ms,
        fast_run_threshold_ms=fast_run_threshold_ms,
        min_run_notes=min_run_notes,
        enable_scale_matching=enable_scale_matching
    )
    
    # Detect
    notes, runs = detector.detect(y, verbose=verbose)
    
    return notes, runs


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fast_legato_runs.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"\n🎵 Processing: {audio_path}\n")
    
    notes, runs = detect_fast_legato_runs(audio_path, verbose=True)
    
    print("\n📋 Detected Legato Runs:")
    print("-" * 70)
    
    for i, run in enumerate(runs):
        print(f"\n  Run {i+1}: {run}")
        print(f"    Time: {run.start_time:.3f}s - {run.end_time:.3f}s")
        print(f"    Notes: {', '.join(n.note_name for n in run.notes)}")
        if run.scale_type:
            root_name = NOTE_NAMES[run.root_note % 12]
            print(f"    Scale: {root_name} {run.scale_type}")
        if run.is_3nps:
            print(f"    Pattern: 3-note-per-string")
    
    print(f"\n   Total runs: {len(runs)}")
    print(f"   Total legato notes: {len(notes)}")
