#!/usr/bin/env python3
"""
Multi-Resolution Pitch Detection for Guitar

Different musical passages need different analysis strategies:
- Fast solos/shred: Short windows (10ms hop) to capture rapid notes
- Sustained notes/bends: Long windows (50ms hop) for stable frequency
- Mixed passages: Adaptive window based on local note density

This module provides:
1. Short-window analysis for fast passages (hop ~10ms)
2. Long-window analysis for sustained notes (hop ~50ms)
3. Adaptive resolution selection based on onset density
4. Multi-resolution fusion combining all analysis scales

Key insight: Fast legato runs sound like garbage with long analysis windows
because the pitch changes faster than the window can track. Conversely,
sustained bends analyzed with tiny windows have excessive pitch jitter.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, medfilt, butter, filtfilt
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings

# Constants
GUITAR_MIN_HZ = 75
GUITAR_MAX_HZ = 1400
GUITAR_MIN_MIDI = 36
GUITAR_MAX_MIDI = 90
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class ResolutionConfig:
    """Configuration for a single analysis resolution."""
    name: str
    frame_length: int      # FFT window size
    hop_length: int        # Hop between frames
    hop_ms: float          # Hop in milliseconds (computed)
    fmin: float = GUITAR_MIN_HZ
    fmax: float = GUITAR_MAX_HZ
    # YIN/pYIN parameters
    yin_threshold: float = 0.15  # Lower = stricter voicing decision
    min_confidence: float = 0.4
    # Onset detection parameters
    onset_threshold: float = 0.3
    onset_backtrack: bool = True
    
    def __post_init__(self):
        # Auto-compute hop_ms if not provided
        pass


@dataclass
class MultiResConfig:
    """Configuration for multi-resolution pitch detection."""
    # Sample rate (set during init)
    sr: int = 22050
    
    # Resolution presets
    resolutions: Dict[str, ResolutionConfig] = field(default_factory=dict)
    
    # Adaptive mode settings
    adaptive_mode: bool = True
    density_window_ms: float = 500  # Window for computing note density
    fast_threshold_nps: float = 6   # Notes/sec above this = "fast"
    slow_threshold_nps: float = 3   # Notes/sec below this = "slow"
    
    # Fusion settings
    fusion_method: str = 'weighted_vote'  # 'weighted_vote', 'confidence_max', 'density_adaptive'
    pitch_tolerance_cents: float = 50     # Tolerance for clustering
    time_tolerance_ms: float = 20         # Time tolerance for matching
    
    # Output filtering
    min_note_duration_ms: float = 15
    min_confidence: float = 0.05  # Very low - let multi-resolution consensus handle quality
    
    def __post_init__(self):
        if not self.resolutions:
            # Default resolutions optimized for guitar
            # Note: pYIN confidence for guitar is often 0.1-0.4
            self.resolutions = {
                'ultra_fast': ResolutionConfig(
                    name='ultra_fast',
                    frame_length=512,     # ~23ms at 22050Hz
                    hop_length=128,       # ~6ms - for extreme shred
                    hop_ms=128 / self.sr * 1000,
                    yin_threshold=0.25,
                    min_confidence=0.1,   # Very low - capture fast notes
                    onset_threshold=0.15,
                ),
                'fast': ResolutionConfig(
                    name='fast',
                    frame_length=1024,    # ~46ms at 22050Hz
                    hop_length=256,       # ~11ms - for fast solos
                    hop_ms=256 / self.sr * 1000,
                    yin_threshold=0.2,
                    min_confidence=0.15,  # Low for solos
                    onset_threshold=0.2,
                ),
                'medium': ResolutionConfig(
                    name='medium',
                    frame_length=2048,    # ~93ms at 22050Hz
                    hop_length=512,       # ~23ms - balanced
                    hop_ms=512 / self.sr * 1000,
                    yin_threshold=0.15,
                    min_confidence=0.2,   # Moderate
                    onset_threshold=0.25,
                ),
                'slow': ResolutionConfig(
                    name='slow',
                    frame_length=4096,    # ~186ms at 22050Hz
                    hop_length=1024,      # ~46ms - for sustained notes
                    hop_ms=1024 / self.sr * 1000,
                    yin_threshold=0.1,
                    min_confidence=0.25,  # Higher for sustained
                    onset_threshold=0.3,
                ),
            }


@dataclass
class MultiResPitch:
    """A pitch detection from multi-resolution analysis."""
    time: float
    midi_note: int
    frequency: float
    confidence: float
    resolution: str       # Which resolution detected this
    onset_strength: float = 0.0  # Onset strength at this time


@dataclass
class MultiResNote:
    """A note with multi-resolution consensus."""
    midi_note: int
    start_time: float
    end_time: float
    confidence: float
    frequency: float
    resolutions_used: List[str]
    pitch_stability: float   # How stable the pitch is across resolutions
    is_fast_passage: bool    # Detected in fast passage
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)
    
    def __repr__(self):
        return f"MultiResNote({self.note_name}, t={self.start_time:.3f}s, dur={self.duration:.3f}s, conf={self.confidence:.2f})"


class MultiResolutionDetector:
    """
    Multi-resolution pitch detection for guitar.
    
    Analyzes audio at multiple time scales and combines results
    for optimal detection of both fast solos and sustained notes.
    """
    
    def __init__(self, sr: int = 22050, config: Optional[MultiResConfig] = None):
        self.sr = sr
        self.config = config or MultiResConfig(sr=sr)
        # Update hop_ms for all resolutions based on actual sr
        for res in self.config.resolutions.values():
            res.hop_ms = res.hop_length / sr * 1000
    
    def detect(self, y: np.ndarray) -> List[MultiResNote]:
        """
        Run multi-resolution pitch detection.
        
        Args:
            y: Audio signal (mono)
            
        Returns:
            List of notes with multi-resolution consensus
        """
        print(f"\n🎸 Multi-Resolution Pitch Detection")
        print(f"   Resolutions: {list(self.config.resolutions.keys())}")
        
        # Step 1: Compute onset density map to guide resolution selection
        density_map = self._compute_density_map(y)
        
        # Step 2: Run detection at each resolution
        all_pitches: Dict[str, List[MultiResPitch]] = {}
        for res_name, res_config in self.config.resolutions.items():
            print(f"\n   [{res_name}] hop={res_config.hop_ms:.1f}ms, frame={res_config.frame_length}")
            pitches = self._detect_at_resolution(y, res_config, density_map)
            all_pitches[res_name] = pitches
            print(f"   ✓ {len(pitches)} pitch detections")
        
        # Step 3: Fuse results from all resolutions
        print(f"\n   Fusing results ({self.config.fusion_method})...")
        notes = self._fuse_resolutions(all_pitches, density_map)
        
        print(f"   ✓ {len(notes)} fused notes")
        
        return notes
    
    def _compute_density_map(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute note density map to guide resolution selection.
        
        Returns dict with:
        - 'onset_env': Onset strength envelope
        - 'density': Note density (onsets per second)
        - 'is_fast': Boolean array where density > fast_threshold
        - 'times': Time array
        """
        print("   Computing density map...")
        
        # Use medium resolution for density estimation
        hop = 512
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=hop, aggregate=np.median
        )
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr, hop_length=hop)
        
        # Find onsets
        onsets = librosa.onset.onset_detect(
            y=y, sr=self.sr, hop_length=hop, backtrack=True,
            units='time'
        )
        
        # Compute density as onsets per second in sliding window
        window_samples = int(self.config.density_window_ms / 1000 * self.sr / hop)
        density = np.zeros_like(onset_env)
        
        onset_frames = librosa.time_to_frames(onsets, sr=self.sr, hop_length=hop)
        onset_indicator = np.zeros_like(onset_env)
        onset_indicator[np.clip(onset_frames, 0, len(onset_indicator)-1)] = 1
        
        # Sliding window count
        for i in range(len(density)):
            start = max(0, i - window_samples // 2)
            end = min(len(density), i + window_samples // 2)
            window_duration = (end - start) * hop / self.sr
            if window_duration > 0:
                density[i] = np.sum(onset_indicator[start:end]) / window_duration
        
        # Smooth density
        density = uniform_filter1d(density, size=5)
        
        # Classify regions
        is_fast = density > self.config.fast_threshold_nps
        is_slow = density < self.config.slow_threshold_nps
        
        fast_pct = np.mean(is_fast) * 100
        slow_pct = np.mean(is_slow) * 100
        print(f"   Density: {fast_pct:.1f}% fast, {slow_pct:.1f}% slow, {100-fast_pct-slow_pct:.1f}% medium")
        
        return {
            'onset_env': onset_env,
            'density': density,
            'is_fast': is_fast,
            'is_slow': is_slow,
            'times': times,
            'hop': hop,
        }
    
    def _detect_at_resolution(
        self, 
        y: np.ndarray, 
        res: ResolutionConfig,
        density_map: Dict
    ) -> List[MultiResPitch]:
        """
        Detect pitches at a specific resolution.
        
        Uses pYIN for robust pitch detection with appropriate parameters.
        """
        hop = res.hop_length
        
        # pYIN pitch detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=res.fmin,
                fmax=res.fmax,
                sr=self.sr,
                frame_length=res.frame_length,
                hop_length=hop,
                fill_na=0.0
            )
        
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=hop)
        
        # Onset strength for this resolution
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=hop, n_fft=res.frame_length
        )
        onset_env = onset_env[:len(f0)]  # Align lengths
        if len(onset_env) < len(f0):
            onset_env = np.pad(onset_env, (0, len(f0) - len(onset_env)))
        
        pitches = []
        voiced_count = 0
        low_conf_count = 0
        out_of_range_count = 0
        
        for i, (freq, voiced, conf) in enumerate(zip(f0, voiced_flag, voiced_probs)):
            if voiced and freq > 0:
                voiced_count += 1
                midi = int(round(librosa.hz_to_midi(freq)))
                
                # More permissive MIDI range (extended for bass and high notes)
                if not (24 <= midi <= 108):
                    out_of_range_count += 1
                    continue
                
                # Very permissive confidence - let the fusion handle quality
                actual_min_conf = max(0.01, res.min_confidence * 0.5)  # Halve the threshold
                if conf < actual_min_conf:
                    low_conf_count += 1
                    continue
                
                pitches.append(MultiResPitch(
                    time=times[i],
                    midi_note=midi,
                    frequency=float(freq),
                    confidence=float(conf),
                    resolution=res.name,
                    onset_strength=float(onset_env[i]) if i < len(onset_env) else 0.0
                ))
        
        if voiced_count > 0 and len(pitches) == 0:
            print(f"      ⚠️ {voiced_count} voiced, {low_conf_count} low conf, {out_of_range_count} out of range")
        
        return pitches
    
    def _fuse_resolutions(
        self,
        all_pitches: Dict[str, List[MultiResPitch]],
        density_map: Dict
    ) -> List[MultiResNote]:
        """
        Fuse pitch detections from multiple resolutions into notes.
        
        Uses density-adaptive weighting:
        - Fast regions: prefer short-window detections
        - Slow regions: prefer long-window detections
        - Mixed regions: weighted combination
        """
        # Collect all detections with their weights
        weighted_pitches = []
        
        # Resolution weights based on speed characteristics
        base_weights = {
            'ultra_fast': 1.2,
            'fast': 1.0,
            'medium': 0.9,
            'slow': 0.8,
        }
        
        for res_name, pitches in all_pitches.items():
            base_weight = base_weights.get(res_name, 1.0)
            
            for p in pitches:
                # Find density at this time
                time_idx = np.searchsorted(density_map['times'], p.time)
                time_idx = min(time_idx, len(density_map['is_fast']) - 1)
                
                is_fast = density_map['is_fast'][time_idx]
                is_slow = density_map['is_slow'][time_idx]
                
                # Adaptive weight based on density
                if is_fast:
                    # Fast passage: boost short windows
                    if res_name in ['ultra_fast', 'fast']:
                        weight = base_weight * 1.5
                    elif res_name == 'medium':
                        weight = base_weight * 0.8
                    else:
                        weight = base_weight * 0.5
                elif is_slow:
                    # Slow passage: boost long windows
                    if res_name == 'slow':
                        weight = base_weight * 1.5
                    elif res_name == 'medium':
                        weight = base_weight * 1.2
                    else:
                        weight = base_weight * 0.7
                else:
                    # Medium density: balanced weights
                    weight = base_weight
                
                weighted_pitches.append((p, weight, is_fast))
        
        # Sort by time
        weighted_pitches.sort(key=lambda x: x[0].time)
        
        if not weighted_pitches:
            return []
        
        # Cluster nearby pitches
        notes = self._cluster_to_notes(weighted_pitches, density_map)
        
        return notes
    
    def _cluster_to_notes(
        self,
        weighted_pitches: List[Tuple[MultiResPitch, float, bool]],
        density_map: Dict
    ) -> List[MultiResNote]:
        """
        Cluster weighted pitch detections into notes using onset-guided segmentation.
        """
        if not weighted_pitches:
            return []
        
        time_tol = self.config.time_tolerance_ms / 1000
        pitch_tol_semitones = self.config.pitch_tolerance_cents / 100
        min_duration = self.config.min_note_duration_ms / 1000
        
        # Group by time windows
        time_groups = []
        current_group = [weighted_pitches[0]]
        
        for wp in weighted_pitches[1:]:
            if wp[0].time - current_group[-1][0].time <= time_tol:
                current_group.append(wp)
            else:
                time_groups.append(current_group)
                current_group = [wp]
        time_groups.append(current_group)
        
        # For each time group, vote on the pitch
        pitch_timeline = []
        for group in time_groups:
            # Weighted vote for pitch
            pitch_votes = defaultdict(float)
            confidences = defaultdict(list)
            resolutions = defaultdict(set)
            freqs = defaultdict(list)
            is_fast_flags = []
            
            for p, weight, is_fast in group:
                pitch_votes[p.midi_note] += weight * p.confidence
                confidences[p.midi_note].append(p.confidence)
                resolutions[p.midi_note].add(p.resolution)
                freqs[p.midi_note].append(p.frequency)
                is_fast_flags.append(is_fast)
            
            if pitch_votes:
                # Winner takes all
                winning_pitch = max(pitch_votes.keys(), key=lambda x: pitch_votes[x])
                avg_time = np.mean([p.time for p, _, _ in group])
                avg_conf = np.mean(confidences[winning_pitch])
                avg_freq = np.mean(freqs[winning_pitch])
                res_used = list(resolutions[winning_pitch])
                is_fast = any(is_fast_flags)
                
                pitch_timeline.append({
                    'midi': winning_pitch,
                    'time': avg_time,
                    'confidence': avg_conf,
                    'frequency': avg_freq,
                    'resolutions': res_used,
                    'is_fast': is_fast,
                    'vote_strength': pitch_votes[winning_pitch],
                })
        
        # Segment into notes based on pitch changes
        notes = []
        if not pitch_timeline:
            return notes
        
        current_note = pitch_timeline[0].copy()
        current_note['start_time'] = current_note['time']
        current_note['end_time'] = current_note['time']
        current_note['all_freqs'] = [current_note['frequency']]
        current_note['all_res'] = set(current_note['resolutions'])
        
        for pt in pitch_timeline[1:]:
            # Check if this continues the current note or starts a new one
            pitch_changed = abs(pt['midi'] - current_note['midi']) > pitch_tol_semitones
            
            if pitch_changed:
                # Finalize current note
                if current_note['end_time'] - current_note['start_time'] >= min_duration:
                    notes.append(self._create_note(current_note))
                
                # Start new note
                current_note = pt.copy()
                current_note['start_time'] = pt['time']
                current_note['end_time'] = pt['time']
                current_note['all_freqs'] = [pt['frequency']]
                current_note['all_res'] = set(pt['resolutions'])
            else:
                # Continue current note
                current_note['end_time'] = pt['time']
                current_note['confidence'] = (current_note['confidence'] + pt['confidence']) / 2
                current_note['all_freqs'].append(pt['frequency'])
                current_note['all_res'].update(pt['resolutions'])
                current_note['is_fast'] = current_note['is_fast'] or pt['is_fast']
        
        # Finalize last note
        if current_note['end_time'] - current_note['start_time'] >= min_duration:
            notes.append(self._create_note(current_note))
        
        # Filter by confidence
        notes = [n for n in notes if n.confidence >= self.config.min_confidence]
        
        return notes
    
    def _create_note(self, note_data: Dict) -> MultiResNote:
        """Create a MultiResNote from accumulated data."""
        all_freqs = note_data['all_freqs']
        freq_std = np.std(all_freqs) if len(all_freqs) > 1 else 0
        avg_freq = np.mean(all_freqs)
        
        # Pitch stability: how much the pitch varies (in cents)
        if avg_freq > 0:
            stability = 1.0 - min(1.0, freq_std / avg_freq * 100)
        else:
            stability = 0.0
        
        return MultiResNote(
            midi_note=note_data['midi'],
            start_time=note_data['start_time'],
            end_time=note_data['end_time'],
            confidence=note_data['confidence'],
            frequency=avg_freq,
            resolutions_used=list(note_data['all_res']),
            pitch_stability=stability,
            is_fast_passage=note_data['is_fast'],
        )
    
    def detect_adaptive(self, y: np.ndarray) -> List[MultiResNote]:
        """
        Adaptive resolution detection - automatically selects resolution
        based on local characteristics.
        
        This is more efficient than running all resolutions, using
        the appropriate resolution for each segment.
        """
        print(f"\n🎸 Adaptive Multi-Resolution Detection")
        
        # Step 1: Segment audio by density
        density_map = self._compute_density_map(y)
        segments = self._segment_by_density(density_map, y)
        
        print(f"   Segmented into {len(segments)} regions")
        
        all_notes = []
        
        # Step 2: Process each segment with appropriate resolution
        for seg in segments:
            start_sample = int(seg['start_time'] * self.sr)
            end_sample = int(seg['end_time'] * self.sr)
            y_seg = y[start_sample:end_sample]
            
            if len(y_seg) < 1024:
                continue
            
            # Select resolution based on segment type
            if seg['type'] == 'fast':
                res_names = ['ultra_fast', 'fast']
            elif seg['type'] == 'slow':
                res_names = ['slow', 'medium']
            else:
                res_names = ['fast', 'medium']
            
            print(f"   [{seg['start_time']:.2f}s-{seg['end_time']:.2f}s] {seg['type']} → {res_names}")
            
            # Run detection at selected resolutions
            seg_pitches = {}
            for res_name in res_names:
                if res_name in self.config.resolutions:
                    res = self.config.resolutions[res_name]
                    pitches = self._detect_at_resolution(y_seg, res, density_map)
                    # Adjust times for segment offset
                    for p in pitches:
                        p.time += seg['start_time']
                    seg_pitches[res_name] = pitches
            
            # Debug: count pitches in this segment
            seg_pitch_count = sum(len(p) for p in seg_pitches.values())
            if seg_pitch_count > 0:
                print(f"      → {seg_pitch_count} pitch detections")
            
            # Fuse segment results
            seg_notes = self._fuse_resolutions(seg_pitches, density_map)
            if seg_notes:
                print(f"      → {len(seg_notes)} fused notes")
            all_notes.extend(seg_notes)
        
        # Sort by time and deduplicate
        all_notes.sort(key=lambda n: n.start_time)
        all_notes = self._deduplicate_notes(all_notes)
        
        print(f"   ✓ {len(all_notes)} total notes")
        
        return all_notes
    
    def _segment_by_density(self, density_map: Dict, y: np.ndarray) -> List[Dict]:
        """Segment audio into regions by note density."""
        times = density_map['times']
        is_fast = density_map['is_fast']
        is_slow = density_map['is_slow']
        
        segments = []
        current_type = 'medium'
        current_start = 0.0
        
        for i, t in enumerate(times):
            if is_fast[i]:
                new_type = 'fast'
            elif is_slow[i]:
                new_type = 'slow'
            else:
                new_type = 'medium'
            
            if new_type != current_type:
                if t - current_start > 0.1:  # Minimum segment length
                    segments.append({
                        'type': current_type,
                        'start_time': current_start,
                        'end_time': t,
                    })
                current_type = new_type
                current_start = t
        
        # Final segment
        if len(times) > 0:
            segments.append({
                'type': current_type,
                'start_time': current_start,
                'end_time': times[-1],
            })
        
        # Merge very short segments
        merged = []
        for seg in segments:
            if merged and seg['start_time'] - merged[-1]['end_time'] < 0.05:
                # Merge with previous if same type
                if seg['type'] == merged[-1]['type']:
                    merged[-1]['end_time'] = seg['end_time']
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        
        return merged
    
    def _deduplicate_notes(self, notes: List[MultiResNote]) -> List[MultiResNote]:
        """Remove duplicate notes that overlap significantly."""
        if not notes:
            return notes
        
        deduplicated = [notes[0]]
        
        for note in notes[1:]:
            last = deduplicated[-1]
            
            # Check for overlap
            overlap = min(last.end_time, note.end_time) - max(last.start_time, note.start_time)
            if overlap > 0:
                # Same pitch = merge
                if note.midi_note == last.midi_note:
                    last.end_time = max(last.end_time, note.end_time)
                    last.confidence = max(last.confidence, note.confidence)
                    last.resolutions_used = list(set(last.resolutions_used + note.resolutions_used))
                elif note.confidence > last.confidence:
                    # Different pitch, higher confidence = replace
                    deduplicated[-1] = note
                # else: keep the original (higher confidence)
            else:
                deduplicated.append(note)
        
        return deduplicated


def detect_multiresolution(
    y: np.ndarray,
    sr: int = 22050,
    adaptive: bool = True,
    config: Optional[MultiResConfig] = None
) -> List[MultiResNote]:
    """
    Convenience function for multi-resolution pitch detection.
    
    Args:
        y: Audio signal
        sr: Sample rate
        adaptive: If True, use adaptive mode (faster)
        config: Optional configuration
        
    Returns:
        List of detected notes
    """
    detector = MultiResolutionDetector(sr=sr, config=config)
    
    if adaptive:
        return detector.detect_adaptive(y)
    else:
        return detector.detect(y)


def convert_to_ensemble_format(notes: List[MultiResNote]) -> List[dict]:
    """Convert MultiResNotes to format compatible with ensemble_pitch."""
    return [
        {
            'midi_note': n.midi_note,
            'start_time': n.start_time,
            'end_time': n.end_time,
            'confidence': n.confidence,
            'frequency': n.frequency,
            'name': n.note_name,
            'duration': n.duration,
            'methods': n.resolutions_used,
        }
        for n in notes
    ]


# CLI interface for testing
if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Multi-resolution pitch detection')
    parser.add_argument('audio_file', help='Input audio file')
    parser.add_argument('--adaptive', action='store_true', default=True,
                       help='Use adaptive mode (default: True)')
    parser.add_argument('--full', action='store_true',
                       help='Run all resolutions (not adaptive)')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--fast-threshold', type=float, default=6,
                       help='Notes/sec threshold for fast passages')
    parser.add_argument('--slow-threshold', type=float, default=3,
                       help='Notes/sec threshold for slow passages')
    
    args = parser.parse_args()
    
    print(f"Loading {args.audio_file}...")
    y, sr = librosa.load(args.audio_file, sr=22050, mono=True)
    print(f"Loaded {len(y)/sr:.1f}s of audio at {sr}Hz")
    
    config = MultiResConfig(
        sr=sr,
        fast_threshold_nps=args.fast_threshold,
        slow_threshold_nps=args.slow_threshold,
    )
    
    adaptive = not args.full
    notes = detect_multiresolution(y, sr=sr, adaptive=adaptive, config=config)
    
    print(f"\n=== Detected {len(notes)} notes ===")
    for n in notes[:20]:
        fast_flag = "⚡" if n.is_fast_passage else "  "
        print(f"  {fast_flag} {n.note_name:4s} {n.start_time:6.3f}s - {n.end_time:6.3f}s "
              f"(dur={n.duration:.3f}s, conf={n.confidence:.2f}, res={n.resolutions_used})")
    
    if len(notes) > 20:
        print(f"  ... and {len(notes)-20} more")
    
    if args.output:
        output_data = {
            'audio_file': args.audio_file,
            'config': {
                'adaptive': adaptive,
                'fast_threshold_nps': config.fast_threshold_nps,
                'slow_threshold_nps': config.slow_threshold_nps,
            },
            'notes': convert_to_ensemble_format(notes),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")
