#!/usr/bin/env python3
"""
Reassigned Spectrogram Pitch Detection for Guitar

Why Reassigned Spectrogram?
---------------------------
Standard STFT smears energy across time-frequency bins due to windowing.
The reassigned spectrogram computes "instantaneous" time and frequency
coordinates for each bin, giving:

1. SHARPER TIME RESOLUTION - Notes start/stop more precisely
2. SHARPER FREQUENCY RESOLUTION - Less spectral leakage between notes
3. BETTER HARMONIC SEPARATION - Cleaner partial tracking
4. REDUCED INTERFERENCE - Less cross-talk between simultaneous tones

librosa.reassigned_spectrogram() returns:
- freqs: Reassigned frequency for each bin (instantaneous frequency)
- times: Reassigned time for each bin (group delay)
- mags: Magnitude values

We extract pitch by:
1. Computing reassigned spectrogram
2. Filtering to guitar frequency range
3. Finding frequency peaks in reassigned representation
4. Tracking pitch trajectories through reassigned coordinates
5. Converting to MIDI notes

This should give CLEANER pitch tracks than standard STFT-based methods,
especially for fast passages and distorted tones.
"""

import numpy as np
import librosa
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Constants
GUITAR_MIN_HZ = 75      # E2 ~82 Hz, but allow some margin
GUITAR_MAX_HZ = 1400    # ~E6 (high frets + harmonics)
GUITAR_MIN_MIDI = 36    # C2
GUITAR_MAX_MIDI = 90    # F#6
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_midi(hz: float) -> float:
    """Convert frequency to MIDI note number."""
    if hz <= 0:
        return 0
    return 69 + 12 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI number to note name."""
    if midi <= 0:
        return '-'
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


@dataclass
class ReassignedConfig:
    """Configuration for reassigned spectrogram pitch detection."""
    # Sample rate (set during analysis)
    sr: int = 22050
    
    # FFT parameters - shorter windows give better time resolution
    n_fft: int = 2048           # FFT size
    hop_length: int = 256       # ~11ms at 22050 Hz
    win_length: int = 1024      # Window shorter than FFT for zero-padding
    
    # Frequency range for guitar
    fmin: float = GUITAR_MIN_HZ
    fmax: float = GUITAR_MAX_HZ
    
    # Peak detection in reassigned spectrogram
    magnitude_threshold_db: float = -40    # Ignore bins below this
    peak_prominence: float = 3.0           # dB prominence for peaks
    
    # Pitch candidate filtering
    min_confidence: float = 0.3
    max_candidates_per_frame: int = 5      # Top N candidates per frame
    
    # Temporal smoothing
    median_filter_frames: int = 3          # Frames for median smoothing
    min_note_frames: int = 4               # Min frames to form a note
    
    # Output quantization
    pitch_tolerance_cents: float = 50      # For clustering nearby pitches
    
    # Multi-resolution (optional)
    multi_resolution: bool = True
    hop_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])


@dataclass
class PitchCandidate:
    """A pitch candidate from reassigned analysis."""
    time: float           # Reassigned time (seconds)
    frequency: float      # Reassigned frequency (Hz)
    magnitude: float      # Magnitude (linear)
    magnitude_db: float   # Magnitude (dB)
    confidence: float     # Derived confidence 0-1
    frame_idx: int        # Original frame index
    
    @property
    def midi(self) -> float:
        return hz_to_midi(self.frequency)
    
    @property
    def midi_rounded(self) -> int:
        return int(round(self.midi))


@dataclass
class DetectedNote:
    """A detected note from reassigned spectrogram analysis."""
    midi_note: int
    start_time: float
    end_time: float
    frequency: float      # Average frequency
    confidence: float     # Average confidence
    magnitude_db: float   # Peak magnitude
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def note_name(self) -> str:
        return midi_to_note_name(self.midi_note)


class ReassignedSpectrogramDetector:
    """
    Pitch detector using librosa's reassigned spectrogram.
    
    The reassigned spectrogram provides instantaneous frequency estimates
    with better time-frequency localization than standard STFT.
    """
    
    def __init__(self, config: Optional[ReassignedConfig] = None):
        self.config = config or ReassignedConfig()
    
    def analyze(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reassigned spectrogram.
        
        Returns:
            freqs: Reassigned frequencies for each bin [freq_bins, time_frames]
            times: Reassigned times for each bin [freq_bins, time_frames]
            mags: Magnitudes [freq_bins, time_frames]
        """
        self.config.sr = sr
        
        # Compute reassigned spectrogram
        freqs, times, mags = librosa.reassigned_spectrogram(
            y=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            center=True,
            pad_mode='reflect'
        )
        
        return freqs, times, mags
    
    def extract_pitch_candidates(
        self, 
        freqs: np.ndarray, 
        times: np.ndarray, 
        mags: np.ndarray,
        sr: int
    ) -> List[List[PitchCandidate]]:
        """
        Extract pitch candidates from reassigned spectrogram.
        
        For each time frame, finds peaks in the reassigned representation
        and returns candidates in guitar frequency range.
        """
        n_bins, n_frames = mags.shape
        config = self.config
        
        # Convert magnitudes to dB
        mags_db = librosa.amplitude_to_db(mags, ref=np.max)
        
        # Frame times
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), 
            sr=sr, 
            hop_length=config.hop_length
        )
        
        candidates_per_frame = []
        
        for frame_idx in range(n_frames):
            frame_candidates = []
            frame_mag = mags[:, frame_idx]
            frame_mag_db = mags_db[:, frame_idx]
            frame_freq = freqs[:, frame_idx]
            frame_time = times[:, frame_idx]
            
            # Mask to guitar frequency range
            valid_mask = (
                (frame_freq >= config.fmin) & 
                (frame_freq <= config.fmax) &
                (frame_mag_db >= config.magnitude_threshold_db) &
                np.isfinite(frame_freq)
            )
            
            if not np.any(valid_mask):
                candidates_per_frame.append([])
                continue
            
            valid_indices = np.where(valid_mask)[0]
            valid_mags = frame_mag_db[valid_indices]
            valid_freqs = frame_freq[valid_indices]
            valid_times_r = frame_time[valid_indices]
            
            # Find peaks in magnitude
            if len(valid_mags) > 2:
                peaks, properties = find_peaks(
                    valid_mags,
                    prominence=config.peak_prominence,
                    distance=3  # Min distance between peaks
                )
            else:
                peaks = np.arange(len(valid_mags))
            
            if len(peaks) == 0:
                candidates_per_frame.append([])
                continue
            
            # Sort by magnitude, take top N
            peak_mags = valid_mags[peaks]
            sorted_peak_indices = np.argsort(peak_mags)[::-1][:config.max_candidates_per_frame]
            top_peaks = peaks[sorted_peak_indices]
            
            # Create candidates
            max_mag_db = np.max(frame_mag_db)
            for peak in top_peaks:
                freq = valid_freqs[peak]
                mag_db = valid_mags[peak]
                reassigned_time = valid_times_r[peak]
                
                # Confidence based on relative magnitude
                # Higher mag relative to frame max = higher confidence
                confidence = np.clip((mag_db - config.magnitude_threshold_db) / 
                                    (max_mag_db - config.magnitude_threshold_db + 1e-6), 0, 1)
                
                candidate = PitchCandidate(
                    time=reassigned_time if np.isfinite(reassigned_time) else frame_times[frame_idx],
                    frequency=freq,
                    magnitude=frame_mag[valid_indices[peak]],
                    magnitude_db=mag_db,
                    confidence=confidence,
                    frame_idx=frame_idx
                )
                frame_candidates.append(candidate)
            
            # Sort by confidence
            frame_candidates.sort(key=lambda c: c.confidence, reverse=True)
            candidates_per_frame.append(frame_candidates)
        
        return candidates_per_frame
    
    def track_pitches(
        self, 
        candidates_per_frame: List[List[PitchCandidate]],
        sr: int
    ) -> List[Tuple[float, float, float]]:
        """
        Track pitches through frames to form continuous trajectories.
        
        Returns list of (time, frequency, confidence) tuples.
        """
        config = self.config
        hop_time = config.hop_length / sr
        
        # For each frame, select the best pitch (highest confidence in valid range)
        tracked = []
        
        for frame_idx, candidates in enumerate(candidates_per_frame):
            frame_time = frame_idx * hop_time
            
            if not candidates:
                tracked.append((frame_time, 0.0, 0.0))
                continue
            
            # Get best candidate
            best = candidates[0]
            if best.confidence >= config.min_confidence:
                tracked.append((frame_time, best.frequency, best.confidence))
            else:
                tracked.append((frame_time, 0.0, 0.0))
        
        return tracked
    
    def smooth_pitches(
        self, 
        tracked: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Apply temporal smoothing to pitch tracks.
        Removes spurious octave jumps and brief dropouts.
        """
        if not tracked:
            return tracked
        
        config = self.config
        times = np.array([t[0] for t in tracked])
        freqs = np.array([t[1] for t in tracked])
        confs = np.array([t[2] for t in tracked])
        
        # Median filter on frequencies (ignoring zeros)
        if config.median_filter_frames > 1:
            # Convert to MIDI for smoother filtering
            midis = np.array([hz_to_midi(f) if f > 0 else 0 for f in freqs])
            
            # Only filter non-zero regions
            voiced = midis > 0
            if np.sum(voiced) > config.median_filter_frames:
                # Median filter voiced regions
                voiced_midis = midis[voiced]
                filtered_midis = medfilt(voiced_midis, kernel_size=min(config.median_filter_frames, len(voiced_midis) | 1))
                midis[voiced] = filtered_midis
                
                # Convert back to Hz
                freqs = np.array([midi_to_hz(m) if m > 0 else 0 for m in midis])
        
        # Interpolate brief gaps (1-2 frames)
        gap_frames = 2
        for i in range(gap_frames, len(freqs) - gap_frames):
            if freqs[i] == 0:
                # Check if surrounded by voiced frames
                before = freqs[max(0, i-gap_frames):i]
                after = freqs[i+1:min(len(freqs), i+gap_frames+1)]
                
                before_voiced = before[before > 0]
                after_voiced = after[after > 0]
                
                if len(before_voiced) > 0 and len(after_voiced) > 0:
                    # Check if frequencies are similar
                    avg_before = np.mean(before_voiced)
                    avg_after = np.mean(after_voiced)
                    cents_diff = abs(1200 * np.log2(avg_after / avg_before))
                    
                    if cents_diff < 200:  # Within 2 semitones
                        freqs[i] = (avg_before + avg_after) / 2
                        confs[i] = min(confs[i-1], confs[i+1]) * 0.8
        
        return list(zip(times, freqs, confs))
    
    def segment_notes(
        self, 
        tracked: List[Tuple[float, float, float]]
    ) -> List[DetectedNote]:
        """
        Segment continuous pitch track into discrete notes.
        """
        if not tracked:
            return []
        
        config = self.config
        notes = []
        
        # Group consecutive frames with same MIDI note
        current_note = None
        current_start = 0
        current_freqs = []
        current_confs = []
        current_mags = []
        
        for time, freq, conf in tracked:
            if freq <= 0:
                # End current note if exists
                if current_note is not None and len(current_freqs) >= config.min_note_frames:
                    notes.append(DetectedNote(
                        midi_note=current_note,
                        start_time=current_start,
                        end_time=time,
                        frequency=np.mean(current_freqs),
                        confidence=np.mean(current_confs),
                        magnitude_db=np.max(current_mags) if current_mags else -60
                    ))
                current_note = None
                current_freqs = []
                current_confs = []
                continue
            
            midi = int(round(hz_to_midi(freq)))
            
            if midi < GUITAR_MIN_MIDI or midi > GUITAR_MAX_MIDI:
                continue
            
            if current_note is None:
                # Start new note
                current_note = midi
                current_start = time
                current_freqs = [freq]
                current_confs = [conf]
                current_mags = [conf * 60 - 60]  # Approx dB from confidence
            elif midi == current_note:
                # Continue current note
                current_freqs.append(freq)
                current_confs.append(conf)
                current_mags.append(conf * 60 - 60)
            else:
                # Note changed - end current and start new
                if len(current_freqs) >= config.min_note_frames:
                    notes.append(DetectedNote(
                        midi_note=current_note,
                        start_time=current_start,
                        end_time=time,
                        frequency=np.mean(current_freqs),
                        confidence=np.mean(current_confs),
                        magnitude_db=np.max(current_mags) if current_mags else -60
                    ))
                
                current_note = midi
                current_start = time
                current_freqs = [freq]
                current_confs = [conf]
                current_mags = [conf * 60 - 60]
        
        # Don't forget last note
        if current_note is not None and len(current_freqs) >= config.min_note_frames:
            notes.append(DetectedNote(
                midi_note=current_note,
                start_time=current_start,
                end_time=tracked[-1][0] + config.hop_length / config.sr,
                frequency=np.mean(current_freqs),
                confidence=np.mean(current_confs),
                magnitude_db=np.max(current_mags) if current_mags else -60
            ))
        
        return notes
    
    def detect_multi_resolution(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> List[DetectedNote]:
        """
        Multi-resolution reassigned analysis.
        
        Runs analysis at multiple hop lengths and combines results
        for better accuracy across different note speeds.
        """
        all_notes = []
        note_counts = defaultdict(list)  # (midi, start_quantized) -> list of notes
        
        for hop_length in self.config.hop_lengths:
            # Create config for this resolution
            res_config = ReassignedConfig(
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=hop_length,
                win_length=min(self.config.win_length, self.config.n_fft // 2),
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                magnitude_threshold_db=self.config.magnitude_threshold_db,
                min_confidence=self.config.min_confidence,
                min_note_frames=max(2, self.config.min_note_frames * 256 // hop_length),
                multi_resolution=False
            )
            
            detector = ReassignedSpectrogramDetector(res_config)
            notes = detector.detect(audio, sr)
            
            # Quantize start times for matching
            time_quantum = 0.05  # 50ms bins
            for note in notes:
                start_q = int(note.start_time / time_quantum)
                key = (note.midi_note, start_q)
                note_counts[key].append(note)
        
        # Merge notes that appear in multiple resolutions
        merged_notes = []
        for key, notes_list in note_counts.items():
            if len(notes_list) >= 1:  # At least one resolution found it
                # Average properties
                avg_start = np.mean([n.start_time for n in notes_list])
                avg_end = np.mean([n.end_time for n in notes_list])
                avg_freq = np.mean([n.frequency for n in notes_list])
                avg_conf = np.mean([n.confidence for n in notes_list])
                max_mag = np.max([n.magnitude_db for n in notes_list])
                
                # Boost confidence if found in multiple resolutions
                boost = 1 + 0.2 * (len(notes_list) - 1)
                
                merged_notes.append(DetectedNote(
                    midi_note=key[0],
                    start_time=avg_start,
                    end_time=avg_end,
                    frequency=avg_freq,
                    confidence=min(1.0, avg_conf * boost),
                    magnitude_db=max_mag
                ))
        
        # Sort by start time
        merged_notes.sort(key=lambda n: n.start_time)
        
        return merged_notes
    
    def detect(self, audio: np.ndarray, sr: int) -> List[DetectedNote]:
        """
        Main entry point: detect notes using reassigned spectrogram.
        
        Args:
            audio: Audio samples (mono)
            sr: Sample rate
            
        Returns:
            List of detected notes
        """
        if self.config.multi_resolution:
            return self.detect_multi_resolution(audio, sr)
        
        # Single resolution analysis
        freqs, times, mags = self.analyze(audio, sr)
        candidates = self.extract_pitch_candidates(freqs, times, mags, sr)
        tracked = self.track_pitches(candidates, sr)
        smoothed = self.smooth_pitches(tracked)
        notes = self.segment_notes(smoothed)
        
        return notes
    
    def detect_with_debug(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> Tuple[List[DetectedNote], Dict]:
        """
        Detect notes with debug information.
        """
        freqs, times, mags = self.analyze(audio, sr)
        candidates = self.extract_pitch_candidates(freqs, times, mags, sr)
        tracked = self.track_pitches(candidates, sr)
        smoothed = self.smooth_pitches(tracked)
        notes = self.segment_notes(smoothed)
        
        debug = {
            'freqs_shape': freqs.shape,
            'times_shape': times.shape,
            'mags_shape': mags.shape,
            'n_candidates': sum(len(c) for c in candidates),
            'n_tracked': sum(1 for t in tracked if t[1] > 0),
            'n_smoothed': sum(1 for s in smoothed if s[1] > 0),
            'n_notes': len(notes),
            'tracked': tracked,
            'smoothed': smoothed,
        }
        
        return notes, debug


def detect_notes_reassigned(
    audio_path: str,
    config: Optional[ReassignedConfig] = None
) -> List[DetectedNote]:
    """
    Convenience function to detect notes from an audio file.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Create detector
    detector = ReassignedSpectrogramDetector(config)
    
    # Detect notes
    return detector.detect(audio, sr)


def notes_to_tab_string(notes: List[DetectedNote]) -> str:
    """Convert detected notes to a simple tab representation."""
    if not notes:
        return "No notes detected"
    
    # Standard tuning MIDI notes
    TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
    STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']
    
    lines = []
    lines.append(f"Detected {len(notes)} notes using Reassigned Spectrogram:")
    lines.append("")
    
    # Group by time bins
    time_quantum = 0.1  # 100ms bins
    time_bins = defaultdict(list)
    
    for note in notes:
        bin_idx = int(note.start_time / time_quantum)
        time_bins[bin_idx].append(note)
    
    # Output each time bin
    for bin_idx in sorted(time_bins.keys()):
        bin_notes = time_bins[bin_idx]
        time_str = f"[{bin_idx * time_quantum:.2f}s]"
        
        note_strs = []
        for note in bin_notes:
            # Find best string
            best_string = 0
            best_fret = note.midi_note - TUNING[0]
            
            for s, open_midi in enumerate(TUNING):
                fret = note.midi_note - open_midi
                if 0 <= fret <= 24:
                    if best_fret < 0 or fret < best_fret:
                        best_string = s
                        best_fret = fret
            
            if 0 <= best_fret <= 24:
                note_strs.append(f"{STRING_NAMES[best_string]}:{best_fret} ({note.note_name}, conf={note.confidence:.2f})")
            else:
                note_strs.append(f"?:{note.midi_note} ({note.note_name})")
        
        lines.append(f"{time_str} {', '.join(note_strs)}")
    
    return '\n'.join(lines)


# Integration with ensemble detector
def detect_for_ensemble(
    audio: np.ndarray, 
    sr: int, 
    config: Optional[ReassignedConfig] = None
) -> List[Tuple[float, int, float, float]]:
    """
    Detect pitches for use in ensemble voting.
    
    Returns list of (time, midi_note, frequency, confidence) tuples
    compatible with ensemble_pitch.py format.
    """
    detector = ReassignedSpectrogramDetector(config or ReassignedConfig())
    
    # Get raw tracked pitches before segmentation
    freqs, times, mags = detector.analyze(audio, sr)
    candidates = detector.extract_pitch_candidates(freqs, times, mags, sr)
    tracked = detector.track_pitches(candidates, sr)
    smoothed = detector.smooth_pitches(tracked)
    
    results = []
    for time, freq, conf in smoothed:
        if freq > 0 and conf > 0.2:
            midi = int(round(hz_to_midi(freq)))
            if GUITAR_MIN_MIDI <= midi <= GUITAR_MAX_MIDI:
                results.append((time, midi, freq, conf))
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reassigned_spectrogram.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Analyzing: {audio_path}")
    print("Using Reassigned Spectrogram for sharper time-frequency resolution...")
    print()
    
    # Load and analyze
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"Loaded: {len(audio)/sr:.2f}s at {sr}Hz")
    
    # Detect with debug info
    config = ReassignedConfig(
        hop_length=256,
        min_confidence=0.3,
        multi_resolution=True,
        hop_lengths=[128, 256, 512]
    )
    detector = ReassignedSpectrogramDetector(config)
    notes, debug = detector.detect_with_debug(audio, sr)
    
    print(f"\nDebug info:")
    print(f"  Spectrogram shape: {debug['freqs_shape']}")
    print(f"  Total candidates: {debug['n_candidates']}")
    print(f"  Tracked frames: {debug['n_tracked']}")
    print(f"  Notes detected: {debug['n_notes']}")
    print()
    
    # Output tabs
    print(notes_to_tab_string(notes))
