#!/usr/bin/env python3
"""
Rhythm Notation Module for Guitar Tab Transcription

Adds proper rhythm notation to guitar tabs:
- Detect note durations (whole, half, quarter, eighth, sixteenth)
- Add timing marks above tabs (w, h, q, e, s)
- Detect rests and add rest symbols
- Handle triplets and dotted notes
- Add tempo detection and BPM marking

Author: Claude (Subagent) for guitar-tabs project
Date: 2026-02-13
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


# ============================================================================
# CONSTANTS
# ============================================================================

class NoteValue(Enum):
    """Musical note values with their beat duration ratios."""
    WHOLE = (1.0, 'w', '𝅝')           # 4 beats
    HALF = (0.5, 'h', '𝅗𝅥')            # 2 beats
    QUARTER = (0.25, 'q', '♩')        # 1 beat
    EIGHTH = (0.125, 'e', '♪')        # 0.5 beats
    SIXTEENTH = (0.0625, 's', '♬')    # 0.25 beats
    THIRTY_SECOND = (0.03125, 't', '𝅘𝅥𝅰') # 0.125 beats
    
    # Dotted notes (1.5x duration)
    DOTTED_HALF = (0.75, 'h.', '𝅗𝅥.')
    DOTTED_QUARTER = (0.375, 'q.', '♩.')
    DOTTED_EIGHTH = (0.1875, 'e.', '♪.')
    
    # Triplets (2/3 of normal duration)
    QUARTER_TRIPLET = (0.1667, '3q', '♩³')
    EIGHTH_TRIPLET = (0.0833, '3e', '♪³')
    
    def __init__(self, beat_ratio: float, ascii_symbol: str, unicode_symbol: str):
        self.beat_ratio = beat_ratio
        self.ascii_symbol = ascii_symbol
        self.unicode_symbol = unicode_symbol


class RestValue(Enum):
    """Musical rest values."""
    WHOLE = (1.0, 'W', '𝄻')
    HALF = (0.5, 'H', '𝄼')
    QUARTER = (0.25, 'Q', '𝄽')
    EIGHTH = (0.125, 'E', '𝄾')
    SIXTEENTH = (0.0625, 'S', '𝄿')
    
    def __init__(self, beat_ratio: float, ascii_symbol: str, unicode_symbol: str):
        self.beat_ratio = beat_ratio
        self.ascii_symbol = ascii_symbol
        self.unicode_symbol = unicode_symbol


# Standard note durations in beats (relative to quarter note = 1)
NOTE_DURATIONS = {
    'whole': 4.0,
    'half': 2.0,
    'dotted_quarter': 1.5,
    'quarter': 1.0,
    'dotted_eighth': 0.75,
    'eighth': 0.5,
    'eighth_triplet': 1/3,
    'sixteenth': 0.25,
    'thirty_second': 0.125,
}

# Tolerance for duration matching (percentage)
DURATION_TOLERANCE = 0.2


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RhythmNote:
    """A note with rhythm information."""
    start_time: float
    duration: float
    note_value: NoteValue
    is_dotted: bool = False
    is_triplet: bool = False
    beat_position: float = 0.0
    confidence: float = 1.0


@dataclass
class RhythmRest:
    """A rest in the rhythm."""
    start_time: float
    duration: float
    rest_value: RestValue
    beat_position: float = 0.0


@dataclass
class RhythmBar:
    """A measure/bar with rhythm elements."""
    bar_number: int
    start_time: float
    end_time: float
    elements: List[Any] = field(default_factory=list)  # RhythmNote or RhythmRest
    time_signature: Tuple[int, int] = (4, 4)


@dataclass
class TempoInfo:
    """Tempo information for a piece."""
    bpm: float
    beat_times: np.ndarray
    confidence: float
    time_signature: Tuple[int, int] = (4, 4)
    beat_strength: np.ndarray = field(default_factory=lambda: np.array([]))


# ============================================================================
# TEMPO DETECTION
# ============================================================================

def detect_tempo(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    method: str = 'auto'
) -> TempoInfo:
    """
    Detect tempo (BPM) from audio using multiple methods.
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis
        start_bpm: Starting tempo estimate
        method: 'beat_track', 'onset', 'plp', or 'auto'
    
    Returns:
        TempoInfo with BPM, beat times, and confidence
    """
    results = []
    
    # Method 1: Beat tracking (librosa's primary method)
    try:
        tempo_bt, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length, start_bpm=start_bpm
        )
        beat_times_bt = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        
        # Handle scalar tempo (older librosa) vs array (newer librosa)
        if isinstance(tempo_bt, np.ndarray):
            tempo_bt = float(tempo_bt[0]) if len(tempo_bt) > 0 else start_bpm
        else:
            tempo_bt = float(tempo_bt)
        
        # Calculate confidence from beat consistency
        if len(beat_times_bt) > 2:
            intervals = np.diff(beat_times_bt)
            expected_interval = 60.0 / tempo_bt
            interval_errors = np.abs(intervals - expected_interval) / expected_interval
            conf_bt = max(0, 1 - np.mean(interval_errors))
        else:
            conf_bt = 0.5
        
        results.append(('beat_track', tempo_bt, beat_times_bt, conf_bt))
    except Exception as e:
        print(f"  Beat track method failed: {e}")
    
    # Method 2: Onset-based tempo (good for percussive attacks like guitar picking)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo_onset = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, start_bpm=start_bpm
        )
        
        # Handle scalar vs array
        if isinstance(tempo_onset, np.ndarray):
            tempo_onset = float(tempo_onset[0]) if len(tempo_onset) > 0 else start_bpm
        else:
            tempo_onset = float(tempo_onset)
        
        # Generate beat times from tempo
        duration = len(y) / sr
        beat_interval = 60.0 / tempo_onset
        beat_times_onset = np.arange(0, duration, beat_interval)
        
        results.append(('onset', tempo_onset, beat_times_onset, 0.7))
    except Exception as e:
        print(f"  Onset tempo method failed: {e}")
    
    # Method 3: Predominant Local Pulse (PLP) for more stable tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        # Find peaks in pulse (these are the beat positions)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pulse, distance=int(sr * 0.3 / hop_length))
        
        if len(peaks) > 2:
            beat_times_plp = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            intervals = np.diff(beat_times_plp)
            tempo_plp = 60.0 / np.median(intervals)
            
            # Confidence from interval consistency
            interval_std = np.std(intervals)
            conf_plp = max(0, 1 - interval_std / np.median(intervals))
            
            results.append(('plp', tempo_plp, beat_times_plp, conf_plp))
    except Exception as e:
        print(f"  PLP tempo method failed: {e}")
    
    # Choose best result based on confidence
    if not results:
        # Fallback: assume 120 BPM
        duration = len(y) / sr
        beat_times = np.arange(0, duration, 0.5)  # 120 BPM
        return TempoInfo(bpm=120.0, beat_times=beat_times, confidence=0.0)
    
    # Sort by confidence and pick best
    results.sort(key=lambda x: x[3], reverse=True)
    best_method, best_tempo, best_beats, best_conf = results[0]
    
    print(f"  Tempo detection ({best_method}): {best_tempo:.1f} BPM (confidence: {best_conf:.2f})")
    
    # Calculate beat strength for each beat
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    beat_frames = librosa.time_to_frames(best_beats, sr=sr, hop_length=hop_length)
    beat_strength = np.array([
        onset_env[min(f, len(onset_env)-1)] for f in beat_frames
    ])
    
    return TempoInfo(
        bpm=best_tempo,
        beat_times=best_beats,
        confidence=best_conf,
        beat_strength=beat_strength
    )


def detect_time_signature(
    tempo_info: TempoInfo,
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Tuple[int, int]:
    """
    Attempt to detect time signature from beat strength patterns.
    
    Returns:
        Tuple (numerator, denominator) e.g., (4, 4) for 4/4 time
    """
    if len(tempo_info.beat_strength) < 8:
        return (4, 4)  # Default
    
    strengths = tempo_info.beat_strength
    
    # Normalize strengths
    strengths = (strengths - np.min(strengths)) / (np.max(strengths) - np.min(strengths) + 1e-10)
    
    # Try different groupings and see which has strongest downbeat pattern
    candidates = []
    
    # Try 4/4: every 4th beat should be strong
    for beats_per_measure in [3, 4, 6]:
        n_measures = len(strengths) // beats_per_measure
        if n_measures < 2:
            continue
        
        # Average strength at each position within measure
        position_strengths = np.zeros(beats_per_measure)
        for i in range(beats_per_measure):
            positions = np.arange(i, len(strengths), beats_per_measure)
            position_strengths[i] = np.mean(strengths[positions])
        
        # Score: downbeat should be strongest
        downbeat_strength = position_strengths[0]
        other_mean = np.mean(position_strengths[1:])
        score = downbeat_strength - other_mean
        
        candidates.append((beats_per_measure, score))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_beats = candidates[0][0]
        
        if best_beats == 3:
            return (3, 4)
        elif best_beats == 6:
            return (6, 8)
    
    return (4, 4)


# ============================================================================
# NOTE DURATION QUANTIZATION
# ============================================================================

def quantize_duration(
    duration_beats: float,
    allow_triplets: bool = True,
    allow_dotted: bool = True
) -> Tuple[NoteValue, float]:
    """
    Quantize a duration in beats to the nearest musical note value.
    
    Args:
        duration_beats: Duration in beats (quarter note = 1.0)
        allow_triplets: Allow triplet note values
        allow_dotted: Allow dotted note values
    
    Returns:
        Tuple of (NoteValue, error) where error is the quantization error
    """
    # Build list of candidate note values
    candidates = [
        (NoteValue.WHOLE, 4.0),
        (NoteValue.HALF, 2.0),
        (NoteValue.QUARTER, 1.0),
        (NoteValue.EIGHTH, 0.5),
        (NoteValue.SIXTEENTH, 0.25),
        (NoteValue.THIRTY_SECOND, 0.125),
    ]
    
    if allow_dotted:
        candidates.extend([
            (NoteValue.DOTTED_HALF, 3.0),
            (NoteValue.DOTTED_QUARTER, 1.5),
            (NoteValue.DOTTED_EIGHTH, 0.75),
        ])
    
    if allow_triplets:
        candidates.extend([
            (NoteValue.QUARTER_TRIPLET, 2/3),
            (NoteValue.EIGHTH_TRIPLET, 1/3),
        ])
    
    # Find closest match
    best_value = NoteValue.QUARTER
    best_error = float('inf')
    
    for note_value, beat_duration in candidates:
        error = abs(duration_beats - beat_duration) / max(beat_duration, 0.001)
        if error < best_error:
            best_error = error
            best_value = note_value
    
    return best_value, best_error


def quantize_rest_duration(duration_beats: float) -> Tuple[RestValue, float]:
    """
    Quantize a rest duration to the nearest rest value.
    
    Returns:
        Tuple of (RestValue, remaining_beats) for compound rests
    """
    candidates = [
        (RestValue.WHOLE, 4.0),
        (RestValue.HALF, 2.0),
        (RestValue.QUARTER, 1.0),
        (RestValue.EIGHTH, 0.5),
        (RestValue.SIXTEENTH, 0.25),
    ]
    
    # Find largest rest that fits
    for rest_value, beat_duration in candidates:
        if duration_beats >= beat_duration * 0.8:  # Allow 20% tolerance
            remaining = duration_beats - beat_duration
            return rest_value, max(0, remaining)
    
    return RestValue.SIXTEENTH, 0


# ============================================================================
# RHYTHM ANALYSIS
# ============================================================================

def analyze_rhythm(
    notes: List[Any],  # List of Note objects with start_time, duration
    tempo_info: TempoInfo,
    detect_rests: bool = True,
    min_rest_beats: float = 0.25,  # Minimum rest duration to show
) -> List[RhythmBar]:
    """
    Analyze notes and create rhythm notation with bars.
    
    Args:
        notes: List of Note objects
        tempo_info: Tempo information from detect_tempo
        detect_rests: Whether to detect and include rests
        min_rest_beats: Minimum rest duration to include (in beats)
    
    Returns:
        List of RhythmBar objects containing rhythm elements
    """
    if not notes:
        return []
    
    bpm = tempo_info.bpm
    beat_duration = 60.0 / bpm  # seconds per beat
    beats_per_bar = tempo_info.time_signature[0]
    bar_duration = beats_per_bar * beat_duration
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    # Find total duration
    max_time = max(n.start_time + n.duration for n in sorted_notes)
    num_bars = int(max_time / bar_duration) + 1
    
    # Create bars
    bars = []
    for bar_num in range(num_bars):
        bar_start = bar_num * bar_duration
        bar_end = bar_start + bar_duration
        
        bar = RhythmBar(
            bar_number=bar_num + 1,
            start_time=bar_start,
            end_time=bar_end,
            time_signature=tempo_info.time_signature
        )
        bars.append(bar)
    
    # Process notes into bars
    for note in sorted_notes:
        bar_idx = int(note.start_time / bar_duration)
        if bar_idx >= len(bars):
            bar_idx = len(bars) - 1
        
        # Calculate beat position within bar
        bar_start = bars[bar_idx].start_time
        beat_position = (note.start_time - bar_start) / beat_duration
        
        # Convert duration to beats
        duration_beats = note.duration / beat_duration
        
        # Quantize to nearest note value
        note_value, error = quantize_duration(duration_beats)
        
        rhythm_note = RhythmNote(
            start_time=note.start_time,
            duration=note.duration,
            note_value=note_value,
            is_dotted='.' in note_value.ascii_symbol,
            is_triplet='3' in note_value.ascii_symbol,
            beat_position=beat_position,
            confidence=getattr(note, 'confidence', 1.0)
        )
        
        bars[bar_idx].elements.append(rhythm_note)
    
    # Detect rests between notes
    if detect_rests:
        for bar in bars:
            if not bar.elements:
                continue
            
            # Sort elements by start time
            bar.elements.sort(key=lambda x: x.start_time)
            
            new_elements = []
            current_time = bar.start_time
            
            for elem in bar.elements:
                gap = elem.start_time - current_time
                gap_beats = gap / beat_duration
                
                # Add rest if gap is significant
                if gap_beats >= min_rest_beats:
                    rest_value, _ = quantize_rest_duration(gap_beats)
                    rest = RhythmRest(
                        start_time=current_time,
                        duration=gap,
                        rest_value=rest_value,
                        beat_position=(current_time - bar.start_time) / beat_duration
                    )
                    new_elements.append(rest)
                
                new_elements.append(elem)
                current_time = elem.start_time + elem.duration
            
            # Check for rest at end of bar
            gap = bar.end_time - current_time
            gap_beats = gap / beat_duration
            if gap_beats >= min_rest_beats:
                rest_value, _ = quantize_rest_duration(gap_beats)
                rest = RhythmRest(
                    start_time=current_time,
                    duration=gap,
                    rest_value=rest_value,
                    beat_position=(current_time - bar.start_time) / beat_duration
                )
                new_elements.append(rest)
            
            bar.elements = new_elements
    
    return bars


# ============================================================================
# RHYTHM FORMATTING
# ============================================================================

def format_rhythm_line(
    notes: List[Any],
    tempo_info: TempoInfo,
    positions_per_beat: int = 4,  # Resolution: 4 = sixteenth notes
    use_unicode: bool = False,
    show_rests: bool = True
) -> str:
    """
    Create a rhythm line showing note values above the tab.
    
    Args:
        notes: List of Note objects
        tempo_info: Tempo information
        positions_per_beat: Grid resolution
        use_unicode: Use Unicode music symbols
        show_rests: Include rest symbols
    
    Returns:
        String with rhythm notation line
    """
    if not notes:
        return ""
    
    bpm = tempo_info.bpm
    beat_duration = 60.0 / bpm
    
    # Sort notes
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    # Find total duration
    max_time = max(n.start_time + n.duration for n in sorted_notes)
    total_positions = int(max_time / beat_duration * positions_per_beat) + 1
    
    # Create rhythm grid
    rhythm_grid = [' '] * total_positions
    rest_grid = [False] * total_positions  # Track where rests are needed
    
    # Mark note positions
    for note in sorted_notes:
        pos = int(note.start_time / beat_duration * positions_per_beat)
        if pos < total_positions:
            # Calculate duration and get note value
            duration_beats = note.duration / beat_duration
            note_value, _ = quantize_duration(duration_beats)
            
            if use_unicode:
                symbol = note_value.unicode_symbol
            else:
                symbol = note_value.ascii_symbol
            
            # Place symbol (handle multi-character symbols)
            for i, char in enumerate(symbol):
                if pos + i < total_positions:
                    rhythm_grid[pos + i] = char
    
    # Detect and mark rests if enabled
    if show_rests:
        # Find gaps between notes
        prev_end = 0
        for note in sorted_notes:
            gap_start = prev_end
            gap_end = note.start_time
            gap_beats = (gap_end - gap_start) / beat_duration
            
            if gap_beats >= 0.25:  # At least a sixteenth note rest
                rest_value, _ = quantize_rest_duration(gap_beats)
                rest_pos = int(gap_start / beat_duration * positions_per_beat)
                
                if rest_pos < total_positions and rhythm_grid[rest_pos] == ' ':
                    if use_unicode:
                        rhythm_grid[rest_pos] = rest_value.unicode_symbol
                    else:
                        rhythm_grid[rest_pos] = rest_value.ascii_symbol.lower()
            
            prev_end = note.start_time + note.duration
    
    return ''.join(rhythm_grid)


def format_tab_with_rhythm(
    tab_notes: List[Any],
    notes: List[Any],
    tempo_info: TempoInfo,
    beats_per_line: int = 8,
    tuning: List[int] = None,
    use_unicode: bool = False,
    show_tempo: bool = True,
    show_time_sig: bool = True
) -> str:
    """
    Format guitar tab with rhythm notation line above.
    
    Args:
        tab_notes: List of TabNote objects (fret positions)
        notes: List of Note objects (for rhythm info)
        tempo_info: Tempo information
        beats_per_line: Beats per line of tab
        tuning: Guitar tuning
        use_unicode: Use Unicode music symbols
        show_tempo: Show BPM at top
        show_time_sig: Show time signature
    
    Returns:
        Formatted tab string with rhythm notation
    """
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
    
    if not tab_notes:
        return "No notes detected!"
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Generate string names
    string_names = [NOTE_NAMES[midi % 12] for midi in tuning]
    
    bpm = tempo_info.bpm
    beat_duration = 60.0 / bpm
    time_sig = tempo_info.time_signature
    
    # Create time-indexed structures
    time_resolution = beat_duration / 4  # Sixteenth note resolution
    max_time = max(n.start_time + n.duration for n in tab_notes)
    num_positions = int(max_time / time_resolution) + 1
    
    # Build note grid and rhythm grid
    note_grid = {i: ['-'] * num_positions for i in range(6)}
    rhythm_grid = [' '] * num_positions
    
    # Map notes to rhythm info
    note_rhythm_map = {}
    for note in notes:
        pos = int(note.start_time / time_resolution)
        duration_beats = note.duration / beat_duration
        note_value, _ = quantize_duration(duration_beats)
        note_rhythm_map[pos] = note_value
    
    # Fill grids
    for tab_note in tab_notes:
        pos = int(tab_note.start_time / time_resolution)
        if pos < num_positions:
            fret_str = str(tab_note.fret) if tab_note.fret < 10 else f"({tab_note.fret})"
            note_grid[tab_note.string][pos] = fret_str
            
            # Get rhythm info
            if pos in note_rhythm_map:
                note_value = note_rhythm_map[pos]
                symbol = note_value.unicode_symbol if use_unicode else note_value.ascii_symbol
                rhythm_grid[pos] = symbol[0]  # First char only for alignment
    
    # Positions per line
    positions_per_beat = int(1 / (time_resolution / beat_duration))
    positions_per_line = beats_per_line * positions_per_beat
    
    # Build output
    lines = []
    
    # Header
    if show_tempo:
        lines.append(f"♩ = {bpm:.0f} BPM")
    if show_time_sig:
        lines.append(f"Time Signature: {time_sig[0]}/{time_sig[1]}")
    if show_tempo or show_time_sig:
        lines.append("")
    
    # Legend
    lines.append("Rhythm: w=whole h=half q=quarter e=eighth s=sixteenth (. = dotted, 3 = triplet)")
    lines.append("")
    
    # Tab with rhythm
    for start in range(0, num_positions, positions_per_line):
        end = min(start + positions_per_line, num_positions)
        
        # Calculate bar numbers
        start_beat = start / positions_per_beat
        bar_num = int(start_beat / time_sig[0]) + 1
        
        # Bar marker
        lines.append(f"[Bar {bar_num}]")
        
        # Rhythm line
        rhythm_str = ''.join(rhythm_grid[start:end])
        lines.append(f"  {rhythm_str}")
        
        # Tab lines (high to low)
        for string in range(5, -1, -1):
            name = string_names[string] if string < len(string_names) else ['E', 'A', 'D', 'G', 'B', 'e'][string]
            notes_str = ''.join(note_grid[string][start:end])
            lines.append(f"{name}|{notes_str}|")
        
        lines.append("")
    
    return '\n'.join(lines)


# ============================================================================
# TRIPLET DETECTION
# ============================================================================

def detect_triplets(
    notes: List[Any],
    tempo_info: TempoInfo,
    tolerance: float = 0.15
) -> List[Tuple[int, int, int]]:
    """
    Detect triplet groupings in notes.
    
    Triplets occur when 3 notes are played in the space of 2.
    
    Returns:
        List of triplet groups as tuples of note indices
    """
    if len(notes) < 3:
        return []
    
    bpm = tempo_info.bpm
    beat_duration = 60.0 / bpm
    
    # Expected triplet spacing (1/3 of a beat for eighth-note triplets)
    triplet_spacing = beat_duration / 3
    
    triplets = []
    sorted_notes = sorted(enumerate(notes), key=lambda x: x[1].start_time)
    
    i = 0
    while i < len(sorted_notes) - 2:
        idx1, n1 = sorted_notes[i]
        idx2, n2 = sorted_notes[i + 1]
        idx3, n3 = sorted_notes[i + 2]
        
        # Check spacing
        gap1 = n2.start_time - n1.start_time
        gap2 = n3.start_time - n2.start_time
        
        # Both gaps should be approximately triplet spacing
        expected = triplet_spacing
        
        gap1_error = abs(gap1 - expected) / expected
        gap2_error = abs(gap2 - expected) / expected
        
        # Check if gaps are similar (within tolerance)
        gap_ratio = gap1 / max(gap2, 0.001)
        
        if gap1_error < tolerance and gap2_error < tolerance and 0.8 < gap_ratio < 1.2:
            triplets.append((idx1, idx2, idx3))
            i += 3  # Skip past this triplet
        else:
            i += 1
    
    return triplets


# ============================================================================
# SWING DETECTION
# ============================================================================

def detect_swing(
    notes: List[Any],
    tempo_info: TempoInfo,
    min_pairs: int = 4
) -> Tuple[bool, float]:
    """
    Detect swing rhythm in notes.
    
    Swing means eighth notes are played with a long-short pattern
    (typically 2:1 or 3:2 ratio).
    
    Returns:
        Tuple of (has_swing, swing_ratio) where ratio is long/short
    """
    if len(notes) < min_pairs * 2:
        return False, 1.0
    
    bpm = tempo_info.bpm
    beat_duration = 60.0 / bpm
    eighth_duration = beat_duration / 2
    
    # Find pairs of notes that might be swung eighths
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    ratios = []
    for i in range(len(sorted_notes) - 1):
        n1 = sorted_notes[i]
        n2 = sorted_notes[i + 1]
        
        gap = n2.start_time - n1.start_time
        
        # Is this roughly within eighth note range?
        if 0.3 * beat_duration < gap < 0.7 * beat_duration:
            # Check if next gap exists and forms a pair
            if i + 2 < len(sorted_notes):
                n3 = sorted_notes[i + 2]
                gap2 = n3.start_time - n2.start_time
                
                # Combined should be close to a beat
                combined = gap + gap2
                if 0.8 * beat_duration < combined < 1.2 * beat_duration:
                    if gap > gap2:
                        ratios.append(gap / max(gap2, 0.001))
    
    if len(ratios) < min_pairs:
        return False, 1.0
    
    avg_ratio = np.mean(ratios)
    
    # Swing is typically 1.5:1 to 2:1
    if 1.3 < avg_ratio < 2.5:
        return True, avg_ratio
    
    return False, 1.0


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_audio_rhythm(
    audio_path: str = None,
    y: np.ndarray = None,
    sr: int = 22050,
    notes: List[Any] = None
) -> Dict[str, Any]:
    """
    Complete rhythm analysis of audio.
    
    Args:
        audio_path: Path to audio file (or provide y, sr)
        y: Audio signal
        sr: Sample rate
        notes: Pre-detected notes (optional)
    
    Returns:
        Dictionary with all rhythm analysis results
    """
    # Load audio if needed
    if y is None and audio_path:
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=sr)
    
    if y is None:
        raise ValueError("Must provide either audio_path or (y, sr)")
    
    print("Analyzing rhythm...")
    
    # Detect tempo
    tempo_info = detect_tempo(y, sr)
    
    # Detect time signature
    time_sig = detect_time_signature(tempo_info, y, sr)
    tempo_info.time_signature = time_sig
    
    result = {
        'tempo': {
            'bpm': tempo_info.bpm,
            'confidence': tempo_info.confidence,
            'beat_times': tempo_info.beat_times.tolist() if isinstance(tempo_info.beat_times, np.ndarray) else tempo_info.beat_times,
        },
        'time_signature': {
            'numerator': time_sig[0],
            'denominator': time_sig[1],
        },
        'tempo_info': tempo_info,
    }
    
    # Analyze notes if provided
    if notes:
        # Detect triplets
        triplets = detect_triplets(notes, tempo_info)
        result['triplets'] = triplets
        
        # Detect swing
        has_swing, swing_ratio = detect_swing(notes, tempo_info)
        result['swing'] = {
            'detected': has_swing,
            'ratio': swing_ratio,
        }
        
        # Analyze rhythm bars
        bars = analyze_rhythm(notes, tempo_info)
        result['bars'] = bars
        
        # Create rhythm line
        rhythm_line = format_rhythm_line(notes, tempo_info)
        result['rhythm_line'] = rhythm_line
    
    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for rhythm analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rhythm Analysis for Guitar Tabs')
    parser.add_argument('audio', help='Audio file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--unicode', action='store_true', help='Use Unicode symbols')
    
    args = parser.parse_args()
    
    # Analyze
    result = analyze_audio_rhythm(audio_path=args.audio)
    
    # Output
    if args.json:
        import json
        # Convert non-serializable objects
        output = {
            'tempo': result['tempo'],
            'time_signature': result['time_signature'],
        }
        if 'swing' in result:
            output['swing'] = result['swing']
        if 'triplets' in result:
            output['triplets'] = result['triplets']
        
        output_str = json.dumps(output, indent=2)
    else:
        output_str = f"""
Rhythm Analysis Results
=======================
Tempo: {result['tempo']['bpm']:.1f} BPM (confidence: {result['tempo']['confidence']:.2f})
Time Signature: {result['time_signature']['numerator']}/{result['time_signature']['denominator']}
"""
        if 'swing' in result and result['swing']['detected']:
            output_str += f"Swing: Yes (ratio: {result['swing']['ratio']:.2f})\n"
        if 'triplets' in result and result['triplets']:
            output_str += f"Triplets detected: {len(result['triplets'])} groups\n"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Saved to: {args.output}")
    else:
        print(output_str)


if __name__ == '__main__':
    main()
