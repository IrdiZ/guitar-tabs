#!/usr/bin/env python3
"""
Vibrato, Trill, and Tremolo Picking Detection for Guitar Tabs

Uses pYIN pitch tracking to analyze within-note pitch modulation and detect:
- Vibrato (~): Periodic pitch oscillation within a note
- Trills (tr): Rapid alternation between two notes  
- Tremolo picking: Rapid repeated notes

Author: Guitar Tab Generator
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TechniqueType(Enum):
    """Types of guitar playing techniques."""
    NONE = ""
    VIBRATO = "~"
    TRILL = "tr"
    TREMOLO = "trem"
    HAMMER_ON = "h"
    PULL_OFF = "p"
    BEND = "b"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"


@dataclass
class PitchModulation:
    """Represents pitch modulation characteristics within a note."""
    rate_hz: float = 0.0          # Modulation rate in Hz
    depth_cents: float = 0.0      # Modulation depth in cents
    regularity: float = 0.0       # How regular/periodic (0-1)
    is_vibrato: bool = False
    is_trill: bool = False
    trill_interval: int = 0       # Semitones between alternating notes


@dataclass
class TechniqueAnnotation:
    """A detected technique annotation for a note."""
    technique: TechniqueType
    start_time: float
    duration: float
    note_index: int              # Index in the note list
    confidence: float = 0.0
    # Additional parameters
    vibrato_rate: float = 0.0    # Hz
    vibrato_depth: float = 0.0   # Cents
    trill_interval: int = 0      # Semitones
    tremolo_rate: float = 0.0    # Notes per second


@dataclass
class AnnotatedNote:
    """A note with technique annotations."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    technique: TechniqueType = TechniqueType.NONE
    technique_params: Dict = field(default_factory=dict)
    
    @property
    def symbol(self) -> str:
        """Return the notation symbol for this technique."""
        return self.technique.value


class VibratoTrillDetector:
    """
    Detects vibrato, trills, and tremolo picking in guitar audio.
    
    Vibrato characteristics:
    - Rate: 4-8 Hz typical for guitar
    - Depth: 10-100 cents (100 cents = 1 semitone)
    - Regular, periodic oscillation
    
    Trill characteristics:
    - Rapid alternation between two specific pitches
    - Usually 1-2 semitones apart
    - Rate: 6-15 Hz typical
    
    Tremolo picking characteristics:
    - Rapid repeated notes at same pitch
    - Very short note durations
    - High onset density
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 128,  # High resolution for modulation tracking
        # Vibrato parameters - relaxed for better detection
        vibrato_min_rate: float = 3.5,    # Hz (slightly lower)
        vibrato_max_rate: float = 10.0,   # Hz (higher to catch fast vibrato)
        vibrato_min_depth: float = 8.0,   # cents (more sensitive)
        vibrato_max_depth: float = 200.0, # cents (allow wider)
        vibrato_min_regularity: float = 0.3,  # Lower threshold
        # Trill parameters  
        trill_min_rate: float = 6.0,      # Hz
        trill_max_rate: float = 15.0,     # Hz
        trill_min_interval: int = 1,      # semitones
        trill_max_interval: int = 3,      # semitones
        # Tremolo parameters
        tremolo_min_rate: float = 8.0,    # notes per second
        tremolo_max_gap: float = 0.08,    # max time between repeated notes
    ):
        self.sr = sr
        self.hop_length = hop_length
        
        # Vibrato thresholds
        self.vibrato_min_rate = vibrato_min_rate
        self.vibrato_max_rate = vibrato_max_rate
        self.vibrato_min_depth = vibrato_min_depth
        self.vibrato_max_depth = vibrato_max_depth
        self.vibrato_min_regularity = vibrato_min_regularity
        
        # Trill thresholds
        self.trill_min_rate = trill_min_rate
        self.trill_max_rate = trill_max_rate
        self.trill_min_interval = trill_min_interval
        self.trill_max_interval = trill_max_interval
        
        # Tremolo thresholds
        self.tremolo_min_rate = tremolo_min_rate
        self.tremolo_max_gap = tremolo_max_gap
    
    def analyze_pitch_modulation(
        self,
        y: np.ndarray,
        start_time: float,
        duration: float,
        base_midi: int
    ) -> PitchModulation:
        """
        Analyze pitch modulation within a note segment.
        
        Args:
            y: Full audio signal
            start_time: Note start time in seconds
            duration: Note duration in seconds
            base_midi: Expected MIDI note number
            
        Returns:
            PitchModulation object with analysis results
        """
        # Extract the note segment with some padding
        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + duration) * self.sr)
        
        # Need at least 0.2s for modulation analysis
        min_samples = int(0.2 * self.sr)
        if end_sample - start_sample < min_samples:
            return PitchModulation()
        
        # Add small padding to avoid edge effects
        pad_samples = int(0.02 * self.sr)
        start_sample = max(0, start_sample - pad_samples)
        end_sample = min(len(y), end_sample + pad_samples)
        
        segment = y[start_sample:end_sample]
        
        if len(segment) < min_samples:
            return PitchModulation()
        
        # Track pitch with pYIN at high resolution
        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment,
            fmin=librosa.midi_to_hz(base_midi - 12),
            fmax=librosa.midi_to_hz(base_midi + 12),
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        # Get valid (voiced) pitch values
        valid_mask = (f0 > 0) & voiced_flag
        if np.sum(valid_mask) < 10:  # Need enough samples
            return PitchModulation()
        
        valid_f0 = f0[valid_mask]
        
        # Convert to cents relative to base pitch
        base_hz = librosa.midi_to_hz(base_midi)
        cents = 1200 * np.log2(valid_f0 / base_hz + 1e-10)
        
        # Remove DC offset (center around 0)
        cents_centered = cents - np.mean(cents)
        
        # Analyze modulation characteristics
        return self._analyze_modulation_signal(cents_centered, valid_f0)
    
    def _analyze_modulation_signal(
        self,
        cents: np.ndarray,
        f0: np.ndarray
    ) -> PitchModulation:
        """
        Analyze a pitch modulation signal in cents.
        
        Returns modulation rate, depth, and regularity.
        """
        if len(cents) < 10:
            return PitchModulation()
        
        # Time resolution
        dt = self.hop_length / self.sr
        
        # Compute FFT to find dominant modulation frequency
        n = len(cents)
        
        # Window the signal to reduce spectral leakage
        window = np.hanning(n)
        cents_windowed = cents * window
        
        # FFT
        spectrum = np.abs(fft(cents_windowed))[:n//2]
        freqs = fftfreq(n, dt)[:n//2]
        
        # Find peaks in modulation spectrum
        # Focus on typical guitar vibrato/trill range (2-20 Hz)
        valid_freq_mask = (freqs >= 2.0) & (freqs <= 20.0)
        
        if not np.any(valid_freq_mask):
            return PitchModulation()
        
        valid_spectrum = spectrum.copy()
        valid_spectrum[~valid_freq_mask] = 0
        
        # Find the dominant modulation frequency
        if np.max(valid_spectrum) == 0:
            return PitchModulation()
        
        peak_idx = np.argmax(valid_spectrum)
        mod_rate = freqs[peak_idx]
        peak_magnitude = spectrum[peak_idx]
        
        # Calculate modulation depth (peak-to-peak in cents)
        depth = np.std(cents) * 2.83  # ~= peak-to-peak for sinusoidal
        
        # Calculate regularity (how periodic is the modulation)
        # Compare energy at fundamental frequency to total energy
        fundamental_energy = peak_magnitude ** 2
        total_energy = np.sum(spectrum ** 2)
        regularity = fundamental_energy / (total_energy + 1e-10)
        
        # Check for bimodal pitch distribution (trill indicator)
        is_trill = False
        trill_interval = 0
        
        # Histogram of pitch values
        hist, bin_edges = np.histogram(f0, bins=24)  # Quarter-tone resolution
        
        # Find peaks in histogram (multiple pitch centers)
        hist_peaks, _ = find_peaks(hist, height=np.max(hist) * 0.3, distance=2)
        
        if len(hist_peaks) >= 2:
            # Check if there are two clear pitch centers
            peak_bins = bin_edges[hist_peaks]
            
            # Calculate intervals between peaks
            peak_midis = librosa.hz_to_midi(peak_bins)
            
            for i in range(len(peak_midis)):
                for j in range(i + 1, len(peak_midis)):
                    interval = abs(peak_midis[j] - peak_midis[i])
                    if self.trill_min_interval <= interval <= self.trill_max_interval:
                        is_trill = True
                        trill_interval = int(round(interval))
                        break
        
        # Determine if this is vibrato
        is_vibrato = (
            self.vibrato_min_rate <= mod_rate <= self.vibrato_max_rate and
            self.vibrato_min_depth <= depth <= self.vibrato_max_depth and
            regularity >= self.vibrato_min_regularity and
            not is_trill  # Trills are distinct from vibrato
        )
        
        # Adjust trill detection based on rate
        if is_trill:
            is_trill = self.trill_min_rate <= mod_rate <= self.trill_max_rate
        
        return PitchModulation(
            rate_hz=mod_rate,
            depth_cents=depth,
            regularity=regularity,
            is_vibrato=is_vibrato,
            is_trill=is_trill,
            trill_interval=trill_interval
        )
    
    def detect_tremolo_picking(
        self,
        notes: List,  # List of Note objects
        same_pitch_tolerance: int = 1  # semitones
    ) -> List[Tuple[int, int, float]]:
        """
        Detect tremolo picking sections (rapid repeated notes).
        
        Args:
            notes: List of Note objects with midi, start_time, duration
            same_pitch_tolerance: MIDI note tolerance for "same pitch"
            
        Returns:
            List of (start_idx, end_idx, rate) tuples for tremolo sections
        """
        if len(notes) < 3:
            return []
        
        tremolo_sections = []
        
        i = 0
        while i < len(notes) - 2:
            # Look for sequences of rapid repeated notes
            section_start = i
            section_notes = [notes[i]]
            base_midi = notes[i].midi
            
            j = i + 1
            while j < len(notes):
                note = notes[j]
                prev_note = section_notes[-1]
                
                # Check if same pitch (within tolerance)
                if abs(note.midi - base_midi) > same_pitch_tolerance:
                    break
                
                # Check if rapid succession
                gap = note.start_time - (prev_note.start_time + prev_note.duration)
                if gap > self.tremolo_max_gap:
                    break
                
                section_notes.append(note)
                j += 1
            
            # Check if we found a tremolo section
            if len(section_notes) >= 3:
                # Calculate note rate
                total_duration = (section_notes[-1].start_time + section_notes[-1].duration -
                                 section_notes[0].start_time)
                if total_duration > 0:
                    rate = len(section_notes) / total_duration
                    
                    if rate >= self.tremolo_min_rate:
                        tremolo_sections.append((section_start, j - 1, rate))
            
            i = max(j, i + 1)
        
        return tremolo_sections


def detect_techniques_with_pitch_analysis(
    y: np.ndarray,
    sr: int,
    notes: List,  # List of Note objects
    verbose: bool = True
) -> List[AnnotatedNote]:
    """
    Detect vibrato, trills, and tremolo picking in notes.
    
    Args:
        y: Audio signal
        sr: Sample rate
        notes: List of Note objects
        verbose: Print detection info
        
    Returns:
        List of AnnotatedNote objects with technique annotations
    """
    detector = VibratoTrillDetector(sr=sr)
    
    annotated = []
    
    # Detect tremolo sections first (works on note patterns)
    tremolo_sections = detector.detect_tremolo_picking(notes)
    tremolo_note_indices = set()
    
    for start_idx, end_idx, rate in tremolo_sections:
        for idx in range(start_idx, end_idx + 1):
            tremolo_note_indices.add(idx)
    
    if verbose and tremolo_sections:
        print(f"  Found {len(tremolo_sections)} tremolo picking sections")
    
    # Analyze each note for vibrato/trill
    vibrato_count = 0
    trill_count = 0
    
    for i, note in enumerate(notes):
        technique = TechniqueType.NONE
        params = {}
        
        # Check if this note is part of a tremolo section
        if i in tremolo_note_indices:
            # Find the tremolo section this belongs to
            for start_idx, end_idx, rate in tremolo_sections:
                if start_idx <= i <= end_idx:
                    technique = TechniqueType.TREMOLO
                    params['tremolo_rate'] = rate
                    break
        
        # Only analyze for vibrato/trill if not tremolo
        # and note is long enough (> 0.15s for vibrato detection)
        elif note.duration >= 0.15:
            modulation = detector.analyze_pitch_modulation(
                y, note.start_time, note.duration, note.midi
            )
            
            if modulation.is_trill:
                technique = TechniqueType.TRILL
                params['trill_interval'] = modulation.trill_interval
                params['rate_hz'] = modulation.rate_hz
                trill_count += 1
                
            elif modulation.is_vibrato:
                technique = TechniqueType.VIBRATO
                params['vibrato_rate'] = modulation.rate_hz
                params['vibrato_depth'] = modulation.depth_cents
                vibrato_count += 1
        
        annotated.append(AnnotatedNote(
            midi=note.midi,
            start_time=note.start_time,
            duration=note.duration,
            confidence=note.confidence,
            technique=technique,
            technique_params=params
        ))
    
    if verbose:
        print(f"  Detected: {vibrato_count} vibrato, {trill_count} trills, "
              f"{len(tremolo_sections)} tremolo sections")
    
    return annotated


def add_technique_symbols_to_tab(
    tab_lines: List[str],
    annotated_notes: List[AnnotatedNote],
    time_resolution: float = 0.125
) -> List[str]:
    """
    Add technique symbols to tab output.
    
    Symbols:
    - ~  : Vibrato (after the fret number)
    - tr : Trill (above the note)
    - ≋  : Tremolo picking (or 'trem')
    
    Args:
        tab_lines: Current tab lines (6 strings)
        annotated_notes: Notes with technique annotations
        time_resolution: Time per tab position
        
    Returns:
        Modified tab lines with technique symbols
    """
    # This modifies the tab format - for now return a technique summary
    techniques_found = []
    
    for note in annotated_notes:
        if note.technique != TechniqueType.NONE:
            techniques_found.append(
                f"  {note.start_time:.2f}s: {note.technique.value} "
                f"({note.technique.name.lower()})"
            )
    
    return techniques_found


def format_tab_with_techniques(
    notes: List[AnnotatedNote],
    tuning: List[int] = None,
    beats_per_line: int = 16
) -> str:
    """
    Format tab output with technique annotations.
    
    Technique notation:
    - Vibrato: fret number followed by ~  (e.g., "7~")
    - Trill: "tr" above the note
    - Tremolo: indicated with "(trem)" 
    
    Args:
        notes: List of AnnotatedNote objects
        tuning: Guitar tuning as MIDI notes
        beats_per_line: Notes per line
        
    Returns:
        Formatted tab string with technique annotations
    """
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
    
    if not notes:
        return "No notes detected!"
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Generate string names
    string_names = [NOTE_NAMES[midi % 12] for midi in tuning]
    
    time_resolution = 0.125  # 1/8 note at 120 BPM
    max_time = max(n.start_time + n.duration for n in notes)
    num_positions = int(max_time / time_resolution) + 1
    
    # Create grid with extra width for technique symbols
    grid = {i: ['-'] * (num_positions * 2) for i in range(6)}
    technique_row = [' '] * (num_positions * 2)  # For "tr" above
    
    for note in notes:
        pos = int(note.start_time / time_resolution) * 2  # Double spacing
        
        # Find best string/fret for this MIDI note
        string_idx = None
        fret = None
        
        for s_idx, open_midi in enumerate(tuning):
            f = note.midi - open_midi
            if 0 <= f <= 24:
                if string_idx is None or f < fret:  # Prefer lower frets
                    string_idx = s_idx
                    fret = f
        
        if string_idx is None:
            continue
        
        # Format fret with technique
        if note.technique == TechniqueType.VIBRATO:
            fret_str = f"{fret}~"
        elif note.technique == TechniqueType.TREMOLO:
            fret_str = f"{fret}*"  # * for tremolo
        else:
            fret_str = str(fret)
        
        # Place trill marker above
        if note.technique == TechniqueType.TRILL:
            if pos < len(technique_row):
                technique_row[pos] = 't'
                if pos + 1 < len(technique_row):
                    technique_row[pos + 1] = 'r'
        
        # Place fret number
        if pos < len(grid[string_idx]):
            for c_idx, char in enumerate(fret_str):
                if pos + c_idx < len(grid[string_idx]):
                    grid[string_idx][pos + c_idx] = char
    
    # Format output
    lines = []
    positions_per_line = beats_per_line * 2
    
    for start in range(0, num_positions * 2, positions_per_line):
        end = min(start + positions_per_line, num_positions * 2)
        
        # Add technique markers line if any
        tech_line = ''.join(technique_row[start:end])
        if tech_line.strip():
            lines.append(f"  {tech_line}")
        
        # String lines
        for string in range(5, -1, -1):  # High e to low E
            name = string_names[string] if string < len(string_names) else 'X'
            notes_str = ''.join(grid[string][start:end])
            lines.append(f"{name}|{notes_str}|")
        lines.append("")
    
    # Add legend
    lines.append("Legend: ~ = vibrato, * = tremolo, tr = trill")
    
    return '\n'.join(lines)


# ============================================================================
# Integration functions
# ============================================================================

def analyze_audio_for_techniques(
    audio_path: str,
    notes: List = None,
    verbose: bool = True
) -> Tuple[List[AnnotatedNote], str]:
    """
    Full pipeline: load audio, detect techniques, format output.
    
    Args:
        audio_path: Path to audio file
        notes: Optional pre-detected notes (if None, will use simple detection)
        verbose: Print progress info
        
    Returns:
        Tuple of (annotated_notes, formatted_tab_string)
    """
    import librosa
    
    if verbose:
        print("🎸 Analyzing audio for playing techniques...")
        print(f"   File: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    if notes is None:
        # Simple note detection if not provided
        if verbose:
            print("   Detecting notes...")
        
        # Use pYIN for pitch detection
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=75, fmax=1400, sr=sr
        )
        
        # Simple onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Create simple Note objects
        @dataclass
        class SimpleNote:
            midi: int
            start_time: float
            duration: float
            confidence: float
        
        notes = []
        hop_length = 512
        
        for i, onset_time in enumerate(onset_times):
            frame = int(onset_time * sr / hop_length)
            if frame < len(f0) and f0[frame] > 0:
                midi = int(round(librosa.hz_to_midi(f0[frame])))
                
                # Estimate duration
                if i < len(onset_times) - 1:
                    duration = onset_times[i + 1] - onset_time
                else:
                    duration = librosa.get_duration(y=y, sr=sr) - onset_time
                
                notes.append(SimpleNote(
                    midi=midi,
                    start_time=onset_time,
                    duration=max(0.1, duration),
                    confidence=voiced_prob[frame] if frame < len(voiced_prob) else 0.5
                ))
    
    if verbose:
        print(f"   Analyzing {len(notes)} notes for techniques...")
    
    # Detect techniques
    annotated = detect_techniques_with_pitch_analysis(y, sr, notes, verbose=verbose)
    
    # Format tab with techniques
    tab_output = format_tab_with_techniques(annotated)
    
    return annotated, tab_output


def main():
    """Test the vibrato/trill detection on a sample file."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vibrato_trill_detection.py <audio_file>")
        print("\nThis module detects:")
        print("  - Vibrato (~): Periodic pitch oscillation within a note")
        print("  - Trills (tr): Rapid alternation between two notes")
        print("  - Tremolo picking (*): Rapid repeated notes")
        return 1
    
    audio_path = sys.argv[1]
    
    # Run analysis
    annotated_notes, tab_output = analyze_audio_for_techniques(
        audio_path, verbose=True
    )
    
    print("\n🎼 Tablature with Technique Annotations:")
    print("-" * 50)
    print(tab_output)
    
    # Summary
    vibratos = sum(1 for n in annotated_notes if n.technique == TechniqueType.VIBRATO)
    trills = sum(1 for n in annotated_notes if n.technique == TechniqueType.TRILL)
    tremolos = sum(1 for n in annotated_notes if n.technique == TechniqueType.TREMOLO)
    
    print("\n📊 Summary:")
    print(f"   Total notes: {len(annotated_notes)}")
    print(f"   Vibratos: {vibratos}")
    print(f"   Trills: {trills}")
    print(f"   Tremolo picks: {tremolos}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
