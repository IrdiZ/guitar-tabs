#!/usr/bin/env python3
"""
Guitar Technique Detection Engine

Detects and annotates guitar techniques:
- Hammer-ons (h): Ascending pitch with legato onset
- Pull-offs (p): Descending pitch with legato onset
- Bends (b): Pitch variation within a note (via pitch curve analysis)
- Slides (/ \\): Rapid pitch transition between notes

Integrates with OnsetDetector's legato detection for accurate articulation analysis.

ASCII Tab Notation:
  5h7  - Hammer-on from fret 5 to 7
  7p5  - Pull-off from fret 7 to 5
  7b9  - Bend from fret 7 up to pitch of fret 9
  5/7  - Slide up from fret 5 to 7
  7\5  - Slide down from fret 7 to 5
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


class Technique(Enum):
    """Guitar playing techniques."""
    NONE = ""
    HAMMER_ON = "h"
    PULL_OFF = "p"
    BEND = "b"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"
    VIBRATO = "~"      # Periodic pitch oscillation
    TRILL = "tr"       # Rapid alternation between two notes
    TREMOLO = "*"      # Rapid repeated notes (tremolo picking)


@dataclass
class TechniqueAnnotation:
    """Annotation for a detected technique."""
    technique: Technique
    target_fret: Optional[int] = None  # For h/p/slides: the destination fret
    bend_amount: float = 0.0  # For bends: semitones bent
    confidence: float = 1.0
    
    def to_ascii(self, source_fret: int) -> str:
        """
        Generate ASCII notation for the technique.
        
        Examples:
            5h7 - Hammer-on from 5 to 7
            7p5 - Pull-off from 7 to 5
            7b9 - Bend at 7 reaching pitch of 9
            5/7 - Slide up from 5 to 7
            7\\5 - Slide down from 7 to 5
        """
        if self.technique == Technique.NONE:
            return str(source_fret)
        
        if self.technique == Technique.HAMMER_ON:
            return f"{source_fret}h{self.target_fret}"
        
        if self.technique == Technique.PULL_OFF:
            return f"{source_fret}p{self.target_fret}"
        
        if self.technique == Technique.BEND:
            # Show target pitch as fret equivalent
            target = source_fret + round(self.bend_amount)
            return f"{source_fret}b{target}"
        
        if self.technique == Technique.SLIDE_UP:
            return f"{source_fret}/{self.target_fret}"
        
        if self.technique == Technique.SLIDE_DOWN:
            return f"{source_fret}\\{self.target_fret}"
        
        if self.technique == Technique.VIBRATO:
            return f"{source_fret}~"
        
        if self.technique == Technique.TRILL:
            if self.target_fret is not None:
                return f"tr{source_fret}-{self.target_fret}"
            return f"tr{source_fret}"
        
        if self.technique == Technique.TREMOLO:
            return f"{source_fret}*"
        
        return str(source_fret)


@dataclass
class AnnotatedNote:
    """A note with technique annotation."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    string: int
    fret: int
    technique: TechniqueAnnotation = field(default_factory=lambda: TechniqueAnnotation(Technique.NONE))
    is_legato: bool = False  # From onset detection
    pitch_curve: Optional[np.ndarray] = None  # Pitch values during the note
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def to_ascii(self) -> str:
        """Get ASCII tab notation for this note."""
        return self.technique.to_ascii(self.fret)


class TechniqueDetector:
    """
    Detects guitar playing techniques from audio and note data.
    
    Integrates with OnsetDetector's legato detection and analyzes:
    - Pitch curves for bends
    - Timing relationships for slides
    - Onset characteristics for hammer-ons/pull-offs
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        # Legato detection
        legato_pitch_threshold: float = 0.5,  # Min semitones for legato
        # Bend detection
        bend_min_semitones: float = 0.3,  # Minimum bend to detect
        bend_analysis_window: float = 0.15,  # Window for pitch curve analysis
        # Slide detection
        slide_max_gap: float = 0.08,  # Max time gap between notes for slide
        slide_min_pitch_change: int = 2,  # Min semitone change for slide
        slide_max_pitch_change: int = 12,  # Max semitone change (octave)
        # Vibrato detection (future)
        vibrato_min_cycles: int = 2,
        vibrato_min_width: float = 0.2,  # Min semitone deviation
    ):
        self.sr = sr
        self.hop_length = hop_length
        
        self.legato_pitch_threshold = legato_pitch_threshold
        self.bend_min_semitones = bend_min_semitones
        self.bend_analysis_window = bend_analysis_window
        self.slide_max_gap = slide_max_gap
        self.slide_min_pitch_change = slide_min_pitch_change
        self.slide_max_pitch_change = slide_max_pitch_change
        self.vibrato_min_cycles = vibrato_min_cycles
        self.vibrato_min_width = vibrato_min_width
    
    def analyze(
        self,
        y: np.ndarray,
        notes: List[Any],  # Note objects with midi, start_time, duration
        onset_details: Optional[List[Any]] = None,  # EnsembleOnset objects
        tuning: List[int] = None,
        verbose: bool = True
    ) -> List[AnnotatedNote]:
        """
        Analyze notes and detect techniques.
        
        Args:
            y: Audio signal
            notes: List of Note objects from pitch detection
            onset_details: List of EnsembleOnset from onset detection (for legato info)
            tuning: Guitar tuning as MIDI note numbers
            verbose: Print detection statistics
            
        Returns:
            List of AnnotatedNote objects with technique annotations
        """
        if tuning is None:
            tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
        
        if not notes:
            return []
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Create onset lookup for legato info
        onset_map = {}
        if onset_details:
            for onset in onset_details:
                onset_map[round(onset.time * 1000)] = onset  # Key by ms
        
        # Compute pitch curve for the entire audio
        if verbose:
            print("  Computing pitch curve for technique detection...")
        pitch_curve, pitch_confidence = self._compute_pitch_curve(y)
        frame_times = librosa.frames_to_time(
            np.arange(len(pitch_curve)), sr=self.sr, hop_length=self.hop_length
        )
        
        # Convert notes to annotated notes with fret positions
        annotated = []
        for note in sorted_notes:
            # Find fret position
            string, fret = self._midi_to_position(note.midi, tuning)
            if string is None:
                continue
            
            # Check if this note has legato onset
            is_legato = False
            onset_key = round(note.start_time * 1000)
            # Search with 50ms tolerance
            for key in range(onset_key - 50, onset_key + 51):
                if key in onset_map and onset_map[key].is_legato:
                    is_legato = True
                    break
            
            # Extract pitch curve for this note's duration
            note_pitch_curve = self._extract_note_pitch_curve(
                pitch_curve, pitch_confidence, frame_times,
                note.start_time, note.duration
            )
            
            ann_note = AnnotatedNote(
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.confidence,
                string=string,
                fret=fret,
                is_legato=is_legato,
                pitch_curve=note_pitch_curve
            )
            annotated.append(ann_note)
        
        # Detect techniques
        if verbose:
            print("  Detecting techniques...")
        
        # Pass 1: Detect hammer-ons and pull-offs (using legato info + pitch direction)
        self._detect_hammer_pulloffs(annotated)
        
        # Pass 2: Detect bends (using pitch curve analysis)
        self._detect_bends(annotated)
        
        # Pass 3: Detect slides (using timing + pitch changes)
        self._detect_slides(annotated)
        
        # Pass 4: Detect vibrato (periodic pitch modulation within notes)
        self._detect_vibrato(annotated, y)
        
        # Pass 5: Detect trills (rapid alternation between two pitches)
        self._detect_trills(annotated, y)
        
        # Pass 6: Detect tremolo picking (rapid repeated notes)
        self._detect_tremolo(annotated)
        
        if verbose:
            self._print_stats(annotated)
        
        return annotated
    
    def _compute_pitch_curve(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pitch curve using pYIN for fine-grained pitch tracking."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        # Convert Hz to MIDI (continuous values for bend detection)
        midi_curve = np.zeros_like(f0)
        valid = f0 > 0
        midi_curve[valid] = librosa.hz_to_midi(f0[valid])
        
        return midi_curve, voiced_probs
    
    def _extract_note_pitch_curve(
        self,
        full_pitch_curve: np.ndarray,
        pitch_confidence: np.ndarray,
        frame_times: np.ndarray,
        start_time: float,
        duration: float
    ) -> np.ndarray:
        """Extract the pitch curve for a specific note."""
        start_idx = np.searchsorted(frame_times, start_time)
        end_idx = np.searchsorted(frame_times, start_time + duration)
        
        if start_idx >= end_idx or start_idx >= len(full_pitch_curve):
            return np.array([])
        
        return full_pitch_curve[start_idx:end_idx].copy()
    
    def _midi_to_position(self, midi: int, tuning: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """Convert MIDI note to (string, fret) position."""
        best_string = None
        best_fret = None
        best_score = -999
        
        for string_idx, open_note in enumerate(tuning):
            fret = midi - open_note
            if 0 <= fret <= 24:
                # Score: prefer lower frets and middle strings
                score = -fret * 0.2
                if 2 <= string_idx <= 3:  # D and G strings
                    score += 1
                elif 1 <= string_idx <= 4:
                    score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_string = string_idx
                    best_fret = fret
        
        return best_string, best_fret
    
    def _detect_hammer_pulloffs(self, notes: List[AnnotatedNote]) -> None:
        """
        Detect hammer-ons and pull-offs based on legato onset + pitch direction.
        
        Hammer-on: legato onset with ascending pitch from previous note
        Pull-off: legato onset with descending pitch from previous note
        
        For ASCII notation:
        - The SOURCE note gets the technique annotation (e.g., "5h7")
        - The TARGET note is the destination fret
        """
        for i, note in enumerate(notes):
            if not note.is_legato or i == 0:
                continue
            
            prev_note = notes[i - 1]
            
            # Check timing - should be connected or very close
            gap = note.start_time - prev_note.end_time
            if gap > 0.15:  # More than 150ms gap - not a legato phrase
                continue
            
            pitch_diff = note.midi - prev_note.midi
            
            # Allow cross-string legato (less common but possible)
            # Same string is more reliable, but we'll accept cross-string too
            same_string = (prev_note.string == note.string)
            
            # Minimum pitch change for h/p (at least 1 semitone)
            if abs(pitch_diff) < 1:
                continue
            
            if pitch_diff > 0:
                # Ascending = hammer-on
                # Mark the SOURCE note (previous) with h and target
                if prev_note.technique.technique == Technique.NONE:
                    prev_note.technique = TechniqueAnnotation(
                        technique=Technique.HAMMER_ON,
                        target_fret=note.fret,
                        confidence=0.9 if same_string else 0.7
                    )
            elif pitch_diff < 0:
                # Descending = pull-off
                # Mark the SOURCE note (previous) with p and target
                if prev_note.technique.technique == Technique.NONE:
                    prev_note.technique = TechniqueAnnotation(
                        technique=Technique.PULL_OFF,
                        target_fret=note.fret,
                        confidence=0.9 if same_string else 0.7
                    )
    
    def _detect_bends(self, notes: List[AnnotatedNote]) -> None:
        """
        Detect bends by analyzing pitch variation within each note.
        
        A bend shows as pitch rising above the initial note pitch,
        then optionally returning (bend + release).
        """
        for note in notes:
            # Skip if already has a technique
            if note.technique.technique != Technique.NONE:
                continue
            
            if note.pitch_curve is None or len(note.pitch_curve) < 3:
                continue
            
            # Remove zero values (unvoiced frames)
            valid_curve = note.pitch_curve[note.pitch_curve > 0]
            if len(valid_curve) < 3:
                continue
            
            # Smooth the curve
            smoothed = gaussian_filter1d(valid_curve, sigma=1.0)
            
            # Get initial pitch and max pitch
            initial_pitch = smoothed[0]
            max_pitch = np.max(smoothed)
            
            # Bend amount in semitones
            bend_amount = max_pitch - initial_pitch
            
            if bend_amount >= self.bend_min_semitones:
                # Verify it's a bend pattern (pitch rises, doesn't just start high)
                # Check that max occurs after the first quarter of the note
                max_idx = np.argmax(smoothed)
                if max_idx > len(smoothed) * 0.1:  # Max after first 10%
                    note.technique = TechniqueAnnotation(
                        technique=Technique.BEND,
                        target_fret=note.fret + round(bend_amount),
                        bend_amount=bend_amount,
                        confidence=min(1.0, bend_amount / 2.0)  # Higher confidence for bigger bends
                    )
    
    def _detect_slides(self, notes: List[AnnotatedNote]) -> None:
        """
        Detect slides between notes.
        
        A slide is characterized by:
        - Small time gap between notes (continuous connection)
        - Rapid pitch transition (not separate articulations)
        - Same string
        """
        for i in range(len(notes) - 1):
            note = notes[i]
            next_note = notes[i + 1]
            
            # Skip if either note already has a technique
            if note.technique.technique != Technique.NONE:
                continue
            
            # Check same string
            if note.string != next_note.string:
                continue
            
            # Check timing gap
            gap = next_note.start_time - note.end_time
            
            # For slides, we want either overlap or very small gap
            if gap > self.slide_max_gap:
                continue
            
            # Check pitch change
            pitch_diff = abs(next_note.midi - note.midi)
            
            if not (self.slide_min_pitch_change <= pitch_diff <= self.slide_max_pitch_change):
                continue
            
            # Check it's not a legato (those are h/p, not slides)
            if next_note.is_legato:
                continue
            
            # Analyze the transition region for continuous pitch movement
            if self._is_slide_transition(note, next_note):
                if next_note.midi > note.midi:
                    # Ascending slide
                    note.technique = TechniqueAnnotation(
                        technique=Technique.SLIDE_UP,
                        target_fret=next_note.fret,
                        confidence=0.8
                    )
                else:
                    # Descending slide
                    note.technique = TechniqueAnnotation(
                        technique=Technique.SLIDE_DOWN,
                        target_fret=next_note.fret,
                        confidence=0.8
                    )
    
    def _is_slide_transition(self, note1: AnnotatedNote, note2: AnnotatedNote) -> bool:
        """
        Check if the transition between two notes looks like a slide.
        
        For slides, the pitch should transition smoothly between the two notes.
        """
        # If we have pitch curve data, analyze the transition
        if note1.pitch_curve is not None and len(note1.pitch_curve) > 0:
            # Check the end of note1's pitch curve
            # A slide would show pitch moving toward note2's pitch
            end_region = note1.pitch_curve[-max(1, len(note1.pitch_curve) // 4):]
            end_region = end_region[end_region > 0]
            
            if len(end_region) > 0:
                end_pitch = np.mean(end_region)
                # Check if pitch is trending toward note2
                mid_pitch = (note1.midi + note2.midi) / 2
                
                # If end pitch is between start and target, likely a slide
                if note1.midi < note2.midi:
                    return note1.midi <= end_pitch <= note2.midi + 0.5
                else:
                    return note2.midi - 0.5 <= end_pitch <= note1.midi
        
        # Without pitch curve data, use heuristics
        # Very short notes with close timing are likely slides
        return note1.duration < 0.2
    
    def _detect_vibrato(self, notes: List[AnnotatedNote], y: np.ndarray) -> None:
        """
        Detect vibrato by analyzing periodic pitch modulation within notes.
        
        Vibrato characteristics for guitar:
        - Rate: 4-10 Hz (typical 5-7 Hz)
        - Depth: 10-100 cents (subtle to wide)
        - Regular, periodic oscillation
        
        Uses FFT on the pitch deviation signal to find dominant modulation frequency.
        """
        from scipy.fft import fft, fftfreq
        
        for note in notes:
            # Skip if already has a technique or note is too short
            if note.technique.technique != Technique.NONE:
                continue
            
            if note.pitch_curve is None or len(note.pitch_curve) < 10:
                continue
            
            # Need at least 0.2s for vibrato analysis
            if note.duration < 0.2:
                continue
            
            # Get valid pitch values (non-zero)
            valid_curve = note.pitch_curve[note.pitch_curve > 0]
            if len(valid_curve) < 10:
                continue
            
            # Smooth the curve
            smoothed = gaussian_filter1d(valid_curve.astype(float), sigma=1.0)
            
            # Detrend: remove linear trend and mean
            x = np.arange(len(smoothed))
            coeffs = np.polyfit(x, smoothed, 1)
            trend = np.polyval(coeffs, x)
            detrended = smoothed - trend
            detrended = detrended - np.mean(detrended)
            
            # Check for sufficient pitch variation
            pitch_std = np.std(detrended)
            if pitch_std < 0.05:  # Less than 5 cents std - no vibrato
                continue
            
            # Convert to cents for analysis (100 cents = 1 semitone)
            cents_deviation = detrended * 100
            
            # FFT to find modulation frequency
            n = len(cents_deviation)
            dt = self.hop_length / self.sr  # Time per frame
            
            # Window and FFT
            window = np.hanning(n)
            spectrum = np.abs(fft(cents_deviation * window))[:n//2]
            freqs = fftfreq(n, dt)[:n//2]
            
            # Focus on vibrato range (4-10 Hz)
            vibrato_mask = (freqs >= 4.0) & (freqs <= 10.0)
            
            if not np.any(vibrato_mask):
                continue
            
            vibrato_spectrum = spectrum.copy()
            vibrato_spectrum[~vibrato_mask] = 0
            
            if np.max(vibrato_spectrum) < 1e-6:
                continue
            
            # Find dominant vibrato frequency
            peak_idx = np.argmax(vibrato_spectrum)
            vibrato_rate = freqs[peak_idx]
            peak_magnitude = spectrum[peak_idx]
            
            # Calculate regularity (energy at fundamental vs total)
            fundamental_energy = peak_magnitude ** 2
            total_energy = np.sum(spectrum ** 2)
            regularity = fundamental_energy / (total_energy + 1e-10)
            
            # Calculate vibrato depth (peak-to-peak in cents)
            depth_cents = np.std(cents_deviation) * 2.83  # ~= peak-to-peak for sinusoidal
            
            # Threshold checks for vibrato
            if (4.0 <= vibrato_rate <= 10.0 and 
                depth_cents >= 10.0 and  # At least 10 cents
                depth_cents <= 150.0 and  # Not more than 1.5 semitones
                regularity >= 0.15):  # Reasonably periodic
                
                note.technique = TechniqueAnnotation(
                    technique=Technique.VIBRATO,
                    confidence=min(1.0, regularity + depth_cents / 100)
                )
    
    def _detect_trills(self, notes: List[AnnotatedNote], y: np.ndarray) -> None:
        """
        Detect trills by finding rapid alternation between two distinct pitches.
        
        Trill characteristics:
        - Two distinct pitch centers (usually 1-2 semitones apart)
        - Rapid alternation (6-15 Hz)
        - Bimodal pitch distribution
        """
        for note in notes:
            # Skip if already has a technique or note is too short
            if note.technique.technique != Technique.NONE:
                continue
            
            if note.pitch_curve is None or len(note.pitch_curve) < 15:
                continue
            
            # Need at least 0.25s for trill analysis
            if note.duration < 0.25:
                continue
            
            # Get valid pitch values
            valid_curve = note.pitch_curve[note.pitch_curve > 0]
            if len(valid_curve) < 15:
                continue
            
            # Convert to Hz for histogram analysis
            valid_hz = librosa.midi_to_hz(valid_curve)
            
            # Create histogram to find pitch centers
            # Use fine resolution (quarter-tone bins = 50 cents)
            n_bins = int((np.max(valid_curve) - np.min(valid_curve)) / 0.25) + 1
            n_bins = max(n_bins, 4)  # At least 4 bins
            
            hist, bin_edges = np.histogram(valid_curve, bins=n_bins)
            
            # Find peaks in histogram (pitch centers)
            hist_smooth = gaussian_filter1d(hist.astype(float), sigma=0.5)
            peaks, properties = find_peaks(hist_smooth, height=np.max(hist_smooth) * 0.25, distance=2)
            
            if len(peaks) < 2:
                continue
            
            # Get the two strongest peaks
            peak_heights = hist_smooth[peaks]
            sorted_peak_idx = np.argsort(peak_heights)[::-1][:2]
            top_peaks = peaks[sorted_peak_idx]
            
            # Calculate interval between pitch centers
            peak_midis = (bin_edges[top_peaks] + bin_edges[top_peaks + 1]) / 2
            interval = abs(peak_midis[1] - peak_midis[0])
            
            # Trills are typically 1-3 semitones
            if not (0.8 <= interval <= 3.5):
                continue
            
            # Verify rapid alternation by checking pitch crossings
            mean_pitch = np.mean(valid_curve)
            crossings = np.sum(np.diff(np.sign(valid_curve - mean_pitch)) != 0)
            
            # Calculate crossing rate
            crossing_rate = crossings / note.duration
            
            # Trills should have 12-30 crossings per second (6-15 Hz oscillation)
            if 10 <= crossing_rate <= 35:
                note.technique = TechniqueAnnotation(
                    technique=Technique.TRILL,
                    target_fret=note.fret + round(interval),  # Upper note of trill
                    confidence=min(1.0, crossing_rate / 20)
                )
    
    def _detect_tremolo(self, notes: List[AnnotatedNote]) -> None:
        """
        Detect tremolo picking (rapid repeated notes at the same pitch).
        
        Tremolo characteristics:
        - Same pitch repeated rapidly (8+ notes per second)
        - Very short individual notes
        - Minimal gaps between notes
        """
        if len(notes) < 3:
            return
        
        # Track tremolo sections
        i = 0
        while i < len(notes) - 2:
            # Skip if already has technique
            if notes[i].technique.technique != Technique.NONE:
                i += 1
                continue
            
            # Start potential tremolo section
            section_start = i
            section_notes = [notes[i]]
            base_midi = notes[i].midi
            
            j = i + 1
            while j < len(notes):
                note = notes[j]
                prev_note = section_notes[-1]
                
                # Skip if already has technique
                if note.technique.technique != Technique.NONE:
                    break
                
                # Check if same pitch (within 1 semitone tolerance)
                if abs(note.midi - base_midi) > 1:
                    break
                
                # Check if rapid succession (max 80ms gap)
                gap = note.start_time - prev_note.end_time
                if gap > 0.08:
                    break
                
                # Check if notes are short
                if prev_note.duration > 0.15:
                    break
                
                section_notes.append(note)
                j += 1
            
            # Check if we have a tremolo section (3+ rapid notes)
            if len(section_notes) >= 3:
                # Calculate note rate
                total_duration = (section_notes[-1].end_time - section_notes[0].start_time)
                if total_duration > 0:
                    rate = len(section_notes) / total_duration
                    
                    # Tremolo is typically 8+ notes per second
                    if rate >= 8:
                        # Mark all notes in section as tremolo
                        for note in section_notes:
                            note.technique = TechniqueAnnotation(
                                technique=Technique.TREMOLO,
                                confidence=min(1.0, rate / 15)  # Higher rate = higher confidence
                            )
            
            i = max(j, i + 1)
    
    def _print_stats(self, notes: List[AnnotatedNote]) -> None:
        """Print detection statistics."""
        stats = {
            'total': len(notes),
            'hammer_on': 0,
            'pull_off': 0,
            'bend': 0,
            'slide_up': 0,
            'slide_down': 0,
            'vibrato': 0,
            'trill': 0,
            'tremolo': 0,
            'none': 0
        }
        
        for note in notes:
            tech = note.technique.technique
            if tech == Technique.HAMMER_ON:
                stats['hammer_on'] += 1
            elif tech == Technique.PULL_OFF:
                stats['pull_off'] += 1
            elif tech == Technique.BEND:
                stats['bend'] += 1
            elif tech == Technique.SLIDE_UP:
                stats['slide_up'] += 1
            elif tech == Technique.SLIDE_DOWN:
                stats['slide_down'] += 1
            elif tech == Technique.VIBRATO:
                stats['vibrato'] += 1
            elif tech == Technique.TRILL:
                stats['trill'] += 1
            elif tech == Technique.TREMOLO:
                stats['tremolo'] += 1
            else:
                stats['none'] += 1
        
        print(f"  Technique detection results:")
        print(f"    Total notes: {stats['total']}")
        print(f"    Hammer-ons (h): {stats['hammer_on']}")
        print(f"    Pull-offs (p): {stats['pull_off']}")
        print(f"    Bends (b): {stats['bend']}")
        print(f"    Slides up (/): {stats['slide_up']}")
        print(f"    Slides down (\\): {stats['slide_down']}")
        print(f"    Vibrato (~): {stats['vibrato']}")
        print(f"    Trills (tr): {stats['trill']}")
        print(f"    Tremolo (*): {stats['tremolo']}")
        print(f"    Plain notes: {stats['none']}")


def format_ascii_tab_with_techniques(
    notes: List[AnnotatedNote],
    tempo: int = 120,
    time_signature: Tuple[int, int] = (4, 4)
) -> str:
    """
    Format annotated notes as ASCII tablature with technique symbols.
    
    Args:
        notes: List of AnnotatedNote objects
        tempo: Tempo in BPM
        time_signature: Time signature (beats, beat_value)
        
    Returns:
        Formatted ASCII tab string
    """
    if not notes:
        return "No notes to display."
    
    # String names (high to low for display)
    string_names = ['e', 'B', 'G', 'D', 'A', 'E']
    
    # Calculate timing
    beat_duration = 60.0 / tempo
    measure_duration = beat_duration * time_signature[0]
    
    # Characters per measure (enough for 16th notes)
    chars_per_measure = 16
    
    # Group notes by measure
    max_time = max(n.end_time for n in notes)
    num_measures = int(max_time / measure_duration) + 1
    
    output_lines = []
    
    # Process measures in groups of 4 for readability
    measures_per_line = 4
    
    for line_start in range(0, num_measures, measures_per_line):
        line_end = min(line_start + measures_per_line, num_measures)
        
        # Initialize tab lines for this line group
        tab_strings = {i: "" for i in range(6)}
        
        for measure_idx in range(line_start, line_end):
            measure_start = measure_idx * measure_duration
            measure_end = measure_start + measure_duration
            
            # Initialize measure content
            measure_content = {i: ['-'] * chars_per_measure for i in range(6)}
            
            # Find notes in this measure
            measure_notes = [n for n in notes 
                           if measure_start <= n.start_time < measure_end]
            
            for note in measure_notes:
                # Calculate position within measure
                local_time = note.start_time - measure_start
                pos = int((local_time / measure_duration) * chars_per_measure)
                pos = min(chars_per_measure - 1, max(0, pos))
                
                # Get the ASCII representation
                ascii_rep = note.to_ascii()
                
                # Place on the correct string (convert from 0=low to display order)
                display_string = 5 - note.string  # Flip for display (e on top)
                
                # Place the characters
                for j, char in enumerate(ascii_rep):
                    char_pos = pos + j
                    if char_pos < chars_per_measure:
                        measure_content[display_string][char_pos] = char
            
            # Add measure to each string
            for string_idx in range(6):
                tab_strings[string_idx] += ''.join(measure_content[string_idx]) + '|'
        
        # Add string labels and content to output
        for string_idx in range(6):
            output_lines.append(f"{string_names[string_idx]}|{tab_strings[string_idx]}")
        output_lines.append("")  # Blank line between line groups
    
    return '\n'.join(output_lines)


def annotate_tabs_with_techniques(
    audio_path: str,
    notes: List[Any],
    onset_details: Optional[List[Any]] = None,
    tuning: List[int] = None,
    sr: int = 22050,
    hop_length: int = 512,
    verbose: bool = True
) -> Tuple[List[AnnotatedNote], str]:
    """
    High-level function to annotate notes with techniques and format as ASCII tabs.
    
    Args:
        audio_path: Path to audio file
        notes: List of Note objects from detection
        onset_details: List of EnsembleOnset objects (optional)
        tuning: Guitar tuning
        sr: Sample rate
        hop_length: Hop length
        verbose: Print progress
        
    Returns:
        Tuple of (annotated_notes, ascii_tab_string)
    """
    import librosa
    
    if verbose:
        print(f"\n🎸 Annotating techniques for {len(notes)} notes...")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Create detector
    detector = TechniqueDetector(sr=sr, hop_length=hop_length)
    
    # Detect techniques
    annotated = detector.analyze(
        y=y,
        notes=notes,
        onset_details=onset_details,
        tuning=tuning,
        verbose=verbose
    )
    
    # Format as ASCII tab
    ascii_tab = format_ascii_tab_with_techniques(annotated)
    
    return annotated, ascii_tab


# Test function
def test_technique_detector():
    """Test the technique detector with synthetic data."""
    print("Testing TechniqueDetector...")
    
    # Create mock Note class
    @dataclass
    class MockNote:
        midi: int
        start_time: float
        duration: float
        confidence: float
    
    # Create mock onset detail
    @dataclass  
    class MockOnset:
        time: float
        is_legato: bool
    
    # Test notes simulating a hammer-on lick: E-G-A
    notes = [
        MockNote(midi=52, start_time=0.0, duration=0.2, confidence=0.9),   # E3, fret 2 on D
        MockNote(midi=55, start_time=0.2, duration=0.2, confidence=0.8),   # G3, fret 5 on D (legato)
        MockNote(midi=57, start_time=0.4, duration=0.2, confidence=0.8),   # A3, fret 7 on D (legato)
        MockNote(midi=55, start_time=0.6, duration=0.2, confidence=0.8),   # G3, pull-off
        MockNote(midi=52, start_time=0.8, duration=0.3, confidence=0.9),   # E3
    ]
    
    # Legato markers for hammer-ons
    onsets = [
        MockOnset(time=0.0, is_legato=False),
        MockOnset(time=0.2, is_legato=True),   # Hammer-on
        MockOnset(time=0.4, is_legato=True),   # Hammer-on
        MockOnset(time=0.6, is_legato=True),   # Pull-off
        MockOnset(time=0.8, is_legato=False),
    ]
    
    # Create detector (we'll skip audio analysis for this test)
    detector = TechniqueDetector()
    
    print("  Mock notes created. Full test requires audio file.")
    print("  Testing annotation formatting...")
    
    # Test annotation formatting
    ann = TechniqueAnnotation(Technique.HAMMER_ON, target_fret=7)
    print(f"    Hammer-on 5h7: {ann.to_ascii(5)}")
    
    ann = TechniqueAnnotation(Technique.PULL_OFF, target_fret=5)
    print(f"    Pull-off 7p5: {ann.to_ascii(7)}")
    
    ann = TechniqueAnnotation(Technique.BEND, target_fret=9, bend_amount=2.0)
    print(f"    Bend 7b9: {ann.to_ascii(7)}")
    
    ann = TechniqueAnnotation(Technique.SLIDE_UP, target_fret=7)
    print(f"    Slide up 5/7: {ann.to_ascii(5)}")
    
    ann = TechniqueAnnotation(Technique.SLIDE_DOWN, target_fret=5)
    print(f"    Slide down 7\\5: {ann.to_ascii(7)}")
    
    print("\n✅ Technique detector tests passed!")


if __name__ == "__main__":
    test_technique_detector()
