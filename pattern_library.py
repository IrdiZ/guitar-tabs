#!/usr/bin/env python3
"""
Guitar Pattern Library & Matcher

Pattern matching is MORE RELIABLE than pure pitch detection because:
- Pitch detection can miss fast notes or octave-shift them
- Common patterns have established sequences that "should" be there
- If 70%+ of detected notes match a known pattern, use the pattern

Library includes:
- Pentatonic runs (minor & major, ascending/descending)
- Blues licks (with blue notes, bends, classic turnarounds)
- Sweep arpeggios (major, minor, diminished, major7, minor7)
- Shred patterns (3-note-per-string, economy picking)
- Common licks (Hendrix, Page, Clapton, SRV, etc.)
- Rock/metal power chord progressions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Union
from enum import Enum
from collections import defaultdict

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class PatternType(Enum):
    """Categories of guitar patterns."""
    PENTATONIC = "pentatonic"
    BLUES = "blues"
    SWEEP_ARPEGGIO = "sweep_arpeggio"
    THREE_NPS = "3nps_run"
    CLASSIC_LICK = "classic_lick"
    CHROMATIC = "chromatic"
    SEQUENCE = "sequence"
    TAPPING = "tapping"
    HYBRID = "hybrid"


@dataclass
class GuitarPattern:
    """A recognized guitar pattern/lick."""
    name: str
    intervals: List[int]  # Semitone intervals from root (relative)
    pattern_type: PatternType
    direction: str = "both"  # "ascending", "descending", "both"
    canonical_bpm_range: Tuple[int, int] = (80, 200)  # Typical tempo range
    difficulty: int = 1  # 1-5 difficulty rating
    description: str = ""
    tab_hint: str = ""  # Preferred string/fret positions if known
    variations: List[List[int]] = field(default_factory=list)  # Common variations
    
    @property
    def length(self) -> int:
        return len(self.intervals)
    
    def transpose(self, semitones: int) -> List[int]:
        """Return pattern intervals shifted by semitones."""
        return [i + semitones for i in self.intervals]
    
    def as_midi_notes(self, root_midi: int) -> List[int]:
        """Convert intervals to MIDI note numbers starting from root."""
        return [root_midi + i for i in self.intervals]
    
    def reversed(self) -> 'GuitarPattern':
        """Return the pattern in reverse (descending)."""
        return GuitarPattern(
            name=f"{self.name}_descending",
            intervals=self.intervals[::-1],
            pattern_type=self.pattern_type,
            direction="descending",
            canonical_bpm_range=self.canonical_bpm_range,
            difficulty=self.difficulty,
            description=self.description + " (descending)",
            tab_hint=self.tab_hint
        )


# =============================================================================
# PENTATONIC PATTERNS
# =============================================================================

# Minor Pentatonic: Root, b3, 4, 5, b7
MINOR_PENTATONIC_INTERVALS = [0, 3, 5, 7, 10]

# Major Pentatonic: Root, 2, 3, 5, 6
MAJOR_PENTATONIC_INTERVALS = [0, 2, 4, 7, 9]

PENTATONIC_PATTERNS = [
    # One octave runs
    GuitarPattern(
        name="minor_pent_1oct_up",
        intervals=[0, 3, 5, 7, 10, 12],
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=1,
        description="Minor pentatonic ascending one octave"
    ),
    GuitarPattern(
        name="major_pent_1oct_up",
        intervals=[0, 2, 4, 7, 9, 12],
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=1,
        description="Major pentatonic ascending one octave"
    ),
    
    # Two octave runs
    GuitarPattern(
        name="minor_pent_2oct",
        intervals=[0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24],
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=2,
        description="Minor pentatonic full two octave run"
    ),
    
    # Box position patterns
    GuitarPattern(
        name="pent_box1_run",
        intervals=[0, 3, 5, 7, 10, 12, 15, 17],  # First position
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=2,
        description="Pentatonic box 1 typical run"
    ),
    
    # Common pentatonic sequences
    GuitarPattern(
        name="pent_groups_of_4",
        intervals=[0, 3, 5, 7, 3, 5, 7, 10, 5, 7, 10, 12],
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=3,
        description="Pentatonic in groups of 4 (ascending sequence)"
    ),
    GuitarPattern(
        name="pent_triplet_pattern",
        intervals=[0, 3, 5, 3, 5, 7, 5, 7, 10, 7, 10, 12],
        pattern_type=PatternType.PENTATONIC,
        direction="ascending",
        difficulty=3,
        description="Pentatonic triplet sequence"
    ),
]

# =============================================================================
# BLUES PATTERNS
# =============================================================================

# Blues Scale: Root, b3, 4, b5 (blue note), 5, b7
BLUES_SCALE_INTERVALS = [0, 3, 5, 6, 7, 10]

BLUES_PATTERNS = [
    # Classic blues licks
    GuitarPattern(
        name="blues_box_run",
        intervals=[0, 3, 5, 6, 7, 10, 12],
        pattern_type=PatternType.BLUES,
        direction="ascending",
        difficulty=2,
        description="Blues scale ascending with blue note"
    ),
    
    # BB King box lick
    GuitarPattern(
        name="bb_king_box",
        intervals=[12, 10, 12, 10, 7, 10, 7, 5, 7, 5, 3, 0],
        pattern_type=PatternType.BLUES,
        direction="descending",
        difficulty=2,
        description="BB King style box position lick"
    ),
    
    # Classic blues turnaround (I-IV-I-V)
    GuitarPattern(
        name="blues_turnaround_1",
        intervals=[0, 4, 5, 6, 7, 5, 3, 0],
        pattern_type=PatternType.BLUES,
        direction="mixed",
        difficulty=2,
        description="Classic blues turnaround lick"
    ),
    
    # Bend + release patterns (represented as semitone shifts)
    GuitarPattern(
        name="blues_bend_lick",
        intervals=[3, 5, 3, 0, 3, 5, 7],  # Bend from b3 to 4, return
        pattern_type=PatternType.BLUES,
        direction="mixed",
        difficulty=2,
        description="Blues lick with characteristic bend"
    ),
    
    # SRV-style double stop licks
    GuitarPattern(
        name="srv_double_stop",
        intervals=[0, 3, 5, 0, 3, 7, 5, 3, 0],
        pattern_type=PatternType.BLUES,
        direction="mixed",
        difficulty=3,
        description="SRV-style double stop pattern"
    ),
    
    # Minor blues run
    GuitarPattern(
        name="minor_blues_run",
        intervals=[0, 3, 5, 6, 7, 6, 5, 3, 0],
        pattern_type=PatternType.BLUES,
        direction="mixed",
        difficulty=2,
        description="Minor blues with blue note emphasis"
    ),
    
    # Hendrix chord-melody
    GuitarPattern(
        name="hendrix_r_and_b",
        intervals=[0, 4, 7, 12, 7, 4, 0],
        pattern_type=PatternType.BLUES,
        direction="mixed",
        difficulty=3,
        description="Hendrix R&B chord embellishment"
    ),
]

# =============================================================================
# SWEEP ARPEGGIO PATTERNS
# =============================================================================

# Arpeggio intervals
MAJOR_ARPEGGIO = [0, 4, 7, 12, 16, 19, 24]  # 2-octave major
MINOR_ARPEGGIO = [0, 3, 7, 12, 15, 19, 24]  # 2-octave minor
DIMINISHED_ARPEGGIO = [0, 3, 6, 12, 15, 18, 24]
MAJOR7_ARPEGGIO = [0, 4, 7, 11, 12, 16, 19, 23, 24]
MINOR7_ARPEGGIO = [0, 3, 7, 10, 12, 15, 19, 22, 24]
DOM7_ARPEGGIO = [0, 4, 7, 10, 12, 16, 19, 22, 24]

SWEEP_PATTERNS = [
    # 5-string sweeps
    GuitarPattern(
        name="sweep_major_5str",
        intervals=[0, 4, 7, 12, 16, 12, 7, 4, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="5-string major sweep up and down"
    ),
    GuitarPattern(
        name="sweep_minor_5str",
        intervals=[0, 3, 7, 12, 15, 12, 7, 3, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="5-string minor sweep up and down"
    ),
    
    # 3-string sweeps (easier, more common)
    GuitarPattern(
        name="sweep_major_3str",
        intervals=[0, 4, 7, 12, 7, 4, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=3,
        description="3-string major arpeggio"
    ),
    GuitarPattern(
        name="sweep_minor_3str",
        intervals=[0, 3, 7, 12, 7, 3, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=3,
        description="3-string minor arpeggio"
    ),
    
    # Economy picking arpeggios
    GuitarPattern(
        name="economy_major_arp",
        intervals=[0, 4, 7, 12, 16, 19, 24, 19, 16, 12, 7, 4, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="Full economy picking major arpeggio"
    ),
    
    # Diminished sweeps
    GuitarPattern(
        name="sweep_dim",
        intervals=[0, 3, 6, 9, 12, 9, 6, 3, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="Diminished arpeggio sweep (symmetric)"
    ),
    
    # 7th arpeggios
    GuitarPattern(
        name="sweep_maj7",
        intervals=[0, 4, 7, 11, 12, 11, 7, 4, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="Major 7th arpeggio"
    ),
    GuitarPattern(
        name="sweep_min7",
        intervals=[0, 3, 7, 10, 12, 10, 7, 3, 0],
        pattern_type=PatternType.SWEEP_ARPEGGIO,
        direction="both",
        difficulty=4,
        description="Minor 7th arpeggio"
    ),
]

# =============================================================================
# 3-NOTE-PER-STRING PATTERNS (SHRED)
# =============================================================================

THREE_NPS_PATTERNS = [
    # Major 3NPS
    GuitarPattern(
        name="3nps_major_asc",
        intervals=[0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19],
        pattern_type=PatternType.THREE_NPS,
        direction="ascending",
        difficulty=4,
        description="3-note-per-string major scale ascending"
    ),
    
    # Minor 3NPS
    GuitarPattern(
        name="3nps_minor_asc",
        intervals=[0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19],
        pattern_type=PatternType.THREE_NPS,
        direction="ascending",
        difficulty=4,
        description="3-note-per-string natural minor ascending"
    ),
    
    # Paul Gilbert sixes
    GuitarPattern(
        name="gilbert_sixes",
        intervals=[0, 2, 4, 2, 4, 5, 4, 5, 7, 5, 7, 9],
        pattern_type=PatternType.THREE_NPS,
        direction="ascending",
        difficulty=5,
        description="Paul Gilbert string skipping sixes"
    ),
    
    # Economy 3NPS
    GuitarPattern(
        name="economy_3nps",
        intervals=[0, 2, 4, 7, 9, 11, 12, 14, 16],
        pattern_type=PatternType.THREE_NPS,
        direction="ascending",
        difficulty=4,
        description="Economy picking 3NPS run"
    ),
]

# =============================================================================
# CHROMATIC PATTERNS
# =============================================================================

CHROMATIC_PATTERNS = [
    GuitarPattern(
        name="chromatic_run_4",
        intervals=[0, 1, 2, 3, 4],
        pattern_type=PatternType.CHROMATIC,
        direction="ascending",
        difficulty=2,
        description="4-fret chromatic run"
    ),
    GuitarPattern(
        name="chromatic_approach",
        intervals=[-1, 0],  # Half-step below to target
        pattern_type=PatternType.CHROMATIC,
        direction="ascending",
        difficulty=1,
        description="Chromatic approach from below"
    ),
    GuitarPattern(
        name="chromatic_enclosure",
        intervals=[-1, 1, 0],  # Below, above, target
        pattern_type=PatternType.CHROMATIC,
        direction="mixed",
        difficulty=2,
        description="Chromatic enclosure (bebop)"
    ),
]

# =============================================================================
# CLASSIC LICKS (Genre-Specific)
# =============================================================================

CLASSIC_LICKS = [
    # Clapton "Crossroads" lick
    GuitarPattern(
        name="crossroads_lick",
        intervals=[0, 3, 5, 7, 5, 3, 0, -2, 0, 3],
        pattern_type=PatternType.CLASSIC_LICK,
        direction="mixed",
        difficulty=3,
        description="Clapton-style pentatonic turnaround"
    ),
    
    # Page "Stairway" arpeggios
    GuitarPattern(
        name="stairway_arp",
        intervals=[0, 7, 12, 16, 12, 7, 0, 4, 7, 12, 7, 4],
        pattern_type=PatternType.CLASSIC_LICK,
        direction="mixed",
        difficulty=3,
        description="Fingerpicking arpeggio pattern"
    ),
    
    # Zakk Wylde pinch + run
    GuitarPattern(
        name="pinch_run",
        intervals=[0, 3, 5, 7, 10, 12, 15, 17],
        pattern_type=PatternType.CLASSIC_LICK,
        direction="ascending",
        difficulty=4,
        description="Pinch harmonic followed by pentatonic run"
    ),
    
    # EVH tapping lick
    GuitarPattern(
        name="evh_tapping_basic",
        intervals=[0, 7, 12, 0, 7, 12, 0, 7],  # Open, 7th fret, tap
        pattern_type=PatternType.TAPPING,
        direction="mixed",
        difficulty=4,
        description="Van Halen style tapping pattern"
    ),
    
    # Yngwie diminished run
    GuitarPattern(
        name="yngwie_dim_run",
        intervals=[0, 1, 3, 6, 7, 9, 12, 13, 15],
        pattern_type=PatternType.CLASSIC_LICK,
        direction="ascending",
        difficulty=5,
        description="Yngwie harmonic minor diminished pattern"
    ),
]

# =============================================================================
# ALL PATTERNS COMBINED
# =============================================================================

def build_pattern_library() -> List[GuitarPattern]:
    """Build complete pattern library including reversed variants."""
    all_patterns = []
    
    # Add all base patterns
    base_patterns = (
        PENTATONIC_PATTERNS +
        BLUES_PATTERNS +
        SWEEP_PATTERNS +
        THREE_NPS_PATTERNS +
        CHROMATIC_PATTERNS +
        CLASSIC_LICKS
    )
    
    for pattern in base_patterns:
        all_patterns.append(pattern)
        
        # Add reversed version for ascending/descending patterns
        if pattern.direction == "ascending":
            all_patterns.append(pattern.reversed())
    
    return all_patterns


PATTERN_LIBRARY = build_pattern_library()


# =============================================================================
# PATTERN MATCHER
# =============================================================================

@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern: GuitarPattern
    root_midi: int  # Root note that anchors the pattern
    start_idx: int  # Index in input sequence where pattern starts
    end_idx: int    # End index
    match_score: float  # 0.0-1.0 (1.0 = perfect match)
    matched_indices: List[int]  # Which input notes matched
    inferred_notes: List[int]  # MIDI notes inferred from pattern
    confidence: float  # Overall confidence
    
    @property
    def pattern_name(self) -> str:
        return self.pattern.name
    
    @property
    def root_name(self) -> str:
        return NOTE_NAMES[self.root_midi % 12]


class PatternMatcher:
    """
    Match detected notes against known guitar patterns.
    
    If 70%+ of a sequence matches a known pattern, use the pattern
    to fill in potentially missed notes or correct octave errors.
    """
    
    def __init__(
        self,
        patterns: List[GuitarPattern] = None,
        match_threshold: float = 0.70,  # 70% match required
        max_semitone_error: int = 1,    # Allow 1 semitone pitch error
        max_timing_drift: float = 0.1,  # 10% timing variance allowed
    ):
        self.patterns = patterns or PATTERN_LIBRARY
        self.match_threshold = match_threshold
        self.max_semitone_error = max_semitone_error
        self.max_timing_drift = max_timing_drift
        
        # Build index by pattern type for faster lookup
        self.patterns_by_type: Dict[PatternType, List[GuitarPattern]] = defaultdict(list)
        for p in self.patterns:
            self.patterns_by_type[p.pattern_type].append(p)
    
    def _normalize_intervals(self, midi_notes: List[int]) -> Tuple[int, List[int]]:
        """Convert MIDI notes to intervals from first note."""
        if not midi_notes:
            return 0, []
        root = midi_notes[0]
        intervals = [m - root for m in midi_notes]
        return root, intervals
    
    def _compute_match_score(
        self,
        detected_intervals: List[int],
        pattern_intervals: List[int]
    ) -> Tuple[float, List[int], List[int]]:
        """
        Compute how well detected intervals match pattern intervals.
        
        Uses dynamic programming to find best alignment, allowing
        for missing notes and small pitch errors.
        
        Returns:
            (score, matched_indices, missing_positions)
        """
        n_detected = len(detected_intervals)
        n_pattern = len(pattern_intervals)
        
        if n_detected == 0 or n_pattern == 0:
            return 0.0, [], []
        
        # Simple matching: for each pattern note, find closest detected note
        matched = []
        used_detected = set()
        
        for pi, p_int in enumerate(pattern_intervals):
            best_match = -1
            best_error = float('inf')
            
            for di, d_int in enumerate(detected_intervals):
                if di in used_detected:
                    continue
                error = abs(d_int - p_int)
                if error <= self.max_semitone_error and error < best_error:
                    best_error = error
                    best_match = di
            
            if best_match >= 0:
                matched.append(best_match)
                used_detected.add(best_match)
        
        # Score based on fraction of pattern matched
        score = len(matched) / n_pattern
        
        # Find missing pattern positions
        matched_pattern_positions = set()
        for di in matched:
            d_int = detected_intervals[di]
            for pi, p_int in enumerate(pattern_intervals):
                if abs(d_int - p_int) <= self.max_semitone_error:
                    matched_pattern_positions.add(pi)
                    break
        
        missing = [i for i in range(n_pattern) if i not in matched_pattern_positions]
        
        return score, matched, missing
    
    def _try_pattern_at_root(
        self,
        detected_midi: List[int],
        pattern: GuitarPattern,
        root_midi: int,
        start_idx: int = 0
    ) -> Optional[PatternMatch]:
        """Try matching pattern starting at specific root."""
        pattern_midi = pattern.as_midi_notes(root_midi)
        
        # Extract relevant portion of detected sequence
        window_size = min(len(pattern_midi) + 2, len(detected_midi) - start_idx)
        if window_size < 3:
            return None
        
        detected_window = detected_midi[start_idx:start_idx + window_size]
        
        # Normalize to intervals
        _, detected_intervals = self._normalize_intervals(detected_window)
        
        # Compute match score
        score, matched, missing = self._compute_match_score(
            detected_intervals,
            pattern.intervals
        )
        
        if score >= self.match_threshold:
            # Build inferred notes (fill in missing from pattern)
            inferred = []
            if missing:
                for m_idx in missing:
                    if m_idx < len(pattern.intervals):
                        inferred.append(root_midi + pattern.intervals[m_idx])
            
            return PatternMatch(
                pattern=pattern,
                root_midi=root_midi,
                start_idx=start_idx,
                end_idx=start_idx + window_size,
                match_score=score,
                matched_indices=[start_idx + m for m in matched],
                inferred_notes=inferred,
                confidence=score * (1 - len(missing) / len(pattern.intervals) * 0.3)
            )
        
        return None
    
    def find_patterns(
        self,
        midi_notes: List[int],
        pattern_types: List[PatternType] = None,
        min_pattern_len: int = 4
    ) -> List[PatternMatch]:
        """
        Find all pattern matches in a sequence of MIDI notes.
        
        Args:
            midi_notes: List of detected MIDI note numbers
            pattern_types: Filter to specific pattern types (optional)
            min_pattern_len: Minimum pattern length to consider
            
        Returns:
            List of PatternMatch results, sorted by confidence
        """
        if len(midi_notes) < min_pattern_len:
            return []
        
        # Filter patterns by type if specified
        patterns_to_try = []
        if pattern_types:
            for pt in pattern_types:
                patterns_to_try.extend(self.patterns_by_type[pt])
        else:
            patterns_to_try = self.patterns
        
        # Filter by minimum length
        patterns_to_try = [p for p in patterns_to_try if p.length >= min_pattern_len]
        
        matches = []
        
        # Sliding window over detected notes
        for start_idx in range(len(midi_notes)):
            if len(midi_notes) - start_idx < min_pattern_len:
                break
            
            # Try each pattern
            for pattern in patterns_to_try:
                # Try pattern anchored at each possible root
                # (based on first note in window as potential root)
                root_midi = midi_notes[start_idx]
                
                match = self._try_pattern_at_root(
                    midi_notes, pattern, root_midi, start_idx
                )
                
                if match:
                    matches.append(match)
        
        # Sort by confidence and remove overlapping matches
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return self._remove_overlaps(matches)
    
    def _remove_overlaps(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping pattern matches, keeping highest confidence."""
        if not matches:
            return []
        
        non_overlapping = []
        used_indices = set()
        
        for match in matches:
            match_range = set(range(match.start_idx, match.end_idx))
            if not match_range.intersection(used_indices):
                non_overlapping.append(match)
                used_indices.update(match_range)
        
        return non_overlapping
    
    def apply_pattern_corrections(
        self,
        midi_notes: List[int],
        matches: List[PatternMatch]
    ) -> Tuple[List[int], List[Tuple[int, int, str]]]:
        """
        Apply pattern-based corrections to detected notes.
        
        Returns:
            (corrected_notes, changes_made)
            where changes_made is list of (index, new_midi, reason)
        """
        corrected = list(midi_notes)
        changes = []
        
        for match in matches:
            pattern = match.pattern
            root = match.root_midi
            expected = pattern.as_midi_notes(root)
            
            # For each expected note in pattern, check if detection was off
            for i, exp_midi in enumerate(expected):
                if i + match.start_idx >= len(corrected):
                    break
                
                actual_idx = match.start_idx + i
                actual_midi = corrected[actual_idx]
                
                # Check for octave error
                if abs(actual_midi - exp_midi) == 12:
                    corrected[actual_idx] = exp_midi
                    changes.append((
                        actual_idx,
                        exp_midi,
                        f"Octave correction from pattern '{pattern.name}'"
                    ))
                
                # Check for missing note that could be inferred
                elif exp_midi in match.inferred_notes:
                    # This position should have a note but was missed
                    changes.append((
                        actual_idx,
                        exp_midi,
                        f"Inferred from pattern '{pattern.name}'"
                    ))
        
        return corrected, changes
    
    def suggest_pattern(
        self,
        midi_notes: List[int]
    ) -> Optional[PatternMatch]:
        """
        Suggest the single best pattern match for a note sequence.
        
        Returns highest-confidence match or None.
        """
        matches = self.find_patterns(midi_notes)
        return matches[0] if matches else None


# =============================================================================
# INTERVAL-BASED DETECTION
# =============================================================================

def detect_interval_pattern(intervals: List[int]) -> Optional[str]:
    """
    Detect common interval patterns in a sequence.
    
    Returns pattern name or None.
    """
    if len(intervals) < 3:
        return None
    
    # Check for common interval sequences
    patterns = {
        'ascending_scale': lambda i: all(0 <= d <= 3 for d in np.diff(i)),
        'descending_scale': lambda i: all(-3 <= d <= 0 for d in np.diff(i)),
        'chromatic_up': lambda i: all(d == 1 for d in np.diff(i)),
        'chromatic_down': lambda i: all(d == -1 for d in np.diff(i)),
        'arpeggio': lambda i: set(np.diff(i)).issubset({3, 4, 5, -3, -4, -5}),
        'pedal_tone': lambda i: sum(1 for d in np.diff(i) if d == 0) > len(i) // 2,
        'octave_jump': lambda i: 12 in np.abs(np.diff(i)),
    }
    
    for name, check in patterns.items():
        try:
            if check(intervals):
                return name
        except:
            continue
    
    return None


def analyze_sequence(midi_notes: List[int]) -> Dict:
    """
    Analyze a note sequence for pattern characteristics.
    
    Returns dict with detected characteristics.
    """
    if len(midi_notes) < 2:
        return {'length': len(midi_notes), 'patterns': []}
    
    intervals = np.diff(midi_notes)
    
    analysis = {
        'length': len(midi_notes),
        'range': max(midi_notes) - min(midi_notes),
        'direction': 'ascending' if sum(intervals) > 0 else 'descending' if sum(intervals) < 0 else 'mixed',
        'avg_interval': float(np.mean(np.abs(intervals))),
        'max_interval': int(max(np.abs(intervals))),
        'is_chromatic': all(abs(i) <= 1 for i in intervals),
        'has_octave_jump': any(abs(i) == 12 for i in intervals),
        'interval_pattern': detect_interval_pattern(midi_notes),
    }
    
    # Detect pentatonic movement
    pent_intervals = {0, 2, 3, 4, 5, 7, 9, 10, 12}
    if all(abs(i) in pent_intervals for i in intervals):
        analysis['likely_pentatonic'] = True
    
    return analysis


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def match_patterns(
    midi_notes: List[int],
    match_threshold: float = 0.70
) -> List[PatternMatch]:
    """
    Convenience function to match patterns in note sequence.
    
    Args:
        midi_notes: List of MIDI note numbers
        match_threshold: Minimum match score (default 70%)
        
    Returns:
        List of pattern matches
    """
    matcher = PatternMatcher(match_threshold=match_threshold)
    return matcher.find_patterns(midi_notes)


def correct_with_patterns(
    midi_notes: List[int],
    match_threshold: float = 0.70
) -> Tuple[List[int], List[PatternMatch]]:
    """
    Detect patterns and apply corrections.
    
    Returns:
        (corrected_notes, matched_patterns)
    """
    matcher = PatternMatcher(match_threshold=match_threshold)
    matches = matcher.find_patterns(midi_notes)
    
    if matches:
        corrected, changes = matcher.apply_pattern_corrections(midi_notes, matches)
        return corrected, matches
    
    return midi_notes, []


def get_patterns_by_type(pattern_type: PatternType) -> List[GuitarPattern]:
    """Get all patterns of a specific type."""
    return [p for p in PATTERN_LIBRARY if p.pattern_type == pattern_type]


def find_similar_patterns(intervals: List[int], threshold: float = 0.6) -> List[Tuple[GuitarPattern, float]]:
    """
    Find patterns similar to given intervals.
    
    Returns list of (pattern, similarity_score) tuples.
    """
    results = []
    
    for pattern in PATTERN_LIBRARY:
        # Use simple correlation
        min_len = min(len(intervals), len(pattern.intervals))
        if min_len < 3:
            continue
        
        truncated_input = intervals[:min_len]
        truncated_pattern = pattern.intervals[:min_len]
        
        # Count matches
        matches = sum(1 for a, b in zip(truncated_input, truncated_pattern) if abs(a - b) <= 1)
        score = matches / min_len
        
        if score >= threshold:
            results.append((pattern, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("Guitar Pattern Library")
    print("=" * 50)
    print(f"Total patterns: {len(PATTERN_LIBRARY)}")
    print()
    
    for ptype in PatternType:
        patterns = get_patterns_by_type(ptype)
        print(f"{ptype.value}: {len(patterns)} patterns")
    
    print()
    print("Testing pattern matcher...")
    
    # Test with a minor pentatonic run
    test_sequence = [64, 67, 69, 71, 74, 76]  # E minor pentatonic from E4
    print(f"\nTest sequence: {test_sequence}")
    print(f"Analysis: {analyze_sequence(test_sequence)}")
    
    matches = match_patterns(test_sequence)
    print(f"\nMatches found: {len(matches)}")
    for m in matches[:3]:
        print(f"  - {m.pattern_name}: {m.match_score:.2%} (root: {m.root_name})")
