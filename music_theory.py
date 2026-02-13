"""
Music Theory Module for Guitar Tab Post-Processing

Provides:
- Key detection (Krumhansl-Schmuckler algorithm)
- Scale definitions and pitch snapping
- Pattern/riff detection
- Timing quantization
- Physical playability validation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import Counter

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (correlation coefficients for key detection)
# Major profile
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
# Minor profile (harmonic minor)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Scale intervals (semitones from root)
SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],  # Ascending
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'chromatic': list(range(12)),  # All notes (no snapping)
}

# Common relative minor/major relationships
RELATIVE_MINORS = {
    0: 9,   # C -> Am
    1: 10,  # C# -> A#m
    2: 11,  # D -> Bm
    3: 0,   # D# -> Cm
    4: 1,   # E -> C#m
    5: 2,   # F -> Dm
    6: 3,   # F# -> D#m
    7: 4,   # G -> Em
    8: 5,   # G# -> Fm
    9: 6,   # A -> F#m
    10: 7,  # A# -> Gm
    11: 8,  # B -> G#m
}


@dataclass
class Key:
    """Represents a musical key"""
    root: int  # 0-11 (C to B)
    mode: str  # 'major' or 'minor'
    confidence: float  # 0-1
    
    @property
    def name(self) -> str:
        """Return key name like 'C major' or 'A minor'"""
        return f"{NOTE_NAMES[self.root]} {self.mode}"
    
    @property
    def short_name(self) -> str:
        """Return short name like 'C' or 'Am'"""
        if self.mode == 'minor':
            return f"{NOTE_NAMES[self.root]}m"
        return NOTE_NAMES[self.root]
    
    def get_scale_notes(self, scale_type: str = None) -> List[int]:
        """Get pitch classes (0-11) that belong to this key's scale."""
        if scale_type is None:
            scale_type = 'major' if self.mode == 'major' else 'natural_minor'
        
        intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS['major'])
        return [(self.root + i) % 12 for i in intervals]


@dataclass 
class Pattern:
    """Represents a detected musical pattern/riff"""
    notes: List[Tuple[int, float]]  # List of (midi, relative_time)
    occurrences: List[float]  # Start times where pattern occurs
    length: int  # Number of notes in pattern
    
    @property
    def count(self) -> int:
        return len(self.occurrences)


def detect_key(notes: List, method: str = 'krumhansl') -> Key:
    """
    Detect the key of a piece from its notes.
    
    Uses Krumhansl-Schmuckler algorithm:
    1. Build pitch class histogram
    2. Correlate with major/minor profiles for each key
    3. Return best match
    
    Args:
        notes: List of Note objects with .midi attribute
        method: Detection method ('krumhansl' or 'simple')
        
    Returns:
        Key object with detected key and confidence
    """
    if not notes:
        return Key(0, 'major', 0.0)  # Default to C major
    
    # Build pitch class histogram (weighted by duration if available)
    pitch_histogram = np.zeros(12)
    
    for note in notes:
        pc = note.midi % 12
        weight = getattr(note, 'duration', 1.0)
        pitch_histogram[pc] += weight
    
    # Normalize
    if pitch_histogram.sum() > 0:
        pitch_histogram = pitch_histogram / pitch_histogram.sum()
    
    if method == 'simple':
        # Simple method: most common pitch class is root
        root = int(np.argmax(pitch_histogram))
        # Determine major/minor by checking 3rd
        major_third = pitch_histogram[(root + 4) % 12]
        minor_third = pitch_histogram[(root + 3) % 12]
        mode = 'major' if major_third >= minor_third else 'minor'
        confidence = pitch_histogram[root]
        return Key(root, mode, float(confidence))
    
    # Krumhansl-Schmuckler algorithm
    best_key = None
    best_correlation = -2
    
    for root in range(12):
        # Rotate histogram to test this root
        rotated = np.roll(pitch_histogram, -root)
        
        # Test major
        major_corr = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = Key(root, 'major', float(major_corr))
        
        # Test minor
        minor_corr = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]
        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = Key(root, 'minor', float(minor_corr))
    
    return best_key if best_key else Key(0, 'major', 0.0)


def snap_to_scale(
    midi_note: int,
    key: Key,
    scale_type: str = None,
    direction: str = 'nearest'
) -> int:
    """
    Snap a MIDI note to the nearest note in the given scale.
    
    Args:
        midi_note: MIDI note number
        key: Key object
        scale_type: Override scale type (e.g., 'pentatonic_minor')
        direction: 'nearest', 'down', or 'up'
        
    Returns:
        Adjusted MIDI note number
    """
    scale_notes = key.get_scale_notes(scale_type)
    pitch_class = midi_note % 12
    
    # Already in scale?
    if pitch_class in scale_notes:
        return midi_note
    
    # Find nearest scale note
    octave = midi_note // 12
    
    if direction == 'nearest':
        # Find closest note in scale
        distances = []
        for sn in scale_notes:
            # Check both directions
            dist_up = (sn - pitch_class) % 12
            dist_down = (pitch_class - sn) % 12
            distances.append((min(dist_up, dist_down), sn, 'up' if dist_up <= dist_down else 'down'))
        
        min_dist, nearest_pc, snap_dir = min(distances)
        
        # Adjust octave if needed
        if snap_dir == 'down' and nearest_pc > pitch_class:
            return (octave - 1) * 12 + nearest_pc
        elif snap_dir == 'up' and nearest_pc < pitch_class:
            return (octave + 1) * 12 + nearest_pc
        return octave * 12 + nearest_pc
        
    elif direction == 'down':
        # Find nearest scale note below
        for offset in range(1, 12):
            test_pc = (pitch_class - offset) % 12
            if test_pc in scale_notes:
                if test_pc > pitch_class:
                    return (octave - 1) * 12 + test_pc
                return octave * 12 + test_pc
                
    else:  # direction == 'up'
        # Find nearest scale note above
        for offset in range(1, 12):
            test_pc = (pitch_class + offset) % 12
            if test_pc in scale_notes:
                if test_pc < pitch_class:
                    return (octave + 1) * 12 + test_pc
                return octave * 12 + test_pc
    
    return midi_note  # Fallback


def is_in_scale(midi_note: int, key: Key, scale_type: str = None) -> bool:
    """Check if a MIDI note is in the given scale."""
    scale_notes = key.get_scale_notes(scale_type)
    return (midi_note % 12) in scale_notes


def calculate_fret_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate fret distance between two positions.
    Returns absolute fret difference.
    """
    _, fret1 = pos1
    _, fret2 = pos2
    return abs(fret2 - fret1)


def is_physically_playable(
    pos1: Tuple[int, int],
    pos2: Tuple[int, int],
    time_diff: float,
    max_fret_jump: int = 5,
    min_time_for_jump: float = 0.05
) -> bool:
    """
    Check if transitioning between two fret positions is physically possible.
    
    Args:
        pos1: (string, fret) of first note
        pos2: (string, fret) of second note
        time_diff: Time in seconds between notes
        max_fret_jump: Maximum fret jump for fast transitions
        min_time_for_jump: Minimum time needed for large jumps
        
    Returns:
        True if transition is physically possible
    """
    if pos1 is None or pos2 is None:
        return True
    
    fret_distance = calculate_fret_distance(pos1, pos2)
    
    # Open strings (fret 0) are always reachable
    if pos1[1] == 0 or pos2[1] == 0:
        return True
    
    # Fast transition with large fret jump is impossible
    if time_diff < min_time_for_jump and fret_distance > max_fret_jump:
        return False
    
    # Very large jumps need more time
    if fret_distance > 12 and time_diff < 0.1:
        return False
    
    return True


def quantize_time(
    time: float,
    tempo: int,
    subdivision: int = 16,
    swing: float = 0.0
) -> float:
    """
    Quantize a time value to the nearest beat subdivision.
    
    Args:
        time: Time in seconds
        tempo: Tempo in BPM
        subdivision: Note subdivision (4=quarter, 8=eighth, 16=sixteenth)
        swing: Swing amount (0.0=straight, 0.33=triplet feel, 0.5=heavy swing)
        
    Returns:
        Quantized time in seconds
    """
    beat_duration = 60.0 / tempo
    grid_duration = beat_duration * 4 / subdivision  # Duration of one grid unit
    
    # Calculate grid position
    grid_pos = round(time / grid_duration)
    quantized = grid_pos * grid_duration
    
    # Apply swing to off-beat notes
    if swing > 0 and grid_pos % 2 == 1:  # Off-beat
        swing_offset = grid_duration * swing
        quantized += swing_offset
    
    return quantized


def quantize_notes(
    notes: List,
    tempo: int,
    subdivision: int = 16,
    swing: float = 0.0
) -> List:
    """
    Quantize all notes to a time grid.
    
    Args:
        notes: List of Note objects with .start_time attribute
        tempo: Tempo in BPM
        subdivision: Note subdivision (4, 8, 16, 32)
        swing: Swing amount (0.0-0.5)
        
    Returns:
        New list of notes with quantized times
    """
    from copy import deepcopy
    
    quantized_notes = []
    for note in notes:
        new_note = deepcopy(note)
        new_note.start_time = quantize_time(note.start_time, tempo, subdivision, swing)
        
        # Also quantize duration
        end_time = note.start_time + note.duration
        quantized_end = quantize_time(end_time, tempo, subdivision, swing)
        new_note.duration = max(0.01, quantized_end - new_note.start_time)
        
        quantized_notes.append(new_note)
    
    return quantized_notes


def detect_patterns(
    notes: List,
    min_pattern_length: int = 3,
    max_pattern_length: int = 16,
    min_occurrences: int = 2,
    time_tolerance: float = 0.1
) -> List[Pattern]:
    """
    Detect repeated musical patterns (riffs) in a note sequence.
    
    Uses melodic interval matching with timing tolerance.
    
    Args:
        notes: List of Note objects
        min_pattern_length: Minimum notes in a pattern
        max_pattern_length: Maximum notes in a pattern
        min_occurrences: Minimum times pattern must appear
        time_tolerance: Timing tolerance ratio (0.1 = 10%)
        
    Returns:
        List of Pattern objects sorted by occurrence count
    """
    if len(notes) < min_pattern_length * min_occurrences:
        return []
    
    patterns = []
    
    # Convert notes to interval sequence for pattern matching
    def notes_to_intervals(note_list):
        """Convert notes to (interval, time_ratio) pairs."""
        if len(note_list) < 2:
            return []
        intervals = []
        for i in range(1, len(note_list)):
            midi_diff = note_list[i].midi - note_list[i-1].midi
            time_diff = note_list[i].start_time - note_list[i-1].start_time
            intervals.append((midi_diff, time_diff))
        return intervals
    
    def intervals_match(int1, int2, tol=time_tolerance):
        """Check if two interval sequences match."""
        if len(int1) != len(int2):
            return False
        for (m1, t1), (m2, t2) in zip(int1, int2):
            if m1 != m2:  # Intervals must match exactly
                return False
            if t1 > 0 and abs(t2 / t1 - 1) > tol:  # Timing must be similar
                return False
        return True
    
    # Search for patterns of different lengths
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    for length in range(min_pattern_length, min(max_pattern_length + 1, len(sorted_notes) // 2 + 1)):
        # Try each starting position as a potential pattern
        for start_idx in range(len(sorted_notes) - length + 1):
            candidate = sorted_notes[start_idx:start_idx + length]
            candidate_intervals = notes_to_intervals(candidate)
            
            if not candidate_intervals:
                continue
            
            # Look for occurrences of this pattern
            occurrences = [candidate[0].start_time]
            
            for search_idx in range(start_idx + length, len(sorted_notes) - length + 1):
                test_notes = sorted_notes[search_idx:search_idx + length]
                test_intervals = notes_to_intervals(test_notes)
                
                if intervals_match(candidate_intervals, test_intervals):
                    occurrences.append(test_notes[0].start_time)
            
            if len(occurrences) >= min_occurrences:
                # Check if this pattern isn't a subset of an existing one
                is_subset = False
                pattern_notes = [(n.midi, n.start_time - candidate[0].start_time) for n in candidate]
                
                for existing in patterns:
                    if existing.length > length and all(
                        any(abs(en[0] - pn[0]) == 0 and abs(en[1] - pn[1]) < 0.1 
                            for en in existing.notes[:length])
                        for pn in pattern_notes
                    ):
                        is_subset = True
                        break
                
                if not is_subset:
                    patterns.append(Pattern(
                        notes=pattern_notes,
                        occurrences=occurrences,
                        length=length
                    ))
    
    # Sort by occurrence count (most frequent first)
    patterns.sort(key=lambda p: -p.count)
    
    # Remove duplicates (keep longer patterns)
    unique_patterns = []
    seen_starts = set()
    
    for pattern in patterns:
        # Create a signature for this pattern
        signature = tuple(round(t, 2) for t in pattern.occurrences)
        if signature not in seen_starts:
            unique_patterns.append(pattern)
            seen_starts.add(signature)
    
    return unique_patterns


def post_process_notes(
    notes: List,
    key: Optional[Key] = None,
    scale_type: str = None,
    snap_to_scale_enabled: bool = True,
    quantize_enabled: bool = False,
    tempo: int = 120,
    subdivision: int = 16,
    detect_patterns_enabled: bool = False
) -> Tuple[List, Optional[Key], List[Pattern]]:
    """
    Apply all post-processing to notes.
    
    Args:
        notes: List of Note objects
        key: Key to use (None for auto-detect)
        scale_type: Scale type override
        snap_to_scale_enabled: Whether to snap pitches to scale
        quantize_enabled: Whether to quantize timing
        tempo: Tempo in BPM
        subdivision: Quantization subdivision
        detect_patterns_enabled: Whether to detect patterns
        
    Returns:
        Tuple of (processed_notes, detected_key, patterns)
    """
    from copy import deepcopy
    
    if not notes:
        return notes, None, []
    
    # Auto-detect key if not provided
    detected_key = key
    if detected_key is None:
        detected_key = detect_key(notes)
        print(f"🎵 Auto-detected key: {detected_key.name} (confidence: {detected_key.confidence:.2f})")
    
    processed = [deepcopy(n) for n in notes]
    
    # Snap pitches to scale
    if snap_to_scale_enabled and detected_key:
        notes_adjusted = 0
        for note in processed:
            original_midi = note.midi
            note.midi = snap_to_scale(note.midi, detected_key, scale_type)
            if note.midi != original_midi:
                notes_adjusted += 1
        if notes_adjusted > 0:
            print(f"🎼 Snapped {notes_adjusted} notes to {detected_key.name} scale")
    
    # Quantize timing
    if quantize_enabled:
        processed = quantize_notes(processed, tempo, subdivision)
        print(f"⏱️  Quantized timing to 1/{subdivision} notes at {tempo} BPM")
    
    # Detect patterns
    patterns = []
    if detect_patterns_enabled:
        patterns = detect_patterns(processed)
        if patterns:
            print(f"🔁 Found {len(patterns)} repeated patterns/riffs")
    
    return processed, detected_key, patterns


def filter_impossible_transitions(
    tab_notes: List,
    notes: List,
    max_fret_jump: int = 5,
    min_time_for_jump: float = 0.05,
    tuning: List[int] = None
) -> List:
    """
    Filter out tab notes with physically impossible fret transitions.
    Replaces impossible notes with better alternatives if possible.
    
    Args:
        tab_notes: List of TabNote objects
        notes: Original Note objects (for MIDI values)
        max_fret_jump: Maximum fret jump for fast transitions
        min_time_for_jump: Minimum time needed for large jumps
        tuning: Guitar tuning
        
    Returns:
        Filtered/adjusted list of TabNote objects
    """
    from copy import deepcopy
    
    if not tab_notes or len(tab_notes) < 2:
        return tab_notes
    
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
    
    # Create time-indexed lookup for notes
    note_lookup = {round(n.start_time, 4): n for n in notes}
    
    result = [tab_notes[0]]
    removed_count = 0
    adjusted_count = 0
    
    for i in range(1, len(tab_notes)):
        current = tab_notes[i]
        prev = result[-1] if result else None
        
        if prev is None:
            result.append(current)
            continue
        
        time_diff = current.start_time - prev.start_time
        prev_pos = (prev.string, prev.fret)
        curr_pos = (current.string, current.fret)
        
        if is_physically_playable(prev_pos, curr_pos, time_diff, max_fret_jump, min_time_for_jump):
            result.append(current)
        else:
            # Try to find alternative position
            original_note = note_lookup.get(round(current.start_time, 4))
            
            if original_note:
                # Find all possible positions for this note
                midi = original_note.midi
                options = []
                
                for string_idx, open_note in enumerate(tuning):
                    fret = midi - open_note
                    if 0 <= fret <= 24:
                        test_pos = (string_idx, fret)
                        if is_physically_playable(prev_pos, test_pos, time_diff, max_fret_jump, min_time_for_jump):
                            # Score: prefer lower frets and staying close
                            fret_penalty = fret * 0.1
                            distance_penalty = abs(fret - prev.fret) * 0.2
                            score = -(fret_penalty + distance_penalty)
                            options.append((score, string_idx, fret))
                
                if options:
                    # Use best alternative
                    options.sort(reverse=True)
                    _, best_string, best_fret = options[0]
                    
                    new_tab = deepcopy(current)
                    new_tab.string = best_string
                    new_tab.fret = best_fret
                    result.append(new_tab)
                    adjusted_count += 1
                else:
                    # No playable alternative - remove note
                    removed_count += 1
            else:
                # Can't find original note - remove
                removed_count += 1
    
    if removed_count > 0 or adjusted_count > 0:
        print(f"🎸 Playability filter: adjusted {adjusted_count}, removed {removed_count} impossible transitions")
    
    return result


def prefer_lower_frets(
    tab_notes: List,
    notes: List,
    tuning: List[int] = None,
    max_position: int = 7,
    context_window: int = 4
) -> List:
    """
    Re-optimize tab notes to prefer lower fret positions when possible,
    while maintaining playability.
    
    Args:
        tab_notes: List of TabNote objects
        notes: Original Note objects (for MIDI values)
        tuning: Guitar tuning
        max_position: Prefer frets below this position
        context_window: Number of surrounding notes to consider
        
    Returns:
        Optimized list of TabNote objects
    """
    from copy import deepcopy
    
    if not tab_notes or tuning is None:
        return tab_notes
    
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]
    
    # Create time-indexed lookup for notes
    note_lookup = {round(n.start_time, 4): n for n in notes}
    
    result = []
    optimized_count = 0
    
    for i, tab_note in enumerate(tab_notes):
        original_note = note_lookup.get(round(tab_note.start_time, 4))
        
        if original_note is None:
            result.append(tab_note)
            continue
        
        midi = original_note.midi
        
        # Get all possible positions
        options = []
        for string_idx, open_note in enumerate(tuning):
            fret = midi - open_note
            if 0 <= fret <= 24:
                options.append((string_idx, fret))
        
        if len(options) <= 1:
            result.append(tab_note)
            continue
        
        # Score each option
        best_option = (tab_note.string, tab_note.fret)
        best_score = float('-inf')
        
        # Get context (surrounding notes)
        context_start = max(0, i - context_window)
        context_end = min(len(tab_notes), i + context_window + 1)
        context = tab_notes[context_start:context_end]
        
        avg_fret = np.mean([t.fret for t in context if t.fret > 0]) if context else 5
        
        for string_idx, fret in options:
            score = 0
            
            # Strong preference for lower frets
            if fret <= max_position:
                score += (max_position - fret) * 2  # Bonus for low frets
            else:
                score -= (fret - max_position) * 1  # Penalty for high frets
            
            # Prefer open strings
            if fret == 0:
                score += 5
            
            # Stay close to average position in context
            score -= abs(fret - avg_fret) * 0.3
            
            # Check playability with neighbors
            if i > 0:
                prev = result[-1]
                time_diff = tab_note.start_time - prev.start_time
                if not is_physically_playable((prev.string, prev.fret), (string_idx, fret), time_diff):
                    score -= 10  # Heavy penalty for impossible transition
            
            if i < len(tab_notes) - 1:
                next_note = tab_notes[i + 1]
                time_diff = next_note.start_time - tab_note.start_time
                if not is_physically_playable((string_idx, fret), (next_note.string, next_note.fret), time_diff):
                    score -= 5  # Moderate penalty
            
            if score > best_score:
                best_score = score
                best_option = (string_idx, fret)
        
        # Create new tab note with optimized position
        new_tab = deepcopy(tab_note)
        if best_option != (tab_note.string, tab_note.fret):
            new_tab.string, new_tab.fret = best_option
            optimized_count += 1
        
        result.append(new_tab)
    
    if optimized_count > 0:
        print(f"🎯 Optimized {optimized_count} notes to lower fret positions")
    
    return result


def format_pattern_info(patterns: List[Pattern], notes: List) -> str:
    """Format detected patterns for display."""
    if not patterns:
        return "No repeated patterns detected."
    
    lines = ["🔁 Detected Patterns/Riffs:", ""]
    
    for i, pattern in enumerate(patterns[:5], 1):  # Show top 5
        # Convert pattern notes to note names
        note_names = []
        for midi, _ in pattern.notes:
            # Find the actual midi value (pattern stores intervals relative to first)
            note_names.append(NOTE_NAMES[midi % 12])
        
        times = [f"{t:.2f}s" for t in pattern.occurrences[:3]]
        if len(pattern.occurrences) > 3:
            times.append("...")
        
        lines.append(f"  Pattern {i}: {pattern.length} notes, {pattern.count}x occurrences")
        lines.append(f"    Times: {', '.join(times)}")
    
    return '\n'.join(lines)


def parse_key_string(key_str: str) -> Key:
    """
    Parse a key string like 'Am', 'C', 'F#m', 'Bb major'.
    
    Returns Key object.
    """
    key_str = key_str.strip()
    
    # Handle explicit mode
    if ' major' in key_str.lower():
        root_str = key_str.lower().replace(' major', '').strip()
        mode = 'major'
    elif ' minor' in key_str.lower():
        root_str = key_str.lower().replace(' minor', '').strip()
        mode = 'minor'
    elif key_str.endswith('m') and len(key_str) > 1:
        root_str = key_str[:-1]
        mode = 'minor'
    else:
        root_str = key_str
        mode = 'major'
    
    # Parse root note
    root_str = root_str.upper()
    
    # Handle flats -> sharps
    root_str = root_str.replace('BB', 'A#').replace('DB', 'C#').replace('EB', 'D#')
    root_str = root_str.replace('GB', 'F#').replace('AB', 'G#')
    
    try:
        root = NOTE_NAMES.index(root_str)
    except ValueError:
        # Try without sharp/flat
        root = NOTE_NAMES.index(root_str[0]) if root_str else 0
    
    return Key(root, mode, 1.0)  # Explicit key has full confidence
