#!/usr/bin/env python3
"""
Fretboard Position Tracking Module

Tracks the guitarist's hand POSITION on the fretboard to constrain pitch detection.
Uses physical constraints (can't jump 10 frets instantly) to improve accuracy.

Key concepts:
- POSITION: Where the index finger is (fret 1, 3, 5, 7, 9, 12, etc.)
- BOX: A shape pattern at a position (pentatonic box, CAGED shape)
- SPAN: How far the hand can stretch (typically 4-5 frets)

This module integrates with pitch detection to:
1. Infer initial position from first few notes
2. Track position changes over time
3. Constrain note possibilities based on current position
4. Apply transition penalties for large position shifts
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
from collections import deque


# ============================================================================
# POSITION DEFINITIONS
# ============================================================================

class PositionType(Enum):
    """Types of fretboard positions/patterns."""
    OPEN = "open"              # Open position (frets 0-4)
    PENTATONIC_BOX1 = "pent1"  # Pentatonic box 1 (minor root)
    PENTATONIC_BOX2 = "pent2"  # Pentatonic box 2
    PENTATONIC_BOX3 = "pent3"  # Pentatonic box 3
    PENTATONIC_BOX4 = "pent4"  # Pentatonic box 4
    PENTATONIC_BOX5 = "pent5"  # Pentatonic box 5 (major root)
    THREE_NPS = "3nps"         # 3-note-per-string patterns
    CAGED_C = "caged_c"        # CAGED C shape
    CAGED_A = "caged_a"        # CAGED A shape
    CAGED_G = "caged_g"        # CAGED G shape
    CAGED_E = "caged_e"        # CAGED E shape
    CAGED_D = "caged_d"        # CAGED D shape
    GENERIC = "generic"        # Generic position (just fret range)


@dataclass
class Position:
    """
    Represents a hand position on the fretboard.
    
    The base_fret is where the index finger sits.
    The span defines how far the pinky can reach.
    """
    base_fret: int              # Index finger position (0 = open position)
    span: int = 4               # Typical 4-fret span
    position_type: PositionType = PositionType.GENERIC
    key_root: int = 0           # Root note (pitch class 0-11)
    
    @property
    def min_fret(self) -> int:
        """Minimum reachable fret."""
        return max(0, self.base_fret)
    
    @property
    def max_fret(self) -> int:
        """Maximum reachable fret (including stretch)."""
        return self.base_fret + self.span
    
    @property
    def comfortable_range(self) -> Tuple[int, int]:
        """Comfortable fret range (less stretch)."""
        return (self.base_fret, self.base_fret + 3)
    
    def contains_fret(self, fret: int) -> bool:
        """Check if a fret is reachable from this position."""
        if fret == 0:  # Open strings always reachable
            return self.base_fret <= 4  # Only from open/low positions
        return self.min_fret <= fret <= self.max_fret
    
    def fret_cost(self, fret: int) -> float:
        """
        Calculate cost of reaching a fret from this position.
        Lower = easier to reach.
        """
        if fret == 0:
            # Open string cost depends on position
            if self.base_fret <= 2:
                return 0.0  # Easy from open position
            elif self.base_fret <= 5:
                return 0.5  # Possible but awkward
            else:
                return 2.0  # Very awkward from high position
        
        if not self.contains_fret(fret):
            # Out of reach - high penalty
            return 10.0 + abs(fret - self.base_fret)
        
        # Distance from index finger
        distance = abs(fret - self.base_fret)
        
        if distance <= 3:
            return distance * 0.2  # Comfortable
        else:
            return 0.6 + (distance - 3) * 0.5  # Stretch
    
    def __repr__(self) -> str:
        return f"Position(fret={self.base_fret}, type={self.position_type.value})"


# ============================================================================
# PENTATONIC BOX DEFINITIONS
# ============================================================================

# Pentatonic minor intervals: 0, 3, 5, 7, 10 (R, b3, 4, 5, b7)
# Each box is defined as fret offsets from the box's starting fret per string
# Format: {string: [(fret_offset, interval), ...]}

PENTATONIC_BOXES = {
    # Box 1 - Root on string 6 (minor root position)
    PositionType.PENTATONIC_BOX1: {
        0: [(0, 0), (3, 3)],      # Low E: R, b3
        1: [(0, 5), (2, 7)],      # A: 4, 5
        2: [(0, 7), (2, 10)],     # D: 5, b7
        3: [(0, 10), (2, 0)],     # G: b7, R (octave)
        4: [(0, 0), (3, 3)],      # B: R, b3
        5: [(0, 3), (3, 5)],      # High e: b3, 4
    },
    
    # Box 2
    PositionType.PENTATONIC_BOX2: {
        0: [(0, 3), (2, 5)],
        1: [(0, 7), (2, 10)],
        2: [(0, 10), (2, 0)],
        3: [(0, 0), (2, 3)],
        4: [(0, 3), (2, 5)],
        5: [(0, 5), (2, 7)],
    },
    
    # Box 3
    PositionType.PENTATONIC_BOX3: {
        0: [(0, 5), (2, 7)],
        1: [(0, 10), (3, 0)],
        2: [(0, 0), (2, 3)],
        3: [(0, 3), (2, 5)],
        4: [(0, 5), (3, 7)],
        5: [(0, 7), (2, 10)],
    },
    
    # Box 4
    PositionType.PENTATONIC_BOX4: {
        0: [(0, 7), (2, 10)],
        1: [(0, 0), (2, 3)],
        2: [(0, 3), (2, 5)],
        3: [(0, 5), (3, 7)],
        4: [(0, 7), (2, 10)],
        5: [(0, 10), (2, 0)],
    },
    
    # Box 5 - Major pentatonic root position
    PositionType.PENTATONIC_BOX5: {
        0: [(0, 10), (3, 0)],
        1: [(0, 3), (2, 5)],
        2: [(0, 5), (2, 7)],
        3: [(0, 7), (2, 10)],
        4: [(0, 10), (3, 0)],
        5: [(0, 0), (3, 3)],
    },
}

# 3-note-per-string patterns (for fast runs/shred)
# Each string has exactly 3 notes spanning ~5-6 frets
THREE_NPS_PATTERN = {
    0: [0, 2, 4],  # Low E
    1: [0, 2, 4],  # A
    2: [0, 2, 4],  # D
    3: [0, 2, 4],  # G
    4: [0, 2, 4],  # B
    5: [0, 2, 4],  # High e
}


# ============================================================================
# CAGED SHAPE DEFINITIONS
# ============================================================================

# CAGED shapes define chord shapes and their scale positions
# Format: starting fret offset for each shape relative to root

CAGED_POSITIONS = {
    # C shape - root on string 5
    PositionType.CAGED_C: {
        'root_string': 1,  # A string
        'fret_offset': 3,  # Starts 3 frets below root
        'span': 4,
    },
    # A shape - root on string 5
    PositionType.CAGED_A: {
        'root_string': 1,
        'fret_offset': 0,
        'span': 4,
    },
    # G shape - root on strings 6 and 1
    PositionType.CAGED_G: {
        'root_string': 0,
        'fret_offset': 0,
        'span': 4,
    },
    # E shape - root on string 6
    PositionType.CAGED_E: {
        'root_string': 0,
        'fret_offset': 0,
        'span': 4,
    },
    # D shape - root on string 4
    PositionType.CAGED_D: {
        'root_string': 2,
        'fret_offset': 0,
        'span': 4,
    },
}


# ============================================================================
# POSITION TRACKER
# ============================================================================

@dataclass
class PositionTransition:
    """Records a position change."""
    from_position: Position
    to_position: Position
    time: float
    cost: float
    reason: str  # Why the transition happened


@dataclass
class PositionTrackerConfig:
    """Configuration for position tracking."""
    # Position preferences
    prefer_low_frets: bool = True
    low_fret_bonus: float = 0.5
    
    # Transition costs
    position_shift_cost: float = 2.0  # Base cost per fret of position shift
    max_instant_shift: int = 4        # Max frets that can shift "instantly"
    large_shift_penalty: float = 5.0  # Additional penalty for large shifts
    
    # Timing constraints
    min_shift_time: float = 0.15      # Minimum time (sec) for position shift
    fast_passage_threshold: float = 0.1  # Notes faster than this are "fast"
    
    # Pattern detection
    detect_pentatonic: bool = True
    detect_caged: bool = True
    detect_3nps: bool = True
    pattern_bonus: float = 1.0        # Bonus for staying in a pattern
    
    # History
    history_size: int = 8             # Notes to consider for position inference
    position_inertia: float = 0.7     # Weight given to current position
    
    # Verbosity
    verbose: bool = False


class PositionTracker:
    """
    Tracks hand position on the fretboard over time.
    
    Uses recent notes to infer position and applies physical
    constraints to note selection.
    """
    
    def __init__(
        self,
        tuning: List[int] = None,
        config: PositionTrackerConfig = None
    ):
        self.tuning = tuning or [40, 45, 50, 55, 59, 64]  # Standard tuning
        self.config = config or PositionTrackerConfig()
        
        # Current state
        self.current_position: Optional[Position] = None
        self.detected_pattern: Optional[PositionType] = None
        self.key_root: Optional[int] = None
        
        # History
        self.position_history: deque = deque(maxlen=20)
        self.note_history: deque = deque(maxlen=self.config.history_size)
        self.transitions: List[PositionTransition] = []
        
        # Pre-compute fret options for all MIDI notes
        self._fret_cache: Dict[int, List[Tuple[int, int]]] = {}
    
    def get_fret_options(self, midi_note: int) -> List[Tuple[int, int]]:
        """Get all (string, fret) options for a MIDI note."""
        if midi_note not in self._fret_cache:
            options = []
            for string_idx, open_note in enumerate(self.tuning):
                fret = midi_note - open_note
                if 0 <= fret <= 24:
                    options.append((string_idx, fret))
            self._fret_cache[midi_note] = options
        return self._fret_cache[midi_note]
    
    def infer_initial_position(self, notes: List) -> Position:
        """
        Infer the starting position from the first few notes.
        
        Args:
            notes: List of Note objects (with .midi attribute)
            
        Returns:
            Inferred starting Position
        """
        if not notes:
            return Position(base_fret=0, position_type=PositionType.OPEN)
        
        # Collect all possible frets for the first N notes
        all_frets = []
        for note in notes[:self.config.history_size]:
            options = self.get_fret_options(note.midi)
            for string, fret in options:
                if fret > 0:  # Exclude open strings for position detection
                    all_frets.append(fret)
        
        if not all_frets:
            # All open strings - open position
            return Position(base_fret=0, position_type=PositionType.OPEN)
        
        # Find the most common fret range
        fret_counts = {}
        for fret in all_frets:
            # Group into 4-fret windows
            base = (fret // 4) * 4
            if base == 0:
                base = 1  # Avoid 0 base for fretted notes
            fret_counts[base] = fret_counts.get(base, 0) + 1
        
        # Choose most common position
        best_base = max(fret_counts.keys(), key=lambda k: fret_counts[k])
        
        # Try to identify pattern
        pattern = self._detect_pattern(notes[:self.config.history_size], best_base)
        
        position = Position(
            base_fret=best_base,
            span=4,
            position_type=pattern,
            key_root=self._detect_key_root(notes[:8])
        )
        
        self.current_position = position
        self.position_history.append((0.0, position))
        
        if self.config.verbose:
            print(f"📍 Inferred initial position: fret {position.base_fret} ({position.position_type.value})")
        
        return position
    
    def _detect_key_root(self, notes: List) -> int:
        """Detect likely key root from notes (pitch class 0-11)."""
        if not notes:
            return 0
        
        # Count pitch classes
        pc_counts = {}
        for note in notes:
            pc = note.midi % 12
            pc_counts[pc] = pc_counts.get(pc, 0) + 1
        
        # Return most common
        return max(pc_counts.keys(), key=lambda k: pc_counts[k])
    
    def _detect_pattern(self, notes: List, base_fret: int) -> PositionType:
        """
        Detect if notes match a known pattern (pentatonic box, etc.).
        """
        if not self.config.detect_pentatonic:
            return PositionType.GENERIC
        
        # Get fret positions used
        frets_used = set()
        for note in notes:
            options = self.get_fret_options(note.midi)
            for string, fret in options:
                if base_fret <= fret <= base_fret + 4:
                    frets_used.add(fret)
        
        if not frets_used:
            return PositionType.GENERIC
        
        # Check fret span
        span = max(frets_used) - min(frets_used)
        
        if span <= 3:
            # Could be pentatonic box
            # For now, use generic but could match against box patterns
            return PositionType.PENTATONIC_BOX1
        elif span <= 5:
            # Likely 3-note-per-string
            return PositionType.THREE_NPS
        
        return PositionType.GENERIC
    
    def score_fret_option(
        self,
        string: int,
        fret: int,
        time: float,
        prev_time: Optional[float] = None,
        prev_string: Optional[int] = None,
        prev_fret: Optional[int] = None
    ) -> float:
        """
        Score a (string, fret) option based on current position.
        
        Lower score = better option.
        """
        score = 0.0
        
        # Base position cost
        if self.current_position:
            score += self.current_position.fret_cost(fret)
        else:
            # No position yet - prefer lower frets
            if fret <= 5:
                score -= self.config.low_fret_bonus * (6 - fret)
            elif fret > 12:
                score += (fret - 12) * 0.1
        
        # Transition cost from previous note
        if prev_fret is not None:
            time_delta = time - prev_time if prev_time else 0.5
            
            fret_jump = abs(fret - prev_fret) if prev_fret > 0 and fret > 0 else 0
            string_jump = abs(string - prev_string) if prev_string is not None else 0
            
            # Penalty for large fret jumps in fast passages
            if time_delta < self.config.fast_passage_threshold:
                if fret_jump > 4:
                    # Physically difficult/impossible
                    score += self.config.large_shift_penalty * (fret_jump - 4)
                elif fret_jump > 2:
                    # Awkward but possible
                    score += (fret_jump - 2) * 0.5
            
            # String skip penalty
            if string_jump > 1:
                score += (string_jump - 1) * 0.3
            
            # Bonus for adjacent strings
            if string_jump == 1:
                score -= 0.2
        
        # Pattern bonus
        if self.detected_pattern and self.detected_pattern != PositionType.GENERIC:
            if self._fret_in_pattern(fret, string, self.detected_pattern):
                score -= self.config.pattern_bonus
        
        return score
    
    def _fret_in_pattern(self, fret: int, string: int, pattern: PositionType) -> bool:
        """Check if a fret/string is in the current pattern."""
        if pattern in PENTATONIC_BOXES:
            box = PENTATONIC_BOXES[pattern]
            if string in box:
                # Check if fret offset matches any note in the box
                base = self.current_position.base_fret if self.current_position else 0
                for fret_offset, _ in box[string]:
                    if fret == base + fret_offset:
                        return True
        return False
    
    def choose_best_fret(
        self,
        midi_note: int,
        time: float,
        prev_time: Optional[float] = None,
        prev_string: Optional[int] = None,
        prev_fret: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Choose the best (string, fret) for a note given current position.
        
        Returns:
            (string, fret) tuple or None if note is out of range
        """
        options = self.get_fret_options(midi_note)
        
        if not options:
            return None
        
        if len(options) == 1:
            return options[0]
        
        # Score each option
        scored = []
        for string, fret in options:
            score = self.score_fret_option(
                string, fret, time,
                prev_time, prev_string, prev_fret
            )
            scored.append((score, string, fret))
        
        # Sort by score (lowest = best)
        scored.sort(key=lambda x: x[0])
        
        _, best_string, best_fret = scored[0]
        
        # Update position if needed
        self._maybe_update_position(best_fret, time)
        
        return (best_string, best_fret)
    
    def _maybe_update_position(self, fret: int, time: float) -> None:
        """Update position if the note suggests a position shift."""
        if fret == 0:
            return  # Open strings don't indicate position
        
        if self.current_position is None:
            self.current_position = Position(base_fret=max(1, fret - 2))
            return
        
        # Check if fret is outside current position
        if not self.current_position.contains_fret(fret):
            # Need to shift
            new_base = max(1, fret - 2)  # Center the fret in the new position
            
            # Apply inertia - don't shift too aggressively
            if len(self.note_history) > 0:
                recent_frets = [f for f in self.note_history if f > 0]
                if recent_frets:
                    avg_recent = np.mean(recent_frets)
                    # Blend new position with recent average
                    new_base = int(
                        new_base * (1 - self.config.position_inertia) +
                        avg_recent * self.config.position_inertia
                    )
                    new_base = max(1, new_base)
            
            old_position = self.current_position
            self.current_position = Position(
                base_fret=new_base,
                position_type=self.detected_pattern or PositionType.GENERIC
            )
            
            # Record transition
            self.transitions.append(PositionTransition(
                from_position=old_position,
                to_position=self.current_position,
                time=time,
                cost=abs(new_base - old_position.base_fret) * self.config.position_shift_cost,
                reason="fret_out_of_range"
            ))
            
            if self.config.verbose:
                print(f"📍 Position shift: fret {old_position.base_fret} → {new_base} at t={time:.2f}s")
        
        # Track note in history
        self.note_history.append(fret)
    
    def constrain_to_position(
        self,
        f0_hz: np.ndarray,
        times: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        tuning: List[int] = None
    ) -> np.ndarray:
        """
        Constrain pitch values to be playable from current position.
        
        This is called during pitch detection to snap pitches to
        physically plausible values.
        
        Args:
            f0_hz: Array of fundamental frequencies in Hz
            times: Array of time stamps
            confidence: Optional confidence values
            tuning: Guitar tuning as MIDI notes
            
        Returns:
            Constrained f0 array
        """
        import librosa
        
        if tuning is None:
            tuning = self.tuning
        
        constrained = f0_hz.copy()
        
        prev_time = None
        prev_fret = None
        prev_string = None
        
        for i in range(len(f0_hz)):
            if f0_hz[i] <= 0:
                continue
            
            # Skip low confidence
            if confidence is not None and confidence[i] < 0.3:
                continue
            
            # Convert to MIDI
            midi = int(round(librosa.hz_to_midi(f0_hz[i])))
            
            # Choose best fret position
            best = self.choose_best_fret(
                midi, times[i],
                prev_time, prev_string, prev_fret
            )
            
            if best is None:
                # Out of guitar range - might need to adjust
                continue
            
            string, fret = best
            
            # Calculate the actual pitch for this fret
            actual_midi = tuning[string] + fret
            
            # If it's different from detected, adjust
            if actual_midi != midi:
                constrained[i] = librosa.midi_to_hz(actual_midi)
            
            prev_time = times[i]
            prev_string = string
            prev_fret = fret
        
        return constrained
    
    def get_position_at_time(self, time: float) -> Optional[Position]:
        """Get the hand position at a specific time."""
        # Find the most recent position before this time
        for t, pos in reversed(list(self.position_history)):
            if t <= time:
                return pos
        return self.current_position
    
    def get_transition_cost(self, from_fret: int, to_fret: int, time_delta: float) -> float:
        """
        Calculate the physical cost of transitioning between frets.
        
        Accounts for time available - faster passages have higher cost
        for the same distance.
        """
        if from_fret == 0 or to_fret == 0:
            return 0.0  # Open string transitions are free
        
        distance = abs(to_fret - from_fret)
        
        if distance <= self.config.max_instant_shift:
            # Within reach - cost based on stretch
            base_cost = distance * 0.2
        else:
            # Requires position shift
            base_cost = self.config.position_shift_cost + distance * 0.3
        
        # Adjust for timing
        if time_delta < self.config.min_shift_time:
            # Not enough time to shift cleanly
            if distance > self.config.max_instant_shift:
                base_cost += self.config.large_shift_penalty
            else:
                base_cost *= (self.config.min_shift_time / max(time_delta, 0.01))
        
        return base_cost
    
    def smooth_positions(
        self,
        tab_notes: List,
        max_iterations: int = 3
    ) -> List:
        """
        Post-process tab notes to smooth position changes.
        
        Iteratively adjusts notes to minimize total position movement
        while keeping notes playable.
        """
        from copy import deepcopy
        
        if len(tab_notes) < 2:
            return tab_notes
        
        smoothed = deepcopy(tab_notes)
        
        for iteration in range(max_iterations):
            changes = 0
            
            for i in range(1, len(smoothed) - 1):
                current = smoothed[i]
                prev = smoothed[i - 1]
                next_note = smoothed[i + 1]
                
                if current.fret == 0:
                    continue  # Open strings stay
                
                # Get the MIDI note
                midi = self.tuning[current.string] + current.fret
                
                # Get alternative positions
                options = self.get_fret_options(midi)
                
                if len(options) <= 1:
                    continue
                
                # Calculate current cost
                current_cost = (
                    self.get_transition_cost(prev.fret, current.fret, 0.1) +
                    self.get_transition_cost(current.fret, next_note.fret, 0.1)
                )
                
                # Try alternatives
                best_cost = current_cost
                best_option = (current.string, current.fret)
                
                for string, fret in options:
                    if (string, fret) == (current.string, current.fret):
                        continue
                    
                    cost = (
                        self.get_transition_cost(prev.fret, fret, 0.1) +
                        self.get_transition_cost(fret, next_note.fret, 0.1)
                    )
                    
                    if cost < best_cost - 0.1:  # Threshold to avoid tiny changes
                        best_cost = cost
                        best_option = (string, fret)
                
                if best_option != (current.string, current.fret):
                    smoothed[i].string = best_option[0]
                    smoothed[i].fret = best_option[1]
                    changes += 1
            
            if changes == 0:
                break
            
            if self.config.verbose:
                print(f"  Position smoothing iteration {iteration + 1}: {changes} changes")
        
        return smoothed
    
    def analyze_positions(self, tab_notes: List) -> Dict:
        """
        Analyze position usage in tab notes.
        
        Returns statistics about position changes, patterns detected, etc.
        """
        if not tab_notes:
            return {}
        
        frets = [n.fret for n in tab_notes if n.fret > 0]
        
        if not frets:
            return {'all_open': True}
        
        # Group frets into positions
        position_ranges = {}
        for fret in frets:
            pos_base = max(1, ((fret - 1) // 4) * 4 + 1)
            position_ranges[pos_base] = position_ranges.get(pos_base, 0) + 1
        
        # Count position changes
        position_changes = 0
        current_pos = None
        for fret in frets:
            pos_base = max(1, ((fret - 1) // 4) * 4 + 1)
            if current_pos is not None and pos_base != current_pos:
                if abs(pos_base - current_pos) >= 4:
                    position_changes += 1
            current_pos = pos_base
        
        # Calculate fret jumps
        fret_jumps = []
        for i in range(1, len(frets)):
            jump = abs(frets[i] - frets[i-1])
            fret_jumps.append(jump)
        
        return {
            'total_notes': len(tab_notes),
            'fretted_notes': len(frets),
            'open_strings': len(tab_notes) - len(frets),
            'position_ranges': position_ranges,
            'primary_position': max(position_ranges.keys(), key=lambda k: position_ranges[k]),
            'position_changes': position_changes,
            'avg_fret': np.mean(frets),
            'max_fret': max(frets),
            'min_fret': min(frets),
            'avg_fret_jump': np.mean(fret_jumps) if fret_jumps else 0,
            'max_fret_jump': max(fret_jumps) if fret_jumps else 0,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_position_tracker(
    tuning: List[int] = None,
    prefer_low_frets: bool = True,
    verbose: bool = False
) -> PositionTracker:
    """Create a position tracker with default settings."""
    config = PositionTrackerConfig(
        prefer_low_frets=prefer_low_frets,
        verbose=verbose
    )
    return PositionTracker(tuning=tuning, config=config)


def apply_position_constraints(
    notes: List,
    tuning: List[int] = None,
    verbose: bool = True
) -> List:
    """
    Apply position tracking to convert notes to tab notes.
    
    Args:
        notes: List of Note objects
        tuning: Guitar tuning
        verbose: Print info
        
    Returns:
        List of TabNote objects with position-optimized fret choices
    """
    # Import TabNote if needed
    try:
        from guitar_tabs import TabNote
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class TabNote:
            string: int
            fret: int
            start_time: float
            duration: float
    
    if not notes:
        return []
    
    tracker = create_position_tracker(tuning=tuning, verbose=verbose)
    
    # Sort by time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    # Infer initial position
    tracker.infer_initial_position(sorted_notes[:8])
    
    # Process each note
    tab_notes = []
    prev_time = None
    prev_string = None
    prev_fret = None
    
    for note in sorted_notes:
        result = tracker.choose_best_fret(
            note.midi, note.start_time,
            prev_time, prev_string, prev_fret
        )
        
        if result is None:
            # Out of range - skip
            continue
        
        string, fret = result
        tab_notes.append(TabNote(
            string=string,
            fret=fret,
            start_time=note.start_time,
            duration=note.duration
        ))
        
        prev_time = note.start_time
        prev_string = string
        prev_fret = fret
    
    # Smooth positions
    tab_notes = tracker.smooth_positions(tab_notes)
    
    if verbose:
        stats = tracker.analyze_positions(tab_notes)
        print(f"\n📍 Position Analysis:")
        print(f"   Primary position: fret {stats.get('primary_position', '?')}")
        print(f"   Position changes: {stats.get('position_changes', 0)}")
        print(f"   Avg fret: {stats.get('avg_fret', 0):.1f}")
        print(f"   Avg fret jump: {stats.get('avg_fret_jump', 0):.1f}")
        print(f"   Max fret jump: {stats.get('max_fret_jump', 0)}")
    
    return tab_notes


def add_position_tracking_args(parser) -> None:
    """Add position tracking arguments to argument parser."""
    group = parser.add_argument_group('Position Tracking Options')
    
    group.add_argument('--position-tracking', action='store_true',
                       help='Enable fretboard position tracking. Tracks hand position '
                            'and applies physical constraints to improve tab accuracy.')
    
    group.add_argument('--position-verbose', action='store_true',
                       help='Print position tracking info')
    
    group.add_argument('--position-low-fret-bonus', type=float, default=0.5,
                       help='Bonus for lower fret positions (default: 0.5)')
    
    group.add_argument('--position-shift-cost', type=float, default=2.0,
                       help='Cost per fret of position shift (default: 2.0)')
    
    group.add_argument('--position-max-instant', type=int, default=4,
                       help='Max frets that can shift instantly (default: 4)')


def config_from_args(args) -> PositionTrackerConfig:
    """Create config from parsed arguments."""
    return PositionTrackerConfig(
        prefer_low_frets=True,
        low_fret_bonus=getattr(args, 'position_low_fret_bonus', 0.5),
        position_shift_cost=getattr(args, 'position_shift_cost', 2.0),
        max_instant_shift=getattr(args, 'position_max_instant', 4),
        verbose=getattr(args, 'position_verbose', False)
    )


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fretboard position tracking')
    parser.add_argument('--test', action='store_true', help='Run test with sample notes')
    parser.add_argument('--scale', type=str, default='pentatonic',
                        choices=['pentatonic', 'major', 'random'],
                        help='Type of test scale to generate')
    args = parser.parse_args()
    
    if args.test:
        # Create test notes
        from dataclasses import dataclass
        
        @dataclass
        class TestNote:
            midi: int
            start_time: float
            duration: float
            confidence: float = 0.9
        
        print("🎸 Testing Fretboard Position Tracking\n")
        
        if args.scale == 'pentatonic':
            # A minor pentatonic (A, C, D, E, G) - common guitar scale
            # Position 5 (fret 5 on low E = A)
            test_notes = [
                TestNote(midi=45, start_time=0.0, duration=0.25),   # A2 (open A string)
                TestNote(midi=48, start_time=0.25, duration=0.25),  # C3
                TestNote(midi=50, start_time=0.5, duration=0.25),   # D3
                TestNote(midi=52, start_time=0.75, duration=0.25),  # E3
                TestNote(midi=55, start_time=1.0, duration=0.25),   # G3
                TestNote(midi=57, start_time=1.25, duration=0.25),  # A3
                TestNote(midi=60, start_time=1.5, duration=0.25),   # C4
                TestNote(midi=62, start_time=1.75, duration=0.25),  # D4
                TestNote(midi=64, start_time=2.0, duration=0.25),   # E4
                TestNote(midi=67, start_time=2.25, duration=0.25),  # G4
                TestNote(midi=69, start_time=2.5, duration=0.25),   # A4
            ]
            print("Testing A minor pentatonic scale...")
        
        elif args.scale == 'major':
            # C major scale
            test_notes = [
                TestNote(midi=60, start_time=0.0, duration=0.5),   # C4
                TestNote(midi=62, start_time=0.5, duration=0.5),   # D4
                TestNote(midi=64, start_time=1.0, duration=0.5),   # E4
                TestNote(midi=65, start_time=1.5, duration=0.5),   # F4
                TestNote(midi=67, start_time=2.0, duration=0.5),   # G4
                TestNote(midi=69, start_time=2.5, duration=0.5),   # A4
                TestNote(midi=71, start_time=3.0, duration=0.5),   # B4
                TestNote(midi=72, start_time=3.5, duration=0.5),   # C5
            ]
            print("Testing C major scale...")
        
        else:  # random
            import random
            test_notes = [
                TestNote(
                    midi=random.randint(40, 84),
                    start_time=i * 0.3,
                    duration=0.25
                )
                for i in range(15)
            ]
            print("Testing random notes...")
        
        # Apply position tracking
        tab_notes = apply_position_constraints(test_notes, verbose=True)
        
        print("\n📋 Tab Output:")
        print("-" * 40)
        for tn in tab_notes:
            string_name = ['E', 'A', 'D', 'G', 'B', 'e'][tn.string]
            print(f"  t={tn.start_time:.2f}s: String {string_name} Fret {tn.fret}")
