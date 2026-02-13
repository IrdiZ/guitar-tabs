#!/usr/bin/env python3
"""
Fingering Optimizer for Guitar Tab Generator

Optimizes fret/string assignments for playability by considering:
- Hand position continuity and stretches
- Minimizing fret jumps between notes
- Technique requirements (bends, vibrato, hammer-ons)
- Open string preferences
- Alternate tuning support
- Chord shape playability
- Look-ahead optimization

Uses dynamic programming with cost-based optimization to find
the most playable fingering for a sequence of notes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from collections import defaultdict

# Import from main module
try:
    from guitar_tabs import (
        Note, TabNote, TUNINGS, NUM_FRETS, NOTE_NAMES,
        midi_to_fret_options, CHORD_TIME_THRESHOLD
    )
except ImportError:
    # Fallback definitions for standalone testing
    TUNINGS = {
        'standard': [40, 45, 50, 55, 59, 64],
        'drop_d': [38, 45, 50, 55, 59, 64],
    }
    NUM_FRETS = 24
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    CHORD_TIME_THRESHOLD = 0.050
    
    @dataclass
    class Note:
        """Fallback Note class for standalone testing."""
        midi: int
        start_time: float
        duration: float
        confidence: float
        
        @property
        def name(self) -> str:
            return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)
    
    @dataclass
    class TabNote:
        """Fallback TabNote class for standalone testing."""
        string: int
        fret: int
        start_time: float
        duration: float
    
    def midi_to_fret_options(midi_note: int, tuning: List[int] = None) -> List[Tuple[int, int]]:
        """Fallback function for standalone testing."""
        if tuning is None:
            tuning = TUNINGS['standard']
        options = []
        for string_idx, open_note in enumerate(tuning):
            fret = midi_note - open_note
            if 0 <= fret <= NUM_FRETS:
                options.append((string_idx, fret))
        return options


@dataclass
class HandPosition:
    """Represents current hand position on the fretboard."""
    base_fret: int  # The index finger position
    span: int = 4   # Typical span is 4 frets (index to pinky)
    
    @property
    def min_fret(self) -> int:
        return max(0, self.base_fret)
    
    @property
    def max_fret(self) -> int:
        return self.base_fret + self.span
    
    def contains(self, fret: int) -> bool:
        """Check if a fret is within comfortable reach."""
        if fret == 0:  # Open strings always reachable
            return True
        return self.min_fret <= fret <= self.max_fret
    
    def distance_to(self, fret: int) -> int:
        """Calculate distance from current position to a fret."""
        if fret == 0:
            return 0  # Open strings are always "close"
        if self.contains(fret):
            return 0
        if fret < self.min_fret:
            return self.min_fret - fret
        return fret - self.max_fret


@dataclass
class FingeringCandidate:
    """A candidate fingering for a note or chord."""
    positions: List[Tuple[int, int]]  # List of (string, fret) pairs
    cost: float = 0.0
    hand_position: Optional[HandPosition] = None
    
    @property
    def strings_used(self) -> Set[int]:
        return {s for s, f in self.positions}
    
    @property
    def frets_used(self) -> Set[int]:
        return {f for s, f in self.positions if f > 0}
    
    @property
    def min_fret(self) -> int:
        frets = [f for s, f in self.positions if f > 0]
        return min(frets) if frets else 0
    
    @property
    def max_fret(self) -> int:
        frets = [f for s, f in self.positions if f > 0]
        return max(frets) if frets else 0
    
    @property
    def fret_span(self) -> int:
        if not self.frets_used:
            return 0
        return self.max_fret - self.min_fret


@dataclass
class FingeringConfig:
    """Configuration for fingering optimization."""
    # Hand position
    max_fret_span: int = 4  # Max stretch between fingers
    position_change_penalty: float = 2.0  # Cost of shifting hand position
    
    # Fret preferences
    low_fret_bonus: float = 0.5  # Bonus per fret closer to nut (up to fret 5)
    open_string_bonus: float = 1.5  # Bonus for using open strings
    high_fret_penalty: float = 0.1  # Penalty per fret above 12
    
    # String preferences
    adjacent_string_bonus: float = 0.3  # Bonus for moving to adjacent string
    string_skip_penalty: float = 0.5  # Penalty per string skipped
    
    # Technique considerations
    bend_string_bonus: Dict[int, float] = field(default_factory=lambda: {
        # Higher strings are easier to bend (G=3, B=4, high e=5)
        3: 0.3, 4: 0.5, 5: 0.4
    })
    
    # Transition costs
    same_string_penalty: float = 0.2  # Small penalty for consecutive notes on same string
    large_jump_threshold: int = 5  # Frets - beyond this is "large"
    large_jump_penalty: float = 1.5  # Per-fret penalty for large jumps
    
    # Timing considerations
    chord_window: float = 0.05  # Seconds - notes within this are simultaneous
    fast_passage_threshold: float = 0.1  # Seconds - faster than this needs careful fingering
    fast_passage_penalty_mult: float = 1.5  # Multiply costs for fast passages
    
    # Look-ahead
    lookahead_notes: int = 3  # How many notes to look ahead
    lookahead_decay: float = 0.5  # Discount factor for future costs


class FingeringOptimizer:
    """
    Optimizes fingering for a sequence of notes using dynamic programming.
    
    The optimizer considers:
    - Current hand position and the cost of position changes
    - Physical stretches required for each fingering
    - String transitions (adjacent vs skip)
    - Technique requirements (bends, vibrato)
    - Open string opportunities
    - Look-ahead for upcoming notes
    """
    
    def __init__(
        self,
        tuning: List[int] = None,
        config: FingeringConfig = None
    ):
        self.tuning = tuning or TUNINGS['standard']
        self.config = config or FingeringConfig()
        
        # Pre-compute MIDI to fret options for efficiency
        self._fret_cache: Dict[int, List[Tuple[int, int]]] = {}
    
    def get_fret_options(self, midi_note: int) -> List[Tuple[int, int]]:
        """Get all possible (string, fret) combinations for a MIDI note."""
        if midi_note not in self._fret_cache:
            options = []
            for string_idx, open_note in enumerate(self.tuning):
                fret = midi_note - open_note
                if 0 <= fret <= NUM_FRETS:
                    options.append((string_idx, fret))
            self._fret_cache[midi_note] = options
        return self._fret_cache[midi_note]
    
    def calculate_single_note_cost(
        self,
        string: int,
        fret: int,
        prev_position: Optional[Tuple[int, int]],
        hand_position: Optional[HandPosition],
        time_delta: float = 0.0
    ) -> float:
        """
        Calculate the cost of playing a single note at (string, fret).
        
        Lower cost = better fingering.
        """
        cost = 0.0
        cfg = self.config
        
        # === Fret position preferences ===
        
        # Bonus for low frets (easier to play)
        if 1 <= fret <= 5:
            cost -= cfg.low_fret_bonus * (6 - fret)  # Max bonus at fret 1
        
        # Bonus for open strings
        if fret == 0:
            cost -= cfg.open_string_bonus
        
        # Penalty for high frets (harder to play)
        if fret > 12:
            cost += cfg.high_fret_penalty * (fret - 12)
        
        # === Hand position considerations ===
        
        if hand_position:
            distance = hand_position.distance_to(fret)
            if distance > 0:
                # Penalty for position shift
                cost += cfg.position_change_penalty * (distance / 4)  # Normalize
        
        # === String transition costs ===
        
        if prev_position:
            prev_string, prev_fret = prev_position
            string_distance = abs(string - prev_string)
            fret_distance = abs(fret - prev_fret) if fret > 0 and prev_fret > 0 else 0
            
            # Bonus for adjacent strings
            if string_distance == 1:
                cost -= cfg.adjacent_string_bonus
            elif string_distance > 1:
                # Penalty for skipping strings
                cost += cfg.string_skip_penalty * (string_distance - 1)
            
            # Same string penalty (need to lift finger)
            if string_distance == 0:
                cost += cfg.same_string_penalty
            
            # Large fret jump penalty
            if fret_distance > cfg.large_jump_threshold:
                excess = fret_distance - cfg.large_jump_threshold
                cost += cfg.large_jump_penalty * excess
            
            # Fast passage consideration
            if time_delta > 0 and time_delta < cfg.fast_passage_threshold:
                # Increase costs for difficult transitions in fast passages
                if string_distance > 1 or fret_distance > 3:
                    cost *= cfg.fast_passage_penalty_mult
        
        # === Technique considerations ===
        
        # Bends are easier on certain strings
        if string in cfg.bend_string_bonus:
            cost -= cfg.bend_string_bonus[string] * 0.5  # Small preference
        
        return cost
    
    def calculate_chord_cost(
        self,
        positions: List[Tuple[int, int]],
        prev_hand_position: Optional[HandPosition]
    ) -> float:
        """
        Calculate the cost of playing a chord (multiple simultaneous notes).
        """
        if not positions:
            return 0.0
        
        cost = 0.0
        cfg = self.config
        
        # Check fret span
        frets = [f for s, f in positions if f > 0]
        if frets:
            span = max(frets) - min(frets)
            if span > cfg.max_fret_span:
                # Penalty for impossible stretch
                cost += 10.0 * (span - cfg.max_fret_span)
        
        # Check for string conflicts (multiple notes on same string)
        strings = [s for s, f in positions]
        if len(strings) != len(set(strings)):
            cost += 100.0  # Impossible - can't play two notes on same string
        
        # Hand position change
        if prev_hand_position and frets:
            new_base = min(frets)
            distance = abs(new_base - prev_hand_position.base_fret)
            if distance > 0:
                cost += cfg.position_change_penalty * (distance / 4)
        
        # Bonus for common chord shapes
        cost += self._evaluate_chord_shape(positions)
        
        # Accumulate individual note costs
        for string, fret in positions:
            cost += self.calculate_single_note_cost(string, fret, None, prev_hand_position)
        
        return cost
    
    def _evaluate_chord_shape(self, positions: List[Tuple[int, int]]) -> float:
        """
        Evaluate how "natural" a chord shape is.
        
        Common shapes get bonuses, awkward shapes get penalties.
        """
        if len(positions) < 2:
            return 0.0
        
        cost = 0.0
        
        # Sort by string
        sorted_pos = sorted(positions, key=lambda x: x[0])
        
        # Check for barre chord potential
        frets = [f for s, f in sorted_pos if f > 0]
        if frets:
            min_fret = min(frets)
            # Count how many strings could be barred at min_fret
            barre_count = sum(1 for s, f in sorted_pos if f == min_fret)
            if barre_count >= 2:
                cost -= 0.5  # Barre chords are efficient
        
        # Penalty for widely spaced strings with frets
        for i in range(len(sorted_pos) - 1):
            s1, f1 = sorted_pos[i]
            s2, f2 = sorted_pos[i + 1]
            
            # Check for awkward string gaps
            if s2 - s1 > 2 and f1 > 0 and f2 > 0:
                cost += 0.5 * (s2 - s1 - 1)
        
        return cost
    
    def optimize_sequence(
        self,
        notes: List[Note],
        verbose: bool = False
    ) -> List[TabNote]:
        """
        Optimize fingering for a sequence of notes using dynamic programming.
        
        Returns optimized list of TabNote objects.
        """
        if not notes:
            return []
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Group simultaneous notes (chords)
        note_groups = self._group_simultaneous_notes(sorted_notes)
        
        if verbose:
            print(f"\n🎸 Fingering Optimizer")
            print(f"   Notes: {len(sorted_notes)}")
            print(f"   Groups (chords/single): {len(note_groups)}")
        
        # Dynamic programming: find optimal path through all fingering options
        tab_notes = self._dp_optimize(note_groups, verbose)
        
        return tab_notes
    
    def _group_simultaneous_notes(
        self,
        notes: List[Note]
    ) -> List[Tuple[float, List[Note]]]:
        """Group notes that play simultaneously into chords."""
        if not notes:
            return []
        
        groups = []
        current_time = notes[0].start_time
        current_group = [notes[0]]
        
        for note in notes[1:]:
            if note.start_time - current_time <= self.config.chord_window:
                current_group.append(note)
            else:
                groups.append((current_time, current_group))
                current_time = note.start_time
                current_group = [note]
        
        groups.append((current_time, current_group))
        return groups
    
    def _generate_chord_candidates(
        self,
        notes: List[Note]
    ) -> List[FingeringCandidate]:
        """Generate all valid fingering candidates for a chord/note group."""
        if len(notes) == 1:
            # Single note - return all options
            options = self.get_fret_options(notes[0].midi)
            return [FingeringCandidate(positions=[(s, f)]) for s, f in options]
        
        # Multiple notes - find compatible combinations
        all_options = [self.get_fret_options(note.midi) for note in notes]
        
        # Generate all combinations
        def generate_combos(idx: int, current: List[Tuple[int, int]], used_strings: Set[int]):
            if idx == len(all_options):
                yield current.copy()
                return
            
            for string, fret in all_options[idx]:
                if string not in used_strings:
                    current.append((string, fret))
                    used_strings.add(string)
                    yield from generate_combos(idx + 1, current, used_strings)
                    current.pop()
                    used_strings.remove(string)
        
        candidates = []
        for combo in generate_combos(0, [], set()):
            # Check if physically playable
            frets = [f for s, f in combo if f > 0]
            if frets:
                span = max(frets) - min(frets)
                if span <= self.config.max_fret_span:
                    candidates.append(FingeringCandidate(positions=combo))
        
        # If no valid combinations, try to find closest approximations
        if not candidates and all_options:
            # Fall back to first available option for each note
            fallback = []
            used = set()
            for options in all_options:
                for s, f in options:
                    if s not in used:
                        fallback.append((s, f))
                        used.add(s)
                        break
            if fallback:
                candidates.append(FingeringCandidate(positions=fallback))
        
        return candidates
    
    def _dp_optimize(
        self,
        note_groups: List[Tuple[float, List[Note]]],
        verbose: bool = False
    ) -> List[TabNote]:
        """
        Use dynamic programming to find optimal fingering sequence.
        
        State: (group_index, candidate_index)
        Cost: accumulated cost to reach this state
        """
        if not note_groups:
            return []
        
        n_groups = len(note_groups)
        
        # Generate candidates for each group
        group_candidates = []
        for time, notes in note_groups:
            candidates = self._generate_chord_candidates(notes)
            if not candidates:
                # Shouldn't happen, but handle gracefully
                candidates = [FingeringCandidate(positions=[])]
            group_candidates.append(candidates)
        
        # DP arrays
        # cost[i][j] = minimum cost to reach group i, candidate j
        # parent[i][j] = (prev_candidate_idx) that led to minimum cost
        
        cost = [[float('inf')] * len(candidates) for candidates in group_candidates]
        parent = [[-1] * len(candidates) for candidates in group_candidates]
        
        # Initialize first group
        for j, cand in enumerate(group_candidates[0]):
            cost[0][j] = self.calculate_chord_cost(cand.positions, None)
        
        # Fill DP table
        for i in range(1, n_groups):
            time_delta = note_groups[i][0] - note_groups[i-1][0]
            
            for j, curr_cand in enumerate(group_candidates[i]):
                for k, prev_cand in enumerate(group_candidates[i-1]):
                    # Calculate transition cost
                    trans_cost = self._calculate_transition_cost(
                        prev_cand, curr_cand, time_delta
                    )
                    
                    # Add look-ahead cost
                    lookahead_cost = 0.0
                    if i + 1 < n_groups and self.config.lookahead_notes > 0:
                        lookahead_cost = self._calculate_lookahead_cost(
                            curr_cand, note_groups, i + 1
                        )
                    
                    total = cost[i-1][k] + trans_cost + lookahead_cost
                    
                    if total < cost[i][j]:
                        cost[i][j] = total
                        parent[i][j] = k
        
        # Backtrack to find optimal path
        best_final = min(range(len(group_candidates[-1])), key=lambda j: cost[-1][j])
        
        path = []
        j = best_final
        for i in range(n_groups - 1, -1, -1):
            path.append((i, j))
            if i > 0:
                j = parent[i][j]
        path.reverse()
        
        # Convert path to TabNotes
        tab_notes = []
        for group_idx, cand_idx in path:
            time, notes = note_groups[group_idx]
            candidate = group_candidates[group_idx][cand_idx]
            
            for (string, fret), note in zip(candidate.positions, notes):
                tab_notes.append(TabNote(
                    string=string,
                    fret=fret,
                    start_time=note.start_time,
                    duration=note.duration
                ))
        
        if verbose:
            final_cost = cost[-1][best_final]
            print(f"   Optimization cost: {final_cost:.2f}")
            
            # Analyze the result
            fret_jumps = []
            prev_fret = None
            for tn in tab_notes:
                if prev_fret is not None and tn.fret > 0:
                    fret_jumps.append(abs(tn.fret - prev_fret))
                if tn.fret > 0:
                    prev_fret = tn.fret
            
            if fret_jumps:
                print(f"   Avg fret jump: {np.mean(fret_jumps):.1f}")
                print(f"   Max fret jump: {max(fret_jumps)}")
            
            open_count = sum(1 for tn in tab_notes if tn.fret == 0)
            print(f"   Open strings used: {open_count}")
        
        return tab_notes
    
    def _calculate_transition_cost(
        self,
        prev_cand: FingeringCandidate,
        curr_cand: FingeringCandidate,
        time_delta: float
    ) -> float:
        """Calculate cost of transitioning between two fingering candidates."""
        cost = 0.0
        
        # Base chord cost
        prev_hand = None
        if prev_cand.positions:
            frets = [f for s, f in prev_cand.positions if f > 0]
            if frets:
                prev_hand = HandPosition(base_fret=min(frets))
        
        cost += self.calculate_chord_cost(curr_cand.positions, prev_hand)
        
        # Additional transition costs
        if prev_cand.positions and curr_cand.positions:
            prev_pos = prev_cand.positions[-1]  # Last note of previous
            
            for pos in curr_cand.positions:
                cost += self.calculate_single_note_cost(
                    pos[0], pos[1], prev_pos, prev_hand, time_delta
                ) * 0.5  # Reduce double-counting
        
        return cost
    
    def _calculate_lookahead_cost(
        self,
        curr_cand: FingeringCandidate,
        note_groups: List[Tuple[float, List[Note]]],
        start_idx: int
    ) -> float:
        """
        Calculate estimated cost for upcoming notes.
        
        This helps choose positions that will be good for future notes too.
        """
        cost = 0.0
        decay = 1.0
        
        curr_hand = None
        if curr_cand.positions:
            frets = [f for s, f in curr_cand.positions if f > 0]
            if frets:
                curr_hand = HandPosition(base_fret=min(frets))
        
        for i in range(min(self.config.lookahead_notes, len(note_groups) - start_idx)):
            decay *= self.config.lookahead_decay
            
            _, notes = note_groups[start_idx + i]
            
            # Find best candidate for this future group given current position
            best_future_cost = float('inf')
            for note in notes:
                options = self.get_fret_options(note.midi)
                for string, fret in options:
                    note_cost = self.calculate_single_note_cost(
                        string, fret, 
                        curr_cand.positions[-1] if curr_cand.positions else None,
                        curr_hand
                    )
                    best_future_cost = min(best_future_cost, note_cost)
            
            if best_future_cost < float('inf'):
                cost += decay * best_future_cost
        
        return cost


def optimize_fingering(
    notes: List[Note],
    tuning: List[int] = None,
    config: FingeringConfig = None,
    verbose: bool = True
) -> List[TabNote]:
    """
    Convenience function to optimize fingering for a list of notes.
    
    Args:
        notes: List of Note objects to optimize
        tuning: Guitar tuning as MIDI note numbers
        config: Fingering configuration
        verbose: Print optimization statistics
        
    Returns:
        List of optimized TabNote objects
    """
    optimizer = FingeringOptimizer(tuning=tuning, config=config)
    return optimizer.optimize_sequence(notes, verbose=verbose)


def compare_fingerings(
    notes: List[Note],
    original_tabs: List[TabNote],
    optimized_tabs: List[TabNote],
    tuning: List[int] = None
) -> Dict[str, float]:
    """
    Compare original and optimized fingerings with metrics.
    
    Returns dict with comparison statistics.
    """
    def calc_metrics(tabs: List[TabNote]) -> Dict[str, float]:
        if not tabs:
            return {}
        
        frets = [t.fret for t in tabs if t.fret > 0]
        fret_jumps = []
        string_jumps = []
        prev = None
        for t in tabs:
            if prev:
                if t.fret > 0 and prev.fret > 0:
                    fret_jumps.append(abs(t.fret - prev.fret))
                string_jumps.append(abs(t.string - prev.string))
            prev = t
        
        return {
            'avg_fret': np.mean(frets) if frets else 0,
            'max_fret': max(frets) if frets else 0,
            'open_strings': sum(1 for t in tabs if t.fret == 0),
            'avg_fret_jump': np.mean(fret_jumps) if fret_jumps else 0,
            'max_fret_jump': max(fret_jumps) if fret_jumps else 0,
            'avg_string_jump': np.mean(string_jumps) if string_jumps else 0,
        }
    
    original_metrics = calc_metrics(original_tabs)
    optimized_metrics = calc_metrics(optimized_tabs)
    
    return {
        'original': original_metrics,
        'optimized': optimized_metrics,
        'improvement': {
            k: original_metrics.get(k, 0) - optimized_metrics.get(k, 0)
            for k in original_metrics
        }
    }


# CLI for testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fingering optimizer')
    parser.add_argument('--test', action='store_true', help='Run test with sample notes')
    args = parser.parse_args()
    
    if args.test:
        # Create some test notes (C major scale)
        test_notes = [
            Note(midi=60, start_time=0.0, duration=0.5, confidence=0.9),  # C4
            Note(midi=62, start_time=0.5, duration=0.5, confidence=0.9),  # D4
            Note(midi=64, start_time=1.0, duration=0.5, confidence=0.9),  # E4
            Note(midi=65, start_time=1.5, duration=0.5, confidence=0.9),  # F4
            Note(midi=67, start_time=2.0, duration=0.5, confidence=0.9),  # G4
            Note(midi=69, start_time=2.5, duration=0.5, confidence=0.9),  # A4
            Note(midi=71, start_time=3.0, duration=0.5, confidence=0.9),  # B4
            Note(midi=72, start_time=3.5, duration=0.5, confidence=0.9),  # C5
        ]
        
        print("Testing fingering optimizer with C major scale...")
        tabs = optimize_fingering(test_notes, verbose=True)
        
        print("\nOptimized fingering:")
        for tab in tabs:
            print(f"  String {tab.string} Fret {tab.fret} at {tab.start_time:.2f}s")
