"""
Probabilistic Note Model with Hidden Markov Model (HMM)

This module provides a mathematically rigorous approach to note sequence
transcription using probabilistic modeling:

1. Prior Model: P(note | key, scale) - probability distribution based on music theory
2. Observation Model: P(observed_freq | true_note) - handles pitch detection uncertainty  
3. Transition Model: P(note_t | note_{t-1}) - note-to-note transition probabilities
4. Viterbi Algorithm: Find most likely note sequence given observations

This handles uncertainty much better than simple pitch snapping.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import librosa

# Import from existing modules
from music_theory import (
    detect_key, Key, SCALE_INTERVALS, NOTE_NAMES, MAJOR_PROFILE, MINOR_PROFILE
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HMMConfig:
    """Configuration for probabilistic HMM model."""
    
    # Note range (MIDI)
    min_midi: int = 40   # E2 (low E on guitar)
    max_midi: int = 88   # E6 (high notes)
    
    # Observation model parameters
    freq_std_cents: float = 25.0    # Standard deviation in cents for freq uncertainty
    octave_error_prob: float = 0.15  # Probability of octave errors in detection
    
    # Prior model parameters
    scale_note_boost: float = 3.0    # Multiplier for in-scale notes
    root_note_boost: float = 1.5     # Additional multiplier for root note
    fifth_note_boost: float = 1.2    # Additional multiplier for fifth
    
    # Transition model parameters
    step_preference: float = 0.6     # Preference for stepwise motion (±2 semitones)
    leap_penalty: float = 0.1        # Penalty per semitone for large leaps
    max_leap: int = 12               # Maximum interval in semitones (octave)
    repeat_probability: float = 0.3  # Probability of repeating same note
    
    # Viterbi parameters
    beam_width: int = 50             # Number of candidates to keep per frame (0=no pruning)
    
    # Smoothing
    transition_smoothing: float = 0.01  # Laplace smoothing for transitions
    
    verbose: bool = True


# =============================================================================
# Prior Model: P(note | key, scale)
# =============================================================================

class NotePrior:
    """
    Prior probability distribution over notes based on key and scale.
    
    P(note | key) is higher for:
    - Notes in the scale
    - Root, third, fifth (chord tones)
    - Notes that appear frequently in the key profile
    """
    
    def __init__(self, key: Key, config: HMMConfig):
        self.key = key
        self.config = config
        self.priors = self._build_prior_distribution()
    
    def _build_prior_distribution(self) -> Dict[int, float]:
        """Build prior probability for each MIDI note."""
        priors = {}
        
        # Get scale notes
        scale_type = 'major' if self.key.mode == 'major' else 'natural_minor'
        scale_intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS['major'])
        scale_pcs = set((self.key.root + i) % 12 for i in scale_intervals)
        
        # Use Krumhansl profile as base
        if self.key.mode == 'major':
            profile = MAJOR_PROFILE / MAJOR_PROFILE.sum()
        else:
            profile = MINOR_PROFILE / MINOR_PROFILE.sum()
        
        # Build prior for each MIDI note
        for midi in range(self.config.min_midi, self.config.max_midi + 1):
            pc = midi % 12
            rotated_pc = (pc - self.key.root) % 12  # Relative to key root
            
            # Base probability from profile
            base_prob = profile[rotated_pc]
            
            # Boost for scale notes
            if pc in scale_pcs:
                base_prob *= self.config.scale_note_boost
            
            # Extra boost for root
            if pc == self.key.root:
                base_prob *= self.config.root_note_boost
            
            # Extra boost for fifth
            fifth_pc = (self.key.root + 7) % 12
            if pc == fifth_pc:
                base_prob *= self.config.fifth_note_boost
            
            priors[midi] = base_prob
        
        # Normalize
        total = sum(priors.values())
        for midi in priors:
            priors[midi] /= total
        
        return priors
    
    def get_prior(self, midi: int) -> float:
        """Get prior probability for a MIDI note."""
        return self.priors.get(midi, 1e-10)
    
    def get_log_prior(self, midi: int) -> float:
        """Get log prior probability for a MIDI note."""
        return np.log(self.get_prior(midi) + 1e-10)


# =============================================================================
# Transition Model: P(note_t | note_{t-1})
# =============================================================================

class NoteTransitionModel:
    """
    Transition probabilities between notes.
    
    Musical transitions favor:
    - Stepwise motion (intervals of 1-2 semitones)
    - Chord arpeggios (thirds, fifths)
    - Repeated notes
    
    The model is learned from data when available, or uses musical priors.
    """
    
    def __init__(self, key: Key, config: HMMConfig):
        self.key = key
        self.config = config
        self.min_midi = config.min_midi
        self.max_midi = config.max_midi
        
        # Transition counts (for learning)
        self._counts = defaultdict(lambda: defaultdict(float))
        self._learned = False
        
        # Pre-compute base transition matrix
        self._base_transitions = self._build_base_transitions()
    
    def _build_base_transitions(self) -> np.ndarray:
        """Build base transition probability matrix based on musical priors."""
        n_notes = self.max_midi - self.min_midi + 1
        trans = np.zeros((n_notes, n_notes))
        
        for i in range(n_notes):
            midi_from = i + self.min_midi
            
            for j in range(n_notes):
                midi_to = j + self.min_midi
                interval = abs(midi_to - midi_from)
                
                # Repeated note
                if interval == 0:
                    trans[i, j] = self.config.repeat_probability
                    continue
                
                # Stepwise motion (1-2 semitones) - highest probability
                if interval <= 2:
                    trans[i, j] = self.config.step_preference / 4
                
                # Small leap (3-4 semitones, thirds)
                elif interval <= 4:
                    trans[i, j] = 0.1
                
                # Medium leap (5-7 semitones, fourth/fifth)
                elif interval <= 7:
                    trans[i, j] = 0.05
                
                # Octave
                elif interval == 12:
                    trans[i, j] = 0.08
                
                # Large leap
                elif interval <= self.config.max_leap:
                    trans[i, j] = 0.02 * np.exp(-self.config.leap_penalty * (interval - 7))
                
                # Very large leap (rare)
                else:
                    trans[i, j] = 1e-4
            
            # Normalize row
            row_sum = trans[i].sum()
            if row_sum > 0:
                trans[i] /= row_sum
        
        return trans
    
    def learn_from_notes(self, notes: List) -> None:
        """
        Learn transition probabilities from a list of notes.
        
        Args:
            notes: List of Note objects with .midi attribute, sorted by time
        """
        if len(notes) < 2:
            return
        
        # Sort by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Count transitions
        for i in range(len(sorted_notes) - 1):
            from_midi = sorted_notes[i].midi
            to_midi = sorted_notes[i + 1].midi
            
            # Clamp to range
            from_midi = max(self.min_midi, min(self.max_midi, from_midi))
            to_midi = max(self.min_midi, min(self.max_midi, to_midi))
            
            self._counts[from_midi][to_midi] += 1
        
        self._learned = True
    
    def get_transition_prob(self, from_midi: int, to_midi: int) -> float:
        """Get transition probability P(to_midi | from_midi)."""
        # Clamp to range
        from_midi = max(self.min_midi, min(self.max_midi, from_midi))
        to_midi = max(self.min_midi, min(self.max_midi, to_midi))
        
        i = from_midi - self.min_midi
        j = to_midi - self.min_midi
        
        if self._learned:
            # Use learned + smoothing + base prior
            count = self._counts[from_midi][to_midi]
            total = sum(self._counts[from_midi].values())
            
            if total > 0:
                learned_prob = (count + self.config.transition_smoothing) / (total + self.config.transition_smoothing * (self.max_midi - self.min_midi + 1))
                # Mix with base
                return 0.5 * learned_prob + 0.5 * self._base_transitions[i, j]
        
        return self._base_transitions[i, j]
    
    def get_log_transition_prob(self, from_midi: int, to_midi: int) -> float:
        """Get log transition probability."""
        return np.log(self.get_transition_prob(from_midi, to_midi) + 1e-10)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get full transition probability matrix."""
        n_notes = self.max_midi - self.min_midi + 1
        matrix = np.zeros((n_notes, n_notes))
        
        for i in range(n_notes):
            for j in range(n_notes):
                matrix[i, j] = self.get_transition_prob(
                    i + self.min_midi,
                    j + self.min_midi
                )
        
        return matrix


# =============================================================================
# Observation Model: P(observed_freq | true_note)
# =============================================================================

class ObservationModel:
    """
    Observation probability model for pitch detection.
    
    P(observed_freq | true_note) models the uncertainty in pitch detection:
    - Frequency measurement noise (Gaussian in cents)
    - Octave errors (common in pitch detection)
    - Harmonic confusion
    """
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.freq_std_cents = config.freq_std_cents
        self.octave_error_prob = config.octave_error_prob
    
    def _freq_to_midi(self, freq: float) -> float:
        """Convert frequency to continuous MIDI value."""
        if freq <= 0:
            return 0
        return 69 + 12 * np.log2(freq / 440.0)
    
    def _cents_diff(self, freq_observed: float, midi_true: int) -> float:
        """Calculate difference in cents between observed freq and true MIDI."""
        if freq_observed <= 0:
            return float('inf')
        
        midi_observed = self._freq_to_midi(freq_observed)
        return 100 * (midi_observed - midi_true)  # 100 cents per semitone
    
    def get_observation_prob(
        self,
        freq_observed: float,
        midi_true: int,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate P(observed_freq | true_midi).
        
        Args:
            freq_observed: Observed frequency in Hz
            midi_true: Hypothesized true MIDI note
            confidence: Pitch detection confidence (0-1)
            
        Returns:
            Observation probability
        """
        if freq_observed <= 0:
            return 1e-10
        
        # Scale uncertainty by confidence (lower confidence = higher uncertainty)
        adjusted_std = self.freq_std_cents / (confidence + 0.1)
        
        # Calculate cents difference
        cents_diff = self._cents_diff(freq_observed, midi_true)
        
        # Gaussian probability for in-tune detection
        gaussian_prob = np.exp(-0.5 * (cents_diff / adjusted_std) ** 2)
        
        # Account for octave errors (±12 semitones)
        octave_up_diff = self._cents_diff(freq_observed, midi_true + 12)
        octave_down_diff = self._cents_diff(freq_observed, midi_true - 12)
        
        octave_prob = (
            self.octave_error_prob * 0.5 * np.exp(-0.5 * (octave_up_diff / adjusted_std) ** 2) +
            self.octave_error_prob * 0.5 * np.exp(-0.5 * (octave_down_diff / adjusted_std) ** 2)
        )
        
        # Combined probability
        total_prob = (1 - self.octave_error_prob) * gaussian_prob + octave_prob
        
        return max(total_prob, 1e-10)
    
    def get_log_observation_prob(
        self,
        freq_observed: float,
        midi_true: int,
        confidence: float = 1.0
    ) -> float:
        """Get log observation probability."""
        return np.log(self.get_observation_prob(freq_observed, midi_true, confidence))


# =============================================================================
# Hidden Markov Model
# =============================================================================

@dataclass
class PitchObservation:
    """A single pitch observation."""
    time: float
    frequency: float  # Hz (0 = unvoiced/silence)
    confidence: float
    
    @property
    def midi_estimate(self) -> int:
        """Get rough MIDI estimate from frequency."""
        if self.frequency <= 0:
            return 0
        return int(round(69 + 12 * np.log2(self.frequency / 440.0)))


@dataclass
class DecodedNote:
    """A decoded note from HMM."""
    midi: int
    start_time: float
    end_time: float
    confidence: float
    raw_observations: List[PitchObservation] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def frequency(self) -> float:
        return librosa.midi_to_hz(self.midi)
    
    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.midi % 12] + str(self.midi // 12 - 1)


class PitchHMM:
    """
    Hidden Markov Model for pitch sequence decoding.
    
    States: MIDI notes in the guitar range
    Observations: Detected frequencies (with uncertainty)
    
    Uses Viterbi algorithm to find most likely note sequence.
    """
    
    def __init__(
        self,
        key: Optional[Key] = None,
        config: Optional[HMMConfig] = None
    ):
        self.config = config or HMMConfig()
        self.key = key
        
        # Models (initialized when key is set)
        self.prior_model: Optional[NotePrior] = None
        self.transition_model: Optional[NoteTransitionModel] = None
        self.observation_model = ObservationModel(self.config)
        
        if key:
            self._setup_models()
    
    def _setup_models(self) -> None:
        """Initialize probability models."""
        if self.key:
            self.prior_model = NotePrior(self.key, self.config)
            self.transition_model = NoteTransitionModel(self.key, self.config)
    
    def set_key(self, key: Key) -> None:
        """Set the musical key and reinitialize models."""
        self.key = key
        self._setup_models()
    
    def detect_key_from_observations(
        self,
        observations: List[PitchObservation]
    ) -> Key:
        """
        Detect musical key from pitch observations.
        
        Uses pitch class histogram with Krumhansl-Schmuckler algorithm.
        """
        # Build pitch class histogram weighted by confidence
        pc_histogram = np.zeros(12)
        
        for obs in observations:
            if obs.frequency > 0 and obs.confidence > 0.3:
                midi = obs.midi_estimate
                pc = midi % 12
                pc_histogram[pc] += obs.confidence
        
        # Normalize
        if pc_histogram.sum() > 0:
            pc_histogram = pc_histogram / pc_histogram.sum()
        
        # Correlate with key profiles
        best_key = None
        best_corr = -2
        
        for root in range(12):
            rotated = np.roll(pc_histogram, -root)
            
            # Test major
            major_corr = np.corrcoef(rotated, MAJOR_PROFILE / MAJOR_PROFILE.sum())[0, 1]
            if not np.isnan(major_corr) and major_corr > best_corr:
                best_corr = major_corr
                best_key = Key(root, 'major', float(major_corr))
            
            # Test minor
            minor_corr = np.corrcoef(rotated, MINOR_PROFILE / MINOR_PROFILE.sum())[0, 1]
            if not np.isnan(minor_corr) and minor_corr > best_corr:
                best_corr = minor_corr
                best_key = Key(root, 'minor', float(minor_corr))
        
        return best_key if best_key else Key(0, 'major', 0.0)
    
    def viterbi_decode(
        self,
        observations: List[PitchObservation]
    ) -> Tuple[List[int], float]:
        """
        Viterbi algorithm to find most likely note sequence.
        
        Args:
            observations: List of PitchObservation objects
            
        Returns:
            Tuple of (best_path, log_probability)
        """
        if not observations:
            return [], 0.0
        
        if not self.key:
            # Auto-detect key
            self.set_key(self.detect_key_from_observations(observations))
            if self.config.verbose:
                print(f"🔑 Auto-detected key: {self.key.name} (confidence: {self.key.confidence:.2f})")
        
        n_obs = len(observations)
        min_midi = self.config.min_midi
        max_midi = self.config.max_midi
        n_states = max_midi - min_midi + 1
        
        # Log probabilities (avoid underflow)
        # V[t, s] = log P(best path to state s at time t)
        V = np.full((n_obs, n_states), -np.inf)
        
        # Backpointers for path reconstruction
        backpointer = np.zeros((n_obs, n_states), dtype=int)
        
        # Initialize first observation
        obs = observations[0]
        for s in range(n_states):
            midi = s + min_midi
            log_prior = self.prior_model.get_log_prior(midi)
            log_obs = self.observation_model.get_log_observation_prob(
                obs.frequency, midi, obs.confidence
            )
            V[0, s] = log_prior + log_obs
        
        # Beam pruning for first step
        if self.config.beam_width > 0:
            threshold_idx = min(self.config.beam_width, n_states) - 1
            sorted_vals = np.sort(V[0])[::-1]
            threshold = sorted_vals[threshold_idx] if threshold_idx < len(sorted_vals) else -np.inf
            V[0, V[0] < threshold] = -np.inf
        
        # Forward pass with Viterbi
        for t in range(1, n_obs):
            obs = observations[t]
            
            for s in range(n_states):
                midi = s + min_midi
                
                # Observation probability
                log_obs = self.observation_model.get_log_observation_prob(
                    obs.frequency, midi, obs.confidence
                )
                
                # Find best previous state
                best_prev = -1
                best_val = -np.inf
                
                for prev_s in range(n_states):
                    if V[t-1, prev_s] == -np.inf:
                        continue
                    
                    prev_midi = prev_s + min_midi
                    log_trans = self.transition_model.get_log_transition_prob(prev_midi, midi)
                    val = V[t-1, prev_s] + log_trans
                    
                    if val > best_val:
                        best_val = val
                        best_prev = prev_s
                
                if best_prev >= 0:
                    V[t, s] = best_val + log_obs
                    backpointer[t, s] = best_prev
            
            # Beam pruning
            if self.config.beam_width > 0:
                valid_vals = V[t, V[t] > -np.inf]
                if len(valid_vals) > self.config.beam_width:
                    sorted_vals = np.sort(valid_vals)[::-1]
                    threshold = sorted_vals[self.config.beam_width - 1]
                    V[t, V[t] < threshold] = -np.inf
        
        # Backtrack to find best path
        best_final_state = np.argmax(V[n_obs - 1])
        best_log_prob = V[n_obs - 1, best_final_state]
        
        path = [best_final_state + min_midi]
        s = best_final_state
        
        for t in range(n_obs - 1, 0, -1):
            s = backpointer[t, s]
            path.append(s + min_midi)
        
        path.reverse()
        
        return path, best_log_prob
    
    def decode_to_notes(
        self,
        observations: List[PitchObservation],
        min_note_duration: float = 0.05,
        merge_threshold_cents: float = 50
    ) -> List[DecodedNote]:
        """
        Decode observations to note events with timing.
        
        Runs Viterbi, then merges consecutive same-notes into note events.
        
        Args:
            observations: List of PitchObservation objects
            min_note_duration: Minimum note duration in seconds
            merge_threshold_cents: Threshold for merging consecutive frames
            
        Returns:
            List of DecodedNote objects
        """
        if not observations:
            return []
        
        # Run Viterbi
        path, log_prob = self.viterbi_decode(observations)
        
        if self.config.verbose:
            print(f"🎯 Viterbi decoding complete (log prob: {log_prob:.2f})")
        
        # Group consecutive same-MIDI values into notes
        notes = []
        current_midi = path[0]
        current_start_idx = 0
        current_obs = [observations[0]]
        
        for i in range(1, len(path)):
            if path[i] == current_midi:
                current_obs.append(observations[i])
            else:
                # End current note
                start_time = observations[current_start_idx].time
                end_time = observations[i - 1].time
                duration = end_time - start_time
                
                if duration >= min_note_duration:
                    avg_confidence = np.mean([o.confidence for o in current_obs])
                    notes.append(DecodedNote(
                        midi=current_midi,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=avg_confidence,
                        raw_observations=current_obs
                    ))
                
                # Start new note
                current_midi = path[i]
                current_start_idx = i
                current_obs = [observations[i]]
        
        # Handle last note
        start_time = observations[current_start_idx].time
        end_time = observations[-1].time
        duration = end_time - start_time
        
        if duration >= min_note_duration:
            avg_confidence = np.mean([o.confidence for o in current_obs])
            notes.append(DecodedNote(
                midi=current_midi,
                start_time=start_time,
                end_time=end_time,
                confidence=avg_confidence,
                raw_observations=current_obs
            ))
        
        if self.config.verbose:
            print(f"📝 Decoded {len(notes)} notes from {len(observations)} observations")
        
        return notes


# =============================================================================
# Integration with Existing Pipeline
# =============================================================================

def create_observations_from_pitch_track(
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray
) -> List[PitchObservation]:
    """
    Create PitchObservation list from pitch tracking arrays.
    
    Args:
        times: Time array in seconds
        f0: Fundamental frequency array in Hz (0 = unvoiced)
        confidence: Confidence array (0-1)
        
    Returns:
        List of PitchObservation objects
    """
    observations = []
    for t, freq, conf in zip(times, f0, confidence):
        observations.append(PitchObservation(
            time=float(t),
            frequency=float(freq),
            confidence=float(conf)
        ))
    return observations


def hmm_decode_pitch_track(
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray,
    key: Optional[Key] = None,
    config: Optional[HMMConfig] = None
) -> Tuple[List[DecodedNote], Key]:
    """
    Apply HMM decoding to a pitch track.
    
    This is the main entry point for integrating HMM into the pipeline.
    
    Args:
        times: Time array in seconds
        f0: Fundamental frequency array in Hz
        confidence: Confidence array
        key: Optional key (auto-detected if None)
        config: HMM configuration
        
    Returns:
        Tuple of (decoded_notes, detected_key)
    """
    config = config or HMMConfig()
    
    # Create observations
    observations = create_observations_from_pitch_track(times, f0, confidence)
    
    # Filter out silent frames (use low threshold for distorted audio)
    voiced_obs = [o for o in observations if o.frequency > 0 and o.confidence > 0.05]
    
    if not voiced_obs:
        return [], Key(0, 'major', 0.0)
    
    # Create HMM
    hmm = PitchHMM(key=key, config=config)
    
    # Decode
    notes = hmm.decode_to_notes(voiced_obs)
    
    return notes, hmm.key


def refine_notes_with_hmm(
    notes: List,
    key: Optional[Key] = None,
    config: Optional[HMMConfig] = None
) -> List:
    """
    Refine already-detected notes using HMM transition model.
    
    This re-decodes notes that may have been detected incorrectly,
    using transition probabilities to fix implausible sequences.
    
    Args:
        notes: List of Note objects with .midi, .start_time, .duration, .confidence
        key: Optional key
        config: HMM configuration
        
    Returns:
        List of refined notes
    """
    from copy import deepcopy
    
    if not notes:
        return notes
    
    config = config or HMMConfig()
    
    # Create pseudo-observations from notes
    observations = []
    for note in notes:
        freq = librosa.midi_to_hz(note.midi)
        conf = getattr(note, 'confidence', 0.8)
        observations.append(PitchObservation(
            time=note.start_time,
            frequency=freq,
            confidence=conf
        ))
    
    # Create HMM
    hmm = PitchHMM(key=key, config=config)
    
    # Run Viterbi
    path, _ = hmm.viterbi_decode(observations)
    
    # Update notes with refined MIDI values
    refined = []
    for i, note in enumerate(notes):
        new_note = deepcopy(note)
        new_note.midi = path[i]
        refined.append(new_note)
    
    # Count changes
    changes = sum(1 for orig, ref in zip(notes, refined) if orig.midi != ref.midi)
    if config.verbose and changes > 0:
        print(f"🔧 HMM refinement changed {changes}/{len(notes)} notes")
    
    return refined


# =============================================================================
# Analysis and Diagnostics
# =============================================================================

def analyze_transition_statistics(notes: List) -> Dict[str, Any]:
    """
    Analyze note transition patterns in a note list.
    
    Returns statistics about intervals, stepwise motion, etc.
    """
    if len(notes) < 2:
        return {'error': 'Not enough notes'}
    
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    intervals = []
    for i in range(len(sorted_notes) - 1):
        interval = sorted_notes[i + 1].midi - sorted_notes[i].midi
        intervals.append(interval)
    
    intervals = np.array(intervals)
    abs_intervals = np.abs(intervals)
    
    stats = {
        'total_transitions': len(intervals),
        'mean_interval': float(np.mean(abs_intervals)),
        'std_interval': float(np.std(abs_intervals)),
        'max_interval': int(np.max(abs_intervals)),
        'stepwise_ratio': float(np.mean(abs_intervals <= 2)),
        'leap_ratio': float(np.mean(abs_intervals > 4)),
        'repeat_ratio': float(np.mean(intervals == 0)),
        'ascending_ratio': float(np.mean(intervals > 0)),
        'descending_ratio': float(np.mean(intervals < 0)),
        'interval_histogram': {
            int(i): int(np.sum(abs_intervals == i))
            for i in range(13)
        }
    }
    
    return stats


def print_hmm_diagnostics(hmm: PitchHMM, observations: List[PitchObservation]) -> None:
    """Print diagnostic information about the HMM."""
    print("\n" + "=" * 60)
    print("HMM DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\nKey: {hmm.key.name if hmm.key else 'Not set'}")
    print(f"Key confidence: {hmm.key.confidence:.2f}" if hmm.key else "")
    
    print(f"\nObservations: {len(observations)}")
    voiced = [o for o in observations if o.frequency > 0]
    print(f"Voiced frames: {len(voiced)} ({100*len(voiced)/len(observations):.1f}%)")
    
    if voiced:
        avg_conf = np.mean([o.confidence for o in voiced])
        print(f"Average confidence: {avg_conf:.2f}")
    
    print(f"\nConfig:")
    print(f"  - MIDI range: {hmm.config.min_midi}-{hmm.config.max_midi}")
    print(f"  - Freq std (cents): {hmm.config.freq_std_cents}")
    print(f"  - Octave error prob: {hmm.config.octave_error_prob}")
    print(f"  - Beam width: {hmm.config.beam_width}")
    
    print("=" * 60 + "\n")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Test HMM with sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HMM pitch decoding')
    parser.add_argument('--test', action='store_true', help='Run test with synthetic data')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Testing HMM with synthetic data...")
        
        # Create synthetic observations (C major scale)
        config = HMMConfig(verbose=True)
        key = Key(0, 'major', 1.0)  # C major
        
        # C4-D4-E4-F4-G4 sequence with some noise
        midi_sequence = [60, 62, 64, 65, 67]
        observations = []
        
        for i, midi in enumerate(midi_sequence):
            true_freq = librosa.midi_to_hz(midi)
            # Add some noise
            noisy_freq = true_freq * (1 + np.random.normal(0, 0.01))
            observations.append(PitchObservation(
                time=i * 0.5,
                frequency=noisy_freq,
                confidence=0.9
            ))
        
        # Add an octave error
        observations[2] = PitchObservation(
            time=1.0,
            frequency=librosa.midi_to_hz(76),  # E5 instead of E4
            confidence=0.7
        )
        
        # Decode
        hmm = PitchHMM(key=key, config=config)
        notes = hmm.decode_to_notes(observations)
        
        print("\nDecoded notes:")
        for note in notes:
            print(f"  {note.note_name} (MIDI {note.midi}) @ {note.start_time:.2f}s")
        
        print("\nExpected: C4, D4, E4, F4, G4")
        print("Note: E4 should be recovered despite octave error in observation")


if __name__ == '__main__':
    main()
