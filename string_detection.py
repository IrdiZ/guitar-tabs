#!/usr/bin/env python3
"""
String Detection Module for Guitar Tab Generation

Uses timbre/spectral analysis to detect which string a note was played on.
Different strings have characteristic timbral qualities:
- Lower strings (E, A, D): warmer, more bass, lower spectral centroid
- Higher strings (G, B, e): brighter, more treble, higher spectral centroid

The same pitch can be played on different strings at different frets,
and each position has a distinct timbral fingerprint.

Key features used:
1. Spectral centroid - measure of "brightness"
2. Spectral bandwidth - frequency spread
3. Spectral rolloff - frequency below which most energy lies
4. MFCCs - timbral texture
5. Zero-crossing rate - related to noisiness/brightness
6. Spectral contrast - valley-to-peak ratios per octave band
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler

# Guitar tunings (MIDI note numbers, low to high: E2 A2 D3 G3 B3 E4)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']

# Typical spectral centroids by string (rough ranges in Hz for open strings)
# These are calibrated for standard steel acoustic guitar
# Lower strings have lower centroids, higher strings have higher centroids
STRING_CENTROID_RANGES = {
    0: (200, 600),    # Low E - very warm
    1: (300, 800),    # A - warm
    2: (400, 1000),   # D - neutral-warm
    3: (600, 1400),   # G - neutral-bright
    4: (800, 1800),   # B - bright
    5: (1000, 2500),  # High e - very bright
}

# Expected spectral bandwidth ranges by string
STRING_BANDWIDTH_RANGES = {
    0: (100, 400),    # Low E - narrow, focused bass
    1: (150, 500),    # A
    2: (200, 600),    # D
    3: (300, 800),    # G
    4: (400, 1000),   # B
    5: (500, 1200),   # High e - wider bandwidth
}


@dataclass
class SpectralFeatures:
    """Spectral features extracted from an audio segment."""
    centroid: float          # Mean spectral centroid
    bandwidth: float         # Mean spectral bandwidth
    rolloff: float           # Mean spectral rolloff (85%)
    zcr: float               # Mean zero-crossing rate
    flatness: float          # Mean spectral flatness
    contrast: np.ndarray     # Mean spectral contrast per band
    mfcc: np.ndarray         # Mean MFCCs (13 coefficients)
    attack_brightness: float # Centroid during attack phase
    sustain_brightness: float # Centroid during sustain phase
    

@dataclass
class StringPrediction:
    """Prediction of which string a note was played on."""
    string: int              # 0-5 (low E to high e)
    fret: int                # 0-24
    confidence: float        # 0-1
    method: str              # Which method made the prediction
    features: Optional[SpectralFeatures] = None
    alternative_positions: List[Tuple[int, int, float]] = field(default_factory=list)


class StringDetector:
    """
    Detects which guitar string a note was played on using spectral analysis.
    
    The detector uses multiple features:
    1. Spectral centroid correlation with expected string brightness
    2. Attack transient analysis (picked notes vs fretted notes sound different)
    3. Harmonic content analysis (different strings emphasize different harmonics)
    4. Machine learning classifier (if training data is available)
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        tuning: List[int] = None,
        # Feature weights for scoring
        centroid_weight: float = 0.35,
        bandwidth_weight: float = 0.15,
        attack_weight: float = 0.20,
        context_weight: float = 0.30,
        # Thresholds
        min_confidence: float = 0.3,
        # Attack detection
        attack_duration_ms: float = 30.0,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.tuning = tuning or STANDARD_TUNING
        
        self.centroid_weight = centroid_weight
        self.bandwidth_weight = bandwidth_weight
        self.attack_weight = attack_weight
        self.context_weight = context_weight
        
        self.min_confidence = min_confidence
        self.attack_frames = int((attack_duration_ms / 1000.0) * sr / hop_length)
        
        # Build lookup: for each MIDI note, which (string, fret) combos are possible
        self._build_position_lookup()
        
        # Scaler for normalizing features (fit on first call)
        self.scaler = StandardScaler()
        self._scaler_fitted = False
    
    def _build_position_lookup(self):
        """Build lookup table for possible positions for each MIDI note."""
        self.positions_by_midi: Dict[int, List[Tuple[int, int]]] = {}
        
        for midi in range(36, 91):  # Guitar range
            positions = []
            for string_idx, open_note in enumerate(self.tuning):
                fret = midi - open_note
                if 0 <= fret <= 24:
                    positions.append((string_idx, fret))
            if positions:
                self.positions_by_midi[midi] = positions
    
    def extract_features(
        self, 
        y: np.ndarray, 
        onset_sample: int,
        duration_samples: int
    ) -> SpectralFeatures:
        """
        Extract spectral features from a note segment.
        
        Args:
            y: Full audio signal
            onset_sample: Sample index where note starts
            duration_samples: Duration of note in samples
        
        Returns:
            SpectralFeatures object
        """
        # Extract note segment with some padding
        start = max(0, onset_sample)
        end = min(len(y), onset_sample + duration_samples)
        segment = y[start:end]
        
        if len(segment) < self.n_fft:
            # Pad short segments
            segment = np.pad(segment, (0, self.n_fft - len(segment)))
        
        # Compute spectrogram
        S = np.abs(librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # 1. Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        mean_centroid = np.mean(centroid) if len(centroid) > 0 else 0
        
        # 2. Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sr)[0]
        mean_bandwidth = np.mean(bandwidth) if len(bandwidth) > 0 else 0
        
        # 3. Spectral rolloff (85%)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr, roll_percent=0.85)[0]
        mean_rolloff = np.mean(rolloff) if len(rolloff) > 0 else 0
        
        # 4. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
        mean_zcr = np.mean(zcr) if len(zcr) > 0 else 0
        
        # 5. Spectral flatness (noise-like vs tonal)
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        mean_flatness = np.mean(flatness) if len(flatness) > 0 else 0
        
        # 6. Spectral contrast (per-band valley-to-peak ratios)
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr)
        mean_contrast = np.mean(contrast, axis=1)
        
        # 7. MFCCs (timbral texture)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=self.sr, n_mfcc=13)
        mean_mfcc = np.mean(mfcc, axis=1)
        
        # 8. Attack vs sustain brightness
        attack_end = min(len(centroid), self.attack_frames + 1)
        if attack_end > 0:
            attack_brightness = np.mean(centroid[:attack_end])
        else:
            attack_brightness = mean_centroid
            
        if len(centroid) > attack_end:
            sustain_brightness = np.mean(centroid[attack_end:])
        else:
            sustain_brightness = mean_centroid
        
        return SpectralFeatures(
            centroid=mean_centroid,
            bandwidth=mean_bandwidth,
            rolloff=mean_rolloff,
            zcr=mean_zcr,
            flatness=mean_flatness,
            contrast=mean_contrast,
            mfcc=mean_mfcc,
            attack_brightness=attack_brightness,
            sustain_brightness=sustain_brightness
        )
    
    def _score_position_by_centroid(
        self,
        features: SpectralFeatures,
        string: int,
        fret: int
    ) -> float:
        """
        Score a (string, fret) position based on spectral centroid.
        
        Higher frets on lower strings can sound similar to lower frets on higher strings,
        so we need to account for fret position.
        """
        # Get expected centroid range for this string
        low, high = STRING_CENTROID_RANGES.get(string, (400, 1200))
        
        # Adjust expected centroid based on fret position
        # Higher frets = shorter string = higher centroid
        fret_factor = 1.0 + (fret * 0.02)  # ~2% brighter per fret
        expected_low = low * fret_factor
        expected_high = high * fret_factor
        expected_center = (expected_low + expected_high) / 2
        
        # Calculate how well the actual centroid matches expected
        actual = features.centroid
        
        # Score based on distance from expected center, normalized by range
        range_width = expected_high - expected_low
        if range_width <= 0:
            range_width = 200  # Fallback
        
        distance = abs(actual - expected_center)
        
        # Score: 1.0 if perfect match, decreasing with distance
        # Use Gaussian-like scoring
        score = np.exp(-0.5 * (distance / range_width) ** 2)
        
        return score
    
    def _score_position_by_bandwidth(
        self,
        features: SpectralFeatures,
        string: int,
        fret: int
    ) -> float:
        """Score based on spectral bandwidth."""
        low, high = STRING_BANDWIDTH_RANGES.get(string, (200, 800))
        
        # Adjust for fret position (higher frets have narrower bandwidth)
        fret_factor = 1.0 - (fret * 0.01)  # Slightly narrower per fret
        expected_low = low * fret_factor
        expected_high = high * fret_factor
        expected_center = (expected_low + expected_high) / 2
        
        actual = features.bandwidth
        range_width = expected_high - expected_low
        if range_width <= 0:
            range_width = 100
        
        distance = abs(actual - expected_center)
        score = np.exp(-0.5 * (distance / range_width) ** 2)
        
        return score
    
    def _score_position_by_attack(
        self,
        features: SpectralFeatures,
        string: int,
        fret: int
    ) -> float:
        """
        Score based on attack transient characteristics.
        
        Higher strings typically have brighter, sharper attacks.
        Lower strings have more "thump" in the attack.
        """
        # Ratio of attack brightness to sustain brightness
        if features.sustain_brightness > 0:
            attack_ratio = features.attack_brightness / features.sustain_brightness
        else:
            attack_ratio = 1.0
        
        # Higher strings typically have attack_ratio closer to 1
        # Lower strings often have higher attack_ratio (brighter attack relative to sustain)
        
        # Expected attack ratio by string
        expected_ratios = {
            0: 1.3,   # Low E - noticeable attack brightness
            1: 1.25,  # A
            2: 1.2,   # D
            3: 1.15,  # G
            4: 1.1,   # B
            5: 1.05,  # High e - attack and sustain similar brightness
        }
        
        expected = expected_ratios.get(string, 1.15)
        distance = abs(attack_ratio - expected)
        
        # Score with tolerance
        score = np.exp(-2.0 * distance ** 2)
        
        return score
    
    def _score_position_by_playability(
        self,
        string: int,
        fret: int,
        prev_position: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Score based on playability/ergonomics.
        
        This is the "context" factor:
        - Prefer lower frets (easier)
        - Prefer positions close to previous position
        - Avoid huge jumps
        """
        score = 1.0
        
        # Prefer lower frets
        if fret <= 5:
            score *= 1.1
        elif fret <= 12:
            score *= 1.0
        else:
            score *= 0.85
        
        # If we have a previous position, prefer staying close
        if prev_position is not None:
            prev_string, prev_fret = prev_position
            
            fret_distance = abs(fret - prev_fret)
            string_distance = abs(string - prev_string)
            
            # Penalize large jumps
            if fret_distance > 5:
                score *= 0.7
            elif fret_distance > 3:
                score *= 0.85
            
            if string_distance > 2:
                score *= 0.8
            elif string_distance > 1:
                score *= 0.9
        
        return score
    
    def predict_string(
        self,
        y: np.ndarray,
        midi_note: int,
        onset_time: float,
        duration: float,
        prev_position: Optional[Tuple[int, int]] = None,
        verbose: bool = False
    ) -> StringPrediction:
        """
        Predict which string a note was played on.
        
        Args:
            y: Full audio signal
            midi_note: MIDI note number
            onset_time: Note onset time in seconds
            duration: Note duration in seconds
            prev_position: Previous (string, fret) if known
            verbose: Print debug info
            
        Returns:
            StringPrediction with best position and alternatives
        """
        # Get possible positions for this MIDI note
        positions = self.positions_by_midi.get(midi_note, [])
        
        if not positions:
            # Note out of range - can't be played on guitar
            return StringPrediction(
                string=-1, fret=-1, confidence=0.0,
                method='none', features=None
            )
        
        if len(positions) == 1:
            # Only one option - return it with high confidence
            string, fret = positions[0]
            return StringPrediction(
                string=string, fret=fret, confidence=0.95,
                method='unique', features=None
            )
        
        # Multiple positions possible - use spectral analysis
        onset_sample = int(onset_time * self.sr)
        duration_samples = int(min(duration, 0.5) * self.sr)  # Cap at 500ms for analysis
        
        # Extract features
        features = self.extract_features(y, onset_sample, duration_samples)
        
        # Score each possible position
        scored_positions = []
        
        for string, fret in positions:
            # Calculate scores from different methods
            centroid_score = self._score_position_by_centroid(features, string, fret)
            bandwidth_score = self._score_position_by_bandwidth(features, string, fret)
            attack_score = self._score_position_by_attack(features, string, fret)
            playability_score = self._score_position_by_playability(string, fret, prev_position)
            
            # Weighted combination
            total_score = (
                self.centroid_weight * centroid_score +
                self.bandwidth_weight * bandwidth_score +
                self.attack_weight * attack_score +
                self.context_weight * playability_score
            )
            
            scored_positions.append((string, fret, total_score, {
                'centroid': centroid_score,
                'bandwidth': bandwidth_score,
                'attack': attack_score,
                'playability': playability_score
            }))
            
            if verbose:
                print(f"  Position ({STRING_NAMES[string]}, fret {fret}): "
                      f"centroid={centroid_score:.3f}, bw={bandwidth_score:.3f}, "
                      f"attack={attack_score:.3f}, play={playability_score:.3f}, "
                      f"TOTAL={total_score:.3f}")
        
        # Sort by total score
        scored_positions.sort(key=lambda x: x[2], reverse=True)
        
        # Best position
        best_string, best_fret, best_score, _ = scored_positions[0]
        
        # Calculate confidence based on margin over second-best
        if len(scored_positions) > 1:
            second_score = scored_positions[1][2]
            margin = best_score - second_score
            # Confidence increases with margin
            confidence = min(0.95, 0.5 + margin * 2)
        else:
            confidence = 0.9
        
        # Normalize confidence
        confidence = max(self.min_confidence, min(1.0, confidence))
        
        # Build alternatives list
        alternatives = [
            (s, f, sc) for s, f, sc, _ in scored_positions[1:]
        ]
        
        return StringPrediction(
            string=best_string,
            fret=best_fret,
            confidence=confidence,
            method='spectral',
            features=features,
            alternative_positions=alternatives
        )
    
    def predict_sequence(
        self,
        y: np.ndarray,
        notes: List,  # List of Note objects
        verbose: bool = False
    ) -> List[StringPrediction]:
        """
        Predict strings for a sequence of notes, using context.
        
        This method tracks the previous position to help with
        context-aware predictions (prefer positions close to previous).
        
        Args:
            y: Full audio signal
            notes: List of Note objects (must have midi, start_time, duration)
            verbose: Print debug info
            
        Returns:
            List of StringPrediction objects
        """
        predictions = []
        prev_position = None
        
        for i, note in enumerate(notes):
            if verbose:
                print(f"\nNote {i+1}: MIDI {note.midi} ({note.name}) at {note.start_time:.3f}s")
            
            pred = self.predict_string(
                y,
                note.midi,
                note.start_time,
                note.duration,
                prev_position=prev_position,
                verbose=verbose
            )
            
            predictions.append(pred)
            
            # Update previous position if prediction is confident
            if pred.confidence > 0.5:
                prev_position = (pred.string, pred.fret)
            
            if verbose:
                if pred.string >= 0:
                    print(f"  -> Predicted: {STRING_NAMES[pred.string]} string, fret {pred.fret} "
                          f"(confidence: {pred.confidence:.2f})")
                else:
                    print(f"  -> No valid position found")
        
        return predictions


def compute_string_brightness_profile(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Dict[str, float]:
    """
    Compute overall brightness profile of the audio.
    
    This can help calibrate the string detector for specific guitars
    (e.g., nylon vs steel strings, acoustic vs electric).
    
    Returns dict with:
    - mean_centroid: Overall brightness
    - centroid_std: Brightness variation
    - brightness_category: 'bright', 'neutral', 'warm'
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    
    mean_centroid = np.mean(centroid)
    std_centroid = np.std(centroid)
    
    # Categorize brightness
    if mean_centroid > 1500:
        category = 'bright'
    elif mean_centroid > 800:
        category = 'neutral'
    else:
        category = 'warm'
    
    return {
        'mean_centroid': float(mean_centroid),
        'centroid_std': float(std_centroid),
        'brightness_category': category
    }


def detect_string_with_harmonics(
    y: np.ndarray,
    sr: int,
    fundamental_hz: float,
    onset_sample: int,
    duration_samples: int
) -> Dict[str, float]:
    """
    Analyze harmonic content to help identify string.
    
    Different strings have different harmonic profiles:
    - Wound strings (E, A, D) have more prominent upper harmonics
    - Plain strings (G, B, e) have cleaner, simpler harmonic series
    
    Returns harmonic profile analysis.
    """
    # Extract segment
    start = max(0, onset_sample)
    end = min(len(y), onset_sample + duration_samples)
    segment = y[start:end]
    
    if len(segment) < 2048:
        segment = np.pad(segment, (0, 2048 - len(segment)))
    
    # Compute FFT
    fft = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1/sr)
    
    # Find harmonic amplitudes
    harmonics = []
    for h in range(1, 7):  # First 6 harmonics
        target_freq = fundamental_hz * h
        # Find closest bin
        idx = np.argmin(np.abs(freqs - target_freq))
        # Take max in small window around target
        window_start = max(0, idx - 2)
        window_end = min(len(fft), idx + 3)
        amplitude = np.max(fft[window_start:window_end])
        harmonics.append(amplitude)
    
    # Normalize to fundamental
    if harmonics[0] > 0:
        harmonics = [h / harmonics[0] for h in harmonics]
    
    # Calculate harmonic features
    harmonic_decay = np.mean([harmonics[i] / harmonics[i-1] 
                             for i in range(1, len(harmonics)) 
                             if harmonics[i-1] > 0.01])
    
    odd_even_ratio = np.mean([harmonics[0], harmonics[2], harmonics[4]]) / \
                     (np.mean([harmonics[1], harmonics[3], harmonics[5]]) + 0.01)
    
    return {
        'harmonics': harmonics,
        'harmonic_decay': float(harmonic_decay),
        'odd_even_ratio': float(odd_even_ratio)
    }


# Convenience function for integration with guitar_tabs.py
def choose_string_position_spectral(
    y: np.ndarray,
    note,  # Note object
    prev_position: Optional[Tuple[int, int]] = None,
    tuning: List[int] = None,
    sr: int = 22050,
    verbose: bool = False
) -> Tuple[int, int, float]:
    """
    Convenience function to get (string, fret, confidence) for a note.
    
    This integrates with the existing guitar_tabs.py workflow.
    
    Args:
        y: Full audio signal at sr sample rate
        note: Note object with midi, start_time, duration
        prev_position: Previous (string, fret) for context
        tuning: Guitar tuning as MIDI notes
        sr: Sample rate
        verbose: Print debug info
        
    Returns:
        (string, fret, confidence) tuple
    """
    detector = StringDetector(sr=sr, tuning=tuning)
    
    pred = detector.predict_string(
        y,
        note.midi,
        note.start_time,
        note.duration,
        prev_position=prev_position,
        verbose=verbose
    )
    
    return pred.string, pred.fret, pred.confidence


def compare_string_detection(
    y: np.ndarray,
    notes: List,
    tuning: List[int] = None,
    sr: int = 22050
) -> Dict:
    """
    Compare spectral string detection with heuristic method.
    
    Returns dict with:
    - agreements: Number of notes where both methods agree
    - disagreements: List of (note, spectral_pos, heuristic_pos)
    - spectral_confidences: Average confidence of spectral method
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    detector = StringDetector(sr=sr, tuning=tuning)
    
    agreements = 0
    disagreements = []
    confidences = []
    
    prev_spectral = None
    prev_heuristic = None
    
    for note in notes:
        # Spectral prediction
        pred = detector.predict_string(
            y, note.midi, note.start_time, note.duration,
            prev_position=prev_spectral
        )
        
        if pred.string >= 0:
            spectral_pos = (pred.string, pred.fret)
            confidences.append(pred.confidence)
            prev_spectral = spectral_pos
        else:
            spectral_pos = None
        
        # Heuristic method (simple)
        from guitar_tabs import midi_to_fret_options
        options = midi_to_fret_options(note.midi, tuning)
        
        if options:
            # Simple heuristic: prefer lower frets, middle strings
            scored = []
            for string, fret in options:
                score = 0
                if fret <= 5:
                    score += 3
                elif fret <= 12:
                    score += 1
                score -= fret * 0.2
                string_pref = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5]
                score += string_pref[string] * 3
                if prev_heuristic:
                    ps, pf = prev_heuristic
                    score -= abs(fret - pf) * 0.3
                    score -= abs(string - ps) * 0.5
                scored.append((score, string, fret))
            scored.sort(reverse=True)
            heuristic_pos = (scored[0][1], scored[0][2])
            prev_heuristic = heuristic_pos
        else:
            heuristic_pos = None
        
        # Compare
        if spectral_pos and heuristic_pos:
            if spectral_pos == heuristic_pos:
                agreements += 1
            else:
                disagreements.append({
                    'note': note.name if hasattr(note, 'name') else f"MIDI {note.midi}",
                    'midi': note.midi,
                    'spectral': spectral_pos,
                    'heuristic': heuristic_pos,
                    'confidence': pred.confidence
                })
    
    return {
        'total': len(notes),
        'agreements': agreements,
        'disagreements': disagreements,
        'avg_confidence': np.mean(confidences) if confidences else 0
    }


if __name__ == "__main__":
    # Test the string detector
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python string_detection.py <audio_file>")
        print("\nThis module detects which guitar string notes are played on")
        print("using spectral/timbre analysis.")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Loading audio: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Compute overall brightness profile
    print("\n📊 Audio Brightness Profile:")
    profile = compute_string_brightness_profile(y, sr)
    print(f"  Mean spectral centroid: {profile['mean_centroid']:.1f} Hz")
    print(f"  Centroid std dev: {profile['centroid_std']:.1f} Hz")
    print(f"  Category: {profile['brightness_category']}")
    
    # Demo: test string detection for a few MIDI notes
    print("\n🎸 String Detection Demo:")
    detector = StringDetector(sr=sr)
    
    # Test notes that can be played on multiple strings
    test_cases = [
        (60, 0.5, 0.3),   # C4 - can be on A string fret 15, D string fret 10, G string fret 5, B string fret 1
        (64, 1.0, 0.3),   # E4 - can be on D string fret 14, G string fret 9, B string fret 5, e string fret 0
        (55, 1.5, 0.3),   # G3 - can be on E string fret 15, A string fret 10, D string fret 5, G string fret 0
    ]
    
    for midi, start, dur in test_cases:
        onset_sample = int(start * sr)
        duration_samples = int(dur * sr)
        
        # Extract features
        features = detector.extract_features(y, onset_sample, duration_samples)
        
        print(f"\nMIDI {midi}:")
        print(f"  Spectral centroid: {features.centroid:.1f} Hz")
        print(f"  Spectral bandwidth: {features.bandwidth:.1f} Hz")
        print(f"  Attack brightness: {features.attack_brightness:.1f} Hz")
        print(f"  Sustain brightness: {features.sustain_brightness:.1f} Hz")
        
        # Get prediction
        from dataclasses import dataclass
        
        @dataclass
        class DummyNote:
            midi: int
            start_time: float
            duration: float
            name: str = ""
        
        note = DummyNote(midi=midi, start_time=start, duration=dur, 
                        name=librosa.midi_to_note(midi))
        
        pred = detector.predict_string(y, midi, start, dur, verbose=True)
        
        if pred.string >= 0:
            print(f"  Prediction: {STRING_NAMES[pred.string]} string, fret {pred.fret}")
            print(f"  Confidence: {pred.confidence:.2%}")
            if pred.alternative_positions:
                print(f"  Alternatives: {[(STRING_NAMES[s], f, f'{c:.2%}') for s, f, c in pred.alternative_positions[:3]]}")
