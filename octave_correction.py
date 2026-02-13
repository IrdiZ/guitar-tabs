#!/usr/bin/env python3
"""
Octave Correction for Guitar Pitch Detection

Common problem: pitch detectors often detect the WRONG OCTAVE (e.g., E2 instead of E3).
This happens because:
1. The fundamental frequency can be weaker than harmonics
2. Pitch detectors may lock onto a harmonic or subharmonic
3. Low-frequency components can be ambiguous

This module implements octave correction using:
1. Harmonic analysis - validates that detected pitch's harmonics are present
2. Guitar range constraints - ensures notes fall within playable range
3. Fundamental vs harmonic energy comparison
4. Subharmonic presence detection
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


# Guitar frequency constraints
GUITAR_LOWEST_STANDARD = 82.4  # E2 in standard tuning
GUITAR_LOWEST_DROP = 61.7      # B1 in drop tunings (giving margin)
GUITAR_HIGHEST = 1320.0        # E6 (22nd fret high E)

# Common guitar tuning lowest notes (MIDI)
TUNING_LOWEST = {
    'standard': 40,      # E2
    'drop_d': 38,        # D2
    'drop_c': 36,        # C2
    'half_step_down': 39,  # Eb2
    'full_step_down': 38,  # D2
}


class OctaveCorrection(Enum):
    """Type of octave correction applied."""
    NONE = "none"
    UP_ONE = "up_one"      # Moved up one octave
    UP_TWO = "up_two"      # Moved up two octaves
    DOWN_ONE = "down_one"  # Moved down one octave


@dataclass
class OctaveCorrectionResult:
    """Result of octave correction analysis."""
    original_freq: float
    corrected_freq: float
    original_midi: int
    corrected_midi: int
    correction: OctaveCorrection
    confidence: float
    harmonic_score: float
    details: Dict


class OctaveCorrector:
    """
    Validates and corrects octave errors in pitch detection using harmonic analysis.
    
    The key insight: if we detect a pitch P, we should see harmonics at 2P, 3P, 4P.
    If instead we see that P itself looks like a harmonic (e.g., P/2 has strong
    harmonics at P, 3P/2, 2P), then we probably detected a harmonic, not fundamental.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 4096,  # Larger FFT for better frequency resolution
        hop_length: int = 512,
        # Guitar constraints
        min_freq: float = GUITAR_LOWEST_DROP,
        max_freq: float = GUITAR_HIGHEST,
        tuning: str = 'standard',
        # Harmonic analysis
        n_harmonics: int = 5,           # Check up to 5th harmonic
        harmonic_tolerance: float = 0.03,  # 3% frequency tolerance
        min_harmonic_ratio: float = 0.1,   # Min ratio to consider harmonic present
        # Confidence thresholds
        correction_threshold: float = 0.6,  # Confidence needed to apply correction
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.tuning = tuning
        
        self.n_harmonics = n_harmonics
        self.harmonic_tolerance = harmonic_tolerance
        self.min_harmonic_ratio = min_harmonic_ratio
        
        self.correction_threshold = correction_threshold
        
        # Frequency resolution
        self.freq_resolution = sr / n_fft
        
        # Precompute frequency bins
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    def _get_spectrum_at_time(
        self,
        S: np.ndarray,
        time: float,
        window_frames: int = 3
    ) -> np.ndarray:
        """Get averaged spectrum around a specific time."""
        frame = int(time * self.sr / self.hop_length)
        start = max(0, frame - window_frames // 2)
        end = min(S.shape[1], frame + window_frames // 2 + 1)
        
        if start >= end:
            return S[:, frame] if frame < S.shape[1] else np.zeros(S.shape[0])
        
        return np.mean(S[:, start:end], axis=1)
    
    def _find_harmonic_energy(
        self,
        spectrum: np.ndarray,
        freq: float,
        tolerance: float = None
    ) -> float:
        """Find energy at a specific frequency in the spectrum."""
        if tolerance is None:
            tolerance = self.harmonic_tolerance
        
        if freq <= 0 or freq > self.sr / 2:
            return 0.0
        
        # Find frequency bin
        target_bin = int(freq * self.n_fft / self.sr)
        
        # Search within tolerance range
        freq_range = freq * tolerance
        bin_range = int(freq_range * self.n_fft / self.sr)
        bin_range = max(1, bin_range)
        
        start_bin = max(0, target_bin - bin_range)
        end_bin = min(len(spectrum), target_bin + bin_range + 1)
        
        if start_bin >= end_bin:
            return 0.0
        
        return np.max(spectrum[start_bin:end_bin])
    
    def _compute_harmonic_profile(
        self,
        spectrum: np.ndarray,
        fundamental: float
    ) -> List[float]:
        """
        Compute energy at each harmonic of a fundamental frequency.
        Returns list of harmonic energies relative to fundamental.
        """
        fund_energy = self._find_harmonic_energy(spectrum, fundamental)
        
        if fund_energy <= 0:
            return [0.0] * self.n_harmonics
        
        harmonic_ratios = []
        for h in range(2, self.n_harmonics + 2):
            harmonic_freq = fundamental * h
            harmonic_energy = self._find_harmonic_energy(spectrum, harmonic_freq)
            ratio = harmonic_energy / (fund_energy + 1e-10)
            harmonic_ratios.append(ratio)
        
        return harmonic_ratios
    
    def _validate_pitch_harmonics(
        self,
        spectrum: np.ndarray,
        detected_freq: float
    ) -> Tuple[float, Dict]:
        """
        Validate a detected pitch by checking its harmonic structure.
        
        Returns:
            score: 0-1 where 1 means strong harmonic validation
            details: Dictionary with analysis details
        """
        details = {
            'fundamental': detected_freq,
            'harmonics_present': [],
            'harmonics_missing': [],
        }
        
        fund_energy = self._find_harmonic_energy(spectrum, detected_freq)
        details['fundamental_energy'] = float(fund_energy)
        
        if fund_energy <= 0:
            return 0.0, details
        
        # Check harmonics
        harmonics_found = 0
        harmonic_energies = []
        
        for h in range(2, self.n_harmonics + 2):
            harmonic_freq = detected_freq * h
            
            # Skip if harmonic is beyond Nyquist
            if harmonic_freq > self.sr / 2:
                break
            
            harmonic_energy = self._find_harmonic_energy(spectrum, harmonic_freq)
            ratio = harmonic_energy / (fund_energy + 1e-10)
            harmonic_energies.append(ratio)
            
            if ratio >= self.min_harmonic_ratio:
                harmonics_found += 1
                details['harmonics_present'].append(h)
            else:
                details['harmonics_missing'].append(h)
        
        details['harmonic_energies'] = harmonic_energies
        
        # Score: what fraction of expected harmonics are present?
        expected_harmonics = min(self.n_harmonics, int((self.sr / 2) / detected_freq) - 1)
        expected_harmonics = max(1, expected_harmonics)
        
        harmonic_score = harmonics_found / expected_harmonics
        
        # Bonus: strong 2nd and 3rd harmonics are typical of guitar
        if len(harmonic_energies) >= 2:
            if harmonic_energies[0] > 0.15:  # 2nd harmonic present
                harmonic_score = min(1.0, harmonic_score + 0.15)
            if harmonic_energies[1] > 0.1:   # 3rd harmonic present
                harmonic_score = min(1.0, harmonic_score + 0.1)
        
        details['harmonic_score'] = harmonic_score
        return harmonic_score, details
    
    def _check_subharmonic(
        self,
        spectrum: np.ndarray,
        detected_freq: float
    ) -> Tuple[bool, float, Dict]:
        """
        Check if the detected frequency might actually be a harmonic of a lower pitch.
        
        If P/2 (or P/3) shows a better harmonic profile than P, we may have
        detected a harmonic instead of the fundamental.
        """
        details = {
            'subharmonic_candidates': []
        }
        
        # Validate the detected pitch first
        detected_score, det_details = self._validate_pitch_harmonics(spectrum, detected_freq)
        
        best_subharmonic = None
        best_score = detected_score
        
        # Check P/2 (one octave lower)
        sub_freq = detected_freq / 2
        if sub_freq >= self.min_freq:
            sub_score, sub_details = self._validate_pitch_harmonics(spectrum, sub_freq)
            
            # Also check if the detected freq appears as a harmonic of sub_freq
            # If detected_freq is at 2x sub_freq, it should appear as 2nd harmonic
            if 'harmonic_energies' in sub_details and len(sub_details['harmonic_energies']) > 0:
                # 2nd harmonic should be strong
                second_harmonic_ratio = sub_details['harmonic_energies'][0]
                if second_harmonic_ratio > 0.5:  # Strong 2nd harmonic
                    sub_score += 0.2
            
            details['subharmonic_candidates'].append({
                'freq': sub_freq,
                'score': sub_score,
                'multiplier': 2
            })
            
            if sub_score > best_score + 0.15:  # Require significant improvement
                best_subharmonic = sub_freq
                best_score = sub_score
        
        # Check P/3 (octave + fifth lower) - less common but possible
        sub_freq_3 = detected_freq / 3
        if sub_freq_3 >= self.min_freq:
            sub_score_3, sub_details_3 = self._validate_pitch_harmonics(spectrum, sub_freq_3)
            
            details['subharmonic_candidates'].append({
                'freq': sub_freq_3,
                'score': sub_score_3,
                'multiplier': 3
            })
            
            if sub_score_3 > best_score + 0.2:  # Higher threshold for P/3
                best_subharmonic = sub_freq_3
                best_score = sub_score_3
        
        has_better_subharmonic = best_subharmonic is not None
        details['original_score'] = detected_score
        details['best_subharmonic'] = best_subharmonic
        details['best_score'] = best_score
        
        return has_better_subharmonic, best_score, details
    
    def _check_octave_up(
        self,
        spectrum: np.ndarray,
        detected_freq: float
    ) -> Tuple[bool, float, Dict]:
        """
        Check if the pitch should be moved UP an octave.
        
        This happens when:
        1. The detected pitch is below guitar range
        2. The octave above has a better harmonic profile
        3. The detected frequency looks like it could be a subharmonic
        """
        details = {}
        
        detected_score, det_details = self._validate_pitch_harmonics(spectrum, detected_freq)
        details['original_score'] = detected_score
        
        # Check octave up
        octave_up = detected_freq * 2
        if octave_up <= self.max_freq:
            up_score, up_details = self._validate_pitch_harmonics(spectrum, octave_up)
            details['octave_up_score'] = up_score
            
            # Check if octave up has energy at its fundamental
            up_energy = self._find_harmonic_energy(spectrum, octave_up)
            detected_energy = self._find_harmonic_energy(spectrum, detected_freq)
            
            energy_ratio = up_energy / (detected_energy + 1e-10)
            details['energy_ratio'] = energy_ratio
            
            # Conditions for moving up:
            # 1. Detected freq is suspiciously low (near guitar's lowest)
            is_very_low = detected_freq < 100  # Below ~G2
            
            # 2. Octave up has comparable or better harmonic structure
            better_harmonics = up_score >= detected_score - 0.1
            
            # 3. Energy at octave up is significant
            has_energy = energy_ratio > 0.3
            
            should_move_up = is_very_low and better_harmonics and has_energy
            
            details['is_very_low'] = is_very_low
            details['better_harmonics'] = better_harmonics
            details['has_energy'] = has_energy
            
            return should_move_up, up_score, details
        
        return False, detected_score, details
    
    def correct_pitch(
        self,
        spectrum: np.ndarray,
        detected_freq: float,
        time: float = 0.0
    ) -> OctaveCorrectionResult:
        """
        Analyze and correct octave errors for a detected pitch.
        
        Args:
            spectrum: Magnitude spectrum at the time of the note
            detected_freq: Detected frequency in Hz
            time: Time stamp (for logging)
            
        Returns:
            OctaveCorrectionResult with corrected pitch and analysis details
        """
        original_midi = int(round(librosa.hz_to_midi(detected_freq)))
        details = {'time': time, 'original_freq': detected_freq}
        
        # First, validate the detected pitch
        harmonic_score, harmonic_details = self._validate_pitch_harmonics(spectrum, detected_freq)
        details['harmonic_validation'] = harmonic_details
        
        # Check if we should move DOWN (detected a harmonic instead of fundamental)
        has_subharmonic, sub_score, sub_details = self._check_subharmonic(spectrum, detected_freq)
        details['subharmonic_check'] = sub_details
        
        if has_subharmonic and sub_score > harmonic_score + 0.15:
            # The subharmonic has a better harmonic profile - consider moving down
            corrected_freq = sub_details['best_subharmonic']
            corrected_midi = int(round(librosa.hz_to_midi(corrected_freq)))
            
            # IMPORTANT: Check if the corrected pitch is still within guitar range
            # Don't move down if it would go below the lowest playable note
            lowest_midi = TUNING_LOWEST.get(self.tuning, 40)
            
            if corrected_midi >= lowest_midi:
                # Determine how many octaves down
                octave_diff = int(round(np.log2(detected_freq / corrected_freq)))
                correction = OctaveCorrection.DOWN_ONE if octave_diff == 1 else OctaveCorrection.NONE
                
                return OctaveCorrectionResult(
                    original_freq=detected_freq,
                    corrected_freq=corrected_freq,
                    original_midi=original_midi,
                    corrected_midi=corrected_midi,
                    correction=correction,
                    confidence=sub_score,
                    harmonic_score=sub_score,
                    details=details
                )
            else:
                # Subharmonic is too low - keep original
                details['subharmonic_rejected'] = f"Would be MIDI {corrected_midi}, below guitar range (min: {lowest_midi})"
        
        # Check if we should move UP (detected subharmonic or very low pitch)
        should_up, up_score, up_details = self._check_octave_up(spectrum, detected_freq)
        details['octave_up_check'] = up_details
        
        if should_up:
            corrected_freq = detected_freq * 2
            corrected_midi = int(round(librosa.hz_to_midi(corrected_freq)))
            
            return OctaveCorrectionResult(
                original_freq=detected_freq,
                corrected_freq=corrected_freq,
                original_midi=original_midi,
                corrected_midi=corrected_midi,
                correction=OctaveCorrection.UP_ONE,
                confidence=up_score,
                harmonic_score=up_score,
                details=details
            )
        
        # No correction needed
        return OctaveCorrectionResult(
            original_freq=detected_freq,
            corrected_freq=detected_freq,
            original_midi=original_midi,
            corrected_midi=original_midi,
            correction=OctaveCorrection.NONE,
            confidence=harmonic_score,
            harmonic_score=harmonic_score,
            details=details
        )
    
    def correct_pitch_from_audio(
        self,
        y: np.ndarray,
        detected_freq: float,
        time: float
    ) -> OctaveCorrectionResult:
        """
        Correct pitch using audio signal directly.
        Computes spectrum internally.
        """
        # Compute STFT
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Get spectrum at the specific time
        spectrum = self._get_spectrum_at_time(S, time)
        
        return self.correct_pitch(spectrum, detected_freq, time)
    
    def correct_notes(
        self,
        y: np.ndarray,
        notes: List[dict],
        verbose: bool = False
    ) -> Tuple[List[dict], List[OctaveCorrectionResult]]:
        """
        Apply octave correction to a list of detected notes.
        
        Args:
            y: Audio signal
            notes: List of note dicts with 'midi' or 'freq', 'time', 'duration'
            verbose: Print correction details
            
        Returns:
            Tuple of (corrected_notes, correction_results)
        """
        # Compute STFT once
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        corrected_notes = []
        results = []
        
        corrections_applied = 0
        
        for note in notes:
            time = note.get('time', note.get('start_time', 0))
            
            # Get frequency
            if 'freq' in note:
                freq = note['freq']
            elif 'midi' in note:
                freq = librosa.midi_to_hz(note['midi'])
            else:
                corrected_notes.append(note.copy())
                continue
            
            # Get spectrum at note time
            spectrum = self._get_spectrum_at_time(S, time)
            
            # Apply correction
            result = self.correct_pitch(spectrum, freq, time)
            results.append(result)
            
            # Create corrected note
            corrected = note.copy()
            if result.correction != OctaveCorrection.NONE:
                corrections_applied += 1
                corrected['midi'] = result.corrected_midi
                corrected['freq'] = result.corrected_freq
                corrected['octave_corrected'] = True
                corrected['original_midi'] = result.original_midi
                
                if verbose:
                    print(f"  [{time:.2f}s] {result.correction.value}: "
                          f"MIDI {result.original_midi} -> {result.corrected_midi} "
                          f"({librosa.midi_to_note(result.original_midi)} -> "
                          f"{librosa.midi_to_note(result.corrected_midi)})")
            
            corrected_notes.append(corrected)
        
        if verbose:
            print(f"\nOctave correction: {corrections_applied}/{len(notes)} notes corrected")
        
        return corrected_notes, results


def compute_spectral_flux_in_band(
    S: np.ndarray,
    sr: int,
    n_fft: int,
    freq_low: float,
    freq_high: float
) -> np.ndarray:
    """Compute spectral flux in a specific frequency band."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find bin indices
    low_bin = np.searchsorted(freqs, freq_low)
    high_bin = np.searchsorted(freqs, freq_high)
    
    # Extract band
    S_band = S[low_bin:high_bin, :]
    
    # Compute flux
    flux = np.zeros(S_band.shape[1])
    for i in range(1, S_band.shape[1]):
        diff = S_band[:, i] - S_band[:, i-1]
        diff = np.maximum(0, diff)  # Half-wave rectification
        flux[i] = np.sum(diff)
    
    return flux


def validate_guitar_range(
    midi_note: int,
    tuning: str = 'standard',
    strict: bool = True
) -> Tuple[bool, str]:
    """
    Check if a MIDI note is within valid guitar range.
    
    Args:
        midi_note: MIDI note number
        tuning: Guitar tuning name
        strict: If True, only allow notes actually playable on that tuning
        
    Returns:
        (is_valid, reason)
    """
    # Get lowest note for tuning
    lowest = TUNING_LOWEST.get(tuning, 40)
    highest = lowest + 48  # ~4 octaves of range (24 frets * 2 octaves)
    
    if strict:
        if midi_note < lowest:
            return False, f"Below lowest string ({librosa.midi_to_note(lowest)})"
        if midi_note > highest:
            return False, f"Above highest fret ({librosa.midi_to_note(highest)})"
    else:
        # Allow some flexibility
        if midi_note < lowest - 2:
            return False, f"Too low for guitar"
        if midi_note > highest + 4:
            return False, f"Too high for guitar"
    
    return True, "Valid"


def apply_octave_correction(
    y: np.ndarray,
    notes: List,
    sr: int = 22050,
    verbose: bool = True
) -> List:
    """
    Convenience function to apply octave correction to Note objects.
    
    Args:
        y: Audio signal
        notes: List of Note objects (with midi, start_time, duration, confidence)
        sr: Sample rate
        verbose: Print correction info
        
    Returns:
        List of corrected Note objects
    """
    corrector = OctaveCorrector(sr=sr)
    
    # Convert to dict format
    note_dicts = []
    for note in notes:
        note_dicts.append({
            'midi': note.midi,
            'time': note.start_time,
            'duration': note.duration,
            'confidence': note.confidence
        })
    
    # Apply correction
    corrected_dicts, results = corrector.correct_notes(y, note_dicts, verbose=verbose)
    
    # Convert back to Note objects
    from dataclasses import dataclass
    
    @dataclass
    class CorrectedNote:
        midi: int
        start_time: float
        duration: float
        confidence: float
        octave_corrected: bool = False
        original_midi: int = None
        
        @property
        def name(self):
            return librosa.midi_to_note(self.midi)
    
    corrected_notes = []
    for d in corrected_dicts:
        corrected_notes.append(CorrectedNote(
            midi=d['midi'],
            start_time=d['time'],
            duration=d['duration'],
            confidence=d['confidence'],
            octave_corrected=d.get('octave_corrected', False),
            original_midi=d.get('original_midi')
        ))
    
    return corrected_notes


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Octave correction for guitar pitch detection'
    )
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--tuning', default='standard', help='Guitar tuning')
    
    args = parser.parse_args()
    
    print(f"Loading audio: {args.audio_file}")
    y, sr = librosa.load(args.audio_file, sr=args.sr)
    print(f"  Duration: {len(y)/sr:.2f}s, Sample rate: {sr}")
    
    # Detect pitches using pyin
    print("\nDetecting pitches with pYIN...")
    f0, voiced, probs = librosa.pyin(
        y, fmin=GUITAR_LOWEST_DROP, fmax=GUITAR_HIGHEST,
        sr=sr, hop_length=512
    )
    
    # Get onset times
    print("Detecting onsets...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    # Build note list from onsets
    f0_times = librosa.times_like(f0, sr=sr, hop_length=512)
    notes = []
    
    for i, onset in enumerate(onset_times):
        # Find pitch at onset
        idx = np.argmin(np.abs(f0_times - onset))
        if f0[idx] and not np.isnan(f0[idx]):
            duration = (onset_times[i+1] - onset) if i + 1 < len(onset_times) else 0.3
            
            # Convert to MIDI
            midi = int(round(librosa.hz_to_midi(f0[idx])))
            
            notes.append({
                'midi': midi,
                'freq': float(f0[idx]),
                'time': float(onset),
                'duration': float(min(duration, 0.5)),
                'confidence': float(probs[idx]) if probs is not None else 0.5
            })
    
    print(f"Detected {len(notes)} notes")
    
    # Apply octave correction
    print("\n=== Applying Octave Correction ===")
    corrector = OctaveCorrector(sr=sr, tuning=args.tuning)
    corrected_notes, results = corrector.correct_notes(y, notes, verbose=True)
    
    # Summary
    corrections = [r for r in results if r.correction != OctaveCorrection.NONE]
    print(f"\n=== Summary ===")
    print(f"Total notes: {len(notes)}")
    print(f"Corrections applied: {len(corrections)}")
    
    if corrections:
        up_count = sum(1 for r in corrections if r.correction in [OctaveCorrection.UP_ONE, OctaveCorrection.UP_TWO])
        down_count = sum(1 for r in corrections if r.correction == OctaveCorrection.DOWN_ONE)
        print(f"  Moved up: {up_count}")
        print(f"  Moved down: {down_count}")
    
    # Show before/after comparison for first few notes
    print("\n=== First 10 Notes (Before -> After) ===")
    for i, (orig, corr, res) in enumerate(zip(notes[:10], corrected_notes[:10], results[:10])):
        orig_note = librosa.midi_to_note(orig['midi'])
        corr_note = librosa.midi_to_note(corr['midi'])
        marker = " *" if res.correction != OctaveCorrection.NONE else ""
        print(f"  {i+1}. [{orig['time']:.2f}s] {orig_note} -> {corr_note}{marker}")
    
    return corrected_notes, results


if __name__ == "__main__":
    main()
