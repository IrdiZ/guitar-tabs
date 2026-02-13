#!/usr/bin/env python3
"""
Palm Mute & Harmonic Detection for Guitar Tabs

Detects:
- Palm mutes (PM): Damped/muted notes with reduced harmonics, shorter decay
- Natural harmonics (NH): Strong overtones at 5th, 7th, 12th fret positions
- Pinch harmonics (PH): Squealing high overtones

Uses spectral analysis to differentiate harmonic content vs fundamental.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks, medfilt
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TechniqueType(Enum):
    """Guitar technique types for muting and harmonics."""
    NORMAL = ""
    PALM_MUTE = "PM"
    NATURAL_HARMONIC = "NH"
    PINCH_HARMONIC = "PH"


@dataclass
class SpectralFeatures:
    """Spectral features extracted for a note segment."""
    time: float
    duration: float
    
    # Harmonic analysis
    fundamental_freq: float
    fundamental_power: float
    harmonic_ratios: List[float]  # Power of harmonics relative to fundamental
    spectral_centroid: float      # Center of mass of spectrum
    spectral_rolloff: float       # Frequency below which 85% of energy lies
    spectral_flatness: float      # How noise-like vs tonal (0=tonal, 1=noise)
    
    # Envelope analysis
    attack_time: float            # Time to reach peak amplitude
    decay_rate: float             # Rate of amplitude decay
    sustain_level: float          # Relative level after initial decay
    
    # Harmonic-to-noise ratio
    hnr: float                    # Higher = more harmonic content


@dataclass
class TechniqueDetection:
    """Result of technique detection for a note."""
    time: float
    duration: float
    technique: TechniqueType
    confidence: float
    details: Dict


class HarmonicDetector:
    """
    Detects palm mutes and harmonics using spectral analysis.
    
    Detection criteria:
    
    Palm Mutes (PM):
    - Reduced high-frequency content (lower spectral centroid)
    - Faster decay than normal notes
    - More noise-like spectrum (higher spectral flatness)
    - Harmonics are suppressed relative to fundamental
    
    Natural Harmonics (NH):
    - Very strong specific harmonic (2nd, 3rd, 4th, etc.)
    - Weak or absent fundamental frequency
    - Long sustain (harmonics ring clearly)
    - Clean, pure tone (low spectral flatness)
    
    Pinch Harmonics (PH):
    - Very high spectral centroid (squealing sound)
    - Strong high harmonics (often 3rd-5th harmonic dominant)
    - Distinctive attack transient
    - Higher frequency than expected for the note
    """
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        # Palm mute thresholds
        pm_centroid_ratio: float = 0.6,      # Centroid below 60% of expected
        pm_decay_threshold: float = 0.15,    # Fast decay (seconds to -6dB)
        pm_flatness_threshold: float = 0.1,  # Higher flatness for muted sound
        pm_harmonic_suppression: float = 0.4, # Harmonics below 40% of normal
        # Natural harmonic thresholds
        nh_harmonic_dominance: float = 3.0,  # Harmonic > 3x fundamental
        nh_sustain_threshold: float = 0.7,   # Sustain above 70% after 0.5s
        nh_flatness_max: float = 0.05,       # Very pure tone
        # Pinch harmonic thresholds
        ph_centroid_ratio: float = 2.5,      # Centroid > 2.5x expected
        ph_high_harmonic_boost: float = 4.0, # High harmonics > 4x normal
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Thresholds
        self.pm_centroid_ratio = pm_centroid_ratio
        self.pm_decay_threshold = pm_decay_threshold
        self.pm_flatness_threshold = pm_flatness_threshold
        self.pm_harmonic_suppression = pm_harmonic_suppression
        
        self.nh_harmonic_dominance = nh_harmonic_dominance
        self.nh_sustain_threshold = nh_sustain_threshold
        self.nh_flatness_max = nh_flatness_max
        
        self.ph_centroid_ratio = ph_centroid_ratio
        self.ph_high_harmonic_boost = ph_high_harmonic_boost
    
    def extract_features(
        self,
        y: np.ndarray,
        start_time: float,
        duration: float,
        expected_freq: float
    ) -> Optional[SpectralFeatures]:
        """Extract spectral features for a note segment."""
        
        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + duration) * self.sr)
        
        if start_sample >= len(y) or end_sample <= start_sample:
            return None
        
        segment = y[start_sample:min(end_sample, len(y))]
        
        if len(segment) < self.n_fft:
            # Pad if too short
            segment = np.pad(segment, (0, self.n_fft - len(segment)))
        
        # Compute STFT
        D = librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Average spectrum
        avg_spectrum = np.mean(S, axis=1)
        
        # Find fundamental peak near expected frequency
        expected_bin = int(expected_freq * self.n_fft / self.sr)
        search_range = max(5, int(50 * self.n_fft / self.sr))  # ±50 Hz
        
        start_bin = max(0, expected_bin - search_range)
        end_bin = min(len(avg_spectrum), expected_bin + search_range)
        
        if start_bin >= end_bin:
            return None
        
        local_peak = np.argmax(avg_spectrum[start_bin:end_bin]) + start_bin
        fundamental_freq = freqs[local_peak] if local_peak < len(freqs) else expected_freq
        fundamental_power = avg_spectrum[local_peak] if local_peak < len(avg_spectrum) else 0
        
        # Analyze harmonics (2nd through 6th)
        harmonic_ratios = []
        for h in range(2, 7):
            harmonic_freq = fundamental_freq * h
            harmonic_bin = int(harmonic_freq * self.n_fft / self.sr)
            
            if harmonic_bin < len(avg_spectrum):
                # Search around expected harmonic position
                h_start = max(0, harmonic_bin - 3)
                h_end = min(len(avg_spectrum), harmonic_bin + 4)
                harmonic_power = np.max(avg_spectrum[h_start:h_end])
                ratio = harmonic_power / (fundamental_power + 1e-10)
                harmonic_ratios.append(ratio)
            else:
                harmonic_ratios.append(0.0)
        
        # Spectral centroid (weighted mean of frequencies)
        spectral_centroid = librosa.feature.spectral_centroid(
            S=S, sr=self.sr
        )[0]
        avg_centroid = np.mean(spectral_centroid)
        
        # Spectral rolloff (frequency below which 85% of energy lies)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=S, sr=self.sr, roll_percent=0.85
        )[0]
        avg_rolloff = np.mean(spectral_rolloff)
        
        # Spectral flatness (0=tonal, 1=noise-like)
        spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
        avg_flatness = np.mean(spectral_flatness)
        
        # Envelope analysis
        envelope = np.abs(librosa.util.normalize(segment))
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.01 * self.sr))
        
        # Attack time (time to reach peak)
        peak_idx = np.argmax(envelope_smooth)
        attack_time = peak_idx / self.sr
        
        # Decay rate (time to fall to -6dB after peak)
        peak_level = envelope_smooth[peak_idx]
        target_level = peak_level * 0.5  # -6dB
        
        decay_rate = 1.0  # Default: slow decay
        for i in range(peak_idx, len(envelope_smooth)):
            if envelope_smooth[i] < target_level:
                decay_rate = (i - peak_idx) / self.sr
                break
        
        # Sustain level (average level in last 30% of note)
        sustain_start = int(0.7 * len(envelope_smooth))
        sustain_level = np.mean(envelope_smooth[sustain_start:]) / (peak_level + 1e-10)
        
        # Harmonic-to-noise ratio
        y_harmonic, y_noise = librosa.effects.hpss(segment, margin=2.0)
        harmonic_energy = np.sum(y_harmonic ** 2)
        noise_energy = np.sum(y_noise ** 2)
        hnr = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
        
        return SpectralFeatures(
            time=start_time,
            duration=duration,
            fundamental_freq=fundamental_freq,
            fundamental_power=fundamental_power,
            harmonic_ratios=harmonic_ratios,
            spectral_centroid=avg_centroid,
            spectral_rolloff=avg_rolloff,
            spectral_flatness=avg_flatness,
            attack_time=attack_time,
            decay_rate=decay_rate,
            sustain_level=sustain_level,
            hnr=hnr
        )
    
    def detect_palm_mute(
        self,
        features: SpectralFeatures,
        expected_freq: float
    ) -> Tuple[bool, float, Dict]:
        """
        Detect if a note is palm muted.
        
        Palm mutes have:
        - Reduced spectral centroid (muffled sound)
        - Faster decay
        - Higher spectral flatness (more percussive)
        - Suppressed harmonics
        """
        scores = []
        details = {}
        
        # Expected centroid for normal note (roughly 2-3x fundamental)
        expected_centroid = expected_freq * 2.5
        centroid_ratio = features.spectral_centroid / (expected_centroid + 1e-10)
        details['centroid_ratio'] = centroid_ratio
        
        # Score: lower centroid = more likely palm mute
        if centroid_ratio < self.pm_centroid_ratio:
            scores.append(1.0 - centroid_ratio / self.pm_centroid_ratio)
        else:
            scores.append(0.0)
        
        # Score: faster decay = more likely palm mute
        if features.decay_rate < self.pm_decay_threshold:
            scores.append(1.0)
        elif features.decay_rate < self.pm_decay_threshold * 2:
            scores.append(0.5)
        else:
            scores.append(0.0)
        details['decay_rate'] = features.decay_rate
        
        # Score: higher flatness = more likely palm mute
        if features.spectral_flatness > self.pm_flatness_threshold:
            scores.append(min(1.0, features.spectral_flatness / 0.2))
        else:
            scores.append(0.0)
        details['flatness'] = features.spectral_flatness
        
        # Score: suppressed harmonics = more likely palm mute
        avg_harmonic_ratio = np.mean(features.harmonic_ratios[:3]) if features.harmonic_ratios else 1.0
        if avg_harmonic_ratio < self.pm_harmonic_suppression:
            scores.append(1.0 - avg_harmonic_ratio / self.pm_harmonic_suppression)
        else:
            scores.append(0.0)
        details['harmonic_ratio'] = avg_harmonic_ratio
        
        # Overall confidence
        confidence = np.mean(scores) if scores else 0.0
        is_palm_mute = confidence > 0.5
        
        return is_palm_mute, confidence, details
    
    def detect_natural_harmonic(
        self,
        features: SpectralFeatures,
        expected_freq: float
    ) -> Tuple[bool, float, Dict]:
        """
        Detect if a note is a natural harmonic.
        
        Natural harmonics have:
        - Strong harmonic overtone (often 2x, 3x, 4x fundamental)
        - Weak or absent fundamental
        - Long sustain
        - Very pure tone (low flatness)
        """
        scores = []
        details = {}
        
        # Check for harmonic dominance
        harmonic_dominance = 0.0
        dominant_harmonic = 0
        
        for i, ratio in enumerate(features.harmonic_ratios):
            if ratio > self.nh_harmonic_dominance:
                harmonic_dominance = ratio
                dominant_harmonic = i + 2  # 2nd, 3rd, 4th, etc.
                break
        
        details['harmonic_dominance'] = harmonic_dominance
        details['dominant_harmonic'] = dominant_harmonic
        
        if harmonic_dominance > self.nh_harmonic_dominance:
            scores.append(min(1.0, harmonic_dominance / (self.nh_harmonic_dominance * 2)))
        else:
            scores.append(0.0)
        
        # Check sustain level
        details['sustain'] = features.sustain_level
        if features.sustain_level > self.nh_sustain_threshold:
            scores.append(1.0)
        elif features.sustain_level > self.nh_sustain_threshold * 0.7:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Check for pure tone (low flatness)
        details['flatness'] = features.spectral_flatness
        if features.spectral_flatness < self.nh_flatness_max:
            scores.append(1.0)
        elif features.spectral_flatness < self.nh_flatness_max * 2:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Additional: check if actual frequency is near a harmonic position
        # Common natural harmonic fret positions: 5 (4th harm), 7 (5th harm), 12 (2nd harm)
        actual_freq = features.spectral_centroid
        freq_ratio = actual_freq / (expected_freq + 1e-10)
        
        # Check if frequency matches a harmonic multiple
        harmonic_match = False
        for h in [2, 3, 4, 5, 6]:
            if abs(freq_ratio - h) < 0.15:
                harmonic_match = True
                details['freq_ratio'] = freq_ratio
                break
        
        if harmonic_match:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        confidence = np.mean(scores) if scores else 0.0
        is_natural_harmonic = confidence > 0.5
        
        return is_natural_harmonic, confidence, details
    
    def detect_pinch_harmonic(
        self,
        features: SpectralFeatures,
        expected_freq: float
    ) -> Tuple[bool, float, Dict]:
        """
        Detect if a note is a pinch harmonic.
        
        Pinch harmonics have:
        - Very high spectral centroid (squealing sound)
        - Strong high harmonics (3rd-5th or higher)
        - Distinctive pick attack
        - Actual frequency much higher than expected fundamental
        """
        scores = []
        details = {}
        
        # Check for extremely high centroid
        expected_centroid = expected_freq * 2.5
        centroid_ratio = features.spectral_centroid / (expected_centroid + 1e-10)
        details['centroid_ratio'] = centroid_ratio
        
        if centroid_ratio > self.ph_centroid_ratio:
            scores.append(min(1.0, centroid_ratio / (self.ph_centroid_ratio * 2)))
        else:
            scores.append(0.0)
        
        # Check for boosted high harmonics
        high_harmonic_ratio = np.mean(features.harmonic_ratios[2:]) if len(features.harmonic_ratios) > 2 else 0
        details['high_harmonic_ratio'] = high_harmonic_ratio
        
        if high_harmonic_ratio > self.ph_high_harmonic_boost:
            scores.append(min(1.0, high_harmonic_ratio / (self.ph_high_harmonic_boost * 2)))
        else:
            scores.append(0.0)
        
        # Pinch harmonics often have a distinctive quick attack
        details['attack_time'] = features.attack_time
        if features.attack_time < 0.02:  # Very fast attack
            scores.append(1.0)
        elif features.attack_time < 0.05:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Check that it's still relatively harmonic (not noise)
        details['hnr'] = features.hnr
        if features.hnr > 10:  # Good harmonic content
            scores.append(1.0)
        elif features.hnr > 5:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        confidence = np.mean(scores) if scores else 0.0
        is_pinch_harmonic = confidence > 0.5
        
        return is_pinch_harmonic, confidence, details
    
    def detect_technique(
        self,
        y: np.ndarray,
        time: float,
        duration: float,
        expected_freq: float
    ) -> TechniqueDetection:
        """
        Detect if a note uses palm muting or harmonics.
        
        Returns the most likely technique with confidence.
        """
        features = self.extract_features(y, time, duration, expected_freq)
        
        if features is None:
            return TechniqueDetection(
                time=time,
                duration=duration,
                technique=TechniqueType.NORMAL,
                confidence=0.0,
                details={'error': 'Could not extract features'}
            )
        
        # Check all techniques
        is_pm, pm_conf, pm_details = self.detect_palm_mute(features, expected_freq)
        is_nh, nh_conf, nh_details = self.detect_natural_harmonic(features, expected_freq)
        is_ph, ph_conf, ph_details = self.detect_pinch_harmonic(features, expected_freq)
        
        # Determine best match
        techniques = [
            (TechniqueType.PALM_MUTE, pm_conf, pm_details),
            (TechniqueType.NATURAL_HARMONIC, nh_conf, nh_details),
            (TechniqueType.PINCH_HARMONIC, ph_conf, ph_details),
        ]
        
        # Find highest confidence
        best_technique, best_conf, best_details = max(techniques, key=lambda x: x[1])
        
        # Only report if confidence is above threshold
        if best_conf < 0.5:
            return TechniqueDetection(
                time=time,
                duration=duration,
                technique=TechniqueType.NORMAL,
                confidence=1.0 - max(pm_conf, nh_conf, ph_conf),
                details={
                    'pm_conf': pm_conf,
                    'nh_conf': nh_conf,
                    'ph_conf': ph_conf,
                    'spectral_features': {
                        'centroid': features.spectral_centroid,
                        'flatness': features.spectral_flatness,
                        'decay_rate': features.decay_rate,
                        'hnr': features.hnr
                    }
                }
            )
        
        return TechniqueDetection(
            time=time,
            duration=duration,
            technique=best_technique,
            confidence=best_conf,
            details=best_details
        )
    
    def analyze_audio(
        self,
        y: np.ndarray,
        notes: List[dict],
        verbose: bool = False
    ) -> List[TechniqueDetection]:
        """
        Analyze an entire audio file with detected notes.
        
        Args:
            y: Audio signal
            notes: List of notes with 'time', 'duration', 'pitch' (MIDI or Hz)
            verbose: Print detection details
            
        Returns:
            List of technique detections for each note
        """
        results = []
        
        for i, note in enumerate(notes):
            time = note.get('time', 0)
            duration = note.get('duration', 0.2)
            
            # Get expected frequency
            pitch = note.get('pitch', 0)
            if pitch > 127:  # Assume Hz
                expected_freq = pitch
            else:  # MIDI note number
                expected_freq = librosa.midi_to_hz(pitch)
            
            detection = self.detect_technique(y, time, duration, expected_freq)
            results.append(detection)
            
            if verbose and detection.technique != TechniqueType.NORMAL:
                print(f"  Note {i} at {time:.2f}s: {detection.technique.value} "
                      f"(conf: {detection.confidence:.2f})")
        
        return results


def annotate_tabs_with_techniques(
    tab_lines: List[str],
    detections: List[TechniqueDetection],
    notes: List[dict]
) -> List[str]:
    """
    Add PM/NH/PH annotations to tab output.
    
    Adds technique markers above the tab lines where detected.
    """
    # This would integrate with the main tab formatting
    # For now, return a summary
    annotated = []
    
    technique_notes = []
    for det, note in zip(detections, notes):
        if det.technique != TechniqueType.NORMAL:
            technique_notes.append({
                'time': det.time,
                'technique': det.technique.value,
                'confidence': det.confidence,
                'fret': note.get('fret', '?'),
                'string': note.get('string', '?')
            })
    
    if technique_notes:
        annotated.append("# Detected Techniques:")
        for tn in technique_notes:
            annotated.append(
                f"#   {tn['time']:.2f}s: {tn['technique']} "
                f"(string {tn['string']}, fret {tn['fret']}) "
                f"[{tn['confidence']*100:.0f}% confidence]"
            )
        annotated.append("")
    
    annotated.extend(tab_lines)
    return annotated


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect palm mutes and harmonics in guitar audio'
    )
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--notes-json', help='JSON file with detected notes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    
    args = parser.parse_args()
    
    print(f"Loading audio: {args.audio_file}")
    y, sr = librosa.load(args.audio_file, sr=args.sr)
    print(f"  Duration: {len(y)/sr:.2f}s, Sample rate: {sr}")
    
    # If no notes provided, do basic onset detection
    if args.notes_json:
        import json
        with open(args.notes_json) as f:
            notes = json.load(f)
    else:
        print("Detecting onsets...")
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # Estimate pitches at onsets
        print("Estimating pitches...")
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'),
            sr=sr
        )
        f0_times = librosa.times_like(f0, sr=sr)
        
        notes = []
        for i, onset in enumerate(onset_times):
            # Find nearest pitch
            idx = np.argmin(np.abs(f0_times - onset))
            if f0[idx] and not np.isnan(f0[idx]):
                duration = (onset_times[i+1] - onset) if i + 1 < len(onset_times) else 0.3
                notes.append({
                    'time': float(onset),
                    'duration': float(min(duration, 0.5)),
                    'pitch': float(f0[idx])  # Hz
                })
        
        print(f"  Detected {len(notes)} notes")
    
    # Create detector
    detector = HarmonicDetector(sr=sr)
    
    print("\nAnalyzing techniques...")
    detections = detector.analyze_audio(y, notes, verbose=args.verbose)
    
    # Summarize results
    pm_count = sum(1 for d in detections if d.technique == TechniqueType.PALM_MUTE)
    nh_count = sum(1 for d in detections if d.technique == TechniqueType.NATURAL_HARMONIC)
    ph_count = sum(1 for d in detections if d.technique == TechniqueType.PINCH_HARMONIC)
    normal_count = sum(1 for d in detections if d.technique == TechniqueType.NORMAL)
    
    print(f"\n=== Detection Results ===")
    print(f"Total notes analyzed: {len(detections)}")
    print(f"  Normal:             {normal_count}")
    print(f"  Palm Mutes (PM):    {pm_count}")
    print(f"  Natural Harmonics:  {nh_count}")
    print(f"  Pinch Harmonics:    {ph_count}")
    
    # Print technique details
    if args.verbose:
        print(f"\n=== Technique Details ===")
        for i, det in enumerate(detections):
            if det.technique != TechniqueType.NORMAL:
                print(f"\nNote {i} at {det.time:.2f}s: {det.technique.value}")
                print(f"  Confidence: {det.confidence:.2%}")
                for k, v in det.details.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.3f}")
                    else:
                        print(f"  {k}: {v}")
    
    return detections


if __name__ == "__main__":
    main()
