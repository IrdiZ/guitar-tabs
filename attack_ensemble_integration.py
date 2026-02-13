#!/usr/bin/env python3
"""
Attack-Based Ensemble Pitch Detection Integration

Integrates attack transient analysis into the ensemble pitch detection pipeline.

For distorted guitar:
1. Use attack transients (first 20-50ms) for PRIMARY pitch detection
2. Use sustain for VALIDATION only (if attack and sustain agree, higher confidence)
3. Weight attack-based detection heavily in the ensemble

The attack-first approach is critical because:
- Distortion compresses dynamics and adds harmonics over time
- The attack is the initial string excitation before effects engage
- First few wave cycles are cleanest and most periodic
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from attack_transient_pitch import (
    AttackTransientPitchDetector, 
    AttackConfig, 
    AttackPitchResult
)

# Try to import ensemble pitch
try:
    from ensemble_pitch import EnsemblePitchDetector, EnsembleConfig, ConsensusNote
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class AttackEnsembleConfig:
    """Configuration for attack-enhanced ensemble detection."""
    
    # Attack analysis settings
    attack_start_ms: float = 5       # Skip first 5ms (pick noise)
    attack_end_ms: float = 50        # Primary analysis window
    extended_attack_ms: float = 100  # Extended window for validation
    
    # Ensemble integration
    attack_weight: float = 2.0       # Weight for attack-based detection
    sustain_weight: float = 0.5      # Weight for sustain-based detection
    
    # When attack and sustain agree, boost confidence
    agreement_boost: float = 0.3
    
    # Fallback behavior
    use_sustain_fallback: bool = True  # Use sustain if attack fails
    
    # Output
    min_confidence: float = 0.35
    verbose: bool = True


@dataclass
class AttackEnhancedNote:
    """A note detected with attack-enhanced analysis."""
    midi_note: int
    start_time: float
    end_time: float
    frequency: float
    confidence: float
    
    # Attack analysis details
    attack_hz: float
    attack_confidence: float
    attack_methods: List[str]
    
    # Sustain analysis (if available)
    sustain_hz: Optional[float] = None
    sustain_confidence: Optional[float] = None
    
    # Agreement flag
    attack_sustain_agree: bool = False
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AttackEnsembleDetector:
    """
    Ensemble pitch detection with attack transient priority.
    
    Strategy:
    1. Detect onsets
    2. For each onset, analyze attack transient (primary pitch source)
    3. Optionally validate with sustain analysis
    4. Combine with weighted voting
    """
    
    def __init__(
        self, 
        sr: int = 22050,
        attack_config: Optional[AttackConfig] = None,
        ensemble_config: Optional[AttackEnsembleConfig] = None
    ):
        self.sr = sr
        self.attack_config = attack_config or AttackConfig(sr=sr)
        self.config = ensemble_config or AttackEnsembleConfig()
        
        # Create attack detector
        self.attack_detector = AttackTransientPitchDetector(self.attack_config)
        
    def detect(
        self, 
        y: np.ndarray,
        sr: Optional[int] = None
    ) -> List[AttackEnhancedNote]:
        """
        Detect notes using attack-first strategy.
        
        Args:
            y: Audio signal
            sr: Sample rate (uses self.sr if not provided)
            
        Returns:
            List of AttackEnhancedNote with attack-based pitch detection
        """
        if sr is not None and sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        
        if self.config.verbose:
            print("🎸 Attack-Enhanced Ensemble Detection")
            print(f"   Attack window: {self.attack_config.attack_start_ms}-{self.attack_config.attack_end_ms}ms")
        
        # Step 1: Get attack-based pitches
        attack_results = self.attack_detector.detect(y)
        
        if self.config.verbose:
            print(f"   Attack detector found {len(attack_results)} notes")
        
        # Step 2: Optionally get sustain-based validation
        sustain_pitches = {}
        if self.config.use_sustain_fallback:
            sustain_pitches = self._analyze_sustain(y, attack_results)
        
        # Step 3: Combine and create enhanced notes
        notes = []
        for ar in attack_results:
            # Get sustain pitch if available
            sustain_hz = sustain_pitches.get(ar.onset_time)
            sustain_midi = None
            if sustain_hz is not None:
                sustain_midi = int(round(librosa.hz_to_midi(sustain_hz)))
            
            # Check agreement
            attack_sustain_agree = False
            confidence = ar.confidence
            
            if sustain_midi is not None:
                midi_diff = abs(ar.midi_note - sustain_midi)
                if midi_diff == 0:
                    attack_sustain_agree = True
                    confidence = min(1.0, confidence + self.config.agreement_boost)
                elif midi_diff == 12:
                    # Octave error - trust attack (HPS handles this better)
                    pass
            
            # Estimate end time (use next onset or default duration)
            idx = attack_results.index(ar)
            if idx < len(attack_results) - 1:
                end_time = attack_results[idx + 1].onset_time
            else:
                end_time = ar.onset_time + 0.5  # Default 500ms
            
            note = AttackEnhancedNote(
                midi_note=ar.midi_note,
                start_time=ar.onset_time,
                end_time=end_time,
                frequency=ar.pitch_hz,
                confidence=confidence,
                attack_hz=ar.pitch_hz,
                attack_confidence=ar.confidence,
                attack_methods=ar.methods_agreed,
                sustain_hz=sustain_hz,
                sustain_confidence=0.5 if sustain_hz else None,
                attack_sustain_agree=attack_sustain_agree
            )
            
            if confidence >= self.config.min_confidence:
                notes.append(note)
        
        if self.config.verbose:
            agreed = sum(1 for n in notes if n.attack_sustain_agree)
            print(f"   Final: {len(notes)} notes ({agreed} with attack-sustain agreement)")
        
        return notes
    
    def _analyze_sustain(
        self, 
        y: np.ndarray, 
        attack_results: List[AttackPitchResult]
    ) -> Dict[float, float]:
        """Analyze sustain portion for validation."""
        sustain_pitches = {}
        
        for ar in attack_results:
            # Sustain starts after attack ends
            sustain_start_sec = ar.onset_time + self.attack_config.attack_end_ms / 1000
            sustain_end_sec = ar.onset_time + 0.25  # 250ms max
            
            start_sample = int(sustain_start_sec * self.sr)
            end_sample = int(sustain_end_sec * self.sr)
            
            if start_sample >= len(y):
                continue
            end_sample = min(end_sample, len(y))
            
            if end_sample - start_sample < 500:  # Need at least ~20ms
                continue
            
            sustain = y[start_sample:end_sample]
            
            # Use pYIN on sustain
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    sustain,
                    fmin=70,
                    fmax=1200,
                    sr=self.sr
                )
                
                voiced_f0 = f0[~np.isnan(f0)]
                if len(voiced_f0) > 0:
                    sustain_pitches[ar.onset_time] = float(np.median(voiced_f0))
            except:
                pass
        
        return sustain_pitches


def detect_with_attack_priority(
    audio_path: str,
    verbose: bool = True
) -> List[AttackEnhancedNote]:
    """
    Convenience function to detect notes with attack-first strategy.
    
    This is the recommended function for distorted guitar.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Configure for distorted guitar
    attack_config = AttackConfig(
        sr=sr,
        attack_start_ms=5,    # Skip pick click
        attack_end_ms=50,     # Primary window
        verbose=verbose,
        use_autocorrelation=True,
        use_yin=True,
        use_cqt=True,
        use_hps=True          # Critical for distortion
    )
    
    ensemble_config = AttackEnsembleConfig(
        verbose=verbose,
        attack_weight=2.0,
        use_sustain_fallback=True
    )
    
    detector = AttackEnsembleDetector(
        sr=sr,
        attack_config=attack_config,
        ensemble_config=ensemble_config
    )
    
    return detector.detect(y)


def format_results(notes: List[AttackEnhancedNote]) -> str:
    """Format notes as readable output."""
    lines = []
    lines.append("Attack-Enhanced Detection Results")
    lines.append("=" * 60)
    lines.append(f"{'Time':<8} {'Note':<5} {'Hz':<8} {'Conf':<6} {'Methods':<20} {'Agree?'}")
    lines.append("-" * 60)
    
    for n in notes:
        agree_str = "✓" if n.attack_sustain_agree else ""
        methods_str = ",".join(n.attack_methods)[:20]
        lines.append(
            f"{n.start_time:6.2f}s  {n.name:<5} {n.frequency:7.1f}  "
            f"{n.confidence:.2f}   {methods_str:<20} {agree_str}"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python attack_ensemble_integration.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"\n{'='*70}")
    print("ATTACK-PRIORITY ENSEMBLE DETECTION")
    print(f"{'='*70}\n")
    
    notes = detect_with_attack_priority(audio_file, verbose=True)
    
    print("\n" + format_results(notes))
