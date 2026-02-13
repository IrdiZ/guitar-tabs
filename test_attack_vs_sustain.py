#!/usr/bin/env python3
"""
Test script comparing attack-based vs sustain-based pitch detection.

Shows why attack transients are cleaner for distorted guitar:
1. Detect pitch from attack only (first 20-50ms)
2. Detect pitch from sustain only (after 50ms)
3. Compare results - attack should be more consistent

This demonstrates the key insight: distortion muddies the sustain portion
but the attack is relatively clean.
"""

import numpy as np
import librosa
import sys
from attack_transient_pitch import AttackTransientPitchDetector, AttackConfig

# Constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi):
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def detect_sustain_pitch(y, onset_time, sr=22050):
    """Detect pitch from sustain portion (after attack)."""
    # Sustain starts after attack (50ms) and goes for ~200ms
    sustain_start = int((onset_time + 0.05) * sr)
    sustain_end = int((onset_time + 0.25) * sr)
    
    if sustain_start >= len(y):
        return None
    sustain_end = min(sustain_end, len(y))
    
    if sustain_end - sustain_start < 1000:
        return None
    
    sustain = y[sustain_start:sustain_end]
    
    # Use pYIN on sustain
    f0, voiced_flag, voiced_probs = librosa.pyin(
        sustain,
        fmin=70,
        fmax=1200,
        sr=sr
    )
    
    # Get median of voiced frames
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) > 0:
        return float(np.median(voiced_f0))
    
    return None


def compare_detection(audio_path):
    """Compare attack-based vs sustain-based pitch detection."""
    print(f"\n{'='*70}")
    print("ATTACK vs SUSTAIN PITCH DETECTION COMPARISON")
    print(f"{'='*70}")
    print(f"File: {audio_path}\n")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Attack-based detection
    config = AttackConfig(
        sr=sr,
        verbose=False,
        attack_start_ms=5,
        attack_end_ms=50
    )
    detector = AttackTransientPitchDetector(config)
    attack_results = detector.detect(y)
    
    print(f"{'Onset':<8} {'Attack Pitch':<15} {'Sustain Pitch':<15} {'Match?':<8} {'Notes'}")
    print("-" * 70)
    
    matches = 0
    octave_errors = 0
    total = 0
    
    for r in attack_results:
        attack_hz = r.pitch_hz
        attack_midi = r.midi_note
        attack_name = r.note_name
        
        # Get sustain pitch
        sustain_hz = detect_sustain_pitch(y, r.onset_time, sr)
        
        if sustain_hz is not None:
            sustain_midi = int(round(librosa.hz_to_midi(sustain_hz)))
            sustain_name = midi_to_name(sustain_midi)
            
            # Check match
            midi_diff = abs(attack_midi - sustain_midi)
            if midi_diff == 0:
                match_str = "✓ SAME"
                matches += 1
            elif midi_diff == 12:
                match_str = "⚠ 8ve"
                octave_errors += 1
            elif midi_diff <= 2:
                match_str = "~ close"
            else:
                match_str = "✗ DIFF"
            
            total += 1
            
            print(f"{r.onset_time:6.2f}s  "
                  f"{attack_name:<4} ({attack_hz:6.1f})  "
                  f"{sustain_name:<4} ({sustain_hz:6.1f})  "
                  f"{match_str:<8} "
                  f"methods={','.join(r.methods_agreed)}")
        else:
            print(f"{r.onset_time:6.2f}s  "
                  f"{attack_name:<4} ({attack_hz:6.1f})  "
                  f"{'N/A':<15}  "
                  f"-        "
                  f"methods={','.join(r.methods_agreed)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total notes analyzed: {len(attack_results)}")
    if total > 0:
        print(f"Attack-Sustain agreement: {matches}/{total} ({100*matches/total:.0f}%)")
        print(f"Octave errors: {octave_errors}")
    print("\n💡 Key insight: Attack transients (first 20-50ms) contain cleaner")
    print("   pitch information than the sustained portion for distorted guitar.")
    print("   Distortion effects (compression, harmonics) take time to engage.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_attack_vs_sustain.py <audio_file>")
        sys.exit(1)
    
    compare_detection(sys.argv[1])
