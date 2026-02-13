#!/usr/bin/env python3
"""
Test script: Compare pitch detection accuracy before and after frequency cleanup.
"""

import numpy as np
import librosa
from freq_cleanup import freq_domain_cleanup, FreqCleanupConfig, analyze_audio_quality
from pitch_accuracy import detect_pitches_accurate, compare_detectors
import sys


def test_pitch_accuracy_improvement(audio_path: str):
    """
    Compare pitch detection accuracy before and after frequency cleanup.
    """
    print("=" * 60)
    print("🔬 PITCH ACCURACY COMPARISON TEST")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print()
    
    # Load audio
    y_original, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y_original) / sr
    print(f"Duration: {duration:.2f}s")
    
    # Apply cleanup
    print("\n🔧 Applying frequency domain cleanup...")
    config = FreqCleanupConfig.default()
    y_cleaned = freq_domain_cleanup(y_original, sr, config=config, verbose=False)
    
    # Test 1: Run pitch detection on ORIGINAL
    print("\n" + "-" * 60)
    print("📊 ORIGINAL AUDIO - Pitch Detection")
    print("-" * 60)
    
    f0_orig, conf_orig, pitches_orig = detect_pitches_accurate(
        y_original, sr, 
        hop_length=512,
        min_votes=2,
        min_confidence=0.3,
        verbose=True
    )
    
    # Test 2: Run pitch detection on CLEANED
    print("\n" + "-" * 60)
    print("📊 CLEANED AUDIO - Pitch Detection")
    print("-" * 60)
    
    f0_clean, conf_clean, pitches_clean = detect_pitches_accurate(
        y_cleaned, sr,
        hop_length=512,
        min_votes=2,
        min_confidence=0.3,
        verbose=True
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("📈 COMPARISON RESULTS")
    print("=" * 60)
    
    # Metrics
    valid_orig = f0_orig > 0
    valid_clean = f0_clean > 0
    
    n_valid_orig = np.sum(valid_orig)
    n_valid_clean = np.sum(valid_clean)
    
    mean_conf_orig = np.mean(conf_orig[valid_orig]) if n_valid_orig > 0 else 0
    mean_conf_clean = np.mean(conf_clean[valid_clean]) if n_valid_clean > 0 else 0
    
    # Pitch stability (lower variance = more stable)
    if n_valid_orig > 10:
        midi_orig = librosa.hz_to_midi(f0_orig[valid_orig])
        stability_orig = np.std(np.diff(midi_orig))
    else:
        stability_orig = float('inf')
    
    if n_valid_clean > 10:
        midi_clean = librosa.hz_to_midi(f0_clean[valid_clean])
        stability_clean = np.std(np.diff(midi_clean))
    else:
        stability_clean = float('inf')
    
    # Note counts (unique notes detected)
    if n_valid_orig > 0:
        notes_orig = set(int(round(m)) for m in librosa.hz_to_midi(f0_orig[valid_orig]))
    else:
        notes_orig = set()
    
    if n_valid_clean > 0:
        notes_clean = set(int(round(m)) for m in librosa.hz_to_midi(f0_clean[valid_clean]))
    else:
        notes_clean = set()
    
    print(f"\n{'Metric':<30} {'Original':>12} {'Cleaned':>12} {'Change':>12}")
    print("-" * 66)
    
    def print_metric(name, orig, clean, fmt=".2f", higher_better=True):
        change = clean - orig
        if higher_better:
            emoji = "✅" if change > 0 else ("⚠️" if change < 0 else "➡️")
        else:
            emoji = "✅" if change < 0 else ("⚠️" if change > 0 else "➡️")
        print(f"{name:<30} {orig:>12{fmt}} {clean:>12{fmt}} {change:>+11{fmt}} {emoji}")
    
    print_metric("Voiced frames", n_valid_orig, n_valid_clean, "d", True)
    print_metric("Voiced %", 100*n_valid_orig/len(f0_orig), 100*n_valid_clean/len(f0_clean), ".1f", True)
    print_metric("Mean confidence", mean_conf_orig, mean_conf_clean, ".3f", True)
    print_metric("Pitch stability (std)", stability_orig, stability_clean, ".3f", False)
    print_metric("Unique notes", len(notes_orig), len(notes_clean), "d", False)  # Fewer = more stable
    
    print("\n📝 Notes detected:")
    print(f"   Original: {sorted(notes_orig)}")
    print(f"   Cleaned:  {sorted(notes_clean)}")
    
    # Overall verdict
    print("\n" + "=" * 60)
    print("🏆 VERDICT")
    print("=" * 60)
    
    improvements = 0
    if n_valid_clean >= n_valid_orig:
        improvements += 1
    if mean_conf_clean >= mean_conf_orig:
        improvements += 1
    if stability_clean <= stability_orig:
        improvements += 1
    if len(notes_clean) <= len(notes_orig):  # Fewer spurious notes
        improvements += 1
    
    if improvements >= 3:
        print("✅ IMPROVEMENT: Frequency cleanup helps pitch detection!")
    elif improvements >= 2:
        print("➡️ MARGINAL: Mixed results, may help in some cases")
    else:
        print("⚠️ NO IMPROVEMENT: Cleanup may not help this audio")
    
    return {
        'original': {
            'voiced_frames': n_valid_orig,
            'mean_confidence': mean_conf_orig,
            'stability': stability_orig,
            'unique_notes': len(notes_orig)
        },
        'cleaned': {
            'voiced_frames': n_valid_clean,
            'mean_confidence': mean_conf_clean,
            'stability': stability_clean,
            'unique_notes': len(notes_clean)
        }
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_freq_cleanup.py <audio_file>")
        sys.exit(1)
    
    test_pitch_accuracy_improvement(sys.argv[1])
