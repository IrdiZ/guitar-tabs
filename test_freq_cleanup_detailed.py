#!/usr/bin/env python3
"""
Detailed test: Test individual cleanup steps and different presets.
"""

import numpy as np
import librosa
from freq_cleanup import (
    freq_domain_cleanup, FreqCleanupConfig,
    remove_dc_offset, apply_bandpass_filter, apply_spectral_whitening,
    reduce_noise_floor, enhance_transients
)
from pitch_accuracy import detect_pitches_accurate
import sys


def evaluate_pitch_quality(y, sr, name=""):
    """Evaluate pitch detection quality."""
    f0, conf, pitches = detect_pitches_accurate(y, sr, verbose=False)
    
    valid = f0 > 0
    n_valid = np.sum(valid)
    
    if n_valid > 0:
        mean_conf = np.mean(conf[valid])
        midi = librosa.hz_to_midi(f0[valid])
        stability = np.std(np.diff(midi))
        unique_notes = len(set(int(round(m)) for m in midi))
    else:
        mean_conf = 0
        stability = float('inf')
        unique_notes = 0
    
    return {
        'name': name,
        'voiced_frames': n_valid,
        'voiced_pct': 100 * n_valid / len(f0),
        'mean_confidence': mean_conf,
        'stability': stability,
        'unique_notes': unique_notes
    }


def test_individual_steps(audio_path: str):
    """Test each processing step individually."""
    print("=" * 70)
    print("🔬 INDIVIDUAL PROCESSING STEP TEST")
    print("=" * 70)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    print(f"Audio: {audio_path}")
    print(f"Duration: {len(y)/sr:.2f}s")
    
    results = []
    
    # 1. Original
    print("\n📊 Testing: Original audio...")
    results.append(evaluate_pitch_quality(y, sr, "Original"))
    
    # 2. DC offset only
    print("📊 Testing: DC offset removal...")
    y_dc = remove_dc_offset(y.copy())
    results.append(evaluate_pitch_quality(y_dc, sr, "DC offset removed"))
    
    # 3. Bandpass only
    print("📊 Testing: Bandpass filter (80-5000Hz)...")
    y_bp = apply_bandpass_filter(y.copy(), sr, 80, 5000)
    results.append(evaluate_pitch_quality(y_bp, sr, "Bandpass 80-5000Hz"))
    
    # 4. Narrower bandpass (guitar fundamentals)
    print("📊 Testing: Bandpass filter (80-2000Hz)...")
    y_bp2 = apply_bandpass_filter(y.copy(), sr, 80, 2000)
    results.append(evaluate_pitch_quality(y_bp2, sr, "Bandpass 80-2000Hz"))
    
    # 5. Spectral whitening only
    print("📊 Testing: Spectral whitening only...")
    y_white = apply_spectral_whitening(y.copy(), sr, relaxation=0.9)
    results.append(evaluate_pitch_quality(y_white, sr, "Spectral whitening"))
    
    # 6. Gentle whitening
    print("📊 Testing: Gentle spectral whitening...")
    y_white_gentle = apply_spectral_whitening(y.copy(), sr, relaxation=0.5)
    results.append(evaluate_pitch_quality(y_white_gentle, sr, "Gentle whitening"))
    
    # 7. Noise reduction only
    print("📊 Testing: Noise floor reduction...")
    y_noise = reduce_noise_floor(y.copy(), sr, threshold_db=-40)
    results.append(evaluate_pitch_quality(y_noise, sr, "Noise reduction -40dB"))
    
    # 8. Transient enhancement only
    print("📊 Testing: Transient enhancement...")
    y_trans = enhance_transients(y.copy(), sr, strength=0.5)
    results.append(evaluate_pitch_quality(y_trans, sr, "Transient enhance"))
    
    # 9. Default preset
    print("📊 Testing: Full default preset...")
    config = FreqCleanupConfig.default()
    y_default = freq_domain_cleanup(y.copy(), sr, config, verbose=False)
    results.append(evaluate_pitch_quality(y_default, sr, "Full default"))
    
    # 10. Gentle preset
    print("📊 Testing: Gentle preset...")
    config = FreqCleanupConfig.gentle()
    y_gentle = freq_domain_cleanup(y.copy(), sr, config, verbose=False)
    results.append(evaluate_pitch_quality(y_gentle, sr, "Full gentle"))
    
    # 11. Custom: Bandpass + DC only (minimal)
    print("📊 Testing: Minimal (DC + bandpass only)...")
    y_min = remove_dc_offset(y.copy())
    y_min = apply_bandpass_filter(y_min, sr, 80, 5000)
    results.append(evaluate_pitch_quality(y_min, sr, "DC + bandpass only"))
    
    # 12. Bandpass + noise reduction
    print("📊 Testing: Bandpass + noise reduction...")
    y_bn = apply_bandpass_filter(y.copy(), sr, 80, 5000)
    y_bn = reduce_noise_floor(y_bn, sr, threshold_db=-35)
    results.append(evaluate_pitch_quality(y_bn, sr, "Bandpass + noise"))
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Voiced%':>8} {'Conf':>7} {'Stab':>7} {'Notes':>6}")
    print("-" * 70)
    
    # Sort by a composite score
    orig = results[0]
    
    for r in results:
        # Calculate improvement vs original
        conf_diff = r['mean_confidence'] - orig['mean_confidence']
        stab_diff = orig['stability'] - r['stability']  # Lower is better
        
        # Score: confidence matters most, then stability
        score = conf_diff * 10 + stab_diff * 2
        r['score'] = score
        
        # Mark best/worst
        marker = ""
        if r['name'] != "Original":
            if r['mean_confidence'] > orig['mean_confidence'] and r['stability'] < orig['stability']:
                marker = " ✅"
            elif r['mean_confidence'] < orig['mean_confidence'] - 0.02:
                marker = " ⚠️"
        
        print(f"{r['name']:<25} {r['voiced_pct']:>7.1f}% {r['mean_confidence']:>7.3f} "
              f"{r['stability']:>7.3f} {r['unique_notes']:>6}{marker}")
    
    # Find best method
    best = max(results[1:], key=lambda r: r['score'])
    worst = min(results[1:], key=lambda r: r['score'])
    
    print("\n" + "=" * 70)
    print("🏆 RECOMMENDATIONS")
    print("=" * 70)
    print(f"✅ Best method: {best['name']}")
    print(f"   Confidence: {best['mean_confidence']:.3f}, Stability: {best['stability']:.3f}")
    print(f"⚠️ Avoid: {worst['name']}")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_freq_cleanup_detailed.py <audio_file>")
        sys.exit(1)
    
    test_individual_steps(sys.argv[1])
