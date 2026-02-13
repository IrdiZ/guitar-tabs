#!/usr/bin/env python3
"""
Test HMM-based probabilistic note detection on real audio.
"""

import numpy as np
import librosa
import json
from pathlib import Path

from probabilistic_hmm import (
    PitchHMM, HMMConfig, PitchObservation, DecodedNote,
    create_observations_from_pitch_track, hmm_decode_pitch_track,
    analyze_transition_statistics, print_hmm_diagnostics
)
from music_theory import Key, NOTE_NAMES


def extract_pitch_track(audio_path: str, sr: int = 22050) -> tuple:
    """
    Extract pitch track using multiple methods and combine.
    """
    print(f"📂 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    print(f"   Duration: {duration:.2f}s, SR: {sr}")
    
    # Use pyin for pitch detection (good for monophonic)
    print("🎤 Running pitch detection (pyin)...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('E2'),  # Low E on guitar
        fmax=librosa.note_to_hz('E6'),  # High notes
        sr=sr,
        frame_length=2048,
        hop_length=512
    )
    
    # Create time array
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    
    # Use voiced probabilities as confidence
    confidence = voiced_probs
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    confidence = np.nan_to_num(confidence, nan=0.0)
    
    print(f"   Extracted {len(f0)} frames")
    voiced_count = np.sum(f0 > 0)
    print(f"   Voiced frames: {voiced_count} ({100*voiced_count/len(f0):.1f}%)")
    
    return times, f0, confidence


def test_hmm_on_audio(audio_path: str):
    """
    Test HMM decoding on a real audio file.
    """
    print("\n" + "=" * 60)
    print("HMM PROBABILISTIC NOTE DETECTION TEST")
    print("=" * 60 + "\n")
    
    # Extract pitch track
    times, f0, confidence = extract_pitch_track(audio_path)
    
    # Configure HMM
    config = HMMConfig(
        verbose=True,
        freq_std_cents=30,          # Allow some pitch uncertainty
        octave_error_prob=0.15,     # Account for octave errors
        step_preference=0.5,        # Moderate preference for stepwise
        beam_width=30               # Keep top 30 candidates
    )
    
    # Decode using HMM
    print("\n🎯 Running HMM Viterbi decoding...")
    notes, detected_key = hmm_decode_pitch_track(
        times, f0, confidence,
        key=None,  # Auto-detect
        config=config
    )
    
    print(f"\n✅ Detected {len(notes)} notes in {detected_key.name}")
    
    # Analyze results
    if notes:
        print("\n📊 Note statistics:")
        midis = [n.midi for n in notes]
        print(f"   Range: {NOTE_NAMES[min(midis)%12]}{min(midis)//12-1} - {NOTE_NAMES[max(midis)%12]}{max(midis)//12-1}")
        print(f"   (MIDI {min(midis)} - {max(midis)})")
        
        durations = [n.duration for n in notes]
        print(f"   Duration range: {min(durations)*1000:.0f}ms - {max(durations)*1000:.0f}ms")
        print(f"   Mean duration: {np.mean(durations)*1000:.0f}ms")
        
        # Transition analysis
        stats = analyze_transition_statistics(notes)
        print(f"\n📈 Transition statistics:")
        print(f"   Stepwise motion (±2 semitones): {stats['stepwise_ratio']*100:.1f}%")
        print(f"   Leaps (>4 semitones): {stats['leap_ratio']*100:.1f}%")
        print(f"   Repeated notes: {stats['repeat_ratio']*100:.1f}%")
        print(f"   Mean interval: {stats['mean_interval']:.1f} semitones")
        
        # Print first 20 notes
        print(f"\n🎵 First 20 notes:")
        for i, note in enumerate(notes[:20]):
            print(f"   {i+1:2d}. {note.note_name:4s} @ {note.start_time:6.2f}s (dur: {note.duration*1000:4.0f}ms, conf: {note.confidence:.2f})")
        
        if len(notes) > 20:
            print(f"   ... and {len(notes) - 20} more notes")
    
    # Save results
    output = {
        'key': detected_key.name,
        'key_confidence': float(detected_key.confidence),
        'total_notes': len(notes),
        'notes': [
            {
                'midi': int(n.midi),
                'note_name': n.note_name,
                'start_time': float(n.start_time),
                'end_time': float(n.end_time),
                'duration': float(n.duration),
                'confidence': float(n.confidence)
            }
            for n in notes
        ]
    }
    
    output_path = Path(audio_path).stem + '_hmm_output.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved results to {output_path}")
    
    return notes, detected_key


def compare_with_naive(audio_path: str):
    """
    Compare HMM decoding with naive pitch-to-note conversion.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: HMM vs NAIVE DETECTION")
    print("=" * 60 + "\n")
    
    # Extract pitch track
    times, f0, confidence = extract_pitch_track(audio_path)
    
    # Naive approach: just convert each frame to MIDI
    print("\n📌 Naive approach (direct frequency-to-MIDI)...")
    naive_notes = []
    current_midi = None
    start_idx = 0
    
    for i, (t, freq, conf) in enumerate(zip(times, f0, confidence)):
        if freq <= 0 or conf < 0.3:
            if current_midi is not None:
                # End current note
                if times[i-1] - times[start_idx] > 0.05:
                    naive_notes.append({
                        'midi': current_midi,
                        'start': times[start_idx],
                        'end': times[i-1]
                    })
                current_midi = None
            continue
        
        midi = int(round(librosa.hz_to_midi(freq)))
        
        if midi != current_midi:
            if current_midi is not None:
                if times[i-1] - times[start_idx] > 0.05:
                    naive_notes.append({
                        'midi': current_midi,
                        'start': times[start_idx],
                        'end': times[i-1]
                    })
            current_midi = midi
            start_idx = i
    
    # Final note
    if current_midi is not None and times[-1] - times[start_idx] > 0.05:
        naive_notes.append({
            'midi': current_midi,
            'start': times[start_idx],
            'end': times[-1]
        })
    
    print(f"   Naive detected {len(naive_notes)} notes")
    
    # HMM approach
    print("\n🎯 HMM approach...")
    config = HMMConfig(verbose=False)
    hmm_notes, key = hmm_decode_pitch_track(times, f0, confidence, config=config)
    print(f"   HMM detected {len(hmm_notes)} notes")
    
    # Compare
    print("\n📊 Comparison:")
    print(f"   Naive note count: {len(naive_notes)}")
    print(f"   HMM note count: {len(hmm_notes)}")
    
    if naive_notes:
        naive_midis = [n['midi'] for n in naive_notes]
        print(f"   Naive MIDI range: {min(naive_midis)} - {max(naive_midis)}")
    
    if hmm_notes:
        hmm_midis = [n.midi for n in hmm_notes]
        print(f"   HMM MIDI range: {min(hmm_midis)} - {max(hmm_midis)}")
    
    # Check for out-of-scale notes
    if key and hmm_notes:
        scale_pcs = set(key.get_scale_notes())
        naive_out_of_scale = sum(1 for n in naive_notes if n['midi'] % 12 not in scale_pcs) if naive_notes else 0
        hmm_out_of_scale = sum(1 for n in hmm_notes if n.midi % 12 not in scale_pcs)
        
        print(f"\n   Key: {key.name}")
        print(f"   Naive out-of-scale: {naive_out_of_scale}/{len(naive_notes) if naive_notes else 0}")
        print(f"   HMM out-of-scale: {hmm_out_of_scale}/{len(hmm_notes)}")
    
    return naive_notes, hmm_notes


if __name__ == '__main__':
    import sys
    
    # Default test file
    audio_path = "/root/.clawdbot/media/inbound/7099892f-2b38-41fb-9813-ea8055cbc385.mp3"
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    
    # Run tests
    test_hmm_on_audio(audio_path)
    compare_with_naive(audio_path)
