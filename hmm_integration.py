#!/usr/bin/env python3
"""
Integration of HMM probabilistic model with existing ensemble pitch detection.

This combines:
1. Ensemble pitch detection (handles distortion, multiple methods)
2. HMM Viterbi decoding (handles uncertainty, enforces musical coherence)
"""

import numpy as np
import librosa
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from probabilistic_hmm import (
    PitchHMM, HMMConfig, PitchObservation, DecodedNote,
    hmm_decode_pitch_track, analyze_transition_statistics,
    NoteTransitionModel, NotePrior
)
from music_theory import Key, NOTE_NAMES, detect_key


def run_ensemble_pitch_detection(audio_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run ensemble pitch detection using existing module.
    Returns dict with 'times', 'f0', 'confidence'.
    """
    try:
        from ensemble_pitch import EnsemblePitchDetector, EnsembleConfig
        
        config = EnsembleConfig(
            methods=['pyin', 'yin', 'crepe_small'],  # Use available methods
            min_confidence=0.2,  # Lower threshold for distorted audio
            octave_voting=True,
            verbose=verbose
        )
        
        detector = EnsemblePitchDetector(config)
        result = detector.detect(audio_path)
        
        return {
            'times': result.times,
            'f0': result.f0_hz,
            'confidence': result.confidence,
            'method': 'ensemble'
        }
    except Exception as e:
        if verbose:
            print(f"⚠️ Ensemble detection failed: {e}, falling back to pyin")
        return run_pyin_detection(audio_path, verbose)


def run_pyin_detection(audio_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run basic pyin pitch detection.
    """
    y, sr = librosa.load(audio_path, sr=22050)
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('E2'),
        fmax=librosa.note_to_hz('E6'),
        sr=sr,
        frame_length=2048,
        hop_length=512
    )
    
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    f0 = np.nan_to_num(f0, nan=0.0)
    confidence = np.nan_to_num(voiced_probs, nan=0.0)
    
    return {
        'times': times,
        'f0': f0,
        'confidence': confidence,
        'method': 'pyin'
    }


def run_yin_detection(audio_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run YIN pitch detection (better for distorted audio).
    """
    try:
        from yin_pitch import yin_pitch_detection, YinConfig
        
        y, sr = librosa.load(audio_path, sr=22050)
        
        config = YinConfig(
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            threshold=0.15,  # Slightly relaxed for distortion
            confidence_threshold=0.2
        )
        
        # YIN returns (f0, voiced_flag, confidence) tuple
        f0, voiced, confidence = yin_pitch_detection(y, sr, config)
        
        # Create time array
        times = np.arange(len(f0)) * config.hop_length / sr
        
        # Zero out unvoiced frames
        f0 = np.where(voiced, f0, 0.0)
        
        return {
            'times': times,
            'f0': f0,
            'confidence': confidence,
            'method': 'yin'
        }
    except Exception as e:
        if verbose:
            print(f"⚠️ YIN detection failed: {e}")
        return run_pyin_detection(audio_path, verbose)


def interpolate_pitch_gaps(
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray,
    max_gap_frames: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate short gaps in pitch detection.
    
    This helps when pitch detection drops out briefly on sustained notes.
    """
    f0_filled = f0.copy()
    conf_filled = confidence.copy()
    
    # Find gaps (f0 == 0 or low confidence)
    valid = (f0 > 0) & (confidence > 0.1)
    
    gap_start = None
    for i in range(len(f0)):
        if not valid[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_length = i - gap_start
                
                # Only fill short gaps with consistent neighbors
                if gap_length <= max_gap_frames and gap_start > 0:
                    before_f0 = f0[gap_start - 1]
                    after_f0 = f0[i]
                    
                    # Check if neighbors are similar (within 2 semitones)
                    if before_f0 > 0 and after_f0 > 0:
                        before_midi = librosa.hz_to_midi(before_f0)
                        after_midi = librosa.hz_to_midi(after_f0)
                        
                        if abs(before_midi - after_midi) <= 2:
                            # Linear interpolation
                            for j in range(gap_start, i):
                                t = (j - gap_start + 1) / (gap_length + 1)
                                f0_filled[j] = before_f0 * (1 - t) + after_f0 * t
                                conf_filled[j] = 0.3  # Lower confidence for interpolated
                
                gap_start = None
    
    return f0_filled, conf_filled


def build_transition_priors_from_audio(
    audio_path: str,
    verbose: bool = True
) -> Tuple[Dict[int, Dict[int, float]], Key]:
    """
    Build transition probability priors from initial note detection.
    
    This does a first pass to learn the specific transition patterns
    in this piece before running full HMM decoding.
    """
    # Quick detection pass
    pitch_data = run_pyin_detection(audio_path, verbose=False)
    
    # Convert to rough notes
    times = pitch_data['times']
    f0 = pitch_data['f0']
    confidence = pitch_data['confidence']
    
    # Get MIDI values for voiced frames
    midis = []
    for freq, conf in zip(f0, confidence):
        if freq > 0 and conf > 0.2:
            midi = int(round(librosa.hz_to_midi(freq)))
            midis.append(midi)
    
    # Detect key from distribution
    if midis:
        class MockNote:
            def __init__(self, midi):
                self.midi = midi
                self.duration = 1.0
        
        notes = [MockNote(m) for m in midis]
        key = detect_key(notes)
    else:
        key = Key(0, 'major', 0.5)
    
    # Build transition counts
    transitions = {}
    for i in range(len(midis) - 1):
        from_m = midis[i]
        to_m = midis[i + 1]
        
        if from_m not in transitions:
            transitions[from_m] = {}
        if to_m not in transitions[from_m]:
            transitions[from_m][to_m] = 0
        transitions[from_m][to_m] += 1
    
    if verbose and transitions:
        total = sum(sum(d.values()) for d in transitions.values())
        print(f"📊 Learned {total} transitions from first pass")
    
    return transitions, key


def hmm_transcribe_audio(
    audio_path: str,
    config: Optional[HMMConfig] = None,
    pitch_method: str = 'pyin',
    interpolate_gaps: bool = True,
    learn_transitions: bool = True,
    verbose: bool = True
) -> Tuple[List[DecodedNote], Key, Dict[str, Any]]:
    """
    Full HMM-based audio transcription.
    
    Args:
        audio_path: Path to audio file
        config: HMM configuration
        pitch_method: 'pyin', 'yin', or 'ensemble'
        interpolate_gaps: Fill short gaps in pitch detection
        learn_transitions: Learn transitions from first pass
        verbose: Print progress
        
    Returns:
        Tuple of (notes, key, metadata)
    """
    config = config or HMMConfig(verbose=verbose)
    
    if verbose:
        print(f"\n{'='*60}")
        print("HMM PROBABILISTIC TRANSCRIPTION")
        print('='*60)
        print(f"\n📂 Input: {audio_path}")
    
    # Step 1: Initial key/transition learning
    learned_transitions = None
    initial_key = None
    
    if learn_transitions:
        if verbose:
            print("\n🔬 Phase 1: Learning from audio...")
        learned_transitions, initial_key = build_transition_priors_from_audio(
            audio_path, verbose=verbose
        )
        if verbose and initial_key:
            print(f"   Initial key estimate: {initial_key.name}")
    
    # Step 2: Pitch detection
    if verbose:
        print(f"\n🎤 Phase 2: Pitch detection ({pitch_method})...")
    
    if pitch_method == 'ensemble':
        pitch_data = run_ensemble_pitch_detection(audio_path, verbose=verbose)
    elif pitch_method == 'yin':
        pitch_data = run_yin_detection(audio_path, verbose=verbose)
    else:
        pitch_data = run_pyin_detection(audio_path, verbose=verbose)
    
    times = pitch_data['times']
    f0 = pitch_data['f0']
    confidence = pitch_data['confidence']
    
    if verbose:
        voiced = np.sum((f0 > 0) & (confidence > 0.1))
        print(f"   {len(f0)} frames, {voiced} voiced ({100*voiced/len(f0):.1f}%)")
    
    # Step 3: Interpolate gaps
    if interpolate_gaps:
        f0, confidence = interpolate_pitch_gaps(times, f0, confidence)
        if verbose:
            new_voiced = np.sum((f0 > 0) & (confidence > 0.1))
            print(f"   After gap filling: {new_voiced} voiced")
    
    # Step 4: Create HMM
    hmm = PitchHMM(key=initial_key, config=config)
    
    # Step 5: Inject learned transitions
    if learned_transitions and hmm.transition_model:
        for from_m, to_dict in learned_transitions.items():
            for to_m, count in to_dict.items():
                hmm.transition_model._counts[from_m][to_m] = count
        hmm.transition_model._learned = True
        if verbose:
            print("   Injected learned transitions")
    
    # Step 6: Create observations
    if verbose:
        print(f"\n🎯 Phase 3: Viterbi decoding...")
    
    observations = []
    for t, freq, conf in zip(times, f0, confidence):
        if freq > 0 and conf > 0.05:  # Very low threshold
            observations.append(PitchObservation(
                time=float(t),
                frequency=float(freq),
                confidence=float(conf)
            ))
    
    if verbose:
        print(f"   {len(observations)} observations for HMM")
    
    # Step 7: Run Viterbi
    notes = hmm.decode_to_notes(
        observations,
        min_note_duration=0.04,  # 40ms minimum
        merge_threshold_cents=50
    )
    
    final_key = hmm.key
    
    # Step 8: Results
    if verbose:
        print(f"\n✅ Results:")
        print(f"   Key: {final_key.name} (confidence: {final_key.confidence:.2f})")
        print(f"   Notes: {len(notes)}")
        
        if notes:
            midis = [n.midi for n in notes]
            print(f"   Range: {NOTE_NAMES[min(midis)%12]}{min(midis)//12-1} - {NOTE_NAMES[max(midis)%12]}{max(midis)//12-1}")
            
            stats = analyze_transition_statistics(notes)
            print(f"   Stepwise motion: {stats['stepwise_ratio']*100:.1f}%")
            
            # Check scale conformance
            scale_pcs = set(final_key.get_scale_notes())
            in_scale = sum(1 for n in notes if n.midi % 12 in scale_pcs)
            print(f"   In-scale notes: {in_scale}/{len(notes)} ({100*in_scale/len(notes):.1f}%)")
    
    metadata = {
        'pitch_method': pitch_data.get('method', pitch_method),
        'observations': len(observations),
        'interpolated_gaps': interpolate_gaps,
        'learned_transitions': learn_transitions
    }
    
    return notes, final_key, metadata


def notes_to_tab_string(notes: List[DecodedNote], key: Key) -> str:
    """
    Convert decoded notes to a simple tab string representation.
    """
    if not notes:
        return "No notes detected."
    
    lines = [
        f"Key: {key.name}",
        f"Notes: {len(notes)}",
        "",
        "Time     Note   MIDI  Dur(ms)  Conf",
        "-" * 45
    ]
    
    for note in notes:
        lines.append(
            f"{note.start_time:7.2f}  {note.note_name:5s}  {note.midi:3d}   "
            f"{note.duration*1000:6.0f}    {note.confidence:.2f}"
        )
    
    return '\n'.join(lines)


def main():
    """
    Main entry point for HMM transcription.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HMM Probabilistic Note Detection')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('--method', choices=['pyin', 'yin', 'ensemble'], default='pyin',
                        help='Pitch detection method')
    parser.add_argument('--no-interpolate', action='store_true',
                        help='Disable gap interpolation')
    parser.add_argument('--no-learn', action='store_true',
                        help='Disable transition learning')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    
    config = HMMConfig(
        verbose=not args.quiet,
        freq_std_cents=30,
        octave_error_prob=0.15,
        beam_width=30
    )
    
    notes, key, metadata = hmm_transcribe_audio(
        args.audio,
        config=config,
        pitch_method=args.method,
        interpolate_gaps=not args.no_interpolate,
        learn_transitions=not args.no_learn,
        verbose=not args.quiet
    )
    
    # Print results
    if not args.quiet:
        print("\n" + notes_to_tab_string(notes, key))
    
    # Save output
    if args.output:
        output = {
            'key': key.name,
            'key_confidence': float(key.confidence),
            'metadata': metadata,
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
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
