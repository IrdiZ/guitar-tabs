#!/usr/bin/env python3
"""
Alternative Pitch Detection Libraries for Guitar Tabs

Tests and compares alternative audio analysis libraries:
1. Essentia - MIR library with multiple pitch detection algorithms
2. Parselmouth (Praat) - Excellent pitch tracking from speech research
3. Librosa (existing) - pYIN, piptrack, CQT-based

This module provides pitch detectors that can be integrated into the
existing ensemble system for improved accuracy.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import warnings

# Import Essentia
try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False
    print("⚠️  Essentia not available")

# Import Parselmouth (Praat bindings)
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("⚠️  Parselmouth not available")

# Constants
GUITAR_MIN_HZ = 75   # Below E2 for drop tunings
GUITAR_MAX_HZ = 1400  # High frets on high E
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class DetectedNote:
    """A detected note from pitch analysis."""
    midi_note: int
    start_time: float
    end_time: float
    confidence: float
    frequency: float
    method: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)


@dataclass 
class PitchFrame:
    """Frame-by-frame pitch data."""
    time: float
    frequency: float
    confidence: float
    midi_note: int = 0
    
    def __post_init__(self):
        if self.frequency > 0 and self.midi_note == 0:
            self.midi_note = int(round(librosa.hz_to_midi(self.frequency)))


class EssentiaPitchDetector:
    """
    Pitch detection using Essentia's algorithms.
    
    Essentia provides several pitch detection methods:
    - PitchYin: Standard YIN algorithm
    - PitchYinFFT: FFT-based YIN (faster)
    - PitchMelodia: Predominant melody extraction
    - MultiPitchMelodia: Polyphonic pitch detection
    """
    
    def __init__(self, sr: int = 22050, hop_size: int = 512, frame_size: int = 2048):
        if not HAS_ESSENTIA:
            raise RuntimeError("Essentia not installed. Install with: pip install essentia")
        
        self.sr = sr
        self.hop_size = hop_size
        self.frame_size = frame_size
    
    def detect_yin(self, audio: np.ndarray) -> List[PitchFrame]:
        """
        Detect pitch using Essentia's YIN algorithm.
        YIN is the classic time-domain pitch detection method.
        """
        # Ensure mono float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Create algorithms
        pitch_yin = es.PitchYin(
            frameSize=self.frame_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ
        )
        
        frames = []
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            
            if len(frame) < self.frame_size:
                continue
            
            pitch, confidence = pitch_yin(frame)
            
            if confidence > 0.3 and GUITAR_MIN_HZ <= pitch <= GUITAR_MAX_HZ:
                frames.append(PitchFrame(
                    time=start / self.sr,
                    frequency=float(pitch),
                    confidence=float(confidence)
                ))
        
        return frames
    
    def detect_yin_fft(self, audio: np.ndarray) -> List[PitchFrame]:
        """
        Detect pitch using Essentia's FFT-based YIN.
        Faster than time-domain YIN, similar accuracy.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        pitch_yin_fft = es.PitchYinFFT(
            frameSize=self.frame_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ
        )
        
        # Need windowing and FFT for YinFFT
        windowing = es.Windowing(type='hann', size=self.frame_size)
        spectrum = es.Spectrum(size=self.frame_size)
        
        frames = []
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            
            if len(frame) < self.frame_size:
                continue
            
            windowed = windowing(frame)
            spec = spectrum(windowed)
            pitch, confidence = pitch_yin_fft(spec)
            
            if confidence > 0.3 and GUITAR_MIN_HZ <= pitch <= GUITAR_MAX_HZ:
                frames.append(PitchFrame(
                    time=start / self.sr,
                    frequency=float(pitch),
                    confidence=float(confidence)
                ))
        
        return frames
    
    def detect_melodia(self, audio: np.ndarray) -> List[PitchFrame]:
        """
        Detect predominant pitch using Melodia algorithm.
        
        Melodia is specifically designed for melody extraction from
        polyphonic audio. Excellent for guitar lead lines.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # PitchMelodia operates on the whole signal
        melodia = es.PitchMelodia(
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ,
            voicingTolerance=0.2,
            filterIterations=3
        )
        
        pitch_values, confidence_values = melodia(audio)
        
        frames = []
        for i, (pitch, conf) in enumerate(zip(pitch_values, confidence_values)):
            time = i * self.hop_size / self.sr
            
            if pitch > 0 and conf > 0.3:
                frames.append(PitchFrame(
                    time=time,
                    frequency=float(pitch),
                    confidence=float(conf)
                ))
        
        return frames
    
    def detect_multipitch_melodia(self, audio: np.ndarray) -> List[PitchFrame]:
        """
        Detect multiple simultaneous pitches using MultiPitchMelodia.
        Useful for chord detection.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        multipitch = es.MultiPitchMelodia(
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ
        )
        
        pitches_per_frame = multipitch(audio)
        
        frames = []
        for i, frame_pitches in enumerate(pitches_per_frame):
            time = i * self.hop_size / self.sr
            
            for pitch in frame_pitches:
                if pitch > 0 and GUITAR_MIN_HZ <= pitch <= GUITAR_MAX_HZ:
                    frames.append(PitchFrame(
                        time=time,
                        frequency=float(pitch),
                        confidence=0.7  # MultiPitch doesn't return confidence
                    ))
        
        return frames
    
    def detect_yin_probabilistic(self, audio: np.ndarray) -> List[PitchFrame]:
        """
        Detect pitch using probabilistic YIN (similar to pYIN).
        Uses HMM for smooth pitch tracking.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Frame-by-frame probabilistic pitch
        yin_probs = es.PitchYinProbabilities(
            frameSize=self.frame_size,
            sampleRate=self.sr,
            lowRMSThreshold=0.01
        )
        
        # HMM smoothing
        yin_hmm = es.PitchYinProbabilitiesHMM(
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ,
            sampleRate=self.sr
        )
        
        # Collect frame probabilities
        all_probs = []
        all_rms = []
        
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            
            if len(frame) < self.frame_size:
                continue
            
            probs, rms = yin_probs(frame)
            all_probs.append(probs)
            all_rms.append(rms)
        
        if not all_probs:
            return []
        
        # Convert to 2D array for HMM
        probs_array = np.array(all_probs)
        rms_array = np.array(all_rms)
        
        # Run HMM
        pitches, voicing = yin_hmm(probs_array, rms_array)
        
        frames = []
        for i, (pitch, voiced) in enumerate(zip(pitches, voicing)):
            if voiced > 0.5 and pitch > 0:
                time = i * self.hop_size / self.sr
                frames.append(PitchFrame(
                    time=time,
                    frequency=float(pitch),
                    confidence=float(voiced)
                ))
        
        return frames


class ParselMouthPitchDetector:
    """
    Pitch detection using Praat via Parselmouth.
    
    Praat is the gold standard for pitch analysis in phonetics research.
    Its autocorrelation-based pitch detection is highly accurate.
    """
    
    def __init__(self, sr: int = 22050):
        if not HAS_PARSELMOUTH:
            raise RuntimeError("Parselmouth not installed. Install with: pip install praat-parselmouth")
        
        self.sr = sr
    
    def detect_ac(self, audio: np.ndarray, time_step: float = 0.01) -> List[PitchFrame]:
        """
        Detect pitch using Praat's autocorrelation method.
        
        This is Praat's "To Pitch (ac)..." command, the most accurate
        method for clean recordings.
        """
        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
        
        # Extract pitch using autocorrelation
        pitch = sound.to_pitch_ac(
            time_step=time_step,
            pitch_floor=GUITAR_MIN_HZ,
            pitch_ceiling=GUITAR_MAX_HZ,
            very_accurate=True
        )
        
        frames = []
        for t in pitch.xs():
            freq = pitch.get_value_at_time(t)
            
            if freq and not np.isnan(freq):
                # Get strength (confidence proxy)
                strength = pitch.get_strength_at_time(t)
                
                if strength and not np.isnan(strength):
                    frames.append(PitchFrame(
                        time=t,
                        frequency=float(freq),
                        confidence=float(strength)
                    ))
        
        return frames
    
    def detect_cc(self, audio: np.ndarray, time_step: float = 0.01) -> List[PitchFrame]:
        """
        Detect pitch using Praat's cross-correlation method.
        
        Better for noisy recordings or recordings with harmonics.
        """
        sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
        
        pitch = sound.to_pitch_cc(
            time_step=time_step,
            pitch_floor=GUITAR_MIN_HZ,
            pitch_ceiling=GUITAR_MAX_HZ,
            very_accurate=True
        )
        
        frames = []
        for t in pitch.xs():
            freq = pitch.get_value_at_time(t)
            
            if freq and not np.isnan(freq):
                strength = pitch.get_strength_at_time(t)
                
                if strength and not np.isnan(strength):
                    frames.append(PitchFrame(
                        time=t,
                        frequency=float(freq),
                        confidence=float(strength)
                    ))
        
        return frames
    
    def detect_shs(self, audio: np.ndarray, time_step: float = 0.01) -> List[PitchFrame]:
        """
        Detect pitch using Subharmonic Summation (SHS).
        
        SHS is good for signals with strong harmonics, like guitar.
        It explicitly models the harmonic series.
        """
        sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
        
        # Use raw Praat command for SHS
        pitch = call(sound, "To Pitch (shs)...",
                     time_step,
                     GUITAR_MIN_HZ,
                     15,  # max candidates
                     GUITAR_MAX_HZ,
                     15,  # max subharmonics
                     0.84,  # compression factor
                     0.0,  # ceiling (0 = use pitch ceiling)
                     0.02  # points per octave
                     )
        
        frames = []
        for i in range(call(pitch, "Get number of frames")):
            t = call(pitch, "Get time from frame number...", i + 1)
            freq = call(pitch, "Get value in frame...", i + 1, "Hertz")
            
            if freq and not np.isnan(freq):
                frames.append(PitchFrame(
                    time=t,
                    frequency=float(freq),
                    confidence=0.7  # SHS doesn't provide confidence directly
                ))
        
        return frames


def frames_to_notes(
    frames: List[PitchFrame],
    min_duration: float = 0.05,
    pitch_stability_threshold: float = 0.5,
    method: str = ""
) -> List[DetectedNote]:
    """
    Convert frame-by-frame pitch data to note events.
    
    Groups consecutive frames with the same pitch into notes.
    """
    if not frames:
        return []
    
    notes = []
    
    # Sort by time
    frames = sorted(frames, key=lambda f: f.time)
    
    # Group by pitch
    current_note: Optional[Dict] = None
    
    for frame in frames:
        midi = frame.midi_note
        
        if current_note is None:
            # Start new note
            current_note = {
                'midi': midi,
                'start': frame.time,
                'end': frame.time,
                'confidences': [frame.confidence],
                'frequencies': [frame.frequency]
            }
        elif midi == current_note['midi'] or abs(midi - current_note['midi']) <= 1:
            # Continue note (allow ±1 semitone jitter)
            current_note['end'] = frame.time
            current_note['confidences'].append(frame.confidence)
            current_note['frequencies'].append(frame.frequency)
        else:
            # End current note, start new
            duration = current_note['end'] - current_note['start']
            if duration >= min_duration:
                notes.append(DetectedNote(
                    midi_note=current_note['midi'],
                    start_time=current_note['start'],
                    end_time=current_note['end'],
                    confidence=float(np.mean(current_note['confidences'])),
                    frequency=float(np.mean(current_note['frequencies'])),
                    method=method
                ))
            
            current_note = {
                'midi': midi,
                'start': frame.time,
                'end': frame.time,
                'confidences': [frame.confidence],
                'frequencies': [frame.frequency]
            }
    
    # Don't forget last note
    if current_note:
        duration = current_note['end'] - current_note['start']
        if duration >= min_duration:
            notes.append(DetectedNote(
                midi_note=current_note['midi'],
                start_time=current_note['start'],
                end_time=current_note['end'],
                confidence=float(np.mean(current_note['confidences'])),
                frequency=float(np.mean(current_note['frequencies'])),
                method=method
            ))
    
    return notes


def run_all_detectors(audio_path: str, sr: int = 22050) -> Dict[str, List[DetectedNote]]:
    """
    Run all available pitch detectors on an audio file.
    
    Returns a dict mapping method name to detected notes.
    """
    print(f"\n📂 Loading audio: {audio_path}")
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Apply harmonic separation
    print("🎵 Separating harmonic component...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
    
    results = {}
    
    # =====================
    # ESSENTIA DETECTORS
    # =====================
    if HAS_ESSENTIA:
        print("\n🔬 Running Essentia detectors...")
        essentia_det = EssentiaPitchDetector(sr=sr)
        
        methods = [
            ('essentia_yin', essentia_det.detect_yin),
            ('essentia_yin_fft', essentia_det.detect_yin_fft),
            ('essentia_melodia', essentia_det.detect_melodia),
            ('essentia_yin_prob', essentia_det.detect_yin_probabilistic),
            ('essentia_multipitch', essentia_det.detect_multipitch_melodia),
        ]
        
        for name, method in methods:
            try:
                start = time.time()
                frames = method(y_harmonic)
                elapsed = (time.time() - start) * 1000
                notes = frames_to_notes(frames, method=name)
                results[name] = notes
                print(f"   ✓ {name}: {len(notes)} notes ({elapsed:.0f}ms)")
            except Exception as e:
                print(f"   ✗ {name}: {e}")
    
    # =====================
    # PARSELMOUTH (PRAAT) DETECTORS
    # =====================
    if HAS_PARSELMOUTH:
        print("\n🔬 Running Parselmouth (Praat) detectors...")
        praat_det = ParselMouthPitchDetector(sr=sr)
        
        methods = [
            ('praat_ac', praat_det.detect_ac),
            ('praat_cc', praat_det.detect_cc),
            ('praat_shs', praat_det.detect_shs),
        ]
        
        for name, method in methods:
            try:
                start = time.time()
                frames = method(y_harmonic)
                elapsed = (time.time() - start) * 1000
                notes = frames_to_notes(frames, method=name)
                results[name] = notes
                print(f"   ✓ {name}: {len(notes)} notes ({elapsed:.0f}ms)")
            except Exception as e:
                print(f"   ✗ {name}: {e}")
    
    # =====================
    # LIBROSA (EXISTING) DETECTORS
    # =====================
    print("\n🔬 Running Librosa (existing) detectors...")
    
    # pYIN
    try:
        start = time.time()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_harmonic,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=sr,
            hop_length=512,
            fill_na=0.0
        )
        elapsed = (time.time() - start) * 1000
        
        frames = []
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
        for t, freq, voiced, conf in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and freq > 0 and conf > 0.3:
                frames.append(PitchFrame(time=t, frequency=freq, confidence=conf))
        
        notes = frames_to_notes(frames, method='librosa_pyin')
        results['librosa_pyin'] = notes
        print(f"   ✓ librosa_pyin: {len(notes)} notes ({elapsed:.0f}ms)")
    except Exception as e:
        print(f"   ✗ librosa_pyin: {e}")
    
    return results


def compare_results(results: Dict[str, List[DetectedNote]], reference: str = None) -> None:
    """
    Compare results from different detectors.
    
    If reference is provided, use it as ground truth for accuracy comparison.
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if not results:
        print("No results to compare!")
        return
    
    # Basic statistics
    print("\n📊 Detection Statistics:")
    print(f"{'Method':<25} {'Notes':>8} {'Avg Conf':>10} {'Avg Dur':>10} {'Pitch Range':>15}")
    print("-" * 70)
    
    for method, notes in sorted(results.items()):
        if not notes:
            print(f"{method:<25} {'0':>8}")
            continue
        
        avg_conf = np.mean([n.confidence for n in notes])
        avg_dur = np.mean([n.duration for n in notes])
        min_midi = min(n.midi_note for n in notes)
        max_midi = max(n.midi_note for n in notes)
        pitch_range = f"{NOTE_NAMES[min_midi % 12]}{min_midi // 12 - 1}-{NOTE_NAMES[max_midi % 12]}{max_midi // 12 - 1}"
        
        print(f"{method:<25} {len(notes):>8} {avg_conf:>10.3f} {avg_dur:>10.3f}s {pitch_range:>15}")
    
    # Find consensus notes (notes detected by multiple methods)
    print("\n🎵 Consensus Analysis:")
    
    # Collect all unique time windows
    all_notes = []
    for method, notes in results.items():
        for note in notes:
            all_notes.append((note.start_time, note.midi_note, method))
    
    # Group by time window (±50ms)
    time_tolerance = 0.05
    pitch_groups = {}
    
    for start, midi, method in all_notes:
        # Find or create group
        found_group = None
        for key in pitch_groups:
            if abs(key[0] - start) <= time_tolerance and abs(key[1] - midi) <= 1:
                found_group = key
                break
        
        if found_group:
            pitch_groups[found_group].add(method)
        else:
            pitch_groups[(start, midi)] = {method}
    
    # Count consensus levels
    consensus_counts = {}
    for methods in pitch_groups.values():
        n_methods = len(methods)
        consensus_counts[n_methods] = consensus_counts.get(n_methods, 0) + 1
    
    print(f"{'Methods Agreeing':<20} {'Count':>10}")
    print("-" * 30)
    for n_methods in sorted(consensus_counts.keys(), reverse=True):
        count = consensus_counts[n_methods]
        print(f"{n_methods:<20} {count:>10}")
    
    # High-consensus notes (detected by 3+ methods)
    high_consensus = [(key, methods) for key, methods in pitch_groups.items() if len(methods) >= 3]
    high_consensus.sort(key=lambda x: x[0][0])  # Sort by time
    
    if high_consensus:
        print(f"\n🎯 High-Consensus Notes (3+ methods agree): {len(high_consensus)}")
        print(f"{'Time':>8} {'Note':>8} {'Methods'}")
        print("-" * 50)
        for (start, midi), methods in high_consensus[:20]:  # Show first 20
            note_name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
            methods_str = ', '.join(sorted(methods))
            print(f"{start:>8.3f} {note_name:>8} {methods_str}")
        
        if len(high_consensus) > 20:
            print(f"   ... and {len(high_consensus) - 20} more")


def integrate_best_detectors() -> str:
    """
    Return code snippet for integrating the best-performing detectors
    into the existing ensemble system.
    """
    code = '''
# Add to ensemble_pitch.py to integrate alternative detectors:

# In EnsemblePitchDetector.__init__, add to available_methods check:
if HAS_ESSENTIA:
    methods.append('essentia_melodia')
    methods.append('essentia_yin_prob')

if HAS_PARSELMOUTH:
    methods.append('praat_ac')

# Add method weights:
self.config.weights.update({
    'essentia_melodia': 1.3,     # Excellent for melody
    'essentia_yin_prob': 1.1,    # Similar to pYIN
    'praat_ac': 1.2,             # High accuracy
})

# Add detector methods:
def _detect_essentia_melodia(self, y: np.ndarray) -> List[PitchCandidate]:
    from alternative_pitch_detection import EssentiaPitchDetector, PitchFrame
    det = EssentiaPitchDetector(sr=self.sr)
    frames = det.detect_melodia(y.astype(np.float32))
    return [
        PitchCandidate(
            time=f.time,
            midi_note=f.midi_note,
            frequency=f.frequency,
            confidence=f.confidence,
            method='essentia_melodia'
        )
        for f in frames
    ]

def _detect_praat_ac(self, y: np.ndarray) -> List[PitchCandidate]:
    from alternative_pitch_detection import ParselMouthPitchDetector
    det = ParselMouthPitchDetector(sr=self.sr)
    frames = det.detect_ac(y)
    return [
        PitchCandidate(
            time=f.time,
            midi_note=f.midi_note,
            frequency=f.frequency,
            confidence=f.confidence,
            method='praat_ac'
        )
        for f in frames
    ]
'''
    return code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Alternative Pitch Detection Libraries")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_all_detectors(args.audio_path, sr=args.sr)
    compare_results(results)
    
    if args.output:
        import json
        output_data = {
            method: [
                {
                    'midi': n.midi_note,
                    'name': n.name,
                    'start': n.start_time,
                    'end': n.end_time,
                    'duration': n.duration,
                    'confidence': n.confidence,
                    'frequency': n.frequency
                }
                for n in notes
            ]
            for method, notes in results.items()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Saved results to {args.output}")
    
    print("\n📝 Integration code:")
    print(integrate_best_detectors())
