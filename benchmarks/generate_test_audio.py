#!/usr/bin/env python3
"""
Synthetic Guitar Audio Generator for Benchmarking

Generates test audio files with KNOWN ground truth notes for testing
pitch detection accuracy. Creates various test cases:
1. Single pure tones (baseline)
2. Single guitar-like notes (with harmonics)
3. Simple sequences
4. Polyphonic chords
5. Fast passages
6. Edge cases (very high/low notes, quiet notes)
"""

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os
from pathlib import Path

# Guitar tuning (standard) - MIDI note numbers
GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def midi_to_name(midi_note: int) -> str:
    """Convert MIDI note number to note name."""
    octave = (midi_note // 12) - 1
    note = NOTE_NAMES[midi_note % 12]
    return f"{note}{octave}"


@dataclass
class GroundTruthNote:
    """A note in the ground truth."""
    midi_note: int
    start_time: float
    end_time: float
    velocity: float = 1.0
    string: Optional[int] = None
    fret: Optional[int] = None
    
    def to_dict(self):
        return {
            'midi_note': self.midi_note,
            'note_name': midi_to_name(self.midi_note),
            'frequency_hz': midi_to_freq(self.midi_note),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'velocity': self.velocity,
            'string': self.string,
            'fret': self.fret
        }


class GuitarSynthesizer:
    """
    Synthesizes guitar-like sounds using the Karplus-Strong algorithm
    with additional harmonics modeling.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def generate_pure_tone(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 0.8
    ) -> np.ndarray:
        """Generate a pure sine wave."""
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # Add slight fade in/out to avoid clicks
        envelope = np.ones_like(t)
        fade_samples = min(int(0.01 * self.sr), len(t) // 4)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        return amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    
    def generate_guitar_note(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 0.8,
        pluck_position: float = 0.2,  # 0-1, where on string it's plucked
        decay_rate: float = 0.996
    ) -> np.ndarray:
        """
        Generate a guitar-like note using Karplus-Strong synthesis
        with harmonic enrichment.
        """
        # Karplus-Strong algorithm
        n_samples = int(self.sr * duration)
        delay_length = int(self.sr / frequency)
        
        if delay_length < 2:
            return self.generate_pure_tone(frequency, duration, amplitude)
        
        # Initialize with noise burst (simulates pluck)
        noise = np.random.uniform(-1, 1, delay_length)
        
        # Apply pluck position filtering (affects harmonic content)
        # Plucking near bridge = more harmonics
        if pluck_position < 0.1:
            # Near bridge - keep more high frequencies
            pass
        else:
            # Apply low-pass based on pluck position
            filter_size = max(2, int(pluck_position * 5))
            noise = np.convolve(noise, np.ones(filter_size)/filter_size, mode='same')
        
        # Generate output using delay line
        output = np.zeros(n_samples)
        buffer = noise.copy()
        
        for i in range(n_samples):
            idx = i % delay_length
            output[i] = buffer[idx]
            
            # Karplus-Strong averaging filter with decay
            next_idx = (idx + 1) % delay_length
            buffer[idx] = decay_rate * 0.5 * (buffer[idx] + buffer[next_idx])
        
        # Add harmonics for richer sound
        harmonics = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]  # Harmonic amplitudes
        for i, harm_amp in enumerate(harmonics[1:], start=2):
            if frequency * i < self.sr / 2:  # Below Nyquist
                t = np.arange(n_samples) / self.sr
                harmonic = harm_amp * np.sin(2 * np.pi * frequency * i * t)
                # Apply decay envelope
                decay_env = np.exp(-3 * t / duration)
                output += harmonic * decay_env
        
        # Normalize and apply amplitude
        output = output / (np.max(np.abs(output)) + 1e-10) * amplitude
        
        # Apply guitar-like envelope: fast attack, slow decay
        envelope = self._guitar_envelope(n_samples, duration)
        output *= envelope
        
        return output
    
    def _guitar_envelope(self, n_samples: int, duration: float) -> np.ndarray:
        """Create a guitar-like ADSR envelope."""
        attack_time = 0.005  # 5ms attack
        decay_time = 0.1     # 100ms decay to sustain level
        sustain_level = 0.7
        release_time = 0.05  # 50ms release
        
        envelope = np.ones(n_samples)
        
        # Attack
        attack_samples = int(attack_time * self.sr)
        if attack_samples > 0 and attack_samples < n_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay to sustain
        decay_samples = int(decay_time * self.sr)
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, n_samples)
        if decay_end > decay_start:
            envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_end - decay_start)
        
        # Sustain with gradual decay
        sustain_start = decay_end
        if sustain_start < n_samples:
            sustain_duration = (n_samples - sustain_start) / self.sr
            t = np.arange(n_samples - sustain_start) / self.sr
            envelope[sustain_start:] = sustain_level * np.exp(-2 * t / max(duration, 0.1))
        
        # Release
        release_samples = int(release_time * self.sr)
        if release_samples > 0 and release_samples < n_samples:
            envelope[-release_samples:] *= np.linspace(1, 0, release_samples)
        
        return envelope
    
    def generate_sequence(
        self,
        notes: List[GroundTruthNote],
        add_noise: float = 0.0,
        use_guitar_synth: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Generate audio from a sequence of notes.
        
        Returns:
            Tuple of (audio array, total duration)
        """
        if not notes:
            return np.zeros(self.sr), 1.0
        
        # Find total duration needed
        total_duration = max(n.end_time for n in notes) + 0.5  # Add buffer
        n_samples = int(total_duration * self.sr)
        audio = np.zeros(n_samples)
        
        for note in notes:
            freq = midi_to_freq(note.midi_note)
            duration = note.end_time - note.start_time
            
            if use_guitar_synth:
                note_audio = self.generate_guitar_note(freq, duration, note.velocity)
            else:
                note_audio = self.generate_pure_tone(freq, duration, note.velocity)
            
            # Place in output buffer
            start_sample = int(note.start_time * self.sr)
            end_sample = start_sample + len(note_audio)
            
            if end_sample <= n_samples:
                audio[start_sample:end_sample] += note_audio
            else:
                audio[start_sample:n_samples] += note_audio[:n_samples - start_sample]
        
        # Add noise if requested
        if add_noise > 0:
            noise = np.random.normal(0, add_noise, n_samples)
            audio += noise
        
        # Normalize to prevent clipping
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.95:
            audio = audio / max_amp * 0.95
        
        return audio, total_duration


class TestCaseGenerator:
    """Generates various test cases for benchmarking."""
    
    def __init__(self, output_dir: str, sr: int = 22050):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.synth = GuitarSynthesizer(sr)
    
    def save_test_case(
        self,
        name: str,
        audio: np.ndarray,
        ground_truth: List[GroundTruthNote],
        metadata: dict = None
    ):
        """Save audio and ground truth JSON."""
        audio_path = self.output_dir / f"{name}.wav"
        json_path = self.output_dir / f"{name}.json"
        
        sf.write(str(audio_path), audio, self.sr)
        
        gt_data = {
            'name': name,
            'sample_rate': self.sr,
            'duration': len(audio) / self.sr,
            'notes': [n.to_dict() for n in ground_truth],
            'metadata': metadata or {}
        }
        
        with open(json_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        print(f"  ✓ {name}: {len(ground_truth)} notes")
        return audio_path, json_path
    
    def generate_single_notes(self):
        """Generate individual note tests covering guitar range."""
        print("\n📝 Generating single note tests...")
        
        # Test notes across guitar range (E2 to E6)
        test_notes = [
            (40, "E2_low_e_open"),
            (45, "A2_a_open"),
            (50, "D3_d_open"),
            (55, "G3_g_open"),
            (59, "B3_b_open"),
            (64, "E4_high_e_open"),
            (52, "E3_low_e_12th"),
            (69, "A4_high_e_5th"),
            (76, "E5_high_e_12th"),
            (81, "A5_high_e_17th"),
        ]
        
        for midi_note, name in test_notes:
            gt = [GroundTruthNote(midi_note, 0.1, 1.1, velocity=0.8)]
            audio, _ = self.synth.generate_sequence(gt, use_guitar_synth=True)
            self.save_test_case(f"single_{name}", audio, gt, {'type': 'single_note'})
        
        # Pure tones for baseline comparison
        print("\n📝 Generating pure tone baselines...")
        for midi_note, name in test_notes[:6]:
            gt = [GroundTruthNote(midi_note, 0.1, 1.1, velocity=0.8)]
            audio, _ = self.synth.generate_sequence(gt, use_guitar_synth=False)
            self.save_test_case(f"pure_{name}", audio, gt, {'type': 'pure_tone'})
    
    def generate_sequences(self):
        """Generate melodic sequences."""
        print("\n📝 Generating sequence tests...")
        
        # Simple ascending scale (C major in first position)
        c_major = [48, 50, 52, 53, 55, 57, 59, 60]  # C3 to C4
        notes = []
        for i, midi in enumerate(c_major):
            notes.append(GroundTruthNote(midi, i * 0.5, (i + 1) * 0.5 - 0.05))
        audio, _ = self.synth.generate_sequence(notes)
        self.save_test_case("seq_c_major_scale", audio, notes, {'type': 'sequence', 'scale': 'C_major'})
        
        # Chromatic passage
        chromatic = list(range(48, 61))  # C3 to C4 chromatic
        notes = []
        for i, midi in enumerate(chromatic):
            notes.append(GroundTruthNote(midi, i * 0.3, (i + 1) * 0.3 - 0.02))
        audio, _ = self.synth.generate_sequence(notes)
        self.save_test_case("seq_chromatic", audio, notes, {'type': 'sequence', 'scale': 'chromatic'})
        
        # Pentatonic lick
        pentatonic = [52, 55, 57, 59, 62, 64, 62, 59, 57, 55, 52]
        notes = []
        for i, midi in enumerate(pentatonic):
            notes.append(GroundTruthNote(midi, i * 0.2, (i + 1) * 0.2 - 0.02))
        audio, _ = self.synth.generate_sequence(notes)
        self.save_test_case("seq_pentatonic_lick", audio, notes, {'type': 'sequence', 'scale': 'pentatonic'})
        
        # Fast passage (16th notes at 120 BPM)
        fast_notes = [48, 52, 55, 60, 55, 52, 48, 52] * 2
        notes = []
        note_duration = 60 / 120 / 4  # 16th notes at 120 BPM
        for i, midi in enumerate(fast_notes):
            notes.append(GroundTruthNote(midi, i * note_duration, (i + 1) * note_duration - 0.01))
        audio, _ = self.synth.generate_sequence(notes)
        self.save_test_case("seq_fast_16ths", audio, notes, {'type': 'sequence', 'tempo': 120})
    
    def generate_chords(self):
        """Generate polyphonic chord tests."""
        print("\n📝 Generating chord tests...")
        
        # Common guitar chords (MIDI notes for each)
        chords = {
            'E_major': [40, 47, 52, 56, 59, 64],     # E2 B2 E3 G#3 B3 E4
            'A_major': [45, 52, 57, 61, 64],         # A2 E3 A3 C#4 E4
            'D_major': [50, 57, 62, 66],             # D3 A3 D4 F#4
            'G_major': [43, 47, 50, 55, 59, 67],     # G2 B2 D3 G3 B3 G4
            'C_major': [48, 52, 55, 60, 64],         # C3 E3 G3 C4 E4
            'Am': [45, 52, 57, 60, 64],              # A2 E3 A3 C4 E4
            'Em': [40, 47, 52, 55, 59, 64],          # E2 B2 E3 G3 B3 E4
        }
        
        for chord_name, midi_notes in chords.items():
            notes = [GroundTruthNote(midi, 0.1, 1.5, velocity=0.7) for midi in midi_notes]
            audio, _ = self.synth.generate_sequence(notes)
            self.save_test_case(f"chord_{chord_name}", audio, notes, {'type': 'chord', 'chord': chord_name})
        
        # Chord progression: Em - G - D - C
        print("\n📝 Generating chord progression...")
        progression = [
            ('Em', [40, 47, 52, 55, 59, 64]),
            ('G', [43, 47, 50, 55, 59, 67]),
            ('D', [50, 57, 62, 66]),
            ('C', [48, 52, 55, 60, 64]),
        ]
        
        notes = []
        for i, (name, midi_notes) in enumerate(progression):
            start = i * 1.5
            end = (i + 1) * 1.5 - 0.1
            for midi in midi_notes:
                notes.append(GroundTruthNote(midi, start, end, velocity=0.7))
        
        audio, _ = self.synth.generate_sequence(notes)
        self.save_test_case("chord_progression_em_g_d_c", audio, notes, {'type': 'progression'})
    
    def generate_edge_cases(self):
        """Generate edge case tests."""
        print("\n📝 Generating edge case tests...")
        
        # Very quiet note
        gt = [GroundTruthNote(52, 0.1, 1.1, velocity=0.15)]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_quiet_note", audio, gt, {'type': 'edge_case', 'case': 'quiet'})
        
        # Very short note (staccato)
        gt = [GroundTruthNote(60, 0.1, 0.15, velocity=0.8)]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_staccato", audio, gt, {'type': 'edge_case', 'case': 'short'})
        
        # Long sustained note
        gt = [GroundTruthNote(48, 0.1, 5.0, velocity=0.7)]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_sustained", audio, gt, {'type': 'edge_case', 'case': 'sustained'})
        
        # Two notes same pitch (re-attack)
        gt = [
            GroundTruthNote(52, 0.1, 0.5),
            GroundTruthNote(52, 0.6, 1.0),
        ]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_repeated_note", audio, gt, {'type': 'edge_case', 'case': 'repeat'})
        
        # Overlapping notes (legato)
        gt = [
            GroundTruthNote(48, 0.1, 0.8),
            GroundTruthNote(50, 0.5, 1.2),
            GroundTruthNote(52, 0.9, 1.6),
        ]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_overlapping", audio, gt, {'type': 'edge_case', 'case': 'overlap'})
        
        # With background noise
        gt = [GroundTruthNote(55, 0.1, 1.1, velocity=0.8)]
        audio, _ = self.synth.generate_sequence(gt, add_noise=0.02)
        self.save_test_case("edge_with_noise", audio, gt, {'type': 'edge_case', 'case': 'noisy'})
        
        # Interval test (perfect fifth)
        gt = [
            GroundTruthNote(48, 0.1, 1.0),  # C3
            GroundTruthNote(55, 0.1, 1.0),  # G3
        ]
        audio, _ = self.synth.generate_sequence(gt)
        self.save_test_case("edge_interval_fifth", audio, gt, {'type': 'edge_case', 'case': 'interval'})
    
    def generate_all(self):
        """Generate all test cases."""
        print(f"🎸 Generating test audio in {self.output_dir}")
        
        self.generate_single_notes()
        self.generate_sequences()
        self.generate_chords()
        self.generate_edge_cases()
        
        # Create index file
        index_path = self.output_dir / "index.json"
        test_files = sorted(self.output_dir.glob("*.json"))
        test_files = [f for f in test_files if f.name != "index.json"]
        
        index = {
            'generated_by': 'generate_test_audio.py',
            'sample_rate': self.sr,
            'test_cases': [f.stem for f in test_files],
            'count': len(test_files)
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n✅ Generated {len(test_files)} test cases")
        print(f"   Index saved to: {index_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic test audio for benchmarking")
    parser.add_argument('-o', '--output', default='test_audio', help='Output directory')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output
    generator = TestCaseGenerator(str(output_dir), sr=args.sr)
    generator.generate_all()


if __name__ == '__main__':
    main()
