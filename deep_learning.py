#!/usr/bin/env python3
"""
Deep Learning Module for Guitar Tab Transcription

This module provides neural network-based approaches for automatic guitar transcription:
1. Basic Pitch (Spotify) - Pre-trained polyphonic AMT
2. CRNN Architecture - Guitar-specific tablature transcription
3. Spectrogram CNN - Lightweight pitch detection

Key Features:
- Pre-trained model support (no training required)
- GPU acceleration when available
- Multiple backend support (TensorFlow, PyTorch, ONNX)
- Guitar-specific post-processing

Author: Claude (Subagent) for guitar-tabs project
Date: 2026-02-13
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Audio processing
import librosa
import soundfile as sf

# Try to import deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available - some features disabled", file=sys.stderr)

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Try to import basic-pitch
try:
    from basic_pitch.inference import predict as bp_predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    HAS_BASIC_PITCH = True
except ImportError:
    HAS_BASIC_PITCH = False
    print("basic-pitch not available - install with: pip install basic-pitch", file=sys.stderr)

# Constants
GUITAR_TUNING_STANDARD = [40, 45, 50, 55, 59, 64]  # MIDI notes E2-E4
GUITAR_MIN_MIDI = 36  # C2 (for drop tunings)
GUITAR_MAX_MIDI = 88  # E6 (high frets on high E)
NUM_STRINGS = 6
NUM_FRETS = 22
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DetectedNote:
    """Represents a detected note from neural network inference."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    pitch_bends: List[float] = field(default_factory=list)
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    @property
    def name(self) -> str:
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.midi // 12) - 1
        note = notes[self.midi % 12]
        return f"{note}{octave}"
    
    def to_dict(self) -> Dict:
        return {
            'midi': self.midi,
            'name': self.name,
            'start_time': self.start_time,
            'duration': self.duration,
            'confidence': self.confidence,
            'pitch_bends': self.pitch_bends
        }


@dataclass
class TablaturePosition:
    """Represents a position on the guitar fretboard."""
    string: int  # 0-5 (low E to high E)
    fret: int    # 0-22
    midi: int
    start_time: float
    duration: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Complete transcription result from neural network."""
    notes: List[DetectedNote]
    tablature: List[TablaturePosition]
    model_used: str
    processing_time: float
    audio_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BASIC PITCH WRAPPER (PRE-TRAINED MODEL)
# ============================================================================

class BasicPitchTranscriber:
    """
    Wrapper for Spotify's Basic Pitch neural network.
    
    Basic Pitch is a lightweight CNN for polyphonic music transcription.
    - Architecture: CNN with ~17k parameters
    - Input: Audio (any sample rate, resampled internally)
    - Output: MIDI notes with pitch bends
    - Pre-trained on diverse instruments including guitar
    """
    
    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: int = 50,  # ms
        minimum_frequency: Optional[float] = None,
        maximum_frequency: Optional[float] = None
    ):
        if not HAS_BASIC_PITCH:
            raise ImportError("basic-pitch not installed. Install with: pip install basic-pitch")
        
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.minimum_frequency = minimum_frequency or 75.0   # E2 and below
        self.maximum_frequency = maximum_frequency or 1400.0 # High guitar range
    
    def transcribe(self, audio_path: str) -> List[DetectedNote]:
        """
        Transcribe audio file using Basic Pitch neural network.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
        
        Returns:
            List of DetectedNote objects
        """
        import time
        start_time = time.time()
        
        # Run Basic Pitch inference
        model_output, midi_data, note_events = bp_predict(
            audio_path,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        
        # Convert to our format
        notes = []
        for start, end, midi_note, amplitude, bends in note_events:
            # Filter to guitar range
            if GUITAR_MIN_MIDI <= midi_note <= GUITAR_MAX_MIDI:
                notes.append(DetectedNote(
                    midi=int(midi_note),
                    start_time=float(start),
                    duration=float(end - start),
                    confidence=float(amplitude),
                    pitch_bends=[float(b) for b in bends] if bends else []
                ))
        
        # Sort by time
        notes.sort(key=lambda n: (n.start_time, n.midi))
        
        elapsed = time.time() - start_time
        print(f"Basic Pitch: Detected {len(notes)} notes in {elapsed:.2f}s", file=sys.stderr)
        
        return notes
    
    def transcribe_to_midi(self, audio_path: str, output_path: str) -> str:
        """Transcribe and save as MIDI file."""
        _, midi_data, _ = bp_predict(
            audio_path,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        midi_data.write(output_path)
        return output_path


# ============================================================================
# CRNN ARCHITECTURE (GUITAR-SPECIFIC)
# ============================================================================

if HAS_TORCH:
    class TabCNN(nn.Module):
        """
        Convolutional encoder for spectrogram features.
        Based on trimplexx/music-transcription architecture.
        """
        
        def __init__(
            self,
            input_channels: int = 1,
            output_channels: List[int] = None,
            dropout: float = 0.3
        ):
            super().__init__()
            
            if output_channels is None:
                output_channels = [32, 64, 128, 128, 128]
            
            layers = []
            in_ch = input_channels
            
            for out_ch in output_channels:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    nn.Dropout2d(dropout)
                ])
                in_ch = out_ch
            
            self.conv = nn.Sequential(*layers)
            self.output_channels = output_channels[-1]
        
        def forward(self, x):
            return self.conv(x)
    
    
    class GuitarTabCRNN(nn.Module):
        """
        CRNN for guitar tablature transcription.
        
        Architecture:
        - 5-layer CNN encoder on CQT spectrogram
        - 2-layer bidirectional GRU
        - Dual heads: onset detection + fret classification
        
        Based on trimplexx/music-transcription which achieves 0.87 F1 on GuitarSet.
        """
        
        def __init__(
            self,
            n_cqt_bins: int = 168,
            rnn_hidden_size: int = 256,
            rnn_layers: int = 2,
            rnn_dropout: float = 0.3,
            rnn_bidirectional: bool = True,
            num_strings: int = NUM_STRINGS,
            num_frets: int = NUM_FRETS,
            cnn_channels: List[int] = None
        ):
            super().__init__()
            
            self.num_strings = num_strings
            self.num_fret_classes = num_frets + 2  # +1 for silence, +1 for open string
            
            # CNN encoder
            self.cnn = TabCNN(
                input_channels=1,
                output_channels=cnn_channels or [32, 64, 128, 128, 128]
            )
            
            # Calculate CNN output size
            # After 5 pooling layers with (2,1): n_cqt_bins / 32
            cnn_freq_out = n_cqt_bins // 32
            self.rnn_input_dim = self.cnn.output_channels * cnn_freq_out
            
            # Bidirectional GRU
            self.rnn = nn.GRU(
                input_size=self.rnn_input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout if rnn_layers > 1 else 0
            )
            
            rnn_out_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
            
            # Dual output heads
            self.onset_head = nn.Linear(rnn_out_size, num_strings)
            self.fret_head = nn.Linear(rnn_out_size, num_strings * self.num_fret_classes)
        
        def forward(self, x):
            """
            Forward pass.
            
            Args:
                x: CQT spectrogram [batch, freq_bins, time_frames]
            
            Returns:
                onset_logits: [batch, time_frames, num_strings]
                fret_logits: [batch, time_frames, num_strings, num_fret_classes]
            """
            # Add channel dimension
            if x.dim() == 3:
                x = x.unsqueeze(1)  # [batch, 1, freq, time]
            
            # CNN encoding
            x = self.cnn(x)  # [batch, channels, reduced_freq, time]
            
            batch_size, channels, freq, time_frames = x.shape
            
            # Reshape for RNN: [batch, time, features]
            x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
            x = x.reshape(batch_size, time_frames, -1)
            
            # RNN
            x, _ = self.rnn(x)  # [batch, time, rnn_out]
            
            # Output heads
            onset_logits = self.onset_head(x)  # [batch, time, strings]
            fret_logits = self.fret_head(x)    # [batch, time, strings * frets]
            
            # Reshape fret logits
            fret_logits = fret_logits.reshape(
                batch_size, time_frames, self.num_strings, self.num_fret_classes
            )
            
            return onset_logits, fret_logits
        
        def predict(self, cqt: np.ndarray, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
            """
            Run inference on CQT spectrogram.
            
            Args:
                cqt: CQT spectrogram [freq_bins, time_frames]
                device: 'cpu' or 'cuda'
            
            Returns:
                onset_probs: [time_frames, num_strings]
                fret_probs: [time_frames, num_strings, num_frets]
            """
            self.eval()
            with torch.no_grad():
                x = torch.from_numpy(cqt).float().unsqueeze(0).to(device)
                onset_logits, fret_logits = self(x)
                
                onset_probs = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
                fret_probs = F.softmax(fret_logits, dim=-1).squeeze(0).cpu().numpy()
            
            return onset_probs, fret_probs


# ============================================================================
# MIDI TO TABLATURE CONVERTER
# ============================================================================

class MIDIToTabConverter:
    """
    Convert MIDI notes to guitar tablature positions.
    
    Uses optimization to find playable fingerings considering:
    - Physical constraints (reach, string assignments)
    - Preference for lower frets
    - Hand position continuity
    """
    
    def __init__(
        self,
        tuning: List[int] = None,
        num_frets: int = NUM_FRETS,
        max_fret_span: int = 5,
        prefer_lower: bool = True
    ):
        self.tuning = tuning or GUITAR_TUNING_STANDARD
        self.num_frets = num_frets
        self.max_fret_span = max_fret_span
        self.prefer_lower = prefer_lower
        
        # Build lookup table: midi -> [(string, fret), ...]
        self._build_lookup()
    
    def _build_lookup(self):
        """Build MIDI to string/fret lookup table."""
        self.midi_to_positions = {}
        
        for string_idx, open_midi in enumerate(self.tuning):
            for fret in range(self.num_frets + 1):
                midi = open_midi + fret
                if midi not in self.midi_to_positions:
                    self.midi_to_positions[midi] = []
                self.midi_to_positions[midi].append((string_idx, fret))
    
    def get_possible_positions(self, midi: int) -> List[Tuple[int, int]]:
        """Get all possible string/fret positions for a MIDI note."""
        return self.midi_to_positions.get(midi, [])
    
    def convert_note(
        self, 
        note: DetectedNote,
        prev_position: Optional[TablaturePosition] = None
    ) -> Optional[TablaturePosition]:
        """
        Convert a single note to tablature position.
        
        Args:
            note: Detected note
            prev_position: Previous tab position for continuity
        
        Returns:
            Best tablature position or None if unplayable
        """
        positions = self.get_possible_positions(note.midi)
        
        if not positions:
            return None
        
        if len(positions) == 1:
            string, fret = positions[0]
            return TablaturePosition(
                string=string,
                fret=fret,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.confidence
            )
        
        # Score each position
        best_pos = None
        best_score = float('inf')
        
        for string, fret in positions:
            score = 0.0
            
            # Prefer lower frets
            if self.prefer_lower:
                score += fret * 0.1
            
            # Prefer continuity with previous position
            if prev_position is not None:
                fret_distance = abs(fret - prev_position.fret)
                string_distance = abs(string - prev_position.string)
                score += fret_distance * 0.5 + string_distance * 0.2
            
            # Penalize extreme positions
            if fret > 12:
                score += (fret - 12) * 0.3
            
            if score < best_score:
                best_score = score
                best_pos = (string, fret)
        
        if best_pos:
            string, fret = best_pos
            return TablaturePosition(
                string=string,
                fret=fret,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                confidence=note.confidence
            )
        
        return None
    
    def convert_notes(self, notes: List[DetectedNote]) -> List[TablaturePosition]:
        """Convert list of notes to tablature positions."""
        tablature = []
        prev_pos = None
        
        for note in sorted(notes, key=lambda n: n.start_time):
            pos = self.convert_note(note, prev_pos)
            if pos:
                tablature.append(pos)
                prev_pos = pos
        
        return tablature
    
    def resolve_chord_positions(
        self,
        notes: List[DetectedNote],
        time_threshold: float = 0.05
    ) -> List[TablaturePosition]:
        """
        Convert notes to tablature with proper chord handling.
        
        Groups simultaneous notes and optimizes string assignments together.
        """
        if not notes:
            return []
        
        # Sort by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Group into chords
        chords = []
        current_chord = [sorted_notes[0]]
        
        for note in sorted_notes[1:]:
            if note.start_time - current_chord[0].start_time <= time_threshold:
                current_chord.append(note)
            else:
                chords.append(current_chord)
                current_chord = [note]
        chords.append(current_chord)
        
        # Process each chord
        tablature = []
        prev_positions = []
        
        for chord in chords:
            if len(chord) == 1:
                # Single note
                prev_pos = prev_positions[-1] if prev_positions else None
                pos = self.convert_note(chord[0], prev_pos)
                if pos:
                    tablature.append(pos)
                    prev_positions = [pos]
            else:
                # Chord - find optimal string assignment
                chord_positions = self._optimize_chord(chord, prev_positions)
                tablature.extend(chord_positions)
                prev_positions = chord_positions
        
        return tablature
    
    def _optimize_chord(
        self,
        notes: List[DetectedNote],
        prev_positions: List[TablaturePosition]
    ) -> List[TablaturePosition]:
        """Optimize string assignment for a chord."""
        from itertools import product
        
        # Get all possible positions for each note
        all_positions = []
        for note in notes:
            positions = self.get_possible_positions(note.midi)
            if positions:
                all_positions.append([(note, s, f) for s, f in positions])
            else:
                # Note not playable on guitar
                pass
        
        if not all_positions:
            return []
        
        # Find combination with no string conflicts and best score
        best_combo = None
        best_score = float('inf')
        
        for combo in product(*all_positions):
            # Check for string conflicts
            strings_used = [pos[1] for pos in combo]
            if len(strings_used) != len(set(strings_used)):
                continue  # Conflict - skip
            
            # Calculate score
            frets = [pos[2] for pos in combo]
            
            # Fret span penalty
            fret_span = max(frets) - min(frets) if frets else 0
            if fret_span > self.max_fret_span:
                continue  # Unplayable span
            
            score = 0.0
            
            # Prefer lower average fret
            avg_fret = sum(frets) / len(frets)
            score += avg_fret * 0.2
            
            # Prefer smaller span
            score += fret_span * 0.5
            
            # Prefer continuity
            if prev_positions:
                prev_avg_fret = sum(p.fret for p in prev_positions) / len(prev_positions)
                score += abs(avg_fret - prev_avg_fret) * 0.3
            
            if score < best_score:
                best_score = score
                best_combo = combo
        
        if best_combo:
            return [
                TablaturePosition(
                    string=string,
                    fret=fret,
                    midi=note.midi,
                    start_time=note.start_time,
                    duration=note.duration,
                    confidence=note.confidence
                )
                for note, string, fret in best_combo
            ]
        
        # Fallback: assign what we can
        result = []
        used_strings = set()
        for note in sorted(notes, key=lambda n: n.midi):  # Low to high
            positions = self.get_possible_positions(note.midi)
            for string, fret in positions:
                if string not in used_strings:
                    result.append(TablaturePosition(
                        string=string,
                        fret=fret,
                        midi=note.midi,
                        start_time=note.start_time,
                        duration=note.duration,
                        confidence=note.confidence
                    ))
                    used_strings.add(string)
                    break
        
        return result


# ============================================================================
# SPECTROGRAM CNN (LIGHTWEIGHT)
# ============================================================================

if HAS_TORCH:
    class SpectrogramCNN(nn.Module):
        """
        Lightweight CNN for pitch detection on mel spectrograms.
        Can be quickly trained on custom data or used for feature extraction.
        """
        
        def __init__(
            self,
            n_mels: int = 128,
            n_classes: int = 88,  # MIDI 21-108 (piano range covers guitar)
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 1)),
            )
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes)
            )
        
        def forward(self, x):
            """
            Args:
                x: Mel spectrogram [batch, n_mels, time]
            Returns:
                logits: [batch, time, n_classes]
            """
            batch_size, n_mels, time_frames = x.shape
            
            # Process each frame independently
            x = x.unsqueeze(1)  # [batch, 1, mels, time]
            x = self.conv(x)   # [batch, 128, 4, time']
            
            # Per-frame classification
            x = x.permute(0, 3, 1, 2)  # [batch, time', 128, 4]
            x = x.reshape(batch_size, -1, 128 * 4)
            
            outputs = []
            for t in range(x.shape[1]):
                out = self.fc[1:](x[:, t])  # Skip flatten for pre-flattened input
                outputs.append(out)
            
            return torch.stack(outputs, dim=1)


# ============================================================================
# MAIN TRANSCRIPTION ENGINE
# ============================================================================

class DeepLearningTranscriber:
    """
    Main class for deep learning-based guitar transcription.
    
    Supports multiple backends:
    - Basic Pitch (Spotify) - pre-trained, no training needed
    - CRNN (trimplexx-style) - requires training or pre-trained weights
    - Spectrogram CNN - lightweight, can be trained quickly
    """
    
    def __init__(
        self,
        model_type: str = 'basic_pitch',
        device: str = 'auto',
        weights_path: Optional[str] = None
    ):
        """
        Initialize transcriber.
        
        Args:
            model_type: 'basic_pitch', 'crnn', or 'spectrogram_cnn'
            device: 'cpu', 'cuda', or 'auto'
            weights_path: Path to pre-trained weights (for crnn/cnn)
        """
        self.model_type = model_type
        
        # Determine device
        if device == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.weights_path = weights_path
        self.model = None
        self.tab_converter = MIDIToTabConverter()
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the selected model."""
        if self.model_type == 'basic_pitch':
            if not HAS_BASIC_PITCH:
                raise ImportError("basic-pitch not available")
            self.model = BasicPitchTranscriber()
        
        elif self.model_type == 'crnn':
            if not HAS_TORCH:
                raise ImportError("PyTorch required for CRNN model")
            self.model = GuitarTabCRNN()
            if self.weights_path and os.path.exists(self.weights_path):
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.to(self.device)
        
        elif self.model_type == 'spectrogram_cnn':
            if not HAS_TORCH:
                raise ImportError("PyTorch required for Spectrogram CNN")
            self.model = SpectrogramCNN()
            if self.weights_path and os.path.exists(self.weights_path):
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def transcribe(
        self,
        audio_path: str,
        return_tablature: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio file to notes and tablature.
        
        Args:
            audio_path: Path to audio file
            return_tablature: Whether to convert to guitar tablature
        
        Returns:
            TranscriptionResult with notes and tablature
        """
        import time
        start_time = time.time()
        
        # Get audio duration
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_duration = len(y) / sr
        
        # Run transcription based on model type
        if self.model_type == 'basic_pitch':
            notes = self.model.transcribe(audio_path)
        
        elif self.model_type == 'crnn':
            notes = self._transcribe_crnn(y)
        
        elif self.model_type == 'spectrogram_cnn':
            notes = self._transcribe_cnn(y)
        
        else:
            notes = []
        
        # Convert to tablature
        tablature = []
        if return_tablature and notes:
            tablature = self.tab_converter.resolve_chord_positions(notes)
        
        elapsed = time.time() - start_time
        
        return TranscriptionResult(
            notes=notes,
            tablature=tablature,
            model_used=self.model_type,
            processing_time=elapsed,
            audio_duration=audio_duration,
            metadata={
                'device': self.device,
                'sample_rate': SAMPLE_RATE,
                'total_notes': len(notes),
                'total_tab_positions': len(tablature)
            }
        )
    
    def _transcribe_crnn(self, y: np.ndarray) -> List[DetectedNote]:
        """Transcribe using CRNN model."""
        # Compute CQT
        cqt = np.abs(librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=168,
            bins_per_octave=24,
            fmin=librosa.note_to_hz('E2')
        ))
        
        # Convert to dB and normalize
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_norm = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min() + 1e-8)
        
        # Run model
        onset_probs, fret_probs = self.model.predict(cqt_norm, self.device)
        
        # Decode predictions to notes
        notes = self._decode_crnn_output(onset_probs, fret_probs)
        
        return notes
    
    def _decode_crnn_output(
        self,
        onset_probs: np.ndarray,
        fret_probs: np.ndarray,
        onset_threshold: float = 0.5
    ) -> List[DetectedNote]:
        """Decode CRNN output to notes."""
        notes = []
        time_per_frame = HOP_LENGTH / SAMPLE_RATE
        
        for t in range(onset_probs.shape[0]):
            for string in range(NUM_STRINGS):
                if onset_probs[t, string] > onset_threshold:
                    # Get predicted fret
                    fret = np.argmax(fret_probs[t, string])
                    if fret == 0:
                        continue  # Silence class
                    
                    # Convert to MIDI
                    midi = GUITAR_TUNING_STANDARD[string] + (fret - 1)
                    
                    # Find note duration
                    duration = time_per_frame
                    for t2 in range(t + 1, min(t + 50, onset_probs.shape[0])):
                        if onset_probs[t2, string] > onset_threshold:
                            break
                        duration += time_per_frame
                    
                    notes.append(DetectedNote(
                        midi=midi,
                        start_time=t * time_per_frame,
                        duration=duration,
                        confidence=float(onset_probs[t, string])
                    ))
        
        return notes
    
    def _transcribe_cnn(self, y: np.ndarray) -> List[DetectedNote]:
        """Transcribe using Spectrogram CNN."""
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_mels=128,
            hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        # Run model
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(mel_norm).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        # Decode
        notes = self._decode_cnn_output(probs)
        
        return notes
    
    def _decode_cnn_output(
        self,
        probs: np.ndarray,
        threshold: float = 0.5
    ) -> List[DetectedNote]:
        """Decode CNN output to notes."""
        notes = []
        time_per_frame = HOP_LENGTH / SAMPLE_RATE
        
        # Find active pitches per frame
        active = probs > threshold
        
        for midi_offset in range(active.shape[1]):
            midi = 21 + midi_offset  # Piano MIDI range
            
            # Skip non-guitar range
            if midi < GUITAR_MIN_MIDI or midi > GUITAR_MAX_MIDI:
                continue
            
            # Find onset positions
            note_active = active[:, midi_offset]
            diff = np.diff(note_active.astype(int))
            onsets = np.where(diff == 1)[0] + 1
            offsets = np.where(diff == -1)[0] + 1
            
            if note_active[0]:
                onsets = np.concatenate([[0], onsets])
            if note_active[-1]:
                offsets = np.concatenate([offsets, [len(note_active)]])
            
            for on, off in zip(onsets, offsets):
                confidence = float(probs[on:off, midi_offset].max())
                notes.append(DetectedNote(
                    midi=midi,
                    start_time=on * time_per_frame,
                    duration=(off - on) * time_per_frame,
                    confidence=confidence
                ))
        
        return notes


# ============================================================================
# CLI INTERFACE
# ============================================================================

def format_tablature(
    positions: List[TablaturePosition],
    width: int = 80
) -> str:
    """Format tablature positions as ASCII tab."""
    if not positions:
        return "No tablature generated"
    
    # Group by time
    time_step = 0.1  # 100ms per column
    max_time = max(p.start_time for p in positions) + 1.0
    n_cols = int(max_time / time_step) + 1
    
    # Initialize strings
    string_names = ['e', 'B', 'G', 'D', 'A', 'E']
    lines = {i: ['-'] * n_cols for i in range(6)}
    
    for pos in positions:
        col = int(pos.start_time / time_step)
        if col < n_cols:
            fret_str = str(pos.fret) if pos.fret < 10 else f"({pos.fret})"
            lines[5 - pos.string][col] = fret_str
    
    # Format output
    result = []
    for i in range(6):
        string_line = string_names[i] + '|' + ''.join(lines[i])
        result.append(string_line[:width])
    
    return '\n'.join(result)


def main():
    """CLI interface for deep learning transcription."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Learning Guitar Tab Transcription'
    )
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument(
        '-m', '--model',
        choices=['basic_pitch', 'crnn', 'spectrogram_cnn'],
        default='basic_pitch',
        help='Model to use (default: basic_pitch)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (JSON)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use'
    )
    parser.add_argument(
        '--weights',
        help='Path to model weights'
    )
    parser.add_argument(
        '--tab-only',
        action='store_true',
        help='Print only ASCII tablature'
    )
    
    args = parser.parse_args()
    
    # Run transcription
    transcriber = DeepLearningTranscriber(
        model_type=args.model,
        device=args.device,
        weights_path=args.weights
    )
    
    result = transcriber.transcribe(args.audio_path)
    
    # Output
    if args.tab_only:
        print(format_tablature(result.tablature))
    else:
        output = {
            'model': result.model_used,
            'processing_time': result.processing_time,
            'audio_duration': result.audio_duration,
            'notes': [n.to_dict() for n in result.notes],
            'tablature': [
                {
                    'string': p.string,
                    'fret': p.fret,
                    'midi': p.midi,
                    'start_time': p.start_time,
                    'duration': p.duration
                }
                for p in result.tablature
            ],
            'metadata': result.metadata
        }
        
        output_json = json.dumps(output, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output_json)
        
        # Also print ASCII tab
        print("\n" + "=" * 60, file=sys.stderr)
        print("ASCII Tablature:", file=sys.stderr)
        print(format_tablature(result.tablature), file=sys.stderr)


if __name__ == '__main__':
    main()
