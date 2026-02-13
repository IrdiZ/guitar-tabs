#!/usr/bin/env python3
"""
Neural Pitch Detection for Guitar Transcription

This module implements WaveNet-style neural pitch detection using:
1. torchcrepe - PyTorch implementation of CREPE (Convolutional REPresentations for Pitch Estimation)
2. WaveNet-inspired dilated convolutions for pitch refinement
3. SPICE (Google) via TensorFlow Hub
4. Ensemble methods combining neural approaches

Key advantages over DSP methods:
- Better handling of distortion (trained on diverse audio)
- More robust to harmonics and noise
- Sub-frame pitch resolution via probabilistic inference
- Handles pitch bends and vibrato naturally

Author: Claude (Subagent) for guitar-tabs project
Date: 2026-02-13
"""

import os
import sys
import json
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from abc import ABC, abstractmethod
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Audio processing
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# PyTorch
try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# torchcrepe - neural pitch detection
try:
    import torchcrepe
    HAS_TORCHCREPE = True
except ImportError:
    HAS_TORCHCREPE = False

# TensorFlow for SPICE
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GUITAR_MIN_HZ = 75  # Low E (82 Hz) with some slack for drop tunings
GUITAR_MAX_HZ = 1500  # High notes on high E string


@dataclass
class NeuralPitchNote:
    """A detected note from neural pitch detection."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    frequency: float = 0.0
    pitch_contour: List[Tuple[float, float]] = field(default_factory=list)  # [(time, freq), ...]
    source_model: str = ""
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    @property
    def name(self) -> str:
        octave = (self.midi // 12) - 1
        note = NOTE_NAMES[self.midi % 12]
        return f"{note}{octave}"
    
    def to_dict(self) -> Dict:
        return {
            'midi': self.midi,
            'name': self.name,
            'start_time': self.start_time,
            'duration': self.duration,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'source_model': self.source_model,
            'pitch_contour': self.pitch_contour[:10]  # Truncate for readability
        }


@dataclass
class NeuralPitchResult:
    """Complete result from neural pitch detection."""
    notes: List[NeuralPitchNote]
    pitch_contour: np.ndarray  # Frame-wise pitch in Hz
    confidence_contour: np.ndarray  # Frame-wise confidence
    times: np.ndarray  # Time axis
    model_name: str
    processing_time: float
    audio_duration: float
    hop_length: int
    sample_rate: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def note_rate(self) -> float:
        return len(self.notes) / self.audio_duration if self.audio_duration > 0 else 0
    
    @property
    def avg_confidence(self) -> float:
        if len(self.notes) == 0:
            return 0.0
        return sum(n.confidence for n in self.notes) / len(self.notes)


class NeuralPitchDetector(ABC):
    """Abstract base class for neural pitch detectors."""
    
    @abstractmethod
    def detect_pitch(self, audio: np.ndarray, sr: int) -> NeuralPitchResult:
        """Detect pitch from audio array."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this detector is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return detector name."""
        pass


class TorchCrepeDetector(NeuralPitchDetector):
    """
    CREPE-based pitch detection using torchcrepe.
    
    CREPE uses a deep CNN trained on millions of synthesized examples.
    It provides frame-level pitch estimates with confidence.
    """
    
    def __init__(
        self,
        model_capacity: str = 'small',  # 'tiny', 'small', 'medium', 'large', 'full' (small is faster on CPU)
        step_size: int = 10,  # ms between frames
        device: Optional[str] = None,
        min_confidence: float = 0.3,
        min_freq: float = GUITAR_MIN_HZ,
        max_freq: float = GUITAR_MAX_HZ,
        decoder: str = 'viterbi'  # 'argmax', 'weighted_argmax', 'viterbi'
    ):
        self.model_capacity = model_capacity
        self.step_size = step_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.decoder = decoder
        
    def is_available(self) -> bool:
        return HAS_TORCHCREPE
    
    @property
    def name(self) -> str:
        return f"torchcrepe-{self.model_capacity}"
    
    def detect_pitch(self, audio: np.ndarray, sr: int) -> NeuralPitchResult:
        """Run CREPE pitch detection."""
        if not self.is_available():
            raise RuntimeError("torchcrepe not available")
        
        start_time = time.time()
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # CREPE expects 16kHz audio
        if sr != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
            sr_crepe = 16000
        else:
            sr_crepe = sr
        
        # Run prediction
        hop_length = int(sr_crepe * self.step_size / 1000)
        
        # Get pitch and confidence
        pitch, confidence = torchcrepe.predict(
            audio_tensor,
            sr_crepe,
            hop_length=hop_length,
            model=self.model_capacity,
            decoder=torchcrepe.decode.viterbi if self.decoder == 'viterbi' else 
                    torchcrepe.decode.weighted_argmax if self.decoder == 'weighted_argmax' else
                    torchcrepe.decode.argmax,
            return_periodicity=True,
            device=self.device,
            batch_size=1024
        )
        
        # Convert to numpy
        pitch = pitch.squeeze().cpu().numpy()
        confidence = confidence.squeeze().cpu().numpy()
        
        # Create time axis
        n_frames = len(pitch)
        times = np.arange(n_frames) * self.step_size / 1000
        audio_duration = len(audio) / sr
        
        # Apply frequency filtering
        valid_mask = (pitch >= self.min_freq) & (pitch <= self.max_freq)
        pitch_filtered = np.where(valid_mask, pitch, 0)
        
        # Segment into notes
        notes = self._segment_to_notes(pitch_filtered, confidence, times, audio_duration)
        
        processing_time = time.time() - start_time
        
        return NeuralPitchResult(
            notes=notes,
            pitch_contour=pitch_filtered,
            confidence_contour=confidence,
            times=times,
            model_name=self.name,
            processing_time=processing_time,
            audio_duration=audio_duration,
            hop_length=hop_length,
            sample_rate=sr_crepe,
            metadata={
                'decoder': self.decoder,
                'model_capacity': self.model_capacity,
                'device': self.device,
                'n_frames': n_frames
            }
        )
    
    def _segment_to_notes(
        self,
        pitch: np.ndarray,
        confidence: np.ndarray,
        times: np.ndarray,
        audio_duration: float
    ) -> List[NeuralPitchNote]:
        """Segment continuous pitch contour into discrete notes."""
        notes = []
        
        # Find voiced regions
        voiced = (pitch > 0) & (confidence > self.min_confidence)
        
        if not np.any(voiced):
            return notes
        
        # Find contiguous voiced regions
        changes = np.diff(voiced.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if voiced[0]:
            starts = np.concatenate([[0], starts])
        if voiced[-1]:
            ends = np.concatenate([ends, [len(voiced)]])
        
        # Process each region
        for start_idx, end_idx in zip(starts, ends):
            if end_idx <= start_idx:
                continue
            
            region_pitch = pitch[start_idx:end_idx]
            region_conf = confidence[start_idx:end_idx]
            region_times = times[start_idx:end_idx]
            
            # Skip very short regions
            if len(region_pitch) < 2:
                continue
            
            # Find median pitch for note assignment
            median_freq = np.median(region_pitch[region_pitch > 0])
            midi = int(np.round(12 * np.log2(median_freq / 440) + 69))
            
            # Build pitch contour
            contour = list(zip(region_times.tolist(), region_pitch.tolist()))
            
            note = NeuralPitchNote(
                midi=midi,
                start_time=float(region_times[0]),
                duration=float(region_times[-1] - region_times[0] + self.step_size / 1000),
                confidence=float(np.mean(region_conf)),
                frequency=float(median_freq),
                pitch_contour=contour,
                source_model=self.name
            )
            notes.append(note)
        
        # Merge notes with same MIDI and close timing
        notes = self._merge_close_notes(notes)
        
        return notes
    
    def _merge_close_notes(self, notes: List[NeuralPitchNote], gap_threshold: float = 0.05) -> List[NeuralPitchNote]:
        """Merge notes with same pitch that are close together."""
        if len(notes) < 2:
            return notes
        
        merged = [notes[0]]
        for note in notes[1:]:
            prev = merged[-1]
            if (note.midi == prev.midi and 
                note.start_time - prev.end_time < gap_threshold):
                # Merge
                merged[-1] = NeuralPitchNote(
                    midi=prev.midi,
                    start_time=prev.start_time,
                    duration=note.end_time - prev.start_time,
                    confidence=(prev.confidence + note.confidence) / 2,
                    frequency=(prev.frequency + note.frequency) / 2,
                    pitch_contour=prev.pitch_contour + note.pitch_contour,
                    source_model=prev.source_model
                )
            else:
                merged.append(note)
        
        return merged


class SPICEDetector(NeuralPitchDetector):
    """
    SPICE (Self-supervised Pitch Estimation) from Google.
    
    Uses a self-supervised neural network trained on unlabeled audio.
    Particularly good at generalizing to unseen data.
    """
    
    SPICE_MODEL_URL = "https://tfhub.dev/google/spice/2"
    
    def __init__(
        self,
        model_url: Optional[str] = None,
        min_confidence: float = 0.8,  # SPICE uses different confidence scale
        min_freq: float = GUITAR_MIN_HZ,
        max_freq: float = GUITAR_MAX_HZ,
        hop_length_ms: float = 32.0
    ):
        self.model_url = model_url or self.SPICE_MODEL_URL
        self.min_confidence = min_confidence
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.hop_length_ms = hop_length_ms
        self._model = None
    
    def is_available(self) -> bool:
        return HAS_TF
    
    @property
    def name(self) -> str:
        return "SPICE"
    
    def _load_model(self):
        """Load SPICE model from TF Hub."""
        if self._model is None:
            print(f"Loading SPICE model from {self.model_url}...")
            self._model = hub.load(self.model_url)
        return self._model
    
    def detect_pitch(self, audio: np.ndarray, sr: int) -> NeuralPitchResult:
        """Run SPICE pitch detection."""
        if not self.is_available():
            raise RuntimeError("TensorFlow/TF Hub not available")
        
        start_time = time.time()
        
        # Ensure mono and normalize
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # SPICE expects 16kHz audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr_spice = 16000
        else:
            sr_spice = sr
        
        # Load model and run inference
        model = self._load_model()
        
        # Convert to float32 tensor
        audio_tensor = tf.constant(audio, dtype=tf.float32)
        
        # Run model
        output = model(audio_tensor)
        
        # Extract pitch and confidence
        pitch = output['pitch'].numpy()
        uncertainty = output['uncertainty'].numpy()
        
        # Convert uncertainty to confidence
        confidence = 1.0 - uncertainty
        
        # Convert SPICE pitch units to Hz
        # SPICE outputs pitch in a custom scale; convert to Hz
        # The model outputs values where pitch = log2(freq/10) roughly
        pitch_hz = np.power(2, pitch) * 10
        
        # Apply frequency filtering
        valid_mask = (pitch_hz >= self.min_freq) & (pitch_hz <= self.max_freq) & (confidence > self.min_confidence)
        pitch_hz = np.where(valid_mask, pitch_hz, 0)
        
        # Create time axis
        n_frames = len(pitch_hz)
        audio_duration = len(audio) / sr_spice
        times = np.linspace(0, audio_duration, n_frames)
        
        # Segment into notes
        notes = self._segment_to_notes(pitch_hz, confidence, times, audio_duration)
        
        processing_time = time.time() - start_time
        
        return NeuralPitchResult(
            notes=notes,
            pitch_contour=pitch_hz,
            confidence_contour=confidence,
            times=times,
            model_name=self.name,
            processing_time=processing_time,
            audio_duration=audio_duration,
            hop_length=int(sr_spice * self.hop_length_ms / 1000),
            sample_rate=sr_spice,
            metadata={'n_frames': n_frames}
        )
    
    def _segment_to_notes(
        self,
        pitch: np.ndarray,
        confidence: np.ndarray,
        times: np.ndarray,
        audio_duration: float
    ) -> List[NeuralPitchNote]:
        """Segment continuous pitch into notes (same as TorchCrepe)."""
        notes = []
        voiced = (pitch > 0)
        
        if not np.any(voiced):
            return notes
        
        changes = np.diff(voiced.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if voiced[0]:
            starts = np.concatenate([[0], starts])
        if voiced[-1]:
            ends = np.concatenate([ends, [len(voiced)]])
        
        hop_time = audio_duration / len(times) if len(times) > 0 else 0.01
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx <= start_idx:
                continue
            
            region_pitch = pitch[start_idx:end_idx]
            region_conf = confidence[start_idx:end_idx]
            region_times = times[start_idx:end_idx]
            
            if len(region_pitch) < 2:
                continue
            
            median_freq = np.median(region_pitch[region_pitch > 0])
            midi = int(np.round(12 * np.log2(median_freq / 440) + 69))
            
            contour = list(zip(region_times.tolist(), region_pitch.tolist()))
            
            note = NeuralPitchNote(
                midi=midi,
                start_time=float(region_times[0]),
                duration=float(region_times[-1] - region_times[0] + hop_time),
                confidence=float(np.mean(region_conf)),
                frequency=float(median_freq),
                pitch_contour=contour,
                source_model=self.name
            )
            notes.append(note)
        
        return notes


class WaveNetStyleRefiner(torch.nn.Module):
    """
    WaveNet-inspired dilated convolution network for pitch refinement.
    
    Uses causal dilated convolutions to model temporal dependencies
    in pitch trajectories, helping to correct octave errors and
    smooth predictions from primary pitch detectors.
    """
    
    def __init__(
        self,
        input_dim: int = 1,  # pitch values
        hidden_dim: int = 64,
        output_dim: int = 128,  # MIDI note bins
        n_layers: int = 6,
        kernel_size: int = 3,
        dilation_base: int = 2
    ):
        super().__init__()
        
        self.input_conv = torch.nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Dilated causal convolutions (WaveNet-style)
        self.dilated_convs = torch.nn.ModuleList()
        self.skip_convs = torch.nn.ModuleList()
        
        for i in range(n_layers):
            dilation = dilation_base ** i
            # Causal padding
            padding = (kernel_size - 1) * dilation
            
            self.dilated_convs.append(
                torch.nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size,
                              dilation=dilation, padding=padding)
            )
            self.skip_convs.append(
                torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
            )
        
        self.output_conv1 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.output_conv2 = torch.nn.Conv1d(hidden_dim, output_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, time) - normalized pitch values
            
        Returns:
            (batch, output_dim, time) - MIDI note logits
        """
        h = self.input_conv(x)
        skip_sum = 0
        
        for dilated, skip in zip(self.dilated_convs, self.skip_convs):
            h_conv = dilated(h)
            # Remove future samples (causal)
            h_conv = h_conv[:, :, :h.shape[2]]
            
            # Gated activation (WaveNet-style)
            filter_out = torch.tanh(h_conv[:, :h_conv.shape[1]//2, :])
            gate_out = torch.sigmoid(h_conv[:, h_conv.shape[1]//2:, :])
            h_gated = filter_out * gate_out
            
            skip_sum = skip_sum + skip(h_gated)
            h = h + h_gated
        
        out = torch.relu(self.output_conv1(skip_sum))
        out = self.output_conv2(out)
        
        return out


class NeuralEnsemblePitchDetector:
    """
    Ensemble neural pitch detection combining multiple models.
    
    Uses confidence-weighted voting to combine predictions from
    different neural pitch detectors.
    """
    
    def __init__(
        self,
        detectors: Optional[List[NeuralPitchDetector]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        if detectors is None:
            # Default: use available detectors
            self.detectors = []
            if HAS_TORCHCREPE:
                self.detectors.append(TorchCrepeDetector(model_capacity='full'))
            if HAS_TF:
                self.detectors.append(SPICEDetector())
        else:
            self.detectors = detectors
        
        self.weights = weights or {d.name: 1.0 for d in self.detectors}
    
    def detect_pitch(self, audio: np.ndarray, sr: int) -> NeuralPitchResult:
        """Run all detectors and combine results."""
        results = []
        
        for detector in self.detectors:
            if detector.is_available():
                try:
                    result = detector.detect_pitch(audio, sr)
                    results.append(result)
                    print(f"  {detector.name}: {len(result.notes)} notes, "
                          f"avg conf={result.avg_confidence:.2f}, "
                          f"time={result.processing_time:.2f}s")
                except Exception as e:
                    print(f"  {detector.name}: FAILED - {e}")
        
        if not results:
            raise RuntimeError("No detectors succeeded")
        
        # Combine results using confidence-weighted voting
        return self._combine_results(results, audio, sr)
    
    def _combine_results(
        self,
        results: List[NeuralPitchResult],
        audio: np.ndarray,
        sr: int
    ) -> NeuralPitchResult:
        """Combine multiple detector results."""
        # Use highest-resolution time grid
        best_result = max(results, key=lambda r: len(r.times))
        
        # Interpolate all pitch contours to same grid
        combined_pitch = np.zeros_like(best_result.pitch_contour)
        combined_conf = np.zeros_like(best_result.confidence_contour)
        total_weight = 0
        
        for result in results:
            weight = self.weights.get(result.model_name, 1.0) * result.avg_confidence
            
            # Interpolate to common grid
            pitch_interp = np.interp(best_result.times, result.times, result.pitch_contour)
            conf_interp = np.interp(best_result.times, result.times, result.confidence_contour)
            
            combined_pitch += pitch_interp * weight * conf_interp
            combined_conf += conf_interp * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_pitch /= (combined_conf + 1e-8)
            combined_conf /= total_weight
        
        # Re-segment combined pitch
        notes = self._segment_combined(combined_pitch, combined_conf, best_result.times)
        
        return NeuralPitchResult(
            notes=notes,
            pitch_contour=combined_pitch,
            confidence_contour=combined_conf,
            times=best_result.times,
            model_name="NeuralEnsemble",
            processing_time=sum(r.processing_time for r in results),
            audio_duration=best_result.audio_duration,
            hop_length=best_result.hop_length,
            sample_rate=best_result.sample_rate,
            metadata={
                'models_used': [r.model_name for r in results],
                'individual_notes': [len(r.notes) for r in results]
            }
        )
    
    def _segment_combined(
        self,
        pitch: np.ndarray,
        confidence: np.ndarray,
        times: np.ndarray
    ) -> List[NeuralPitchNote]:
        """Segment combined pitch into notes."""
        notes = []
        voiced = (pitch > 0) & (confidence > 0.3)
        
        if not np.any(voiced):
            return notes
        
        changes = np.diff(voiced.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if voiced[0]:
            starts = np.concatenate([[0], starts])
        if voiced[-1]:
            ends = np.concatenate([ends, [len(voiced)]])
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx <= start_idx:
                continue
            
            region_pitch = pitch[start_idx:end_idx]
            region_conf = confidence[start_idx:end_idx]
            region_times = times[start_idx:end_idx]
            
            if len(region_pitch) < 2:
                continue
            
            median_freq = np.median(region_pitch[region_pitch > 0])
            if median_freq <= 0 or np.isnan(median_freq):
                continue
                
            midi = int(np.round(12 * np.log2(median_freq / 440) + 69))
            
            note = NeuralPitchNote(
                midi=midi,
                start_time=float(region_times[0]),
                duration=float(region_times[-1] - region_times[0]),
                confidence=float(np.mean(region_conf)),
                frequency=float(median_freq),
                pitch_contour=list(zip(region_times.tolist(), region_pitch.tolist()))[:10],
                source_model="NeuralEnsemble"
            )
            notes.append(note)
        
        return notes


def compare_with_dsp_methods(audio_path: str) -> Dict[str, Any]:
    """
    Compare neural pitch detection with existing DSP methods.
    
    Returns comparison metrics and detected notes from each method.
    """
    print(f"\n{'='*60}")
    print(f"Pitch Detection Comparison: {Path(audio_path).name}")
    print(f"{'='*60}\n")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr
    print(f"Audio: {duration:.2f}s @ {sr}Hz\n")
    
    results = {}
    
    # 1. Neural: torchcrepe
    if HAS_TORCHCREPE:
        print("Running torchcrepe (CREPE neural network)...")
        detector = TorchCrepeDetector(model_capacity='full')
        try:
            result = detector.detect_pitch(audio, sr)
            results['torchcrepe'] = {
                'notes': [n.to_dict() for n in result.notes],
                'n_notes': len(result.notes),
                'avg_confidence': result.avg_confidence,
                'processing_time': result.processing_time,
                'note_rate': result.note_rate
            }
            print(f"  ✓ {len(result.notes)} notes detected ({result.processing_time:.2f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results['torchcrepe'] = {'error': str(e)}
    
    # 2. Neural: Ensemble
    print("\nRunning neural ensemble...")
    try:
        ensemble = NeuralEnsemblePitchDetector()
        result = ensemble.detect_pitch(audio, sr)
        results['neural_ensemble'] = {
            'notes': [n.to_dict() for n in result.notes],
            'n_notes': len(result.notes),
            'avg_confidence': result.avg_confidence,
            'processing_time': result.processing_time,
            'models_used': result.metadata.get('models_used', [])
        }
        print(f"  ✓ {len(result.notes)} notes (combined)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['neural_ensemble'] = {'error': str(e)}
    
    # 3. DSP: pyin (from librosa)
    print("\nRunning pyin (DSP method)...")
    try:
        start = time.time()
        f0, voiced, probs = librosa.pyin(
            audio, 
            fmin=GUITAR_MIN_HZ, 
            fmax=GUITAR_MAX_HZ,
            sr=sr
        )
        pyin_time = time.time() - start
        
        # Count notes (voiced segments)
        pyin_notes = np.sum(np.diff(voiced.astype(int)) == 1)
        avg_conf = np.mean(probs[voiced]) if np.any(voiced) else 0
        
        results['pyin'] = {
            'n_notes': int(pyin_notes),
            'avg_confidence': float(avg_conf),
            'processing_time': pyin_time,
            'voiced_ratio': float(np.mean(voiced))
        }
        print(f"  ✓ {pyin_notes} notes ({pyin_time:.2f}s)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['pyin'] = {'error': str(e)}
    
    # 4. DSP: HPS from distortion_pitch module
    print("\nRunning HPS (Harmonic Product Spectrum)...")
    try:
        sys.path.insert(0, str(Path(audio_path).parent))
        from distortion_pitch import DistortionPitchDetector, DistortionConfig
        
        start = time.time()
        config = DistortionConfig(verbose=False)
        dsp_detector = DistortionPitchDetector(config)
        dsp_results = dsp_detector.detect(audio, sr)
        hps_time = time.time() - start
        
        results['hps_distortion'] = {
            'n_notes': len(dsp_results),
            'processing_time': hps_time,
            'method': 'Harmonic Product Spectrum'
        }
        print(f"  ✓ {len(dsp_results)} notes ({hps_time:.2f}s)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['hps_distortion'] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method, data in results.items():
        if 'error' in data:
            print(f"  {method}: ERROR - {data['error']}")
        else:
            notes = data.get('n_notes', '?')
            conf = data.get('avg_confidence', 0)
            time_s = data.get('processing_time', 0)
            print(f"  {method}: {notes} notes, conf={conf:.2f}, time={time_s:.2f}s")
    
    return results


def process_audio_file(
    audio_path: str,
    output_json: Optional[str] = None,
    detector_type: str = 'torchcrepe',
    model_capacity: str = 'full'
) -> NeuralPitchResult:
    """
    Process an audio file with neural pitch detection.
    
    Args:
        audio_path: Path to audio file
        output_json: Optional path to save JSON results
        detector_type: 'torchcrepe', 'spice', or 'ensemble'
        model_capacity: For torchcrepe: 'tiny', 'small', 'medium', 'large', 'full'
    
    Returns:
        NeuralPitchResult with detected notes
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Create detector
    if detector_type == 'torchcrepe':
        detector = TorchCrepeDetector(model_capacity=model_capacity)
    elif detector_type == 'spice':
        detector = SPICEDetector()
    elif detector_type == 'ensemble':
        detector = NeuralEnsemblePitchDetector()
    else:
        raise ValueError(f"Unknown detector: {detector_type}")
    
    # Run detection
    result = detector.detect_pitch(audio, sr)
    
    # Save results
    if output_json:
        output_data = {
            'audio_file': audio_path,
            'model': result.model_name,
            'audio_duration': result.audio_duration,
            'processing_time': result.processing_time,
            'n_notes': len(result.notes),
            'avg_confidence': result.avg_confidence,
            'notes': [n.to_dict() for n in result.notes]
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_json}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Pitch Detection for Guitar")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-d", "--detector", default="torchcrepe",
                       choices=["torchcrepe", "spice", "ensemble"],
                       help="Detector type")
    parser.add_argument("-m", "--model", default="full",
                       choices=["tiny", "small", "medium", "large", "full"],
                       help="torchcrepe model capacity")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with DSP methods")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_dsp_methods(args.audio)
    else:
        result = process_audio_file(
            args.audio,
            output_json=args.output,
            detector_type=args.detector,
            model_capacity=args.model
        )
        
        print(f"\nDetected {len(result.notes)} notes:")
        for note in result.notes[:20]:  # First 20 notes
            print(f"  {note.name:4} @ {note.start_time:.3f}s "
                  f"(dur={note.duration:.3f}s, conf={note.confidence:.2f})")
        
        if len(result.notes) > 20:
            print(f"  ... and {len(result.notes) - 20} more")
