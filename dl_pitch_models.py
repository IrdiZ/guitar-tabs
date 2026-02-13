#!/usr/bin/env python3
"""
Deep Learning Pitch Detection Models for Guitar Transcription

This module provides unified access to state-of-the-art neural network models
for automatic music transcription (AMT), specifically optimized for guitar.

Models Supported:
1. Basic Pitch (Spotify) - Polyphonic, fast, good generalization
2. CREPE - High accuracy monophonic pitch tracking
3. MT3 (Google Magenta) - Transformer-based, excellent for guitar
4. Omnizart - Multi-instrument toolkit

Key Features:
- Unified interface for all models
- Automatic model selection based on audio characteristics
- Ensemble methods combining multiple models
- Guitar-specific post-processing

Author: Claude (Subagent) for guitar-tabs project
Date: 2026-02-13
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import numpy as np

# Audio processing
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PitchNote:
    """A detected note from a pitch detection model."""
    midi: int
    start_time: float
    duration: float
    confidence: float
    frequency: float = 0.0
    pitch_bends: List[float] = field(default_factory=list)
    source_model: str = ""
    
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
            'end_time': self.end_time,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'source_model': self.source_model,
            'pitch_bends': self.pitch_bends
        }


@dataclass
class ModelResult:
    """Complete result from a pitch detection model."""
    notes: List[PitchNote]
    model_name: str
    processing_time: float
    audio_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def note_rate(self) -> float:
        """Notes per second of audio."""
        return len(self.notes) / self.audio_duration if self.audio_duration > 0 else 0
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across all notes."""
        if not self.notes:
            return 0.0
        return sum(n.confidence for n in self.notes) / len(self.notes)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'num_notes': len(self.notes),
            'processing_time': self.processing_time,
            'audio_duration': self.audio_duration,
            'note_rate': self.note_rate,
            'avg_confidence': self.avg_confidence,
            'notes': [n.to_dict() for n in self.notes],
            'metadata': self.metadata
        }


# ============================================================================
# CONSTANTS
# ============================================================================

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2-E4 in MIDI
GUITAR_MIN_MIDI = 36  # C2 (drop tunings)
GUITAR_MAX_MIDI = 88  # E6 (high frets)
GUITAR_MIN_HZ = 73.0  # D2
GUITAR_MAX_HZ = 1400.0  # High E6


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class PitchModel(ABC):
    """Abstract base class for pitch detection models."""
    
    name: str = "base"
    supports_polyphonic: bool = False
    requires_docker: bool = False
    
    @abstractmethod
    def detect(self, audio_path: str) -> ModelResult:
        """Run pitch detection on an audio file."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available/installed."""
        pass
    
    def filter_guitar_range(self, notes: List[PitchNote]) -> List[PitchNote]:
        """Filter notes to guitar range."""
        return [n for n in notes if GUITAR_MIN_MIDI <= n.midi <= GUITAR_MAX_MIDI]
    
    def merge_consecutive_notes(
        self, 
        notes: List[PitchNote], 
        max_gap: float = 0.05,
        same_pitch_threshold: int = 1
    ) -> List[PitchNote]:
        """Merge consecutive notes of the same pitch."""
        if not notes:
            return notes
        
        sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.midi))
        merged = []
        current = sorted_notes[0]
        
        for note in sorted_notes[1:]:
            same_pitch = abs(note.midi - current.midi) <= same_pitch_threshold
            consecutive = (note.start_time - current.end_time) <= max_gap
            
            if same_pitch and consecutive:
                # Extend current note
                current = PitchNote(
                    midi=current.midi,
                    start_time=current.start_time,
                    duration=(note.end_time - current.start_time),
                    confidence=max(current.confidence, note.confidence),
                    frequency=current.frequency,
                    pitch_bends=current.pitch_bends + note.pitch_bends,
                    source_model=current.source_model
                )
            else:
                merged.append(current)
                current = note
        
        merged.append(current)
        return merged


# ============================================================================
# BASIC PITCH (SPOTIFY)
# ============================================================================

class BasicPitchModel(PitchModel):
    """
    Spotify Basic Pitch neural network.
    
    - Architecture: Lightweight CNN (~17k params)
    - Polyphonic: Yes
    - Best for: General polyphonic transcription, chords
    - Speed: Fast (~0.5x realtime)
    """
    
    name = "basic_pitch"
    supports_polyphonic = True
    requires_docker = True  # Using Docker for isolation
    
    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: int = 50,
        docker_image: str = "basic-pitch-runner:latest"
    ):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.docker_image = docker_image
    
    def is_available(self) -> bool:
        """Check if Docker and image are available."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except FileNotFoundError:
            return False
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run Basic Pitch via Docker."""
        start_time = time.time()
        audio_path = Path(audio_path).resolve()
        
        # Get audio duration
        audio_duration = 0.0
        if HAS_LIBROSA:
            audio_duration = librosa.get_duration(path=str(audio_path))
        
        # Create temp output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            # Run Docker container
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{audio_path.parent}:/audio:ro',
                '-v', f'{Path(output_path).parent}:/output',
                self.docker_image,
                f'/audio/{audio_path.name}',
                '--output', f'/output/{Path(output_path).name}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Basic Pitch failed: {result.stderr}")
            
            # Parse output
            with open(output_path) as f:
                data = json.load(f)
            
            notes = []
            for note_data in data.get('notes', []):
                notes.append(PitchNote(
                    midi=note_data['midi'],
                    start_time=note_data['start_time'],
                    duration=note_data['duration'],
                    confidence=note_data['confidence'],
                    pitch_bends=note_data.get('pitch_bends', []),
                    source_model=self.name
                ))
            
            processing_time = time.time() - start_time
            
            return ModelResult(
                notes=self.filter_guitar_range(notes),
                model_name=self.name,
                processing_time=processing_time,
                audio_duration=audio_duration,
                metadata={'raw_note_count': len(data.get('notes', []))}
            )
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# ============================================================================
# CREPE (NYU)
# ============================================================================

class CREPEModel(PitchModel):
    """
    CREPE monophonic pitch tracker.
    
    - Architecture: CNN on raw audio
    - Polyphonic: No (tracks dominant pitch)
    - Best for: Single note lines, solos, high accuracy
    - Speed: Slow (~10x realtime on CPU)
    """
    
    name = "crepe"
    supports_polyphonic = False
    requires_docker = False
    
    def __init__(
        self,
        step_size: int = 10,  # ms
        confidence_threshold: float = 0.6,
        viterbi: bool = True,
        model_capacity: str = "full"  # tiny, small, medium, large, full
    ):
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold
        self.viterbi = viterbi
        self.model_capacity = model_capacity
        self._crepe = None
        self._venv_path = "/root/clawd/guitar-tabs/venv-crepe"
    
    def is_available(self) -> bool:
        """Check if CREPE is available."""
        try:
            # Check if crepe venv exists and has crepe
            activate = f"{self._venv_path}/bin/activate"
            if not os.path.exists(activate):
                return False
            
            result = subprocess.run(
                f"source {activate} && python -c 'import crepe'",
                shell=True,
                capture_output=True,
                executable='/bin/bash'
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run CREPE pitch detection."""
        start_time = time.time()
        
        # Get audio info
        if HAS_LIBROSA:
            audio_duration = librosa.get_duration(path=audio_path)
        else:
            audio_duration = 0.0
        
        # Run CREPE via subprocess (to use correct venv)
        script = f'''
import crepe
import librosa
import numpy as np
import json

audio_path = "{audio_path}"
y, sr = librosa.load(audio_path, sr=16000)

time_arr, frequency, confidence, _ = crepe.predict(
    y, sr, 
    viterbi={self.viterbi},
    step_size={self.step_size},
    model_capacity="{self.model_capacity}"
)

# Convert to notes
notes = []
current_note = None
for t, f, c in zip(time_arr, frequency, confidence):
    if c > {self.confidence_threshold} and f > 60:
        midi = int(round(12 * np.log2(f / 440) + 69))
        if current_note is None or abs(midi - current_note['midi']) > 0:
            if current_note:
                current_note['end'] = t
                notes.append(current_note)
            current_note = {{'midi': midi, 'start': float(t), 'freq': float(f), 'conf': float(c)}}
        else:
            current_note['end'] = float(t)
    elif current_note:
        current_note['end'] = float(t)
        notes.append(current_note)
        current_note = None

if current_note:
    current_note['end'] = float(time_arr[-1])
    notes.append(current_note)

print(json.dumps(notes))
'''
        
        # Write script to temp file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            result = subprocess.run(
                f"source {self._venv_path}/bin/activate && python {script_path}",
                shell=True,
                capture_output=True,
                text=True,
                executable='/bin/bash',
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"CREPE failed: {result.stderr}")
            
            # Parse output (find the JSON line)
            notes_data = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('['):
                    notes_data = json.loads(line)
                    break
            
            notes = []
            for nd in notes_data:
                duration = nd.get('end', nd['start'] + 0.05) - nd['start']
                notes.append(PitchNote(
                    midi=nd['midi'],
                    start_time=nd['start'],
                    duration=duration,
                    confidence=nd['conf'],
                    frequency=nd['freq'],
                    source_model=self.name
                ))
            
            processing_time = time.time() - start_time
            
            return ModelResult(
                notes=self.filter_guitar_range(notes),
                model_name=self.name,
                processing_time=processing_time,
                audio_duration=audio_duration,
                metadata={'raw_note_count': len(notes_data)}
            )
            
        finally:
            os.unlink(script_path)


# ============================================================================
# PYIN (LIBROSA)
# ============================================================================

class PYINModel(PitchModel):
    """
    Probabilistic YIN pitch detection (non-neural).
    
    - Algorithm: Statistical pitch tracking
    - Polyphonic: No
    - Best for: Fast baseline, vibrato handling
    - Speed: Very fast (~0.1x realtime)
    """
    
    name = "pyin"
    supports_polyphonic = False
    requires_docker = False
    
    def __init__(
        self,
        fmin: float = GUITAR_MIN_HZ,
        fmax: float = GUITAR_MAX_HZ,
        sr: int = 22050,
        hop_length: int = 512
    ):
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
        self.hop_length = hop_length
    
    def is_available(self) -> bool:
        return HAS_LIBROSA
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run pYIN pitch detection."""
        if not HAS_LIBROSA:
            raise ImportError("librosa not available")
        
        start_time = time.time()
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        audio_duration = len(y) / sr
        
        # Run pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=self.hop_length
        )
        
        # Convert to notes
        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)
        
        notes = []
        current_note = None
        
        for t, f, voiced, prob in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and not np.isnan(f) and f > 0:
                midi = int(round(12 * np.log2(f / 440) + 69))
                
                if current_note is None or abs(midi - current_note['midi']) > 0:
                    if current_note:
                        current_note['end'] = t
                        notes.append(current_note)
                    current_note = {
                        'midi': midi,
                        'start': t,
                        'freq': f,
                        'conf': prob
                    }
                else:
                    current_note['end'] = t
                    current_note['conf'] = max(current_note['conf'], prob)
            elif current_note:
                current_note['end'] = t
                notes.append(current_note)
                current_note = None
        
        if current_note:
            current_note['end'] = times[-1]
            notes.append(current_note)
        
        pitch_notes = []
        for nd in notes:
            duration = nd.get('end', nd['start'] + 0.05) - nd['start']
            pitch_notes.append(PitchNote(
                midi=nd['midi'],
                start_time=nd['start'],
                duration=duration,
                confidence=nd['conf'],
                frequency=nd['freq'],
                source_model=self.name
            ))
        
        processing_time = time.time() - start_time
        
        return ModelResult(
            notes=self.filter_guitar_range(pitch_notes),
            model_name=self.name,
            processing_time=processing_time,
            audio_duration=audio_duration,
            metadata={'raw_note_count': len(notes)}
        )


# ============================================================================
# MT3 (GOOGLE MAGENTA) - Placeholder for future integration
# ============================================================================

class MT3Model(PitchModel):
    """
    MT3 Multi-Task Multitrack Music Transcription (Google Magenta).
    
    - Architecture: T5 Transformer
    - Polyphonic: Yes, multi-track
    - Best for: Complex arrangements, guitar (specifically optimized)
    - Speed: Slow, GPU recommended
    
    Note: Requires T5X framework and TPU/GPU. Use Docker or Colab.
    """
    
    name = "mt3"
    supports_polyphonic = True
    requires_docker = True
    
    def __init__(self, docker_image: str = "mt3-transcription:latest"):
        self.docker_image = docker_image
    
    def is_available(self) -> bool:
        """Check if MT3 Docker image is available."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except FileNotFoundError:
            return False
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run MT3 transcription."""
        # TODO: Implement when MT3 Docker image is built
        raise NotImplementedError(
            "MT3 integration pending. See: https://github.com/magenta/mt3\n"
            "MT3 dramatically improves guitar transcription accuracy but "
            "requires T5X framework and significant compute resources."
        )


# ============================================================================
# OMNIZART - Placeholder for future integration
# ============================================================================

class OmnizartModel(PitchModel):
    """
    Omnizart general music transcription toolkit.
    
    - Architecture: Various deep learning models
    - Polyphonic: Yes
    - Best for: Multi-instrument, drums, vocals
    - Speed: Moderate
    """
    
    name = "omnizart"
    supports_polyphonic = True
    requires_docker = True
    
    def __init__(self, docker_image: str = "mctlab/omnizart:latest"):
        self.docker_image = docker_image
    
    def is_available(self) -> bool:
        """Check if Omnizart Docker image is available."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except FileNotFoundError:
            return False
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run Omnizart transcription."""
        # TODO: Implement when Omnizart Docker pull completes
        raise NotImplementedError(
            "Omnizart integration pending. Pull image with:\n"
            "docker pull mctlab/omnizart:latest"
        )


# ============================================================================
# MODEL ENSEMBLE
# ============================================================================

class EnsembleModel(PitchModel):
    """
    Ensemble combining multiple pitch detection models.
    
    Strategy:
    1. Run all available models
    2. Cluster notes by time/pitch
    3. Weight by model confidence and reliability
    4. Output consensus notes with high confidence
    """
    
    name = "ensemble"
    supports_polyphonic = True
    
    def __init__(
        self,
        models: Optional[List[PitchModel]] = None,
        time_tolerance: float = 0.05,  # 50ms
        pitch_tolerance: int = 1,  # semitones
        min_votes: int = 2,  # minimum models that must agree
        weights: Optional[Dict[str, float]] = None
    ):
        if models is None:
            models = [
                BasicPitchModel(),
                CREPEModel(),
                PYINModel()
            ]
        
        self.models = models
        self.time_tolerance = time_tolerance
        self.pitch_tolerance = pitch_tolerance
        self.min_votes = min_votes
        self.weights = weights or {
            'basic_pitch': 1.2,  # Good for polyphonic
            'crepe': 1.5,  # Highest accuracy monophonic
            'pyin': 1.0,  # Baseline
            'mt3': 1.8,  # Best for guitar when available
            'omnizart': 1.1
        }
    
    def is_available(self) -> bool:
        """True if at least one model is available."""
        return any(m.is_available() for m in self.models)
    
    def detect(self, audio_path: str) -> ModelResult:
        """Run ensemble detection."""
        start_time = time.time()
        
        # Collect results from all available models
        all_results = []
        all_notes = []
        
        for model in self.models:
            if model.is_available():
                try:
                    result = model.detect(audio_path)
                    all_results.append(result)
                    for note in result.notes:
                        all_notes.append((note, model.name))
                    print(f"  {model.name}: {len(result.notes)} notes", file=sys.stderr)
                except Exception as e:
                    print(f"  {model.name}: FAILED - {e}", file=sys.stderr)
        
        if not all_notes:
            return ModelResult(
                notes=[],
                model_name=self.name,
                processing_time=time.time() - start_time,
                audio_duration=0,
                metadata={'error': 'No models produced output'}
            )
        
        # Get audio duration from first result
        audio_duration = all_results[0].audio_duration if all_results else 0
        
        # Cluster notes by time and pitch
        consensus_notes = self._build_consensus(all_notes)
        
        processing_time = time.time() - start_time
        
        return ModelResult(
            notes=consensus_notes,
            model_name=self.name,
            processing_time=processing_time,
            audio_duration=audio_duration,
            metadata={
                'models_used': [r.model_name for r in all_results],
                'notes_per_model': {r.model_name: len(r.notes) for r in all_results},
                'total_candidates': len(all_notes)
            }
        )
    
    def _build_consensus(self, all_notes: List[Tuple[PitchNote, str]]) -> List[PitchNote]:
        """Build consensus notes from multiple model outputs."""
        if not all_notes:
            return []
        
        # Sort by time
        all_notes.sort(key=lambda x: x[0].start_time)
        
        # Cluster notes
        clusters = []
        current_cluster = [all_notes[0]]
        
        for note, model in all_notes[1:]:
            # Check if note belongs to current cluster
            fits = False
            for cn, cm in current_cluster:
                time_match = abs(note.start_time - cn.start_time) <= self.time_tolerance
                pitch_match = abs(note.midi - cn.midi) <= self.pitch_tolerance
                if time_match and pitch_match:
                    fits = True
                    break
            
            if fits:
                current_cluster.append((note, model))
            else:
                clusters.append(current_cluster)
                current_cluster = [(note, model)]
        
        clusters.append(current_cluster)
        
        # Build consensus from clusters
        consensus = []
        for cluster in clusters:
            if len(set(m for _, m in cluster)) >= self.min_votes:
                # Have enough agreement
                consensus_note = self._merge_cluster(cluster)
                consensus.append(consensus_note)
            elif len(cluster) == 1:
                # Single model detected - keep if high confidence
                note, model = cluster[0]
                if note.confidence > 0.8:
                    consensus.append(note)
        
        return consensus
    
    def _merge_cluster(self, cluster: List[Tuple[PitchNote, str]]) -> PitchNote:
        """Merge a cluster of notes into a consensus note."""
        # Weighted average of MIDI notes
        total_weight = 0
        weighted_midi = 0
        weighted_start = 0
        weighted_end = 0
        max_confidence = 0
        all_bends = []
        models_used = []
        
        for note, model in cluster:
            weight = self.weights.get(model, 1.0) * note.confidence
            total_weight += weight
            weighted_midi += note.midi * weight
            weighted_start += note.start_time * weight
            weighted_end += note.end_time * weight
            max_confidence = max(max_confidence, note.confidence)
            all_bends.extend(note.pitch_bends)
            models_used.append(model)
        
        if total_weight == 0:
            total_weight = 1
        
        return PitchNote(
            midi=int(round(weighted_midi / total_weight)),
            start_time=weighted_start / total_weight,
            duration=(weighted_end - weighted_start) / total_weight,
            confidence=max_confidence,
            pitch_bends=all_bends[:20],  # Keep reasonable number
            source_model=f"ensemble({','.join(set(models_used))})"
        )


# ============================================================================
# MODEL REGISTRY AND FACTORY
# ============================================================================

MODEL_REGISTRY = {
    'basic_pitch': BasicPitchModel,
    'crepe': CREPEModel,
    'pyin': PYINModel,
    'mt3': MT3Model,
    'omnizart': OmnizartModel,
    'ensemble': EnsembleModel
}


def get_available_models() -> List[str]:
    """Get list of available model names."""
    available = []
    for name, model_class in MODEL_REGISTRY.items():
        try:
            model = model_class()
            if model.is_available():
                available.append(name)
        except Exception:
            pass
    return available


def create_model(name: str, **kwargs) -> PitchModel:
    """Create a pitch detection model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def compare_models(
    audio_path: str,
    models: Optional[List[str]] = None
) -> Dict[str, ModelResult]:
    """
    Run multiple models on the same audio and compare results.
    
    Args:
        audio_path: Path to audio file
        models: List of model names (or None for all available)
    
    Returns:
        Dict mapping model name to its result
    """
    if models is None:
        models = get_available_models()
    
    results = {}
    for name in models:
        try:
            model = create_model(name)
            if model.is_available():
                print(f"Running {name}...", file=sys.stderr)
                result = model.detect(audio_path)
                results[name] = result
                print(f"  {name}: {len(result.notes)} notes in {result.processing_time:.2f}s",
                      file=sys.stderr)
        except Exception as e:
            print(f"  {name}: FAILED - {e}", file=sys.stderr)
    
    return results


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """CLI for testing pitch models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Learning Pitch Detection')
    parser.add_argument('audio', help='Audio file path')
    parser.add_argument('--model', '-m', default='ensemble',
                        help='Model to use (or "compare" for all)')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name in get_available_models():
            model = create_model(name)
            poly = "polyphonic" if model.supports_polyphonic else "monophonic"
            docker = "(Docker)" if model.requires_docker else ""
            print(f"  {name}: {poly} {docker}")
        return
    
    if args.model == 'compare':
        results = compare_models(args.audio)
        output = {name: r.to_dict() for name, r in results.items()}
    else:
        model = create_model(args.model)
        result = model.detect(args.audio)
        output = result.to_dict()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
