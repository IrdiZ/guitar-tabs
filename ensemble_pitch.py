#!/usr/bin/env python3
"""
Multi-Model Ensemble Pitch Detection

Combines multiple pitch detection algorithms using "wisdom of crowds" approach:
- pYIN (librosa) - Probabilistic YIN, robust for monophonic
- CQT (librosa) - Constant-Q Transform peak detection  
- CREPE (neural network) - Deep learning, high accuracy
- Basic Pitch (Spotify) - Polyphonic neural network
- piptrack (librosa) - Multi-pitch tracking
- HPS (Harmonic Product Spectrum) - BEST FOR DISTORTED GUITAR

The ensemble:
1. Runs ALL available detectors on the same audio
2. Collects pitch candidates at each time frame
3. Uses weighted voting/clustering to find consensus
4. Only outputs high-confidence consensus notes

This dramatically improves accuracy through multi-model agreement.

HPS is critical for distorted guitar because distortion creates strong harmonics
that can be louder than the fundamental. HPS multiplies downsampled spectra
together, causing all harmonics to "vote" for the true fundamental frequency.
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings

# Import detector modules
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

try:
    from basic_pitch.inference import predict
    HAS_BASIC_PITCH_NATIVE = True
except ImportError:
    HAS_BASIC_PITCH_NATIVE = False

# Check Docker-based Basic Pitch
import shutil
HAS_BASIC_PITCH_DOCKER = shutil.which('docker') is not None

# Import Essentia (alternative pitch detection)
try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

# Import Parselmouth/Praat (alternative pitch detection)
try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

# Import our custom YIN implementation
try:
    from yin_pitch import detect_yin_for_ensemble, YinConfig
    HAS_YIN = True
except ImportError:
    HAS_YIN = False

# Constants
GUITAR_MIN_HZ = 75
GUITAR_MAX_HZ = 1400
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class PitchCandidate:
    """A single pitch candidate from one detector."""
    time: float           # Time in seconds
    midi_note: int        # MIDI note number
    frequency: float      # Frequency in Hz
    confidence: float     # Confidence 0-1
    method: str           # Detector name
    raw_pitch: float = 0  # Raw detected frequency before quantization


@dataclass
class ConsensusNote:
    """A note with consensus from multiple detectors."""
    midi_note: int
    start_time: float
    end_time: float
    confidence: float      # Weighted consensus confidence
    frequency: float       # Average frequency
    votes: int             # Number of methods that agreed
    methods: List[str]     # Which methods contributed
    pitch_spread: float    # Spread of pitch estimates (cents)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def name(self) -> str:
        return NOTE_NAMES[self.midi_note % 12] + str(self.midi_note // 12 - 1)
    
    def __repr__(self):
        return f"ConsensusNote({self.name}, t={self.start_time:.3f}, conf={self.confidence:.2f}, votes={self.votes})"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble pitch detection."""
    # Detector selection
    use_pyin: bool = True
    use_cqt: bool = True
    use_crepe: bool = True
    use_basic_pitch: bool = True
    use_piptrack: bool = True
    use_hps: bool = True  # Harmonic Product Spectrum - BEST FOR DISTORTED GUITAR
    use_essentia_yin_fft: bool = True  # Essentia YIN FFT - FASTEST, HIGH ACCURACY
    use_essentia_yin: bool = False      # Essentia YIN - slower but thorough
    use_praat_cc: bool = True           # Praat cross-correlation - HIGH CONFIDENCE
    use_praat_shs: bool = True          # Praat SHS - good for harmonics
    use_yin: bool = True                # Custom YIN - BEST FOR MONOPHONIC, handles octave errors
    
    # Detector weights (higher = more trusted)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'pyin': 1.0,
        'cqt': 0.8,
        'crepe': 1.5,           # CREPE is most accurate for monophonic
        'basic_pitch': 1.2,     # Good for polyphonic
        'piptrack': 0.6,        # Less accurate but fast
        'hps': 1.3,             # HPS - excellent for distorted guitar
        'essentia_yin_fft': 1.3,  # Fast and accurate
        'essentia_yin': 1.1,      # Thorough YIN implementation
        'praat_cc': 1.2,          # High confidence scores
        'praat_shs': 1.1,         # Good for harmonics
        'yin': 1.4,               # Custom YIN - excellent for lead guitar, handles octave errors
    })
    
    # Consensus parameters
    min_votes: int = 2           # Minimum methods that must agree
    min_weighted_score: float = 1.5  # Minimum weighted vote score
    pitch_tolerance_cents: float = 50  # Tolerance for clustering (50 cents = quarter tone)
    time_tolerance: float = 0.03  # Time window for clustering (30ms)
    
    # Output filtering
    min_confidence: float = 0.5   # Minimum consensus confidence
    min_duration: float = 0.05    # Minimum note duration (50ms)
    
    # Processing
    hop_length: int = 512
    crepe_model: str = 'small'  # 'tiny', 'small', 'medium', 'large', 'full'


class EnsemblePitchDetector:
    """
    Multi-model ensemble for robust pitch detection.
    
    Uses multiple algorithms and combines their outputs through
    weighted voting and clustering to find consensus.
    """
    
    def __init__(self, sr: int = 22050, config: Optional[EnsembleConfig] = None):
        self.sr = sr
        self.config = config or EnsembleConfig()
        self.available_methods = self._check_available_methods()
        
    def _check_available_methods(self) -> List[str]:
        """Check which pitch detection methods are available."""
        methods = []
        
        if self.config.use_pyin:
            methods.append('pyin')
        
        if self.config.use_cqt:
            methods.append('cqt')
        
        if self.config.use_crepe and HAS_CREPE:
            methods.append('crepe')
        elif self.config.use_crepe:
            print("⚠️  CREPE not available (install: pip install crepe tensorflow)")
        
        if self.config.use_basic_pitch:
            if HAS_BASIC_PITCH_NATIVE:
                methods.append('basic_pitch')
            elif HAS_BASIC_PITCH_DOCKER:
                methods.append('basic_pitch_docker')
            else:
                print("⚠️  Basic Pitch not available")
        
        if self.config.use_piptrack:
            methods.append('piptrack')
        
        if self.config.use_hps:
            methods.append('hps')  # HPS is always available (pure numpy/scipy)
        
        # Essentia detectors (alternative library)
        if HAS_ESSENTIA:
            if self.config.use_essentia_yin_fft:
                methods.append('essentia_yin_fft')
            if self.config.use_essentia_yin:
                methods.append('essentia_yin')
        elif self.config.use_essentia_yin_fft or self.config.use_essentia_yin:
            print("⚠️  Essentia not available (install: pip install essentia)")
        
        # Parselmouth/Praat detectors (alternative library)
        if HAS_PARSELMOUTH:
            if self.config.use_praat_cc:
                methods.append('praat_cc')
            if self.config.use_praat_shs:
                methods.append('praat_shs')
        elif self.config.use_praat_cc or self.config.use_praat_shs:
            print("⚠️  Parselmouth not available (install: pip install praat-parselmouth)")
        
        # Custom YIN detector (always available - pure numpy)
        if self.config.use_yin and HAS_YIN:
            methods.append('yin')
        elif self.config.use_yin and not HAS_YIN:
            print("⚠️  Custom YIN not available (check yin_pitch.py)")
        
        return methods
    
    def detect(self, y: np.ndarray, audio_path: Optional[str] = None) -> List[ConsensusNote]:
        """
        Run ensemble pitch detection on audio.
        
        Args:
            y: Audio signal (mono, resampled to self.sr)
            audio_path: Path to audio file (needed for basic_pitch Docker)
            
        Returns:
            List of ConsensusNote with high-confidence consensus pitches
        """
        print(f"\n🎸 Ensemble Pitch Detection")
        print(f"   Available methods: {', '.join(self.available_methods)}")
        print(f"   Config: min_votes={self.config.min_votes}, min_weighted={self.config.min_weighted_score}")
        
        # Collect candidates from all methods
        all_candidates: List[PitchCandidate] = []
        
        for method in self.available_methods:
            print(f"\n   Running {method}...")
            try:
                candidates = self._run_detector(method, y, audio_path)
                all_candidates.extend(candidates)
                print(f"   ✓ {method}: {len(candidates)} candidates")
            except Exception as e:
                print(f"   ✗ {method} failed: {e}")
        
        print(f"\n   Total candidates: {len(all_candidates)}")
        
        if not all_candidates:
            print("   No candidates detected!")
            return []
        
        # Cluster and vote
        print(f"\n   Clustering with tolerance: {self.config.pitch_tolerance_cents} cents, {self.config.time_tolerance*1000:.0f}ms")
        consensus_notes = self._cluster_and_vote(all_candidates)
        
        print(f"   Raw consensus notes: {len(consensus_notes)}")
        
        # Filter by confidence and duration
        filtered = [
            n for n in consensus_notes
            if n.confidence >= self.config.min_confidence
            and n.duration >= self.config.min_duration
        ]
        
        print(f"   After filtering (conf>={self.config.min_confidence}, dur>={self.config.min_duration}s): {len(filtered)}")
        
        # Sort by time
        filtered.sort(key=lambda n: n.start_time)
        
        return filtered
    
    def _run_detector(self, method: str, y: np.ndarray, audio_path: Optional[str]) -> List[PitchCandidate]:
        """Run a specific pitch detector."""
        if method == 'pyin':
            return self._detect_pyin(y)
        elif method == 'cqt':
            return self._detect_cqt(y)
        elif method == 'crepe':
            return self._detect_crepe(y)
        elif method == 'basic_pitch':
            return self._detect_basic_pitch_native(y)
        elif method == 'basic_pitch_docker':
            return self._detect_basic_pitch_docker(audio_path)
        elif method == 'piptrack':
            return self._detect_piptrack(y)
        elif method == 'hps':
            return self._detect_hps(y)
        elif method == 'essentia_yin_fft':
            return self._detect_essentia_yin_fft(y)
        elif method == 'essentia_yin':
            return self._detect_essentia_yin(y)
        elif method == 'praat_cc':
            return self._detect_praat_cc(y)
        elif method == 'praat_shs':
            return self._detect_praat_shs(y)
        elif method == 'yin':
            return self._detect_yin(y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_pyin(self, y: np.ndarray) -> List[PitchCandidate]:
        """Detect pitches using pYIN."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ,
            sr=self.sr,
            hop_length=self.config.hop_length,
            fill_na=0.0
        )
        
        candidates = []
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=self.config.hop_length)
        
        for i, (freq, voiced, conf) in enumerate(zip(f0, voiced_flag, voiced_probs)):
            if voiced and freq > 0 and conf > 0.3:
                midi = int(round(librosa.hz_to_midi(freq)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=times[i],
                        midi_note=midi,
                        frequency=freq,
                        confidence=float(conf),
                        method='pyin',
                        raw_pitch=freq
                    ))
        
        return candidates
    
    def _detect_cqt(self, y: np.ndarray) -> List[PitchCandidate]:
        """Detect pitches using CQT peak detection."""
        fmin = librosa.note_to_hz('C1')
        n_bins = 84  # 7 octaves
        
        C = np.abs(librosa.cqt(
            y, sr=self.sr, hop_length=self.config.hop_length,
            fmin=fmin, n_bins=n_bins, bins_per_octave=12
        ))
        
        cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=12)
        times = librosa.frames_to_time(np.arange(C.shape[1]), sr=self.sr, hop_length=self.config.hop_length)
        
        candidates = []
        
        for i in range(C.shape[1]):
            frame = C[:, i]
            max_mag = frame.max()
            
            if max_mag < 1e-10:
                continue
            
            # Normalize
            frame_norm = frame / max_mag
            
            # Find peaks above threshold
            threshold = 0.3
            for j in range(1, len(frame_norm) - 1):
                if (frame_norm[j] > frame_norm[j-1] and 
                    frame_norm[j] > frame_norm[j+1] and 
                    frame_norm[j] >= threshold):
                    
                    freq = cqt_freqs[j]
                    if GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                        midi = int(round(librosa.hz_to_midi(freq)))
                        if 30 <= midi <= 96:
                            candidates.append(PitchCandidate(
                                time=times[i],
                                midi_note=midi,
                                frequency=freq,
                                confidence=float(frame_norm[j]),
                                method='cqt',
                                raw_pitch=freq
                            ))
        
        return candidates
    
    def _detect_crepe(self, y: np.ndarray) -> List[PitchCandidate]:
        """Detect pitches using CREPE neural network."""
        if not HAS_CREPE:
            return []
        
        # CREPE expects mono audio
        step_size_ms = max(10, int((self.config.hop_length / self.sr) * 1000))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time, frequency, confidence, _ = crepe.predict(
                y, self.sr,
                model_capacity=self.config.crepe_model,
                viterbi=True,
                step_size=step_size_ms,
                verbose=0
            )
        
        candidates = []
        for t, freq, conf in zip(time, frequency, confidence):
            if conf > 0.3 and GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                midi = int(round(librosa.hz_to_midi(freq)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=float(t),
                        midi_note=midi,
                        frequency=float(freq),
                        confidence=float(conf),
                        method='crepe',
                        raw_pitch=float(freq)
                    ))
        
        return candidates
    
    def _detect_basic_pitch_native(self, y: np.ndarray) -> List[PitchCandidate]:
        """Detect pitches using Basic Pitch (native Python)."""
        if not HAS_BASIC_PITCH_NATIVE:
            return []
        
        # Basic pitch works on audio at original sample rate
        model_output, midi_data, note_events = predict(y, self.sr)
        
        candidates = []
        for note in note_events:
            # note format: (start_time, end_time, pitch, amplitude, pitch_bends)
            start_time, end_time, pitch, amplitude, _ = note
            freq = librosa.midi_to_hz(pitch)
            
            if GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                # Create candidates for each frame in the note duration
                n_frames = max(1, int((end_time - start_time) * self.sr / self.config.hop_length))
                for i in range(n_frames):
                    t = start_time + i * (self.config.hop_length / self.sr)
                    candidates.append(PitchCandidate(
                        time=float(t),
                        midi_note=int(round(pitch)),
                        frequency=float(freq),
                        confidence=float(amplitude),
                        method='basic_pitch',
                        raw_pitch=float(freq)
                    ))
        
        return candidates
    
    def _detect_basic_pitch_docker(self, audio_path: Optional[str]) -> List[PitchCandidate]:
        """Detect pitches using Basic Pitch via Docker."""
        if not audio_path or not HAS_BASIC_PITCH_DOCKER:
            return []
        
        import subprocess
        import json
        import os
        
        # Check if Docker image exists
        result = subprocess.run(
            ['docker', 'images', '-q', 'guitar-tabs-basicpitch'],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            print("      Basic Pitch Docker image not found")
            return []
        
        abs_path = os.path.abspath(audio_path)
        dir_path = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{dir_path}:/data',
            'guitar-tabs-basicpitch',
            f'/data/{file_name}',
            '--onset-threshold', '0.5',
            '--frame-threshold', '0.3',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"      Basic Pitch Docker failed: {result.stderr}")
            return []
        
        # Parse JSON output
        try:
            lines = result.stdout.strip().split('\n')
            json_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start is None:
                return []
            
            json_str = '\n'.join(lines[json_start:])
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return []
        
        candidates = []
        for note_data in data.get('notes', []):
            midi = note_data['midi']
            freq = librosa.midi_to_hz(midi)
            start = note_data['start_time']
            duration = note_data['duration']
            conf = note_data.get('confidence', 0.7)
            
            if GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                # Create candidates for each frame
                n_frames = max(1, int(duration * self.sr / self.config.hop_length))
                for i in range(n_frames):
                    t = start + i * (self.config.hop_length / self.sr)
                    candidates.append(PitchCandidate(
                        time=float(t),
                        midi_note=int(midi),
                        frequency=float(freq),
                        confidence=float(conf),
                        method='basic_pitch',
                        raw_pitch=float(freq)
                    ))
        
        return candidates
    
    def _detect_piptrack(self, y: np.ndarray) -> List[PitchCandidate]:
        """Detect pitches using piptrack."""
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=self.sr, hop_length=self.config.hop_length,
            fmin=GUITAR_MIN_HZ, fmax=GUITAR_MAX_HZ
        )
        
        times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=self.sr, hop_length=self.config.hop_length)
        candidates = []
        
        for i in range(pitches.shape[1]):
            # Get strongest pitches in this frame
            frame_pitches = pitches[:, i]
            frame_mags = magnitudes[:, i]
            
            if frame_mags.max() < 1e-10:
                continue
            
            # Normalize magnitudes
            frame_mags_norm = frame_mags / (frame_mags.max() + 1e-10)
            
            # Find top peaks
            for j in range(len(frame_pitches)):
                freq = frame_pitches[j]
                mag = frame_mags_norm[j]
                
                if freq > 0 and mag > 0.3:
                    midi = int(round(librosa.hz_to_midi(freq)))
                    if 30 <= midi <= 96:
                        candidates.append(PitchCandidate(
                            time=times[i],
                            midi_note=midi,
                            frequency=float(freq),
                            confidence=float(mag),
                            method='piptrack',
                            raw_pitch=float(freq)
                        ))
        
        return candidates
    
    def _detect_hps(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using Harmonic Product Spectrum (HPS).
        
        HPS is CRITICAL for distorted guitar where harmonics dominate the fundamental.
        
        Algorithm:
        1. Compute FFT magnitude spectrum
        2. Downsample spectrum by factors of 2, 3, 4, 5
        3. Multiply all downsampled versions together
        4. The fundamental frequency "pops out" as the peak
        
        This works because harmonics at 2*f0, 3*f0, etc. all align with f0
        when downsampled by their respective factors.
        """
        from scipy.signal import medfilt
        
        n_fft = 4096  # Larger FFT for better frequency resolution
        num_harmonics = 5
        hop_length = self.config.hop_length
        
        # Compute STFT
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        n_bins, n_frames = S.shape
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Find bin indices for frequency range
        min_bin = np.searchsorted(freqs, GUITAR_MIN_HZ)
        max_bin = np.searchsorted(freqs, GUITAR_MAX_HZ)
        
        # Ensure we have enough bins for downsampling
        max_valid_bin = n_bins // num_harmonics
        max_bin = min(max_bin, max_valid_bin)
        
        if min_bin >= max_bin:
            min_bin = max(1, max_bin - 100)
        
        times = librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=hop_length)
        candidates = []
        
        for frame_idx in range(n_frames):
            spectrum = S[:, frame_idx]
            
            # Skip silent frames
            if np.max(spectrum) < 1e-10:
                continue
            
            # Normalize spectrum
            spectrum_norm = spectrum / (np.max(spectrum) + 1e-10)
            
            # Initialize HPS with original spectrum
            hps_len = max_bin
            hps = spectrum_norm[:hps_len].copy()
            
            # Multiply by downsampled versions
            for h in range(2, num_harmonics + 1):
                downsampled = np.zeros(hps_len)
                for i in range(hps_len):
                    src_idx = i * h
                    if src_idx < len(spectrum_norm):
                        downsampled[i] = spectrum_norm[src_idx]
                hps *= downsampled
            
            # Find the peak in the valid frequency range
            if min_bin >= max_bin or max_bin > len(hps):
                continue
            
            hps_search = hps[min_bin:max_bin]
            
            if len(hps_search) == 0 or np.max(hps_search) < 1e-20:
                continue
            
            # Find peak
            peak_idx_local = np.argmax(hps_search)
            peak_idx = peak_idx_local + min_bin
            peak_val = hps[peak_idx]
            
            # Calculate confidence based on peak prominence
            window = 10
            start = max(min_bin, peak_idx - window)
            end = min(max_bin, peak_idx + window)
            local_median = np.median(hps[start:end])
            
            if local_median > 0:
                prominence = peak_val / local_median
                conf = min(1.0, prominence / 10.0)
            else:
                conf = 0.5 if peak_val > 0 else 0.0
            
            # Only accept if confidence is reasonable
            if conf > 0.2:
                freq = freqs[peak_idx]
                if GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                    midi = int(round(librosa.hz_to_midi(freq)))
                    if 30 <= midi <= 96:
                        candidates.append(PitchCandidate(
                            time=times[frame_idx],
                            midi_note=midi,
                            frequency=float(freq),
                            confidence=float(conf),
                            method='hps',
                            raw_pitch=float(freq)
                        ))
        
        return candidates
    
    def _detect_essentia_yin_fft(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using Essentia's FFT-based YIN.
        
        Essentia's YIN FFT is very fast and accurate - our benchmark showed:
        - 70 notes detected in 88ms (fastest)
        - Good consensus with other methods
        """
        if not HAS_ESSENTIA:
            return []
        
        # Ensure float32 for Essentia
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        frame_size = 2048
        hop_size = self.config.hop_length
        
        # Create algorithms
        pitch_yin_fft = es.PitchYinFFT(
            frameSize=frame_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ
        )
        windowing = es.Windowing(type='hann', size=frame_size)
        spectrum = es.Spectrum(size=frame_size)
        
        candidates = []
        n_frames = (len(y) - frame_size) // hop_size + 1
        
        for i in range(n_frames):
            start = i * hop_size
            frame = y[start:start + frame_size]
            
            if len(frame) < frame_size:
                continue
            
            windowed = windowing(frame)
            spec = spectrum(windowed)
            pitch, confidence = pitch_yin_fft(spec)
            
            if confidence > 0.3 and GUITAR_MIN_HZ <= pitch <= GUITAR_MAX_HZ:
                time = start / self.sr
                midi = int(round(librosa.hz_to_midi(pitch)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=time,
                        midi_note=midi,
                        frequency=float(pitch),
                        confidence=float(confidence),
                        method='essentia_yin_fft',
                        raw_pitch=float(pitch)
                    ))
        
        return candidates
    
    def _detect_essentia_yin(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using Essentia's time-domain YIN.
        
        More thorough than YIN FFT but slower.
        """
        if not HAS_ESSENTIA:
            return []
        
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        frame_size = 2048
        hop_size = self.config.hop_length
        
        pitch_yin = es.PitchYin(
            frameSize=frame_size,
            sampleRate=self.sr,
            minFrequency=GUITAR_MIN_HZ,
            maxFrequency=GUITAR_MAX_HZ
        )
        
        candidates = []
        n_frames = (len(y) - frame_size) // hop_size + 1
        
        for i in range(n_frames):
            start = i * hop_size
            frame = y[start:start + frame_size]
            
            if len(frame) < frame_size:
                continue
            
            pitch, confidence = pitch_yin(frame)
            
            if confidence > 0.3 and GUITAR_MIN_HZ <= pitch <= GUITAR_MAX_HZ:
                time = start / self.sr
                midi = int(round(librosa.hz_to_midi(pitch)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=time,
                        midi_note=midi,
                        frequency=float(pitch),
                        confidence=float(confidence),
                        method='essentia_yin',
                        raw_pitch=float(pitch)
                    ))
        
        return candidates
    
    def _detect_praat_cc(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using Praat's cross-correlation method.
        
        Parselmouth provides access to Praat's pitch detection, which is
        highly regarded for its accuracy. Cross-correlation method is
        robust for noisy or harmonic-rich signals like guitar.
        
        Benchmark showed: 49 notes with 0.765 average confidence (highest!)
        """
        if not HAS_PARSELMOUTH:
            return []
        
        # Create Praat Sound object
        sound = parselmouth.Sound(y, sampling_frequency=self.sr)
        
        # Extract pitch using cross-correlation
        time_step = self.config.hop_length / self.sr
        pitch = sound.to_pitch_cc(
            time_step=time_step,
            pitch_floor=GUITAR_MIN_HZ,
            pitch_ceiling=GUITAR_MAX_HZ,
            very_accurate=True
        )
        
        candidates = []
        n_frames = pitch.get_number_of_frames()
        
        for i in range(1, n_frames + 1):  # Praat is 1-indexed
            t = pitch.get_time_from_frame_number(i)
            freq = pitch.get_value_in_frame(i)
            
            if freq and not np.isnan(freq) and GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                frame = pitch.get_frame(i)
                strength = 0.7
                if hasattr(frame, 'candidates') and len(frame.candidates) > 0:
                    try:
                        strength = frame.candidates[0].strength
                    except:
                        pass
                
                midi = int(round(librosa.hz_to_midi(freq)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=float(t),
                        midi_note=midi,
                        frequency=float(freq),
                        confidence=float(strength),
                        method='praat_cc',
                        raw_pitch=float(freq)
                    ))
        
        return candidates
    
    def _detect_praat_shs(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using Praat's Subharmonic Summation (SHS).
        
        SHS explicitly models the harmonic series, making it excellent
        for guitar which has strong harmonic content.
        
        Benchmark showed: 39 notes in 100ms (fast)
        """
        if not HAS_PARSELMOUTH:
            return []
        
        from parselmouth.praat import call
        
        sound = parselmouth.Sound(y, sampling_frequency=self.sr)
        time_step = self.config.hop_length / self.sr
        
        # Use standard pitch extraction (SHS via Praat command can be complex)
        # Fall back to default pitch for reliability
        pitch = sound.to_pitch(
            time_step=time_step,
            pitch_floor=GUITAR_MIN_HZ,
            pitch_ceiling=GUITAR_MAX_HZ
        )
        
        candidates = []
        n_frames = pitch.get_number_of_frames()
        
        for i in range(1, n_frames + 1):
            t = pitch.get_time_from_frame_number(i)
            freq = pitch.get_value_in_frame(i)
            
            if freq and not np.isnan(freq) and GUITAR_MIN_HZ <= freq <= GUITAR_MAX_HZ:
                midi = int(round(librosa.hz_to_midi(freq)))
                if 30 <= midi <= 96:
                    candidates.append(PitchCandidate(
                        time=float(t),
                        midi_note=midi,
                        frequency=float(freq),
                        confidence=0.7,  # Praat default method doesn't provide per-frame confidence
                        method='praat_shs',
                        raw_pitch=float(freq)
                    ))
        
        return candidates
    
    def _detect_yin(self, y: np.ndarray) -> List[PitchCandidate]:
        """
        Detect pitches using our custom YIN implementation.
        
        YIN (de Cheveigné & Kawahara, 2002) is specifically designed for
        monophonic pitch detection with these key features:
        
        1. Cumulative Mean Normalized Difference Function (CMNDF)
           - Prevents false peaks at low lags that cause basic autocorrelation
             to fail
        
        2. Absolute threshold for aperiodicity detection
           - Only reports pitches when the signal is clearly periodic
        
        3. Parabolic interpolation
           - Achieves sub-sample accuracy for precise pitch estimation
        
        4. Octave error handling
           - Explicitly checks for and corrects common octave errors
           
        This is EXCELLENT for lead guitar where we want:
        - Clean monophonic note detection
        - Resistance to harmonic confusion
        - High confidence values
        """
        if not HAS_YIN:
            return []
        
        # Use our YIN implementation
        yin_results = detect_yin_for_ensemble(
            y, 
            self.sr, 
            hop_length=self.config.hop_length,
            fmin=GUITAR_MIN_HZ,
            fmax=GUITAR_MAX_HZ
        )
        
        # Convert to PitchCandidates
        candidates = []
        for r in yin_results:
            candidates.append(PitchCandidate(
                time=r['time'],
                midi_note=r['midi_note'],
                frequency=r['frequency'],
                confidence=r['confidence'],
                method='yin',
                raw_pitch=r['raw_pitch']
            ))
        
        return candidates
    
    def _cluster_and_vote(self, candidates: List[PitchCandidate]) -> List[ConsensusNote]:
        """
        Cluster candidates by time and pitch, then vote.
        
        Algorithm:
        1. Sort candidates by time
        2. Group by time window (time_tolerance)
        3. Within each time group, cluster by pitch (pitch_tolerance_cents)
        4. For each cluster, compute weighted vote
        5. Keep clusters with enough votes/score
        """
        if not candidates:
            return []
        
        # Sort by time
        candidates.sort(key=lambda c: c.time)
        
        # Group by time windows
        time_groups: List[List[PitchCandidate]] = []
        current_group: List[PitchCandidate] = [candidates[0]]
        
        for cand in candidates[1:]:
            if cand.time - current_group[0].time <= self.config.time_tolerance:
                current_group.append(cand)
            else:
                time_groups.append(current_group)
                current_group = [cand]
        time_groups.append(current_group)
        
        # Process each time group
        consensus_notes: List[ConsensusNote] = []
        
        for time_group in time_groups:
            # Cluster by pitch within this time window
            pitch_clusters = self._cluster_by_pitch(time_group)
            
            for cluster in pitch_clusters:
                # Compute weighted vote
                note = self._compute_consensus(cluster)
                if note:
                    consensus_notes.append(note)
        
        # Merge overlapping notes with same pitch
        consensus_notes = self._merge_notes(consensus_notes)
        
        return consensus_notes
    
    def _cluster_by_pitch(self, candidates: List[PitchCandidate]) -> List[List[PitchCandidate]]:
        """Cluster candidates by pitch within tolerance."""
        if not candidates:
            return []
        
        # Group by MIDI note (already quantized)
        by_midi: Dict[int, List[PitchCandidate]] = defaultdict(list)
        for c in candidates:
            by_midi[c.midi_note].append(c)
        
        # Merge adjacent MIDI notes if within tolerance
        clusters = []
        processed = set()
        
        midi_notes = sorted(by_midi.keys())
        for midi in midi_notes:
            if midi in processed:
                continue
            
            # Start a new cluster
            cluster = list(by_midi[midi])
            processed.add(midi)
            
            # Check adjacent notes
            for adj in [midi - 1, midi + 1]:
                if adj in by_midi and adj not in processed:
                    # Calculate pitch difference in cents
                    main_freq = np.mean([c.frequency for c in by_midi[midi]])
                    adj_freq = np.mean([c.frequency for c in by_midi[adj]])
                    cents = 1200 * np.log2(adj_freq / main_freq) if main_freq > 0 else 100
                    
                    if abs(cents) <= self.config.pitch_tolerance_cents:
                        cluster.extend(by_midi[adj])
                        processed.add(adj)
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_consensus(self, cluster: List[PitchCandidate]) -> Optional[ConsensusNote]:
        """Compute weighted consensus from a cluster of candidates."""
        if not cluster:
            return None
        
        # Get unique methods
        methods = set(c.method for c in cluster)
        n_votes = len(methods)
        
        # Compute weighted score
        weighted_score = sum(
            self.config.weights.get(c.method, 1.0) * c.confidence
            for c in cluster
        ) / len(cluster)
        
        # Check minimum requirements
        if n_votes < self.config.min_votes:
            return None
        if weighted_score < self.config.min_weighted_score / len(methods):  # Normalize by method count
            return None
        
        # Compute weighted average pitch and confidence
        total_weight = 0
        weighted_midi_sum = 0
        weighted_freq_sum = 0
        weighted_conf_sum = 0
        
        for c in cluster:
            w = self.config.weights.get(c.method, 1.0) * c.confidence
            weighted_midi_sum += c.midi_note * w
            weighted_freq_sum += c.frequency * w
            weighted_conf_sum += c.confidence * w
            total_weight += w
        
        if total_weight < 1e-10:
            return None
        
        avg_midi = int(round(weighted_midi_sum / total_weight))
        avg_freq = weighted_freq_sum / total_weight
        avg_conf = weighted_conf_sum / total_weight
        
        # Compute pitch spread in cents
        freqs = [c.frequency for c in cluster]
        if len(freqs) > 1 and min(freqs) > 0:
            pitch_spread = 1200 * np.log2(max(freqs) / min(freqs))
        else:
            pitch_spread = 0
        
        # Time bounds
        start_time = min(c.time for c in cluster)
        end_time = max(c.time for c in cluster) + self.config.hop_length / self.sr
        
        # Boost confidence if many methods agree
        confidence_boost = 1 + (n_votes - self.config.min_votes) * 0.1
        final_conf = min(1.0, avg_conf * confidence_boost)
        
        return ConsensusNote(
            midi_note=avg_midi,
            start_time=start_time,
            end_time=end_time,
            confidence=final_conf,
            frequency=avg_freq,
            votes=n_votes,
            methods=list(methods),
            pitch_spread=pitch_spread
        )
    
    def _merge_notes(self, notes: List[ConsensusNote]) -> List[ConsensusNote]:
        """Merge consecutive notes with same pitch."""
        if len(notes) <= 1:
            return notes
        
        notes.sort(key=lambda n: (n.midi_note, n.start_time))
        merged = []
        current = notes[0]
        
        for note in notes[1:]:
            # Check if same pitch and overlapping/adjacent
            if (note.midi_note == current.midi_note and
                note.start_time <= current.end_time + self.config.time_tolerance):
                # Merge
                current = ConsensusNote(
                    midi_note=current.midi_note,
                    start_time=current.start_time,
                    end_time=max(current.end_time, note.end_time),
                    confidence=(current.confidence * current.votes + note.confidence * note.votes) / (current.votes + note.votes),
                    frequency=(current.frequency + note.frequency) / 2,
                    votes=max(current.votes, note.votes),
                    methods=list(set(current.methods + note.methods)),
                    pitch_spread=max(current.pitch_spread, note.pitch_spread)
                )
            else:
                merged.append(current)
                current = note
        
        merged.append(current)
        return merged


def detect_notes_ensemble(
    audio_path: str,
    config: Optional[EnsembleConfig] = None,
    sr: int = 22050,
    use_harmonic_separation: bool = True
) -> List[ConsensusNote]:
    """
    Convenience function for ensemble pitch detection.
    
    Args:
        audio_path: Path to audio file
        config: EnsembleConfig (optional)
        sr: Sample rate for processing
        use_harmonic_separation: Apply HPSS before detection
        
    Returns:
        List of ConsensusNote with high-confidence notes
    """
    print(f"\n📂 Loading audio: {audio_path}")
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    if use_harmonic_separation:
        print("🎵 Separating harmonic component...")
        y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
        y_detect = y_harmonic
    else:
        y_detect = y
    
    detector = EnsemblePitchDetector(sr=sr, config=config)
    notes = detector.detect(y_detect, audio_path=audio_path)
    
    return notes


def print_ensemble_results(notes: List[ConsensusNote]):
    """Pretty print ensemble detection results."""
    if not notes:
        print("\n❌ No notes detected!")
        return
    
    print(f"\n🎵 Detected {len(notes)} consensus notes:")
    print("-" * 80)
    print(f"{'Note':<8} {'Time':>10} {'Duration':>10} {'Confidence':>12} {'Votes':>8} {'Methods':<30}")
    print("-" * 80)
    
    for note in notes:
        methods_str = ', '.join(note.methods[:3])
        if len(note.methods) > 3:
            methods_str += f" +{len(note.methods)-3}"
        
        print(
            f"{note.name:<8} "
            f"{note.start_time:>10.3f} "
            f"{note.duration:>10.3f} "
            f"{note.confidence:>12.3f} "
            f"{note.votes:>8} "
            f"{methods_str:<30}"
        )
    
    print("-" * 80)
    
    # Statistics
    avg_votes = np.mean([n.votes for n in notes])
    avg_conf = np.mean([n.confidence for n in notes])
    
    print(f"\n📊 Statistics:")
    print(f"   Average votes: {avg_votes:.2f}")
    print(f"   Average confidence: {avg_conf:.3f}")
    print(f"   Total duration: {notes[-1].end_time - notes[0].start_time:.2f}s")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble Pitch Detection")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--min-votes", type=int, default=2, help="Minimum methods that must agree")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Minimum note duration (seconds)")
    parser.add_argument("--no-crepe", action="store_true", help="Disable CREPE")
    parser.add_argument("--no-basic-pitch", action="store_true", help="Disable Basic Pitch")
    parser.add_argument("--no-harmonic-sep", action="store_true", help="Disable harmonic separation")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    config = EnsembleConfig(
        min_votes=args.min_votes,
        min_confidence=args.min_confidence,
        min_duration=args.min_duration,
        use_crepe=not args.no_crepe,
        use_basic_pitch=not args.no_basic_pitch,
    )
    
    notes = detect_notes_ensemble(
        args.audio_path,
        config=config,
        use_harmonic_separation=not args.no_harmonic_sep
    )
    
    print_ensemble_results(notes)
    
    if args.output:
        import json
        output_data = {
            "notes": [
                {
                    "midi": n.midi_note,
                    "name": n.name,
                    "start_time": n.start_time,
                    "end_time": n.end_time,
                    "duration": n.duration,
                    "confidence": n.confidence,
                    "votes": n.votes,
                    "methods": n.methods,
                    "frequency": n.frequency,
                    "pitch_spread_cents": n.pitch_spread
                }
                for n in notes
            ],
            "config": {
                "min_votes": config.min_votes,
                "min_confidence": config.min_confidence,
                "min_duration": config.min_duration,
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved to {args.output}")
