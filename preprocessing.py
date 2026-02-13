"""
Audio Preprocessing Pipeline for Guitar Tab Transcription

Provides various audio processing techniques to clean up guitar recordings:
- Noise reduction / gate
- Compression to even out dynamics
- High-pass filter to remove rumble
- De-essing / reducing pick noise
- Harmonic enhancement to boost fundamental frequencies
- Auto-gain normalization
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Optional, Tuple
import librosa

# Try to import noisereduce
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing pipeline."""
    # Master enable
    enabled: bool = True
    
    # Noise reduction
    noise_reduce: bool = True
    noise_reduce_strength: float = 1.0  # 0.0-2.0, higher = more aggressive
    
    # Noise gate
    noise_gate: bool = True
    gate_threshold_db: float = -40.0  # dB below which to attenuate
    gate_attack_ms: float = 5.0
    gate_release_ms: float = 50.0
    gate_ratio: float = 10.0  # Attenuation ratio
    
    # Compression
    compress: bool = True
    compress_threshold_db: float = -20.0
    compress_ratio: float = 4.0
    compress_attack_ms: float = 10.0
    compress_release_ms: float = 100.0
    compress_makeup_db: float = 0.0  # Auto-calculated if 0
    
    # High-pass filter (remove rumble)
    highpass: bool = True
    highpass_freq: float = 70.0  # Hz, below E2
    highpass_order: int = 4
    
    # Low-pass filter (remove extreme highs/noise)
    lowpass: bool = False
    lowpass_freq: float = 8000.0  # Hz
    lowpass_order: int = 4
    
    # De-essing / pick noise reduction
    deess: bool = True
    deess_freq: float = 4000.0  # Target frequency for pick noise
    deess_threshold_db: float = -10.0
    deess_ratio: float = 3.0
    deess_bandwidth: float = 2000.0  # Hz bandwidth around target
    
    # Harmonic enhancement
    harmonic_enhance: bool = True
    harmonic_strength: float = 0.3  # 0.0-1.0
    harmonic_freq_range: Tuple[float, float] = (80.0, 1200.0)  # Guitar fundamental range
    
    # Auto-gain normalization
    normalize: bool = True
    target_db: float = -3.0  # Target peak level
    
    @classmethod
    def from_args(cls, args) -> 'PreprocessingConfig':
        """Create config from argparse namespace."""
        config = cls()
        
        # Check if preprocessing is enabled at all
        config.enabled = getattr(args, 'preprocess', False)
        
        if not config.enabled:
            return config
        
        # Individual toggles (enabled by default when preprocessing is on)
        config.noise_reduce = not getattr(args, 'no_noise_reduce', False)
        config.noise_gate = not getattr(args, 'no_gate', False)
        config.compress = not getattr(args, 'no_compress', False)
        config.highpass = not getattr(args, 'no_highpass', False)
        config.deess = not getattr(args, 'no_deess', False)
        config.harmonic_enhance = not getattr(args, 'no_harmonic', False)
        config.normalize = not getattr(args, 'no_normalize', False)
        
        # Parameter overrides
        if hasattr(args, 'noise_strength') and args.noise_strength is not None:
            config.noise_reduce_strength = args.noise_strength
        if hasattr(args, 'gate_threshold') and args.gate_threshold is not None:
            config.gate_threshold_db = args.gate_threshold
        if hasattr(args, 'compress_threshold') and args.compress_threshold is not None:
            config.compress_threshold_db = args.compress_threshold
        if hasattr(args, 'compress_ratio') and args.compress_ratio is not None:
            config.compress_ratio = args.compress_ratio
        if hasattr(args, 'highpass_freq') and args.highpass_freq is not None:
            config.highpass_freq = args.highpass_freq
        if hasattr(args, 'harmonic_strength') and args.harmonic_strength is not None:
            config.harmonic_strength = args.harmonic_strength
        if hasattr(args, 'target_db') and args.target_db is not None:
            config.target_db = args.target_db
        
        return config


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20 * np.log10(max(linear, 1e-10))


def apply_noise_reduction(
    y: np.ndarray,
    sr: int,
    strength: float = 1.0
) -> np.ndarray:
    """
    Apply noise reduction using spectral gating.
    
    Args:
        y: Audio signal
        sr: Sample rate
        strength: Reduction strength (0.0-2.0)
        
    Returns:
        Noise-reduced audio
    """
    if not HAS_NOISEREDUCE:
        print("  ⚠️  noisereduce not installed, skipping noise reduction")
        return y
    
    # Use the stationary noise reduction method
    # prop_decrease controls how much of the noise to remove
    prop_decrease = min(1.0, 0.5 * strength)
    
    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=prop_decrease,
        n_fft=2048,
        hop_length=512
    )
    
    return reduced


def apply_noise_gate(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    ratio: float = 10.0
) -> np.ndarray:
    """
    Apply a noise gate to reduce quiet passages.
    
    Args:
        y: Audio signal
        sr: Sample rate
        threshold_db: Gate threshold in dB
        attack_ms: Attack time in ms
        release_ms: Release time in ms
        ratio: Attenuation ratio below threshold
        
    Returns:
        Gated audio
    """
    threshold = db_to_linear(threshold_db)
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    # Calculate envelope using RMS
    frame_length = 2048
    hop_length = 512
    
    # Get RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Interpolate back to sample rate
    rms_interp = np.interp(
        np.arange(len(y)),
        np.linspace(0, len(y), len(rms)),
        rms
    )
    
    # Calculate gain reduction
    gain = np.ones_like(rms_interp)
    below_threshold = rms_interp < threshold
    
    # Apply ratio-based reduction below threshold
    if ratio > 1:
        reduction = 1.0 / ratio
        gain[below_threshold] = reduction + (1 - reduction) * (rms_interp[below_threshold] / threshold)
    
    # Smooth gain envelope (attack/release)
    smoothing_window = max(attack_samples, release_samples)
    if smoothing_window > 0:
        gain = uniform_filter1d(gain, size=smoothing_window)
    
    return y * gain


def apply_compression(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_db: float = 0.0
) -> np.ndarray:
    """
    Apply dynamic range compression.
    
    Args:
        y: Audio signal
        sr: Sample rate
        threshold_db: Compression threshold in dB
        ratio: Compression ratio (e.g., 4:1)
        attack_ms: Attack time in ms
        release_ms: Release time in ms
        makeup_db: Makeup gain in dB (0 = auto)
        
    Returns:
        Compressed audio
    """
    threshold = db_to_linear(threshold_db)
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    # Calculate envelope
    envelope = np.abs(y)
    
    # Smooth envelope with attack/release
    smoothed = np.zeros_like(envelope)
    smoothed[0] = envelope[0]
    
    attack_coeff = np.exp(-1.0 / max(attack_samples, 1))
    release_coeff = np.exp(-1.0 / max(release_samples, 1))
    
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            coeff = attack_coeff
        else:
            coeff = release_coeff
        smoothed[i] = coeff * smoothed[i-1] + (1 - coeff) * envelope[i]
    
    # Calculate gain reduction
    gain = np.ones_like(smoothed)
    above_threshold = smoothed > threshold
    
    if np.any(above_threshold):
        # Soft knee compression
        overshoot_db = 20 * np.log10(smoothed[above_threshold] / threshold + 1e-10)
        gain_reduction_db = overshoot_db * (1 - 1/ratio)
        gain[above_threshold] = db_to_linear(-gain_reduction_db)
    
    # Apply gain
    output = y * gain
    
    # Auto makeup gain: compensate for average gain reduction
    if makeup_db == 0 and np.any(above_threshold):
        avg_reduction = np.mean(gain[above_threshold])
        makeup_db = -linear_to_db(avg_reduction) * 0.5  # 50% compensation
    
    if makeup_db > 0:
        output = output * db_to_linear(makeup_db)
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output


def apply_highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff: float = 70.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply high-pass filter to remove low frequency rumble.
    
    Args:
        y: Audio signal
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        return y
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    filtered = signal.filtfilt(b, a, y)
    
    return filtered


def apply_lowpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff: float = 8000.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply low-pass filter to remove high frequency noise.
    
    Args:
        y: Audio signal
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        return y
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, y)
    
    return filtered


def apply_deessing(
    y: np.ndarray,
    sr: int,
    center_freq: float = 4000.0,
    bandwidth: float = 2000.0,
    threshold_db: float = -10.0,
    ratio: float = 3.0
) -> np.ndarray:
    """
    Apply de-essing / pick noise reduction.
    
    Reduces harsh high-frequency transients common with guitar picks.
    
    Args:
        y: Audio signal
        sr: Sample rate
        center_freq: Target frequency for reduction
        bandwidth: Bandwidth around target frequency
        threshold_db: Threshold for reduction
        ratio: Reduction ratio
        
    Returns:
        De-essed audio
    """
    nyquist = sr / 2
    
    # Design bandpass filter to isolate pick noise frequencies
    low_freq = max(100, center_freq - bandwidth/2)
    high_freq = min(nyquist - 100, center_freq + bandwidth/2)
    
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    if low_norm >= high_norm or high_norm >= 1.0:
        return y
    
    # Extract the sibilant frequency band
    b, a = signal.butter(2, [low_norm, high_norm], btype='band')
    sibilant = signal.filtfilt(b, a, y)
    
    # Calculate envelope of sibilant band
    envelope = np.abs(sibilant)
    smoothing = int(sr * 0.005)  # 5ms smoothing
    if smoothing > 1:
        envelope = uniform_filter1d(envelope, size=smoothing)
    
    # Calculate gain reduction
    threshold = db_to_linear(threshold_db) * np.max(envelope)
    gain = np.ones_like(envelope)
    
    above_threshold = envelope > threshold
    if np.any(above_threshold):
        overshoot = envelope[above_threshold] / threshold
        gain[above_threshold] = 1 / (1 + (overshoot - 1) * (1 - 1/ratio))
    
    # Smooth gain transitions
    gain = uniform_filter1d(gain, size=max(1, int(sr * 0.002)))
    
    # Apply gain only to sibilant band, keep rest intact
    sibilant_reduced = sibilant * gain
    
    # Subtract original sibilant, add reduced sibilant
    output = y - sibilant + sibilant_reduced
    
    return output


def apply_harmonic_enhancement(
    y: np.ndarray,
    sr: int,
    strength: float = 0.3,
    freq_range: Tuple[float, float] = (80.0, 1200.0)
) -> np.ndarray:
    """
    Enhance harmonic content to boost fundamental guitar frequencies.
    
    Uses harmonic/percussive separation to isolate and boost harmonics.
    
    Args:
        y: Audio signal
        sr: Sample rate
        strength: Enhancement strength (0.0-1.0)
        freq_range: Frequency range to enhance
        
    Returns:
        Harmonically enhanced audio
    """
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
    
    # Bandpass filter to isolate guitar fundamental range
    nyquist = sr / 2
    low_norm = freq_range[0] / nyquist
    high_norm = min(freq_range[1] / nyquist, 0.99)
    
    if low_norm < high_norm:
        b, a = signal.butter(2, [low_norm, high_norm], btype='band')
        harmonic_band = signal.filtfilt(b, a, y_harmonic)
        
        # Add enhanced harmonics back to original
        output = y + harmonic_band * strength
    else:
        # Fallback: just boost harmonic component
        output = y + y_harmonic * strength * 0.5
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output


def apply_normalization(
    y: np.ndarray,
    target_db: float = -3.0
) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        y: Audio signal
        target_db: Target peak level in dB
        
    Returns:
        Normalized audio
    """
    peak = np.max(np.abs(y))
    
    if peak < 1e-10:
        return y
    
    target_linear = db_to_linear(target_db)
    gain = target_linear / peak
    
    return y * gain


def preprocess_audio(
    y: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to audio.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        config: Preprocessing configuration
        verbose: Print progress messages
        
    Returns:
        Preprocessed audio
    """
    if not config.enabled:
        return y
    
    output = y.copy()
    
    def log(msg):
        if verbose:
            print(f"  {msg}")
    
    # 1. High-pass filter (remove rumble first)
    if config.highpass:
        log(f"🔊 Applying high-pass filter @ {config.highpass_freq:.0f} Hz")
        output = apply_highpass_filter(
            output, sr,
            cutoff=config.highpass_freq,
            order=config.highpass_order
        )
    
    # 2. Noise reduction
    if config.noise_reduce:
        log(f"🔇 Applying noise reduction (strength: {config.noise_reduce_strength:.1f})")
        output = apply_noise_reduction(
            output, sr,
            strength=config.noise_reduce_strength
        )
    
    # 3. Noise gate
    if config.noise_gate:
        log(f"🚪 Applying noise gate (threshold: {config.gate_threshold_db:.0f} dB)")
        output = apply_noise_gate(
            output, sr,
            threshold_db=config.gate_threshold_db,
            attack_ms=config.gate_attack_ms,
            release_ms=config.gate_release_ms,
            ratio=config.gate_ratio
        )
    
    # 4. De-essing / pick noise reduction
    if config.deess:
        log(f"✂️  Reducing pick noise @ {config.deess_freq:.0f} Hz")
        output = apply_deessing(
            output, sr,
            center_freq=config.deess_freq,
            bandwidth=config.deess_bandwidth,
            threshold_db=config.deess_threshold_db,
            ratio=config.deess_ratio
        )
    
    # 5. Compression
    if config.compress:
        log(f"📊 Applying compression ({config.compress_ratio:.1f}:1 @ {config.compress_threshold_db:.0f} dB)")
        output = apply_compression(
            output, sr,
            threshold_db=config.compress_threshold_db,
            ratio=config.compress_ratio,
            attack_ms=config.compress_attack_ms,
            release_ms=config.compress_release_ms,
            makeup_db=config.compress_makeup_db
        )
    
    # 6. Harmonic enhancement
    if config.harmonic_enhance:
        log(f"🎵 Enhancing harmonics (strength: {config.harmonic_strength:.1f})")
        output = apply_harmonic_enhancement(
            output, sr,
            strength=config.harmonic_strength,
            freq_range=config.harmonic_freq_range
        )
    
    # 7. Low-pass filter (optional)
    if config.lowpass:
        log(f"🔉 Applying low-pass filter @ {config.lowpass_freq:.0f} Hz")
        output = apply_lowpass_filter(
            output, sr,
            cutoff=config.lowpass_freq,
            order=config.lowpass_order
        )
    
    # 8. Final normalization
    if config.normalize:
        log(f"📈 Normalizing to {config.target_db:.0f} dB")
        output = apply_normalization(output, target_db=config.target_db)
    
    return output


def add_preprocessing_args(parser):
    """Add preprocessing arguments to an argument parser."""
    preproc = parser.add_argument_group('Audio Preprocessing')
    
    preproc.add_argument(
        '--preprocess', '-P',
        action='store_true',
        help='Enable audio preprocessing pipeline'
    )
    
    preproc.add_argument(
        '--save-preprocessed',
        metavar='PATH',
        help='Save preprocessed audio to file (for debugging)'
    )
    
    # Individual feature toggles (disable flags)
    preproc.add_argument(
        '--no-noise-reduce',
        action='store_true',
        help='Disable noise reduction'
    )
    preproc.add_argument(
        '--no-gate',
        action='store_true',
        help='Disable noise gate'
    )
    preproc.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable compression'
    )
    preproc.add_argument(
        '--no-highpass',
        action='store_true',
        help='Disable high-pass filter'
    )
    preproc.add_argument(
        '--no-deess',
        action='store_true',
        help='Disable de-essing / pick noise reduction'
    )
    preproc.add_argument(
        '--no-harmonic',
        action='store_true',
        help='Disable harmonic enhancement'
    )
    preproc.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable normalization'
    )
    
    # Parameter tuning
    preproc.add_argument(
        '--noise-strength',
        type=float,
        metavar='N',
        help='Noise reduction strength (0.0-2.0, default: 1.0)'
    )
    preproc.add_argument(
        '--gate-threshold',
        type=float,
        metavar='DB',
        help='Noise gate threshold in dB (default: -40)'
    )
    preproc.add_argument(
        '--compress-threshold',
        type=float,
        metavar='DB',
        help='Compression threshold in dB (default: -20)'
    )
    preproc.add_argument(
        '--compress-ratio',
        type=float,
        metavar='R',
        help='Compression ratio (default: 4.0)'
    )
    preproc.add_argument(
        '--highpass-freq',
        type=float,
        metavar='HZ',
        help='High-pass filter cutoff in Hz (default: 70)'
    )
    preproc.add_argument(
        '--harmonic-strength',
        type=float,
        metavar='N',
        help='Harmonic enhancement strength (0.0-1.0, default: 0.3)'
    )
    preproc.add_argument(
        '--target-db',
        type=float,
        metavar='DB',
        help='Target normalization level in dB (default: -3)'
    )
    
    return preproc
