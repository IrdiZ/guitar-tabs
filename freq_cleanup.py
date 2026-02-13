#!/usr/bin/env python3
"""
Frequency Domain Audio Cleanup for Guitar Tab Transcription

Pre-pitch-detection audio cleanup in the frequency domain:
1. Bandpass filter for guitar range (80Hz - 5kHz)
2. Remove DC offset
3. Apply spectral whitening (flatten spectrum)
4. Reduce noise floor (spectral gating)
5. Enhance transients (attack emphasis)

The goal: cleaner audio = more accurate pitch detection.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class FreqCleanupConfig:
    """Configuration for frequency domain cleanup."""
    # Master enable
    enabled: bool = True
    
    # Bandpass filter for guitar range
    bandpass: bool = True
    bandpass_low: float = 80.0    # Hz - E2 fundamental is ~82Hz
    bandpass_high: float = 5000.0  # Hz - high harmonics, above pick noise
    bandpass_order: int = 4
    
    # DC offset removal
    remove_dc: bool = True
    
    # Spectral whitening (flattens spectrum for better pitch detection)
    spectral_whitening: bool = True
    whitening_relaxation: float = 0.9  # 0-1, higher = more whitening
    whitening_floor: float = 0.01  # Minimum magnitude to prevent division by zero
    
    # Noise floor reduction (spectral gating)
    noise_reduction: bool = True
    noise_threshold_db: float = -40.0  # dB below peak to consider noise
    noise_smoothing_frames: int = 5  # Temporal smoothing for noise estimate
    
    # Transient enhancement
    transient_enhance: bool = True
    transient_strength: float = 0.5  # 0-1, how much to boost attack
    transient_decay_ms: float = 30.0  # ms for attack envelope decay
    
    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 512
    
    @classmethod
    def default(cls) -> 'FreqCleanupConfig':
        """Return default configuration."""
        return cls()
    
    @classmethod
    def aggressive(cls) -> 'FreqCleanupConfig':
        """More aggressive cleanup for noisy audio."""
        return cls(
            whitening_relaxation=0.95,
            noise_threshold_db=-35.0,
            transient_strength=0.7
        )
    
    @classmethod
    def gentle(cls) -> 'FreqCleanupConfig':
        """Gentle cleanup for clean recordings."""
        return cls(
            whitening_relaxation=0.7,
            noise_threshold_db=-50.0,
            transient_strength=0.3
        )
    
    @classmethod
    def pitch_optimized(cls) -> 'FreqCleanupConfig':
        """
        Optimized for pitch detection accuracy.
        
        Based on testing: transient enhancement and noise reduction
        help most. Aggressive bandpass/whitening can hurt clean audio.
        """
        return cls(
            # Skip bandpass - can hurt clean audio
            bandpass=False,
            # Always remove DC
            remove_dc=True,
            # Skip whitening - doesn't help much
            spectral_whitening=False,
            # Moderate noise reduction
            noise_reduction=True,
            noise_threshold_db=-45.0,
            # Transient enhancement is the most helpful
            transient_enhance=True,
            transient_strength=0.6,
            transient_decay_ms=25.0
        )
    
    @classmethod
    def noisy_recording(cls) -> 'FreqCleanupConfig':
        """
        For noisy recordings with background noise/hum.
        
        Full cleanup pipeline for challenging audio.
        """
        return cls(
            bandpass=True,
            bandpass_low=75.0,
            bandpass_high=6000.0,
            remove_dc=True,
            spectral_whitening=True,
            whitening_relaxation=0.8,
            noise_reduction=True,
            noise_threshold_db=-35.0,
            noise_smoothing_frames=7,
            transient_enhance=True,
            transient_strength=0.7
        )


def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio signal.
    
    Args:
        y: Audio signal
        
    Returns:
        Signal with DC offset removed
    """
    return y - np.mean(y)


def apply_bandpass_filter(
    y: np.ndarray,
    sr: int,
    low_freq: float = 80.0,
    high_freq: float = 5000.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter for guitar frequency range.
    
    Args:
        y: Audio signal
        sr: Sample rate
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Bandpass filtered signal
    """
    nyquist = sr / 2
    
    # Normalize frequencies
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Clamp to valid range
    low_norm = max(0.001, min(0.99, low_norm))
    high_norm = max(low_norm + 0.01, min(0.999, high_norm))
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # Apply zero-phase filtering
    filtered = signal.filtfilt(b, a, y)
    
    return filtered


def apply_spectral_whitening(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    relaxation: float = 0.9,
    floor: float = 0.01
) -> np.ndarray:
    """
    Apply spectral whitening to flatten the spectrum.
    
    Whitening helps pitch detectors by making all frequencies
    equally prominent, reducing harmonic confusion.
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        relaxation: Whitening strength (0-1, higher = more whitening)
        floor: Minimum magnitude floor
        
    Returns:
        Spectrally whitened signal
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Compute smoothed magnitude envelope (per frequency bin)
    # Using temporal average as the spectral shape
    mean_mag = np.mean(magnitude, axis=1, keepdims=True) + 1e-10
    
    # Compute whitening filter (inverse of spectral shape)
    whitening_filter = 1.0 / np.maximum(mean_mag, floor)
    
    # Normalize filter to preserve overall energy
    whitening_filter = whitening_filter / np.max(whitening_filter)
    
    # Apply with relaxation (blend original and whitened)
    whitened_mag = magnitude * (1 - relaxation + relaxation * whitening_filter)
    
    # Reconstruct with original phase
    D_whitened = whitened_mag * np.exp(1j * phase)
    
    # Inverse STFT
    y_whitened = librosa.istft(D_whitened, hop_length=hop_length, length=len(y))
    
    return y_whitened


def reduce_noise_floor(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    threshold_db: float = -40.0,
    smoothing_frames: int = 5
) -> np.ndarray:
    """
    Reduce noise floor using spectral gating.
    
    Attenuates frequency bins below a threshold, estimated
    from the quietest parts of the signal.
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        threshold_db: Threshold below peak (in dB)
        smoothing_frames: Temporal smoothing for noise estimate
        
    Returns:
        Noise-reduced signal
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Convert to dB
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Estimate noise floor as minimum over smoothed time windows
    # Use percentile instead of min to be more robust
    noise_floor_db = np.percentile(magnitude_db, 10, axis=1, keepdims=True)
    
    # Compute gate threshold
    peak_db = np.max(magnitude_db)
    gate_threshold_db = peak_db + threshold_db
    
    # Create soft gate (smooth transition)
    gate_db_range = 6.0  # dB range for smooth transition
    
    # Gate based on whether signal is above noise floor
    mask = np.clip(
        (magnitude_db - gate_threshold_db) / gate_db_range,
        0.0, 1.0
    )
    
    # Smooth mask temporally
    if smoothing_frames > 1:
        mask = uniform_filter1d(mask, size=smoothing_frames, axis=1)
    
    # Apply gate
    gated_magnitude = magnitude * mask
    
    # Reconstruct
    D_gated = gated_magnitude * np.exp(1j * phase)
    y_gated = librosa.istft(D_gated, hop_length=hop_length, length=len(y))
    
    return y_gated


def enhance_transients(
    y: np.ndarray,
    sr: int,
    strength: float = 0.5,
    decay_ms: float = 30.0,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Enhance transients (note attacks) for better onset detection.
    
    Uses spectral flux to identify transients and boosts them.
    
    Args:
        y: Audio signal
        sr: Sample rate
        strength: Transient enhancement strength (0-1)
        decay_ms: Attack envelope decay time in ms
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        Transient-enhanced signal
    """
    # Compute onset envelope (spectral flux)
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Normalize onset envelope
    onset_env = onset_env / (np.max(onset_env) + 1e-10)
    
    # Create attack envelope with decay
    decay_samples = int(decay_ms * sr / 1000)
    attack_env = np.zeros(len(y))
    
    # Map onset envelope to sample domain
    frame_times = librosa.frames_to_samples(
        np.arange(len(onset_env)),
        hop_length=hop_length
    )
    
    # Create decay kernel
    decay_kernel = np.exp(-np.arange(decay_samples) / (decay_samples / 4))
    
    # Find onset peaks
    onset_peaks = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units='samples'
    )
    
    # Apply attack emphasis at onset locations
    for peak_sample in onset_peaks:
        peak_frame = peak_sample // hop_length
        if peak_frame < len(onset_env):
            peak_strength = onset_env[peak_frame]
            
            # Create attack burst
            start = peak_sample
            end = min(len(y), start + decay_samples)
            length = end - start
            
            if length > 0:
                attack_env[start:end] = np.maximum(
                    attack_env[start:end],
                    peak_strength * decay_kernel[:length]
                )
    
    # Create boost envelope (1.0 + strength * attack)
    boost = 1.0 + strength * attack_env
    
    # Apply boost
    y_enhanced = y * boost
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(y_enhanced))
    if peak > 1.0:
        y_enhanced = y_enhanced / peak * 0.99
    
    return y_enhanced


def freq_domain_cleanup(
    y: np.ndarray,
    sr: int,
    config: Optional[FreqCleanupConfig] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply full frequency domain cleanup pipeline.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        config: Cleanup configuration
        verbose: Print progress messages
        
    Returns:
        Cleaned audio signal
    """
    if config is None:
        config = FreqCleanupConfig.default()
    
    if not config.enabled:
        return y
    
    output = y.copy()
    
    def log(msg):
        if verbose:
            print(f"  {msg}")
    
    # 1. Remove DC offset (always first)
    if config.remove_dc:
        dc_offset = np.mean(output)
        output = remove_dc_offset(output)
        log(f"⚡ Removed DC offset: {dc_offset:.6f}")
    
    # 2. Bandpass filter for guitar range
    if config.bandpass:
        log(f"🎸 Applying bandpass filter: {config.bandpass_low:.0f}Hz - {config.bandpass_high:.0f}Hz")
        output = apply_bandpass_filter(
            output, sr,
            low_freq=config.bandpass_low,
            high_freq=config.bandpass_high,
            order=config.bandpass_order
        )
    
    # 3. Spectral whitening
    if config.spectral_whitening:
        log(f"🎨 Applying spectral whitening (relaxation: {config.whitening_relaxation:.2f})")
        output = apply_spectral_whitening(
            output, sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            relaxation=config.whitening_relaxation,
            floor=config.whitening_floor
        )
    
    # 4. Noise floor reduction
    if config.noise_reduction:
        log(f"🔇 Reducing noise floor (threshold: {config.noise_threshold_db:.0f}dB)")
        output = reduce_noise_floor(
            output, sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            threshold_db=config.noise_threshold_db,
            smoothing_frames=config.noise_smoothing_frames
        )
    
    # 5. Transient enhancement
    if config.transient_enhance:
        log(f"⚡ Enhancing transients (strength: {config.transient_strength:.2f})")
        output = enhance_transients(
            output, sr,
            strength=config.transient_strength,
            decay_ms=config.transient_decay_ms,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
    
    # Final normalization to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    elif peak < 0.1:
        # Boost very quiet signals
        output = output / peak * 0.5
    
    return output


def analyze_audio_quality(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    verbose: bool = True
) -> dict:
    """
    Analyze audio quality metrics relevant to pitch detection.
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT size
        verbose: Print results
        
    Returns:
        Dictionary of quality metrics
    """
    # Compute spectrum
    D = librosa.stft(y, n_fft=n_fft)
    magnitude = np.abs(D)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Guitar range mask (80Hz - 5kHz)
    guitar_mask = (freqs >= 80) & (freqs <= 5000)
    
    # Metrics
    metrics = {}
    
    # 1. DC offset
    metrics['dc_offset'] = float(np.abs(np.mean(y)))
    
    # 2. Dynamic range
    peak_db = np.max(magnitude_db)
    noise_floor_db = np.percentile(magnitude_db, 5)
    metrics['dynamic_range_db'] = float(peak_db - noise_floor_db)
    
    # 3. Spectral flatness (how "white" the spectrum is)
    # Higher = more flat/white
    mean_energy = np.mean(magnitude[guitar_mask, :], axis=0)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft)
    metrics['spectral_flatness'] = float(np.mean(spectral_flatness))
    
    # 4. Energy concentration in guitar range
    total_energy = np.sum(magnitude ** 2)
    guitar_energy = np.sum(magnitude[guitar_mask, :] ** 2)
    metrics['guitar_range_energy_pct'] = float(100 * guitar_energy / (total_energy + 1e-10))
    
    # 5. Harmonic content (useful for pitch detection)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic ** 2)
    total_energy = np.sum(y ** 2)
    metrics['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-10))
    
    # 6. Transient clarity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    metrics['transient_clarity'] = float(np.max(onset_env) / (np.mean(onset_env) + 1e-10))
    
    if verbose:
        print("\n📊 Audio Quality Analysis:")
        print(f"   DC Offset: {metrics['dc_offset']:.6f}")
        print(f"   Dynamic Range: {metrics['dynamic_range_db']:.1f} dB")
        print(f"   Spectral Flatness: {metrics['spectral_flatness']:.4f}")
        print(f"   Guitar Range Energy: {metrics['guitar_range_energy_pct']:.1f}%")
        print(f"   Harmonic Ratio: {metrics['harmonic_ratio']:.2f}")
        print(f"   Transient Clarity: {metrics['transient_clarity']:.1f}")
    
    return metrics


def compare_cleanup(
    audio_path: str,
    output_path: Optional[str] = None,
    config: Optional[FreqCleanupConfig] = None
) -> Tuple[dict, dict]:
    """
    Compare audio quality before and after cleanup.
    
    Args:
        audio_path: Path to input audio
        output_path: Optional path to save cleaned audio
        config: Cleanup configuration
        
    Returns:
        Tuple of (before_metrics, after_metrics)
    """
    import soundfile as sf
    
    print(f"🎸 Frequency Domain Cleanup Test")
    print(f"   Input: {audio_path}")
    print()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr
    print(f"   Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
    
    # Analyze before
    print("\n📈 BEFORE cleanup:")
    before_metrics = analyze_audio_quality(y, sr)
    
    # Apply cleanup
    print("\n🔧 Applying frequency domain cleanup...")
    y_clean = freq_domain_cleanup(y, sr, config=config, verbose=True)
    
    # Analyze after
    print("\n📈 AFTER cleanup:")
    after_metrics = analyze_audio_quality(y_clean, sr)
    
    # Summary
    print("\n📊 IMPROVEMENT SUMMARY:")
    for key in before_metrics:
        before = before_metrics[key]
        after = after_metrics[key]
        if key == 'dc_offset':
            improvement = (before - after) / (before + 1e-10) * 100
            print(f"   {key}: {before:.6f} → {after:.6f} ({improvement:+.1f}% reduction)")
        elif key.endswith('_pct'):
            print(f"   {key}: {before:.1f}% → {after:.1f}%")
        elif key.endswith('_db'):
            print(f"   {key}: {before:.1f}dB → {after:.1f}dB ({after - before:+.1f}dB)")
        else:
            improvement = (after - before) / (before + 1e-10) * 100
            print(f"   {key}: {before:.4f} → {after:.4f} ({improvement:+.1f}%)")
    
    # Save if requested
    if output_path:
        sf.write(output_path, y_clean, sr)
        print(f"\n💾 Saved cleaned audio to: {output_path}")
    
    return before_metrics, after_metrics


def add_freq_cleanup_args(parser):
    """Add frequency cleanup arguments to an argument parser."""
    cleanup = parser.add_argument_group('Frequency Domain Cleanup')
    
    cleanup.add_argument(
        '--freq-cleanup',
        action='store_true',
        help='Enable frequency domain cleanup before pitch detection'
    )
    
    cleanup.add_argument(
        '--cleanup-preset',
        choices=['default', 'aggressive', 'gentle', 'pitch', 'noisy'],
        default='default',
        help='Cleanup preset: default, aggressive, gentle, pitch (optimized for pitch detection), noisy (for noisy recordings)'
    )
    
    cleanup.add_argument(
        '--bandpass-low',
        type=float,
        metavar='HZ',
        help='Bandpass low cutoff in Hz (default: 80)'
    )
    
    cleanup.add_argument(
        '--bandpass-high',
        type=float,
        metavar='HZ',
        help='Bandpass high cutoff in Hz (default: 5000)'
    )
    
    cleanup.add_argument(
        '--whitening',
        type=float,
        metavar='N',
        help='Spectral whitening strength 0-1 (default: 0.9)'
    )
    
    cleanup.add_argument(
        '--noise-threshold',
        type=float,
        metavar='DB',
        help='Noise floor threshold in dB (default: -40)'
    )
    
    cleanup.add_argument(
        '--transient-strength',
        type=float,
        metavar='N',
        help='Transient enhancement strength 0-1 (default: 0.5)'
    )
    
    cleanup.add_argument(
        '--save-cleaned',
        metavar='PATH',
        help='Save cleaned audio to file (for debugging)'
    )
    
    return cleanup


def config_from_args(args) -> FreqCleanupConfig:
    """Create config from argparse namespace."""
    # Start with preset
    preset = getattr(args, 'cleanup_preset', 'default')
    if preset == 'aggressive':
        config = FreqCleanupConfig.aggressive()
    elif preset == 'gentle':
        config = FreqCleanupConfig.gentle()
    elif preset == 'pitch':
        config = FreqCleanupConfig.pitch_optimized()
    elif preset == 'noisy':
        config = FreqCleanupConfig.noisy_recording()
    else:
        config = FreqCleanupConfig.default()
    
    # Override with specific args
    config.enabled = getattr(args, 'freq_cleanup', False)
    
    if hasattr(args, 'bandpass_low') and args.bandpass_low is not None:
        config.bandpass_low = args.bandpass_low
    if hasattr(args, 'bandpass_high') and args.bandpass_high is not None:
        config.bandpass_high = args.bandpass_high
    if hasattr(args, 'whitening') and args.whitening is not None:
        config.whitening_relaxation = args.whitening
    if hasattr(args, 'noise_threshold') and args.noise_threshold is not None:
        config.noise_threshold_db = args.noise_threshold
    if hasattr(args, 'transient_strength') and args.transient_strength is not None:
        config.transient_strength = args.transient_strength
    
    return config


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python freq_cleanup.py <audio_file> [output_file]")
        print("\nTests frequency domain cleanup on an audio file.")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    compare_cleanup(audio_path, output_path)
