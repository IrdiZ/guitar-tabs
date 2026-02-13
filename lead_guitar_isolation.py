#!/usr/bin/env python3
"""
Lead Guitar Isolation for Guitar Tab Transcription

Isolates lead guitar from a mix using multiple techniques:
1. Frequency band isolation - focus on lead guitar range (1-4kHz)
2. Source separation (Demucs) - extract guitar stem
3. Mid/side processing - extract centered lead guitar
4. Spectral peak analysis - find prominent single-note lines

Lead guitar characteristics:
- Higher pitch range (above rhythm guitar)
- More prominent in mix (often centered and louder)
- Single note lines (mostly monophonic)
- Specific frequency band (1-4kHz often boosted)
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import tempfile
import os
import subprocess
import warnings


@dataclass
class LeadIsolationConfig:
    """Configuration for lead guitar isolation."""
    # Master enable
    enabled: bool = True
    
    # =========================================================================
    # FREQUENCY BAND ISOLATION
    # =========================================================================
    freq_isolation: bool = True
    # Lead guitar fundamental range (typically higher than rhythm)
    lead_freq_low: float = 200.0      # Hz - above low E (82Hz) 
    lead_freq_high: float = 2000.0    # Hz - lead guitar fundamentals
    # Presence/cut-through frequencies
    presence_freq_low: float = 2000.0  # Hz
    presence_freq_high: float = 4000.0  # Hz - lead guitar "bite"
    # Boost amounts
    lead_band_boost_db: float = 3.0    # Boost lead fundamental range
    presence_boost_db: float = 2.0     # Boost presence range
    # Cut rhythm guitar low-end
    rhythm_cut_freq: float = 150.0     # Hz - cut below this
    rhythm_cut_db: float = 6.0         # dB to attenuate
    
    # =========================================================================
    # SOURCE SEPARATION (Demucs)
    # =========================================================================
    source_separation: bool = True
    demucs_model: str = "htdemucs"     # htdemucs, htdemucs_ft, mdx_extra
    # Which stems to combine for lead (usually just "other" contains guitar)
    use_other_stem: bool = True        # "other" often has guitar
    use_bass_stem: bool = False        # Sometimes bass bleeds into guitar
    # Post-separation filtering
    filter_separated: bool = True
    
    # =========================================================================
    # MID/SIDE PROCESSING
    # =========================================================================
    mid_side: bool = True
    # Lead guitar is typically center-panned
    mid_boost_db: float = 3.0          # Boost mid (center) channel
    side_cut_db: float = 6.0           # Cut side (stereo) channel
    # Mid-only mode: completely remove sides
    mid_only: bool = False
    
    # =========================================================================
    # SPECTRAL PEAK ENHANCEMENT
    # =========================================================================
    spectral_peak_enhance: bool = True
    # Number of peaks to keep per frame (lead is often single notes)
    max_peaks_per_frame: int = 3
    # Minimum peak prominence (relative to median)
    peak_prominence_ratio: float = 2.0
    # Frequency range for peak detection
    peak_freq_low: float = 150.0
    peak_freq_high: float = 3000.0
    
    # =========================================================================
    # TRANSIENT DETECTION (pick attacks)
    # =========================================================================
    transient_enhance: bool = True
    transient_boost_db: float = 2.0
    transient_decay_ms: float = 50.0
    
    # =========================================================================
    # OUTPUT SETTINGS
    # =========================================================================
    normalize_output: bool = True
    target_db: float = -3.0
    
    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 512


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20 * np.log10(max(linear, 1e-10))


def apply_frequency_isolation(
    y: np.ndarray,
    sr: int,
    config: LeadIsolationConfig
) -> np.ndarray:
    """
    Apply frequency band isolation for lead guitar.
    
    - Boosts the lead guitar fundamental range (200-2000Hz)
    - Boosts presence/cut-through frequencies (2-4kHz)
    - Cuts low rhythm guitar frequencies
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        config: Isolation configuration
        
    Returns:
        Frequency-isolated audio
    """
    if not config.freq_isolation:
        return y
    
    nyquist = sr / 2
    
    # Compute STFT
    D = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)
    
    # Create frequency-dependent gain curve
    gain = np.ones(len(freqs))
    
    # 1. Cut low frequencies (rhythm guitar territory)
    low_mask = freqs < config.rhythm_cut_freq
    gain[low_mask] *= db_to_linear(-config.rhythm_cut_db)
    
    # 2. Boost lead guitar fundamental range
    lead_mask = (freqs >= config.lead_freq_low) & (freqs <= config.lead_freq_high)
    gain[lead_mask] *= db_to_linear(config.lead_band_boost_db)
    
    # 3. Boost presence range
    presence_mask = (freqs >= config.presence_freq_low) & (freqs <= config.presence_freq_high)
    gain[presence_mask] *= db_to_linear(config.presence_boost_db)
    
    # Apply gain curve to magnitude
    magnitude_filtered = magnitude * gain[:, np.newaxis]
    
    # Reconstruct
    D_filtered = magnitude_filtered * np.exp(1j * phase)
    y_filtered = librosa.istft(D_filtered, hop_length=config.hop_length, length=len(y))
    
    return y_filtered


def run_demucs_separation(
    audio_path: str,
    model: str = "htdemucs",
    output_dir: Optional[str] = None
) -> dict:
    """
    Run Demucs source separation.
    
    Args:
        audio_path: Path to audio file
        model: Demucs model name
        output_dir: Output directory (temp if None)
        
    Returns:
        Dict with stem paths: {'drums': path, 'bass': path, 'other': path, 'vocals': path}
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="demucs_")
    
    # Run demucs
    cmd = [
        "python", "-m", "demucs",
        "--two-stems=other",  # Simplified: just separate into vocals + other
        "-n", model,
        "-o", output_dir,
        audio_path
    ]
    
    try:
        # Try full 4-stem separation first
        cmd_full = [
            "python", "-m", "demucs",
            "-n", model,
            "-o", output_dir,
            audio_path
        ]
        subprocess.run(cmd_full, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Fall back to two-stem
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Demucs separation failed: {e.stderr.decode()}")
    
    # Find output files
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, base_name)
    
    stems = {}
    for stem_name in ['drums', 'bass', 'other', 'vocals']:
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if os.path.exists(stem_path):
            stems[stem_name] = stem_path
    
    # Also check for no_vocals.wav from two-stem
    no_vocals = os.path.join(stem_dir, "no_vocals.wav")
    if os.path.exists(no_vocals) and 'other' not in stems:
        stems['other'] = no_vocals
    
    return stems


def apply_source_separation(
    audio_path: str,
    config: LeadIsolationConfig,
    verbose: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Use Demucs to isolate guitar from the mix.
    
    Args:
        audio_path: Path to audio file
        config: Isolation configuration
        verbose: Print progress
        
    Returns:
        Tuple of (isolated audio, sample rate)
    """
    if not config.source_separation:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    
    if verbose:
        print("  🎛️  Running Demucs source separation...")
    
    # Run separation
    try:
        stems = run_demucs_separation(audio_path, model=config.demucs_model)
    except Exception as e:
        warnings.warn(f"Demucs separation failed: {e}. Using original audio.")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    
    # Load and combine requested stems
    combined = None
    sr = None
    
    if config.use_other_stem and 'other' in stems:
        y, sr = librosa.load(stems['other'], sr=None, mono=True)
        combined = y
        if verbose:
            print("    ✓ Using 'other' stem (contains guitar)")
    
    if config.use_bass_stem and 'bass' in stems:
        y_bass, sr_bass = librosa.load(stems['bass'], sr=sr, mono=True)
        if combined is not None:
            # Resample if needed
            if len(y_bass) != len(combined):
                y_bass = librosa.resample(y_bass, orig_sr=sr_bass, target_sr=sr)
                if len(y_bass) > len(combined):
                    y_bass = y_bass[:len(combined)]
                else:
                    y_bass = np.pad(y_bass, (0, len(combined) - len(y_bass)))
            combined = combined + y_bass * 0.5  # Add bass at lower volume
        else:
            combined = y_bass
            sr = sr_bass
        if verbose:
            print("    ✓ Added 'bass' stem")
    
    if combined is None:
        warnings.warn("No stems extracted. Using original audio.")
        combined, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Apply post-separation filtering
    if config.filter_separated:
        combined = apply_frequency_isolation(combined, sr, config)
    
    return combined, sr


def apply_mid_side_processing(
    y_stereo: np.ndarray,
    sr: int,
    config: LeadIsolationConfig
) -> np.ndarray:
    """
    Apply mid/side processing to extract centered lead guitar.
    
    Lead guitar is typically panned center, while rhythm guitars
    are often panned left/right.
    
    Args:
        y_stereo: Stereo audio (2, samples) or mono (samples,)
        sr: Sample rate
        config: Isolation configuration
        
    Returns:
        Processed mono audio
    """
    if not config.mid_side:
        if y_stereo.ndim == 2:
            return np.mean(y_stereo, axis=0)
        return y_stereo
    
    # Handle mono input
    if y_stereo.ndim == 1:
        return y_stereo  # Can't do M/S on mono
    
    if y_stereo.shape[0] != 2:
        # Might be (samples, 2), transpose
        if y_stereo.shape[1] == 2:
            y_stereo = y_stereo.T
        else:
            return np.mean(y_stereo, axis=0)
    
    left = y_stereo[0]
    right = y_stereo[1]
    
    # Convert to mid/side
    mid = (left + right) / 2
    side = (left - right) / 2
    
    if config.mid_only:
        # Return just the mid channel
        return mid
    
    # Apply gain adjustments
    mid_gain = db_to_linear(config.mid_boost_db)
    side_gain = db_to_linear(-config.side_cut_db)
    
    mid_boosted = mid * mid_gain
    side_cut = side * side_gain
    
    # Convert back to stereo, then to mono
    left_out = mid_boosted + side_cut
    right_out = mid_boosted - side_cut
    
    # Return mono mix
    return (left_out + right_out) / 2


def apply_spectral_peak_enhancement(
    y: np.ndarray,
    sr: int,
    config: LeadIsolationConfig
) -> np.ndarray:
    """
    Enhance spectral peaks to focus on lead guitar notes.
    
    Lead guitar typically plays single notes that stand out
    as prominent spectral peaks.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        config: Isolation configuration
        
    Returns:
        Peak-enhanced audio
    """
    if not config.spectral_peak_enhance:
        return y
    
    # Compute STFT
    D = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)
    
    # Find frequency range indices
    freq_mask = (freqs >= config.peak_freq_low) & (freqs <= config.peak_freq_high)
    freq_indices = np.where(freq_mask)[0]
    
    # Create output magnitude
    magnitude_out = np.zeros_like(magnitude)
    
    # Process each frame
    for frame_idx in range(magnitude.shape[1]):
        frame_mag = magnitude[freq_indices, frame_idx]
        
        if len(frame_mag) == 0 or np.max(frame_mag) < 1e-10:
            continue
        
        # Find peaks
        median_mag = np.median(frame_mag)
        threshold = median_mag * config.peak_prominence_ratio
        
        # Get peak indices
        peaks, properties = signal.find_peaks(
            frame_mag,
            height=threshold,
            prominence=median_mag * 0.5
        )
        
        if len(peaks) == 0:
            # No prominent peaks, keep original
            magnitude_out[freq_indices, frame_idx] = frame_mag
            continue
        
        # Sort by height and keep top N
        peak_heights = frame_mag[peaks]
        top_peak_idx = np.argsort(peak_heights)[-config.max_peaks_per_frame:]
        top_peaks = peaks[top_peak_idx]
        
        # Create mask: boost peaks, attenuate others
        mask = np.zeros_like(frame_mag)
        
        # Gaussian windows around each peak
        for peak in top_peaks:
            # Width based on frequency (narrower at higher freq)
            width = max(3, int(10 * (1 - peak / len(frame_mag))))
            start = max(0, peak - width)
            end = min(len(frame_mag), peak + width + 1)
            
            # Gaussian boost
            x = np.arange(start, end) - peak
            gaussian = np.exp(-0.5 * (x / (width / 2)) ** 2)
            mask[start:end] = np.maximum(mask[start:end], gaussian)
        
        # Apply mask: peaks at full, others attenuated
        attenuation = 0.3  # Keep 30% of non-peak content
        frame_out = frame_mag * (mask + (1 - mask) * attenuation)
        magnitude_out[freq_indices, frame_idx] = frame_out
    
    # Keep frequencies outside our range unchanged
    magnitude_out[~freq_mask, :] = magnitude[~freq_mask, :] * 0.5
    
    # Reconstruct
    D_out = magnitude_out * np.exp(1j * phase)
    y_out = librosa.istft(D_out, hop_length=config.hop_length, length=len(y))
    
    return y_out


def apply_transient_enhancement(
    y: np.ndarray,
    sr: int,
    config: LeadIsolationConfig
) -> np.ndarray:
    """
    Enhance transients (pick attacks) for clearer note detection.
    
    Args:
        y: Audio signal
        sr: Sample rate
        config: Isolation configuration
        
    Returns:
        Transient-enhanced audio
    """
    if not config.transient_enhance:
        return y
    
    # Detect transients using onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
    
    # Create transient envelope at sample rate
    transient_samples = int(config.transient_decay_ms * sr / 1000)
    
    # Interpolate onset envelope to sample rate
    onset_interp = np.interp(
        np.arange(len(y)),
        np.linspace(0, len(y), len(onset_env)),
        onset_env
    )
    
    # Normalize
    onset_interp = onset_interp / (np.max(onset_interp) + 1e-10)
    
    # Apply exponential decay
    decayed = np.zeros_like(onset_interp)
    decay_factor = np.exp(-1.0 / max(transient_samples, 1))
    
    for i in range(len(onset_interp)):
        if i == 0:
            decayed[i] = onset_interp[i]
        else:
            decayed[i] = max(onset_interp[i], decayed[i-1] * decay_factor)
    
    # Convert to gain
    boost = db_to_linear(config.transient_boost_db)
    gain = 1.0 + (boost - 1.0) * decayed
    
    return y * gain


def normalize_audio(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to target peak level."""
    peak = np.max(np.abs(y))
    if peak < 1e-10:
        return y
    target_linear = db_to_linear(target_db)
    return y * (target_linear / peak)


def isolate_lead_guitar(
    audio_path: str,
    config: Optional[LeadIsolationConfig] = None,
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Full lead guitar isolation pipeline.
    
    Args:
        audio_path: Path to audio file
        config: Isolation configuration (default if None)
        verbose: Print progress messages
        save_path: Optional path to save isolated audio
        
    Returns:
        Tuple of (isolated audio, sample rate)
    """
    if config is None:
        config = LeadIsolationConfig()
    
    if not config.enabled:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    
    if verbose:
        print("\n🎸 Lead Guitar Isolation Pipeline")
        print("-" * 40)
    
    # 1. Source separation (Demucs)
    y, sr = apply_source_separation(audio_path, config, verbose)
    if verbose:
        print(f"  ✓ Source separation complete")
    
    # 2. Load original stereo for M/S processing
    if config.mid_side:
        try:
            y_stereo, sr_stereo = librosa.load(audio_path, sr=sr, mono=False)
            if y_stereo.ndim == 2:
                y_ms = apply_mid_side_processing(y_stereo, sr, config)
                # Blend with source-separated
                y = y * 0.6 + y_ms * 0.4
                if verbose:
                    print(f"  ✓ Mid/Side processing applied")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Mid/Side processing skipped: {e}")
    
    # 3. Frequency band isolation
    if config.freq_isolation:
        y = apply_frequency_isolation(y, sr, config)
        if verbose:
            print(f"  ✓ Frequency isolation applied")
    
    # 4. Spectral peak enhancement
    if config.spectral_peak_enhance:
        y = apply_spectral_peak_enhancement(y, sr, config)
        if verbose:
            print(f"  ✓ Spectral peak enhancement applied")
    
    # 5. Transient enhancement
    if config.transient_enhance:
        y = apply_transient_enhancement(y, sr, config)
        if verbose:
            print(f"  ✓ Transient enhancement applied")
    
    # 6. Normalize
    if config.normalize_output:
        y = normalize_audio(y, config.target_db)
        if verbose:
            print(f"  ✓ Normalized to {config.target_db} dB")
    
    # Save if requested
    if save_path:
        import soundfile as sf
        sf.write(save_path, y, sr)
        if verbose:
            print(f"  💾 Saved to: {save_path}")
    
    return y, sr


def isolate_lead_from_array(
    y: np.ndarray,
    sr: int,
    config: Optional[LeadIsolationConfig] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Isolate lead guitar from an audio array (no source separation).
    
    This version works on already-loaded audio, applying frequency
    and spectral processing but skipping Demucs.
    
    Args:
        y: Audio signal (can be stereo or mono)
        sr: Sample rate
        config: Isolation configuration
        verbose: Print progress
        
    Returns:
        Isolated mono audio
    """
    if config is None:
        config = LeadIsolationConfig()
        config.source_separation = False  # No file path available
    
    if not config.enabled:
        if y.ndim == 2:
            return np.mean(y, axis=0)
        return y
    
    # Handle stereo
    if y.ndim == 2:
        if config.mid_side:
            y = apply_mid_side_processing(y, sr, config)
        else:
            y = np.mean(y, axis=0)
    
    # Apply processing chain
    if config.freq_isolation:
        y = apply_frequency_isolation(y, sr, config)
    
    if config.spectral_peak_enhance:
        y = apply_spectral_peak_enhancement(y, sr, config)
    
    if config.transient_enhance:
        y = apply_transient_enhancement(y, sr, config)
    
    if config.normalize_output:
        y = normalize_audio(y, config.target_db)
    
    return y


# =============================================================================
# ARGPARSE INTEGRATION
# =============================================================================

def add_lead_isolation_args(parser):
    """Add lead guitar isolation arguments to an argument parser."""
    lead_group = parser.add_argument_group('Lead Guitar Isolation')
    
    lead_group.add_argument(
        '--lead-only',
        action='store_true',
        help='Enable lead guitar isolation (separates lead from rhythm/bass)'
    )
    
    lead_group.add_argument(
        '--lead-no-separation',
        action='store_true',
        help='Disable Demucs source separation (use frequency processing only)'
    )
    
    lead_group.add_argument(
        '--lead-no-midside',
        action='store_true',
        help='Disable mid/side processing'
    )
    
    lead_group.add_argument(
        '--lead-no-peaks',
        action='store_true',
        help='Disable spectral peak enhancement'
    )
    
    lead_group.add_argument(
        '--lead-demucs-model',
        type=str,
        default='htdemucs',
        choices=['htdemucs', 'htdemucs_ft', 'mdx_extra'],
        help='Demucs model for source separation (default: htdemucs)'
    )
    
    lead_group.add_argument(
        '--lead-freq-low',
        type=float,
        default=200.0,
        metavar='HZ',
        help='Low frequency cutoff for lead band (default: 200 Hz)'
    )
    
    lead_group.add_argument(
        '--lead-freq-high',
        type=float,
        default=2000.0,
        metavar='HZ',
        help='High frequency cutoff for lead band (default: 2000 Hz)'
    )
    
    lead_group.add_argument(
        '--lead-mid-only',
        action='store_true',
        help='Extract only mid channel (removes all stereo content)'
    )
    
    lead_group.add_argument(
        '--save-lead-audio',
        type=str,
        metavar='PATH',
        help='Save isolated lead guitar audio to file'
    )
    
    return lead_group


def config_from_args(args) -> LeadIsolationConfig:
    """Create LeadIsolationConfig from argparse namespace."""
    config = LeadIsolationConfig()
    
    config.enabled = getattr(args, 'lead_only', False)
    
    if not config.enabled:
        return config
    
    # Source separation
    config.source_separation = not getattr(args, 'lead_no_separation', False)
    config.demucs_model = getattr(args, 'lead_demucs_model', 'htdemucs')
    
    # Mid/side
    config.mid_side = not getattr(args, 'lead_no_midside', False)
    config.mid_only = getattr(args, 'lead_mid_only', False)
    
    # Spectral peaks
    config.spectral_peak_enhance = not getattr(args, 'lead_no_peaks', False)
    
    # Frequency settings
    if hasattr(args, 'lead_freq_low') and args.lead_freq_low:
        config.lead_freq_low = args.lead_freq_low
    if hasattr(args, 'lead_freq_high') and args.lead_freq_high:
        config.lead_freq_high = args.lead_freq_high
    
    return config


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(
        description="Isolate lead guitar from audio mix"
    )
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-o', '--output', help='Output audio file')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    add_lead_isolation_args(parser)
    
    # Override: always enable for CLI
    args = parser.parse_args()
    args.lead_only = True
    
    config = config_from_args(args)
    
    output_path = args.output or args.input.replace('.', '_lead.')
    
    y, sr = isolate_lead_guitar(
        args.input,
        config=config,
        verbose=args.verbose,
        save_path=output_path
    )
    
    print(f"\n✅ Lead guitar isolated: {output_path}")
