#!/usr/bin/env python3
"""
Debug Spectrogram Visualization for Guitar Tab Generator

Generates spectrogram images with overlaid pitch detection results
to help visualize where detection is failing.

Features:
- Mel spectrogram with pitch overlays
- Shows detected notes as colored markers
- Shows onset times as vertical lines
- Confidence-based opacity
- Multiple view modes (full, zoomed sections)
- Guitar frequency range highlighting
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import os


# Guitar frequency ranges (Hz)
GUITAR_FREQ_MIN = 80    # Lowest note (E2 ≈ 82 Hz)
GUITAR_FREQ_MAX = 1400  # Highest practical fret (around 24th fret high E)

# MIDI to frequency conversion
def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def freq_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_name(midi_note: int) -> str:
    """Convert MIDI note to note name (e.g., 60 -> C4)."""
    octave = (midi_note // 12) - 1
    note = midi_note % 12
    return f"{NOTE_NAMES[note]}{octave}"


@dataclass
class DetectedNote:
    """Represents a detected note for visualization."""
    midi_note: int
    start_time: float
    end_time: float
    confidence: float
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = midi_to_name(self.midi_note)


def generate_debug_spectrogram(
    audio_path: str,
    detected_notes: List[DetectedNote],
    output_path: str = "debug_spectrogram.png",
    title: str = None,
    sr: int = 22050,
    hop_length: int = 512,
    n_mels: int = 256,
    fmin: float = 60,
    fmax: float = 2000,
    figsize: Tuple[int, int] = (20, 10),
    show_note_labels: bool = True,
    show_confidence: bool = True,
    time_range: Optional[Tuple[float, float]] = None,
    show_onset_lines: bool = True,
    colormap: str = 'magma',
) -> str:
    """
    Generate a debug spectrogram with pitch overlays.
    
    Args:
        audio_path: Path to the audio file
        detected_notes: List of DetectedNote objects to overlay
        output_path: Where to save the PNG
        title: Plot title (defaults to filename)
        sr: Sample rate for analysis
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        fmin: Minimum frequency for mel spectrogram
        fmax: Maximum frequency for mel spectrogram
        figsize: Figure size (width, height)
        show_note_labels: Whether to show note names
        show_confidence: Whether to vary opacity by confidence
        time_range: Optional (start, end) to zoom into a time range
        show_onset_lines: Show vertical lines at note onsets
        colormap: Colormap for spectrogram
        
    Returns:
        Path to saved PNG
    """
    print(f"📊 Generating debug spectrogram for: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    
    # Apply time range if specified
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        y = y[start_sample:end_sample]
        time_offset = time_range[0]
    else:
        time_offset = 0
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        hop_length=hop_length, 
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 0.5])
    
    # =========================================================================
    # SUBPLOT 1: Mel Spectrogram with Note Overlays
    # =========================================================================
    ax1 = axes[0]
    
    # Plot mel spectrogram
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time', 
        y_axis='mel',
        ax=ax1,
        cmap=colormap,
        fmin=fmin,
        fmax=fmax
    )
    
    # Overlay detected notes
    if detected_notes:
        # Filter notes to time range
        if time_range:
            notes_in_range = [
                n for n in detected_notes 
                if n.end_time >= time_range[0] and n.start_time <= time_range[1]
            ]
        else:
            notes_in_range = detected_notes
        
        # Create note rectangles
        for note in notes_in_range:
            freq = midi_to_freq(note.midi_note)
            
            # Skip if frequency is outside display range
            if freq < fmin or freq > fmax:
                continue
            
            # Adjust time for offset
            start = note.start_time - time_offset
            end = note.end_time - time_offset
            
            # Confidence-based color/alpha
            alpha = 0.3 + 0.5 * note.confidence if show_confidence else 0.6
            
            # Color based on confidence (red = low, green = high)
            color = plt.cm.RdYlGn(note.confidence)
            
            # Draw horizontal line at note frequency
            ax1.axhline(y=freq, xmin=start/duration, xmax=end/duration, 
                       color=color, linewidth=2, alpha=alpha)
            
            # Draw note marker at onset
            ax1.scatter([start], [freq], s=50, c=[color], 
                       edgecolors='white', linewidth=1, zorder=5, alpha=alpha)
            
            # Add note label
            if show_note_labels:
                ax1.annotate(
                    note.name, 
                    (start, freq),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=7,
                    color='white',
                    fontweight='bold',
                    alpha=min(alpha + 0.3, 1.0)
                )
        
        # Show onset lines
        if show_onset_lines:
            onset_times = sorted(set(n.start_time - time_offset for n in notes_in_range))
            for onset in onset_times:
                if onset >= 0:
                    ax1.axvline(x=onset, color='cyan', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('')
    
    # Add colorbar
    plt.colorbar(img, ax=ax1, format='%+2.0f dB', label='Power (dB)')
    
    # Title
    if title is None:
        title = os.path.basename(audio_path)
    ax1.set_title(f'{title}\nMel Spectrogram with Detected Pitches ({len(notes_in_range) if detected_notes else 0} notes)', 
                  fontsize=14, fontweight='bold')
    
    # =========================================================================
    # SUBPLOT 2: Pitch Detection Timeline (Piano Roll View)
    # =========================================================================
    ax2 = axes[1]
    
    if detected_notes:
        # Create piano roll visualization
        for note in notes_in_range if time_range else detected_notes:
            start = note.start_time - time_offset
            end = note.end_time - time_offset
            midi = note.midi_note
            
            # Confidence-based color
            color = plt.cm.RdYlGn(note.confidence)
            alpha = 0.5 + 0.4 * note.confidence if show_confidence else 0.8
            
            # Draw rectangle for each note
            rect = Rectangle(
                (start, midi - 0.4),
                end - start,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=alpha
            )
            ax2.add_patch(rect)
        
        # Set y-axis to MIDI range of detected notes
        midi_notes = [n.midi_note for n in (notes_in_range if time_range else detected_notes)]
        if midi_notes:
            ax2.set_ylim(min(midi_notes) - 2, max(midi_notes) + 2)
        
        # Add guitar string reference lines (standard tuning)
        guitar_strings = [40, 45, 50, 55, 59, 64]  # E A D G B e (open strings)
        string_names = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
        for midi, name in zip(guitar_strings, string_names):
            ax2.axhline(y=midi, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
            ax2.text(-0.02 * duration, midi, name, fontsize=8, ha='right', va='center', alpha=0.7)
    
    ax2.set_xlim(0, duration if not time_range else time_range[1] - time_range[0])
    ax2.set_ylabel('MIDI Note')
    ax2.set_xlabel('')
    ax2.set_title('Piano Roll View (Note Timeline)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 3: Confidence Distribution
    # =========================================================================
    ax3 = axes[2]
    
    if detected_notes:
        confidences = [n.confidence for n in (notes_in_range if time_range else detected_notes)]
        times = [n.start_time - time_offset for n in (notes_in_range if time_range else detected_notes)]
        
        # Scatter plot of confidence over time
        colors = [plt.cm.RdYlGn(c) for c in confidences]
        ax3.scatter(times, confidences, c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # Add moving average line
        if len(confidences) > 5:
            window = min(10, len(confidences) // 2)
            sorted_indices = np.argsort(times)
            sorted_times = np.array(times)[sorted_indices]
            sorted_conf = np.array(confidences)[sorted_indices]
            smoothed = np.convolve(sorted_conf, np.ones(window)/window, mode='valid')
            ax3.plot(sorted_times[window//2:window//2+len(smoothed)], smoothed, 
                    'b-', linewidth=2, alpha=0.7, label=f'Moving avg ({window} notes)')
            ax3.legend(loc='lower right', fontsize=8)
        
        ax3.axhline(y=0.5, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Typical threshold')
    
    ax3.set_xlim(0, duration if not time_range else time_range[1] - time_range[0])
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel('Confidence')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Detection Confidence Over Time', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Finalize and Save
    # =========================================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved debug spectrogram to: {output_path}")
    
    # Print statistics
    if detected_notes:
        confs = [n.confidence for n in detected_notes]
        print(f"\n📈 Detection Statistics:")
        print(f"   Total notes: {len(detected_notes)}")
        print(f"   Confidence: min={min(confs):.2f}, max={max(confs):.2f}, mean={np.mean(confs):.2f}")
        print(f"   Notes below 0.5 confidence: {sum(1 for c in confs if c < 0.5)}")
        print(f"   Duration: {duration:.2f}s")
        
        # Note frequency distribution
        midi_notes = [n.midi_note for n in detected_notes]
        print(f"   MIDI range: {min(midi_notes)} ({midi_to_name(min(midi_notes))}) - {max(midi_notes)} ({midi_to_name(max(midi_notes))})")
    
    return output_path


def generate_zoomed_spectrograms(
    audio_path: str,
    detected_notes: List[DetectedNote],
    output_dir: str = "debug_spectrograms",
    num_sections: int = 4,
    section_duration: float = 5.0,
    **kwargs
) -> List[str]:
    """
    Generate multiple zoomed spectrogram views for different sections of audio.
    
    Useful for long audio files where the full spectrogram is too compressed.
    
    Args:
        audio_path: Path to audio file
        detected_notes: List of detected notes
        output_dir: Directory to save spectrograms
        num_sections: Number of sections to generate
        section_duration: Duration of each section in seconds
        **kwargs: Additional arguments passed to generate_debug_spectrogram
        
    Returns:
        List of paths to generated PNGs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total duration
    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = len(y) / sr
    
    # Calculate section positions
    if num_sections == 1:
        sections = [(0, min(section_duration, total_duration))]
    else:
        # Evenly space sections, or use note-dense regions
        step = (total_duration - section_duration) / (num_sections - 1)
        sections = [(i * step, i * step + section_duration) for i in range(num_sections)]
    
    # Ensure sections don't exceed duration
    sections = [(max(0, s), min(e, total_duration)) for s, e in sections]
    
    output_paths = []
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    
    for i, (start, end) in enumerate(sections):
        output_path = os.path.join(output_dir, f"{basename}_section_{i+1}_{start:.1f}s-{end:.1f}s.png")
        generate_debug_spectrogram(
            audio_path,
            detected_notes,
            output_path=output_path,
            time_range=(start, end),
            title=f"{basename} - Section {i+1} ({start:.1f}s - {end:.1f}s)",
            **kwargs
        )
        output_paths.append(output_path)
    
    return output_paths


def generate_pitch_comparison_spectrogram(
    audio_path: str,
    detections_dict: Dict[str, List[DetectedNote]],
    output_path: str = "pitch_comparison.png",
    time_range: Optional[Tuple[float, float]] = None,
    **kwargs
) -> str:
    """
    Generate a spectrogram comparing multiple pitch detection methods.
    
    Args:
        audio_path: Path to audio file
        detections_dict: Dict mapping method names to their detected notes
        output_path: Where to save PNG
        time_range: Optional time range to zoom into
        **kwargs: Additional arguments
        
    Returns:
        Path to saved PNG
    """
    print(f"📊 Generating pitch comparison spectrogram...")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        y = y[start_sample:end_sample]
        time_offset = time_range[0]
        duration = time_range[1] - time_range[0]
    else:
        time_offset = 0
    
    num_methods = len(detections_dict)
    fig, axes = plt.subplots(num_methods + 1, 1, figsize=(20, 4 * (num_methods + 1)))
    
    # Compute mel spectrogram once
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmin=60, fmax=2000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # First subplot: raw spectrogram
    ax0 = axes[0]
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax0, cmap='magma', fmin=60, fmax=2000)
    ax0.set_title('Raw Mel Spectrogram (no overlays)', fontsize=12, fontweight='bold')
    ax0.set_xlabel('')
    
    # Method comparison subplots
    colors = plt.cm.tab10(np.linspace(0, 1, num_methods))
    
    for idx, (method_name, notes) in enumerate(detections_dict.items()):
        ax = axes[idx + 1]
        
        # Plot spectrogram
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='gray', fmin=60, fmax=2000)
        
        # Overlay notes with method-specific color
        color = colors[idx]
        
        if time_range:
            notes_in_range = [n for n in notes if n.end_time >= time_range[0] and n.start_time <= time_range[1]]
        else:
            notes_in_range = notes
        
        for note in notes_in_range:
            freq = midi_to_freq(note.midi_note)
            if 60 <= freq <= 2000:
                start = note.start_time - time_offset
                end = note.end_time - time_offset
                ax.plot([start, end], [freq, freq], color=color, linewidth=3, alpha=0.7)
                ax.scatter([start], [freq], s=40, c=[color], edgecolors='white', linewidth=1, zorder=5)
        
        ax.set_title(f'{method_name}: {len(notes_in_range)} notes detected', fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('')
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved comparison spectrogram to: {output_path}")
    return output_path


def main():
    """CLI interface for debug spectrogram generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate debug spectrogram visualizations for pitch detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic spectrogram (no notes, just visualize audio)
  python debug_spectrogram.py audio.mp3
  
  # With detected notes from JSON
  python debug_spectrogram.py audio.mp3 --notes-json transcribed.json
  
  # Zoomed view of first 10 seconds
  python debug_spectrogram.py audio.mp3 --time-range 0 10
  
  # Generate multiple section views
  python debug_spectrogram.py audio.mp3 --sections 4 --section-duration 5
"""
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('-o', '--output', default='debug_spectrogram.png', help='Output PNG path')
    parser.add_argument('--notes-json', help='JSON file with detected notes (from guitar_tabs.py)')
    parser.add_argument('--time-range', nargs=2, type=float, metavar=('START', 'END'),
                        help='Time range to visualize (start end) in seconds')
    parser.add_argument('--sections', type=int, default=0,
                        help='Generate multiple section views (0 = single full view)')
    parser.add_argument('--section-duration', type=float, default=5.0,
                        help='Duration of each section in seconds')
    parser.add_argument('--no-labels', action='store_true', help='Hide note labels')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 10], metavar=('W', 'H'),
                        help='Figure size (width height)')
    parser.add_argument('--colormap', default='magma', 
                        choices=['magma', 'viridis', 'plasma', 'inferno', 'hot', 'gray'],
                        help='Colormap for spectrogram')
    
    args = parser.parse_args()
    
    # Load notes from JSON if provided
    detected_notes = []
    if args.notes_json:
        import json
        with open(args.notes_json, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        notes_data = data.get('notes', data.get('detected_notes', data))
        if isinstance(notes_data, list):
            for n in notes_data:
                if isinstance(n, dict):
                    detected_notes.append(DetectedNote(
                        midi_note=n.get('midi_note', n.get('midi', 60)),
                        start_time=n.get('start_time', n.get('start', 0)),
                        end_time=n.get('end_time', n.get('end', n.get('start_time', 0) + 0.1)),
                        confidence=n.get('confidence', 0.5),
                        name=n.get('name', '')
                    ))
        
        print(f"📥 Loaded {len(detected_notes)} notes from {args.notes_json}")
    
    # Generate spectrogram(s)
    time_range = tuple(args.time_range) if args.time_range else None
    
    if args.sections > 0:
        # Generate multiple section views
        output_dir = os.path.splitext(args.output)[0] + "_sections"
        paths = generate_zoomed_spectrograms(
            args.audio_file,
            detected_notes,
            output_dir=output_dir,
            num_sections=args.sections,
            section_duration=args.section_duration,
            show_note_labels=not args.no_labels,
            figsize=tuple(args.figsize),
            colormap=args.colormap
        )
        print(f"\n✅ Generated {len(paths)} section spectrograms in {output_dir}/")
    else:
        # Generate single spectrogram
        generate_debug_spectrogram(
            args.audio_file,
            detected_notes,
            output_path=args.output,
            time_range=time_range,
            show_note_labels=not args.no_labels,
            figsize=tuple(args.figsize),
            colormap=args.colormap
        )


if __name__ == '__main__':
    main()
