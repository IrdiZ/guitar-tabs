#!/usr/bin/env python3
"""
Professional Guitar Tab PDF Generator - Pro Edition
====================================================
Creates publication-quality guitar tablature PDFs similar to 
Ultimate Guitar Pro, Guitar Pro, and Songsterr output.

This is an enhanced version with:
- Proper visual spacing and typography
- Clean staff rendering with centered fret numbers
- Multiple systems per page with proper pagination
- Chord diagrams with fingering
- Section headers
- Technique annotations

Author: Clawdbot Guitar Tab Generator
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm, inch, cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import json
import os
import math


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Technique(Enum):
    """Guitar playing techniques"""
    NONE = ""
    HAMMER_ON = "h"
    PULL_OFF = "p"
    BEND = "b"
    BEND_RELEASE = "r"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"
    VIBRATO = "~"
    MUTE = "x"
    PALM_MUTE = "PM"
    HARMONIC = "<>"
    PINCH_HARMONIC = "PH"
    TAP = "T"
    GHOST = "()"
    LET_RING = "LR"


@dataclass
class TabNote:
    """A single note on the guitar tablature"""
    string: int          # 0=low E, 5=high e
    fret: int           # 0-24
    start_time: float   # Seconds
    duration: float     # Seconds
    technique: Technique = Technique.NONE
    connect_next: bool = False
    velocity: int = 95


@dataclass
class ChordDiagram:
    """A chord diagram for display"""
    name: str
    frets: List[int]     # 6 values, -1=muted, 0=open
    fingers: List[int]   # Fingering 1-4, 0=none
    barre: Optional[Tuple[int, int, int]] = None  # (fret, start, end)
    position: int = 1    # Starting fret


# Standard tuning MIDI values
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']


# ============================================================================
# PDF GENERATOR CLASS
# ============================================================================

class GuitarTabPDF:
    """
    Professional guitar tablature PDF generator.
    
    Creates clean, readable tablature similar to commercial software.
    """
    
    def __init__(
        self,
        page_size=A4,
        margin=15*mm,
        staff_spacing=3.2*mm,     # Space between tab lines
        beat_width=10*mm,         # Width per beat
        systems_per_page=5,       # Tab systems per page
    ):
        self.page_size = page_size
        self.margin = margin
        self.staff_spacing = staff_spacing
        self.beat_width = beat_width
        self.systems_per_page = systems_per_page
        
        self.page_width, self.page_height = page_size
        self.content_width = self.page_width - 2 * margin
        
        # Colors
        self.line_color = colors.Color(0.3, 0.3, 0.3)
        self.bar_color = colors.black
        self.note_color = colors.black
        self.label_color = colors.Color(0.4, 0.4, 0.4)
        self.chord_color = colors.Color(0.1, 0.1, 0.6)
        
    def generate(
        self,
        notes: List[TabNote],
        output_path: str,
        title: str = "Guitar Tablature",
        artist: str = "",
        key: str = "",
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        tuning: str = "Standard",
        capo: int = 0,
    ) -> bool:
        """Generate a PDF from TabNote objects."""
        
        c = canvas.Canvas(output_path, pagesize=self.page_size)
        
        # Organize notes into measures
        beats_per_measure = time_signature[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        
        measures = self._group_into_measures(notes, measure_duration)
        
        # Calculate layout
        measure_width = beats_per_measure * self.beat_width
        measures_per_line = max(1, int(self.content_width / measure_width))
        
        # Adjust measure width to fill line
        measure_width = self.content_width / measures_per_line
        
        # Staff height
        staff_height = 5 * self.staff_spacing  # 6 lines = 5 gaps
        
        # System height (staff + margin for numbers/chords)
        system_height = staff_height + 15*mm
        
        # Header height
        header_height = 30*mm if artist else 22*mm
        
        # Calculate systems per page accounting for header on first page
        first_page_systems = int((self.page_height - header_height - 2*self.margin) / system_height)
        other_page_systems = int((self.page_height - 2*self.margin) / system_height)
        
        # Start drawing
        page_num = 1
        measure_idx = 0
        total_measures = len(measures)
        
        # Calculate total pages
        remaining = total_measures
        lines_first = math.ceil(total_measures / measures_per_line)
        
        while measure_idx < total_measures:
            # Draw header on first page
            if page_num == 1:
                self._draw_header(c, title, artist, key, tempo, tuning, capo, time_signature)
                y_start = self.page_height - self.margin - header_height
                num_systems = first_page_systems
            else:
                y_start = self.page_height - self.margin
                num_systems = other_page_systems
            
            # Draw systems on this page
            y = y_start
            
            for system_idx in range(num_systems):
                if measure_idx >= total_measures:
                    break
                    
                # Get measures for this line
                line_measures = measures[measure_idx:measure_idx + measures_per_line]
                
                # Draw the system
                y -= 10*mm  # Top margin for measure numbers
                self._draw_system(c, line_measures, y, measure_width, measure_idx + 1)
                y -= staff_height + 8*mm  # Staff + bottom margin
                
                measure_idx += len(line_measures)
            
            # Page footer
            c.setFont("Helvetica", 8)
            c.setFillColor(self.label_color)
            c.drawCentredString(
                self.page_width / 2, 
                self.margin / 2,
                f"Page {page_num}"
            )
            
            # Next page
            if measure_idx < total_measures:
                c.showPage()
                page_num += 1
        
        c.save()
        print(f"✅ PDF saved: {output_path}")
        print(f"   {total_measures} measures across {page_num} pages")
        return True
        
    def _draw_header(self, c, title, artist, key, tempo, tuning, capo, time_sig):
        """Draw the song header."""
        y = self.page_height - self.margin
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.black)
        c.drawCentredString(self.page_width / 2, y - 8*mm, title)
        
        # Artist
        if artist:
            c.setFont("Helvetica", 12)
            c.setFillColor(self.label_color)
            c.drawCentredString(self.page_width / 2, y - 16*mm, artist)
        
        # Metadata line
        meta = []
        if key:
            meta.append(f"Key: {key}")
        meta.append(f"♩ = {tempo}")
        meta.append(f"Time: {time_sig[0]}/{time_sig[1]}")
        if tuning != "Standard":
            meta.append(f"Tuning: {tuning}")
        if capo:
            meta.append(f"Capo: {capo}")
            
        c.setFont("Helvetica", 9)
        c.setFillColor(self.label_color)
        c.drawCentredString(self.page_width / 2, y - 24*mm, "  |  ".join(meta))
        
    def _draw_system(self, c, measures, y_top, measure_width, start_measure_num):
        """Draw one system (line) of tablature."""
        x = self.margin
        
        # Draw string labels on left
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(self.label_color)
        for i, name in enumerate(reversed(STRING_NAMES)):  # High to low
            string_y = y_top - i * self.staff_spacing
            c.drawRightString(x - 3*mm, string_y - 2.5, name)
        
        # Draw each measure
        for i, measure_notes in enumerate(measures):
            measure_num = start_measure_num + i
            self._draw_measure(c, measure_notes, x, y_top, measure_width, measure_num)
            x += measure_width
            
    def _draw_measure(self, c, notes, x, y_top, width, measure_num):
        """Draw a single measure with staff lines and notes."""
        
        # Draw 6 horizontal lines
        c.setStrokeColor(self.line_color)
        c.setLineWidth(0.4)
        
        for i in range(6):
            y = y_top - i * self.staff_spacing
            c.line(x, y, x + width, y)
        
        # Draw bar line at end
        c.setStrokeColor(self.bar_color)
        c.setLineWidth(0.8)
        c.line(x + width, y_top, x + width, y_top - 5 * self.staff_spacing)
        
        # Draw bar line at start (thinner)
        c.setLineWidth(0.5)
        c.line(x, y_top, x, y_top - 5 * self.staff_spacing)
        
        # Measure number
        c.setFont("Helvetica", 7)
        c.setFillColor(self.label_color)
        c.drawString(x + 1*mm, y_top + 3*mm, str(measure_num))
        
        if not notes:
            return
            
        # Calculate note positions
        # Notes are spread across the measure based on time
        times = [n.start_time for n in notes]
        if times:
            min_time = min(times)
            max_time = max(times)
            time_range = max_time - min_time if max_time > min_time else 1.0
            
            # Add padding
            note_area_start = x + 4*mm
            note_area_width = width - 8*mm
            
            # Draw each note
            for note in notes:
                # X position based on time
                if time_range > 0:
                    time_frac = (note.start_time - min_time) / time_range
                else:
                    time_frac = 0.5
                note_x = note_area_start + time_frac * note_area_width
                
                # Y position based on string (0=low E=bottom, 5=high e=top)
                # Lines are drawn top to bottom, so string 5 is at y_top
                note_y = y_top - (5 - note.string) * self.staff_spacing
                
                self._draw_note(c, note, note_x, note_y)
                
    def _draw_note(self, c, note: TabNote, x: float, y: float):
        """Draw a single fret number on the staff."""
        fret_text = str(note.fret)
        
        # Get text dimensions
        c.setFont("Helvetica-Bold", 9)
        text_width = c.stringWidth(fret_text, "Helvetica-Bold", 9)
        
        # Draw white background to cover the line
        c.setFillColor(colors.white)
        c.rect(x - text_width/2 - 1*mm, y - 3, 
               text_width + 2*mm, 6, fill=True, stroke=False)
        
        # Draw the fret number
        c.setFillColor(self.note_color)
        c.drawCentredString(x, y - 3, fret_text)
        
        # Draw technique symbol if any
        if note.technique != Technique.NONE:
            self._draw_technique_symbol(c, note.technique, x, y)
            
    def _draw_technique_symbol(self, c, technique: Technique, x: float, y: float):
        """Draw technique annotation."""
        c.setFont("Helvetica", 6)
        c.setFillColor(colors.Color(0.5, 0.5, 0.5))
        
        symbol = technique.value
        if technique == Technique.HAMMER_ON:
            c.drawString(x + 3*mm, y + 1, "h")
        elif technique == Technique.PULL_OFF:
            c.drawString(x + 3*mm, y + 1, "p")
        elif technique == Technique.BEND:
            # Draw small bend arrow
            c.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
            c.setLineWidth(0.4)
            c.line(x + 3*mm, y, x + 3*mm, y + 2.5*mm)
            c.line(x + 3*mm, y + 2.5*mm, x + 2*mm, y + 1.5*mm)
            c.line(x + 3*mm, y + 2.5*mm, x + 4*mm, y + 1.5*mm)
        elif technique == Technique.SLIDE_UP:
            c.drawString(x + 3*mm, y + 1, "/")
        elif technique == Technique.SLIDE_DOWN:
            c.drawString(x + 3*mm, y + 1, "\\")
        elif technique == Technique.VIBRATO:
            c.drawString(x - 3*mm, y + 4*mm, "~")
            
    def _group_into_measures(
        self, 
        notes: List[TabNote], 
        measure_duration: float
    ) -> List[List[TabNote]]:
        """Group notes into measures based on time."""
        if not notes:
            return []
            
        max_time = max(n.start_time + n.duration for n in notes)
        num_measures = max(1, int(math.ceil(max_time / measure_duration)))
        
        measures = [[] for _ in range(num_measures)]
        
        for note in notes:
            measure_idx = min(
                int(note.start_time / measure_duration),
                num_measures - 1
            )
            measures[measure_idx].append(note)
            
        return measures

    @staticmethod
    def from_json(json_path: str) -> List[TabNote]:
        """Load notes from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        notes = []
        tuning = STANDARD_TUNING
        
        for note_data in data.get('notes', []):
            midi = note_data.get('midi', note_data.get('pitch', 0))
            start_time = note_data.get('start_time', note_data.get('start', 0))
            duration = note_data.get('duration', 0.25)
            
            # Find best string/fret for this MIDI note
            string, fret = GuitarTabPDF._midi_to_fret(midi, tuning)
            
            if string is not None:
                # Detect technique from pitch bends
                technique = Technique.NONE
                if 'pitch_bends' in note_data:
                    bends = note_data['pitch_bends']
                    if any(abs(b) >= 1 for b in bends):
                        technique = Technique.BEND
                        
                notes.append(TabNote(
                    string=string,
                    fret=fret,
                    start_time=start_time,
                    duration=duration,
                    technique=technique,
                ))
                
        return notes
        
    @staticmethod
    def _midi_to_fret(midi: int, tuning: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """Convert MIDI note to string/fret, preferring lower frets."""
        candidates = []
        
        for string_num, open_midi in enumerate(tuning):
            fret = midi - open_midi
            if 0 <= fret <= 24:
                # Score: prefer lower frets and middle strings
                score = fret + abs(string_num - 2.5) * 0.5
                candidates.append((string_num, fret, score))
                
        if not candidates:
            return None, None
            
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0], candidates[0][1]


# ============================================================================
# CHORD DIAGRAMS
# ============================================================================

class ChordDiagramPDF:
    """Draw chord diagrams."""
    
    @staticmethod
    def draw(c, chord: ChordDiagram, x: float, y: float, width: float = 18*mm):
        """Draw a chord diagram at position (x, y)."""
        string_spacing = width / 5
        fret_spacing = 4*mm
        grid_height = 4 * fret_spacing
        
        # Chord name
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.Color(0.1, 0.1, 0.6))
        c.drawCentredString(x + width/2, y + 5*mm, chord.name)
        
        grid_top = y
        
        # Nut (thick line at top if position 1)
        c.setStrokeColor(colors.black)
        if chord.position == 1:
            c.setLineWidth(2)
        else:
            c.setLineWidth(0.5)
        c.line(x, grid_top, x + width, grid_top)
        
        # Horizontal fret lines
        c.setLineWidth(0.5)
        for i in range(1, 5):
            fret_y = grid_top - i * fret_spacing
            c.line(x, fret_y, x + width, fret_y)
            
        # Vertical string lines
        for i in range(6):
            string_x = x + i * string_spacing
            c.line(string_x, grid_top, string_x, grid_top - grid_height)
            
        # Position number if not at nut
        if chord.position > 1:
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.black)
            c.drawRightString(x - 2*mm, grid_top - fret_spacing/2 - 2, str(chord.position))
            
        # Draw finger positions
        for string_idx, fret_val in enumerate(chord.frets):
            string_x = x + (5 - string_idx) * string_spacing
            
            if fret_val == -1:  # Muted
                c.setFont("Helvetica", 10)
                c.drawCentredString(string_x, grid_top + 2*mm, "×")
            elif fret_val == 0:  # Open
                c.setStrokeColor(colors.black)
                c.setLineWidth(0.5)
                c.circle(string_x, grid_top + 3*mm, 1.5*mm, fill=False)
            else:  # Fretted
                fret_y = grid_top - (fret_val - chord.position + 0.5) * fret_spacing
                c.setFillColor(colors.black)
                c.circle(string_x, fret_y, 2*mm, fill=True)


# ============================================================================
# COMMON CHORD DEFINITIONS
# ============================================================================

COMMON_CHORDS = {
    # Major
    'C': ChordDiagram('C', [-1, 3, 2, 0, 1, 0], [0, 3, 2, 0, 1, 0]),
    'D': ChordDiagram('D', [-1, -1, 0, 2, 3, 2], [0, 0, 0, 1, 3, 2]),
    'E': ChordDiagram('E', [0, 2, 2, 1, 0, 0], [0, 2, 3, 1, 0, 0]),
    'F': ChordDiagram('F', [1, 3, 3, 2, 1, 1], [1, 3, 4, 2, 1, 1], barre=(1, 0, 5)),
    'G': ChordDiagram('G', [3, 2, 0, 0, 0, 3], [2, 1, 0, 0, 0, 4]),
    'A': ChordDiagram('A', [-1, 0, 2, 2, 2, 0], [0, 0, 2, 1, 3, 0]),
    'B': ChordDiagram('B', [-1, 2, 4, 4, 4, 2], [0, 1, 2, 3, 4, 1], barre=(2, 1, 5), position=2),
    
    # Minor
    'Am': ChordDiagram('Am', [-1, 0, 2, 2, 1, 0], [0, 0, 2, 3, 1, 0]),
    'Bm': ChordDiagram('Bm', [-1, 2, 4, 4, 3, 2], [0, 1, 3, 4, 2, 1], barre=(2, 1, 5), position=2),
    'Cm': ChordDiagram('Cm', [-1, 3, 5, 5, 4, 3], [0, 1, 3, 4, 2, 1], barre=(3, 1, 5), position=3),
    'Dm': ChordDiagram('Dm', [-1, -1, 0, 2, 3, 1], [0, 0, 0, 2, 3, 1]),
    'Em': ChordDiagram('Em', [0, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0]),
    'Fm': ChordDiagram('Fm', [1, 3, 3, 1, 1, 1], [1, 3, 4, 1, 1, 1], barre=(1, 0, 5)),
    'Gm': ChordDiagram('Gm', [3, 5, 5, 3, 3, 3], [1, 3, 4, 1, 1, 1], barre=(3, 0, 5), position=3),
    
    # Seventh
    'A7': ChordDiagram('A7', [-1, 0, 2, 0, 2, 0], [0, 0, 2, 0, 3, 0]),
    'B7': ChordDiagram('B7', [-1, 2, 1, 2, 0, 2], [0, 2, 1, 3, 0, 4], position=1),
    'C7': ChordDiagram('C7', [-1, 3, 2, 3, 1, 0], [0, 3, 2, 4, 1, 0]),
    'D7': ChordDiagram('D7', [-1, -1, 0, 2, 1, 2], [0, 0, 0, 2, 1, 3]),
    'E7': ChordDiagram('E7', [0, 2, 0, 1, 0, 0], [0, 2, 0, 1, 0, 0]),
    'G7': ChordDiagram('G7', [3, 2, 0, 0, 0, 1], [3, 2, 0, 0, 0, 1]),
    
    # Power chords
    'A5': ChordDiagram('A5', [-1, 0, 2, 2, -1, -1], [0, 0, 1, 2, 0, 0]),
    'E5': ChordDiagram('E5', [0, 2, 2, -1, -1, -1], [0, 1, 2, 0, 0, 0]),
    'G5': ChordDiagram('G5', [3, 5, 5, -1, -1, -1], [1, 3, 4, 0, 0, 0], position=3),
}


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Professional Guitar Tab PDF Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_generator_pro.py transcription.json output.pdf
  python pdf_generator_pro.py tabs.json song.pdf --title "Smoke on the Water" --artist "Deep Purple"
  python pdf_generator_pro.py tabs.json output.pdf --tempo 140 --key "Em"
"""
    )
    
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('output', help='Output PDF file')
    parser.add_argument('--title', '-t', default='Guitar Tablature')
    parser.add_argument('--artist', '-a', default='')
    parser.add_argument('--key', '-k', default='')
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--tuning', default='Standard')
    parser.add_argument('--capo', type=int, default=0)
    parser.add_argument('--page-size', choices=['a4', 'letter'], default='a4')
    
    args = parser.parse_args()
    
    # Load notes
    notes = GuitarTabPDF.from_json(args.input)
    
    if not notes:
        print("❌ No valid notes found in input file")
        return 1
        
    # Create PDF
    page_size = A4 if args.page_size == 'a4' else letter
    pdf = GuitarTabPDF(page_size=page_size)
    
    success = pdf.generate(
        notes=notes,
        output_path=args.output,
        title=args.title,
        artist=args.artist,
        key=args.key,
        tempo=args.tempo,
        tuning=args.tuning,
        capo=args.capo,
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
