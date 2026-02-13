#!/usr/bin/env python3
"""
Professional Guitar Tab PDF Generator
=====================================
Creates publication-quality guitar tablature PDFs similar to 
Ultimate Guitar, Guitar Pro, and Songsterr output.

Features:
- Proper 6-line tab staff with precise spacing
- Fret numbers centered on lines (not between)
- Technique symbols (h, p, b, /, \\, ~, x) properly positioned
- Chord diagrams above staff
- Bar lines and measure numbers
- Title, artist, key, tempo header
- Multiple pages with proper pagination
- Optional standard notation staff above tab

Author: Clawdbot Guitar Tab Generator
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Flowable, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, String, Circle, Rect, Group
from reportlab.graphics import renderPDF
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
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
    RELEASE = "r"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"
    VIBRATO = "~"
    MUTE = "x"
    HARMONIC = "*"
    TAP = "t"
    GHOST = "()"
    

@dataclass
class TabNote:
    """A single note on the guitar tablature"""
    string: int          # 0-5 (low E=0 to high e=5)
    fret: int           # 0-24
    start_time: float   # In seconds
    duration: float     # In seconds
    technique: Technique = Technique.NONE
    connect_to_next: bool = False  # For ties/slides/hammer-ons
    velocity: int = 95  # 0-127
    
    
@dataclass
class TabMeasure:
    """A single measure of tablature"""
    number: int
    notes: List[TabNote] = field(default_factory=list)
    time_signature: Tuple[int, int] = (4, 4)
    tempo: Optional[int] = None  # Only set if tempo changes
    

@dataclass 
class ChordDiagram:
    """A chord diagram for display above the staff"""
    name: str           # e.g., "Am", "G", "Cmaj7"
    frets: List[int]    # 6 values, -1 = muted, 0 = open
    finger: List[int]   # Fingering (1-4), 0 = no finger
    barre: Optional[Tuple[int, int, int]] = None  # (fret, start_string, end_string)
    position: int = 1   # Starting fret position
    

@dataclass
class TabSection:
    """A section of the song (verse, chorus, etc.)"""
    name: str
    measures: List[TabMeasure]
    chord_changes: List[Tuple[float, ChordDiagram]] = field(default_factory=list)
    

@dataclass
class TabSong:
    """Complete song structure"""
    title: str
    artist: str = ""
    album: str = ""
    key: str = ""
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    tuning: str = "Standard"
    capo: int = 0
    sections: List[TabSection] = field(default_factory=list)
    notes: List[TabNote] = field(default_factory=list)  # Flat list for simple songs


# ============================================================================
# PDF DRAWING CONSTANTS
# ============================================================================

class TabStyle:
    """Configuration for tab appearance"""
    # Page settings
    page_size = A4
    margin_top = 25 * mm
    margin_bottom = 20 * mm
    margin_left = 15 * mm
    margin_right = 15 * mm
    
    # Staff settings
    staff_line_spacing = 3.5 * mm      # Space between tab lines
    staff_height = 5 * 3.5 * mm        # Total height of 6 lines
    staff_margin_top = 8 * mm          # Space above staff for chord names
    staff_margin_bottom = 6 * mm       # Space below staff for measure numbers
    
    # Note settings
    note_font_size = 9                 # Font size for fret numbers
    note_width = 4 * mm                # Minimum horizontal space per note
    beat_width = 8 * mm                # Horizontal space per beat
    measure_padding = 2 * mm           # Padding inside measure
    
    # Lines
    staff_line_width = 0.5             # Thickness of tab lines
    bar_line_width = 1.0               # Thickness of bar lines
    double_bar_width = 1.5             # Thickness of section end bars
    
    # Colors
    line_color = colors.black
    note_color = colors.black
    technique_color = colors.Color(0.3, 0.3, 0.3)
    measure_num_color = colors.Color(0.4, 0.4, 0.4)
    chord_color = colors.Color(0.1, 0.1, 0.5)
    
    # Chord diagram settings
    chord_diagram_width = 16 * mm
    chord_diagram_height = 20 * mm
    chord_fret_spacing = 3.5 * mm
    chord_string_spacing = 2.5 * mm
    
    # Standard notation (optional)
    notation_staff_height = 12 * mm    # Height for standard notation
    notation_line_spacing = 3 * mm     # Space between notation lines
    
    # Fonts
    title_font_size = 18
    subtitle_font_size = 12
    section_font_size = 11
    tempo_font_size = 9
    
    
# ============================================================================
# PDF GENERATION
# ============================================================================

class TabStaffFlowable(Flowable):
    """
    A Flowable that draws a system (row) of guitar tablature.
    
    Each system contains:
    - Optional chord diagrams above
    - Optional standard notation staff
    - 6-line tab staff with notes
    - Bar lines and measure numbers
    """
    
    def __init__(
        self,
        measures: List[TabMeasure],
        style: TabStyle,
        show_measure_numbers: bool = True,
        show_chord_diagrams: bool = True,
        show_notation: bool = False,
        chords: Optional[List[Tuple[int, ChordDiagram]]] = None,  # (measure_num, chord)
        string_names: List[str] = None,
        first_system: bool = False,
    ):
        super().__init__()
        self.measures = measures
        self.style = style
        self.show_measure_numbers = show_measure_numbers
        self.show_chord_diagrams = show_chord_diagrams
        self.show_notation = show_notation
        self.chords = chords or []
        self.string_names = string_names or ['e', 'B', 'G', 'D', 'A', 'E']
        self.first_system = first_system
        
        # Calculate dimensions
        self._calculate_dimensions()
        
    def _calculate_dimensions(self):
        """Calculate the width and height of this system"""
        s = self.style
        
        # Height components
        self.height = 0
        
        # Chord diagram space
        if self.show_chord_diagrams and self.chords:
            self.height += s.chord_diagram_height + 3 * mm
        
        # Standard notation space
        if self.show_notation:
            self.height += s.notation_staff_height + 2 * mm
            
        # Tab staff
        self.height += s.staff_height + s.staff_margin_top + s.staff_margin_bottom
        
        # Width = page width minus margins
        page_width = s.page_size[0]
        self.width = page_width - s.margin_left - s.margin_right
        
    def wrap(self, availWidth, availHeight):
        """Return the dimensions of this flowable"""
        return (self.width, self.height)
    
    def draw(self):
        """Draw the tablature system"""
        c = self.canv
        s = self.style
        
        y_offset = self.height
        
        # Draw chord diagrams if present
        if self.show_chord_diagrams and self.chords:
            y_offset -= s.chord_diagram_height + 3 * mm
            self._draw_chord_diagrams(c, y_offset + s.chord_diagram_height)
            
        # Draw standard notation if enabled
        if self.show_notation:
            y_offset -= s.notation_staff_height + 2 * mm
            self._draw_notation_staff(c, y_offset + s.notation_staff_height)
            
        # Draw tab staff
        y_offset -= s.staff_margin_top
        self._draw_tab_staff(c, y_offset)
        
    def _draw_tab_staff(self, c, y_top):
        """Draw the 6-line tab staff with notes"""
        s = self.style
        
        # Draw string names on first system
        if self.first_system:
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(s.line_color)
            for i, name in enumerate(self.string_names):
                y = y_top - i * s.staff_line_spacing
                c.drawString(-12 * mm, y - 2.5, name)
        
        # Draw the 6 horizontal lines
        c.setStrokeColor(s.line_color)
        c.setLineWidth(s.staff_line_width)
        
        for i in range(6):
            y = y_top - i * s.staff_line_spacing
            c.line(0, y, self.width, y)
            
        # Calculate measure widths
        total_beats = sum(m.time_signature[0] for m in self.measures)
        available_width = self.width - s.measure_padding * 2 * len(self.measures)
        beat_width = available_width / total_beats
        
        # Draw measures
        x = 0
        for measure in self.measures:
            measure_width = measure.time_signature[0] * beat_width + s.measure_padding * 2
            self._draw_measure(c, measure, x, y_top, measure_width, beat_width)
            x += measure_width
            
            # Draw bar line
            c.setLineWidth(s.bar_line_width)
            c.line(x, y_top, x, y_top - 5 * s.staff_line_spacing)
            
    def _draw_measure(self, c, measure: TabMeasure, x: float, y_top: float, 
                      measure_width: float, beat_width: float):
        """Draw a single measure with its notes"""
        s = self.style
        
        # Draw measure number
        if self.show_measure_numbers:
            c.setFont("Helvetica", 7)
            c.setFillColor(s.measure_num_color)
            c.drawString(x + 1 * mm, y_top - 5 * s.staff_line_spacing - 4 * mm, 
                        str(measure.number))
        
        # Get tempo marking if this is a tempo change
        if measure.tempo:
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(s.note_color)
            c.drawString(x + 1 * mm, y_top + 3 * mm, f"♩= {measure.tempo}")
        
        # Calculate beat duration
        time_sig = measure.time_signature
        beats_per_measure = time_sig[0]
        
        if not measure.notes:
            return
            
        # Find time range for this measure
        min_time = min(n.start_time for n in measure.notes) if measure.notes else 0
        max_time = max(n.start_time + n.duration for n in measure.notes) if measure.notes else 1
        time_range = max_time - min_time if max_time > min_time else 1
        
        # Draw each note
        c.setFont("Helvetica-Bold", s.note_font_size)
        c.setFillColor(s.note_color)
        
        for note in measure.notes:
            # Calculate x position based on time
            note_x = x + s.measure_padding + (note.start_time - min_time) / time_range * (measure_width - 2 * s.measure_padding)
            
            # Calculate y position based on string
            # String 0 = low E = bottom line, String 5 = high e = top line
            note_y = y_top - (5 - note.string) * s.staff_line_spacing
            
            # Draw white background for legibility
            fret_text = str(note.fret)
            text_width = c.stringWidth(fret_text, "Helvetica-Bold", s.note_font_size)
            c.setFillColor(colors.white)
            c.rect(note_x - text_width/2 - 0.5*mm, note_y - 2.5, 
                   text_width + 1*mm, 5, fill=True, stroke=False)
            
            # Draw fret number
            c.setFillColor(s.note_color)
            c.drawCentredString(note_x, note_y - 2.5, fret_text)
            
            # Draw technique symbol
            if note.technique != Technique.NONE:
                self._draw_technique(c, note, note_x, note_y)
                
    def _draw_technique(self, c, note: TabNote, x: float, y: float):
        """Draw technique symbols (h, p, b, /, etc.)"""
        s = self.style
        c.setFont("Helvetica", 7)
        c.setFillColor(s.technique_color)
        
        tech = note.technique
        
        if tech == Technique.HAMMER_ON:
            # Draw 'h' above and to the right
            c.drawString(x + 3*mm, y + 2, "h")
            
        elif tech == Technique.PULL_OFF:
            c.drawString(x + 3*mm, y + 2, "p")
            
        elif tech == Technique.BEND:
            # Draw bend arrow
            c.setLineWidth(0.5)
            c.setStrokeColor(s.technique_color)
            # Arrow pointing up
            c.line(x + 4*mm, y, x + 4*mm, y + 3*mm)
            c.line(x + 4*mm, y + 3*mm, x + 3*mm, y + 2*mm)
            c.line(x + 4*mm, y + 3*mm, x + 5*mm, y + 2*mm)
            
        elif tech == Technique.SLIDE_UP:
            c.drawString(x + 3*mm, y + 2, "/")
            
        elif tech == Technique.SLIDE_DOWN:
            c.drawString(x + 3*mm, y + 2, "\\")
            
        elif tech == Technique.VIBRATO:
            # Draw wavy line above
            c.setStrokeColor(s.technique_color)
            c.setLineWidth(0.5)
            # Simple wave
            for i in range(4):
                x1 = x - 2*mm + i * 1.5*mm
                y1 = y + 3*mm + (1 if i % 2 == 0 else -1) * 0.5*mm
                x2 = x - 2*mm + (i + 1) * 1.5*mm
                y2 = y + 3*mm + (1 if (i+1) % 2 == 0 else -1) * 0.5*mm
                c.line(x1, y1, x2, y2)
                
        elif tech == Technique.MUTE:
            c.drawString(x + 3*mm, y + 2, "x")
            
        elif tech == Technique.HARMONIC:
            # Draw diamond around the number
            c.setStrokeColor(s.technique_color)
            c.setLineWidth(0.5)
            points = [(x-2*mm, y), (x, y+2*mm), (x+2*mm, y), (x, y-2*mm)]
            c.lines([(points[i][0], points[i][1], points[(i+1)%4][0], points[(i+1)%4][1]) 
                    for i in range(4)])
    
    def _draw_chord_diagrams(self, c, y_top):
        """Draw chord diagrams above the staff"""
        s = self.style
        
        # Calculate x positions for each chord
        for measure_num, chord in self.chords:
            # Find measure position
            x = 0
            for m in self.measures:
                if m.number == measure_num:
                    break
                x += m.time_signature[0] * s.beat_width + s.measure_padding * 2
            
            self._draw_single_chord(c, chord, x + 5*mm, y_top)
            
    def _draw_single_chord(self, c, chord: ChordDiagram, x: float, y: float):
        """Draw a single chord diagram"""
        s = self.style
        
        # Chord name
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(s.chord_color)
        c.drawString(x, y, chord.name)
        
        # Chord grid
        grid_y = y - 5*mm
        grid_width = 5 * s.chord_string_spacing
        grid_height = 4 * s.chord_fret_spacing
        
        c.setStrokeColor(s.line_color)
        c.setLineWidth(0.5)
        
        # Vertical lines (strings)
        for i in range(6):
            c.line(x + i * s.chord_string_spacing, grid_y,
                   x + i * s.chord_string_spacing, grid_y - grid_height)
                   
        # Horizontal lines (frets)
        for i in range(5):
            line_width = 2 if i == 0 and chord.position == 1 else 0.5
            c.setLineWidth(line_width)
            c.line(x, grid_y - i * s.chord_fret_spacing,
                   x + grid_width, grid_y - i * s.chord_fret_spacing)
        
        # Draw finger positions
        c.setFillColor(s.note_color)
        for string, fret in enumerate(chord.frets):
            string_x = x + (5 - string) * s.chord_string_spacing
            
            if fret == -1:  # Muted
                c.setFont("Helvetica", 8)
                c.drawCentredString(string_x, grid_y + 2*mm, "x")
            elif fret == 0:  # Open
                c.setStrokeColor(s.note_color)
                c.setLineWidth(0.5)
                c.circle(string_x, grid_y + 2*mm, 1.5*mm, fill=False)
            else:  # Fretted
                fret_y = grid_y - (fret - chord.position + 0.5) * s.chord_fret_spacing
                c.circle(string_x, fret_y, 1.5*mm, fill=True)
        
        # Position number if not at first fret
        if chord.position > 1:
            c.setFont("Helvetica", 7)
            c.drawString(x - 4*mm, grid_y - s.chord_fret_spacing, str(chord.position))
            
    def _draw_notation_staff(self, c, y_top):
        """Draw standard notation staff (5 lines)"""
        s = self.style
        
        c.setStrokeColor(s.line_color)
        c.setLineWidth(s.staff_line_width)
        
        for i in range(5):
            y = y_top - i * s.notation_line_spacing
            c.line(0, y, self.width, y)
            

class TabPDFGenerator:
    """
    Main PDF generator class for guitar tablature.
    
    Usage:
        generator = TabPDFGenerator()
        generator.generate(song, "output.pdf")
    
    Or for simple note lists:
        generator.generate_from_notes(notes, "output.pdf", 
                                      title="My Song", tempo=120)
    """
    
    def __init__(self, style: TabStyle = None):
        self.style = style or TabStyle()
        
    def generate(self, song: TabSong, output_path: str, 
                 show_notation: bool = False) -> bool:
        """
        Generate a complete PDF from a TabSong structure.
        
        Args:
            song: TabSong object with all song data
            output_path: Path to save the PDF
            show_notation: Whether to include standard notation staff
            
        Returns:
            True if successful
        """
        s = self.style
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=s.page_size,
            topMargin=s.margin_top,
            bottomMargin=s.margin_bottom,
            leftMargin=s.margin_left,
            rightMargin=s.margin_right,
        )
        
        # Build content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=s.title_font_size,
            spaceAfter=3 * mm,
            alignment=1,  # Center
        )
        story.append(Paragraph(song.title, title_style))
        
        # Artist
        if song.artist:
            artist_style = ParagraphStyle(
                'Artist',
                parent=styles['Normal'],
                fontSize=s.subtitle_font_size,
                spaceAfter=2 * mm,
                alignment=1,
                textColor=colors.gray,
            )
            story.append(Paragraph(song.artist, artist_style))
            
        # Metadata line (key, tempo, tuning, capo)
        meta_parts = []
        if song.key:
            meta_parts.append(f"Key: {song.key}")
        if song.tempo:
            meta_parts.append(f"Tempo: {song.tempo} BPM")
        if song.tuning and song.tuning != "Standard":
            meta_parts.append(f"Tuning: {song.tuning}")
        if song.capo:
            meta_parts.append(f"Capo: Fret {song.capo}")
            
        if meta_parts:
            meta_style = ParagraphStyle(
                'Meta',
                parent=styles['Normal'],
                fontSize=s.tempo_font_size,
                spaceAfter=5 * mm,
                alignment=1,
                textColor=colors.Color(0.4, 0.4, 0.4),
            )
            story.append(Paragraph(" | ".join(meta_parts), meta_style))
            
        story.append(Spacer(1, 5 * mm))
        
        # Generate measures from notes if not already structured
        if song.notes and not song.sections:
            measures = self._notes_to_measures(song.notes, song.tempo, song.time_signature)
        elif song.sections:
            # Flatten sections into measures
            measures = []
            for section in song.sections:
                # Add section header
                section_style = ParagraphStyle(
                    'Section',
                    parent=styles['Heading3'],
                    fontSize=s.section_font_size,
                    spaceBefore=4 * mm,
                    spaceAfter=2 * mm,
                    textColor=s.chord_color,
                )
                story.append(Paragraph(f"[{section.name}]", section_style))
                measures.extend(section.measures)
        else:
            measures = []
            
        # Calculate how many measures fit per system
        measures_per_system = self._calculate_measures_per_system(measures)
        
        # Create systems
        systems = []
        for i in range(0, len(measures), measures_per_system):
            system_measures = measures[i:i + measures_per_system]
            
            system = TabStaffFlowable(
                measures=system_measures,
                style=self.style,
                show_measure_numbers=True,
                show_chord_diagrams=True,
                show_notation=show_notation,
                chords=[],  # TODO: Extract chords for this system
                first_system=(i == 0),
            )
            systems.append(system)
            
        # Add systems to story with spacing
        for system in systems:
            story.append(system)
            story.append(Spacer(1, 3 * mm))
            
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to generate PDF: {e}")
            return False
            
    def generate_from_notes(
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
        show_notation: bool = False,
    ) -> bool:
        """
        Generate PDF from a flat list of TabNote objects.
        
        This is the simpler interface for transcription output.
        """
        song = TabSong(
            title=title,
            artist=artist,
            key=key,
            tempo=tempo,
            time_signature=time_signature,
            tuning=tuning,
            capo=capo,
            notes=notes,
        )
        return self.generate(song, output_path, show_notation)
        
    def generate_from_json(self, json_path: str, output_path: str, **kwargs) -> bool:
        """
        Generate PDF from JSON transcription output.
        
        Expected JSON format (from guitar_tabs.py or basic-pitch):
        {
            "notes": [
                {"midi": 43, "start_time": 0.0, "duration": 0.5, ...},
                ...
            ],
            "tempo": 120,
            "key": "Am",
            ...
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convert JSON notes to TabNote objects
        notes = self._convert_json_notes(data.get('notes', []))
        
        return self.generate_from_notes(
            notes=notes,
            output_path=output_path,
            title=kwargs.get('title', data.get('title', 'Guitar Tablature')),
            artist=kwargs.get('artist', data.get('artist', '')),
            key=kwargs.get('key', data.get('key', '')),
            tempo=kwargs.get('tempo', data.get('tempo', 120)),
            **{k: v for k, v in kwargs.items() if k not in ['title', 'artist', 'key', 'tempo']}
        )
        
    def _convert_json_notes(self, json_notes: List[Dict]) -> List[TabNote]:
        """Convert JSON note format to TabNote objects"""
        # Standard tuning MIDI values: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
        tuning = [40, 45, 50, 55, 59, 64]
        
        tab_notes = []
        for note_data in json_notes:
            midi = note_data.get('midi', note_data.get('pitch', 0))
            start_time = note_data.get('start_time', note_data.get('start', 0))
            duration = note_data.get('duration', 0.25)
            
            # Find best string/fret combination
            string, fret = self._midi_to_tab(midi, tuning)
            
            if string is not None and fret is not None:
                # Detect technique from pitch bends or metadata
                technique = Technique.NONE
                if 'technique' in note_data:
                    tech_map = {
                        'h': Technique.HAMMER_ON,
                        'p': Technique.PULL_OFF,
                        'b': Technique.BEND,
                        '/': Technique.SLIDE_UP,
                        '\\': Technique.SLIDE_DOWN,
                        '~': Technique.VIBRATO,
                    }
                    technique = tech_map.get(note_data['technique'], Technique.NONE)
                elif 'pitch_bends' in note_data:
                    bends = note_data['pitch_bends']
                    if any(abs(b) > 0.5 for b in bends):
                        technique = Technique.BEND
                
                tab_notes.append(TabNote(
                    string=string,
                    fret=fret,
                    start_time=start_time,
                    duration=duration,
                    technique=technique,
                ))
                
        return tab_notes
        
    def _midi_to_tab(self, midi: int, tuning: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert MIDI note to best string/fret combination.
        
        Prefers lower frets and middle strings for playability.
        """
        candidates = []
        
        for string_num, open_note in enumerate(tuning):
            fret = midi - open_note
            if 0 <= fret <= 24:  # Valid fret range
                # Score based on playability
                # Prefer: lower frets, middle strings
                fret_penalty = fret * 0.5
                string_penalty = abs(string_num - 2.5) * 0.3  # Prefer D/G strings
                
                score = fret_penalty + string_penalty
                candidates.append((string_num, fret, score))
                
        if not candidates:
            return None, None
            
        # Return best candidate (lowest score)
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0], candidates[0][1]
        
    def _notes_to_measures(
        self, 
        notes: List[TabNote], 
        tempo: int, 
        time_signature: Tuple[int, int]
    ) -> List[TabMeasure]:
        """Convert a flat list of notes into measures"""
        if not notes:
            return []
            
        beats_per_measure = time_signature[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        
        # Find total duration
        max_time = max(n.start_time + n.duration for n in notes)
        num_measures = int(max_time / measure_duration) + 1
        
        measures = []
        for i in range(num_measures):
            measure_start = i * measure_duration
            measure_end = (i + 1) * measure_duration
            
            measure_notes = [
                n for n in notes 
                if measure_start <= n.start_time < measure_end
            ]
            
            measures.append(TabMeasure(
                number=i + 1,
                notes=measure_notes,
                time_signature=time_signature,
                tempo=tempo if i == 0 else None,
            ))
            
        return measures
        
    def _calculate_measures_per_system(self, measures: List[TabMeasure]) -> int:
        """Calculate how many measures fit on one line"""
        s = self.style
        
        # Available width
        page_width = s.page_size[0] - s.margin_left - s.margin_right
        
        # Average measure width
        if measures:
            avg_beats = sum(m.time_signature[0] for m in measures) / len(measures)
        else:
            avg_beats = 4
            
        measure_width = avg_beats * s.beat_width + s.measure_padding * 2
        
        # How many fit?
        measures_per_line = max(1, int(page_width / measure_width))
        
        # Limit to reasonable amount
        return min(measures_per_line, 6)


# ============================================================================
# STANDALONE DRAWING (for direct canvas use)
# ============================================================================

def draw_tab_page(
    c: canvas.Canvas,
    notes: List[TabNote],
    page_num: int,
    total_pages: int,
    title: str = "Guitar Tablature",
    artist: str = "",
    tempo: int = 120,
    key: str = "",
    style: TabStyle = None,
):
    """
    Draw a complete page of tablature directly on a canvas.
    
    Useful for custom page layouts or integrating with other content.
    """
    s = style or TabStyle()
    width, height = s.page_size
    
    # Header
    c.setFont("Helvetica-Bold", s.title_font_size)
    c.drawCentredString(width/2, height - s.margin_top, title)
    
    if artist:
        c.setFont("Helvetica", s.subtitle_font_size)
        c.setFillColor(colors.gray)
        c.drawCentredString(width/2, height - s.margin_top - 8*mm, artist)
        
    # Metadata
    if tempo or key:
        c.setFont("Helvetica", s.tempo_font_size)
        c.setFillColor(colors.Color(0.4, 0.4, 0.4))
        meta = []
        if key:
            meta.append(f"Key: {key}")
        if tempo:
            meta.append(f"♩= {tempo}")
        c.drawCentredString(width/2, height - s.margin_top - 14*mm, " | ".join(meta))
        
    # Footer with page number
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.gray)
    c.drawCentredString(width/2, s.margin_bottom/2, f"Page {page_num} of {total_pages}")
    

# ============================================================================
# CHORD DIAGRAM GENERATOR
# ============================================================================

COMMON_CHORDS = {
    # Major chords
    'C': ChordDiagram('C', [-1, 3, 2, 0, 1, 0], [0, 3, 2, 0, 1, 0]),
    'D': ChordDiagram('D', [-1, -1, 0, 2, 3, 2], [0, 0, 0, 1, 3, 2]),
    'E': ChordDiagram('E', [0, 2, 2, 1, 0, 0], [0, 2, 3, 1, 0, 0]),
    'F': ChordDiagram('F', [1, 3, 3, 2, 1, 1], [1, 3, 4, 2, 1, 1], barre=(1, 0, 5)),
    'G': ChordDiagram('G', [3, 2, 0, 0, 0, 3], [2, 1, 0, 0, 0, 3]),
    'A': ChordDiagram('A', [-1, 0, 2, 2, 2, 0], [0, 0, 1, 2, 3, 0]),
    'B': ChordDiagram('B', [-1, 2, 4, 4, 4, 2], [0, 1, 2, 3, 4, 1], barre=(2, 1, 5), position=2),
    
    # Minor chords
    'Am': ChordDiagram('Am', [-1, 0, 2, 2, 1, 0], [0, 0, 2, 3, 1, 0]),
    'Bm': ChordDiagram('Bm', [-1, 2, 4, 4, 3, 2], [0, 1, 3, 4, 2, 1], barre=(2, 1, 5), position=2),
    'Cm': ChordDiagram('Cm', [-1, 3, 5, 5, 4, 3], [0, 1, 3, 4, 2, 1], barre=(3, 1, 5), position=3),
    'Dm': ChordDiagram('Dm', [-1, -1, 0, 2, 3, 1], [0, 0, 0, 2, 3, 1]),
    'Em': ChordDiagram('Em', [0, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0]),
    'Fm': ChordDiagram('Fm', [1, 3, 3, 1, 1, 1], [1, 3, 4, 1, 1, 1], barre=(1, 0, 5)),
    'Gm': ChordDiagram('Gm', [3, 5, 5, 3, 3, 3], [1, 3, 4, 1, 1, 1], barre=(3, 0, 5), position=3),
    
    # Seventh chords
    'C7': ChordDiagram('C7', [-1, 3, 2, 3, 1, 0], [0, 3, 2, 4, 1, 0]),
    'D7': ChordDiagram('D7', [-1, -1, 0, 2, 1, 2], [0, 0, 0, 2, 1, 3]),
    'E7': ChordDiagram('E7', [0, 2, 0, 1, 0, 0], [0, 2, 0, 1, 0, 0]),
    'G7': ChordDiagram('G7', [3, 2, 0, 0, 0, 1], [3, 2, 0, 0, 0, 1]),
    'A7': ChordDiagram('A7', [-1, 0, 2, 0, 2, 0], [0, 0, 1, 0, 2, 0]),
    
    # Power chords
    'E5': ChordDiagram('E5', [0, 2, 2, -1, -1, -1], [0, 1, 2, 0, 0, 0]),
    'A5': ChordDiagram('A5', [-1, 0, 2, 2, -1, -1], [0, 0, 1, 2, 0, 0]),
    'G5': ChordDiagram('G5', [3, 5, 5, -1, -1, -1], [1, 2, 3, 0, 0, 0], position=3),
}


def get_chord_diagram(name: str) -> Optional[ChordDiagram]:
    """Get a chord diagram by name"""
    return COMMON_CHORDS.get(name)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for PDF generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate professional guitar tablature PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON transcription output:
  python pdf_generator.py transcription.json output.pdf --title "My Song" --artist "Me"
  
  # With standard notation:
  python pdf_generator.py tabs.json output.pdf --notation
  
  # Custom tempo and key:
  python pdf_generator.py tabs.json output.pdf --tempo 140 --key "Am"
"""
    )
    
    parser.add_argument('input', help='Input JSON file with tab data')
    parser.add_argument('output', help='Output PDF file path')
    parser.add_argument('--title', '-t', default='Guitar Tablature', 
                        help='Song title')
    parser.add_argument('--artist', '-a', default='', help='Artist name')
    parser.add_argument('--key', '-k', default='', help='Musical key')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--tuning', default='Standard', help='Guitar tuning')
    parser.add_argument('--capo', type=int, default=0, help='Capo position')
    parser.add_argument('--notation', action='store_true', 
                        help='Include standard notation staff')
    parser.add_argument('--page-size', choices=['a4', 'letter'], default='a4',
                        help='Page size')
    
    args = parser.parse_args()
    
    # Configure style
    style = TabStyle()
    if args.page_size == 'letter':
        style.page_size = letter
        
    # Generate PDF
    generator = TabPDFGenerator(style)
    
    if args.input.endswith('.json'):
        success = generator.generate_from_json(
            args.input,
            args.output,
            title=args.title,
            artist=args.artist,
            key=args.key,
            tempo=args.tempo,
            tuning=args.tuning,
            capo=args.capo,
            show_notation=args.notation,
        )
    else:
        print(f"❌ Unsupported input format: {args.input}")
        print("Supported formats: .json")
        return 1
        
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
