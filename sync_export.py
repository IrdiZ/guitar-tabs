#!/usr/bin/env python3
"""
Audio Playback Sync Export Module

Generates JSON timeline data mapping tab positions to audio timestamps
for "follow along" playback in web players.

The sync data can be used by web UIs to:
- Highlight notes as audio plays
- Show current position in tablature
- Display beat/measure markers
- Enable scrolling playback

Output Format:
{
    "version": "1.0",
    "metadata": { ... },
    "timeline": {
        "events": [ ... ],      # All events sorted by time
        "measures": [ ... ],    # Measure markers
        "beats": [ ... ]        # Beat markers
    },
    "tab_data": {
        "strings": 6,
        "tuning": [...],
        "notes": [ ... ]
    }
}
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class EventType(str, Enum):
    """Types of timeline events."""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CHORD_START = "chord_start"
    CHORD_END = "chord_end"
    BEAT = "beat"
    MEASURE = "measure"
    SECTION = "section"
    TECHNIQUE = "technique"


@dataclass
class TimelineEvent:
    """A single event in the playback timeline."""
    time: float  # Timestamp in seconds
    type: str  # EventType value
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "time": round(self.time, 4),
            "type": self.type,
            **self.data
        }


@dataclass
class SyncNote:
    """A note with full sync information."""
    id: int
    start_time: float
    end_time: float
    duration: float
    string: int  # 0-5 (E A D G B e)
    fret: int
    midi: int
    note_name: str
    frequency: float
    confidence: float
    # Technique info (optional)
    technique: Optional[str] = None  # hammer-on, pull-off, slide, bend, vibrato
    technique_target: Optional[int] = None  # Target fret for slides/bends
    
    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "start": round(self.start_time, 4),
            "end": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "string": self.string,
            "fret": self.fret,
            "midi": self.midi,
            "note": self.note_name,
            "freq": round(self.frequency, 2),
            "confidence": round(self.confidence, 3)
        }
        if self.technique:
            d["technique"] = self.technique
        if self.technique_target is not None:
            d["technique_target"] = self.technique_target
        return d


@dataclass
class SyncChord:
    """A chord with sync information."""
    id: int
    start_time: float
    end_time: float
    name: str
    root: str
    quality: str
    note_ids: List[int]  # References to SyncNote ids
    is_barre: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "start": round(self.start_time, 4),
            "end": round(self.end_time, 4),
            "name": self.name,
            "root": self.root,
            "quality": self.quality,
            "note_ids": self.note_ids,
            "is_barre": self.is_barre
        }


@dataclass
class Beat:
    """A beat marker."""
    time: float
    beat_number: int  # Beat within measure (1-indexed)
    measure_number: int
    is_downbeat: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "time": round(self.time, 4),
            "beat": self.beat_number,
            "measure": self.measure_number,
            "downbeat": self.is_downbeat
        }


@dataclass 
class Measure:
    """A measure/bar marker."""
    start_time: float
    end_time: float
    number: int
    beat_count: int = 4
    
    def to_dict(self) -> Dict:
        return {
            "start": round(self.start_time, 4),
            "end": round(self.end_time, 4),
            "number": self.number,
            "beats": self.beat_count
        }


@dataclass
class SyncData:
    """Complete sync data for export."""
    # Metadata
    title: str
    artist: str
    duration: float  # Total audio duration
    tempo: int
    time_signature: Tuple[int, int] = (4, 4)
    key: Optional[str] = None
    tuning: List[int] = field(default_factory=lambda: [40, 45, 50, 55, 59, 64])
    tuning_name: str = "standard"
    
    # Content
    notes: List[SyncNote] = field(default_factory=list)
    chords: List[SyncChord] = field(default_factory=list)
    beats: List[Beat] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    
    # Computed statistics
    total_notes: int = 0
    total_chords: int = 0
    avg_notes_per_second: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "version": "1.0",
            "metadata": {
                "title": self.title,
                "artist": self.artist,
                "duration": round(self.duration, 3),
                "tempo": self.tempo,
                "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
                "key": self.key,
                "tuning": {
                    "name": self.tuning_name,
                    "midi": self.tuning,
                    "strings": 6
                },
                "statistics": {
                    "total_notes": self.total_notes,
                    "total_chords": self.total_chords,
                    "avg_notes_per_second": round(self.avg_notes_per_second, 2)
                }
            },
            "timeline": {
                "events": [e.to_dict() for e in sorted(self.events, key=lambda x: x.time)],
                "measures": [m.to_dict() for m in self.measures],
                "beats": [b.to_dict() for b in self.beats]
            },
            "tab_data": {
                "notes": [n.to_dict() for n in self.notes],
                "chords": [c.to_dict() for c in self.chords]
            },
            # Quick lookup arrays for efficient web player access
            "playback": {
                "note_starts": [round(n.start_time, 4) for n in self.notes],
                "note_ends": [round(n.end_time, 4) for n in self.notes],
                "note_strings": [n.string for n in self.notes],
                "note_frets": [n.fret for n in self.notes],
                "beat_times": [round(b.time, 4) for b in self.beats],
                "measure_times": [round(m.start_time, 4) for m in self.measures]
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str) -> None:
        """Save sync data to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# Note name mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name with octave."""
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def midi_to_frequency(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def detect_technique_between_notes(
    prev_note: 'TabNote',
    curr_note: 'TabNote',
    prev_midi: int,
    curr_midi: int
) -> Tuple[Optional[str], Optional[int]]:
    """
    Detect playing technique between two consecutive notes.
    
    Returns:
        (technique_name, target_value) or (None, None)
    """
    if prev_note is None:
        return None, None
    
    time_gap = curr_note.start_time - (prev_note.start_time + prev_note.duration)
    pitch_diff = curr_midi - prev_midi
    same_string = curr_note.string == prev_note.string
    
    # Legato techniques (same string, quick succession)
    if same_string and time_gap < 0.08:
        if pitch_diff > 0 and pitch_diff <= 4:
            return "hammer-on", curr_note.fret
        elif pitch_diff < 0 and pitch_diff >= -4:
            return "pull-off", curr_note.fret
    
    # Slide (same string, larger pitch change)
    if same_string and abs(pitch_diff) >= 2 and time_gap < 0.05:
        return "slide", curr_note.fret
    
    return None, None


def generate_beat_markers(
    duration: float,
    tempo: int,
    time_signature: Tuple[int, int] = (4, 4)
) -> Tuple[List[Beat], List[Measure]]:
    """
    Generate beat and measure markers for the entire audio duration.
    
    Args:
        duration: Total audio duration in seconds
        tempo: Tempo in BPM
        time_signature: (beats_per_measure, beat_unit)
        
    Returns:
        (beats, measures) lists
    """
    beat_duration = 60.0 / tempo
    beats_per_measure = time_signature[0]
    
    beats = []
    measures = []
    
    current_time = 0.0
    beat_count = 0
    measure_num = 1
    measure_start = 0.0
    
    while current_time < duration:
        beat_in_measure = (beat_count % beats_per_measure) + 1
        is_downbeat = beat_in_measure == 1
        
        beats.append(Beat(
            time=current_time,
            beat_number=beat_in_measure,
            measure_number=measure_num,
            is_downbeat=is_downbeat
        ))
        
        # If this was the last beat in the measure
        if beat_in_measure == beats_per_measure:
            measure_end = min(current_time + beat_duration, duration)
            measures.append(Measure(
                start_time=measure_start,
                end_time=measure_end,
                number=measure_num,
                beat_count=beats_per_measure
            ))
            measure_num += 1
            measure_start = measure_end
        
        beat_count += 1
        current_time += beat_duration
    
    # Handle partial final measure
    if measure_start < duration:
        partial_beats = beat_count % beats_per_measure
        if partial_beats == 0:
            partial_beats = beats_per_measure
        measures.append(Measure(
            start_time=measure_start,
            end_time=duration,
            number=measure_num,
            beat_count=partial_beats
        ))
    
    return beats, measures


def generate_timeline_events(
    notes: List[SyncNote],
    chords: List[SyncChord],
    beats: List[Beat],
    measures: List[Measure]
) -> List[TimelineEvent]:
    """
    Generate a unified timeline of all events.
    
    This creates note_on/note_off events that a web player can use
    to trigger highlight changes.
    """
    events = []
    
    # Note events
    for note in notes:
        # Note on
        events.append(TimelineEvent(
            time=note.start_time,
            type=EventType.NOTE_ON.value,
            data={
                "note_id": note.id,
                "string": note.string,
                "fret": note.fret,
                "midi": note.midi,
                "note": note.note_name
            }
        ))
        # Note off
        events.append(TimelineEvent(
            time=note.end_time,
            type=EventType.NOTE_OFF.value,
            data={"note_id": note.id}
        ))
        # Technique event
        if note.technique:
            events.append(TimelineEvent(
                time=note.start_time,
                type=EventType.TECHNIQUE.value,
                data={
                    "note_id": note.id,
                    "technique": note.technique,
                    "target": note.technique_target
                }
            ))
    
    # Chord events
    for chord in chords:
        events.append(TimelineEvent(
            time=chord.start_time,
            type=EventType.CHORD_START.value,
            data={
                "chord_id": chord.id,
                "name": chord.name,
                "note_ids": chord.note_ids
            }
        ))
        events.append(TimelineEvent(
            time=chord.end_time,
            type=EventType.CHORD_END.value,
            data={"chord_id": chord.id}
        ))
    
    # Beat events
    for beat in beats:
        events.append(TimelineEvent(
            time=beat.time,
            type=EventType.BEAT.value,
            data={
                "beat": beat.beat_number,
                "measure": beat.measure_number,
                "downbeat": beat.is_downbeat
            }
        ))
    
    # Measure events
    for measure in measures:
        events.append(TimelineEvent(
            time=measure.start_time,
            type=EventType.MEASURE.value,
            data={
                "number": measure.number,
                "beats": measure.beat_count
            }
        ))
    
    # Sort by time
    events.sort(key=lambda e: (e.time, 0 if e.type == EventType.MEASURE.value else 1))
    
    return events


def generate_sync_data(
    tab_notes: List['TabNote'],
    notes: List['Note'],
    audio_duration: float,
    tempo: int = 120,
    title: str = "Untitled",
    artist: str = "Unknown",
    tuning: List[int] = None,
    tuning_name: str = "standard",
    chords: List['Chord'] = None,
    key: str = None,
    time_signature: Tuple[int, int] = (4, 4)
) -> SyncData:
    """
    Generate complete sync data from tab notes and original notes.
    
    Args:
        tab_notes: List of TabNote objects (string/fret positions)
        notes: List of Note objects (MIDI/timing info)
        audio_duration: Total audio duration in seconds
        tempo: Tempo in BPM
        title: Song title
        artist: Artist name
        tuning: Guitar tuning as MIDI note numbers
        tuning_name: Name of the tuning (e.g., "standard", "drop_d")
        chords: Optional list of detected chords
        key: Musical key (e.g., "Am", "C major")
        time_signature: Time signature as (beats, beat_unit)
        
    Returns:
        SyncData object ready for export
    """
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
    
    # Build sync notes with full information
    sync_notes = []
    prev_tab = None
    prev_note = None
    
    for i, (tab_note, note) in enumerate(zip(tab_notes, notes)):
        # Calculate MIDI from string/fret position
        midi = tuning[tab_note.string] + tab_note.fret
        
        # Detect technique
        technique, technique_target = None, None
        if prev_tab and prev_note:
            prev_midi = tuning[prev_tab.string] + prev_tab.fret
            technique, technique_target = detect_technique_between_notes(
                prev_tab, tab_note, prev_midi, midi
            )
        
        sync_note = SyncNote(
            id=i,
            start_time=tab_note.start_time,
            end_time=tab_note.start_time + tab_note.duration,
            duration=tab_note.duration,
            string=tab_note.string,
            fret=tab_note.fret,
            midi=midi,
            note_name=midi_to_note_name(midi),
            frequency=midi_to_frequency(midi),
            confidence=note.confidence if hasattr(note, 'confidence') else 1.0,
            technique=technique,
            technique_target=technique_target
        )
        sync_notes.append(sync_note)
        
        prev_tab = tab_note
        prev_note = note
    
    # Build sync chords
    sync_chords = []
    if chords:
        for i, chord in enumerate(chords):
            # Find note IDs that belong to this chord
            chord_note_ids = []
            for j, sync_note in enumerate(sync_notes):
                # Check if note starts within the chord's time window
                if (abs(sync_note.start_time - chord.start_time) < 0.05 and
                    sync_note.start_time < chord.start_time + chord.duration):
                    chord_note_ids.append(j)
            
            sync_chord = SyncChord(
                id=i,
                start_time=chord.start_time,
                end_time=chord.start_time + chord.duration,
                name=chord.name,
                root=chord.root_name if hasattr(chord, 'root_name') else NOTE_NAMES[chord.root],
                quality=chord.quality if hasattr(chord, 'quality') else "",
                note_ids=chord_note_ids,
                is_barre=chord.is_barre if hasattr(chord, 'is_barre') else False
            )
            sync_chords.append(sync_chord)
    
    # Generate beat and measure markers
    beats, measures = generate_beat_markers(audio_duration, tempo, time_signature)
    
    # Generate timeline events
    events = generate_timeline_events(sync_notes, sync_chords, beats, measures)
    
    # Calculate statistics
    total_notes = len(sync_notes)
    total_chords = len(sync_chords)
    avg_notes_per_second = total_notes / audio_duration if audio_duration > 0 else 0
    
    # Build final sync data
    sync_data = SyncData(
        title=title,
        artist=artist,
        duration=audio_duration,
        tempo=tempo,
        time_signature=time_signature,
        key=key,
        tuning=tuning,
        tuning_name=tuning_name,
        notes=sync_notes,
        chords=sync_chords,
        beats=beats,
        measures=measures,
        events=events,
        total_notes=total_notes,
        total_chords=total_chords,
        avg_notes_per_second=avg_notes_per_second
    )
    
    return sync_data


def export_sync_json(
    tab_notes: List['TabNote'],
    notes: List['Note'],
    output_path: str,
    audio_path: str = None,
    tempo: int = 120,
    title: str = None,
    artist: str = "Guitar Tab Generator",
    tuning: List[int] = None,
    tuning_name: str = "standard",
    chords: List['Chord'] = None,
    key: str = None,
    time_signature: Tuple[int, int] = (4, 4),
    verbose: bool = True
) -> bool:
    """
    Export sync data to JSON file.
    
    This is the main entry point for generating playback sync data.
    
    Args:
        tab_notes: List of TabNote objects
        notes: List of Note objects
        output_path: Path to save JSON file
        audio_path: Path to source audio (for duration calculation)
        tempo: Tempo in BPM
        title: Song title (auto-detected from audio_path if None)
        artist: Artist name
        tuning: Guitar tuning
        tuning_name: Tuning name
        chords: Detected chords
        key: Musical key
        time_signature: Time signature
        verbose: Print status messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import librosa
    except ImportError:
        print("❌ librosa required for sync export")
        return False
    
    # Auto-detect title from filename
    if title is None and audio_path:
        title = os.path.splitext(os.path.basename(audio_path))[0]
    elif title is None:
        title = "Untitled"
    
    # Calculate audio duration
    if audio_path and os.path.exists(audio_path):
        audio_duration = librosa.get_duration(path=audio_path)
    elif tab_notes:
        # Estimate from last note
        audio_duration = max(n.start_time + n.duration for n in tab_notes)
    else:
        audio_duration = 0
    
    if verbose:
        print(f"\n🔄 Generating playback sync data...")
        print(f"   Audio duration: {audio_duration:.2f}s")
        print(f"   Notes: {len(tab_notes)}")
        print(f"   Tempo: {tempo} BPM")
    
    # Generate sync data
    sync_data = generate_sync_data(
        tab_notes=tab_notes,
        notes=notes,
        audio_duration=audio_duration,
        tempo=tempo,
        title=title,
        artist=artist,
        tuning=tuning,
        tuning_name=tuning_name,
        chords=chords,
        key=key,
        time_signature=time_signature
    )
    
    # Save to file
    sync_data.save(output_path)
    
    if verbose:
        print(f"   Timeline events: {len(sync_data.events)}")
        print(f"   Measures: {len(sync_data.measures)}")
        print(f"   Beats: {len(sync_data.beats)}")
        print(f"✅ Saved sync data: {output_path}")
    
    return True


# Example web player usage documentation
WEB_PLAYER_EXAMPLE = """
// Example JavaScript for using sync data in a web player

class GuitarTabPlayer {
    constructor(syncData, audioElement) {
        this.syncData = syncData;
        this.audio = audioElement;
        this.activeNotes = new Set();
        this.currentMeasure = 0;
        this.eventIndex = 0;
    }
    
    start() {
        this.audio.play();
        this.animate();
    }
    
    animate() {
        const currentTime = this.audio.currentTime;
        
        // Process events up to current time
        while (this.eventIndex < this.syncData.timeline.events.length) {
            const event = this.syncData.timeline.events[this.eventIndex];
            if (event.time > currentTime) break;
            
            this.processEvent(event);
            this.eventIndex++;
        }
        
        if (!this.audio.paused) {
            requestAnimationFrame(() => this.animate());
        }
    }
    
    processEvent(event) {
        switch (event.type) {
            case 'note_on':
                this.highlightNote(event.note_id, event.string, event.fret);
                this.activeNotes.add(event.note_id);
                break;
            case 'note_off':
                this.unhighlightNote(event.note_id);
                this.activeNotes.delete(event.note_id);
                break;
            case 'beat':
                this.showBeat(event.measure, event.beat, event.downbeat);
                break;
            case 'measure':
                this.scrollToMeasure(event.number);
                break;
            case 'chord_start':
                this.showChord(event.name);
                break;
        }
    }
    
    // Seek to specific time
    seek(time) {
        this.audio.currentTime = time;
        this.eventIndex = this.syncData.timeline.events.findIndex(e => e.time >= time);
        this.activeNotes.clear();
        this.clearHighlights();
    }
    
    // Get active notes at a specific time (for scrubbing)
    getActiveNotesAtTime(time) {
        const notes = this.syncData.tab_data.notes;
        return notes.filter(n => n.start <= time && n.end > time);
    }
}
"""


if __name__ == "__main__":
    # Quick test/demo
    print("Sync Export Module")
    print("==================")
    print("\nThis module generates JSON timeline data for web player integration.")
    print("\nUsage in guitar_tabs.py:")
    print("  python guitar_tabs.py audio.mp3 --export-sync output.sync.json")
    print("\nThe sync data includes:")
    print("  - Note on/off events with timing")
    print("  - Tab positions (string, fret)")
    print("  - Beat and measure markers")
    print("  - Chord events (if detected)")
    print("  - Technique markers (hammer-on, pull-off, slide)")
