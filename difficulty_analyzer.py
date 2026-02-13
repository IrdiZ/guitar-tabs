#!/usr/bin/env python3
"""
Guitar Tab Difficulty Analyzer

Analyzes transcribed guitar tabs and provides:
- Overall difficulty rating (beginner/intermediate/advanced/expert)
- Technique requirements (bends, hammer-ons, slides, etc.)
- Maximum stretch required (fret span)
- Speed requirements (notes per second)
- Highlight difficult passages
- Suggest practice sections

Difficulty levels:
  🟢 Beginner    - Simple melodies, open chords, slow tempo
  🟡 Intermediate - Barre chords, basic techniques, moderate speed
  🟠 Advanced    - Fast passages, complex techniques, wide stretches
  🔴 Expert      - Extreme speed, advanced techniques, challenging patterns
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from collections import Counter


class DifficultyLevel(Enum):
    """Overall difficulty rating."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    
    @property
    def emoji(self) -> str:
        return {
            DifficultyLevel.BEGINNER: "🟢",
            DifficultyLevel.INTERMEDIATE: "🟡",
            DifficultyLevel.ADVANCED: "🟠",
            DifficultyLevel.EXPERT: "🔴",
        }[self]
    
    @property
    def score(self) -> int:
        """Numeric score for averaging."""
        return {
            DifficultyLevel.BEGINNER: 1,
            DifficultyLevel.INTERMEDIATE: 2,
            DifficultyLevel.ADVANCED: 3,
            DifficultyLevel.EXPERT: 4,
        }[self]


@dataclass
class TechniqueStats:
    """Statistics about techniques used in the piece."""
    hammer_ons: int = 0
    pull_offs: int = 0
    bends: int = 0
    slides: int = 0
    vibrato: int = 0
    trills: int = 0
    tremolo: int = 0
    harmonics: int = 0
    palm_mutes: int = 0
    
    @property
    def total_techniques(self) -> int:
        return (self.hammer_ons + self.pull_offs + self.bends + 
                self.slides + self.vibrato + self.trills + 
                self.tremolo + self.harmonics + self.palm_mutes)
    
    def technique_density(self, total_notes: int) -> float:
        """Percentage of notes with techniques."""
        if total_notes == 0:
            return 0.0
        return (self.total_techniques / total_notes) * 100
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'hammer_ons': self.hammer_ons,
            'pull_offs': self.pull_offs,
            'bends': self.bends,
            'slides': self.slides,
            'vibrato': self.vibrato,
            'trills': self.trills,
            'tremolo': self.tremolo,
            'harmonics': self.harmonics,
            'palm_mutes': self.palm_mutes,
        }


@dataclass
class DifficultPassage:
    """A passage identified as challenging."""
    start_time: float
    end_time: float
    reason: str
    difficulty_score: float  # 0-1
    notes_involved: int
    suggested_practice_tempo: Optional[int] = None  # % of original tempo
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PracticeSection:
    """A suggested practice section."""
    start_time: float
    end_time: float
    focus: str  # What to practice (technique, speed, stretches)
    difficulty: DifficultyLevel
    tips: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DifficultyReport:
    """Complete difficulty analysis report."""
    overall_rating: DifficultyLevel
    overall_score: float  # 0-100
    
    # Detailed metrics
    technique_stats: TechniqueStats
    max_fret_span: int
    max_fret_used: int
    avg_fret_span: float
    
    # Speed metrics
    max_notes_per_second: float
    avg_notes_per_second: float
    fastest_passage_nps: float
    
    # Complexity metrics
    total_notes: int
    unique_positions: int
    string_changes: int
    position_changes: int
    
    # Passages and practice
    difficult_passages: List[DifficultPassage]
    practice_sections: List[PracticeSection]
    
    # Sub-ratings (0-100)
    speed_rating: float
    technique_rating: float
    stretch_rating: float
    position_rating: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            'overall_rating': self.overall_rating.value,
            'overall_score': round(self.overall_score, 1),
            'metrics': {
                'total_notes': self.total_notes,
                'max_fret_span': self.max_fret_span,
                'max_fret_used': self.max_fret_used,
                'avg_fret_span': round(self.avg_fret_span, 2),
                'max_notes_per_second': round(self.max_notes_per_second, 1),
                'avg_notes_per_second': round(self.avg_notes_per_second, 1),
                'unique_positions': self.unique_positions,
                'string_changes': self.string_changes,
                'position_changes': self.position_changes,
            },
            'techniques': self.technique_stats.to_dict(),
            'sub_ratings': {
                'speed': round(self.speed_rating, 1),
                'technique': round(self.technique_rating, 1),
                'stretch': round(self.stretch_rating, 1),
                'position': round(self.position_rating, 1),
            },
            'difficult_passages': [
                {
                    'start_time': round(p.start_time, 2),
                    'end_time': round(p.end_time, 2),
                    'reason': p.reason,
                    'difficulty_score': round(p.difficulty_score, 2),
                }
                for p in self.difficult_passages
            ],
            'practice_sections': [
                {
                    'start_time': round(s.start_time, 2),
                    'end_time': round(s.end_time, 2),
                    'focus': s.focus,
                    'difficulty': s.difficulty.value,
                    'tips': s.tips,
                }
                for s in self.practice_sections
            ],
        }
    
    def format_report(self) -> str:
        """Format as human-readable report."""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("🎸 GUITAR TAB DIFFICULTY ANALYSIS")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall rating with visual bar
        bar_filled = int(self.overall_score / 5)  # 0-20 scale
        bar_empty = 20 - bar_filled
        progress_bar = "█" * bar_filled + "░" * bar_empty
        
        lines.append(f"📊 OVERALL DIFFICULTY: {self.overall_rating.emoji} {self.overall_rating.value.upper()}")
        lines.append(f"   Score: {self.overall_score:.0f}/100 [{progress_bar}]")
        lines.append("")
        
        # Sub-ratings
        lines.append("📈 BREAKDOWN:")
        lines.append(f"   Speed:      {self._rating_bar(self.speed_rating)} ({self.speed_rating:.0f}/100)")
        lines.append(f"   Technique:  {self._rating_bar(self.technique_rating)} ({self.technique_rating:.0f}/100)")
        lines.append(f"   Stretches:  {self._rating_bar(self.stretch_rating)} ({self.stretch_rating:.0f}/100)")
        lines.append(f"   Positions:  {self._rating_bar(self.position_rating)} ({self.position_rating:.0f}/100)")
        lines.append("")
        
        # Basic stats
        lines.append("📋 STATISTICS:")
        lines.append(f"   Total notes: {self.total_notes}")
        lines.append(f"   Max fret used: {self.max_fret_used}")
        lines.append(f"   Max fret span: {self.max_fret_span} frets")
        lines.append(f"   Avg fret span: {self.avg_fret_span:.1f} frets")
        lines.append(f"   Unique positions: {self.unique_positions}")
        lines.append(f"   Position changes: {self.position_changes}")
        lines.append(f"   String changes: {self.string_changes}")
        lines.append("")
        
        # Speed
        lines.append("⚡ SPEED REQUIREMENTS:")
        lines.append(f"   Average: {self.avg_notes_per_second:.1f} notes/second")
        lines.append(f"   Maximum: {self.max_notes_per_second:.1f} notes/second")
        lines.append(f"   Fastest passage: {self.fastest_passage_nps:.1f} notes/second")
        lines.append("")
        
        # Techniques
        if self.technique_stats.total_techniques > 0:
            lines.append("🎯 TECHNIQUE REQUIREMENTS:")
            techniques = self.technique_stats.to_dict()
            for tech, count in sorted(techniques.items(), key=lambda x: -x[1]):
                if count > 0:
                    tech_name = tech.replace('_', '-').title().replace('-', ' ')
                    lines.append(f"   {tech_name}: {count}")
            density = self.technique_stats.technique_density(self.total_notes)
            lines.append(f"   Technique density: {density:.1f}% of notes")
            lines.append("")
        
        # Difficult passages
        if self.difficult_passages:
            lines.append("⚠️  DIFFICULT PASSAGES:")
            for i, passage in enumerate(self.difficult_passages[:5], 1):
                time_str = f"{passage.start_time:.1f}s - {passage.end_time:.1f}s"
                lines.append(f"   {i}. [{time_str}] {passage.reason}")
                if passage.suggested_practice_tempo:
                    lines.append(f"      → Practice at {passage.suggested_practice_tempo}% tempo")
            if len(self.difficult_passages) > 5:
                lines.append(f"   ... and {len(self.difficult_passages) - 5} more")
            lines.append("")
        
        # Practice sections
        if self.practice_sections:
            lines.append("📚 SUGGESTED PRACTICE SECTIONS:")
            for i, section in enumerate(self.practice_sections[:5], 1):
                time_str = f"{section.start_time:.1f}s - {section.end_time:.1f}s"
                lines.append(f"   {i}. [{time_str}] Focus: {section.focus}")
                for tip in section.tips[:2]:
                    lines.append(f"      • {tip}")
            lines.append("")
        
        # Summary tips
        lines.append("💡 RECOMMENDATIONS:")
        lines.extend(self._generate_recommendations())
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _rating_bar(self, score: float, width: int = 10) -> str:
        """Create a visual rating bar."""
        filled = int(score / 10)
        empty = width - filled
        return "▓" * filled + "░" * empty
    
    def _generate_recommendations(self) -> List[str]:
        """Generate practice recommendations based on analysis."""
        recs = []
        
        # Speed recommendations
        if self.speed_rating >= 70:
            recs.append("   • Use a metronome - start at 50% tempo, gradually increase")
        elif self.speed_rating >= 40:
            recs.append("   • Practice transitions between positions slowly")
        
        # Technique recommendations
        if self.technique_stats.bends > 3:
            recs.append("   • Focus on bend accuracy - use a tuner to verify pitch")
        if self.technique_stats.hammer_ons + self.technique_stats.pull_offs > 5:
            recs.append("   • Practice legato sequences for smooth hammer-ons/pull-offs")
        
        # Stretch recommendations
        if self.stretch_rating >= 60:
            recs.append("   • Work on finger independence and stretching exercises")
            recs.append("   • Consider position shifts to reduce stretches")
        
        # Position recommendations
        if self.position_rating >= 50:
            recs.append("   • Practice position shifts - aim for smooth transitions")
        
        if not recs:
            if self.overall_rating == DifficultyLevel.BEGINNER:
                recs.append("   • Great for beginners! Focus on clean, even timing")
            else:
                recs.append("   • Break into sections and master each before combining")
        
        return recs


class DifficultyAnalyzer:
    """
    Analyzes guitar tab difficulty across multiple dimensions.
    """
    
    def __init__(
        self,
        # Speed thresholds (notes per second)
        beginner_nps: float = 3.0,
        intermediate_nps: float = 6.0,
        advanced_nps: float = 10.0,
        # Stretch thresholds (fret span)
        beginner_span: int = 3,
        intermediate_span: int = 4,
        advanced_span: int = 5,
        # Position change thresholds
        beginner_positions: int = 3,
        intermediate_positions: int = 5,
        advanced_positions: int = 8,
        # Window for speed calculation
        speed_window: float = 1.0,  # seconds
        # Passage detection window
        passage_window: float = 2.0,  # seconds
    ):
        self.beginner_nps = beginner_nps
        self.intermediate_nps = intermediate_nps
        self.advanced_nps = advanced_nps
        
        self.beginner_span = beginner_span
        self.intermediate_span = intermediate_span
        self.advanced_span = advanced_span
        
        self.beginner_positions = beginner_positions
        self.intermediate_positions = intermediate_positions
        self.advanced_positions = advanced_positions
        
        self.speed_window = speed_window
        self.passage_window = passage_window
    
    def analyze(
        self,
        tab_notes: List[Any],  # TabNote objects with string, fret, start_time, duration
        notes: Optional[List[Any]] = None,  # Original Note objects
        annotated_notes: Optional[List[Any]] = None,  # AnnotatedNote with techniques
        tempo: int = 120,
        verbose: bool = True
    ) -> DifficultyReport:
        """
        Perform complete difficulty analysis.
        
        Args:
            tab_notes: List of TabNote objects
            notes: Optional list of original Note objects
            annotated_notes: Optional list of AnnotatedNote with technique info
            tempo: Song tempo in BPM
            verbose: Print progress
            
        Returns:
            DifficultyReport with complete analysis
        """
        if verbose:
            print("\n🔍 Analyzing difficulty...")
        
        if not tab_notes:
            # Empty tab - return minimal report
            return DifficultyReport(
                overall_rating=DifficultyLevel.BEGINNER,
                overall_score=0,
                technique_stats=TechniqueStats(),
                max_fret_span=0,
                max_fret_used=0,
                avg_fret_span=0,
                max_notes_per_second=0,
                avg_notes_per_second=0,
                fastest_passage_nps=0,
                total_notes=0,
                unique_positions=0,
                string_changes=0,
                position_changes=0,
                difficult_passages=[],
                practice_sections=[],
                speed_rating=0,
                technique_rating=0,
                stretch_rating=0,
                position_rating=0,
            )
        
        # Sort by time
        sorted_tabs = sorted(tab_notes, key=lambda n: n.start_time)
        
        # Calculate basic metrics
        total_notes = len(sorted_tabs)
        if verbose:
            print(f"   Analyzing {total_notes} notes...")
        
        # Fret analysis
        frets = [n.fret for n in sorted_tabs]
        max_fret = max(frets)
        
        # Calculate spans (within small time windows - representing hand positions)
        spans = self._calculate_fret_spans(sorted_tabs)
        max_span = max(spans) if spans else 0
        avg_span = np.mean(spans) if spans else 0
        
        # Position analysis
        positions = self._identify_positions(sorted_tabs)
        unique_positions = len(set(positions))
        position_changes = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
        
        # String changes
        strings = [n.string for n in sorted_tabs]
        string_changes = sum(1 for i in range(1, len(strings)) if strings[i] != strings[i-1])
        
        # Speed analysis
        speed_metrics = self._analyze_speed(sorted_tabs)
        
        # Technique analysis
        technique_stats = self._analyze_techniques(annotated_notes or sorted_tabs)
        
        # Difficult passages
        difficult_passages = self._find_difficult_passages(
            sorted_tabs, speed_metrics, spans, technique_stats
        )
        
        # Practice sections
        practice_sections = self._suggest_practice_sections(
            sorted_tabs, difficult_passages, technique_stats
        )
        
        # Calculate sub-ratings (0-100)
        speed_rating = self._calculate_speed_rating(speed_metrics)
        technique_rating = self._calculate_technique_rating(technique_stats, total_notes)
        stretch_rating = self._calculate_stretch_rating(max_span, avg_span)
        position_rating = self._calculate_position_rating(position_changes, unique_positions, total_notes)
        
        # Overall score (weighted average)
        overall_score = (
            speed_rating * 0.35 +
            technique_rating * 0.25 +
            stretch_rating * 0.20 +
            position_rating * 0.20
        )
        
        # Convert to rating
        overall_rating = self._score_to_rating(overall_score)
        
        if verbose:
            print(f"   Speed rating: {speed_rating:.0f}/100")
            print(f"   Technique rating: {technique_rating:.0f}/100")
            print(f"   Stretch rating: {stretch_rating:.0f}/100")
            print(f"   Position rating: {position_rating:.0f}/100")
            print(f"   Overall: {overall_score:.0f}/100 ({overall_rating.value})")
        
        return DifficultyReport(
            overall_rating=overall_rating,
            overall_score=overall_score,
            technique_stats=technique_stats,
            max_fret_span=max_span,
            max_fret_used=max_fret,
            avg_fret_span=avg_span,
            max_notes_per_second=speed_metrics['max_nps'],
            avg_notes_per_second=speed_metrics['avg_nps'],
            fastest_passage_nps=speed_metrics['fastest_passage_nps'],
            total_notes=total_notes,
            unique_positions=unique_positions,
            string_changes=string_changes,
            position_changes=position_changes,
            difficult_passages=difficult_passages,
            practice_sections=practice_sections,
            speed_rating=speed_rating,
            technique_rating=technique_rating,
            stretch_rating=stretch_rating,
            position_rating=position_rating,
        )
    
    def _calculate_fret_spans(self, tab_notes: List[Any], window: float = 0.5) -> List[int]:
        """Calculate fret spans in overlapping windows."""
        if not tab_notes:
            return []
        
        spans = []
        
        for i, note in enumerate(tab_notes):
            # Find notes within window
            window_notes = [note]
            for j in range(i - 1, -1, -1):
                if note.start_time - tab_notes[j].start_time <= window:
                    window_notes.append(tab_notes[j])
                else:
                    break
            
            for j in range(i + 1, len(tab_notes)):
                if tab_notes[j].start_time - note.start_time <= window:
                    window_notes.append(tab_notes[j])
                else:
                    break
            
            # Calculate span (excluding open strings)
            fretted = [n.fret for n in window_notes if n.fret > 0]
            if len(fretted) >= 2:
                span = max(fretted) - min(fretted)
                spans.append(span)
            elif fretted:
                spans.append(0)
            else:
                spans.append(0)
        
        return spans
    
    def _identify_positions(self, tab_notes: List[Any]) -> List[int]:
        """Identify hand positions for each note."""
        positions = []
        
        for note in tab_notes:
            if note.fret == 0:
                pos = 0  # Open position
            elif note.fret <= 4:
                pos = 1  # First position
            elif note.fret <= 7:
                pos = 5  # Fifth position
            elif note.fret <= 11:
                pos = 9  # Ninth position
            else:
                pos = 12  # Twelfth position or higher
            positions.append(pos)
        
        return positions
    
    def _analyze_speed(self, tab_notes: List[Any]) -> Dict[str, float]:
        """Analyze playing speed requirements."""
        if not tab_notes or len(tab_notes) < 2:
            return {
                'max_nps': 0,
                'avg_nps': 0,
                'fastest_passage_nps': 0,
                'speed_by_section': [],
            }
        
        # Calculate intervals between notes
        times = [n.start_time for n in tab_notes]
        intervals = np.diff(times)
        
        # Filter out unreasonably small intervals (< 20ms = 50 nps max)
        intervals = intervals[intervals >= 0.02]
        
        if len(intervals) == 0:
            return {
                'max_nps': 0,
                'avg_nps': 0,
                'fastest_passage_nps': 0,
                'speed_by_section': [],
            }
        
        # Notes per second
        nps = 1.0 / intervals
        
        # Windowed NPS (sliding window)
        window_nps = []
        total_duration = tab_notes[-1].start_time - tab_notes[0].start_time
        
        for window_start in np.arange(0, total_duration, self.speed_window / 2):
            window_end = window_start + self.speed_window
            notes_in_window = sum(
                1 for n in tab_notes 
                if window_start <= n.start_time < window_end
            )
            actual_duration = min(window_end, total_duration) - window_start
            if actual_duration > 0:
                window_nps.append(notes_in_window / actual_duration)
        
        return {
            'max_nps': float(np.max(nps)) if len(nps) > 0 else 0,
            'avg_nps': float(np.mean(nps)) if len(nps) > 0 else 0,
            'fastest_passage_nps': float(np.max(window_nps)) if window_nps else 0,
            'speed_by_section': window_nps,
        }
    
    def _analyze_techniques(self, notes: List[Any]) -> TechniqueStats:
        """Analyze technique usage from annotated notes."""
        stats = TechniqueStats()
        
        for note in notes:
            # Check for technique attribute (AnnotatedNote)
            if hasattr(note, 'technique') and hasattr(note.technique, 'technique'):
                tech = note.technique.technique
                tech_name = tech.name if hasattr(tech, 'name') else str(tech)
                
                if 'HAMMER' in tech_name:
                    stats.hammer_ons += 1
                elif 'PULL' in tech_name:
                    stats.pull_offs += 1
                elif 'BEND' in tech_name:
                    stats.bends += 1
                elif 'SLIDE' in tech_name:
                    stats.slides += 1
                elif 'VIBRATO' in tech_name:
                    stats.vibrato += 1
                elif 'TRILL' in tech_name:
                    stats.trills += 1
                elif 'TREMOLO' in tech_name:
                    stats.tremolo += 1
        
        return stats
    
    def _find_difficult_passages(
        self,
        tab_notes: List[Any],
        speed_metrics: Dict[str, float],
        spans: List[int],
        technique_stats: TechniqueStats
    ) -> List[DifficultPassage]:
        """Identify difficult passages in the tab."""
        passages = []
        
        if not tab_notes:
            return passages
        
        # Sliding window analysis
        window_size = self.passage_window
        total_duration = tab_notes[-1].start_time - tab_notes[0].start_time
        
        for window_start in np.arange(0, total_duration, window_size / 2):
            window_end = window_start + window_size
            
            # Get notes in window
            window_notes = [
                (i, n) for i, n in enumerate(tab_notes)
                if window_start <= n.start_time < window_end
            ]
            
            if len(window_notes) < 3:
                continue
            
            reasons = []
            scores = []
            
            # Check speed
            actual_duration = window_notes[-1][1].start_time - window_notes[0][1].start_time
            if actual_duration > 0:
                nps = len(window_notes) / actual_duration
                if nps >= self.advanced_nps:
                    reasons.append(f"Fast passage ({nps:.1f} notes/sec)")
                    scores.append(min(1.0, nps / 15.0))
                elif nps >= self.intermediate_nps:
                    reasons.append(f"Moderate speed ({nps:.1f} notes/sec)")
                    scores.append(nps / 15.0)
            
            # Check stretches
            window_spans = [spans[i] for i, _ in window_notes if i < len(spans)]
            if window_spans:
                max_span = max(window_spans)
                if max_span >= self.advanced_span:
                    reasons.append(f"Wide stretch ({max_span} frets)")
                    scores.append(min(1.0, max_span / 7.0))
            
            # Check position changes
            positions = self._identify_positions([n for _, n in window_notes])
            pos_changes = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
            if pos_changes >= 3:
                reasons.append(f"Multiple position shifts ({pos_changes})")
                scores.append(min(1.0, pos_changes / 5.0))
            
            # Create passage if difficult
            if reasons and max(scores) >= 0.4:
                passages.append(DifficultPassage(
                    start_time=window_start,
                    end_time=window_end,
                    reason="; ".join(reasons),
                    difficulty_score=max(scores),
                    notes_involved=len(window_notes),
                    suggested_practice_tempo=max(30, int(50 / max(scores)))
                ))
        
        # Merge overlapping passages
        merged = []
        for passage in sorted(passages, key=lambda p: p.start_time):
            if merged and passage.start_time < merged[-1].end_time:
                # Extend existing passage
                merged[-1].end_time = max(merged[-1].end_time, passage.end_time)
                merged[-1].difficulty_score = max(merged[-1].difficulty_score, passage.difficulty_score)
                merged[-1].reason = merged[-1].reason + "; " + passage.reason
            else:
                merged.append(passage)
        
        # Return top passages
        return sorted(merged, key=lambda p: -p.difficulty_score)[:10]
    
    def _suggest_practice_sections(
        self,
        tab_notes: List[Any],
        difficult_passages: List[DifficultPassage],
        technique_stats: TechniqueStats
    ) -> List[PracticeSection]:
        """Generate practice section suggestions."""
        sections = []
        
        # Add sections from difficult passages
        for passage in difficult_passages[:5]:
            tips = []
            if "Fast" in passage.reason:
                tips.append("Start at 50% tempo with metronome")
                tips.append("Increase speed by 5 BPM when comfortable")
            if "stretch" in passage.reason.lower():
                tips.append("Practice finger stretching exercises first")
                tips.append("Ensure proper thumb position on neck")
            if "position" in passage.reason.lower():
                tips.append("Practice position shifts in isolation")
                tips.append("Use guide fingers for smooth transitions")
            
            sections.append(PracticeSection(
                start_time=passage.start_time,
                end_time=passage.end_time,
                focus=passage.reason.split(";")[0].strip(),
                difficulty=DifficultyLevel.ADVANCED if passage.difficulty_score > 0.7 else DifficultyLevel.INTERMEDIATE,
                tips=tips,
            ))
        
        # Add technique-specific sections if techniques are present
        if technique_stats.bends > 2:
            sections.append(PracticeSection(
                start_time=0,
                end_time=0,
                focus="Bend accuracy practice",
                difficulty=DifficultyLevel.INTERMEDIATE,
                tips=[
                    "Use a tuner to verify bend pitch accuracy",
                    "Practice whole step and half step bends separately",
                    "Focus on consistent vibrato at bend peak",
                ],
            ))
        
        if technique_stats.hammer_ons + technique_stats.pull_offs > 3:
            sections.append(PracticeSection(
                start_time=0,
                end_time=0,
                focus="Legato technique",
                difficulty=DifficultyLevel.INTERMEDIATE,
                tips=[
                    "Practice hammer-on strength for clear tone",
                    "Keep pull-offs quick and snappy",
                    "Try exercises on a single string first",
                ],
            ))
        
        return sections
    
    def _calculate_speed_rating(self, speed_metrics: Dict[str, float]) -> float:
        """Calculate speed difficulty rating (0-100)."""
        max_nps = speed_metrics['max_nps']
        fastest = speed_metrics['fastest_passage_nps']
        
        # Use faster of max instantaneous or windowed
        effective_speed = max(max_nps, fastest)
        
        if effective_speed <= self.beginner_nps:
            return effective_speed / self.beginner_nps * 25
        elif effective_speed <= self.intermediate_nps:
            return 25 + (effective_speed - self.beginner_nps) / (self.intermediate_nps - self.beginner_nps) * 25
        elif effective_speed <= self.advanced_nps:
            return 50 + (effective_speed - self.intermediate_nps) / (self.advanced_nps - self.intermediate_nps) * 25
        else:
            return min(100, 75 + (effective_speed - self.advanced_nps) / 5.0 * 25)
    
    def _calculate_technique_rating(self, technique_stats: TechniqueStats, total_notes: int) -> float:
        """Calculate technique difficulty rating (0-100)."""
        if total_notes == 0:
            return 0
        
        # Base on technique density and complexity
        density = technique_stats.technique_density(total_notes)
        
        # Weight certain techniques higher
        complexity_score = (
            technique_stats.bends * 3 +
            technique_stats.trills * 3 +
            technique_stats.vibrato * 2 +
            technique_stats.hammer_ons * 1 +
            technique_stats.pull_offs * 1 +
            technique_stats.slides * 1.5 +
            technique_stats.tremolo * 2
        ) / max(total_notes, 1) * 100
        
        # Combine density and complexity
        return min(100, (density * 2 + complexity_score) / 2)
    
    def _calculate_stretch_rating(self, max_span: int, avg_span: float) -> float:
        """Calculate stretch difficulty rating (0-100)."""
        # Max span contribution
        if max_span <= self.beginner_span:
            max_score = max_span / self.beginner_span * 25
        elif max_span <= self.intermediate_span:
            max_score = 25 + (max_span - self.beginner_span) / (self.intermediate_span - self.beginner_span) * 25
        elif max_span <= self.advanced_span:
            max_score = 50 + (max_span - self.intermediate_span) / (self.advanced_span - self.intermediate_span) * 25
        else:
            max_score = min(100, 75 + (max_span - self.advanced_span) / 3.0 * 25)
        
        # Average span contribution
        avg_score = min(100, avg_span / 4.0 * 50)
        
        # Combine (max is more important)
        return max_score * 0.7 + avg_score * 0.3
    
    def _calculate_position_rating(self, position_changes: int, unique_positions: int, total_notes: int) -> float:
        """Calculate position change difficulty rating (0-100)."""
        if total_notes == 0:
            return 0
        
        # Position changes per note
        change_rate = position_changes / total_notes
        
        # Scale: 0.1 changes/note = ~50/100
        rate_score = min(100, change_rate * 500)
        
        # Unique positions contribution
        if unique_positions <= self.beginner_positions:
            pos_score = unique_positions / self.beginner_positions * 25
        elif unique_positions <= self.intermediate_positions:
            pos_score = 25 + (unique_positions - self.beginner_positions) / (self.intermediate_positions - self.beginner_positions) * 25
        else:
            pos_score = min(100, 50 + (unique_positions - self.intermediate_positions) / 5.0 * 50)
        
        return rate_score * 0.6 + pos_score * 0.4
    
    def _score_to_rating(self, score: float) -> DifficultyLevel:
        """Convert numeric score to difficulty level."""
        if score < 25:
            return DifficultyLevel.BEGINNER
        elif score < 50:
            return DifficultyLevel.INTERMEDIATE
        elif score < 75:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT


def analyze_difficulty(
    tab_notes: List[Any],
    notes: Optional[List[Any]] = None,
    annotated_notes: Optional[List[Any]] = None,
    tempo: int = 120,
    verbose: bool = True
) -> DifficultyReport:
    """
    Convenience function to analyze tab difficulty.
    
    Args:
        tab_notes: List of TabNote objects
        notes: Optional list of original Note objects
        annotated_notes: Optional list of AnnotatedNote with technique info
        tempo: Song tempo in BPM
        verbose: Print progress
        
    Returns:
        DifficultyReport with complete analysis
    """
    analyzer = DifficultyAnalyzer()
    return analyzer.analyze(
        tab_notes=tab_notes,
        notes=notes,
        annotated_notes=annotated_notes,
        tempo=tempo,
        verbose=verbose
    )


# Test
if __name__ == "__main__":
    print("Testing DifficultyAnalyzer...")
    
    # Create mock TabNote
    @dataclass
    class MockTabNote:
        string: int
        fret: int
        start_time: float
        duration: float
    
    # Create a test sequence (basic melody)
    test_tabs = [
        MockTabNote(4, 3, 0.0, 0.25),
        MockTabNote(4, 5, 0.25, 0.25),
        MockTabNote(3, 5, 0.5, 0.25),
        MockTabNote(3, 7, 0.75, 0.25),
        MockTabNote(2, 5, 1.0, 0.5),
        MockTabNote(2, 7, 1.5, 0.25),
        MockTabNote(2, 8, 1.75, 0.25),
        MockTabNote(1, 5, 2.0, 0.5),
    ]
    
    report = analyze_difficulty(test_tabs, verbose=True)
    print(report.format_report())
