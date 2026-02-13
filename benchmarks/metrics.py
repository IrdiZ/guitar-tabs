#!/usr/bin/env python3
"""
Accuracy Metrics for Guitar Tab Transcription Benchmarking

Metrics implemented:
1. Note Detection Rate (Recall) - What % of true notes were detected?
2. Precision - What % of detected notes were correct?
3. F1 Score - Harmonic mean of precision and recall
4. Pitch Accuracy - How close are detected pitches to ground truth?
5. Timing Accuracy - How close are onset/offset times?
6. Polyphonic Accuracy - How well are chords detected?
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json


@dataclass
class DetectedNote:
    """A note detected by a pitch detection method."""
    midi_note: int
    start_time: float
    end_time: float
    confidence: float = 1.0
    frequency: float = 0.0  # Raw detected frequency
    
    def __post_init__(self):
        if self.frequency == 0.0:
            self.frequency = 440.0 * (2 ** ((self.midi_note - 69) / 12))


@dataclass 
class GroundTruthNote:
    """A note from the ground truth."""
    midi_note: int
    start_time: float
    end_time: float
    velocity: float = 1.0


@dataclass
class NoteMatch:
    """A matched pair of detected and ground truth notes."""
    ground_truth: GroundTruthNote
    detected: DetectedNote
    pitch_error_semitones: float  # Signed error in semitones
    onset_error_ms: float  # Signed error in milliseconds
    offset_error_ms: float  # Signed error in milliseconds
    is_correct: bool  # Within tolerance


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a single test case."""
    test_name: str
    method_name: str
    
    # Core metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Detailed counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Accuracy metrics
    mean_pitch_error: float = 0.0  # In semitones
    std_pitch_error: float = 0.0
    mean_onset_error: float = 0.0  # In milliseconds
    std_onset_error: float = 0.0
    mean_offset_error: float = 0.0  # In milliseconds
    
    # Tolerance used
    pitch_tolerance: float = 0.5  # semitones
    onset_tolerance: float = 50.0  # ms
    
    # Processing info
    processing_time_ms: float = 0.0
    
    # Matched notes for analysis
    matches: List[NoteMatch] = field(default_factory=list)
    unmatched_gt: List[GroundTruthNote] = field(default_factory=list)
    unmatched_det: List[DetectedNote] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'test_name': self.test_name,
            'method_name': self.method_name,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'mean_pitch_error_semitones': round(self.mean_pitch_error, 4),
            'std_pitch_error_semitones': round(self.std_pitch_error, 4),
            'mean_onset_error_ms': round(self.mean_onset_error, 2),
            'std_onset_error_ms': round(self.std_onset_error, 2),
            'mean_offset_error_ms': round(self.mean_offset_error, 2),
            'processing_time_ms': round(self.processing_time_ms, 2),
            'pitch_tolerance': self.pitch_tolerance,
            'onset_tolerance_ms': self.onset_tolerance,
        }


class TranscriptionEvaluator:
    """
    Evaluates pitch detection results against ground truth.
    
    Uses a matching algorithm that:
    1. Finds optimal one-to-one matches between detected and ground truth notes
    2. Computes precision, recall, and F1 score
    3. Computes pitch and timing accuracy for matched notes
    """
    
    def __init__(
        self,
        pitch_tolerance: float = 0.5,  # semitones (0.5 = quarter tone)
        onset_tolerance: float = 50.0,  # milliseconds
        offset_tolerance: float = 100.0,  # milliseconds (more lenient)
        overlap_threshold: float = 0.3,  # minimum IoU for time overlap
    ):
        self.pitch_tolerance = pitch_tolerance
        self.onset_tolerance = onset_tolerance
        self.offset_tolerance = offset_tolerance
        self.overlap_threshold = overlap_threshold
    
    def evaluate(
        self,
        ground_truth: List[GroundTruthNote],
        detected: List[DetectedNote],
        test_name: str = "test",
        method_name: str = "unknown"
    ) -> BenchmarkResult:
        """
        Evaluate detected notes against ground truth.
        
        Returns BenchmarkResult with all metrics.
        """
        result = BenchmarkResult(
            test_name=test_name,
            method_name=method_name,
            pitch_tolerance=self.pitch_tolerance,
            onset_tolerance=self.onset_tolerance
        )
        
        if not ground_truth:
            # No ground truth - all detections are false positives
            result.false_positives = len(detected)
            result.unmatched_det = list(detected)
            return result
        
        if not detected:
            # No detections - all ground truth are false negatives
            result.false_negatives = len(ground_truth)
            result.unmatched_gt = list(ground_truth)
            return result
        
        # Find matches using greedy matching algorithm
        matches, unmatched_gt, unmatched_det = self._match_notes(ground_truth, detected)
        
        result.matches = matches
        result.unmatched_gt = unmatched_gt
        result.unmatched_det = unmatched_det
        result.true_positives = len(matches)
        result.false_positives = len(unmatched_det)
        result.false_negatives = len(unmatched_gt)
        
        # Compute precision, recall, F1
        total_detected = len(detected)
        total_gt = len(ground_truth)
        
        result.precision = result.true_positives / total_detected if total_detected > 0 else 0.0
        result.recall = result.true_positives / total_gt if total_gt > 0 else 0.0
        
        if result.precision + result.recall > 0:
            result.f1_score = 2 * result.precision * result.recall / (result.precision + result.recall)
        
        # Compute error statistics from matches
        if matches:
            pitch_errors = [abs(m.pitch_error_semitones) for m in matches]
            onset_errors = [abs(m.onset_error_ms) for m in matches]
            offset_errors = [abs(m.offset_error_ms) for m in matches]
            
            result.mean_pitch_error = float(np.mean(pitch_errors))
            result.std_pitch_error = float(np.std(pitch_errors))
            result.mean_onset_error = float(np.mean(onset_errors))
            result.std_onset_error = float(np.std(onset_errors))
            result.mean_offset_error = float(np.mean(offset_errors))
        
        return result
    
    def _match_notes(
        self,
        ground_truth: List[GroundTruthNote],
        detected: List[DetectedNote]
    ) -> Tuple[List[NoteMatch], List[GroundTruthNote], List[DetectedNote]]:
        """
        Find optimal one-to-one matching between ground truth and detected notes.
        
        Uses greedy matching: for each ground truth note, find the best unmatched
        detected note within tolerance.
        """
        # Sort by start time for efficiency
        gt_sorted = sorted(ground_truth, key=lambda n: n.start_time)
        det_sorted = sorted(detected, key=lambda n: n.start_time)
        
        matches: List[NoteMatch] = []
        used_gt = set()
        used_det = set()
        
        # Build cost matrix and find matches greedily
        for i, gt in enumerate(gt_sorted):
            best_match_idx = -1
            best_score = float('inf')
            
            for j, det in enumerate(det_sorted):
                if j in used_det:
                    continue
                
                # Check if match is possible
                pitch_error, onset_error, offset_error, overlap = self._compute_errors(gt, det)
                
                # Skip if outside tolerance
                if abs(pitch_error) > self.pitch_tolerance:
                    continue
                if abs(onset_error) > self.onset_tolerance:
                    continue
                if overlap < self.overlap_threshold:
                    continue
                
                # Compute match score (lower is better)
                # Weight pitch error more heavily
                score = abs(pitch_error) * 2 + abs(onset_error) / 50.0
                
                if score < best_score:
                    best_score = score
                    best_match_idx = j
            
            if best_match_idx >= 0:
                det = det_sorted[best_match_idx]
                pitch_error, onset_error, offset_error, _ = self._compute_errors(gt, det)
                
                match = NoteMatch(
                    ground_truth=gt,
                    detected=det,
                    pitch_error_semitones=pitch_error,
                    onset_error_ms=onset_error,
                    offset_error_ms=offset_error,
                    is_correct=True
                )
                matches.append(match)
                used_gt.add(i)
                used_det.add(best_match_idx)
        
        # Collect unmatched notes
        unmatched_gt = [gt for i, gt in enumerate(gt_sorted) if i not in used_gt]
        unmatched_det = [det for j, det in enumerate(det_sorted) if j not in used_det]
        
        return matches, unmatched_gt, unmatched_det
    
    def _compute_errors(
        self,
        gt: GroundTruthNote,
        det: DetectedNote
    ) -> Tuple[float, float, float, float]:
        """
        Compute errors between a ground truth and detected note.
        
        Returns:
            (pitch_error_semitones, onset_error_ms, offset_error_ms, time_overlap_iou)
        """
        # Pitch error in semitones
        pitch_error = det.midi_note - gt.midi_note
        
        # Timing errors in milliseconds
        onset_error = (det.start_time - gt.start_time) * 1000
        offset_error = (det.end_time - gt.end_time) * 1000
        
        # Time overlap (IoU)
        intersection_start = max(gt.start_time, det.start_time)
        intersection_end = min(gt.end_time, det.end_time)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(gt.start_time, det.start_time)
        union_end = max(gt.end_time, det.end_time)
        union = union_end - union_start
        
        overlap = intersection / union if union > 0 else 0.0
        
        return pitch_error, onset_error, offset_error, overlap


def aggregate_results(results: List[BenchmarkResult]) -> Dict[str, Dict]:
    """
    Aggregate results across multiple test cases.
    
    Returns summary statistics by method.
    """
    by_method: Dict[str, List[BenchmarkResult]] = {}
    
    for r in results:
        if r.method_name not in by_method:
            by_method[r.method_name] = []
        by_method[r.method_name].append(r)
    
    summary = {}
    
    for method, method_results in by_method.items():
        precisions = [r.precision for r in method_results]
        recalls = [r.recall for r in method_results]
        f1s = [r.f1_score for r in method_results]
        pitch_errors = [r.mean_pitch_error for r in method_results if r.mean_pitch_error > 0]
        onset_errors = [r.mean_onset_error for r in method_results if r.mean_onset_error > 0]
        times = [r.processing_time_ms for r in method_results]
        
        total_tp = sum(r.true_positives for r in method_results)
        total_fp = sum(r.false_positives for r in method_results)
        total_fn = sum(r.false_negatives for r in method_results)
        
        summary[method] = {
            'num_tests': len(method_results),
            'mean_precision': round(float(np.mean(precisions)), 4),
            'std_precision': round(float(np.std(precisions)), 4),
            'mean_recall': round(float(np.mean(recalls)), 4),
            'std_recall': round(float(np.std(recalls)), 4),
            'mean_f1': round(float(np.mean(f1s)), 4),
            'std_f1': round(float(np.std(f1s)), 4),
            'overall_precision': round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0,
            'overall_recall': round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0,
            'mean_pitch_error': round(float(np.mean(pitch_errors)), 4) if pitch_errors else 0,
            'mean_onset_error_ms': round(float(np.mean(onset_errors)), 2) if onset_errors else 0,
            'mean_processing_time_ms': round(float(np.mean(times)), 2) if times else 0,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
        }
    
    return summary


def format_results_table(summary: Dict[str, Dict]) -> str:
    """Format summary as ASCII table for display."""
    lines = []
    
    # Header
    lines.append("=" * 100)
    lines.append(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Pitch Err':>12} {'Onset Err':>12} {'Time (ms)':>12}")
    lines.append("=" * 100)
    
    # Sort by F1 score descending
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    
    for method, stats in sorted_methods:
        pitch_err = f"{stats['mean_pitch_error']:.3f} st" if stats['mean_pitch_error'] else "N/A"
        onset_err = f"{stats['mean_onset_error_ms']:.1f} ms" if stats['mean_onset_error_ms'] else "N/A"
        
        lines.append(
            f"{method:<20} "
            f"{stats['mean_precision']:>10.3f} "
            f"{stats['mean_recall']:>10.3f} "
            f"{stats['mean_f1']:>10.3f} "
            f"{pitch_err:>12} "
            f"{onset_err:>12} "
            f"{stats['mean_processing_time_ms']:>12.1f}"
        )
    
    lines.append("=" * 100)
    return "\n".join(lines)


def format_detailed_results(results: List[BenchmarkResult]) -> str:
    """Format detailed per-test results."""
    lines = []
    
    # Group by test
    by_test: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.test_name not in by_test:
            by_test[r.test_name] = []
        by_test[r.test_name].append(r)
    
    for test_name, test_results in sorted(by_test.items()):
        lines.append(f"\n{'─'*60}")
        lines.append(f"📝 {test_name}")
        lines.append(f"{'─'*60}")
        
        # Sort by F1
        test_results.sort(key=lambda r: r.f1_score, reverse=True)
        
        for r in test_results:
            status = "✓" if r.f1_score > 0.8 else ("◐" if r.f1_score > 0.5 else "✗")
            lines.append(
                f"  {status} {r.method_name:<15} "
                f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1_score:.2f} "
                f"TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives}"
            )
    
    return "\n".join(lines)
