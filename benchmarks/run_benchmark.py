#!/usr/bin/env python3
"""
Guitar Tab Transcription Benchmark Runner

Runs all pitch detection methods against test audio files and computes
accuracy metrics.

Usage:
    # Generate test audio and run benchmark
    python run_benchmark.py --generate --run
    
    # Run benchmark on existing test audio
    python run_benchmark.py --run
    
    # Generate test audio only
    python run_benchmark.py --generate
    
    # Run specific methods
    python run_benchmark.py --run --methods pyin,piptrack
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa

from benchmarks.generate_test_audio import TestCaseGenerator
from benchmarks.metrics import (
    TranscriptionEvaluator, BenchmarkResult, GroundTruthNote, DetectedNote,
    aggregate_results, format_results_table, format_detailed_results
)
from benchmarks.pitch_detectors import (
    get_all_detectors, get_detector, DETECTORS
)


def load_ground_truth(json_path: Path) -> List[GroundTruthNote]:
    """Load ground truth notes from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    notes = []
    for n in data.get('notes', []):
        notes.append(GroundTruthNote(
            midi_note=n['midi_note'],
            start_time=n['start_time'],
            end_time=n['end_time'],
            velocity=n.get('velocity', 1.0)
        ))
    
    return notes


def run_benchmark(
    test_dir: Path,
    output_dir: Path,
    methods: Optional[List[str]] = None,
    sr: int = 22050,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Run benchmark on all test cases.
    
    Args:
        test_dir: Directory containing test audio and JSON files
        output_dir: Directory to save results
        methods: List of method names to run (None = all)
        sr: Sample rate
        verbose: Print progress
        
    Returns:
        List of BenchmarkResult for each (test, method) pair
    """
    # Load index
    index_path = test_dir / "index.json"
    if not index_path.exists():
        print(f"Error: No index.json found in {test_dir}")
        print("Run with --generate first to create test audio")
        return []
    
    with open(index_path) as f:
        index = json.load(f)
    
    test_cases = index['test_cases']
    print(f"📊 Running benchmark on {len(test_cases)} test cases")
    
    # Initialize detectors
    if methods:
        detectors = [get_detector(m, sr=sr) for m in methods if get_detector(m, sr=sr)]
    else:
        detectors = get_all_detectors(sr=sr)
    
    print(f"🔧 Using {len(detectors)} detection methods: {[d.name for d in detectors]}")
    
    # Initialize evaluator
    evaluator = TranscriptionEvaluator(
        pitch_tolerance=0.5,  # Half semitone
        onset_tolerance=50.0,  # 50ms
    )
    
    all_results: List[BenchmarkResult] = []
    
    # Run each test case
    for test_name in test_cases:
        audio_path = test_dir / f"{test_name}.wav"
        json_path = test_dir / f"{test_name}.json"
        
        if not audio_path.exists() or not json_path.exists():
            print(f"  ⚠ Skipping {test_name} - missing files")
            continue
        
        if verbose:
            print(f"\n  🎸 {test_name}")
        
        # Load audio and ground truth
        audio, _ = librosa.load(str(audio_path), sr=sr)
        ground_truth = load_ground_truth(json_path)
        
        # Run each detector
        for detector in detectors:
            try:
                result = detector.detect(audio)
                
                # Convert to evaluation format
                detected = result.notes
                
                # Evaluate
                eval_result = evaluator.evaluate(
                    ground_truth=ground_truth,
                    detected=detected,
                    test_name=test_name,
                    method_name=detector.name
                )
                eval_result.processing_time_ms = result.processing_time_ms
                
                all_results.append(eval_result)
                
                if verbose:
                    status = "✓" if eval_result.f1_score > 0.8 else ("◐" if eval_result.f1_score > 0.5 else "✗")
                    print(f"      {status} {detector.name:<15} F1={eval_result.f1_score:.3f} "
                          f"(P={eval_result.precision:.2f} R={eval_result.recall:.2f}) "
                          f"{eval_result.processing_time_ms:.1f}ms")
                
            except Exception as e:
                print(f"      ❌ {detector.name}: {e}")
    
    return all_results


def save_results(
    results: List[BenchmarkResult],
    output_dir: Path,
    summary: Dict
):
    """Save benchmark results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_path = output_dir / f"results_{timestamp}.json"
    detailed_data = {
        'timestamp': timestamp,
        'num_results': len(results),
        'results': [r.to_dict() for r in results],
        'summary': summary
    }
    with open(detailed_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    # Save latest symlink
    latest_path = output_dir / "results_latest.json"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(detailed_path.name)
    
    # Save summary table as text
    table_path = output_dir / f"summary_{timestamp}.txt"
    with open(table_path, 'w') as f:
        f.write(f"Benchmark Results - {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(format_results_table(summary))
        f.write("\n\n")
        f.write(format_detailed_results(results))
    
    print(f"\n💾 Results saved to:")
    print(f"   {detailed_path}")
    print(f"   {table_path}")


def print_summary(results: List[BenchmarkResult], summary: Dict):
    """Print summary to console."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(format_results_table(summary))
    
    # Print best method
    if summary:
        best_method = max(summary.items(), key=lambda x: x[1]['mean_f1'])
        print(f"\n🏆 Best method: {best_method[0]} (F1={best_method[1]['mean_f1']:.3f})")
    
    # Print category breakdown
    print("\n📊 Results by category:")
    
    categories = {
        'single': [],
        'pure': [],
        'seq': [],
        'chord': [],
        'edge': []
    }
    
    for r in results:
        for cat in categories:
            if r.test_name.startswith(cat):
                categories[cat].append(r)
                break
    
    for cat, cat_results in categories.items():
        if not cat_results:
            continue
        
        by_method = {}
        for r in cat_results:
            if r.method_name not in by_method:
                by_method[r.method_name] = []
            by_method[r.method_name].append(r.f1_score)
        
        print(f"\n  {cat.upper()} tests ({len(cat_results)//len(by_method)} tests):")
        for method, f1s in sorted(by_method.items(), key=lambda x: np.mean(x[1]), reverse=True):
            print(f"    {method:<15} avg F1={np.mean(f1s):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Guitar Tab Transcription Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --generate --run       Generate test audio and run benchmark
    %(prog)s --run                  Run benchmark on existing test audio
    %(prog)s --methods pyin,hybrid  Run specific methods only
    %(prog)s --list                 List available methods
        """
    )
    
    parser.add_argument('--generate', action='store_true',
                        help='Generate synthetic test audio')
    parser.add_argument('--run', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to run')
    parser.add_argument('--test-dir', type=str, default='test_audio',
                        help='Directory containing test audio')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate')
    parser.add_argument('--list', action='store_true',
                        help='List available detection methods')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    benchmark_dir = Path(__file__).parent
    test_dir = benchmark_dir / args.test_dir
    output_dir = benchmark_dir / args.output_dir
    
    if args.list:
        print("Available pitch detection methods:")
        for name in DETECTORS:
            print(f"  - {name}")
        return 0
    
    if not args.generate and not args.run:
        parser.print_help()
        return 1
    
    # Generate test audio
    if args.generate:
        print("🎵 Generating synthetic test audio...")
        generator = TestCaseGenerator(str(test_dir), sr=args.sr)
        generator.generate_all()
    
    # Run benchmark
    if args.run:
        methods = args.methods.split(',') if args.methods else None
        
        results = run_benchmark(
            test_dir=test_dir,
            output_dir=output_dir,
            methods=methods,
            sr=args.sr,
            verbose=args.verbose or True
        )
        
        if results:
            summary = aggregate_results(results)
            save_results(results, output_dir, summary)
            print_summary(results, summary)
        else:
            print("No results generated")
            return 1
    
    print("\n✅ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
