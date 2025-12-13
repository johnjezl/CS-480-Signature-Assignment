#!/usr/bin/env python3
"""
Preprocessor Metrics Analyzer

Analyzes preprocessor performance data collected during cube solving sessions.
Provides insights on which preprocessing methods work best for segmentation
and color classification, organized by segmenter algorithm.

Usage:
    python tools/analyze_preprocessor_metrics.py
    python tools/analyze_preprocessor_metrics.py --file custom_metrics.json
    python tools/analyze_preprocessor_metrics.py --segmenter contour-neighbor
    python tools/analyze_preprocessor_metrics.py --sort confidence
    python tools/analyze_preprocessor_metrics.py --context seg_preprocess
    python tools/analyze_preprocessor_metrics.py --top 10
    python tools/analyze_preprocessor_metrics.py --export csv
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PreprocessorMetrics import PreprocessorMetrics


def load_metrics(filepath: str) -> dict:
    """Load raw metrics data from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Metrics file not found: {filepath}")
        sys.exit(1)

    with open(filepath, 'r') as f:
        return json.load(f)


def format_duration(start_iso: str, end_iso: str) -> str:
    """Format duration between two ISO timestamps."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        delta = end - start

        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)
    except (ValueError, TypeError):
        return "unknown"


def print_header(title: str, width: int = 90):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 90):
    """Print a formatted subheader."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print("-" * width)


def analyze_basic_stats(data: dict):
    """Print basic statistics about the metrics data."""
    print_header("METRICS OVERVIEW")

    metadata = data.get('metadata', {})
    segmenters = data.get('segmenters', {})

    created = metadata.get('created', 'unknown')
    updated = metadata.get('last_updated', 'unknown')
    version = metadata.get('version', 1)

    print(f"  Data version:  {version}")
    print(f"  Created:       {created[:19] if created != 'unknown' else created}")
    print(f"  Last updated:  {updated[:19] if updated != 'unknown' else updated}")

    if created != 'unknown' and updated != 'unknown':
        print(f"  Duration:      {format_duration(created, updated)}")

    print(f"\n  Segmenters tracked: {len(segmenters)}")
    if segmenters:
        for seg_name in sorted(segmenters.keys()):
            seg_data = segmenters[seg_name]
            seg_preprocess_count = len(seg_data.get('seg_preprocess', {}))
            cc_preprocess_count = len(seg_data.get('cc_preprocess', {}))
            combo_count = len(seg_data.get('combinations', {}))

            # Count total attempts for this segmenter
            total_attempts = sum(
                s.get('attempts', 0)
                for s in seg_data.get('combinations', {}).values()
            )
            total_successes = sum(
                s.get('successes', 0)
                for s in seg_data.get('combinations', {}).values()
            )
            rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0

            print(f"    - {seg_name}: {combo_count} combinations, {total_attempts} attempts, {rate:.1f}% success")


def analyze_segmenter_summary(data: dict, segmenter: str = None):
    """Summarize overall performance by segmenter."""
    segmenters = data.get('segmenters', {})

    if not segmenters:
        print("\n  No data recorded yet.")
        return

    print_header("SEGMENTER PERFORMANCE SUMMARY")

    seg_list = [segmenter] if segmenter else sorted(segmenters.keys())

    print(f"{'Segmenter':<25} {'Combinations':>12} {'Attempts':>10} {'Success':>10} {'Rate':>10} {'Avg Conf':>12}")
    print("-" * 90)

    total_attempts = 0
    total_successes = 0

    for seg_name in seg_list:
        if seg_name not in segmenters:
            continue

        seg_data = segmenters[seg_name]
        combos = seg_data.get('combinations', {})

        attempts = sum(s.get('attempts', 0) for s in combos.values())
        successes = sum(s.get('successes', 0) for s in combos.values())
        total_conf = sum(s.get('total_confidence', 0) for s in combos.values())

        rate = (successes / attempts * 100) if attempts > 0 else 0
        avg_conf = (total_conf / attempts) if attempts > 0 else 0

        total_attempts += attempts
        total_successes += successes

        print(f"{seg_name:<25} {len(combos):>12} {attempts:>10} {successes:>10} {rate:>9.1f}% {avg_conf:>11.1f}")

    print("-" * 90)
    overall_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
    print(f"{'TOTAL':<25} {'':<12} {total_attempts:>10} {total_successes:>10} {overall_rate:>9.1f}%")


def analyze_context(data: dict, segmenter: str, context: str, sort_by: str = 'rate', top_n: int = None, min_attempts: int = 1):
    """Analyze and display metrics for a specific context within a segmenter."""
    segmenters = data.get('segmenters', {})

    if segmenter not in segmenters:
        print(f"\n  No data for segmenter: {segmenter}")
        return

    seg_data = segmenters[segmenter]
    context_data = seg_data.get(context, {})

    if not context_data:
        print(f"\n  No {context} data for segmenter: {segmenter}")
        return

    # Collect stats
    stats = []
    for method, method_data in context_data.items():
        attempts = method_data.get('attempts', 0)
        if attempts >= min_attempts:
            successes = method_data.get('successes', 0)
            failures = method_data.get('failures', 0)
            total_conf = method_data.get('total_confidence', 0.0)

            stats.append({
                'method': method,
                'attempts': attempts,
                'successes': successes,
                'failures': failures,
                'rate': (successes / attempts * 100) if attempts > 0 else 0,
                'avg_conf': (total_conf / attempts) if attempts > 0 else 0
            })

    if not stats:
        print(f"\n  No data with {min_attempts}+ attempts for {context} in {segmenter}")
        return

    # Sort
    if sort_by == 'rate':
        stats.sort(key=lambda x: (x['rate'], x['avg_conf']), reverse=True)
    elif sort_by == 'confidence':
        stats.sort(key=lambda x: (x['avg_conf'], x['rate']), reverse=True)
    elif sort_by == 'attempts':
        stats.sort(key=lambda x: x['attempts'], reverse=True)
    elif sort_by == 'name':
        stats.sort(key=lambda x: x['method'])

    # Limit results
    if top_n:
        stats = stats[:top_n]

    # Display
    ctx_title = context.upper().replace('_', ' ')
    print_subheader(f"{ctx_title} ({segmenter})")

    print(f"{'Rank':<6} {'Method':<35} {'Attempts':>10} {'Success':>10} {'Rate':>10} {'Avg Conf':>12}")
    print("-" * 90)

    for i, s in enumerate(stats, 1):
        rate_str = f"{s['rate']:.1f}%"
        print(f"{i:<6} {s['method']:<35} {s['attempts']:>10} {s['successes']:>10} {rate_str:>10} {s['avg_conf']:>12.1f}")

    print("-" * 90)

    # Summary
    total_attempts = sum(s['attempts'] for s in stats)
    total_successes = sum(s['successes'] for s in stats)
    overall_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
    print(f"{'':6} {'TOTAL':<35} {total_attempts:>10} {total_successes:>10} {overall_rate:>9.1f}%")


def analyze_cc_by_seg_preprocess(data: dict, segmenter: str, seg_preprocess: str = None, min_attempts: int = 1):
    """Analyze CC preprocessing filtered by segmentation preprocessing."""
    segmenters = data.get('segmenters', {})

    if segmenter not in segmenters:
        print(f"\n  No data for segmenter: {segmenter}")
        return

    seg_data = segmenters[segmenter]
    cc_data = seg_data.get('cc_preprocess', {})

    if not cc_data:
        print(f"\n  No cc_preprocess data for segmenter: {segmenter}")
        return

    # Aggregate by CC method, optionally filtered by seg_preprocess
    cc_stats = {}
    for key, stats in cc_data.items():
        parts = key.split('|')
        if len(parts) != 2:
            continue
        cc_method, seg_method = parts

        # Filter by seg_preprocess if specified
        if seg_preprocess and seg_method != seg_preprocess.lower():
            continue

        if stats.get('attempts', 0) < min_attempts:
            continue

        if cc_method not in cc_stats:
            cc_stats[cc_method] = {'attempts': 0, 'successes': 0, 'total_confidence': 0}

        cc_stats[cc_method]['attempts'] += stats.get('attempts', 0)
        cc_stats[cc_method]['successes'] += stats.get('successes', 0)
        cc_stats[cc_method]['total_confidence'] += stats.get('total_confidence', 0)

    if not cc_stats:
        filter_str = f" (seg_preprocess={seg_preprocess})" if seg_preprocess else ""
        print(f"\n  No CC data{filter_str} with {min_attempts}+ attempts for {segmenter}")
        return

    # Convert to list and calculate rates
    stats_list = []
    for method, s in cc_stats.items():
        attempts = s['attempts']
        stats_list.append({
            'method': method,
            'attempts': attempts,
            'successes': s['successes'],
            'rate': (s['successes'] / attempts * 100) if attempts > 0 else 0,
            'avg_conf': (s['total_confidence'] / attempts) if attempts > 0 else 0
        })

    stats_list.sort(key=lambda x: (x['rate'], x['avg_conf']), reverse=True)

    # Display
    filter_str = f" (when seg_preprocess={seg_preprocess})" if seg_preprocess else ""
    print_subheader(f"CC PREPROCESSING{filter_str} ({segmenter})")

    print(f"{'Rank':<6} {'CC Method':<25} {'Attempts':>10} {'Success':>10} {'Rate':>10} {'Avg Conf':>12}")
    print("-" * 90)

    for i, s in enumerate(stats_list, 1):
        rate_str = f"{s['rate']:.1f}%"
        print(f"{i:<6} {s['method']:<25} {s['attempts']:>10} {s['successes']:>10} {rate_str:>10} {s['avg_conf']:>12.1f}")


def analyze_best_combinations(data: dict, segmenter: str = None, top_n: int = 10, min_attempts: int = 1):
    """Analyze best performing seg+cc combinations."""
    segmenters = data.get('segmenters', {})

    seg_list = [segmenter] if segmenter else sorted(segmenters.keys())

    for seg_name in seg_list:
        if seg_name not in segmenters:
            continue

        combos = segmenters[seg_name].get('combinations', {})

        stats = []
        for method, method_data in combos.items():
            attempts = method_data.get('attempts', 0)
            if attempts >= min_attempts:
                successes = method_data.get('successes', 0)
                total_conf = method_data.get('total_confidence', 0.0)

                parts = method.split('+', 1)
                seg_pp = parts[0] if len(parts) > 0 else 'unknown'
                cc_pp = parts[1] if len(parts) > 1 else 'unknown'

                stats.append({
                    'seg': seg_pp,
                    'cc': cc_pp,
                    'combined': method,
                    'attempts': attempts,
                    'successes': successes,
                    'rate': (successes / attempts * 100) if attempts > 0 else 0,
                    'avg_conf': (total_conf / attempts) if attempts > 0 else 0
                })

        if not stats:
            print(f"\n  No combination data for {seg_name} with {min_attempts}+ attempts")
            continue

        # Sort by success rate, then confidence
        stats.sort(key=lambda x: (x['rate'], x['avg_conf']), reverse=True)

        print_subheader(f"TOP {min(top_n, len(stats))} COMBINATIONS ({seg_name})")

        print(f"{'Rank':<6} {'Seg Preprocess':<20} {'CC Preprocess':<20} {'Rate':>10} {'Avg Conf':>12}")
        print("-" * 90)

        for i, s in enumerate(stats[:top_n], 1):
            rate_str = f"{s['rate']:.1f}%"
            print(f"{i:<6} {s['seg']:<20} {s['cc']:<20} {rate_str:>10} {s['avg_conf']:>12.1f}")

        # Show worst combinations if we have enough data
        if len(stats) > top_n * 2:
            print_subheader(f"BOTTOM {min(top_n, len(stats))} COMBINATIONS ({seg_name})")
            print(f"{'Rank':<6} {'Seg Preprocess':<20} {'CC Preprocess':<20} {'Rate':>10} {'Avg Conf':>12}")
            print("-" * 90)

            for i, s in enumerate(stats[-top_n:], len(stats) - top_n + 1):
                rate_str = f"{s['rate']:.1f}%"
                print(f"{i:<6} {s['seg']:<20} {s['cc']:<20} {rate_str:>10} {s['avg_conf']:>12.1f}")


def analyze_recommendations(data: dict, segmenter: str = None, min_attempts: int = 5):
    """Generate recommendations based on the data."""
    segmenters = data.get('segmenters', {})

    seg_list = [segmenter] if segmenter else sorted(segmenters.keys())

    print_header("RECOMMENDATIONS")

    for seg_name in seg_list:
        if seg_name not in segmenters:
            continue

        seg_data = segmenters[seg_name]

        print(f"\n  SEGMENTER: {seg_name.upper()}")
        print("  " + "-" * 40)

        for context in ['seg_preprocess', 'cc_preprocess', 'combinations']:
            context_data = seg_data.get(context, {})

            stats = []
            for method, method_stats in context_data.items():
                attempts = method_stats.get('attempts', 0)
                if attempts >= min_attempts:
                    successes = method_stats.get('successes', 0)
                    total_conf = method_stats.get('total_confidence', 0.0)
                    stats.append({
                        'method': method,
                        'attempts': attempts,
                        'rate': (successes / attempts * 100) if attempts > 0 else 0,
                        'avg_conf': (total_conf / attempts) if attempts > 0 else 0
                    })

            if not stats:
                print(f"    {context}: Not enough data (need {min_attempts}+ attempts)")
                continue

            stats.sort(key=lambda x: (x['rate'], x['avg_conf']), reverse=True)

            ctx_display = context.replace('_', ' ').upper()
            print(f"\n    {ctx_display}:")

            # Best performers
            best = stats[:3]
            best_strs = [f"{s['method']} ({s['rate']:.0f}%)" for s in best]
            best_str = ', '.join(best_strs)
            print(f"      Best:   {best_str}")

            # Worst performers
            worst = [s for s in stats if s['rate'] < 50][-3:]
            if worst:
                worst_strs = [f"{s['method']} ({s['rate']:.0f}%)" for s in worst]
                worst_str = ', '.join(worst_strs)
                print(f"      Worst:  {worst_str}")

            # Consider removing (< 20% success rate)
            remove = [s for s in stats if s['rate'] < 20]
            if remove:
                remove_str = ', '.join(s['method'] for s in remove)
                print(f"      Consider removing: {remove_str}")


def analyze_confidence_distribution(data: dict, segmenter: str, context: str = 'seg_preprocess'):
    """Show confidence score distribution."""
    segmenters = data.get('segmenters', {})

    if segmenter not in segmenters:
        print(f"\n  No data for segmenter: {segmenter}")
        return

    context_data = segmenters[segmenter].get(context, {})

    if not context_data:
        print(f"\n  No {context} data for segmenter: {segmenter}")
        return

    print_subheader(f"CONFIDENCE DISTRIBUTION ({context.upper()}, {segmenter})")

    stats = []
    for method, method_data in context_data.items():
        attempts = method_data.get('attempts', 0)
        if attempts > 0:
            total_conf = method_data.get('total_confidence', 0.0)
            avg_conf = total_conf / attempts
            stats.append({
                'method': method,
                'avg_conf': avg_conf,
                'attempts': attempts
            })

    if not stats:
        print("  No data available.")
        return

    stats.sort(key=lambda x: x['avg_conf'], reverse=True)

    # Find range
    max_conf = max(s['avg_conf'] for s in stats)
    min_conf = min(s['avg_conf'] for s in stats)

    print(f"  Confidence range: {min_conf:.1f} - {max_conf:.1f}")
    print(f"  (Max possible: 5400 = 54 facelets x 100% confidence)\n")

    # Visual bar chart
    bar_width = 40

    for s in stats[:15]:  # Top 15
        conf = s['avg_conf']
        pct = (conf / 5400) * 100 if conf > 0 else 0
        bar_len = int((conf / max_conf) * bar_width) if max_conf > 0 else 0
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        print(f"  {s['method']:<25} {bar} {conf:>7.1f} ({pct:>5.1f}%)")


def export_csv(data: dict, output_file: str):
    """Export metrics to CSV format."""
    segmenters = data.get('segmenters', {})

    rows = []
    for seg_name, seg_data in segmenters.items():
        for context in ['seg_preprocess', 'cc_preprocess', 'combinations']:
            context_data = seg_data.get(context, {})
            for method, method_stats in context_data.items():
                attempts = method_stats.get('attempts', 0)
                if attempts > 0:
                    successes = method_stats.get('successes', 0)
                    total_conf = method_stats.get('total_confidence', 0.0)
                    rows.append({
                        'segmenter': seg_name,
                        'context': context,
                        'method': method,
                        'attempts': attempts,
                        'successes': successes,
                        'failures': method_stats.get('failures', 0),
                        'success_rate': (successes / attempts * 100) if attempts > 0 else 0,
                        'avg_confidence': (total_conf / attempts) if attempts > 0 else 0
                    })

    with open(output_file, 'w') as f:
        # Header
        f.write("segmenter,context,method,attempts,successes,failures,success_rate,avg_confidence\n")
        # Data
        for row in rows:
            f.write(f"{row['segmenter']},{row['context']},{row['method']},{row['attempts']},"
                   f"{row['successes']},{row['failures']},"
                   f"{row['success_rate']:.2f},{row['avg_confidence']:.2f}\n")

    print(f"Exported {len(rows)} rows to {output_file}")


def export_json(data: dict, output_file: str):
    """Export processed metrics to JSON format."""
    segmenters = data.get('segmenters', {})

    output = {
        'metadata': data.get('metadata', {}),
        'segmenters': {}
    }

    for seg_name, seg_data in segmenters.items():
        output['segmenters'][seg_name] = {}

        for context in ['seg_preprocess', 'cc_preprocess', 'combinations']:
            output['segmenters'][seg_name][context] = []
            context_data = seg_data.get(context, {})

            for method, method_stats in context_data.items():
                attempts = method_stats.get('attempts', 0)
                if attempts > 0:
                    successes = method_stats.get('successes', 0)
                    total_conf = method_stats.get('total_confidence', 0.0)
                    output['segmenters'][seg_name][context].append({
                        'method': method,
                        'attempts': attempts,
                        'successes': successes,
                        'failures': method_stats.get('failures', 0),
                        'success_rate': round((successes / attempts * 100), 2) if attempts > 0 else 0,
                        'avg_confidence': round((total_conf / attempts), 2) if attempts > 0 else 0
                    })

            # Sort by success rate
            output['segmenters'][seg_name][context].sort(
                key=lambda x: (x['success_rate'], x['avg_confidence']),
                reverse=True
            )

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze preprocessor performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                             # Full analysis with default file
  %(prog)s --segmenter contour-neighbor  # Show only contour-neighbor data
  %(prog)s --sort confidence           # Sort by average confidence
  %(prog)s --context seg_preprocess    # Show only segmentation preprocessing stats
  %(prog)s --top 5                     # Show top 5 only
  %(prog)s --export csv                # Export to CSV
  %(prog)s --min-attempts 10           # Only show methods with 10+ attempts
"""
    )

    parser.add_argument('--file', '-f', type=str,
                        default='preprocessor_metrics.json',
                        help='Path to metrics JSON file (default: preprocessor_metrics.json)')

    parser.add_argument('--segmenter', '-g', type=str,
                        help='Filter by segmenter name (e.g., contour-neighbor, brightness-otsu)')

    parser.add_argument('--sort', '-s', type=str,
                        choices=['rate', 'confidence', 'attempts', 'name'],
                        default='rate',
                        help='Sort by: rate (default), confidence, attempts, or name')

    parser.add_argument('--context', '-c', type=str,
                        choices=['seg_preprocess', 'cc_preprocess', 'combinations', 'all'],
                        default='all',
                        help='Show specific context or all (default: all)')

    parser.add_argument('--seg-preprocess-filter', type=str,
                        help='When showing CC preprocessing, filter by this seg preprocessing method')

    parser.add_argument('--top', '-t', type=int, default=None,
                        help='Show only top N results per category')

    parser.add_argument('--min-attempts', '-m', type=int, default=1,
                        help='Minimum attempts to include in analysis (default: 1)')

    parser.add_argument('--export', '-e', type=str,
                        choices=['csv', 'json'],
                        help='Export data to CSV or JSON format')

    parser.add_argument('--output', '-o', type=str,
                        help='Output file for export (default: metrics_export.[csv|json])')

    parser.add_argument('--brief', '-b', action='store_true',
                        help='Brief output (skip detailed breakdown)')

    args = parser.parse_args()

    # Load data
    data = load_metrics(args.file)

    # Check for new data structure
    if 'segmenters' not in data:
        print("Error: Old metrics format detected. Please clear metrics and re-collect data.")
        print("       (The metrics structure has been updated to track by segmenter)")
        sys.exit(1)

    # Handle export
    if args.export:
        output_file = args.output or f"metrics_export.{args.export}"
        if args.export == 'csv':
            export_csv(data, output_file)
        else:
            export_json(data, output_file)
        return 0

    # Display analysis
    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + "PREPROCESSOR METRICS ANALYSIS".center(88) + "║")
    print("╚" + "═" * 88 + "╝")

    # Basic stats
    analyze_basic_stats(data)

    # Segmenter summary
    analyze_segmenter_summary(data, args.segmenter)

    if args.brief:
        # Brief mode - just show recommendations
        analyze_recommendations(data, segmenter=args.segmenter, min_attempts=args.min_attempts)
        return 0

    # Get list of segmenters to analyze
    segmenters = data.get('segmenters', {})
    seg_list = [args.segmenter] if args.segmenter else sorted(segmenters.keys())

    # Context-specific analysis for each segmenter
    for seg_name in seg_list:
        if seg_name not in segmenters:
            continue

        print_header(f"DETAILED ANALYSIS: {seg_name.upper()}")

        if args.context == 'all':
            analyze_context(data, seg_name, 'seg_preprocess', args.sort, args.top, args.min_attempts)

            # CC preprocessing - optionally filtered by seg preprocess
            if args.seg_preprocess_filter:
                analyze_cc_by_seg_preprocess(data, seg_name, args.seg_preprocess_filter, args.min_attempts)
            else:
                analyze_context(data, seg_name, 'cc_preprocess', args.sort, args.top, args.min_attempts)

            analyze_best_combinations(data, seg_name, args.top or 10, args.min_attempts)
        else:
            analyze_context(data, seg_name, args.context, args.sort, args.top, args.min_attempts)
            if args.context == 'combinations':
                analyze_best_combinations(data, seg_name, args.top or 10, args.min_attempts)

        # Confidence distribution
        if not args.brief and args.context in ['all', 'seg_preprocess']:
            analyze_confidence_distribution(data, seg_name, 'seg_preprocess')

    # Recommendations
    analyze_recommendations(data, segmenter=args.segmenter, min_attempts=max(args.min_attempts, 3))

    print("\n" + "=" * 90)
    print(f"  Data source: {args.file}")
    print("=" * 90 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
