"""
Preprocessor Metrics Tracker

Tracks success/failure metrics for preprocessing methods, organized by:
- Segmenter algorithm used
- Segmentation preprocessing method
- Color classification preprocessing method (includes seg preprocessing context)

Data Structure:
    {
        "metadata": {...},
        "segmenters": {
            "contour-neighbor": {
                "seg_preprocess": {
                    "none": {stats},
                    "bilateral": {stats},
                    ...
                },
                "cc_preprocess": {
                    "none|bilateral": {stats},  # cc_method|seg_method
                    "clahe|none": {stats},
                    ...
                },
                "combinations": {
                    "bilateral+clahe": {stats},  # seg_preprocess+cc_preprocess
                    ...
                }
            },
            ...
        }
    }

Usage:
    from PreprocessorMetrics import PreprocessorMetrics, get_metrics

    metrics = get_metrics()
    metrics.record_all_combinations(results, segmenter_name='contour-neighbor')

    # Get summary for a specific segmenter
    summary = metrics.get_summary(segmenter='contour-neighbor', context='seg_preprocess')

    # Print report
    metrics.print_report()
"""

import json
import os
from datetime import datetime
from threading import Lock
from typing import Optional, Dict, List
import numpy as np


# Default metrics from historical data (top 10 combinations per segmenter, excluding auto)
# Used as fallback when metrics file cannot be loaded
DEFAULT_METRICS = {
    'metadata': {
        'created': '2024-12-14T00:00:00',
        'last_updated': None,
        'version': 2,
        'note': 'Default metrics from historical data'
    },
    'segmenters': {
        'brightness-otsu': {
            'seg_preprocess': {},
            'cc_preprocess': {},
            'combinations': {
                'morph-open+bilateral-strong': {'successes': 3, 'failures': 1, 'total_confidence': 21261.85, 'attempts': 4},
                'morph-open+gaussian': {'successes': 3, 'failures': 1, 'total_confidence': 21240.67, 'attempts': 4},
                'morph-open+morph-close': {'successes': 3, 'failures': 1, 'total_confidence': 21252.81, 'attempts': 4},
                'cube-color-mask+bilateral': {'successes': 2, 'failures': 2, 'total_confidence': 21091.23, 'attempts': 4},
                'cube-color-mask+bilateral-strong': {'successes': 2, 'failures': 2, 'total_confidence': 21117.09, 'attempts': 4},
                'cube-color-mask+gaussian': {'successes': 2, 'failures': 2, 'total_confidence': 21118.89, 'attempts': 4},
                'cube-color-mask+median': {'successes': 2, 'failures': 2, 'total_confidence': 21107.44, 'attempts': 4},
                'cube-color-mask+morph-close': {'successes': 2, 'failures': 2, 'total_confidence': 21102.22, 'attempts': 4},
                'cube-color-mask+morph-open': {'successes': 2, 'failures': 2, 'total_confidence': 21128.46, 'attempts': 4},
                'cube-color-mask+none': {'successes': 2, 'failures': 2, 'total_confidence': 21094.11, 'attempts': 4},
            }
        },
        'canny-square': {
            'seg_preprocess': {},
            'cc_preprocess': {},
            'combinations': {
                'clahe-lab+cube-color-mask': {'successes': 1, 'failures': 3, 'total_confidence': 19792.60, 'attempts': 4},
                'clahe-lab+cube-color-mask-soft': {'successes': 1, 'failures': 3, 'total_confidence': 19807.44, 'attempts': 4},
                'contrast-stretch+bilateral': {'successes': 1, 'failures': 3, 'total_confidence': 18842.36, 'attempts': 4},
                'contrast-stretch+bilateral-strong': {'successes': 1, 'failures': 3, 'total_confidence': 18873.60, 'attempts': 4},
                'contrast-stretch+cube-color-mask': {'successes': 1, 'failures': 3, 'total_confidence': 19177.92, 'attempts': 4},
                'contrast-stretch+cube-color-mask-soft': {'successes': 1, 'failures': 3, 'total_confidence': 19226.25, 'attempts': 4},
                'contrast-stretch+gaussian': {'successes': 1, 'failures': 3, 'total_confidence': 18897.91, 'attempts': 4},
                'contrast-stretch+median': {'successes': 1, 'failures': 3, 'total_confidence': 18836.73, 'attempts': 4},
                'contrast-stretch+morph-close': {'successes': 1, 'failures': 3, 'total_confidence': 19052.30, 'attempts': 4},
                'contrast-stretch+morph-open': {'successes': 1, 'failures': 3, 'total_confidence': 18799.97, 'attempts': 4},
            }
        },
        'contour-neighbor': {
            'seg_preprocess': {},
            'cc_preprocess': {},
            'combinations': {
                'clahe-hsv+bilateral': {'successes': 3, 'failures': 1, 'total_confidence': 21168.38, 'attempts': 4},
                'clahe-hsv+bilateral-strong': {'successes': 3, 'failures': 1, 'total_confidence': 21164.23, 'attempts': 4},
                'clahe-hsv+cube-color-mask-soft': {'successes': 3, 'failures': 1, 'total_confidence': 21357.23, 'attempts': 4},
                'clahe-hsv+gaussian': {'successes': 3, 'failures': 1, 'total_confidence': 21167.17, 'attempts': 4},
                'clahe-hsv+median': {'successes': 3, 'failures': 1, 'total_confidence': 21171.24, 'attempts': 4},
                'clahe-hsv+morph-close': {'successes': 3, 'failures': 1, 'total_confidence': 21177.07, 'attempts': 4},
                'clahe-hsv+morph-open': {'successes': 3, 'failures': 1, 'total_confidence': 21167.51, 'attempts': 4},
                'clahe-hsv+none': {'successes': 3, 'failures': 1, 'total_confidence': 21178.46, 'attempts': 4},
                'clahe-hsv+unsharp': {'successes': 3, 'failures': 1, 'total_confidence': 21184.25, 'attempts': 4},
                'cube-color-mask-soft+bilateral': {'successes': 3, 'failures': 1, 'total_confidence': 21452.38, 'attempts': 4},
            }
        },
        'contour-perspective': {
            'seg_preprocess': {},
            'cc_preprocess': {},
            'combinations': {
                'gaussian+bilateral': {'successes': 3, 'failures': 1, 'total_confidence': 20946.55, 'attempts': 4},
                'gaussian+bilateral-strong': {'successes': 3, 'failures': 1, 'total_confidence': 20980.87, 'attempts': 4},
                'gaussian+cube-color-mask': {'successes': 3, 'failures': 1, 'total_confidence': 20981.13, 'attempts': 4},
                'gaussian+cube-color-mask-soft': {'successes': 3, 'failures': 1, 'total_confidence': 21153.31, 'attempts': 4},
                'gaussian+gaussian': {'successes': 3, 'failures': 1, 'total_confidence': 20986.11, 'attempts': 4},
                'gaussian+median': {'successes': 3, 'failures': 1, 'total_confidence': 20928.53, 'attempts': 4},
                'gaussian+morph-close': {'successes': 3, 'failures': 1, 'total_confidence': 20942.13, 'attempts': 4},
                'gaussian+none': {'successes': 3, 'failures': 1, 'total_confidence': 20926.16, 'attempts': 4},
                'bilateral+bilateral': {'successes': 2, 'failures': 2, 'total_confidence': 20910.87, 'attempts': 4},
                'bilateral+bilateral-strong': {'successes': 2, 'failures': 2, 'total_confidence': 20941.39, 'attempts': 4},
            }
        },
        'grid-division': {
            'seg_preprocess': {},
            'cc_preprocess': {},
            'combinations': {
                'cube-color-mask+gaussian': {'successes': 3, 'failures': 2, 'total_confidence': 25734.39, 'attempts': 5},
                'bilateral+bilateral': {'successes': 2, 'failures': 2, 'total_confidence': 20305.92, 'attempts': 4},
                'bilateral+bilateral-strong': {'successes': 2, 'failures': 2, 'total_confidence': 20342.00, 'attempts': 4},
                'bilateral+cube-color-mask': {'successes': 2, 'failures': 2, 'total_confidence': 20361.04, 'attempts': 4},
                'bilateral+cube-color-mask-soft': {'successes': 2, 'failures': 2, 'total_confidence': 20468.43, 'attempts': 4},
                'bilateral+gaussian': {'successes': 2, 'failures': 2, 'total_confidence': 20331.34, 'attempts': 4},
                'bilateral+median': {'successes': 2, 'failures': 2, 'total_confidence': 20278.25, 'attempts': 4},
                'bilateral+morph-close': {'successes': 2, 'failures': 2, 'total_confidence': 20321.96, 'attempts': 4},
                'bilateral+morph-open': {'successes': 2, 'failures': 2, 'total_confidence': 20253.91, 'attempts': 4},
                'bilateral+none': {'successes': 2, 'failures': 2, 'total_confidence': 20213.11, 'attempts': 4},
            }
        },
    }
}


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PreprocessorMetrics:
    """Tracks and persists preprocessor performance metrics by segmenter."""

    DEFAULT_FILE = 'preprocessor_metrics.json'

    def __init__(self, metrics_file: str = None):
        """
        Initialize the metrics tracker.

        Args:
            metrics_file: Path to JSON file for storing metrics.
                         Defaults to 'preprocessor_metrics.json' in current directory.
        """
        self.metrics_file = metrics_file or self.DEFAULT_FILE
        self._lock = Lock()
        self._data = self._load_data()

    def _load_data(self) -> dict:
        """Load existing metrics from file, use defaults on failure."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                # Check for new structure
                if 'segmenters' in data:
                    return data
                # Old structure - start fresh with defaults
                print(f"Note: Old metrics format detected, using default metrics.")
                return self._get_default_metrics()
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse metrics file '{self.metrics_file}': {e}")
                print("         Using default metrics from historical data.")
                return self._get_default_metrics()
            except IOError as e:
                print(f"Warning: Could not read metrics file '{self.metrics_file}': {e}")
                print("         Using default metrics from historical data.")
                return self._get_default_metrics()
            except Exception as e:
                print(f"Warning: Unexpected error loading metrics: {e}")
                print("         Using default metrics from historical data.")
                return self._get_default_metrics()

        # File doesn't exist - use default metrics
        return self._get_default_metrics()

    def _get_default_metrics(self) -> dict:
        """Return a copy of default metrics with current timestamp."""
        import copy
        data = copy.deepcopy(DEFAULT_METRICS)
        data['metadata']['created'] = datetime.now().isoformat()
        return data

    def _save_data(self):
        """Save metrics to file. Warns on failure but continues."""
        self._data['metadata']['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self._data, f, indent=2, cls=NumpyJSONEncoder)
        except IOError as e:
            print(f"Warning: Could not save metrics to '{self.metrics_file}': {e}")
            print("         Metrics will not be persisted for this session.")
        except Exception as e:
            print(f"Warning: Unexpected error saving metrics: {e}")
            print("         Metrics will not be persisted for this session.")

    def _empty_stats(self) -> dict:
        """Return an empty stats entry."""
        return {'successes': 0, 'failures': 0, 'total_confidence': 0.0, 'attempts': 0}

    def _ensure_segmenter(self, segmenter: str):
        """Ensure a segmenter entry exists."""
        if segmenter not in self._data['segmenters']:
            self._data['segmenters'][segmenter] = {
                'seg_preprocess': {},
                'cc_preprocess': {},
                'combinations': {}
            }

    def _update_stats(self, stats: dict, is_valid: bool, confidence: float):
        """Update a stats entry."""
        stats['attempts'] += 1
        stats['total_confidence'] += confidence
        if is_valid:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

    def record_all_combinations(self, results: list, segmenter_name: str = 'unknown'):
        """
        Record metrics for all evaluated preprocessing combinations.

        Args:
            results: List of dicts with keys:
                - seg_method: Segmentation preprocessing method
                - cc_method: Color classification preprocessing method
                - total_confidence: Confidence score
                - is_valid: Whether the result was a valid cube configuration
            segmenter_name: Name of the segmenter algorithm used
        """
        segmenter = segmenter_name.lower()

        with self._lock:
            self._ensure_segmenter(segmenter)
            seg_data = self._data['segmenters'][segmenter]

            for result in results:
                seg_method = (result.get('seg_method') or 'none').lower()
                cc_method = (result.get('cc_method') or 'none').lower()
                confidence = result.get('total_confidence', 0.0)
                is_valid = result.get('is_valid', False)

                # 1. Record segmentation preprocessing stats
                if seg_method not in seg_data['seg_preprocess']:
                    seg_data['seg_preprocess'][seg_method] = self._empty_stats()
                self._update_stats(seg_data['seg_preprocess'][seg_method], is_valid, confidence)

                # 2. Record CC preprocessing stats (keyed by cc_method|seg_method)
                #    This tracks how well each CC method works given a specific seg preprocessing
                cc_key = f"{cc_method}|{seg_method}"
                if cc_key not in seg_data['cc_preprocess']:
                    seg_data['cc_preprocess'][cc_key] = self._empty_stats()
                self._update_stats(seg_data['cc_preprocess'][cc_key], is_valid, confidence)

                # 3. Record full combination stats
                combo_key = f"{seg_method}+{cc_method}"
                if combo_key not in seg_data['combinations']:
                    seg_data['combinations'][combo_key] = self._empty_stats()
                self._update_stats(seg_data['combinations'][combo_key], is_valid, confidence)

            # Single save at the end
            self._save_data()

    def get_segmenters(self) -> List[str]:
        """Get list of segmenters with recorded data."""
        return sorted(self._data['segmenters'].keys())

    def get_summary(self, segmenter: str = None, context: str = 'seg_preprocess',
                    min_attempts: int = 0) -> dict:
        """
        Get summary of methods for a segmenter.

        Args:
            segmenter: Segmenter name, or None to aggregate across all segmenters
            context: 'seg_preprocess', 'cc_preprocess', or 'combinations'
            min_attempts: Only include methods with at least this many attempts

        Returns:
            Dict mapping method names to their stats
        """
        summary = {}

        with self._lock:
            if segmenter:
                # Single segmenter
                if segmenter not in self._data['segmenters']:
                    return {}
                self._aggregate_context(
                    self._data['segmenters'][segmenter].get(context, {}),
                    summary, min_attempts
                )
            else:
                # All segmenters combined
                for seg_name, seg_data in self._data['segmenters'].items():
                    self._aggregate_context(
                        seg_data.get(context, {}),
                        summary, min_attempts
                    )

        return summary

    def _aggregate_context(self, context_data: dict, summary: dict, min_attempts: int):
        """Aggregate context data into summary."""
        for method, stats in context_data.items():
            attempts = stats['attempts']
            if attempts < min_attempts:
                continue

            if method not in summary:
                summary[method] = self._empty_stats()

            summary[method]['attempts'] += attempts
            summary[method]['successes'] += stats['successes']
            summary[method]['failures'] += stats['failures']
            summary[method]['total_confidence'] += stats['total_confidence']

        # Compute rates
        for method, stats in summary.items():
            attempts = stats['attempts']
            stats['success_rate'] = (stats['successes'] / attempts * 100) if attempts > 0 else 0.0
            stats['avg_confidence'] = (stats['total_confidence'] / attempts) if attempts > 0 else 0.0

    def get_cc_summary_by_seg_preprocess(self, segmenter: str = None,
                                          seg_preprocess: str = None,
                                          min_attempts: int = 0) -> dict:
        """
        Get CC preprocessing summary, optionally filtered by segmentation preprocessing.

        Args:
            segmenter: Segmenter name, or None for all
            seg_preprocess: Filter to only this seg preprocessing method, or None for all
            min_attempts: Minimum attempts to include

        Returns:
            Dict mapping CC method names to stats
        """
        summary = {}

        with self._lock:
            segmenters = [segmenter] if segmenter else self._data['segmenters'].keys()

            for seg_name in segmenters:
                if seg_name not in self._data['segmenters']:
                    continue
                cc_data = self._data['segmenters'][seg_name].get('cc_preprocess', {})

                for key, stats in cc_data.items():
                    # Key format: cc_method|seg_method
                    parts = key.split('|')
                    if len(parts) != 2:
                        continue
                    cc_method, seg_method = parts

                    # Filter by seg_preprocess if specified
                    if seg_preprocess and seg_method != seg_preprocess.lower():
                        continue

                    if stats['attempts'] < min_attempts:
                        continue

                    if cc_method not in summary:
                        summary[cc_method] = self._empty_stats()

                    summary[cc_method]['attempts'] += stats['attempts']
                    summary[cc_method]['successes'] += stats['successes']
                    summary[cc_method]['failures'] += stats['failures']
                    summary[cc_method]['total_confidence'] += stats['total_confidence']

            # Compute rates
            for method, stats in summary.items():
                attempts = stats['attempts']
                stats['success_rate'] = (stats['successes'] / attempts * 100) if attempts > 0 else 0.0
                stats['avg_confidence'] = (stats['total_confidence'] / attempts) if attempts > 0 else 0.0

        return summary

    def print_report(self, segmenter: str = None, min_attempts: int = 1):
        """
        Print a formatted report of preprocessor performance.

        Args:
            segmenter: Filter by segmenter or None for all
            min_attempts: Only show methods with at least this many attempts
        """
        print("\n" + "=" * 90)
        print("PREPROCESSOR PERFORMANCE METRICS")
        print("=" * 90)

        metadata = self._data.get('metadata', {})
        if metadata.get('created'):
            print(f"Tracking since: {metadata['created'][:19]}")
        if metadata.get('last_updated'):
            print(f"Last updated: {metadata['last_updated'][:19]}")

        segmenters = [segmenter] if segmenter else self.get_segmenters()

        if not segmenters:
            print("\n  No data recorded yet.")
            print("=" * 90)
            return

        for seg_name in segmenters:
            print(f"\n{'#' * 90}")
            print(f"  SEGMENTER: {seg_name.upper()}")
            print("#" * 90)

            # Segmentation preprocessing
            print(f"\n{'-' * 90}")
            print(f"  Segmentation Preprocessing")
            print("-" * 90)
            self._print_summary_table(
                self.get_summary(segmenter=seg_name, context='seg_preprocess', min_attempts=min_attempts)
            )

            # CC preprocessing (show just CC method, aggregated)
            print(f"\n{'-' * 90}")
            print(f"  Color Classification Preprocessing")
            print("-" * 90)
            cc_summary = self.get_cc_summary_by_seg_preprocess(
                segmenter=seg_name, min_attempts=min_attempts
            )
            self._print_summary_table(cc_summary)

            # Top combinations
            print(f"\n{'-' * 90}")
            print(f"  Top Combinations (seg_preprocess + cc_preprocess)")
            print("-" * 90)
            combo_summary = self.get_summary(
                segmenter=seg_name, context='combinations', min_attempts=min_attempts
            )
            self._print_summary_table(combo_summary, limit=10)

        print("\n" + "=" * 90)

    def _print_summary_table(self, summary: dict, limit: int = None):
        """Print a summary table."""
        if not summary:
            print("  No data recorded yet.")
            return

        print(f"{'Method':<35} {'Attempts':>10} {'Success':>10} {'Rate':>10} {'Avg Conf':>12}")
        print("-" * 90)

        # Sort by success rate (descending), then by attempts (descending)
        sorted_methods = sorted(
            summary.items(),
            key=lambda x: (x[1]['success_rate'], x[1]['attempts']),
            reverse=True
        )

        if limit:
            sorted_methods = sorted_methods[:limit]

        for method, stats in sorted_methods:
            print(f"{method:<35} {stats['attempts']:>10} {stats['successes']:>10} "
                  f"{stats['success_rate']:>9.1f}% {stats['avg_confidence']:>11.1f}")

        print("-" * 90)

        # Summary stats
        total_attempts = sum(s['attempts'] for s in summary.values())
        total_successes = sum(s['successes'] for s in summary.values())
        overall_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
        print(f"{'TOTAL':<35} {total_attempts:>10} {total_successes:>10} {overall_rate:>9.1f}%")

    def get_recommendations(self, segmenter: str = None, context: str = 'seg_preprocess',
                           min_attempts: int = 10) -> dict:
        """
        Get recommendations based on collected metrics.

        Args:
            segmenter: Segmenter name, or None for all
            context: 'seg_preprocess', 'cc_preprocess', or 'combinations'
            min_attempts: Minimum attempts to consider method reliable

        Returns:
            Dict with 'best', 'worst', 'consider_removing' lists
        """
        summary = self.get_summary(segmenter=segmenter, context=context, min_attempts=min_attempts)

        if not summary:
            return {'best': [], 'worst': [], 'consider_removing': []}

        sorted_methods = sorted(
            summary.items(),
            key=lambda x: (x[1]['success_rate'], x[1]['avg_confidence']),
            reverse=True
        )

        best = [m[0] for m in sorted_methods[:3]]
        worst = [m[0] for m in sorted_methods[-3:] if m[1]['success_rate'] < 50]

        consider_removing = [
            m[0] for m in sorted_methods
            if m[1]['success_rate'] < 20 and m[1]['attempts'] >= min_attempts
        ]

        return {
            'best': best,
            'worst': worst,
            'consider_removing': consider_removing
        }

    def clear_data(self):
        """Clear all recorded metrics."""
        with self._lock:
            self._data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'last_updated': None,
                    'version': 2
                },
                'segmenters': {}
            }
            self._save_data()


# Global instance for easy access
_metrics_instance = None


def get_metrics() -> PreprocessorMetrics:
    """Get or create the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PreprocessorMetrics()
    return _metrics_instance


if __name__ == '__main__':
    # Demo/test
    import random

    metrics = PreprocessorMetrics('test_metrics.json')

    # Simulate results for different segmenters
    segmenters = ['contour-neighbor', 'brightness-otsu', 'grid-division']
    preprocess_methods = ['none', 'clahe', 'bilateral', 'sharpen', 'histeq']

    for _ in range(100):
        segmenter = random.choice(segmenters)

        # Generate fake results
        results = []
        for seg_m in random.sample(preprocess_methods, 3):
            for cc_m in random.sample(preprocess_methods, 3):
                success = random.random() > 0.3
                confidence = random.uniform(4000, 5400) if success else random.uniform(2000, 4000)
                results.append({
                    'seg_method': seg_m,
                    'cc_method': cc_m,
                    'total_confidence': confidence,
                    'is_valid': success
                })

        metrics.record_all_combinations(results, segmenter_name=segmenter)

    metrics.print_report()

    print("\nRecommendations by segmenter:")
    for seg in segmenters:
        recs = metrics.get_recommendations(segmenter=seg, context='seg_preprocess', min_attempts=5)
        print(f"  {seg}: best={recs['best']}")

    # Cleanup test file
    os.remove('test_metrics.json')
