"""
Adaptive Cube Evaluator

Uses historical metrics data to intelligently select preprocessing combinations.
Requires TWO IDENTICAL valid results for confirmation, interleaves across
segmenters to ensure niche performers get tried early.

Strategy:
1. Load historical metrics and rank combinations per segmenter
2. Interleave: Round N tries the Nth best combo for EACH segmenter
3. Require two identical valid results to confirm (catches false positives)
4. After finding valid result, do background random sampling for data collection
5. Return the confirmed valid result

Usage:
    from adaptive_evaluator import AdaptiveEvaluator

    evaluator = AdaptiveEvaluator()
    result = evaluator.evaluate(face_images, classifier, preprocessor)

    if result['is_valid']:
        cube_data = result['cube_data']
"""

import json
import os
import random
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Debug flag - set via environment variable
DEBUG = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')
LOG_DIR = "log"
_debug_log_file = None

def debug_print(msg):
    """Write debug message to log file if DEBUG is enabled."""
    global _debug_log_file
    if DEBUG:
        os.makedirs(LOG_DIR, exist_ok=True)
        if _debug_log_file is None:
            _debug_log_file = open(os.path.join(LOG_DIR, "debug.log"), "a")
        _debug_log_file.write(f"[DEBUG] {msg}\n")
        _debug_log_file.flush()

from Segmenter import Segmenter
from cube_evaluation import (
    evaluate_preprocessing_combination_batch,
    _preprocess_facelets_vectorized,
    FACE_NAMES
)
from PreprocessorMetrics import get_metrics


@dataclass
class RankedCombination:
    """A preprocessing combination with its historical performance."""
    segmenter_name: str
    seg_preprocess: str
    cc_preprocess: str
    success_rate: float
    avg_confidence: float
    attempts: int

    @property
    def score(self) -> float:
        """Combined score for ranking (success rate primary, confidence secondary)."""
        # Weight success rate heavily, use confidence as tiebreaker
        # Also give slight boost to combinations with more attempts (more reliable data)
        reliability = min(1.0, self.attempts / 10)  # Cap at 10 attempts
        return self.success_rate * 100 + (self.avg_confidence / 100) + reliability


def cube_data_matches(data1: Dict, data2: Dict) -> bool:
    """
    Check if two cube data results are identical.

    Args:
        data1: First cube data dict (face_name -> list of colors)
        data2: Second cube data dict

    Returns:
        True if all facelets match
    """
    if data1 is None or data2 is None:
        return False

    for face_name in FACE_NAMES:
        if face_name not in data1 or face_name not in data2:
            return False
        if data1[face_name] != data2[face_name]:
            return False

    return True


class AdaptiveEvaluator:
    """
    Evaluates cube faces using an adaptive strategy based on historical metrics.

    Key features:
    - Requires TWO IDENTICAL valid results for confirmation
    - Interleaves across segmenters so all get tried early
    - Background random sampling for continued data collection
    """

    def __init__(self, metrics_file: str = 'preprocessor_metrics.json'):
        """
        Initialize the adaptive evaluator.

        Args:
            metrics_file: Path to the metrics JSON file
        """
        self.metrics_file = metrics_file
        self.ranked_by_segmenter: Dict[str, List[RankedCombination]] = {}
        self.all_segmenters: List[str] = []
        self._segmenter_cache: Dict[str, Any] = {}
        self._background_thread: Optional[threading.Thread] = None
        self._background_results: List[Dict] = []

    def _load_metrics(self) -> Dict:
        """Load metrics data from file."""
        if not os.path.exists(self.metrics_file):
            return {}

        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _build_rankings_by_segmenter(self, metrics_data: Dict,
                                      preprocessor) -> None:
        """
        Build ranked lists of combinations organized by segmenter.

        This enables interleaved evaluation where we try the best combo
        from each segmenter before trying the second-best from any.
        """
        self.ranked_by_segmenter = defaultdict(list)
        self.all_segmenters = Segmenter.get_available_segmenters()

        segmenters_data = metrics_data.get('segmenters', {})
        methods = preprocessor.get_available_methods()

        for seg_name in self.all_segmenters:
            seg_data = segmenters_data.get(seg_name, {})
            combinations = seg_data.get('combinations', {})

            # Build combinations for this segmenter
            seg_combos = []
            seen_combos = set()

            # First, add combinations we have data for
            for combo_key, combo_data in combinations.items():
                attempts = combo_data.get('attempts', 0)
                if attempts == 0:
                    continue

                successes = combo_data.get('successes', 0)
                total_conf = combo_data.get('total_confidence', 0)

                # Parse combination key (format: "seg_preprocess+cc_preprocess")
                parts = combo_key.split('+', 1)
                seg_pp = parts[0] if len(parts) > 0 else 'none'
                cc_pp = parts[1] if len(parts) > 1 else 'none'

                ranked = RankedCombination(
                    segmenter_name=seg_name,
                    seg_preprocess=seg_pp,
                    cc_preprocess=cc_pp,
                    success_rate=successes / attempts if attempts > 0 else 0,
                    avg_confidence=total_conf / attempts if attempts > 0 else 0,
                    attempts=attempts
                )
                seg_combos.append(ranked)
                seen_combos.add((seg_pp, cc_pp))

            # Add untried combinations with zero score
            # For brightness-otsu, only use 'none' for seg preprocessing (Otsu is adaptive)
            seg_pp_methods = ['none'] if seg_name == 'brightness-otsu' else methods
            for seg_pp in seg_pp_methods:
                for cc_pp in methods:
                    if (seg_pp, cc_pp) not in seen_combos:
                        ranked = RankedCombination(
                            segmenter_name=seg_name,
                            seg_preprocess=seg_pp,
                            cc_preprocess=cc_pp,
                            success_rate=0,
                            avg_confidence=0,
                            attempts=0
                        )
                        seg_combos.append(ranked)

            # Sort by score (descending)
            seg_combos.sort(key=lambda x: x.score, reverse=True)

            # Move locked pairs to the front (in order)
            locked_pairs = metrics_data.get('locked_pairs', {}).get(seg_name, [])
            if locked_pairs:
                locked_combos = []
                unlocked_combos = []
                locked_set = {(p[0], p[1]) for p in locked_pairs}

                for combo in seg_combos:
                    if (combo.seg_preprocess, combo.cc_preprocess) in locked_set:
                        locked_combos.append(combo)
                    else:
                        unlocked_combos.append(combo)

                # Sort locked combos by their order in locked_pairs
                locked_order = {(p[0], p[1]): i for i, p in enumerate(locked_pairs)}
                locked_combos.sort(key=lambda x: locked_order.get(
                    (x.seg_preprocess, x.cc_preprocess), 999))

                seg_combos = locked_combos + unlocked_combos

            self.ranked_by_segmenter[seg_name] = seg_combos

    def _get_segmenter(self, name: str) -> Any:
        """Get or create a segmenter by name (cached)."""
        if name not in self._segmenter_cache:
            self._segmenter_cache[name] = Segmenter.create(name, debug=False)
        return self._segmenter_cache[name]

    def _evaluate_combination(self, face_images: Dict, segmenter_name: str,
                               seg_preprocess: str, cc_preprocess: str,
                               classifier, preprocessor,
                               force_centers: bool = False) -> Dict:
        """
        Evaluate a single combination.

        Returns:
            Dict with evaluation results
        """
        eval_start = time.time()
        segmenter = self._get_segmenter(segmenter_name)

        # Stage 1: Segment all faces with preprocessing
        # Skip seg preprocessing for brightness-otsu (Otsu is adaptive)
        skip_seg_preprocess = segmenter_name == 'brightness-otsu'
        facelets_by_face = {}
        seg_total_time = 0
        for face_name, image in face_images.items():
            face_start = time.time()

            # Apply segmentation preprocessing (unless brightness-otsu)
            if not skip_seg_preprocess and seg_preprocess and seg_preprocess.lower() != 'none':
                preproc_start = time.time()
                processed = preprocessor.apply(seg_preprocess, image)
                preproc_time = time.time() - preproc_start
            else:
                processed = image
                preproc_time = 0

            # Segment
            segment_start = time.time()
            facelets = segmenter.segment(processed)
            segment_time = time.time() - segment_start

            face_total = time.time() - face_start
            seg_total_time += face_total
            debug_print(f"  {segmenter_name}/{face_name}: preproc={preproc_time*1000:.1f}ms, seg={segment_time*1000:.1f}ms")

            facelets_by_face[face_name] = facelets

        # Stage 2: Apply CC preprocessing and classify
        cc_start = time.time()
        preprocessed_facelets = {}
        for face_name, facelets in facelets_by_face.items():
            preprocessed_facelets[face_name] = _preprocess_facelets_vectorized(
                facelets, preprocessor, cc_preprocess
            )
        cc_time = time.time() - cc_start

        # Stage 3: Classify
        classify_start = time.time()
        cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination_batch(
            preprocessed_facelets, classifier, force_centers
        )
        classify_time = time.time() - classify_start

        eval_total = time.time() - eval_start
        debug_print(f"  {segmenter_name} total: seg={seg_total_time*1000:.1f}ms, cc_preproc={cc_time*1000:.1f}ms, classify={classify_time*1000:.1f}ms, total={eval_total*1000:.1f}ms")

        return {
            'segmenter_name': segmenter_name,
            'seg_method': seg_preprocess,
            'cc_method': cc_preprocess,
            'cube_data': cube_data,
            'confidence_scores': conf_scores,
            'is_valid': is_valid,
            'total_confidence': total_conf,
            'details': details
        }

    def _generate_interleaved_order(self, combos_per_round: int = 5) -> List[RankedCombination]:
        """
        Generate interleaved evaluation order.

        Round N tries the Nth best combos from EACH segmenter.
        This ensures niche performers (like grid-division) get tried early.

        Args:
            combos_per_round: Number of combinations to try per segmenter per round

        Returns:
            List of combinations in interleaved order
        """
        interleaved = []

        # Find the max number of combinations across all segmenters
        max_combos = max(len(combos) for combos in self.ranked_by_segmenter.values())

        # Interleave: round-robin through segmenters, N combos at a time
        for round_start in range(0, max_combos, combos_per_round):
            for seg_name in self.all_segmenters:
                combos = self.ranked_by_segmenter.get(seg_name, [])
                round_end = min(round_start + combos_per_round, len(combos))
                for i in range(round_start, round_end):
                    if i < len(combos):
                        interleaved.append(combos[i])

        return interleaved

    def _run_background_sampling(self, face_images: Dict, classifier, preprocessor,
                                  tried_combinations: set, num_samples: int,
                                  force_centers: bool):
        """
        Run random sampling in background thread for data collection.

        Results are stored in self._background_results for later retrieval.
        """
        self._background_results = []

        # Get all possible combinations
        all_combos = []
        methods = preprocessor.get_available_methods()
        for seg_name in self.all_segmenters:
            # For brightness-otsu, only use 'none' for seg preprocessing
            seg_pp_methods = ['none'] if seg_name == 'brightness-otsu' else methods
            for seg_pp in seg_pp_methods:
                for cc_pp in methods:
                    combo_key = (seg_name, seg_pp, cc_pp)
                    if combo_key not in tried_combinations:
                        all_combos.append(combo_key)

        if not all_combos:
            return

        # Sample random combinations
        samples = random.sample(all_combos, min(num_samples, len(all_combos)))

        for seg_name, seg_pp, cc_pp in samples:
            try:
                result = self._evaluate_combination(
                    face_images, seg_name, seg_pp, cc_pp,
                    classifier, preprocessor, force_centers
                )
                self._background_results.append(result)
            except Exception:
                continue

        # Record all background results to metrics at once
        if self._background_results:
            metrics = get_metrics()
            for result in self._background_results:
                metrics.record_all_combinations([result], segmenter_name=result['segmenter_name'])

    def evaluate(self, face_images: Dict, classifier, preprocessor,
                 max_attempts: int = 100,
                 combos_per_round: int = 5,
                 background_samples: int = 10,
                 fallback_after: int = 30,
                 force_centers: bool = False,
                 record_metrics: bool = True,
                 verbose: bool = True) -> Dict:
        """
        Adaptively evaluate cube faces using historical metrics.

        Strategy:
        1. Interleave across segmenters (Round N tries combos N*5 to N*5+4 for each)
        2. Require TWO IDENTICAL valid results for confirmation
        3. If can't confirm after fallback_after attempts, accept single valid result
        4. Start background random sampling for data collection
        5. Return the result

        Args:
            face_images: Dict of face_name -> image (BGR)
            classifier: FaceletColorClassifier instance
            preprocessor: ImagePreprocessor instance
            max_attempts: Maximum number of combinations to try before giving up
            combos_per_round: Number of combinations per segmenter per round
            background_samples: Number of random samples for background data collection
            fallback_after: Accept unconfirmed result after this many attempts post first valid
            force_centers: If True, require center colors to match expected
            record_metrics: If True, record results to metrics file
            verbose: If True, print progress information

        Returns:
            Dict with:
                - is_valid: bool (True if confirmed OR fallback accepted)
                - cube_data: Dict of face colors (if valid)
                - confidence_scores: Dict of confidence scores
                - segmenter_name: Name of segmenter used
                - seg_method: Segmentation preprocessing method
                - cc_method: Color classification preprocessing method
                - attempts_to_confirm: Number of attempts to get confirmed result
                - confirmation_method: "two_identical", "fallback", or None
                - total_attempts: Total attempts including after confirmation
                - all_results: List of all evaluation results
        """
        start_time = time.time()

        # Load metrics and build rankings
        metrics_data = self._load_metrics()
        self._build_rankings_by_segmenter(metrics_data, preprocessor)

        # Generate interleaved evaluation order
        interleaved_order = self._generate_interleaved_order(combos_per_round)

        total_combos = sum(len(c) for c in self.ranked_by_segmenter.values())

        if verbose:
            print(f"\nAdaptive evaluation: searching for confirmed result...", end='', flush=True)

        debug_print(f"{len(self.all_segmenters)} segmenters, {total_combos} total combinations")
        debug_print(f"Max attempts: {max_attempts}, combos per round per segmenter: {combos_per_round}")
        debug_print(f"Will accept unconfirmed after {fallback_after} attempts if needed")

        all_results = []
        valid_results = []  # Store all valid results to find matching pair
        confirmed_result = None
        fallback_result = None  # Best unconfirmed result to use as fallback
        attempts_to_confirm = 0
        first_valid_at = None  # Track when we found first valid result

        # Track which combinations we've tried
        tried_combinations = set()

        current_segmenter = None

        for i, combo in enumerate(interleaved_order):
            if i >= max_attempts:
                debug_print(f"Reached max attempts ({max_attempts})")
                break

            combo_key = (combo.segmenter_name, combo.seg_preprocess, combo.cc_preprocess)
            if combo_key in tried_combinations:
                continue
            tried_combinations.add(combo_key)

            # Show segmenter change
            if combo.segmenter_name != current_segmenter:
                current_segmenter = combo.segmenter_name
                valid_count = len(valid_results)
                debug_print(f"--- {combo.segmenter_name} --- (valid so far: {valid_count})")

            attempt_num = len(all_results) + 1
            rate_str = f"{combo.success_rate*100:.0f}%" if combo.attempts > 0 else "new"
            # Log attempt start (result will be logged separately after evaluation)
            debug_print(f"[{attempt_num}] {combo.segmenter_name} seg={combo.seg_preprocess}, cc={combo.cc_preprocess} ({rate_str})")

            try:
                result = self._evaluate_combination(
                    face_images, combo.segmenter_name,
                    combo.seg_preprocess, combo.cc_preprocess,
                    classifier, preprocessor, force_centers
                )
                all_results.append(result)

                # Print progress dot after each evaluation
                if verbose and not DEBUG:
                    print('.', end='', flush=True)

                status = "VALID" if result['is_valid'] else "invalid"
                debug_print(f"  -> {status} (conf: {result['total_confidence']:.0f})")

                if result['is_valid']:
                    # Track when we found first valid result
                    if first_valid_at is None:
                        first_valid_at = len(all_results)
                        fallback_result = result  # First valid is our fallback candidate

                    # Check if this matches any previous valid result
                    for prev_result in valid_results:
                        if cube_data_matches(result['cube_data'], prev_result['cube_data']):
                            # Found matching pair - confirmed!
                            confirmed_result = result
                            attempts_to_confirm = len(all_results)
                            debug_print(f"CONFIRMED! Two identical valid results found!")
                            debug_print(f"  Match: {prev_result['segmenter_name']} | seg={prev_result['seg_method']} | cc={prev_result['cc_method']}")
                            debug_print(f"  With:  {result['segmenter_name']} | seg={result['seg_method']} | cc={result['cc_method']}")
                            break

                    valid_results.append(result)

                    # Update fallback to highest confidence valid result
                    if result['total_confidence'] > fallback_result['total_confidence']:
                        fallback_result = result

                    if confirmed_result:
                        break

                # Check if we should accept fallback (have valid but can't confirm)
                if first_valid_at is not None and not confirmed_result:
                    attempts_since_first = len(all_results) - first_valid_at
                    if attempts_since_first >= fallback_after:
                        debug_print(f"FALLBACK: Accepting unconfirmed result after {attempts_since_first} attempts")
                        debug_print(f"  Using: {fallback_result['segmenter_name']} | seg={fallback_result['seg_method']} | cc={fallback_result['cc_method']}")
                        break

            except Exception as e:
                debug_print(f" ERROR: {e}")
                continue

        # Record all results to metrics (grouped by segmenter for efficiency)
        if record_metrics and all_results:
            metrics = get_metrics()
            # Group results by segmenter
            by_segmenter = {}
            for result in all_results:
                seg_name = result['segmenter_name']
                if seg_name not in by_segmenter:
                    by_segmenter[seg_name] = []
                by_segmenter[seg_name].append(result)
            # Record each group
            for seg_name, results in by_segmenter.items():
                metrics.record_all_combinations(results, segmenter_name=seg_name)

        # Determine final result (confirmed or fallback)
        final_result = confirmed_result or fallback_result
        use_fallback = final_result is not None and confirmed_result is None

        # Start background sampling for data collection (non-blocking)
        if background_samples > 0 and final_result:
            debug_print(f"  Starting background sampling ({background_samples} combinations)...")

            self._background_thread = threading.Thread(
                target=self._run_background_sampling,
                args=(face_images, classifier, preprocessor,
                      tried_combinations, background_samples, force_centers),
                daemon=True
            )
            self._background_thread.start()

        elapsed = time.time() - start_time

        # Debug details
        debug_print(f"Completed in {elapsed:.1f}s: {len(all_results)} combinations evaluated")
        debug_print(f"Valid results found: {len(valid_results)}")

        # Simple user-facing output
        if verbose:
            if confirmed_result:
                print(f" confirmed in {elapsed:.1f}s ({attempts_to_confirm} attempts)")
                debug_print(f"Result: {confirmed_result['segmenter_name']} | seg={confirmed_result['seg_method']} | cc={confirmed_result['cc_method']}")
            elif use_fallback:
                print(f" fallback in {elapsed:.1f}s (could not confirm)")
                debug_print(f"Result: {fallback_result['segmenter_name']} | seg={fallback_result['seg_method']} | cc={fallback_result['cc_method']}")
            elif valid_results:
                print(f"\n  WARNING: {len(valid_results)} valid results but could not confirm")
            else:
                print("\n  No valid result found")

        # Build return value
        if confirmed_result:
            return {
                'is_valid': True,
                'cube_data': confirmed_result['cube_data'],
                'confidence_scores': confirmed_result['confidence_scores'],
                'total_confidence': confirmed_result['total_confidence'],
                'segmenter_name': confirmed_result['segmenter_name'],
                'seg_method': confirmed_result['seg_method'],
                'cc_method': confirmed_result['cc_method'],
                'attempts_to_confirm': attempts_to_confirm,
                'confirmation_method': 'two_identical',
                'valid_results_count': len(valid_results),
                'total_attempts': len(all_results),
                'elapsed_seconds': elapsed,
                'all_results': all_results
            }
        elif use_fallback:
            # Fallback: accepted unconfirmed result after threshold
            return {
                'is_valid': True,  # Accept the fallback as valid
                'cube_data': fallback_result['cube_data'],
                'confidence_scores': fallback_result['confidence_scores'],
                'total_confidence': fallback_result['total_confidence'],
                'segmenter_name': fallback_result['segmenter_name'],
                'seg_method': fallback_result['seg_method'],
                'cc_method': fallback_result['cc_method'],
                'attempts_to_confirm': len(all_results) - first_valid_at,
                'confirmation_method': 'fallback',
                'valid_results_count': len(valid_results),
                'total_attempts': len(all_results),
                'elapsed_seconds': elapsed,
                'all_results': all_results
            }
        elif valid_results:
            # Have valid results but didn't reach fallback threshold
            best_valid = max(valid_results, key=lambda r: r['total_confidence'])
            return {
                'is_valid': False,  # Not confirmed and not fallback
                'cube_data': best_valid['cube_data'],
                'confidence_scores': best_valid['confidence_scores'],
                'total_confidence': best_valid['total_confidence'],
                'segmenter_name': best_valid['segmenter_name'],
                'seg_method': best_valid['seg_method'],
                'cc_method': best_valid['cc_method'],
                'attempts_to_confirm': 0,
                'confirmation_method': None,
                'valid_results_count': len(valid_results),
                'unconfirmed_reason': 'max_attempts_before_fallback',
                'total_attempts': len(all_results),
                'elapsed_seconds': elapsed,
                'all_results': all_results
            }
        else:
            # No valid results at all
            best_invalid = max(all_results, key=lambda r: r['total_confidence']) if all_results else None
            return {
                'is_valid': False,
                'cube_data': best_invalid['cube_data'] if best_invalid else None,
                'confidence_scores': best_invalid['confidence_scores'] if best_invalid else None,
                'total_confidence': best_invalid['total_confidence'] if best_invalid else 0,
                'segmenter_name': best_invalid['segmenter_name'] if best_invalid else None,
                'seg_method': best_invalid['seg_method'] if best_invalid else None,
                'cc_method': best_invalid['cc_method'] if best_invalid else None,
                'attempts_to_confirm': 0,
                'confirmation_method': None,
                'valid_results_count': 0,
                'total_attempts': len(all_results),
                'elapsed_seconds': elapsed,
                'all_results': all_results
            }

    def wait_for_background_sampling(self, timeout: float = 60.0) -> List[Dict]:
        """
        Wait for background sampling to complete and return results.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            List of background sampling results
        """
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=timeout)
        return self._background_results

    def get_rankings_by_segmenter(self, n_per_segmenter: int = 10) -> Dict[str, List[RankedCombination]]:
        """Get top N ranked combinations for each segmenter."""
        metrics_data = self._load_metrics()
        # Need a preprocessor to build rankings
        from ImagePreprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        self._build_rankings_by_segmenter(metrics_data, preprocessor)

        return {
            seg: combos[:n_per_segmenter]
            for seg, combos in self.ranked_by_segmenter.items()
        }

    def print_rankings(self, n_per_segmenter: int = 5):
        """Print top ranked combinations for each segmenter."""
        rankings = self.get_rankings_by_segmenter(n_per_segmenter)

        print(f"\nTop {n_per_segmenter} Combinations per Segmenter:")
        print("=" * 90)

        for seg_name in self.all_segmenters:
            combos = rankings.get(seg_name, [])
            if not combos:
                continue

            print(f"\n{seg_name}:")
            print("-" * 80)

            for i, combo in enumerate(combos, 1):
                rate = f"{combo.success_rate*100:.0f}%" if combo.attempts > 0 else "new"
                attempts = f"({combo.attempts} attempts)" if combo.attempts > 0 else ""
                print(f"  {i}. seg={combo.seg_preprocess:<20} cc={combo.cc_preprocess:<20} {rate:>5} {attempts}")


# Convenience function
def adaptive_evaluate(face_images, classifier, preprocessor, **kwargs):
    """
    Convenience function for adaptive evaluation.

    See AdaptiveEvaluator.evaluate() for full documentation.
    """
    evaluator = AdaptiveEvaluator()
    return evaluator.evaluate(face_images, classifier, preprocessor, **kwargs)


if __name__ == '__main__':
    # Demo/test
    import cv2
    from FaceletColorClassifier import FaceletColorClassifier
    from ImagePreprocessor import ImagePreprocessor
    from cube_evaluation import load_face_images

    print("Adaptive Evaluator Demo (Two-Result Confirmation)")
    print("=" * 60)

    # Show current rankings
    evaluator = AdaptiveEvaluator()
    evaluator.print_rankings(5)

    # Test on an image set if available
    test_dirs = [
        'input_faces',
        'input_faces/Black Background',
    ]

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        print(f"\n{'=' * 60}")
        print(f"Testing on: {test_dir}")
        print('=' * 60)

        face_images = load_face_images(test_dir)
        if face_images is None:
            print("  Could not load face images")
            continue

        # Load classifier and preprocessor
        try:
            classifier = FaceletColorClassifier()
            preprocessor = ImagePreprocessor()
        except Exception as e:
            print(f"  Could not load classifier: {e}")
            continue

        # Run adaptive evaluation
        result = evaluator.evaluate(
            face_images, classifier, preprocessor,
            max_attempts=100,
            combos_per_round=5,
            background_samples=10,
            verbose=True
        )

        print(f"\nResult: {'CONFIRMED VALID' if result['is_valid'] else 'NOT CONFIRMED'}")
        if result['is_valid']:
            print(f"  Confirmed after {result['attempts_to_confirm']} attempts")
            print(f"  Total confidence: {result['total_confidence']:.0f}")
        elif result.get('valid_results_count', 0) > 0:
            print(f"  Had {result['valid_results_count']} valid results but no matching pair")

        # Wait for background sampling
        bg_results = evaluator.wait_for_background_sampling(timeout=30)
        if bg_results:
            valid_bg = sum(1 for r in bg_results if r['is_valid'])
            print(f"\nBackground sampling: {len(bg_results)} results ({valid_bg} valid)")

        break  # Just test one directory
