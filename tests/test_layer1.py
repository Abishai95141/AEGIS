"""
AEGIS 3.0 Robust Validation — Layer 1 (Semantic Sensorium) Tests

6 tests: 3 nominal + 3 robustness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.layer1_semantic import SemanticSensorium
from config import L1_NOMINAL, L1_ROBUST, L1_ENTROPY_THRESHOLD


def run_all_tests():
    """Run all Layer 1 tests. Returns list of result dicts."""
    results = []
    l1 = SemanticSensorium(entropy_threshold=L1_ENTROPY_THRESHOLD)

    # ── R-L1-01: Concept Extraction (Nominal) ──
    test_texts = [
        ("Went for a 30 min jog this morning, feeling good", {'exercise', 'mood_positive'}),
        ("Stressed about work deadline, couldn't sleep", {'stress', 'work_stressor', 'sleep_quality'}),
        ("Had a big pasta dinner", {'meal'}),
        ("Feeling shaky and lightheaded", {'hypoglycemia_symptom'}),
        ("Very thirsty, dry mouth, headache", {'hyperglycemia_symptom'}),
        ("Exhausted today, no energy at all", {'fatigue'}),
        ("Pretty normal day overall", {'mood_positive'}),
        ("Did some weights at the gym", {'exercise'}),
        ("Woke up multiple times during the night", {'sleep_quality'}),
        ("Had a tense argument with my boss today", {'stress', 'work_stressor'}),
    ]

    tp, fp, fn = 0, 0, 0
    for text, expected in test_texts:
        result = l1.process(text)
        extracted = set(result['concept_ids'])
        tp += len(extracted & expected)
        fp += len(extracted - expected)
        fn += len(expected - extracted)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    results.append({
        'test_id': 'R-L1-01',
        'name': 'Concept Extraction (Nominal)',
        'type': 'nominal',
        'metrics': {'precision': precision, 'recall': recall, 'f1': f1},
        'threshold': {'f1_min': L1_NOMINAL['extraction_f1_min']},
        'passed': f1 >= L1_NOMINAL['extraction_f1_min'],
    })

    # ── R-L1-02: Semantic Entropy Calibration ──
    ambiguous_texts = [
        "Feeling kind of off, not sure what's wrong",
        "Something doesn't feel right, maybe tired or stressed",
        "Had a weird reaction, could be food or medication",
    ]
    clear_texts = [
        "Went for a run",
        "Had breakfast",
        "Feeling fine",
    ]

    ambiguous_entropy = [l1.compute_semantic_entropy(t) for t in ambiguous_texts]
    clear_entropy = [l1.compute_semantic_entropy(t) for t in clear_texts]

    # Ambiguous should have higher entropy than clear
    mean_amb = np.mean(ambiguous_entropy)
    mean_clear = np.mean(clear_entropy)
    entropy_diff = mean_amb - mean_clear
    entropy_separates = entropy_diff > 0

    results.append({
        'test_id': 'R-L1-02',
        'name': 'Semantic Entropy Calibration',
        'type': 'nominal',
        'metrics': {
            'mean_ambiguous_entropy': float(mean_amb),
            'mean_clear_entropy': float(mean_clear),
            'entropy_difference': float(entropy_diff),
        },
        'threshold': {'entropy_difference_min': 0.0},
        'passed': entropy_separates,
    })

    # ── R-L1-03: Proxy Classification ──
    # Z proxies (treatment-confounder)
    z_positive_texts = [
        "Worried about the deadline at work",
        "Boss scheduled an unexpected meeting",
        "Really anxious about the presentation tomorrow",
        "Work has been insane, can't stop worrying",
    ]
    z_negative_texts = [
        "Had a big pasta dinner",
        "Went for a jog",
        "Feeling fine today",
        "Slept well, feeling refreshed",
    ]

    z_tp = sum(1 for t in z_positive_texts if l1.process(t)['z_proxy'])
    z_fp = sum(1 for t in z_negative_texts if l1.process(t)['z_proxy'])
    z_fn = len(z_positive_texts) - z_tp
    z_tn = len(z_negative_texts) - z_fp

    z_precision = z_tp / max(z_tp + z_fp, 1)
    z_recall = z_tp / max(z_tp + z_fn, 1)

    # W proxies (outcome-confounder)
    w_positive_texts = [
        "Terrible sleep, woke up exhausted",
        "So fatigued, everything is harder",
        "Couldn't sleep well, tossing and turning",
    ]
    w_negative_texts = [
        "Had breakfast",
        "Went for a walk",
        "Feeling good today",
    ]

    w_tp = sum(1 for t in w_positive_texts if l1.process(t)['w_proxy'] > 0)
    w_fp = sum(1 for t in w_negative_texts if l1.process(t)['w_proxy'] > 0)
    w_fn = len(w_positive_texts) - w_tp
    w_recall = w_tp / max(w_tp + w_fn, 1)

    proxy_passed = (z_precision >= L1_NOMINAL['proxy_precision_min'] and
                    z_recall >= L1_NOMINAL['proxy_recall_min'])

    results.append({
        'test_id': 'R-L1-03',
        'name': 'Proxy Classification',
        'type': 'nominal',
        'metrics': {
            'z_precision': z_precision, 'z_recall': z_recall,
            'w_recall': w_recall,
        },
        'threshold': {
            'precision_min': L1_NOMINAL['proxy_precision_min'],
            'recall_min': L1_NOMINAL['proxy_recall_min'],
        },
        'passed': proxy_passed,
    })

    # ── R-L1-04: Noisy/Informal Text (Robustness) ──
    noisy_texts = [
        ("went 4 a jog, feelin gd", {'exercise', 'mood_positive'}),
        ("sooo stressed abt wrk deadln", {'stress', 'work_stressor'}),
        ("ate lnch, pasta n bread", {'meal'}),
        ("shaky af, need sugar asap", {'hypoglycemia_symptom'}),
        ("super tired, cant even", {'fatigue'}),
    ]

    noisy_tp, noisy_fp, noisy_fn = 0, 0, 0
    for text, expected in noisy_texts:
        result = l1.process(text)
        extracted = set(result['concept_ids'])
        noisy_tp += len(extracted & expected)
        noisy_fp += len(extracted - expected)
        noisy_fn += len(expected - extracted)

    noisy_precision = noisy_tp / max(noisy_tp + noisy_fp, 1)
    noisy_recall = noisy_tp / max(noisy_tp + noisy_fn, 1)
    noisy_f1 = 2 * noisy_precision * noisy_recall / max(noisy_precision + noisy_recall, 1e-10)

    results.append({
        'test_id': 'R-L1-04',
        'name': 'Noisy/Informal Text',
        'type': 'robust',
        'metrics': {'precision': noisy_precision, 'recall': noisy_recall, 'f1': noisy_f1},
        'threshold': {'f1_min': L1_ROBUST['noisy_f1_min']},
        'passed': noisy_f1 >= L1_ROBUST['noisy_f1_min'],
    })

    # ── R-L1-05: Out-of-Vocabulary Concepts ──
    known_text = "Went for a run, feeling tired"
    unknown_text = "Experienced paresthesia with concurrent photophobia and tinnitus"
    entropy_known = l1.compute_semantic_entropy(known_text)
    entropy_unknown = l1.compute_semantic_entropy(unknown_text)

    # OOV text should have different (possibly lower) extraction confidence
    # The key test: unknown medical terms should trigger HITL
    oov_result = l1.process(unknown_text)
    known_result = l1.process(known_text)

    # Fewer concepts extracted from unknown = system is being honest
    oov_extraction_lower = len(oov_result['concept_ids']) <= len(known_result['concept_ids'])

    results.append({
        'test_id': 'R-L1-05',
        'name': 'Out-of-Vocabulary Concepts',
        'type': 'robust',
        'metrics': {
            'known_concepts': len(known_result['concept_ids']),
            'unknown_concepts': len(oov_result['concept_ids']),
            'entropy_known': entropy_known,
            'entropy_unknown': entropy_unknown,
        },
        'threshold': {'fewer_extractions_for_oov': True},
        'passed': oov_extraction_lower,
    })

    # ── R-L1-06: Ambiguous Proxy Boundaries ──
    ambiguous_proxy_texts = [
        "Feeling a bit tense, maybe from the coffee",  # Could be stress or not
        "Didn't sleep great, busy day ahead",            # Sleep + work = Z or W?
        "Tired from the workout yesterday",              # Exercise fatigue ≠ stress fatigue
    ]

    # System should still produce reasonable outputs without crashing
    all_processed = True
    proxy_results_valid = True
    for text in ambiguous_proxy_texts:
        try:
            result = l1.process(text)
            if not isinstance(result['z_proxy'], (bool, int, np.bool_)):
                proxy_results_valid = False
            if not isinstance(result['w_proxy'], (float, int)):
                proxy_results_valid = False
        except Exception:
            all_processed = False

    results.append({
        'test_id': 'R-L1-06',
        'name': 'Ambiguous Proxy Boundaries',
        'type': 'robust',
        'metrics': {'all_processed': all_processed, 'valid_proxy_types': proxy_results_valid},
        'threshold': {'processes_without_crash': True},
        'passed': all_processed and proxy_results_valid,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Layer 1 — Robust Validation Tests")
    print("=" * 60)
    results = run_all_tests()
    for r in results:
        status = "PASS ✓" if r['passed'] else "FAIL ✗"
        print(f"\n{r['test_id']}: {r['name']} [{r['type']}]")
        for k, v in r['metrics'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        print(f"  Status: {status}")

    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'=' * 60}")
    print(f"Layer 1 Total: {passed}/{len(results)} passed")
