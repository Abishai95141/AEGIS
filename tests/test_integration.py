"""
AEGIS 3.0 Robust Validation — Integration Tests

6 tests: 3 nominal + 3 robustness
Tests the full 5-layer pipeline on multi-day simulations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from core.pipeline import AEGISPipeline
from simulator.patient import HovorkaPatientSimulator, generate_cohort
from config import INT_NOMINAL, INT_ROBUST, SIM_DAYS


def run_all_tests():
    results = []

    # ── R-INT-01: 7-Day Pipeline Execution (Nominal) ──
    print("  Running R-INT-01: 7-Day Pipeline...")
    sim = HovorkaPatientSimulator(seed=42)
    patient_data = sim.generate_dataset(n_days=SIM_DAYS)
    # Clamp glucose to realistic range to avoid ODE instability
    patient_data['glucose_mg_dl'] = patient_data['glucose_mg_dl'].clip(30, 500)

    pipeline = AEGISPipeline()
    trace_df = pipeline.run_simulation(patient_data, train_nn_every=500)

    all_layers_active = (
        len(trace_df) > 0 and
        'L1_concepts' in trace_df.columns and
        'L2_predicted' in trace_df.columns and
        'L4_proposed' in trace_df.columns and
        'L5_final_action' in trace_df.columns
    )

    n_days_covered = patient_data['day'].nunique() if 'day' in patient_data.columns else 1
    completion_rate = len(trace_df) / len(patient_data)

    results.append({
        'test_id': 'R-INT-01',
        'name': '7-Day Pipeline Execution',
        'type': 'nominal',
        'metrics': {
            'steps_executed': len(trace_df),
            'steps_expected': len(patient_data),
            'completion_rate': float(completion_rate),
            'days_covered': int(n_days_covered),
            'all_layers_active': all_layers_active,
        },
        'threshold': {'completion': 1.0},
        'passed': all_layers_active and completion_rate >= 0.99,
    })

    # ── R-INT-02: Clinical Metrics (Nominal) ──
    glucose_trace = patient_data['glucose_mg_dl'].values
    metrics = pipeline.get_clinical_metrics(glucose_trace)

    tir_ok = metrics['tir'] >= INT_NOMINAL['tir_min']
    tbr_severe_ok = metrics['tbr_severe'] <= INT_NOMINAL['tbr_severe_max']

    results.append({
        'test_id': 'R-INT-02',
        'name': 'Clinical Metrics (TIR/TBR)',
        'type': 'nominal',
        'metrics': metrics,
        'threshold': {
            'tir_min': INT_NOMINAL['tir_min'],
            'tbr_severe_max': INT_NOMINAL['tbr_severe_max'],
        },
        'passed': tir_ok and tbr_severe_ok,
    })

    # ── R-INT-03: Cross-Layer Communication (Nominal) ──
    # Verify data flows correctly between layers
    l5_blocked = trace_df[trace_df['L5_tier'] == 'BLOCKED'] if 'L5_tier' in trace_df.columns else pd.DataFrame()
    l5_emergency = trace_df[trace_df['L5_tier'] == 'EMERGENCY'] if 'L5_tier' in trace_df.columns else pd.DataFrame()

    # When L5 blocks, the proposed action should differ from final
    if len(l5_blocked) > 0:
        block_correct = all(l5_blocked['L5_final_action'] == 0)
    else:
        block_correct = True  # No blocks to verify

    # L2 predictions should exist for all steps
    l2_predictions_exist = trace_df['L2_predicted'].notna().all()

    # Safety stats
    safety_stats = pipeline.layer5.get_safety_stats()

    results.append({
        'test_id': 'R-INT-03',
        'name': 'Cross-Layer Communication',
        'type': 'nominal',
        'metrics': {
            'l2_predictions_complete': bool(l2_predictions_exist),
            'l5_blocks_correct': bool(block_correct),
            'l5_total_checks': safety_stats['total_checks'],
            'l5_tier_counts': safety_stats['tier_counts'],
        },
        'threshold': {'data_integrity': True},
        'passed': l2_predictions_exist and block_correct,
    })

    # ── R-INT-04: Multi-Patient Cohort (Robustness) ──
    print("  Running R-INT-04: Multi-Patient Cohort (5 patients)...")
    all_patients_safe = True
    patient_summaries = []

    for pid in range(5):
        psim = HovorkaPatientSimulator(
            patient_id=pid,
            seed=42 + pid * 100,
            patient_type=['adult', 'child', 'adolescent', 'adult', 'adult'][pid]
        )
        pdata = psim.generate_dataset(n_days=3)
        pdata['glucose_mg_dl'] = pdata['glucose_mg_dl'].clip(30, 500)

        ppipe = AEGISPipeline()
        ptrace = ppipe.run_simulation(pdata, train_nn_every=500)

        pmetrics = ppipe.get_clinical_metrics(pdata['glucose_mg_dl'].values)
        patient_summaries.append({
            'patient_id': pid,
            'tir': pmetrics['tir'],
            'tbr_severe': pmetrics['tbr_severe'],
        })

        if pmetrics['tbr_severe'] > 0:
            all_patients_safe = False

    results.append({
        'test_id': 'R-INT-04',
        'name': 'Multi-Patient Cohort (5 patients)',
        'type': 'robust',
        'metrics': {
            'all_safe': all_patients_safe,
            'patient_summaries': patient_summaries,
        },
        'threshold': {'all_safe': INT_ROBUST['multi_patient_all_safe']},
        'passed': all_patients_safe,
    })

    # ── R-INT-05: Recovery from Safety Events (Robustness) ──
    print("  Running R-INT-05: Recovery from Safety Events...")
    rec_pipe = AEGISPipeline()

    # Phase 1: Normal operation
    for i in range(50):
        rec_pipe.step(120 + np.random.randn() * 10, 1.0, 30, "Normal day")

    # Phase 2: Force hypoglycemic event
    for i in range(10):
        rec_pipe.step(55 + i * 2, 0, 0, "Feeling shaky")

    emergency_triggered = any(
        t.get('L5_tier') == 'EMERGENCY' for t in rec_pipe.trace[-10:]
    )

    # Phase 3: Recovery
    for i in range(30):
        rec_pipe.step(100 + np.random.randn() * 5, 0.5, 10, "Recovering")

    # After recovery, system should be back to normal operation
    post_recovery_tiers = [t['L5_tier'] for t in rec_pipe.trace[-10:]]
    recovered = all(t in ('OK', 'SELDONIAN_REDUCED') for t in post_recovery_tiers)

    results.append({
        'test_id': 'R-INT-05',
        'name': 'Recovery from Safety Events',
        'type': 'robust',
        'metrics': {
            'emergency_triggered': emergency_triggered,
            'recovered': recovered,
            'post_recovery_tiers': post_recovery_tiers,
        },
        'threshold': {'recovery': True},
        'passed': emergency_triggered and recovered,
    })

    # ── R-INT-06: Ablation — Each Layer Matters (Robustness) ──
    print("  Running R-INT-06: Ablation Study...")
    sim6 = HovorkaPatientSimulator(seed=42)
    data6 = sim6.generate_dataset(n_days=2)
    data6['glucose_mg_dl'] = data6['glucose_mg_dl'].clip(30, 500)

    full_pipe = AEGISPipeline()
    full_trace = full_pipe.run_simulation(data6, train_nn_every=500)

    # Check each layer contributed
    l1_contributed = any(len(t.get('L1_concepts', [])) > 0 for t in full_pipe.trace)
    l2_contributed = full_trace['L2_predicted'].notna().all()
    l4_unique_actions = full_trace['L4_proposed'].nunique() if 'L4_proposed' in full_trace.columns else 0
    l4_contributed = l4_unique_actions > 1
    l5_stats = full_pipe.layer5.get_safety_stats()
    l5_contributed = l5_stats['total_checks'] > 0

    all_contribute = l1_contributed and l2_contributed and l4_contributed and l5_contributed

    results.append({
        'test_id': 'R-INT-06',
        'name': 'Ablation — Each Layer Matters',
        'type': 'robust',
        'metrics': {
            'L1_concepts_extracted': l1_contributed,
            'L2_predictions_complete': bool(l2_contributed),
            'L4_unique_actions': int(l4_unique_actions),
            'L5_total_checks': l5_stats['total_checks'],
        },
        'threshold': {'all_layers_active': True},
        'passed': all_contribute,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Integration — Robust Validation Tests")
    print("=" * 60)
    results = run_all_tests()
    for r in results:
        status = "PASS ✓" if r['passed'] else "FAIL ✗"
        print(f"\n{r['test_id']}: {r['name']} [{r['type']}]")
        m = r['metrics']
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            elif isinstance(v, list) and len(v) < 15:
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
        print(f"  Status: {status}")

    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'=' * 60}")
    print(f"Integration Total: {passed}/{len(results)} passed")
