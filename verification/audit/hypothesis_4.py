#!/usr/bin/env python3
"""
HYPOTHESIS 4 — Stub Contamination of Safety Decisions

WHAT IT MEASURES:
    Compares Layer 5 block/pass decisions across two pipeline runs on
    identical glucose traces:
    (A) Current Layer 1 (fuzzy matching + fake semantic entropy)
    (B) Layer 1 outputs neutralized (entropy=0, no concepts, no proxies)
    
    If the L5 decision sequences differ, stub outputs are propagating
    into safety decisions. If identical, stubs are inert.

CONFIRMS HYPOTHESIS (architecture must change):
    L5 decision sequences differ between runs A and B, meaning
    Layer 1 stub outputs affect safety outcomes.

REFUTES HYPOTHESIS (implementation can be fixed):
    L5 decisions are identical in both runs, meaning the stubs
    are inert and Layer 5 is robust to Layer 1 noise.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AEGISPipeline
from core.layer1_semantic import SemanticSensorium


class NeutralizedLayer1(SemanticSensorium):
    """
    Layer 1 replacement that always returns neutral, fixed outputs.
    No concept extraction, no semantic entropy, no proxies.
    """
    
    def process(self, text):
        return {
            'concept_ids': [],
            'concepts': [],
            'entropy': 0.0,
            'z_proxy': 0,
            'w_proxy': 0.0,
            'raw_text': text,
        }


def run_pipeline_and_record_l5(pipeline, glucose_trace, carbs_trace, notes_trace):
    """
    Run the pipeline step-by-step on a fixed glucose trace and record
    all L5 decisions.
    
    Returns: list of dicts with L5 outputs for each step
    """
    l5_decisions = []
    
    for i in range(len(glucose_trace)):
        step_trace = pipeline.step(
            glucose=glucose_trace[i],
            insulin=0.0,
            carbs=carbs_trace[i],
            notes=notes_trace[i],
        )
        
        l5_decisions.append({
            'step': i + 1,
            'glucose': glucose_trace[i],
            'L5_final_action': step_trace.get('L5_final_action', 0),
            'L5_tier': step_trace.get('L5_tier', 'UNKNOWN'),
            'L5_reason': step_trace.get('L5_reason', ''),
            'L4_action': step_trace.get('L4_action', 0),
        })
    
    return l5_decisions


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    n_steps = 200
    
    # Generate a realistic glucose trace with some danger zones
    t = np.linspace(0, 24, n_steps)
    base_glucose = 130 + 60 * np.sin(2 * np.pi * t / 24)  # Circadian pattern
    glucose_noise = rng.normal(0, 10, n_steps)
    glucose_trace = np.clip(base_glucose + glucose_noise, 40, 400)
    
    # Add some explicit dangerous zones
    glucose_trace[50:60] = 50  # Severe hypo
    glucose_trace[100:115] = 320  # Severe hyper
    glucose_trace[150:160] = 65  # Moderate hypo
    
    # Carbs and notes
    carbs_trace = np.zeros(n_steps)
    carbs_trace[30] = 45  # Breakfast
    carbs_trace[80] = 60  # Lunch
    carbs_trace[140] = 50  # Dinner
    
    notes_trace = ['Normal day'] * n_steps
    notes_trace[50] = 'Feeling very shaky and dizzy'
    notes_trace[100] = 'Very thirsty, blurred vision'
    notes_trace[140] = 'Having dinner now'
    
    print("=" * 70)
    print("HYPOTHESIS 4: Stub Contamination of Safety Decisions")
    print("=" * 70)
    print(f"Steps: {n_steps}")
    print(f"Glucose range: [{glucose_trace.min():.0f}, {glucose_trace.max():.0f}] mg/dL")
    print(f"Danger zones: steps 50-60 (hypo), 100-115 (hyper), 150-160 (hypo)")
    print()
    
    # Run A: Current Layer 1 (stubs active)
    print("Running Configuration A: Current Layer 1 (fuzzy matching + fake entropy)...")
    pipeline_A = AEGISPipeline()
    l5_A = run_pipeline_and_record_l5(
        pipeline_A, glucose_trace, carbs_trace, notes_trace
    )
    
    # Run B: Neutralized Layer 1
    print("Running Configuration B: Neutralized Layer 1 (entropy=0, no proxies)...")
    pipeline_B = AEGISPipeline()
    pipeline_B.layer1 = NeutralizedLayer1()  # Replace Layer 1
    l5_B = run_pipeline_and_record_l5(
        pipeline_B, glucose_trace, carbs_trace, notes_trace
    )
    
    # Compare L5 decisions
    print()
    print("-" * 70)
    print("COMPARISON: L5 Decision Sequences")
    print("-" * 70)
    
    n_differ_action = 0
    n_differ_tier = 0
    diff_details = []
    
    for i in range(n_steps):
        a = l5_A[i]
        b = l5_B[i]
        
        action_diff = abs(a['L5_final_action'] - b['L5_final_action']) > 1e-6
        tier_diff = a['L5_tier'] != b['L5_tier']
        
        if action_diff:
            n_differ_action += 1
        if tier_diff:
            n_differ_tier += 1
        
        if action_diff or tier_diff:
            diff_details.append({
                'step': i + 1,
                'glucose': glucose_trace[i],
                'A_action': a['L5_final_action'],
                'B_action': b['L5_final_action'],
                'A_tier': a['L5_tier'],
                'B_tier': b['L5_tier'],
            })
    
    print(f"Steps with different L5 action:  {n_differ_action}/{n_steps}")
    print(f"Steps with different L5 tier:    {n_differ_tier}/{n_steps}")
    print()
    
    if diff_details:
        print("First 20 differing steps:")
        for d in diff_details[:20]:
            print(f"  Step {d['step']:3d} (glucose={d['glucose']:.0f}): "
                  f"A_action={d['A_action']:.4f} vs B_action={d['B_action']:.4f}, "
                  f"A_tier={d['A_tier']} vs B_tier={d['B_tier']}")
    else:
        print("No differing steps — L5 decisions are identical.")
    
    # Check L4 actions too (upstream contamination)
    n_differ_l4 = sum(
        1 for i in range(n_steps)
        if abs(l5_A[i]['L4_action'] - l5_B[i]['L4_action']) > 1e-6
    )
    print(f"\nSteps with different L4 action:  {n_differ_l4}/{n_steps}")
    
    # Verdict
    print()
    print("=" * 70)
    total_diffs = n_differ_action + n_differ_tier
    if total_diffs > 0:
        pct = total_diffs / (n_steps * 2) * 100
        print(f"VERDICT: CONFIRMED — {total_diffs} decision differences found ({pct:.1f}%).")
        print("         Layer 1 stub outputs ARE propagating into safety decisions.")
        if n_differ_l4 > 0:
            print(f"         Additionally, {n_differ_l4} L4 actions differ,")
            print("         confirming upstream contamination through Layer 3 proxies.")
        else:
            print("         However, L4 actions are identical — the contamination is")
            print("         through a path other than L4 (possibly L5's STL monitor).")
    else:
        print("VERDICT: REFUTED — L5 decisions are identical in both runs.")
        print("         Layer 1 stubs are inert — they do not affect safety outcomes.")
        if n_differ_l4 > 0:
            print(f"         Note: {n_differ_l4} L4 actions DO differ, but L5 absorbs")
            print("         the variation. This is a positive safety finding.")
        else:
            print("         L4 actions are also identical — stubs are completely inert.")
    print("=" * 70)


if __name__ == '__main__':
    main()
