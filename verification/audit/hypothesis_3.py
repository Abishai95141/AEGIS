#!/usr/bin/env python3
"""
HYPOTHESIS 3 — IOB Guard Cannot Resolve the TIR/Hypo Tradeoff

WHAT IT MEASURES:
    Sweeps the IOB guard parameter space (τ, max_iob) across a 2D grid
    and records (TIR, TBR_severe) for each combination using the full
    closed-loop simulation with a challenging adolescent patient.

CONFIRMS HYPOTHESIS (architecture must change):
    No point on the (τ, max_iob) surface simultaneously achieves
    TIR ≥ 70% AND TBR_severe = 0%. The Pareto frontier has a gap.

REFUTES HYPOTHESIS (implementation can be fixed):
    There exists at least one (τ, max_iob) combination that passes both
    criteria, indicating the current IOB guard just needs better tuning.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AEGISPipeline
from simulator.patient import HovorkaPatientSimulator


def run_with_iob_params(tau_min, max_iob_frac, patient_id=2, seed=242,
                         patient_type='adolescent', n_days=3):
    """
    Run the full closed-loop with specific IOB guard parameters.
    
    Args:
        tau_min: IOB decay time constant (minutes)
        max_iob_frac: max_iob as a fraction of BW (e.g., 0.125 = BW/8)
    
    Returns: dict with TIR, TBR_severe, mean glucose
    """
    sim = HovorkaPatientSimulator(
        patient_id=patient_id, seed=seed, patient_type=patient_type
    )
    bw = sim.BW
    pipeline = AEGISPipeline()
    
    # Clinical personalization (same as pipeline._run_closed_loop)
    weight_ratio = bw / 75.0
    pipeline.layer4.icr = 1200.0 / bw
    pipeline.layer4.kp = 0.02 * weight_ratio
    pipeline.layer4.ki = 0.0
    pipeline.layer4.kd = 0.03 * weight_ratio
    pipeline.layer4.max_dose = 2.5 * weight_ratio
    pipeline.layer4.actions = [a * weight_ratio for a in [-0.05, 0.0, 0.05, 0.1]]
    
    total_steps = sim.init_scenario(n_days=n_days)
    pipeline._controlled_glucose = []
    prev_day = -1
    
    # IOB guard with custom parameters
    iob = 0.0
    iob_decay = np.exp(-5.0 / tau_min)
    max_iob = max_iob_frac * bw
    
    for step_idx in range(total_steps):
        scenario = sim.get_scenario(step_idx)
        glucose = scenario['glucose_mg_dl']
        pipeline._controlled_glucose.append(glucose)
        
        current_day = scenario['day']
        if current_day != prev_day and current_day > 0:
            pipeline.layer5.advance_day()
            prev_day = current_day
        
        step_trace = pipeline.step(
            glucose=glucose, insulin=0,
            carbs=scenario['carb_input'],
            notes=scenario['notes'],
            timestamp=scenario['timestamp'],
        )
        
        l5_insulin = float(step_trace.get('L5_final_action', 0))
        basal_contribution = scenario['basal_rate'] * (5.0 / 60.0)
        
        # Proportional IOB scaling (same logic as pipeline)
        if l5_insulin > 0 and iob > max_iob * 0.5:
            scale = max(0.0, 1.0 - (iob - max_iob * 0.5) / (max_iob * 0.5))
            l5_insulin *= scale
        
        # PLGS
        if (l5_insulin == 0.0 and step_trace.get('L5_tier') in 
            ('BLOCKED', 'EMERGENCY', 'STL_BLOCKED')) or glucose < 90:
            total_insulin = 0.0
        else:
            total_insulin = l5_insulin + basal_contribution
        
        iob = iob * iob_decay + total_insulin
        
        sim.sim_step(
            insulin_input=total_insulin,
            carb_input=scenario['carb_input'],
        )
    
    # Compute metrics
    glucose_values = np.array(pipeline._controlled_glucose)
    tir = np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100
    tbr_severe = np.mean(glucose_values < 54) * 100
    mean_glucose = np.mean(glucose_values)
    
    return {
        'tir': tir,
        'tbr_severe': tbr_severe,
        'mean': mean_glucose,
    }


def main():
    print("=" * 70)
    print("HYPOTHESIS 3: IOB Guard Parameter Sweep (TIR vs Safety Surface)")
    print("=" * 70)
    print("Patient: adolescent (id=2, seed=242)")
    print()
    
    # Parameter grid
    tau_values = [30, 45, 60, 75, 90, 120, 180, 300]  # minutes
    max_iob_fracs = [1/8, 1/6, 1/5, 1/4, 1/3, 1/2]   # fraction of BW
    
    print(f"τ values (min):    {tau_values}")
    print(f"max_iob fractions: {[f'1/{int(1/f)}' for f in max_iob_fracs]}")
    print()
    
    # Also run WITHOUT any IOB guard as baseline
    print("--- Baseline (no IOB guard) ---")
    baseline = run_with_iob_params(tau_min=9999, max_iob_frac=999, n_days=3)
    print(f"  TIR={baseline['tir']:.1f}%, TBR_severe={baseline['tbr_severe']:.1f}%")
    print()
    
    # Sweep
    results = {}
    feasible = []
    
    print(f"{'τ (min)':>10} | {'max_iob':>10} | {'TIR (%)':>8} | {'TBR_sev (%)':>12} | {'Status':>10}")
    print("-" * 60)
    
    for tau in tau_values:
        for frac in max_iob_fracs:
            r = run_with_iob_params(tau_min=tau, max_iob_frac=frac, n_days=3)
            results[(tau, frac)] = r
            
            label = f"BW/{int(1/frac)}"
            status = "✓ BOTH" if r['tir'] >= 70 and r['tbr_severe'] == 0 else (
                "✓ TIR" if r['tir'] >= 70 else (
                "✓ SAFE" if r['tbr_severe'] == 0 else "✗ BOTH"))
            
            print(f"{tau:>10} | {label:>10} | {r['tir']:>8.1f} | {r['tbr_severe']:>12.1f} | {status:>10}")
            
            if r['tir'] >= 70 and r['tbr_severe'] == 0:
                feasible.append((tau, frac, r['tir'], r['tbr_severe']))
    
    # Verdict
    print()
    print("=" * 70)
    if len(feasible) == 0:
        print("VERDICT: CONFIRMED — No (τ, max_iob) combination achieves both")
        print("         TIR ≥ 70% AND TBR_severe = 0% for this patient.")
        print("         The IOB guard CANNOT resolve the tradeoff.")
        print("         The controller architecture must change.")
    else:
        print(f"VERDICT: REFUTED — {len(feasible)} feasible point(s) found:")
        for tau, frac, tir, tbr in feasible:
            print(f"  τ={tau}min, max_iob=BW/{int(1/frac)}: "
                  f"TIR={tir:.1f}%, TBR_severe={tbr:.1f}%")
        print("  The current IOB guard CAN work with proper tuning.")
    print("=" * 70)


if __name__ == '__main__':
    main()
