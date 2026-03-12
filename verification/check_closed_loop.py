"""
AEGIS Pipeline Closed-Loop Integrity Check
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
from core.pipeline import AEGISPipeline
from simulator.patient import HovorkaPatientSimulator

sim = HovorkaPatientSimulator(seed=42)
patient_data = sim.generate_dataset(n_days=1)
patient_data['glucose_mg_dl'] = patient_data['glucose_mg_dl'].clip(30, 500)

pipeline = AEGISPipeline()
trace_df = pipeline.run_simulation(patient_data, train_nn_every=500)

n = len(trace_df)
print("=" * 80)
print("CLOSED-LOOP INTEGRITY CHECK")
print("=" * 80)

sample_idx = np.linspace(10, n-1, 10, dtype=int)
print(f"\n{'Step':>5} | {'L4 Rec':>8} | {'L5 Out':>8} | {'Sim Ins':>8} | {'Sim Gluc':>9} | {'Pipe Gluc':>10} | {'L5→Sim?':>8}")
print("-" * 75)

mismatches = 0
for i in sample_idx:
    l4_action = trace_df.iloc[i].get('L4_action', 'N/A')
    l5_final = float(trace_df.iloc[i].get('L5_final_action', 0))
    sim_insulin = float(patient_data.iloc[i].get('insulin_bolus_u', 0))
    sim_glucose = float(patient_data.iloc[i]['glucose_mg_dl'])
    pipe_glucose = float(trace_df.iloc[i].get('glucose_input', 0))

    if i + 1 < n:
        next_sim_insulin = float(patient_data.iloc[i + 1].get('insulin_bolus_u', 0))
        match = "YES" if abs(l5_final - next_sim_insulin) < 0.01 else "NO"
    else:
        match = "N/A"
    if match == "NO":
        mismatches += 1

    print(f"{i:5d} | {l4_action:>8} | {l5_final:>8.3f} | {sim_insulin:>8.3f} | {sim_glucose:>9.1f} | {pipe_glucose:>10.1f} | {match:>8}")

print("-" * 75)
print(f"\nQ2 ANSWER: {'NO — L5 output is NOT fed to simulator' if mismatches > 0 else 'YES'}")
print(f"  Mismatches: {mismatches}/{len(sample_idx)}")
print(f"\n  Simulator insulin mean: {patient_data['insulin_bolus_u'].mean():.3f} u/step")
print(f"  Pipeline L5 output mean: {trace_df['L5_final_action'].mean():.3f} u/step")
corr = np.corrcoef(patient_data['insulin_bolus_u'].values[:n], trace_df['L5_final_action'].values)[0,1]
print(f"  Correlation: {corr:.4f}")
print(f"\nCONCLUSION: The IOB scaffold modifies glucose MEASUREMENT, not insulin DELIVERY.")
