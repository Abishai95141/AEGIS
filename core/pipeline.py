"""
AEGIS 3.0 — Full Pipeline Orchestrator

Wires all 5 layers together with bidirectional communication:
    L1 (Semantic) → L2 (Digital Twin) → L3 (Causal) → L4 (Decision) → L5 (Safety)
    L5 → L4 (blocking signal), L4 → L3 (randomization probs), etc.
"""

import numpy as np
import pandas as pd

from .layer1_semantic import SemanticSensorium
from .layer2_digital_twin import DigitalTwin
from .layer3_causal import CausalInferenceEngine
from .layer4_decision import DecisionEngine
from .layer5_safety import SafetySupervisor


class AEGISPipeline:
    """
    Full AEGIS 3.0 pipeline with all 5 layers.

    Supports:
    - Step-by-step execution
    - Full simulation over a DataFrame
    - 7-day runs with day-level cold-start advancement
    """

    def __init__(self):
        self.layer1 = SemanticSensorium()
        self.layer2 = DigitalTwin()
        self.layer3 = CausalInferenceEngine()
        self.layer4 = DecisionEngine()
        self.layer5 = SafetySupervisor()

        self.trace = []
        self.step_count = 0
        self.current_day = 0

    def step(self, glucose, insulin, carbs, notes='', timestamp=None,
             proxy_z=None, proxy_w=None):
        """
        Execute one pipeline step.

        Returns: step_trace dict with all layer outputs
        """
        self.step_count += 1
        step_trace = {
            'step': self.step_count,
            'timestamp': timestamp,
            'glucose_input': glucose,
        }

        # ── Layer 1: Semantic Sensorium ──
        l1 = self.layer1.process(notes)
        step_trace['L1_concepts'] = l1['concept_ids']
        step_trace['L1_entropy'] = l1['entropy']
        step_trace['L1_z_proxy'] = l1['z_proxy']
        step_trace['L1_w_proxy'] = l1['w_proxy']

        # ── Layer 2: Digital Twin ──
        pred_state = self.layer2.predict(insulin, carbs)
        updated_state = self.layer2.update(glucose, insulin, carbs)
        step_trace['L2_predicted'] = float(pred_state[0])
        step_trace['L2_updated'] = float(updated_state[0])
        step_trace['L2_innovation'] = float(glucose - pred_state[0])
        step_trace['L2_filter'] = self.layer2.active_filter

        # ── Layer 3: Causal Inference ──
        treatment = 'bolus' if insulin > 0 else 'no_bolus'
        outcome = -abs(glucose - 120)  # Simplified reward for pipeline flow
        step_trace['L3_treatment'] = treatment
        step_trace['L3_outcome'] = outcome
        
        l3 = None
        proxy_z = l1.get('proxy_z')
        proxy_w = l1.get('proxy_w')
        if proxy_z is not None and proxy_w is not None:
            l3 = self.layer3.process(
                z=proxy_z,
                w=proxy_w,
                treatment=treatment,
                outcome=outcome,
                glucose=glucose,
                insulin=insulin,
                carbs=carbs
            )
            step_trace['L3_causal_effect'] = l3.get('causal_effect')
            step_trace['L3_randomization_prob'] = l3.get('randomization_prob')
        # ── Layer 4: Decision Engine ──
        action_idx, l4_info = self.layer4.select_action(glucose, l3)
        proposed_action = self.layer4.actions[action_idx]
        step_trace['L4_action'] = proposed_action
        step_trace['L4_reason'] = l4_info.get('reason', 'Thompson Sampling')
        step_trace['L4_epsilon'] = l4_info.get('epsilon', 0.0)
        # ── Layer 5: Safety Supervisor ──
        final_action, tier, reason = self.layer5.evaluate(glucose, proposed_action)
        step_trace['L5_final_action'] = final_action
        step_trace['L5_tier'] = tier
        step_trace['L5_reason'] = reason

        # ── Feedback: L5 → L4 ──
        if tier in ('BLOCKED', 'EMERGENCY', 'STL_BLOCKED'):
            # Blocked → counterfactual update
            self.layer4.counterfactual_update(action_idx)
        else:
            # Executed → real update with action centering
            baseline = outcome  # Use outcome as baseline for centering
            reward = -abs(glucose - 120.0) / 100.0
            self.layer4.update(action_idx, reward, baseline=None)

        self.trace.append(step_trace)
        return step_trace

    def run_simulation(self, patient_data, train_nn_every=200):
        """
        Run full pipeline on a patient DataFrame.

        Args:
            patient_data: DataFrame with glucose_mg_dl, insulin_bolus_u, carbs_g, notes
            train_nn_every: Train neural residual every N steps

        Returns: DataFrame of step traces
        """
        prev_day = -1

        for idx, row in patient_data.iterrows():
            # Advance cold-start day
            current_day = row.get('day', 0)
            if current_day != prev_day and current_day > 0:
                self.layer5.advance_day()
                prev_day = current_day

            # Execute step
            self.step(
                glucose=row['glucose_mg_dl'],
                insulin=row.get('insulin_bolus_u', 0),
                carbs=row.get('carbs_g', 0),
                notes=row.get('notes', ''),
                timestamp=row.get('timestamp', None),
            )

            # Periodically train neural residual
            if self.step_count % train_nn_every == 0:
                self.layer2.train_neural_residual(epochs=50)

        return pd.DataFrame(self.trace)

    def get_clinical_metrics(self, glucose_values):
        """Compute standard clinical glucose metrics."""
        g = np.array(glucose_values)
        return {
            'tir': float(np.mean((g >= 70) & (g <= 180)) * 100),
            'tbr': float(np.mean(g < 70) * 100),
            'tbr_severe': float(np.mean(g < 54) * 100),
            'tar': float(np.mean(g > 180) * 100),
            'tar_severe': float(np.mean(g > 250) * 100),
            'mean': float(np.mean(g)),
            'std': float(np.std(g)),
            'cv': float(np.std(g) / np.mean(g) * 100),
        }
