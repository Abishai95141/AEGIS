"""
AEGIS 3.0 — Full Pipeline Orchestrator

Wires all 5 layers together with bidirectional communication:
    L1 (Semantic) → L2 (Digital Twin) → L3 (Causal) → L4 (Decision) → L5 (Safety)
    L5 → L4 (blocking signal), L4 → L3 (randomization probs), etc.
"""

import numpy as np
import pandas as pd

from .layer1_semantic import SemanticSensorium
from .layer2_digital_twin import DigitalTwin, BergmanParameterAdapter
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

        # Wire MPC to Digital Twin for forward prediction
        self.layer4.mpc.set_digital_twin(self.layer2)

        # Phase 3: p1-Only parameter adapter (wraps DigitalTwin externally)
        self.p1_adapter = BergmanParameterAdapter(self.layer2)

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

        # Phase 3: p1 adapter — update p1 estimate after UKF step
        p1_est = self.p1_adapter.update(glucose)

        step_trace['L2_predicted'] = float(pred_state[0])
        step_trace['L2_updated'] = float(updated_state[0])
        step_trace['L2_innovation'] = float(glucose - pred_state[0])
        step_trace['L2_filter'] = self.layer2.active_filter
        step_trace['L2_p1_est'] = p1_est

        # ── Layer 3: Causal Inference ──
        treatment = 'bolus' if insulin > 0 else 'no_bolus'
        outcome = -abs(glucose - 120)  # Simplified reward for pipeline flow
        step_trace['L3_treatment'] = treatment
        step_trace['L3_outcome'] = outcome

        # ── L1 FIREWALL (Priority 1) ──
        # When Layer 1 is operating in stub mode (fuzzy matching + fake
        # semantic entropy), bypass Layer 3 proxy adjustment entirely.
        # Layer 4's bandit must not receive any input that originates
        # from a STUB-flagged Layer 1 component.
        # See: STUB_REGISTRY.md → L1_FIREWALL, audit/hypothesis_4.py
        l1_stub_active = l1.get('stub_active', True)
        step_trace['L1_stub_active'] = l1_stub_active

        l3 = None
        if not l1_stub_active:
            # Real SLM outputs — safe to use L3 proxy adjustment
            pz = l1.get('z_proxy')
            pw = l1.get('w_proxy')
            if pz is not None and pw is not None:
                l3 = self.layer3.process(
                    z=pz,
                    w=pw,
                    treatment=treatment,
                    outcome=outcome,
                    glucose=glucose,
                    insulin=insulin,
                    carbs=carbs
                )
                step_trace['L3_causal_effect'] = l3.get('causal_effect')
                step_trace['L3_randomization_prob'] = l3.get('randomization_prob')
        else:
            step_trace['L3_causal_effect'] = None
            step_trace['L3_randomization_prob'] = None

        # ── Layer 4: Decision Engine ──
        # When L1 is stub, l3=None → L4 receives only glucose state + PK output
        action_idx, l4_info = self.layer4.select_action(glucose, l3, carbs)
        proposed_action = l4_info['action']
        step_trace['L4_action'] = proposed_action
        step_trace['L4_reason'] = l4_info.get('reason', 'Thompson Sampling')
        step_trace['L4_epsilon'] = l4_info.get('epsilon', 0.0)
        step_trace['L4_base_dose'] = l4_info.get('base_dose', 0.0)
        # ── Layer 5: Safety Supervisor ──
        final_action, tier, reason = self.layer5.evaluate(glucose, proposed_action)
        step_trace['L5_final_action'] = final_action
        step_trace['L5_tier'] = tier
        step_trace['L5_reason'] = reason

        # ── Feedback: L5 → L4 ──
        if tier in ('BLOCKED', 'EMERGENCY', 'STL_BLOCKED', 'STL_CORRECTED'):
            # Blocked → counterfactual update (with Digital Twin prediction)
            dt_pred = float(pred_state[0]) if pred_state is not None else None
            self.layer4.counterfactual_update(action_idx, digital_twin_prediction=dt_pred)
        else:
            # Executed → real update with action centering
            # Priority 2 FIX: Pass baseline=outcome so ACB learns τ(S_t)
            # (the treatment effect), not PID residuals confounded by
            # the controller's own reward contribution. See Eq. 7.
            baseline = outcome  # f(S_t) component
            reward = -abs(glucose - 120.0) / 100.0
            self.layer4.update(action_idx, reward, baseline=baseline)

        # ── λ Calibration: record DT prediction vs actual glucose ──
        # This feeds the isotonic regression calibrator with (prediction, actual)
        # pairs so it can learn the DT's prediction accuracy over time
        if pred_state is not None:
            self.layer4.cts.lambda_calibrator.record(
                float(pred_state[0]), float(glucose)
            )

        self.trace.append(step_trace)
        return step_trace

    def run_simulation(self, patient_data_or_sim, train_nn_every=200,
                        n_days=None, sim_seed=None):
        """
        Run full pipeline simulation.

        Accepts EITHER:
          - A HovorkaPatientSimulator object (true closed-loop)
          - A DataFrame (backward-compatible, but open-loop)

        In closed-loop mode, L5's insulin decision is fed to the simulator
        as the ODE's insulin input. This is the correct architecture:
          Simulator → glucose → Pipeline → L5 insulin → Simulator → next glucose

        Args:
            patient_data_or_sim: HovorkaPatientSimulator or DataFrame
            train_nn_every: Train neural residual every N steps
            n_days: Number of days (only for simulator mode)
            sim_seed: unused, kept for backward compat
        """
        from simulator.patient import HovorkaPatientSimulator

        if isinstance(patient_data_or_sim, HovorkaPatientSimulator):
            return self._run_closed_loop(patient_data_or_sim, n_days or 7,
                                          train_nn_every)
        else:
            return self._run_from_dataframe(patient_data_or_sim,
                                             train_nn_every)

    def _ode_based_iob_check(self, proposed_dose, weight_kg):
        """
        ODE-grounded safety check using glucose trajectory prediction.
        
        Uses the Digital Twin's own predict_trajectory (Bergman ODE + bias
        correction) to project glucose 30 min forward assuming zero additional
        insulin. If the trajectory shows glucose dropping dangerously, the
        proposed dose is scaled down or zeroed.
        
        This is NOT a fixed-τ IOB model — it uses the same forward model as
        the MPC, reading directly from the Bergman ODE state.
        
        Threshold scales with weight: lighter patients have a higher (more
        conservative) hypo guard threshold because insulin acts proportionally
        stronger per unit of body weight.
        
        Returns the dose, possibly reduced or zeroed.
        """
        import numpy as np
        
        # Use the Digital Twin's forward model to predict glucose trajectory
        # with zero future insulin over the next 6 steps (30 minutes)
        horizon = 6
        zero_insulin = np.zeros(horizon)
        zero_carbs = np.zeros(horizon)
        
        try:
            glucose_traj, _ = self.layer2.predict_trajectory(
                zero_insulin, zero_carbs, horizon
            )
            min_projected = np.min(glucose_traj)
        except Exception:
            return proposed_dose  # If prediction fails, don't block

        # Weight-adjusted hypo threshold:
        #   75 kg adult → 70 mg/dL (standard hypoglycemia boundary)
        #   30 kg child → 100 mg/dL (more conservative: insulin acts ~2.5× stronger)
        weight_ratio = weight_kg / 75.0
        hypo_threshold = 70.0 + 30.0 * (1.0 - weight_ratio)
        # Clamp threshold to reasonable range
        hypo_threshold = max(70.0, min(hypo_threshold, 120.0))
        
        if min_projected < hypo_threshold:
            # Scale dose proportionally: zero if projected glucose below threshold-30
            safe_margin = 30.0
            scale = max(0.0, (min_projected - (hypo_threshold - safe_margin)) / safe_margin)
            return proposed_dose * scale
        
        return proposed_dose

    def _run_closed_loop(self, sim, n_days, train_nn_every):
        """
        True closed-loop simulation: L5 insulin → Simulator → glucose → Pipeline.

        At each timestep:
        1. Simulator provides glucose from its current ODE state
        2. Pipeline processes glucose → L1→L2→L3→L4→L5 → insulin decision
        3. L5's insulin + scenario basal rate are fed to simulator as ODE input
        4. Simulator advances one step, producing the next glucose
        """
        # --- Clinical Personalization ---
        # Initialize the baseline controller with the patient's weight to prevent
        # severe overdosing in pediatric cohorts (R-INT-04).
        bw = getattr(sim, 'BW', 75.0)
        weight_ratio = bw / 75.0

        # Pass patient weight to L5 for weight-scaled φ₄ correction
        self.layer5.bw = bw
        # 500 Rule: ICR = 500 / TDD. TDD ~ 0.5 * BW. So ICR ~ 1000 / BW -> modified to 1200 to be gentler
        self.layer4.icr = 1200.0 / bw          # e.g., Adult 75kg -> ICR 16.0; Child 30kg -> ICR 40.0
        self.layer4.kp = 0.02 * weight_ratio   # Scale Proportional gain by weight
        self.layer4.ki = 0.0                   # NO INTEGRAL GAIN: Causes delayed hypoglycemia (insulin stacking)
        self.layer4.kd = 0.03 * weight_ratio   # Scale Derivative gain (reduced to avoid oscillation)
        self.layer4.max_dose = 2.5 * weight_ratio  # Weight-proportional dose cap
        self.layer4.weight_ratio = weight_ratio    # MPC uses this for candidate scaling

        # MPC day tracking — cold-start dose cap on days 0-2
        self.layer4.mpc.current_day = 0

        total_steps = sim.init_scenario(n_days=n_days)
        self._controlled_glucose = []
        prev_day = -1

        for step_idx in range(total_steps):
            # 1. Get current environment (carbs, text, proxies) and glucose
            scenario = sim.get_scenario(step_idx)
            glucose = scenario['glucose_mg_dl']
            self._controlled_glucose.append(glucose)

            current_day = scenario['day']
            if current_day != prev_day and current_day > 0:
                self.layer5.advance_day()
                self.layer4.mpc.current_day = current_day
                prev_day = current_day

            # 2. Pipeline processes glucose → insulin decision
            step_trace = self.step(
                glucose=glucose,
                insulin=0,  # Don't pass pre-computed insulin to L2
                carbs=scenario['carb_input'],
                notes=scenario['notes'],
                timestamp=scenario['timestamp'],
            )

            # 3. L5's insulin decision → simulator's ODE input
            # MPC forward prediction already models IOB via Bergman ODE state.
            # L5 safety supervisor handles emergency reflex (G<54 → suspend).
            l5_insulin = float(step_trace.get('L5_final_action', 0))
            basal_contribution = scenario['basal_rate'] * (5.0 / 60.0)

            # L5 safety blocks → zero bolus but keep basal for basal survival
            # STL_CORRECTED delivers L5's correction dose (φ₄ hyperglycemia correction)
            if step_trace.get('L5_tier') in ('BLOCKED', 'EMERGENCY', 'STL_BLOCKED'):
                total_insulin = basal_contribution  # Maintain basal, zero bolus
            else:
                # Phase 3+: ODE-grounded pediatric IOB guard
                # Reads Bergman X directly — no fixed τ, no external PK model
                l5_insulin = self._ode_based_iob_check(l5_insulin, bw)
                total_insulin = l5_insulin + basal_contribution

            # 4. Advance simulator ODE with pipeline's insulin
            sim.sim_step(
                insulin_input=total_insulin,
                carb_input=scenario['carb_input'],
            )

            # Log the closed-loop data flow
            step_trace['sim_insulin_delivered'] = total_insulin
            step_trace['sim_basal'] = basal_contribution
            step_trace['sim_l5_bolus'] = l5_insulin

            # Periodically train neural residual
            if self.step_count % train_nn_every == 0:
                self.layer2.train_neural_residual(epochs=50)

        return pd.DataFrame(self.trace)

    def _run_from_dataframe(self, patient_data, train_nn_every):
        """
        Open-loop simulation from a pre-generated DataFrame.

        WARNING: This is NOT a true closed loop. The glucose trace is
        pre-determined. L5's decisions do NOT affect subsequent glucose.
        Use _run_closed_loop for integration testing and clinical metrics.
        """
        prev_day = -1
        self._controlled_glucose = []

        for idx, row in patient_data.iterrows():
            current_day = row.get('day', 0)
            if current_day != prev_day and current_day > 0:
                self.layer5.advance_day()
                prev_day = current_day

            glucose = row['glucose_mg_dl']
            self._controlled_glucose.append(glucose)

            self.step(
                glucose=glucose,
                insulin=row.get('insulin_bolus_u', 0),
                carbs=row.get('carbs_g', 0),
                notes=row.get('notes', ''),
                timestamp=row.get('timestamp', None),
            )

            if self.step_count % train_nn_every == 0:
                self.layer2.train_neural_residual(epochs=50)

        return pd.DataFrame(self.trace)

    def get_clinical_metrics(self, glucose_values=None):
        """
        Compute standard clinical glucose metrics.

        If called after run_simulation, uses the controlled glucose trace
        (reflecting pipeline decisions). Falls back to provided values.
        """
        if glucose_values is None and hasattr(self, '_controlled_glucose'):
            g = np.array(self._controlled_glucose)
        elif glucose_values is not None:
            g = np.array(glucose_values)
        else:
            return {}

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
