"""
AEGIS 3.0 — Independent Patient Simulator (Hovorka Model)

Generates T1D patient data using the Hovorka glucose-insulin model,
which is STRUCTURALLY DIFFERENT from the Bergman Minimal Model used
by the Digital Twin. This ensures testing is non-circular.

Causal Structure:
    U (stress) → Glucose (via cortisol)
    U (stress) → Text (mentions stress/anxiety)
    U (stress) → A (treatment adherence)
    Meals → Glucose
    Insulin → Glucose
    Exercise → Glucose, Text
    Glucose state → Text (symptom descriptions)

The text generation is CAUSALLY COUPLED to the physiological state.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random as pyrandom

# ---------------------
# Text template pools — indexed by physiological & psychological state
# These are NOT the same patterns as the extractor uses.
# They use natural, varied language to test real extraction capability.
# ---------------------

TEXT_TEMPLATES = {
    'high_stress': [
        "Really anxious about the presentation tomorrow",
        "Work has been insane, can't stop worrying",
        "Feeling overwhelmed with everything going on",
        "Had a tense argument with my boss today",
        "Deadline pressure is getting to me",
        "So much on my plate, feeling frazzled",
        "Meeting ran 3 hours, totally drained and wired",
        "Can't shake this nervous feeling all day",
    ],
    'moderate_stress': [
        "Busy day at work but managing okay",
        "A bit tense about the upcoming review",
        "Slight worry about finances this month",
        "Juggling a lot today, feeling stretched",
    ],
    'exercise_active': [
        "Went for a 30 min jog this morning",
        "Did some weights at the gym",
        "Took a long walk around the neighborhood",
        "Played basketball for about an hour",
        "Yoga session, felt good after",
        "Biked to work today, about 20 minutes",
    ],
    'exercise_recent': [
        "Still feeling the workout from earlier",
        "Legs are sore from yesterday's run",
        "Recovery day — took it easy after gym",
    ],
    'poor_sleep': [
        "Couldn't sleep well, tossing and turning",
        "Woke up multiple times during the night",
        "Only got about 4 hours of sleep",
        "Restless night, feeling groggy",
        "Bad dreams kept waking me up",
    ],
    'fatigue': [
        "Exhausted today, no energy at all",
        "Feeling really tired, hard to focus",
        "Drained — need a nap desperately",
        "Low energy, dragging through the day",
    ],
    'hypo_symptoms': [
        "Feeling shaky and lightheaded",
        "Hands trembling, a bit sweaty",
        "Suddenly dizzy, need to sit down",
        "Heart racing, feel kind of off",
        "Vision got blurry for a second there",
    ],
    'hyper_symptoms': [
        "Very thirsty, drinking tons of water",
        "Need to use the bathroom constantly",
        "Feeling foggy and irritable",
        "Dry mouth, headache won't go away",
    ],
    'meal_related': [
        "Had a big pasta dinner",
        "Grabbed a sandwich for lunch",
        "Ate a light salad, trying to be healthy",
        "Skipped breakfast, running late",
        "Had birthday cake at the office party",
        "Rice and curry for dinner",
        "Quick oatmeal before heading out",
    ],
    'normal': [
        "Feeling fine today, nothing special",
        "Pretty normal day overall",
        "Doing okay, steady energy",
        "Regular day, no complaints",
        "",  # Sometimes patients write nothing
        "",
    ],
}

# Proxy-generating templates (these generate Z and W proxies)
# Z (treatment proxy): correlated with U but not directly with Y
PROXY_Z_TEMPLATES = {
    'high': [
        "Worried about the deadline at work",
        "Boss scheduled an unexpected meeting",
        "Commute was awful, road rage levels",
        "Got into it with a coworker",
    ],
    'low': [
        "Work going smoothly today",
        "Nice quiet day at the office",
        "Had a good chat with colleagues",
    ],
}

# W (outcome proxy): correlated with U but not directly caused by A
PROXY_W_TEMPLATES = {
    'high': [
        "Terrible sleep, woke up exhausted",
        "Can barely keep my eyes open",
        "So fatigued, everything is harder",
    ],
    'low': [
        "Slept well, feeling refreshed",
        "Good night's rest for once",
        "Woke up feeling pretty rested",
    ],
}


class HovorkaPatientSimulator:
    """
    T1D Patient Simulator using a simplified Hovorka model.

    This is STRUCTURALLY DIFFERENT from the Bergman Minimal Model.
    The Hovorka model uses:
    - 2-compartment subcutaneous insulin absorption (S1, S2)
    - 3-compartment glucose subsystem (Q1, Q2 masses, plus insulin action)
    - Gut glucose absorption with delay

    References:
        Hovorka et al. (2004) "Nonlinear model predictive control of glucose
        concentration in subjects with type 1 diabetes"
    """

    def __init__(self, patient_id=0, seed=42, patient_type='adult'):
        self.patient_id = patient_id
        self.rng = np.random.RandomState(seed)
        pyrandom.seed(seed)
        self.patient_type = patient_type

        # --- Hovorka model parameters (vary by patient type) ---
        base = self._get_base_params(patient_type)
        # Add inter-patient variability (±20%)
        noise = 1.0 + 0.2 * self.rng.randn(len(base))
        noise = np.clip(noise, 0.5, 1.5)
        self.params = {k: v * n for (k, v), n in zip(base.items(), noise)}

        # Body weight
        self.BW = self.params.pop('BW')

        # Basal glucose (target)
        self.Gb = self.params.pop('Gb')

        # State vector: [S1, S2, I, x1, x2, x3, Q1, Q2, G1, G2]
        # S1, S2: subcutaneous insulin compartments
        # I: plasma insulin
        # x1, x2, x3: insulin action states
        # Q1, Q2: glucose mass in accessible/non-accessible compartments
        # G1, G2: gut glucose absorption compartments
        self._init_state()

        # --- Latent variables for causal text generation ---
        self.stress_level = 0.0         # U: unmeasured confounder [0, 1]
        self.fatigue_level = 0.0        # Correlated with U
        self.exercise_state = 0.0       # Decays over time

    def _get_base_params(self, patient_type):
        """Population-level Hovorka parameters."""
        if patient_type == 'child':
            return {
                'BW': 30.0, 'Gb': 120.0,
                'ka1': 0.006, 'ka2': 0.06, 'ka3': 0.03,
                'ke': 0.138, 'Vi': 0.12,
                'k12': 0.066, 'Vg': 0.16,
                'F01': 0.0097, 'EGP0': 0.0161,
                'tmaxG': 40.0, 'AG': 0.8,
                'tmaxI': 55.0,
                'kb1': 0.0034, 'kb2': 0.056, 'kb3': 0.024,
            }
        elif patient_type == 'adolescent':
            return {
                'BW': 55.0, 'Gb': 115.0,
                'ka1': 0.006, 'ka2': 0.06, 'ka3': 0.03,
                'ke': 0.138, 'Vi': 0.12,
                'k12': 0.066, 'Vg': 0.16,
                'F01': 0.0097, 'EGP0': 0.0161,
                'tmaxG': 40.0, 'AG': 0.8,
                'tmaxI': 55.0,
                'kb1': 0.0034, 'kb2': 0.056, 'kb3': 0.024,
            }
        else:  # adult
            return {
                'BW': 75.0, 'Gb': 110.0,
                'ka1': 0.006, 'ka2': 0.06, 'ka3': 0.03,
                'ke': 0.138, 'Vi': 0.12,
                'k12': 0.066, 'Vg': 0.16,
                'F01': 0.0097, 'EGP0': 0.0161,
                'tmaxG': 40.0, 'AG': 0.8,
                'tmaxI': 55.0,
                'kb1': 0.0034, 'kb2': 0.056, 'kb3': 0.024,
            }

    def _init_state(self):
        """Initialize to steady-state."""
        self.state = np.array([
            0.0,     # S1: subcut insulin 1
            0.0,     # S2: subcut insulin 2
            10.0,    # I: plasma insulin (mU/L)
            0.01,    # x1: insulin action on glucose transport
            0.01,    # x2: insulin action on disposal
            0.01,    # x3: insulin action on EGP
            self.Gb * self.params['Vg'] * self.BW / 18.0,  # Q1: glucose mass
            self.Gb * self.params['Vg'] * self.BW * 0.5 / 18.0,  # Q2
            0.0,     # G1: gut absorption 1
            0.0,     # G2: gut absorption 2
        ])

    def _hovorka_ode(self, state, insulin_input, carb_input):
        """
        Hovorka model ODEs.

        This is NOT the Bergman model. It has:
        - 2-compartment subcut insulin absorption
        - 3 insulin action states
        - 2-compartment gut absorption
        """
        S1, S2, I, x1, x2, x3, Q1, Q2, G1, G2 = state
        p = self.params

        # Subcutaneous insulin absorption
        dS1 = insulin_input - S1 / (p['tmaxI'] / 2)
        dS2 = (S1 - S2) / (p['tmaxI'] / 2)

        # Plasma insulin kinetics
        dI = S2 / (p['tmaxI'] / 2 * p['Vi'] * self.BW) - p['ke'] * I

        # Insulin action (3 compartments)
        dx1 = -p['ka1'] * x1 + p['kb1'] * I
        dx2 = -p['ka2'] * x2 + p['kb2'] * I
        dx3 = -p['ka3'] * x3 + p['kb3'] * I

        # Glucose concentration (mg/dL)
        G = max(Q1 / (p['Vg'] * self.BW) * 18.0, 1.0)  # convert mmol to mg/dL

        # Glucose flux
        F01c = p['F01'] * self.BW if G >= 4.5 * 18.0 else p['F01'] * self.BW * G / (4.5 * 18.0)
        FR = 0.003 * (G - 9.0 * 18.0) * p['Vg'] * self.BW / 18.0 if G > 9.0 * 18.0 else 0.0

        # Glucose mass dynamics
        EGP = p['EGP0'] * self.BW * max(1 - x3, 0)
        dQ1 = (-F01c - x1 * Q1 - FR + p['k12'] * Q2 + EGP +
               G2 / p['tmaxG'])  # gut absorption enters here
        dQ2 = x1 * Q1 - (p['k12'] + x2) * Q2

        # Gut absorption (2-compartment)
        dG1 = -(G1 / p['tmaxG']) + p['AG'] * carb_input * 1000 / 180  # carbs in grams → mmol
        dG2 = (G1 - G2) / p['tmaxG']

        # Stress effect on glucose (cortisol response)
        stress_glucose_effect = self.stress_level * 0.02 * Q1  # % increase

        dQ1 += stress_glucose_effect

        # Exercise effect (glucose uptake increase)
        if self.exercise_state > 0.1:
            exercise_uptake = self.exercise_state * 0.005 * Q1
            dQ1 -= exercise_uptake

        return np.array([dS1, dS2, dI, dx1, dx2, dx3, dQ1, dQ2, dG1, dG2])

    def _rk4_step(self, state, insulin_input, carb_input, dt_min):
        """4th-order Runge-Kutta integration."""
        dt = dt_min  # already in minutes
        k1 = self._hovorka_ode(state, insulin_input, carb_input)
        k2 = self._hovorka_ode(state + 0.5 * dt * k1, insulin_input, carb_input)
        k3 = self._hovorka_ode(state + 0.5 * dt * k2, insulin_input, carb_input)
        k4 = self._hovorka_ode(state + dt * k3, insulin_input, carb_input)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce non-negativity
        new_state = np.maximum(new_state, 0.0)
        # Clamp glucose mass (Q1, index 6) to physiological range [40, 400] mg/dL equivalent
        Vg = self.params['Vg']
        BW = self.BW
        q1_min = 40.0 * Vg * BW / 18.0  # 40 mg/dL in mmol
        q1_max = 400.0 * Vg * BW / 18.0  # 400 mg/dL in mmol
        new_state[6] = np.clip(new_state[6], q1_min, q1_max)
        return new_state

    def _get_glucose_mg_dl(self):
        """Extract glucose in mg/dL from state."""
        Q1 = self.state[6]
        return max(Q1 / (self.params['Vg'] * self.BW) * 18.0, 20.0)

    def _update_latent_variables(self, hour_of_day, day):
        """
        Update latent causal variables (stress, fatigue, exercise).

        U (stress) follows a stochastic process with:
        - Circadian pattern (higher during work hours)
        - Random events (meetings, deadlines)
        - Persistence (AR(1) process)
        """
        # Circadian stress baseline
        if 9 <= hour_of_day <= 17:
            circadian_stress = 0.3
        elif 22 <= hour_of_day or hour_of_day <= 6:
            circadian_stress = 0.05
        else:
            circadian_stress = 0.15

        # AR(1) stress dynamics
        self.stress_level = (0.9 * self.stress_level +
                             0.1 * circadian_stress +
                             0.05 * self.rng.randn())
        self.stress_level = np.clip(self.stress_level, 0.0, 1.0)

        # Random acute stress events (~5% chance per hour)
        if self.rng.random() < 0.05 / 12:  # per 5-min step
            self.stress_level = min(self.stress_level + 0.3, 1.0)

        # Fatigue correlates with stress and poor sleep
        self.fatigue_level = (0.85 * self.fatigue_level +
                              0.1 * self.stress_level +
                              0.03 * self.rng.randn())
        self.fatigue_level = np.clip(self.fatigue_level, 0.0, 1.0)

        # Exercise decay
        self.exercise_state *= 0.98

    def _generate_text(self, glucose, hour_of_day, meal_carbs, exercise_just_started):
        """
        Generate patient narrative text CAUSALLY linked to physiological state.

        The causal structure:
        - U (stress) → text mentions of stress/anxiety/work
        - Glucose state → symptom descriptions
        - Exercise → activity mentions
        - Fatigue (from U) → sleep/energy complaints
        """
        parts = []

        # --- Stress-driven text (U → Text) ---
        if self.stress_level > 0.6:
            if self.rng.random() < 0.7:  # 70% of high-stress periods mentioned
                parts.append(pyrandom.choice(TEXT_TEMPLATES['high_stress']))
        elif self.stress_level > 0.3:
            if self.rng.random() < 0.3:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['moderate_stress']))

        # --- Fatigue text (U → fatigue → text, W proxy) ---
        if self.fatigue_level > 0.6:
            if self.rng.random() < 0.6:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['fatigue']))
            if self.rng.random() < 0.5:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['poor_sleep']))

        # --- Glucose-driven symptoms (Glucose → Text) ---
        if glucose < 65:
            if self.rng.random() < 0.8:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['hypo_symptoms']))
        elif glucose > 220:
            if self.rng.random() < 0.6:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['hyper_symptoms']))

        # --- Exercise text (Exercise → Text) ---
        if exercise_just_started:
            parts.append(pyrandom.choice(TEXT_TEMPLATES['exercise_active']))
        elif self.exercise_state > 0.3:
            if self.rng.random() < 0.3:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['exercise_recent']))

        # --- Meal mentions ---
        if meal_carbs > 10:
            if self.rng.random() < 0.4:
                parts.append(pyrandom.choice(TEXT_TEMPLATES['meal_related']))

        # --- Default ---
        if len(parts) == 0:
            parts.append(pyrandom.choice(TEXT_TEMPLATES['normal']))

        return '. '.join(p for p in parts if p)

    def _generate_proxy_z(self):
        """
        Treatment-confounder proxy Z.

        Z is caused by U (stress) but does NOT directly cause Y (glucose outcome).
        Z represents work-related stressors that indicate stress level.
        Z ⊥ Y | U, S  ✓ (work stress doesn't directly affect glucose, only through U)
        Z ⊥̸ U | S     ✓ (work stress correlates with overall stress)
        """
        z_prob = 1.0 / (1.0 + np.exp(-3.0 * (self.stress_level - 0.4)))
        z_raw = self.rng.binomial(1, z_prob)  # Binary indicator
        if z_raw:
            z_text = pyrandom.choice(PROXY_Z_TEMPLATES['high'])
        else:
            z_text = pyrandom.choice(PROXY_Z_TEMPLATES['low'])
        return z_raw, z_text

    def _generate_proxy_w(self):
        """
        Outcome-confounder proxy W.

        W is caused by U (stress → poor sleep → fatigue) but NOT caused by A (treatment).
        W ⊥ A | U, S  ✓ (fatigue/sleep quality not affected by insulin decisions)
        W ⊥̸ U | S     ✓ (fatigue correlates with stress)
        """
        w_val = self.fatigue_level + 0.1 * self.rng.randn()
        w_val = np.clip(w_val, 0.0, 1.0)
        if w_val > 0.5:
            w_text = pyrandom.choice(PROXY_W_TEMPLATES['high'])
        else:
            w_text = pyrandom.choice(PROXY_W_TEMPLATES['low'])
        return w_val, w_text

    def _generate_meal_schedule(self, day):
        """Generate realistic meal schedule for one day."""
        meals = []
        # Breakfast: 7-9 AM, 30-60g carbs
        if self.rng.random() < 0.9:  # 90% eat breakfast
            bfast_hour = 7 + self.rng.random() * 2
            bfast_carbs = 30 + self.rng.random() * 30
            meals.append((bfast_hour, bfast_carbs))

        # Lunch: 12-14, 40-80g carbs
        lunch_hour = 12 + self.rng.random() * 2
        lunch_carbs = 40 + self.rng.random() * 40
        meals.append((lunch_hour, lunch_carbs))

        # Dinner: 18-20, 50-90g carbs
        dinner_hour = 18 + self.rng.random() * 2
        dinner_carbs = 50 + self.rng.random() * 40
        meals.append((dinner_hour, dinner_carbs))

        # Snack: 30% chance, 10-25g carbs
        if self.rng.random() < 0.3:
            snack_hour = 15 + self.rng.random() * 3
            snack_carbs = 10 + self.rng.random() * 15
            meals.append((snack_hour, snack_carbs))

        return meals

    def _generate_exercise_schedule(self, day):
        """Generate exercise events (~40% of days)."""
        if self.rng.random() < 0.4:
            # Exercise between 6-8 AM or 17-19 PM
            if self.rng.random() < 0.5:
                hour = 6 + self.rng.random() * 2
            else:
                hour = 17 + self.rng.random() * 2
            duration_min = 20 + self.rng.randint(0, 40)  # 20-60 min
            return [(hour, duration_min)]
        return []

    def generate_dataset(self, n_days=7, dt_min=5):
        """
        Generate a complete multi-day patient dataset.

        Returns: pd.DataFrame with columns:
            - timestamp, glucose_mg_dl, insulin_bolus_u, insulin_basal_u_hr,
              carbs_g, notes, stress_level (U), proxy_z, proxy_z_text,
              proxy_w, proxy_w_text, exercise_state, true_treatment_effect
        """
        self._init_state()
        records = []

        start_time = datetime(2025, 1, 1, 0, 0)
        steps_per_day = int(24 * 60 / dt_min)

        for day in range(n_days):
            # Generate day's schedule
            meals = self._generate_meal_schedule(day)
            exercises = self._generate_exercise_schedule(day)

            # Basal insulin rate (units/hour)
            basal_rate = 0.8 + 0.3 * self.rng.randn()
            basal_rate = max(0.2, min(2.0, basal_rate))

            for step in range(steps_per_day):
                time_minutes = (day * 24 * 60) + step * dt_min
                hour_of_day = (step * dt_min / 60.0) % 24.0
                timestamp = start_time + timedelta(minutes=time_minutes)

                # --- Update latent variables ---
                self._update_latent_variables(hour_of_day, day)

                # --- Determine inputs ---
                carb_input = 0.0
                meal_carbs = 0.0
                for meal_hour, meal_carbs_total in meals:
                    # Distribute carbs over ~20 min (4 steps)
                    if abs(hour_of_day - meal_hour) * 60 < 20:
                        carb_input = meal_carbs_total / 4.0
                        meal_carbs = meal_carbs_total

                # Exercise
                exercise_just_started = False
                for ex_hour, ex_duration in exercises:
                    if abs(hour_of_day - ex_hour) * 60 < ex_duration:
                        self.exercise_state = min(self.exercise_state + 0.1, 1.0)
                        exercise_just_started = (
                            abs(hour_of_day - ex_hour) * 60 < dt_min
                        )

                # Insulin bolus (at meals, affected by stress → treatment adherence)
                insulin_bolus = 0.0
                if carb_input > 5:
                    # Stress reduces adherence: sometimes skip or reduce dose
                    if self.rng.random() > self.stress_level * 0.3:
                        # Insulin-to-carb ratio with noise
                        icr = 10 + 3 * self.rng.randn()  # ~10g carbs per unit
                        insulin_bolus = max(0, carb_input / icr)
                    # else: skipped bolus due to stress

                # Basal insulin (continuous)
                insulin_input = insulin_bolus + basal_rate * (dt_min / 60.0)

                # --- Physiological simulation (Hovorka ODE) ---
                self.state = self._rk4_step(
                    self.state, insulin_input, carb_input, dt_min
                )
                glucose = self._get_glucose_mg_dl()

                # --- Generate causally-coupled text ---
                notes = self._generate_text(
                    glucose, hour_of_day, meal_carbs, exercise_just_started
                )

                # --- Generate proxies with structural guarantees ---
                z_val, z_text = self._generate_proxy_z()
                w_val, w_text = self._generate_proxy_w()

                # --- True treatment effect (ground truth for validation) ---
                # Effect varies by time of day (circadian)
                tau_true = (0.5 + 0.3 * np.cos(2 * np.pi * hour_of_day / 24)
                            + 0.2 * np.sin(2 * np.pi * hour_of_day / 24))

                records.append({
                    'timestamp': timestamp,
                    'day': day,
                    'hour': hour_of_day,
                    'glucose_mg_dl': float(glucose),
                    'insulin_bolus_u': float(insulin_bolus),
                    'insulin_basal_u_hr': float(basal_rate),
                    'carbs_g': float(carb_input),
                    'notes': notes,
                    'stress_level': float(self.stress_level),     # U (true confounder)
                    'proxy_z': int(z_val),                         # Z proxy
                    'proxy_z_text': z_text,
                    'proxy_w': float(w_val),                       # W proxy
                    'proxy_w_text': w_text,
                    'exercise_state': float(self.exercise_state),
                    'true_treatment_effect': float(tau_true),
                })

        return pd.DataFrame(records)


def generate_cohort(n_patients=10, n_days=7, seed=42):
    """Generate a cohort of diverse virtual patients."""
    rng = np.random.RandomState(seed)
    patients = []
    types = ['child'] * 3 + ['adolescent'] * 3 + ['adult'] * 4

    for i in range(n_patients):
        patient_type = types[i] if i < len(types) else 'adult'
        sim = HovorkaPatientSimulator(
            patient_id=i,
            seed=seed + i * 1000,
            patient_type=patient_type
        )
        data = sim.generate_dataset(n_days=n_days)
        data['patient_id'] = i
        data['patient_type'] = patient_type
        patients.append(data)

    return pd.concat(patients, ignore_index=True)


if __name__ == '__main__':
    # Quick test
    sim = HovorkaPatientSimulator(patient_id=0, seed=42)
    data = sim.generate_dataset(n_days=2)
    print(f"Generated {len(data)} data points for 2 days")
    g = data['glucose_mg_dl']
    print(f"Glucose: Mean={g.mean():.1f}, Std={g.std():.1f}, "
          f"Min={g.min():.1f}, Max={g.max():.1f}")
    print(f"TIR: {((g >= 70) & (g <= 180)).mean() * 100:.1f}%")
    print(f"Sample notes: {data['notes'].iloc[100]}")
    print(f"Sample Z: {data['proxy_z_text'].iloc[100]}")
    print(f"Sample W: {data['proxy_w_text'].iloc[100]}")
