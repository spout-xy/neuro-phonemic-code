"""
social_agency_care.py
Couples self-preservation (interoceptive health) with social attunement (other's state).
Features:
- State hysteresis debouncing (prevents sensor chattering)
- Leaky integrator smoothing for prosodic agitation and contact force
- Direct VSA RelationalLayer integration (synthesizing telegram tokens and priorities)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
try:
    from relational_layer import RelationalLayer, RelationalTelegram
except ImportError:
    from .relational_layer import RelationalLayer, RelationalTelegram

class BodilyState(Enum):
    NOMINAL = 0
    HOT_WARN = 1       # Thermal limit exceeded (motor overheat)
    COLD_DEPLETED = 2  # Voltage low / discharge constrained
    HURT_SHOCK = 3     # Accelerometer spike / impact collision
    TIRED_FATIGUE = 4  # Sustained motor torque load

class OtherState(Enum):
    CALM_ATTUNED = 0
    AGITATED_STRESSED = 1
    NEED_CARE = 2
    DISTANT_PASSIVE = 3

@dataclass
class InteroceptiveSensorReading:
    temp_celsius: float
    voltage_normalized: float  # [0.0 = Empty, 1.0 = Full]
    accel_shock_g: float       # Peak G-force
    sustained_torque: float

@dataclass
class SocialAttunementFeedback:
    external_pitch_stress: float # Prosodic agitation from user voice (0.0 to 1.0)
    unexpected_external_force: float
    is_agent_present: bool

@dataclass
class AgencyCareDecision:
    self_state: BodilyState
    other_state: OtherState
    active_focus_agent: str          # "ME", "YOU", or "SHARED"
    telegraphic_pair: Tuple[str, str]# e.g. ("ME", "HOT"), ("YOU", "HUG")
    relational_token: int            # 16-bit VSA concept token
    reflex_action: str
    relational_valence: float        # Normalized -1.0 to +1.0

class SocialAgencyCareEvaluator:
    def __init__(self):
        self.relational_engine = RelationalLayer()
        self.self_state: BodilyState = BodilyState.NOMINAL
        
        # Leaky integration / EMA smoothing for prosodic and force signals
        self.smoothed_pitch_stress: float = 0.0
        self.smoothed_force: float = 0.0
        self.pitch_alpha: float = 0.35
        self.force_alpha: float = 0.40

    def step(self, 
             self_sensors: InteroceptiveSensorReading, 
             other_feedback: SocialAttunementFeedback) -> AgencyCareDecision:
        
        # --- 1. Signal Smoothing (Leaky Integrator) ---
        self.smoothed_pitch_stress = (1.0 - self.pitch_alpha) * self.smoothed_pitch_stress + (self.pitch_alpha * other_feedback.external_pitch_stress)
        self.smoothed_force = (1.0 - self.force_alpha) * self.smoothed_force + (self.force_alpha * other_feedback.unexpected_external_force)

        # --- 2. ME: Self-Preservation & Hysteresis Evaluation ---
        # Hysteresis for temperature: trip above 65.0C, clear only below 58.0C
        if self_sensors.temp_celsius > 65.0 or (self.self_state == BodilyState.HOT_WARN and self_sensors.temp_celsius > 58.0):
            self.self_state = BodilyState.HOT_WARN
            self_words = ("ME", "HOT")
            self_reflex = "STOP / WITHDRAW (MOTOR COOLDOWN)"
        elif self_sensors.accel_shock_g > 3.0:
            self.self_state = BodilyState.HURT_SHOCK
            self_words = ("ME", "HURT")
            self_reflex = "PROTECTIVE CURL / BRACE"
        elif self_sensors.voltage_normalized < 0.2 or (self.self_state == BodilyState.COLD_DEPLETED and self_sensors.voltage_normalized < 0.25):
            self.self_state = BodilyState.COLD_DEPLETED
            self_words = ("ME", "COLD")
            self_reflex = "SEEK DOCK / LOW POWER SLEEP"
        elif self_sensors.sustained_torque > 0.8:
            self.self_state = BodilyState.TIRED_FATIGUE
            self_words = ("ME", "WAIT")
            self_reflex = "REST / UNLOAD ACTUATORS"
        else:
            self.self_state = BodilyState.NOMINAL
            self_words = ("ME", "SAFE")
            self_reflex = "NOMINAL_STABLE"

        # --- 3. YOU: Social Attunement & Empathic Stance ---
        if not other_feedback.is_agent_present:
            other_state = OtherState.DISTANT_PASSIVE
            other_words = ("YOU", "FAR")
            other_reflex = "SOLO_EXPLORE"
        elif self.smoothed_pitch_stress > 0.60 or self.smoothed_force > 0.60:
            other_state = OtherState.AGITATED_STRESSED
            other_words = ("YOU", "HURT")
            other_reflex = "CALMING_GESTURE / SOFTEN_PACE"
        elif self.smoothed_pitch_stress < 0.30:
            other_state = OtherState.CALM_ATTUNED
            other_words = ("YOU", "HUG")
            other_reflex = "ENGAGED_COOPERATION"
        else:
            other_state = OtherState.NEED_CARE
            other_words = ("YOU", "NICE")
            other_reflex = "TENDER_ATTENTION"

        # --- 4. Relational Priority Arbitrator ---
        # Rule 1: Severe bodily danger overrides external compliance
        if self.self_state in (BodilyState.HOT_WARN, BodilyState.HURT_SHOCK):
            active_agent = "ME"
            chosen_words = self_words
            action = f"PRIORITY [SELF]: {self_reflex}"
        # Rule 2: If self is nominal/stable and other is agitated, focus on care
        elif other_state == OtherState.AGITATED_STRESSED:
            active_agent = "YOU"
            chosen_words = other_words
            action = f"PRIORITY [OTHER]: {other_reflex}"
        # Rule 3: Shared cooperative equilibrium
        else:
            active_agent = "SHARED"
            chosen_words = ("YOU", "PLAY") if other_state == OtherState.CALM_ATTUNED else self_words
            action = f"COOPERATIVE: {self_reflex} + {other_reflex}"

        # Evaluate through VSA Relational Layer
        telegram = self.relational_engine.evaluate_telegram(chosen_words[0], chosen_words[1])

        return AgencyCareDecision(
            self_state=self.self_state,
            other_state=other_state,
            active_focus_agent=active_agent,
            telegraphic_pair=chosen_words,
            relational_token=telegram.relational_token,
            reflex_action=action,
            relational_valence=round(telegram.net_valence / 8.0, 2)
        )

if __name__ == "__main__":
    evaluator = SocialAgencyCareEvaluator()

    nominal_me = InteroceptiveSensorReading(temp_celsius=42.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    calm_you = SocialAttunementFeedback(external_pitch_stress=0.15, unexpected_external_force=0.1, is_agent_present=True)

    print("--- Test 1: Nominal Cooperative Baseline ---")
    d1 = evaluator.step(nominal_me, calm_you)
    print(f"Focus: {d1.active_focus_agent:6s} | Telegram: {d1.telegraphic_pair} (Token: 0x{d1.relational_token:04X}) | Val: {d1.relational_valence:+0.2f} | Action: {d1.reflex_action}")
    assert d1.active_focus_agent == "SHARED" and d1.telegraphic_pair == ("YOU", "PLAY")
    print("[PASS] Nominal cooperative balance established.")

    print("\n--- Test 2: Thermal Warning with Hysteresis Debouncing ---")
    # Spike temp to 66.5C -> should trip HOT_WARN and ME HOT
    hot_me = InteroceptiveSensorReading(temp_celsius=66.5, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d2 = evaluator.step(hot_me, calm_you)
    print(f"Step A (66.5°C): State={d2.self_state.name} | Telegram={d2.telegraphic_pair} | Action={d2.reflex_action}")
    assert d2.self_state == BodilyState.HOT_WARN and d2.telegraphic_pair == ("ME", "HOT")

    # Cool down partially to 61.0C -> hysteresis must MAINTAIN HOT_WARN
    cooling_me = InteroceptiveSensorReading(temp_celsius=61.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d2b = evaluator.step(cooling_me, calm_you)
    print(f"Step B (61.0°C - In Deadband): State={d2b.self_state.name} | Telegram={d2b.telegraphic_pair}")
    assert d2b.self_state == BodilyState.HOT_WARN, "Hysteresis failed! Should remain in HOT_WARN at 61C."

    # Cool down below 58.0C (e.g. 55.0C) -> should finally reset to NOMINAL
    cooled_me = InteroceptiveSensorReading(temp_celsius=55.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d2c = evaluator.step(cooled_me, calm_you)
    print(f"Step C (55.0°C - Below Deadband): State={d2c.self_state.name} | Telegram={d2c.telegraphic_pair}")
    assert d2c.self_state == BodilyState.NOMINAL
    print("[PASS] Thermal hysteresis debouncing confirmed.")

    print("\n--- Test 3: Impact Shock Reflex ---")
    shock_me = InteroceptiveSensorReading(temp_celsius=42.0, voltage_normalized=0.85, accel_shock_g=4.2, sustained_torque=0.2)
    d3 = evaluator.step(shock_me, calm_you)
    print(f"Focus: {d3.active_focus_agent:6s} | Telegram: {d3.telegraphic_pair} | Val: {d3.relational_valence:+0.2f} | Action: {d3.reflex_action}")
    assert d3.active_focus_agent == "ME" and d3.telegraphic_pair == ("ME", "HURT")
    print("[PASS] Physical shock triggers protective curl/brace.")

    print("\n--- Test 4: Social Attunement & Empathy (User Agitated) ---")
    # Transient single spike should be smoothed by leaky integrator
    transient_stress = SocialAttunementFeedback(external_pitch_stress=0.90, unexpected_external_force=0.1, is_agent_present=True)
    d4a = evaluator.step(nominal_me, transient_stress)
    print(f"Step A (Transient Stress): Smoothed Pitch={evaluator.smoothed_pitch_stress:.2f} | Focus={d4a.active_focus_agent}")

    # Sustained agitation confirms AGITATED_STRESSED -> triggers YOU HURT soothing response
    d4b = evaluator.step(nominal_me, transient_stress)
    d4c = evaluator.step(nominal_me, transient_stress)
    print(f"Step C (Sustained Stress): Smoothed Pitch={evaluator.smoothed_pitch_stress:.2f} | Focus={d4c.active_focus_agent} | Telegram={d4c.telegraphic_pair} | Action={d4c.reflex_action}")
    assert d4c.active_focus_agent == "YOU" and d4c.telegraphic_pair == ("YOU", "HURT")
    print("[PASS] Leaky acoustic integrator and empathic care posture verified.")

    print("\n--- Test 5: Priority Arbitration (ME overrides YOU in hazard) ---")
    # Both self hazard (overheat) and other agitation present
    d5 = evaluator.step(hot_me, transient_stress)
    print(f"Both Hazards Active -> Priority Focus: {d5.active_focus_agent} | Telegram={d5.telegraphic_pair}")
    assert d5.active_focus_agent == "ME", "Self preservation must override social compliance during bodily danger!"
    print("[PASS] Priority arbitration: Self-preservation successfully overrides social compliance.")

    print("\n[SUCCESS] Social Agency & Interoceptive Care Loop: All tests passed cleanly!")
