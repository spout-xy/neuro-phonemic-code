"""
temporal_size_integrator.py
Computes time duration, pulse counting (acoustic/tactile), and motor trace magnitude
directly from physical sensor feedback without a master CPU clock or measuring tape.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class MagnitudeCategory(Enum):
    TINY = 0      # < 120 ms interaction (flick / tap / short tick)
    SMALL = 1     # Hand-sized loop (120–450 ms)
    MEDIUM = 2    # Reachable arm sweep (450–1500 ms)
    BIG = 3       # Full chassis movement (1.5–3.5 s)
    HUGE = 4      # Prolonged transit / Room scale (> 3.5 s)

@dataclass
class MagnitudeEstimate:
    pulse_count: int
    duration_ms: float
    magnitude: MagnitudeCategory
    relative_label: str

class TemporalSizeIntegrator:
    def __init__(self):
        self.pulse_counter: int = 0
        self.last_bit0: int = 0
        self.active_stroke_start_ms: Optional[float] = None
        self.elapsed_stroke_ms: float = 0.0

    def step_pulse(self, bit0: int, contact_burst: bool = False) -> int:
        """Counts rhythmic transitions (0 -> 1 rising edges) or tactile contact bursts."""
        bit0 = bit0 & 1
        rising_edge = (self.last_bit0 == 0 and bit0 == 1)
        self.last_bit0 = bit0
        if rising_edge or contact_burst:
            self.pulse_counter += 1
        return self.pulse_counter

    def start_spatial_trace(self, timestamp_ms: float):
        """Initiates tracking for a physical stroke or audio gesture."""
        self.active_stroke_start_ms = timestamp_ms
        self.elapsed_stroke_ms = 0.0

    def end_spatial_trace(self, timestamp_ms: float, is_boundary_closed: bool = True) -> MagnitudeEstimate:
        """Concludes the spatial/temporal stroke and categorizes magnitude by duration."""
        if self.active_stroke_start_ms is None:
            self.elapsed_stroke_ms = 50.0
        else:
            self.elapsed_stroke_ms = max(10.0, timestamp_ms - self.active_stroke_start_ms)
        
        duration = self.elapsed_stroke_ms
        pulses = self.pulse_counter

        if duration < 120.0:
            category = MagnitudeCategory.TINY
            label = "SHORT / TINY"
        elif duration < 450.0:
            category = MagnitudeCategory.SMALL
            label = "LITTLE / NEAR"
        elif duration < 1500.0:
            category = MagnitudeCategory.MEDIUM
            label = "MEDIUM"
        elif duration < 3500.0:
            category = MagnitudeCategory.BIG
            label = "BIG / FAR"
        else:
            category = MagnitudeCategory.HUGE
            label = "LONG / HUGE"

        return MagnitudeEstimate(
            pulse_count=pulses,
            duration_ms=round(duration, 1),
            magnitude=category,
            relative_label=label
        )

    def reset_counter(self):
        self.pulse_counter = 0
        self.active_stroke_start_ms = None

if __name__ == "__main__":
    integrator = TemporalSizeIntegrator()

    # Test 1: Pulse counting via Bit-0 alternating waveform (0 -> 1 transitions)
    # Simulate: Consonant (0) -> Vowel (1) -> Consonant (0) -> Vowel (1) -> Vowel (1)
    seq = [0, 1, 0, 1, 1, 0]
    for b in seq:
        integrator.step_pulse(bit0=b)
    # 0->1 happened twice (beat 1 and beat 2)
    assert integrator.pulse_counter == 2, f"Expected 2 pulses, got {integrator.pulse_counter}"
    print(f"[PASS] Pulse counting: Successfully counted {integrator.pulse_counter} rising-edge syllable pulses.")

    # Test 2: Contact burst directly increments pulse counter
    integrator.step_pulse(bit0=0, contact_burst=True)
    assert integrator.pulse_counter == 3
    print(f"[PASS] Tactile contact burst incremented pulse counter to {integrator.pulse_counter}.")

    # Test 3: Magnitude duration categorization across full spectrum
    test_cases = [
        (50.0, MagnitudeCategory.TINY, "TINY"),
        (250.0, MagnitudeCategory.SMALL, "SMALL"),
        (800.0, MagnitudeCategory.MEDIUM, "MEDIUM"),
        (2000.0, MagnitudeCategory.BIG, "BIG"),
        (4500.0, MagnitudeCategory.HUGE, "HUGE"),
    ]

    for duration, expected_cat, name in test_cases:
        integrator.start_spatial_trace(timestamp_ms=1000.0)
        est = integrator.end_spatial_trace(timestamp_ms=1000.0 + duration)
        assert est.magnitude == expected_cat, f"Failed for {name}: got {est.magnitude}"
        print(f"[PASS] Duration {duration}ms correctly categorized as {est.relative_label} ({est.magnitude.name}).")

    print("\n[SUCCESS] Temporal Size Integrator: All duration and pulse checks passed cleanly!")
