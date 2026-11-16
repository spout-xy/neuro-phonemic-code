"""
embodied_physics.py
Translates 6-bit phonesthemic coordinates into 4 invariant physical registers:
Resistance, Kinematics, Boundary, and Resonance.
"""
from dataclasses import dataclass

@dataclass
class DualWorldState:
    structural_resistance: float  # 0.0 (flow) to 1.0 (barrier/stop)
    kinematic_reach: float        # dx (-1.0 to 1.0)
    kinematic_elevation: float    # dy (-1.0 to 1.0)
    topological_boundary: float   # 0.0 (open) to 1.0 (enclosed)
    harmonic_resonance: float     # vocalic tone / openness
    is_halted: bool
    expectant_attention: bool

class EmbodiedPhysicsLayer:
    def __init__(self):
        self.prev_elevation = 0.0
        self.prev_halted = False

    def decode_invariants(self, coord: int) -> DualWorldState:
        # Defensive mask to 6-bit Boolean coordinate space (0x00 - 0x3F)
        coord = coord & 0x3F

        # b0: MSB (Bit 5 in 0-5 index)
        b0_is_vowel = (coord >> 5) & 1
        b1_is_voiced = (coord >> 4) & 1
        b2_is_plosive = (coord >> 3) & 1
        zone = (coord >> 1) & 0x03 # 00: Glottal, 01: Throat, 10: Tongue, 11: Lips
        b5_is_curved = coord & 1

        resistance = 1.0 if (b2_is_plosive and not b0_is_vowel) else (0.2 if b0_is_vowel else 0.5)
        reach = (zone - 1.5) / 1.5
        elevation = 1.0 if b0_is_vowel else (0.5 if zone == 2 else 0.0)
        boundary = 1.0 if b5_is_curved else 0.2
        resonance = 1.0 if (b0_is_vowel and b1_is_voiced) else (0.5 if b1_is_voiced else 0.0)
        
        is_halted = resistance >= 0.9
        expectant_attention = (self.prev_elevation > 0.5) and is_halted

        self.prev_elevation = elevation
        self.prev_halted = is_halted

        return DualWorldState(
            structural_resistance=resistance,
            kinematic_reach=reach,
            kinematic_elevation=elevation,
            topological_boundary=boundary,
            harmonic_resonance=resonance,
            is_halted=is_halted,
            expectant_attention=expectant_attention
        )

if __name__ == "__main__":
    engine = EmbodiedPhysicsLayer()

    # Test 1: Vowel-to-Plosive sequence (U 0x34 -> P 0x0F) triggers expectant attention
    state_u = engine.decode_invariants(0x34)
    print("State U (0x34):", state_u)
    assert not state_u.is_halted and not state_u.expectant_attention

    state_p = engine.decode_invariants(0x0F)
    print("State P (0x0F):", state_p)
    assert state_p.is_halted and state_p.expectant_attention
    print("[PASS] Scenario 1: U -> P latches expectant attention.")

    # Test 2: Subsequent step clears expectant attention (transient turn-taking pulse)
    state_next = engine.decode_invariants(0x34)
    assert not state_next.expectant_attention
    print("[PASS] Scenario 2: Expectant attention resets on subsequent input.")

    # Test 3: Flow/Nasal consonant (M 0x17) does not trigger motor halt
    state_m = engine.decode_invariants(0x17)
    assert not state_m.is_halted
    print(f"[PASS] Scenario 3: Continuous flow consonant (M) maintains movement (resistance={state_m.structural_resistance}).")

    # Test 4: Standalone plosive (T 0x0C) halts motor without false-positive attention latch
    state_t = engine.decode_invariants(0x0C)
    assert state_t.is_halted and not state_t.expectant_attention
    print("[PASS] Scenario 4: Standalone plosive barrier (T) halts motor without false attention trigger.")

    print("\n[SUCCESS] All Embodied Physics Layer unit checks passed cleanly!")
