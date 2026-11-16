"""
test_module1_baseline.py
Comprehensive automated test suite for Petbot Module 1: Embodied Cognition Baseline.
Copyright (c) 2026 Spout Labs.
"""
import sys
import os
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compile_toddler_lexicon import (
    TODDLER_RAW_LEXICON, COORDINATES, compute_concept_token, rotate_6bit
)
from embodied_physics import EmbodiedPhysicsLayer, DualWorldState
from temporal_size_integrator import TemporalSizeIntegrator, MagnitudeCategory
from relational_layer import RelationalLayer, RelationalTelegram
from social_agency_care import (
    SocialAgencyCareEvaluator, BodilyState, InteroceptiveSensorReading,
    SocialAttunementFeedback, AgencyCareDecision
)
from syllable_clock import SyllableClockFilter
from memory_vault import MemoryVault


def test_lexicon_zero_collisions():
    """Verify all 46 words map to distinct 16-bit tokens with 0 collisions."""
    tokens_seen = {}
    collisions = []
    assert len(TODDLER_RAW_LEXICON) == 46, "Lexicon must contain exactly 46 words."
    
    for word, dx, dy, res_bnd, val in TODDLER_RAW_LEXICON:
        token = compute_concept_token(word)
        if token in tokens_seen:
            collisions.append((word, tokens_seen[token], hex(token)))
        tokens_seen[token] = word
        
    assert len(collisions) == 0, f"Token collisions detected: {collisions}"
    assert len(tokens_seen) == 46, "All 46 words must map to unique tokens."


def test_vsa_delay_line_non_commutativity():
    """Verify non-commutative temporal delay-line binding: UP != PU with >=7 bit Hamming separation."""
    token_up = compute_concept_token("UP")
    token_pu = compute_concept_token("PU")
    assert token_up != token_pu, "UP and PU must not collide."
    
    xor_diff = token_up ^ token_pu
    hamming_dist = bin(xor_diff).count("1")
    assert hamming_dist >= 7, f"Hamming separation must be at least 7 bits (got {hamming_dist})."


def test_embodied_physics_invariants():
    """Verify 4 physical invariants and turn-taking attention latch."""
    engine = EmbodiedPhysicsLayer()
    
    # State U (0x34) -> vowel, not halted
    state_u = engine.decode_invariants(0x34)
    assert not state_u.is_halted and not state_u.expectant_attention
    assert state_u.harmonic_resonance > 0.0

    # State P (0x0F) -> plosive stop halts motor and latches expectant attention
    state_p = engine.decode_invariants(0x0F)
    assert state_p.is_halted and state_p.expectant_attention
    assert state_p.structural_resistance >= 0.9

    # Standalone plosive T (0x0C) -> halts motor without false attention trigger
    engine_fresh = EmbodiedPhysicsLayer()
    state_t = engine_fresh.decode_invariants(0x0C)
    assert state_t.is_halted and not state_t.expectant_attention


def test_temporal_size_integrator():
    """Verify rising-edge pulse counting and 5 discrete magnitude categories."""
    integrator = TemporalSizeIntegrator()
    
    test_cases = [
        (50.0, MagnitudeCategory.TINY),
        (250.0, MagnitudeCategory.SMALL),
        (800.0, MagnitudeCategory.MEDIUM),
        (2000.0, MagnitudeCategory.BIG),
        (4500.0, MagnitudeCategory.HUGE),
    ]

    for duration, expected_cat in test_cases:
        integrator.start_spatial_trace(timestamp_ms=1000.0)
        est = integrator.end_spatial_trace(timestamp_ms=1000.0 + duration)
        assert est.magnitude == expected_cat, f"Failed for duration {duration}: expected {expected_cat}, got {est.magnitude}"


def test_relational_syntax_non_commutative():
    """Verify two-word telegraphic binding: non-commutative tokens and semantic roles."""
    engine = RelationalLayer()
    
    # Non-commutativity
    t_you_hug = engine.evaluate_telegram("YOU", "HUG")
    t_hug_you = engine.evaluate_telegram("HUG", "YOU")
    assert t_you_hug.relational_token != t_hug_you.relational_token, "Relational binding must be non-commutative."
    
    # Semantic roles
    t_hurt = engine.evaluate_telegram("ME", "HURT")
    assert t_hurt.focus_agent == "ME" and t_hurt.net_valence == -8

    t_nogo = engine.evaluate_telegram("NO", "GO")
    assert t_nogo.is_inhibited is True

    t_vbig = engine.evaluate_telegram("VERY", "BIG")
    assert t_vbig.is_amplified is True


def test_social_care_thermal_hysteresis_and_arbitration():
    """Verify thermal hysteresis debouncing (65C / 58C) and priority arbitration."""
    evaluator = SocialAgencyCareEvaluator()
    
    nominal_me = InteroceptiveSensorReading(temp_celsius=42.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    calm_you = SocialAttunementFeedback(external_pitch_stress=0.15, unexpected_external_force=0.1, is_agent_present=True)
    
    # Nominal cooperative state
    d1 = evaluator.step(nominal_me, calm_you)
    assert d1.active_focus_agent == "SHARED" and d1.telegraphic_pair == ("YOU", "PLAY")

    # Thermal trip at 66.5C
    hot_me = InteroceptiveSensorReading(temp_celsius=66.5, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d2 = evaluator.step(hot_me, calm_you)
    assert d2.self_state.name == "HOT_WARN" and d2.telegraphic_pair == ("ME", "HOT")

    # Hysteresis: stays locked at 60C (>58C clear floor)
    cooling_me_60 = InteroceptiveSensorReading(temp_celsius=60.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d3 = evaluator.step(cooling_me_60, calm_you)
    assert d3.self_state.name == "HOT_WARN" and d3.telegraphic_pair == ("ME", "HOT")

    # Clears below 58C
    cooled_me_55 = InteroceptiveSensorReading(temp_celsius=55.0, voltage_normalized=0.85, accel_shock_g=0.1, sustained_torque=0.2)
    d4 = evaluator.step(cooled_me_55, calm_you)
    assert d4.self_state.name != "HOT_WARN"

    # Priority arbitration: Self-preservation (hot_me) overrides other agitation during danger
    transient_stress = SocialAttunementFeedback(external_pitch_stress=0.90, unexpected_external_force=0.1, is_agent_present=True)
    d5 = evaluator.step(hot_me, transient_stress)
    assert d5.active_focus_agent == "ME", "Self preservation must override social compliance during bodily danger!"


def test_syllable_cadence_tracker():
    """Verify bit-0 syllable pacing and vowel sustained resonance."""
    clk = SyllableClockFilter()
    
    # Test word "BANANA" (3 syllables: BA-NA-NA)
    for ch in "BANANA":
        clk.feed_character(ch)
    assert clk.syllable_pulse_count == 3, f"BANANA must pace exactly 3 syllable pulses (got {clk.syllable_pulse_count})"

    # Test elongated vowel "SOOO" -> exactly 1 sustained pulse
    clk.reset()
    for ch in "SOOO":
        clk.feed_character(ch)
    assert clk.syllable_pulse_count == 1, "Elongated vowel must remain a single sustained pulse"


def test_memory_vault_16byte_pack_and_seek():
    """Verify 16-byte fixed struct packing and O(1) byte-offset seek."""
    test_vault_file = "test_pytest_vault.bin"
    if os.path.exists(test_vault_file):
        os.remove(test_vault_file)
        
    vault = MemoryVault(test_vault_file)
    vault.commit_event(coord=0x1F, valence=5, arousal=3, beats=2, context_id=0xA101)
    vault.commit_event(coord=0x2A, valence=-2, arousal=1, beats=1, context_id=0xB202)
    
    rec0 = vault.seek_record_by_index(0)
    assert rec0["coord_hex"] == "0x1F"
    assert rec0["valence"] == 5
    assert rec0["beats"] == 2
    assert rec0["context_id"] == "0xA101"

    rec1 = vault.seek_record_by_index(1)
    assert rec1["coord_hex"] == "0x2A"
    assert rec1["valence"] == -2
    assert rec1["byte_offset"] == 16
    
    if os.path.exists(test_vault_file):
        os.remove(test_vault_file)
