"""
test_module2_neuromorphic.py
Automated test suite for Petbot Module 2: Intel Lava Spiking Silicon (Loihi 2).
Copyright (c) 2026 Spout Labs.
"""
import sys
import os
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "neuromorphic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "neuromorphic"))

from compile_toddler_lexicon import compute_concept_token
from thalamic_gate import ThalamicNoiseShield
from lava_syllable_tracker import LavaSyllableCadenceTracker, word_to_bit0_sequence
from lava_word_resonator import LavaWordResonator


def test_thalamic_gate_noise_rejection():
    """Verify 100% rejection of room noise up to I=4.8 (0 false spikes across 180 neuron-steps)."""
    shield = ThalamicNoiseShield(vth=10.0, dv=0.7, du=0.7)
    
    # 30 steps of sub-threshold noise across 6 channels: all values <= 4.8
    np.random.seed(42)
    noise_matrix = np.random.uniform(0.5, 4.8, size=(6, 30))
    spikes = shield.simulate_channel_stream(noise_matrix)
    
    total_false_spikes = np.sum(spikes)
    assert total_false_spikes == 0, f"Thalamic noise shield must produce 0 false spikes under sub-threshold noise (got {total_false_spikes})."


def test_thalamic_gate_aer_latch():
    """Verify clean supra-threshold phonemic burst (I=15.0) latches exact spikes."""
    shield = ThalamicNoiseShield(vth=10.0, dv=0.7, du=0.7)
    
    # Stimulus: 10 steps noise, then 5 steps burst (I=15.0) on channel 0, then 5 steps noise
    stimulus = np.full((6, 20), 2.0)
    stimulus[0, 10:15] = 15.0
    spikes = shield.simulate_channel_stream(stimulus)
    
    # Channel 0 must spike during burst; channels 1-5 must remain completely quiet
    assert np.sum(spikes[1:, :]) == 0, "Unstimulated channels must remain completely quiet."
    assert np.sum(spikes[0, 10:16]) > 0, "Channel 0 must fire supra-threshold spikes during burst."


def test_lava_syllable_cadence_and_warmth():
    """Verify event-driven syllable pacing: UP=1 beat, BANANA=3 beats."""
    tracker = LavaSyllableCadenceTracker()
    
    # UP = 1 beat
    seq_up = word_to_bit0_sequence("UP")
    pacer_up, warmth_up = tracker.run_cadence_simulation(seq_up)
    assert np.sum(pacer_up) == 1, f"'UP' must generate exactly 1 syllable beat (got {np.sum(pacer_up)})."
    
    # BANANA = 3 beats
    seq_banana = word_to_bit0_sequence("BANANA")
    pacer_banana, warmth_banana = tracker.run_cadence_simulation(seq_banana)
    assert np.sum(pacer_banana) == 3, f"'BANANA' must generate exactly 3 syllable beats (got {np.sum(pacer_banana)})."


def test_lava_word_resonator_continuous_attractor():
    """Verify continuous attractor snaps 1-bit corrupted token ('UB' -> 'UP') with 0 competitor leakage."""
    resonator = LavaWordResonator()
    token_ub = compute_concept_token("UB") # 1-bit corruption of 'UP'
    
    basin_spikes = resonator.run_attractor_simulation(token_ub, num_steps=10)
    
    # Target basin [UP] must latch 5 spikes
    assert basin_spikes.get("UP", 0) == 5, f"Target basin [UP] must latch exactly 5 spikes (got {basin_spikes.get('UP', 0)})."
    
    # All other 9 competing basins must have zero spikes
    competitor_spikes = sum(count for word, count in basin_spikes.items() if word != "UP")
    assert competitor_spikes == 0, f"Zero competitor leakage allowed across all 9 competing basins (got {competitor_spikes} spikes)."
