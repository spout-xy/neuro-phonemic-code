"""
lava_syllable_tracker.py
Module 2 / Assignment 7: The Lava Syllable-Cadence Tracker.
Connects the alternating Bit-0 wave (0 <-> 1) to dual-stream spiking channels in Intel Lava.
Demonstrates:
  1. Event-driven clock pacing: Speech rhythm paces network time without an external software oscillator.
  2. Prosodic duty-cycle integration: Distinguishes excited staccato syllables from soothing prolonged resonance.
"""
import numpy as np
from typing import Dict, List, Tuple
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compile_toddler_lexicon import COORDINATES

class LavaSyllableCadenceTracker:
    def __init__(self):
        # 2-compartment neuromorphic process:
        # Neuron 0 (Pacer)    : Sharp differentiator detecting 0 -> 1 rising edge (Syllable Beat)
        # Neuron 1 (Resonance): Slow leaky integrator measuring sustained vowel warmth (Duty Cycle)
        self.vth = np.array([10.0, 10.0])
        self.dv = np.array([0.9, 0.2])   # Pacer resets instantly; Resonance integrates slowly
        self.du = np.array([0.9, 0.2])

    def encode_bit0_streams(self, bit0_sequence: np.ndarray) -> np.ndarray:
        """
        Converts a 1D binary Bit-0 wave into dual-channel input currents:
        Channel 0: Rising-edge pulse current (15.0 on 0 -> 1 transitions)
        Channel 1: Vowel level current (5.0 sustained when Bit-0 == 1)
        """
        b0 = bit0_sequence.astype(float)
        diff = np.diff(b0, prepend=0.0)
        curr_pacer = np.where(diff > 0.5, 15.0, 0.0)
        curr_resonance = np.where(b0 > 0.5, 5.0, 0.0)
        return np.vstack([curr_pacer, curr_resonance])

    def run_cadence_simulation(self, bit0_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the dual-channel cadence tracker in Lava under Loihi2SimCfg.
        Returns: (pacing_spikes, resonance_spikes)
        """
        num_steps = len(bit0_sequence)
        curr_matrix = self.encode_bit0_streams(bit0_sequence)

        source = Source(data=curr_matrix)
        tracker = LIF(shape=(2,), vth=self.vth, dv=self.dv, du=self.du)
        sink = Sink(shape=(2,), buffer=num_steps)

        source.s_out.connect(tracker.a_in)
        tracker.s_out.connect(sink.a_in)

        tracker.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
        spikes = sink.data.get().copy()
        tracker.stop()

        pacing_spikes = spikes[0, :]
        resonance_spikes = spikes[1, :]
        return pacing_spikes, resonance_spikes

def word_to_bit0_sequence(word: str, steps_per_char: int = 2) -> np.ndarray:
    """Converts a word into an expanded Bit-0 sequence based on its 6-bit phonetic coordinates."""
    seq = []
    for ch in word.upper():
        coord = COORDINATES.get(ch, 0)
        bit0 = (coord >> 5) & 1 # MSB: 1=Vowel, 0=Consonant
        seq.extend([bit0] * steps_per_char)
    return np.array(seq, dtype=int)

def main():
    print("=================================================================")
    print("Intel Lava: Experiment 3 - Event-Driven Cadence & Syllable Tracker")
    print("=================================================================\n")

    tracker = LavaSyllableCadenceTracker()

    # -------------------------------------------------------------
    # Test 1: Event-Driven Word Pacing (Zero Software Oscillator)
    # -------------------------------------------------------------
    print("--- Test 1: Event-Driven Syllable Pacing Across Words ---")
    test_words = [
        ("UP", 1),         # U (vowel) -> P (consonant) = 1 beat
        ("PETBOT", 2),     # P-E (beat 1) -> T-B-O (beat 2) -> T = 2 beats
        ("BANANA", 3),     # B-A (beat 1) -> N-A (beat 2) -> N-A (beat 3) = 3 beats
    ]

    for word, expected_syllables in test_words:
        bit0_seq = word_to_bit0_sequence(word, steps_per_char=3)
        pacing, resonance = tracker.run_cadence_simulation(bit0_seq)
        beat_count = int(np.sum(pacing))
        beat_times = np.where(pacing == 1)[0].tolist()

        print(f"Word: {word:8s} | Expected: {expected_syllables} beats | Detected: {beat_count} beats at t={beat_times}")
        assert beat_count == expected_syllables, f"Failed for {word}: expected {expected_syllables}, got {beat_count}"

    print("[PASS] Speech rhythm alone successfully paces syllable ticks without a master CPU clock.\n")

    # -------------------------------------------------------------
    # Test 2: Prosodic Duty-Cycle (Staccato vs. Prolonged Resonance)
    # -------------------------------------------------------------
    print("--- Test 2: Prosodic Cadence Discrimination (Staccato vs. Prolonged) ---")
    
    # Excited/Urgent Staccato: "GO-GO-GO" (rapid vowel pulses, 2 steps each)
    staccato_seq = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0])
    pacing_stac, res_stac = tracker.run_cadence_simulation(staccato_seq)
    stac_pacing_count = int(np.sum(pacing_stac))
    stac_res_count = int(np.sum(res_stac))
    stac_ratio = stac_res_count / max(1, stac_pacing_count)

    # Soothing/Calming Prolonged: "NOOOOO" (consonant + prolonged vowel, 8 steps)
    prolonged_seq = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    pacing_prol, res_prol = tracker.run_cadence_simulation(prolonged_seq)
    prol_pacing_count = int(np.sum(pacing_prol))
    prol_res_count = int(np.sum(res_prol))
    prol_ratio = prol_res_count / max(1, prol_pacing_count)

    print(f"Pattern: STACCATO ('GO GO GO') | Pacing Beats: {stac_pacing_count} | Resonance Spikes: {stac_res_count:2d} | Warmth Ratio: {stac_ratio:.2f}")
    print(f"Pattern: PROLONGED ('NOOOOO')  | Pacing Beats: {prol_pacing_count} | Resonance Spikes: {prol_res_count:2d} | Warmth Ratio: {prol_ratio:.2f}")

    assert prol_ratio > 3.0 * stac_ratio, "Prolonged resonance ratio must significantly exceed staccato ratio!"
    print("[PASS] Duty-cycle integrator cleanly distinguishes soothing resonance from rapid staccato commands.\n")

    # -------------------------------------------------------------
    # Test 3: Natural Conversational Cadence & Quiescent Zero-Power Silence
    # Stream: "HI" (beat 1) -> [PAUSE of 10 steps] -> "PETBOT" (beats 2 & 3)
    # -------------------------------------------------------------
    print("--- Test 3: Conversational Stream with Inter-Utterance Silence ---")
    hi_seq = word_to_bit0_sequence("HI", steps_per_char=3)
    pause_seq = np.zeros(12, dtype=int)
    petbot_seq = word_to_bit0_sequence("PETBOT", steps_per_char=3)
    stream_conv = np.concatenate([hi_seq, pause_seq, petbot_seq])

    pacing_conv, res_conv = tracker.run_cadence_simulation(stream_conv)
    total_conv_beats = int(np.sum(pacing_conv))
    pause_window_spikes = int(np.sum(pacing_conv[len(hi_seq):len(hi_seq) + 12]))

    print(f"Total Stream Steps  : {len(stream_conv)}")
    print(f"Total Beats Latched : {total_conv_beats} (Expected 3: 1 for 'HI', 2 for 'PETBOT')")
    print(f"Spikes During Pause : {pause_window_spikes} across 12 quiescent steps")
    assert total_conv_beats == 3
    assert pause_window_spikes == 0, "Spikes detected during conversational silence!"
    print("[PASS] Absolute zero-power silence verified during inter-utterance conversational pauses.")

    print("\n=================================================================")
    print("[SUCCESS] Experiment 3 Completed: Lava Cadence Tracker Verified!")
    print("=================================================================")

if __name__ == "__main__":
    main()
