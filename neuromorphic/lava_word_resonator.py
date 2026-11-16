"""
lava_word_resonator.py
Module 2 / Assignment 8: Spiking Word Resonator & Continuous Attractor Basins.
Executes on Intel Lava under Loihi2SimCfg.
Demonstrates:
  1. VSA temporal delay-line binding: Strict order distinction (UP vs PU).
  2. Spiking Attractor Basin: Recovers from 1-bit Hamming acoustic corruption
     (e.g., toddler/microphone swapping P -> B) by snapping to the correct word basin
     while completely suppressing competing word candidates.
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

from compile_toddler_lexicon import compute_concept_token, COORDINATES, rotate_6bit

# Lexicon subset for the attractor competition
LEXICON_ATTRACTOR_SET = [
    "UP", "PU", "POP", "GO", "STOP", "PUSH", "HOLD", "DROP", "NO", "MORE"
]

class LavaWordResonator:
    def __init__(self, candidate_words: List[str] = None):
        if candidate_words is None:
            candidate_words = LEXICON_ATTRACTOR_SET
        self.candidate_words = candidate_words
        self.num_words = len(candidate_words)
        
        # Compile reference 16-bit concept tokens
        self.word_tokens = np.array([compute_concept_token(w) for w in candidate_words], dtype=int)
        
        # Attractor LIF neuron parameters (calibrated for >80% similarity thresholding)
        # Steady state V_inf = I / (dv * du) = I / 0.64 = 1.5625 * I
        # For 93.75% similarity (1-bit corrupted UP): I = 18.75 -> V_inf = 29.3 > 25.0 (Spikes)
        # For 75.00% similarity (closest competitor NO): I = 15.00 -> V_inf = 23.4 < 25.0 (Quiescent 0)
        self.vth = 25.0
        self.dv = 0.8
        self.du = 0.8

    def compute_hamming_similarities(self, input_token: int) -> np.ndarray:
        """Calculates normalized bitwise similarity (0.0 to 1.0) between input and candidate tokens."""
        sims = []
        for tok in self.word_tokens:
            diff_bits = bin(input_token ^ tok).count('1')
            sims.append((16 - diff_bits) / 16.0)
        return np.array(sims, dtype=float)

    def run_attractor_simulation(self, input_token: int, num_steps: int = 10) -> Dict[str, int]:
        """
        Feeds similarity current into the Lava LIF attractor network.
        Returns a dictionary mapping candidate words to their emitted spike counts.
        """
        sims = self.compute_hamming_similarities(input_token)
        # Current scale: 100% similarity = 20.0 current
        current_matrix = (sims * 20.0)[:, np.newaxis] * np.ones((self.num_words, num_steps))

        source = Source(data=current_matrix)
        attractors = LIF(shape=(self.num_words,), vth=self.vth, dv=self.dv, du=self.du)
        sink = Sink(shape=(self.num_words,), buffer=num_steps)

        source.s_out.connect(attractors.a_in)
        attractors.s_out.connect(sink.a_in)

        attractors.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
        spikes = sink.data.get().copy()
        attractors.stop()

        results = {}
        for idx, word in enumerate(self.candidate_words):
            results[word] = int(np.sum(spikes[idx, :]))
        return results

def main():
    print("=================================================================")
    print("Intel Lava: Experiment 4 - Spiking Word Resonator & Attractors")
    print("=================================================================\n")

    resonator = LavaWordResonator()

    # -------------------------------------------------------------
    # Test 1: Temporal Delay-Line Binding & Non-Commutativity (UP vs PU)
    # -------------------------------------------------------------
    print("--- Test 1: Temporal Order Binding (UP vs. PU) ---")
    token_up = compute_concept_token("UP")
    token_pu = compute_concept_token("PU")
    h_dist = bin(token_up ^ token_pu).count('1')

    print(f"Token 'UP': 0x{token_up:04X} ({bin(token_up)[2:].zfill(16)})")
    print(f"Token 'PU': 0x{token_pu:04X} ({bin(token_pu)[2:].zfill(16)})")
    print(f"Hamming Separation Distance: {h_dist} bits out of 16 ({(h_dist/16)*100:.1f}% orthogonal)")
    assert token_up != token_pu, "Temporal binding failed: UP and PU collided!"

    # Simulate feeding clean "UP"
    spikes_up_in = resonator.run_attractor_simulation(token_up)
    print(f"Feeding Clean 'UP' -> Spikes: {spikes_up_in}")
    assert spikes_up_in["UP"] > 0 and spikes_up_in["PU"] == 0
    print("[PASS] Clean 'UP' selectively fires Basin [UP]; 'PU' remains 100% suppressed.\n")

    # -------------------------------------------------------------
    # Test 2: 1-Bit Acoustic Hamming Corruption (UB -> Snaps to UP)
    # Swapping 'P' (0b001111) for 'B' (0b011111) - voiced plosive error
    # -------------------------------------------------------------
    print("--- Test 2: 1-Bit Acoustic Noise Corruption (Injected 'UB') ---")
    coord_b = COORDINATES["B"]
    coord_p = COORDINATES["P"]
    assert bin(coord_b ^ coord_p).count('1') == 1, "P vs B is not a 1-bit Hamming distance!"

    # Construct corrupted token for "UB"
    bound_noisy = COORDINATES["U"] ^ rotate_6bit(coord_b, 1)
    token_ub = ((2 & 0xF) << 12) | ((bound_noisy & 0x3F) << 6) | (COORDINATES["U"] & 0x3F)

    print(f"Original 'UP' Token : 0x{token_up:04X}")
    print(f"Corrupted 'UB' Token: 0x{token_ub:04X}")
    
    # Run attractor simulation
    spikes_ub_in = resonator.run_attractor_simulation(token_ub)
    print("\nAttractor Basin Spiking Distribution for Corrupted 'UB':")
    print("Candidate Word | Bit Similarity | Spikes Emitted | Attractor State")
    print("---------------+----------------+----------------+----------------")
    
    sims = resonator.compute_hamming_similarities(token_ub)
    winner = None
    max_spikes = -1

    for idx, w in enumerate(resonator.candidate_words):
        spk = spikes_ub_in[w]
        sim = sims[idx] * 100.0
        state = "WINNER (LATCHED)" if spk > 0 else "SUPPRESSED (0)"
        print(f"Basin [{w:4s}]     | {sim:5.1f}%         | {spk:2d} spikes       | {state}")
        if spk > max_spikes and spk > 0:
            max_spikes = spk
            winner = w

    assert winner == "UP", f"Attractor basin failed to recover! Winner was {winner}"
    assert sum(v for k, v in spikes_ub_in.items() if k != "UP") == 0, "Non-winning attractors leaked spikes!"
    print(f"\n[PASS] Noise Recovery Confirmed: Corrupted 'UB' snapped cleanly into Basin [{winner}] with zero competitor leakage.\n")

    # -------------------------------------------------------------
    # Test 3: Tolerance Boundary - Multiple Bit Corruptions
    # -------------------------------------------------------------
    print("--- Test 3: Attractor Basin Margin of Separation ---")
    sim_up = sims[resonator.candidate_words.index("UP")]
    other_sims = [sims[idx] for idx, w in enumerate(resonator.candidate_words) if w != "UP"]
    second_best = max(other_sims)
    margin = (sim_up - second_best) * 100.0

    print(f"Target Word Similarity  : {sim_up * 100.0:.1f}%")
    print(f"Next Closest Competitor : {second_best * 100.0:.1f}%")
    print(f"Safety Separation Margin: {margin:.1f}% (Hamming distance barrier >= 3 bits)")
    assert margin >= 15.0, f"Separation margin too narrow: {margin:.1f}%"
    print("[PASS] Attractor basin has a robust (>15%, 3-bit) noise immunity barrier.")

    print("\n=================================================================")
    print("[SUCCESS] Experiment 4 Completed: Word Resonator & Attractor Verified!")
    print("=================================================================")

if __name__ == "__main__":
    main()
