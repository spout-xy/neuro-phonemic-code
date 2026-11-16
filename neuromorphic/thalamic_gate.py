"""
thalamic_gate.py
Module 2 / Assignment 6: Thalamic Noise-Shield Process in Intel Lava.
Filters analog room noise across a 6-channel neuromorphic sensory bus.
Rejects sub-threshold ambient fluctuations via membrane leak (tau-decay)
and emits synchronous 6-bit Address-Event Representation (AER) spike packets
strictly upon coordinated phoneme bursts.
"""
import numpy as np
from typing import Tuple, List, Dict
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

# 6-Bit Coordinates: [b5(MSB):Vowel, b4:Voiced, b3:Turbulent, b2:Zone_H, b1:Zone_L, b0:Curvature]
COORDINATES: Dict[str, int] = {
    "U": 0b110100,  # 0x34: High back vowel
    "P": 0b001111,  # 0x0F: Voiceless labial plosive stop
    "M": 0b010111,  # 0x17: Bilabial nasal flow
    "S": 0b000101,  # 0x05: Alveolar fricative
}

class ThalamicNoiseShield:
    def __init__(self, vth: float = 10.0, dv: float = 0.7, du: float = 0.7):
        self.num_channels = 6
        self.vth = vth
        self.dv = dv
        self.du = du

    def simulate_channel_stream(self, input_current_matrix: np.ndarray) -> np.ndarray:
        """
        Runs the 6-channel LIF Thalamic Gate under Loihi2SimCfg.
        input_current_matrix: shape (6, num_steps)
        Returns: output binary spike matrix shape (6, num_steps)
        """
        num_steps = input_current_matrix.shape[1]

        source = Source(data=input_current_matrix)
        gate = LIF(shape=(self.num_channels,), vth=self.vth, dv=self.dv, du=self.du)
        sink = Sink(shape=(self.num_channels,), buffer=num_steps)

        # Connect ports across all 6 coordinate channels
        source.s_out.connect(gate.a_in)
        gate.s_out.connect(sink.a_in)

        gate.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
        spikes = sink.data.get().copy()
        gate.stop()

        return spikes

    @staticmethod
    def decode_aer_coordinate(spike_column: np.ndarray) -> int:
        """Decodes a 6-channel spike column at time t into a 6-bit integer (0x00 - 0x3F)."""
        val = 0
        for bit_idx in range(6):
            if spike_column[bit_idx] > 0.5:
                val |= (1 << (5 - bit_idx))
        return val

def main():
    print("=================================================================")
    print("Intel Lava: Experiment 2 - Thalamic Noise-Shield & AER Gating")
    print("=================================================================\n")

    gate = ThalamicNoiseShield(vth=10.0, dv=0.7, du=0.7)
    np.random.seed(42) # Deterministic noise seed for repeatability

    # -------------------------------------------------------------
    # Test 1: Pure Ambient Room Rumble (Noise-Shield Attenuation)
    # -------------------------------------------------------------
    print("--- Test 1: Continuous Ambient Noise Floor (30 Time-Steps) ---")
    num_steps_noise = 30
    ambient_noise = np.random.uniform(0.5, 1.8, (6, num_steps_noise)) # Sub-threshold rumble
    noise_spikes = gate.simulate_channel_stream(ambient_noise)
    total_false_spikes = int(np.sum(noise_spikes))

    print(f"Noise Current Range : [{np.min(ambient_noise):.2f}, {np.max(ambient_noise):.2f}] across 6 channels")
    print(f"Total False Spikes  : {total_false_spikes} across {num_steps_noise * 6} neuron-steps")
    assert total_false_spikes == 0, f"Noise shield breached! Got {total_false_spikes} false spikes."
    print("[PASS] Noise-Shield 100% Effective: Zero false-alarm spikes emitted during ambient noise.\n")

    # -------------------------------------------------------------
    # Test 2: Synchronized Speech Bursts Embedded in Noise ("UP" = U -> P)
    # Timeline:
    #   t=0..9   : Ambient Noise Floor
    #   t=10..14 : Vowel 'U' (0b110100 = 0x34) Burst
    #   t=15..19 : Co-articulation Inter-Phoneme Gap (Noise only)
    #   t=20..24 : Plosive 'P' (0b001111 = 0x0F) Burst
    #   t=25..34 : Post-Speech Quiet (Noise only)
    # -------------------------------------------------------------
    print("--- Test 2: Ingestion of 'UP' (U 0x34 -> P 0x0F) Embedded in Noise ---")
    total_steps = 35
    stream = np.random.uniform(0.5, 1.8, (6, total_steps)) # Continuous background noise

    # Inject 'U' (0b110100) at t=10..14
    coord_u = COORDINATES["U"]
    for b in range(6):
        if (coord_u >> (5 - b)) & 1:
            stream[b, 10:15] += 15.0

    # Inject 'P' (0b001111) at t=20..24
    coord_p = COORDINATES["P"]
    for b in range(6):
        if (coord_p >> (5 - b)) & 1:
            stream[b, 20:25] += 15.0

    stream_spikes = gate.simulate_channel_stream(stream)

    # Decode latched AER coordinates at key timestamps
    latched_coords = []
    print("Time | Injected Event     | Latched AER Coordinate | Decoded Hex | Match?")
    print("-----+--------------------+------------------------+-------------+-------")

    for t in range(total_steps):
        col = stream_spikes[:, t]
        if np.any(col > 0):
            dec_coord = gate.decode_aer_coordinate(col)
            latched_coords.append((t, dec_coord))
            bin_str = bin(dec_coord)[2:].zfill(6)
            hex_str = f"0x{dec_coord:02X}"
            
            if t in range(10, 16):
                match = "YES (U)" if dec_coord == coord_u else "NO"
                print(f"t={t:02d} | Vowel 'U' Burst    | {bin_str}                 | {hex_str}        | {match}")
            elif t in range(20, 26):
                match = "YES (P)" if dec_coord == coord_p else "NO"
                print(f"t={t:02d} | Plosive 'P' Burst  | {bin_str}                 | {hex_str}        | {match}")

    # Assertions
    # Verify zero spikes during quiet intervals (0..9, 15..19, 25..34)
    quiet_spikes = np.sum(stream_spikes[:, 0:10]) + np.sum(stream_spikes[:, 15:20]) + np.sum(stream_spikes[:, 25:35])
    assert quiet_spikes == 0, f"False alarm spikes detected in quiet windows! Count: {quiet_spikes}"

    # Verify exact coordinate recovery during bursts
    u_decoded = gate.decode_aer_coordinate(stream_spikes[:, 10])
    p_decoded = gate.decode_aer_coordinate(stream_spikes[:, 20])
    assert u_decoded == coord_u, f"Failed to latch 'U': expected {hex(coord_u)}, got {hex(u_decoded)}"
    assert p_decoded == coord_p, f"Failed to latch 'P': expected {hex(coord_p)}, got {hex(p_decoded)}"

    print("\n[PASS] Noise-Shield Gating Verified: Perfectly latched U (0x34) and P (0x0F) with zero false triggers.")

    # -------------------------------------------------------------
    # Test 3: Critical Noise Ceiling Sweep
    # -------------------------------------------------------------
    print("\n--- Test 3: Signal-to-Noise Ratio (SNR) Noise Ceiling Sweep ---")
    noise_levels = [1.0, 2.0, 3.0, 4.0, 4.8, 5.5]
    print("Noise Amplitude | Steady-State Voltage (V_inf) | False Spikes Emitted | Shield Status")
    print("----------------+------------------------------+----------------------+--------------")

    for nl in noise_levels:
        noise_mat = np.full((6, 20), nl)
        spks = gate.simulate_channel_stream(noise_mat)
        count = int(np.sum(spks))
        v_inf = nl / (0.7 * 0.7) # V_inf = I / (dv * du)
        status = "SECURE (100% Reject)" if count == 0 else "BREACHED"
        print(f"I_noise = {nl:4.1f}   | V_inf = {v_inf:4.1f} (Vth=10.0)      | {count:4d} spikes emitted  | {status}")

    print("\n=================================================================")
    print("[SUCCESS] Experiment 2 Completed: Thalamic Noise-Shield Verified in Lava!")
    print("=================================================================")

if __name__ == "__main__":
    main()
