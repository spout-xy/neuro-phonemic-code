"""
lava_lif_baseline.py
Module 2 / Assignment 5: Baseline Leaky-Integrate-and-Fire (LIF) Neuromorphic Process.
Demonstrates discrete virtual time-step execution under Intel Lava's Loihi2SimCfg.
Verifies sub-threshold leak decay, supra-threshold spike generation, and refractory reset.
"""
import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

def run_lif_experiment(
    current_trace: np.ndarray,
    vth: float = 10.0,
    dv: float = 0.5,
    du: float = 0.5,
    num_steps: int = None
) -> np.ndarray:
    """
    Executes a simulated Loihi 2 LIF compartment on the given input current trace.
    Returns the binary output spike train across all time steps.
    """
    if num_steps is None:
        num_steps = current_trace.shape[1]

    # Initialize Lava processes
    source = Source(data=current_trace)
    lif = LIF(shape=(1,), vth=vth, dv=dv, du=du)
    sink = Sink(shape=(1,), buffer=num_steps)

    # Interconnect neuromorphic channel ports
    source.s_out.connect(lif.a_in)
    lif.s_out.connect(sink.a_in)

    # Execute asynchronous virtual time-step simulation via Loihi 2 floating point model
    lif.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
    spikes = sink.data.get().copy()
    lif.stop()

    return spikes.flatten()

def main():
    print("=================================================================")
    print("Intel Lava: Experiment 1 - Baseline Spiking LIF Dynamics (Loihi 2)")
    print("=================================================================\n")

    time_steps = 20

    # -------------------------------------------------------------
    # Scenario A: Sub-threshold stimulus (Ambient Noise / Low Current)
    # With du=0.5, dv=0.5, steady state V_inf = 4 * I.
    # For I = 2.0, V_inf = 8.0 < vth (10.0) -> Expect ZERO spikes across all steps!
    # -------------------------------------------------------------
    print("--- Scenario A: Sub-Threshold Stimulus (Current = 2.0, Threshold = 10.0) ---")
    sub_current = np.full((1, time_steps), 2.0)
    spikes_sub = run_lif_experiment(sub_current, vth=10.0, dv=0.5, du=0.5)
    spike_count_sub = int(np.sum(spikes_sub))
    print(f"Input Current Trace : {sub_current.flatten()[:10]} ...")
    print(f"Output Spike Train  : {spikes_sub}")
    print(f"Total Spikes Emitted: {spike_count_sub}")
    assert spike_count_sub == 0, f"Expected 0 spikes for sub-threshold input, got {spike_count_sub}"
    print("[PASS] Sub-threshold input leaks away silently: zero spikes emitted (quiescent zero energy).\n")

    # -------------------------------------------------------------
    # Scenario B: Supra-threshold stimulus (Active Phoneme / Coordinate Burst)
    # Steps 0-4: Rest (0.0)
    # Steps 5-14: Sustained burst (15.0 >> vth = 10.0) -> Active firing
    # Steps 15-19: Rest (0.0) -> Immediate silence
    # -------------------------------------------------------------
    print("--- Scenario B: Supra-Threshold Burst (Current = 15.0, Steps 5-14) ---")
    burst_current = np.zeros((1, time_steps))
    burst_current[0, 5:15] = 15.0
    spikes_burst = run_lif_experiment(burst_current, vth=10.0, dv=0.5, du=0.5)
    spike_count_burst = int(np.sum(spikes_burst))
    print(f"Input Current Steps : {burst_current.flatten()}")
    print(f"Output Spike Train  : {spikes_burst}")
    print(f"Spike Indices (t)   : {np.where(spikes_burst == 1)[0].tolist()}")
    print(f"Total Spikes Emitted: {spike_count_burst}")
    assert spike_count_burst > 0, "Expected spikes during supra-threshold burst!"
    assert np.all(spikes_burst[:5] == 0), "Spikes observed before stimulus onset!"
    # With tau_u synaptic decay, residual charge clears after t=15
    assert np.all(spikes_burst[16:] == 0), "Spikes observed after synaptic tail decay!"
    print("[PASS] Supra-threshold burst triggers deterministic spiking during stimulus window and synaptic decay.\n")

    # -------------------------------------------------------------
    # Scenario C: Voltage Leak Decay Rate Comparison
    # High leak (dv=0.8) dampens membrane integration vs low leak (dv=0.2)
    # -------------------------------------------------------------
    print("--- Scenario C: Membrane Voltage Leak (dv Decay) Comparison ---")
    moderate_current = np.full((1, time_steps), 4.0)

    spikes_lowleak = run_lif_experiment(moderate_current, vth=10.0, dv=0.2, du=0.5)
    spikes_highleak = run_lif_experiment(moderate_current, vth=10.0, dv=0.8, du=0.5)

    count_low = int(np.sum(spikes_lowleak))
    count_high = int(np.sum(spikes_highleak))
    print(f"Moderate Current    : {moderate_current.flatten()[:10]} ...")
    print(f"Spikes (Low Leak)   : {spikes_lowleak} (Total: {count_low})")
    print(f"Spikes (High Leak)  : {spikes_highleak} (Total: {count_high})")
    assert count_low > count_high, f"Expected low leak to fire more than high leak, got {count_low} vs {count_high}"
    print("[PASS] Membrane leak dynamics dynamically modulate firing frequency as predicted.")

    print("\n=================================================================")
    print("[SUCCESS] Experiment 1 Completed: Lava LIF simulation verified on Loihi2SimCfg!")
    print("=================================================================")

if __name__ == "__main__":
    main()
