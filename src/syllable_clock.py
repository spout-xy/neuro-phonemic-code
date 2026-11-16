"""
syllable_clock.py
Tracks the alternating Bit-0 wave (0 <-> 1, where MSB 1 = Vowel Resonance, 0 = Consonant Obstruction)
directly across 6-bit Boolean coordinate streams.
Extracts syllable pulses and rhythmic cadence without ASCII text, dictionaries, or master CPU clocks.
"""
from typing import Dict, Optional, List, Tuple

# 6-Bit Boolean Coordinates (0x00 to 0x3F)
COORDINATES: Dict[str, int] = {
    "B": 0b011111, "P": 0b001111, "M": 0b010111, "F": 0b001110,
    "V": 0b011110, "W": 0b010110, "D": 0b011101, "T": 0b001100,
    "Z": 0b011100, "S": 0b000101, "N": 0b010100, "L": 0b010000,
    "R": 0b010101, "J": 0b011001, "G": 0b011011, "K": 0b001010,
    "H": 0b001000, "C": 0b001101, "Q": 0b001011, "X": 0b000110, "Y": 0b010001,
    "A": 0b100001, "E": 0b101000, "I": 0b111000, "O": 0b100101, "U": 0b110100
}

# Reverse lookup for debugging terminal displays only
REV_COORDINATES: Dict[int, str] = {v: k for k, v in COORDINATES.items()}

class SyllableClockFilter:
    def __init__(self):
        self.last_bit0: int = 0
        self.syllable_pulse_count: int = 0
        self.waveform_trace: List[str] = []
        self.pulse_events: List[Tuple[int, str]] = []

    def feed_coordinate(self, coord: int) -> dict:
        """
        Ingests a raw 6-bit binary coordinate (0x00 - 0x3F).
        Evaluates Bit 0 (MSB bit 5):
          1 = Vowel Resonance (Acoustic Opening)
          0 = Consonant Obstruction (Acoustic Barrier)
        """
        coord = coord & 0x3F
        bit0 = (coord >> 5) & 1
        event = None

        if self.last_bit0 == 0 and bit0 == 1:
            self.syllable_pulse_count += 1
            event = f"SYLLABLE_RISE (Pulse #{self.syllable_pulse_count})"
            self.waveform_trace.append("^")
            self.pulse_events.append((self.syllable_pulse_count, "RISE"))
        elif self.last_bit0 == 1 and bit0 == 0:
            event = "CONSONANT_CLOSURE"
            self.waveform_trace.append("v")
        elif bit0 == 1:
            event = "SUSTAINED_VOWEL"
            self.waveform_trace.append("-")
        else:
            event = "CONSONANT_CLUSTER"
            self.waveform_trace.append("_")

        self.last_bit0 = bit0
        return {
            "coord_hex": f"0x{coord:02X}",
            "coord_bin": bin(coord)[2:].zfill(6),
            "bit0": bit0,
            "event": event
        }

    def feed_character(self, char: str) -> Optional[dict]:
        """Convenience bridge for human testing strings."""
        char = char.upper()
        if char not in COORDINATES:
            self.last_bit0 = 0
            self.waveform_trace.append(" ")
            return None
        return self.feed_coordinate(COORDINATES[char])

    def get_visual_trace(self) -> str:
        return "".join(self.waveform_trace)

    def reset(self):
        self.last_bit0 = 0
        self.syllable_pulse_count = 0
        self.waveform_trace.clear()
        self.pulse_events.clear()

if __name__ == "__main__":
    clk = SyllableClockFilter()

    print("--- Test 1: Raw 6-Bit Binary Coordinate Stream ---")
    # Feed coordinates for "PETBOT" directly as 6-bit numbers:
    # P: 0b001111 (0x0F) -> bit0 = 0 (Consonant)
    # E: 0b101000 (0x28) -> bit0 = 1 (Vowel) -> PULSE #1
    # T: 0b001100 (0x0C) -> bit0 = 0 (Consonant)
    # B: 0b011111 (0x1F) -> bit0 = 0 (Consonant)
    # O: 0b100101 (0x25) -> bit0 = 1 (Vowel) -> PULSE #2
    # T: 0b001100 (0x0C) -> bit0 = 0 (Consonant)
    petbot_raw_coords = [0x0F, 0x28, 0x0C, 0x1F, 0x25, 0x0C]
    for c in petbot_raw_coords:
        out = clk.feed_coordinate(c)
        evt_str = f" -> {out['event']}" if out['event'] else ""
        print(f"Coord: {out['coord_hex']} ({out['coord_bin']}) | Bit-0: {out['bit0']}{evt_str}")

    print("Waveform Trace:", clk.get_visual_trace())
    assert clk.syllable_pulse_count == 2, f"Expected 2 pulses for PET-BOT, got {clk.syllable_pulse_count}"
    print("[PASS] Raw coordinate stream accurately detected 2 syllable pulses.")

    print("\n--- Test 2: Natural Rhythm Cadence Pacing ---")
    clk.reset()
    # Test word "BANANA" (3 syllables: BA-NA-NA)
    for ch in "BANANA":
        clk.feed_character(ch)
    print("Trace for 'BANANA':", clk.get_visual_trace())
    assert clk.syllable_pulse_count == 3
    print(f"[PASS] 'BANANA' correctly generated {clk.syllable_pulse_count} rising-edge pulses.")

    print("\n--- Test 3: Vowel Sustained Resonance ---")
    clk.reset()
    # Test elongated vowel "SOOO": S(0) -> O(1) -> O(1) -> O(1)
    for ch in "SOOO":
        clk.feed_character(ch)
    trace = clk.get_visual_trace()
    print("Trace for 'SOOO':", trace)
    assert clk.syllable_pulse_count == 1, "Elongated vowel must remain a single sustained pulse"
    assert "-" in trace, "Sustained resonance marker must be present"
    print("[PASS] Sustained vowel resonance correctly tracked without false syllable splits.")

    print("\n[SUCCESS] Syllable Clock Filter: All binary cadence and pulse checks passed cleanly!")
