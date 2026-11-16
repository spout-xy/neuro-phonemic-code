"""
memory_vault.py
Episodic memory store serializing 16-byte fixed binary records (<IBBHQ)
with O(1) seek indexing via a 64-slot spatial jump table.
"""
import struct
import time
from typing import Dict, List

class MemoryVault:
    def __init__(self, filename: str = "petbot_vault.bin"):
        self.filename = filename
        # Layout: uint32 ts (4B), uint8 coord (1B), uint8 emotion (1B), uint16 beats (2B), uint64 ctx (8B) = 16B
        self.record_struct = struct.Struct("<IBBHQ")
        self.spatial_jump_table: Dict[int, List[int]] = {slot: [] for slot in range(64)}
        self.record_count: int = 0

    def commit_event(self, coord: int, valence: int, arousal: int, beats: int, context_id: int = 0, timestamp: int = None) -> bytes:
        ts = timestamp if timestamp is not None else int(time.time())
        coord = coord & 0x3F
        packed_emotion = (((valence + 8) & 0x0F) << 4) | (arousal & 0x0F)
        binary_record = self.record_struct.pack(ts, coord, packed_emotion, beats, context_id)

        with open(self.filename, "ab") as f:
            f.write(binary_record)

        byte_offset = self.record_count * self.record_struct.size
        self.spatial_jump_table[coord].append(byte_offset)
        self.record_count += 1
        return binary_record

    def seek_record_by_index(self, index: int) -> dict:
        offset = index * self.record_struct.size
        with open(self.filename, "rb") as f:
            f.seek(offset)
            data = f.read(self.record_struct.size)
            ts, coord, emo, beats, ctx = self.record_struct.unpack(data)
            return {
                "index": index,
                "byte_offset": offset,
                "timestamp": ts,
                "coord_hex": f"0x{coord:02X}",
                "valence": ((emo >> 4) & 0x0F) - 8,
                "arousal": emo & 0x0F,
                "beats": beats,
                "context_id": f"0x{ctx:04X}"
            }

if __name__ == "__main__":
    vault = MemoryVault("test_petbot_vault.bin")
    vault.commit_event(coord=0x1F, valence=5, arousal=3, beats=2, context_id=0xA101)
    rec = vault.seek_record_by_index(0)
    print("Record #0 Recalled:", rec)
    assert rec["coord_hex"] == "0x1F"
    assert rec["valence"] == 5
    print("Verification Passed: 16-byte pack & seek works cleanly!")
