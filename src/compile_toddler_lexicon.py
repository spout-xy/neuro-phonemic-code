"""
compile_toddler_lexicon.py
Generates the immutable 368-byte binary table (kernel_lexicon.bin)
for Petbot's 46 toddler words using 6-bit coordinates and VSA temporal delay binding.
"""
import struct

# 6-Bit Coordinates: [b0:Vowel, b1:Voiced, b2:Turbulent, b3:Zone_H, b4:Zone_L, b5:Curvature]
CONSONANTS = {
    "B": 0b011111, "P": 0b001111, "M": 0b010111, "W": 0b010110,
    "F": 0b001110, "V": 0b011110, "D": 0b011101, "T": 0b001100,
    "Z": 0b011100, "S": 0b000101, "N": 0b010100, "L": 0b010000,
    "R": 0b010101, "J": 0b011001, "G": 0b011011, "K": 0b001010,
    "H": 0b001000, "C": 0b001101, "Q": 0b001011, "X": 0b000110,
    "Y": 0b010001
}

VOWELS = {
    "A": 0b100001, "E": 0b101000, "I": 0b111000,
    "O": 0b100101, "U": 0b110100
}

COORDINATES = {**CONSONANTS, **VOWELS}

def rotate_6bit(val: int, shift: int) -> int:
    """Circular permutation (rho) across 6-bit space."""
    shift = shift % 6
    return ((val << shift) & 0x3F) | ((val >> (6 - shift)) & 0x3F)

def compute_concept_token(word: str) -> int:
    """Computes deterministic 16-bit concept token via temporal VSA binding.
    16-bit Token Layout: [15:12] length (4b) | [11:6] VSA bound vector (6b) | [5:0] root coordinate (6b)
    """
    bound_vector = 0
    root_coord = COORDINATES.get(word[0].upper(), 0)
    for t, char in enumerate(word.upper()):
        coord = COORDINATES.get(char, 0)
        bound_vector ^= rotate_6bit(coord, t)
    return ((len(word) & 0x0F) << 12) | ((bound_vector & 0x3F) << 6) | (root_coord & 0x3F)

TODDLER_RAW_LEXICON = [
    ("UP", 0, 7, 0xCA, 2), ("DOWN", 0, -7, 0x62, 0), ("IN", 2, 0, 0x4F, 1),
    ("OUT", 6, 0, 0x20, 1), ("GO", 7, 0, 0x10, 5), ("STOP", 0, 0, 0xFF, 0),
    ("PUSH", 7, 0, 0xE4, -1), ("POP", 1, 5, 0xFC, 4), ("HOLD", 0, 0, 0x8E, 2),
    ("DROP", 0, -6, 0x21, -2), ("YES", 4, 4, 0x12, 6), ("NO", -5, 0, 0xCE, -3),
    ("MORE", 5, 2, 0x26, 4), ("DONE", 0, -2, 0xAF, 1), ("UHOH", -2, -3, 0x68, -5),
    ("LOOK", 6, 4, 0x46, 5), ("COME", 4, 0, 0x30, 3), ("BYE", -4, 0, 0x15, 1),
    ("HI", 2, 3, 0x10, 6), ("EAT", 1, 0, 0x88, 5), ("DRINK", 2, -1, 0x72, 4),
    ("SLEEP", 0, -5, 0x05, 5), ("HUG", 2, 1, 0x6E, 7), ("KISS", 1, 2, 0x5C, 7),
    ("PLAY", 5, 5, 0x3B, 6), ("FALL", 0, -7, 0x12, -4), ("OUCH", 0, 0, 0x99, -7),
    ("HOT", 0, 0, 0x45, -3), ("COLD", 0, 0, 0x82, -2), ("WET", 1, -2, 0x31, 0),
    ("OPEN", 5, 0, 0x08, 3), ("SHUT", -5, 0, 0xF1, -1), ("BIG", 6, 6, 0x40, 3),
    ("TINY", -6, -6, 0x02, 2), ("LOUD", 0, 8, 0x70, 1), ("SHH", 0, -8, 0x01, 3),
    ("FAST", 8, 0, 0x18, 4), ("SLOW", -8, 0, 0x90, 0), ("HELP", 3, 3, 0x8A, 2),
    ("WAIT", 0, 0, 0x9C, -1), ("MINE", 1, 0, 0xEE, 2), ("GIVE", 5, 0, 0x14, 4),
    ("SEE", 4, 2, 0x11, 4), ("HEAR", 3, 3, 0x19, 4), ("RUN", 8, 0, 0x24, 5),
    ("SIT", 0, -4, 0x83, 1)
]

def compile_lexicon(filename="kernel_lexicon.bin"):
    tokens_seen = {}
    collisions = []
    
    with open(filename, "wb") as f:
        for word, dx, dy, res_bnd, val in TODDLER_RAW_LEXICON:
            token = compute_concept_token(word)
            if token in tokens_seen:
                collisions.append((word, tokens_seen[token], hex(token)))
            tokens_seen[token] = word
            
            root = COORDINATES[word[0]]
            # 8 bytes: token(2, uint16), root(1, uint8), dx(1, int8), dy(1, int8), res_bnd(1, uint8), val(1, int8), pad(1, uint8)
            packed = struct.pack("<HBbbBbB", token, root, dx, dy, res_bnd, val, 0)
            f.write(packed)

    if collisions:
        print(f"[ERROR] Found {len(collisions)} collision(s): {collisions}")
        raise ValueError("Token collisions detected during lexicon compilation.")
    else:
        print(f"[SUCCESS] Collision Check Passed: All {len(tokens_seen)} unique words mapped to distinct 16-bit concept tokens.")
        
    print(f"[SUCCESS] Compiled {len(TODDLER_RAW_LEXICON)} words into '{filename}' ({len(TODDLER_RAW_LEXICON) * 8} bytes).")

if __name__ == "__main__":
    compile_lexicon()
