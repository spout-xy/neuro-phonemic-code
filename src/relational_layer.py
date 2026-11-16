"""
relational_layer.py
Two-Word Telegraphic Relational Engine for Petbot.
Implements zero-parameter Vector Symbolic Architecture (VSA) binding:
    Relational_Token = Token_1 ^ rotate_16bit(Token_2, shift=5)
Synthesizes physical registers and interoceptive focus without high-power matrix multiplication.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
try:
    from compile_toddler_lexicon import compute_concept_token
except ImportError:
    from .compile_toddler_lexicon import compute_concept_token

def rotate_16bit(val: int, shift: int = 5) -> int:
    """Circular permutation across 16-bit space to strictly preserve telegram word order."""
    shift = shift % 16
    return ((val << shift) & 0xFFFF) | ((val >> (16 - shift)) & 0xFFFF)

@dataclass
class RelationalTelegram:
    word1: str
    word2: str
    relational_token: int
    focus_agent: str          # "ME", "YOU", "SHARED", or "ENVIRONMENT"
    net_valence: int          # -8 to +7
    is_inhibited: bool        # True if negated (e.g., "NO GO", "STOP PUSH")
    is_amplified: bool        # True if scaled (e.g., "VERY BIG")
    action_intent: str        # Synthesized motor/social directive

class RelationalLayer:
    def __init__(self):
        # Cache of compiled 16-bit tokens for fast lookup
        self.token_cache: Dict[str, int] = {}

    def get_token(self, word: str) -> int:
        word = word.upper()
        if word not in self.token_cache:
            self.token_cache[word] = compute_concept_token(word)
        return self.token_cache[word]

    def bind_telegram(self, word1: str, word2: str) -> int:
        """Computes deterministic 16-bit relational token using VSA circular shift."""
        t1 = self.get_token(word1)
        t2 = self.get_token(word2)
        # Order-preserving non-commutative binding: W1 ^ rho(W2)
        return t1 ^ rotate_16bit(t2, shift=5)

    def evaluate_telegram(self, word1: str, word2: str) -> RelationalTelegram:
        w1 = word1.upper()
        w2 = word2.upper()
        rel_token = self.bind_telegram(w1, w2)

        # 1. Focus Agent Resolution
        if w1 == "ME" or w2 == "ME":
            focus = "ME"
        elif w1 == "YOU" or w2 == "YOU":
            focus = "YOU"
        elif w1 in ("HUG", "PLAY", "NICE", "HELP") or w2 in ("HUG", "PLAY", "NICE", "HELP"):
            focus = "SHARED"
        else:
            focus = "ENVIRONMENT"

        # 2. Negation / Inhibition Flag
        is_inhibited = (w1 in ("NO", "STOP") or w2 in ("NO", "STOP"))

        # 3. Amplification / Scaling Flag
        is_amplified = (w1 in ("VERY", "BIG", "MORE") or w2 in ("VERY", "BIG", "MORE"))

        # 4. Net Valence Synthesis
        # Default baseline
        valence = 0
        if w1 == "HURT" or w2 == "HURT" or w1 == "OUCH" or w2 == "OUCH":
            valence = -8 if focus == "ME" else -5
        elif w1 in ("NICE", "HUG", "GOOD", "PLAY") or w2 in ("NICE", "HUG", "GOOD", "PLAY"):
            valence = 6
        elif w1 in ("BAD", "NO") or w2 in ("BAD", "NO"):
            valence = -4

        if is_amplified and valence > 0:
            valence = min(7, valence + 2)
        elif is_amplified and valence < 0:
            valence = max(-8, valence - 2)

        # 5. Action Directive Formulation
        if is_inhibited:
            action = f"INHIBIT / HALT ACTION ({w2})"
        elif focus == "ME" and valence < -4:
            action = f"SELF-PRESERVATION / DEFENSIVE BRACE ({w1} {w2})"
        elif focus == "YOU" and valence < 0:
            action = f"EMPATHIC SOOTHING / ATTUNEMENT ({w1} {w2})"
        elif focus in ("YOU", "SHARED") and valence > 4:
            action = f"AFFILIATIVE BONDING / PLAY ENGAGEMENT ({w1} {w2})"
        elif is_amplified:
            action = f"SCALE TRAJECTORY / INTENSIFY ({w1} {w2})"
        else:
            action = f"STANDARD EMBODIED DIRECTIVE ({w1} {w2})"

        return RelationalTelegram(
            word1=w1,
            word2=w2,
            relational_token=rel_token,
            focus_agent=focus,
            net_valence=valence,
            is_inhibited=is_inhibited,
            is_amplified=is_amplified,
            action_intent=action
        )

if __name__ == "__main__":
    engine = RelationalLayer()

    print("--- 1. Order Preservation (Non-Commutativity) ---")
    t_you_hug = engine.bind_telegram("YOU", "HUG")
    t_hug_you = engine.bind_telegram("HUG", "YOU")
    print(f"'YOU HUG': 0x{t_you_hug:04X} vs 'HUG YOU': 0x{t_hug_you:04X}")
    assert t_you_hug != t_hug_you, "Order symmetry failure! W1 ^ rho(W2) must be asymmetric."
    print("[PASS] Word order strictly preserved.")

    print("\n--- 2. Telephonic Telegram Evaluator ---")
    test_telegrams = [
        ("ME", "HURT"),    # Interoceptive distress -> self preservation
        ("YOU", "HUG"),    # Affiliative bonding
        ("NO", "GO"),      # Motor inhibition
        ("VERY", "BIG"),   # Magnitude scaling
        ("YOU", "HURT"),   # Empathic other-care
    ]

    for w1, w2 in test_telegrams:
        res = engine.evaluate_telegram(w1, w2)
        print(f"[{w1} {w2}] -> Token: 0x{res.relational_token:04X} | Focus: {res.focus_agent:5s} | Val: {res.net_valence:+2d} | Action: {res.action_intent}")

    # Specific assertions
    t_hurt = engine.evaluate_telegram("ME", "HURT")
    assert t_hurt.focus_agent == "ME" and t_hurt.net_valence == -8

    t_nogo = engine.evaluate_telegram("NO", "GO")
    assert t_nogo.is_inhibited == True

    t_vbig = engine.evaluate_telegram("VERY", "BIG")
    assert t_vbig.is_amplified == True

    print("\n[SUCCESS] Relational Layer: All telegraphic composition checks passed cleanly!")
