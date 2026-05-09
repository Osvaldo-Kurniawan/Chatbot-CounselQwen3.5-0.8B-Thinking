"""
tokenizer.py
Initializes and exports the HuggingFace tokenizer.
Swap MODEL_NAME to your custom model path or Hub repo ID.
"""

from __future__ import annotations
from transformers import AutoTokenizer

# ── Set this to your custom HuggingFace model path or repo ID ─────────────────
MODEL_NAME = "tegarganang/CounselQwen3.5-0.8B-Thinking"
# ──────────────────────────────────────────────────────────────────────────────

_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """
    Returns a singleton tokenizer instance.
    Loads from MODEL_NAME on first call; returns the cached instance afterwards.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Most causal-LM tokenizers need a pad token
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer