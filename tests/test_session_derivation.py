"""Tests for per-session derivation of (S, M, A).

Covered properties:
  1. determinism: same (key, nonce) -> bitwise identical matrices;
  2. isolation: different nonce or key -> independent matrices;
  3. functional round-trip through SessionKVCloak (both fused modes);
  4. cross-session deobfuscation produces garbage.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from defense.core.keyderive import (
    SessionKVCloak,
    derive_session_config,
    new_session_nonce,
    session_seed,
)
from tests.test_kvcloak_markers import row_multiset_diff

B, D, LAYERS, HEADS = 8, 16, 2, 3
KEY = b"unit-test-user-key"


def theta_config(theta=10.0):
    return {
        f"layer_{i}": {
            "key_max_values": [theta] * HEADS,
            "value_max_values": [theta] * HEADS,
        }
        for i in range(LAYERS)
    }


def make_session(nonce, **kw):
    return SessionKVCloak(
        user_key=KEY,
        session_nonce=nonce,
        num_hidden_layers=LAYERS,
        num_key_value_heads=HEADS,
        head_dim=D,
        block_size=B,
        theta_config=theta_config(),
        dtype=torch.float32,
        **kw,
    )


def make_cache(layers, heads, seq, d):
    torch.manual_seed(123)
    from transformers.cache_utils import DynamicCache

    legacy = []
    for _ in range(layers):
        k = torch.randn(1, heads, seq, d)
        v = torch.randn(1, heads, seq, d)
        legacy.append((k, v))
    return DynamicCache.from_legacy_cache(tuple(legacy))


# ---------------------------------------------------------------- 1 & 2 ----
def test_derivation_is_deterministic():
    c1 = derive_session_config(
        LAYERS, HEADS, D, B, theta_config(),
        user_key=KEY, session_nonce=b"nonce-1",
    )
    c2 = derive_session_config(
        LAYERS, HEADS, D, B, theta_config(),
        user_key=KEY, session_nonce=b"nonce-1",
    )
    for h1, h2 in zip(c1[0][0], c2[0][0]):
        for key in ("S", "M_angles", "S_ratios", "M_ratios", "A"):
            assert torch.equal(h1[key], h2[key]), f"{key} must be bitwise identical"


def test_different_nonce_or_key_gives_independent_matrices():
    base = derive_session_config(
        LAYERS, HEADS, D, B, theta_config(),
        user_key=KEY, session_nonce=b"nonce-1",
    )
    other_nonce = derive_session_config(
        LAYERS, HEADS, D, B, theta_config(),
        user_key=KEY, session_nonce=b"nonce-2",
    )
    other_key = derive_session_config(
        LAYERS, HEADS, D, B, theta_config(),
        user_key=b"another-user-key", session_nonce=b"nonce-1",
    )
    for other in (other_nonce, other_key):
        assert not torch.equal(base[0][0][0]["S"], other[0][0][0]["S"])
        assert not torch.equal(base[0][0][1]["A"], other[0][0][1]["A"])
    assert session_seed(KEY, b"n1", "g") != session_seed(KEY, b"n2", "g")
    assert session_seed(KEY, b"n1", "g") != session_seed(KEY2 := b"k2", b"n1", "g")


def test_session_nonces_are_unique():
    nonces = {new_session_nonce() for _ in range(128)}
    assert len(nonces) == 128


# -------------------------------------------------------------------- 3 ----
@pytest.mark.parametrize("fused", [True, False])
def test_roundtrip_same_session(fused):
    cloak = make_session(b"roundtrip-nonce", fused=fused)
    cache = make_cache(LAYERS, HEADS, 32, D)
    snap = tuple((k.clone(), v.clone()) for k, v in cache)
    rec = cloak.deobfuscate(cloak.obfuscate(cache))
    for (k0, v0), (k1, v1) in zip(snap, rec):
        assert k1.shape == k0.shape
        assert row_multiset_diff(k0, k1) < 1e-3
        assert row_multiset_diff(v0, v1) < 1e-3


# -------------------------------------------------------------------- 4 ----
def test_cross_session_deobfuscation_is_garbage():
    cloak_a = make_session(b"nonce-A")
    cloak_b = make_session(b"nonce-B")
    cache = make_cache(LAYERS, HEADS, 32, D)
    snap = tuple((k.clone(), v.clone()) for k, v in cache)
    protected = cloak_a.obfuscate(cache)
    rec = cloak_b.deobfuscate(protected)
    garbage = True
    for (k0, _), (k1, _) in zip(snap, rec):
        if k1.shape == k0.shape:
            garbage = garbage and row_multiset_diff(k0, k1) > 1.0
        # shape mismatch alone already proves the failure
    assert garbage, "another session's matrices must not decode this cache"
