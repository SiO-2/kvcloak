"""Unit tests for KV-Cloak staggered markers.

The deobfuscator cannot restore the random within-block row permutation P
(by design: K and V are permuted synchronously, which attention is
invariant to). Correctness is therefore checked permutation-invariantly:
the multiset of recovered rows must equal the multiset of original rows,
sequence length must be exactly restored (padding removed), and the
caller's cache must not be mutated.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from defense.core.kvcloak import KVCloak, create_test_kv_config

B = 8
D = 16
LAYERS, HEADS = 2, 3


def make_protector(fused, device):
    torch.manual_seed(7)
    # theta = 2.5 * expected max of N(0,1) data over this many samples
    theta = [10.0] * (2 * HEADS)
    cfg = create_test_kv_config(LAYERS, HEADS, theta, 1.0, 1.0, B, D, device, torch.float32)
    return KVCloak(cfg, torch.float32, fused, False, True), cfg


def make_cache(batch, seq, device):
    torch.manual_seed(123)
    legacy = []
    for _ in range(LAYERS):
        k = torch.randn(batch, HEADS, seq, D, device=device)
        v = torch.randn(batch, HEADS, seq, D, device=device)
        legacy.append((k, v))
    from transformers.cache_utils import DynamicCache

    return DynamicCache.from_legacy_cache(tuple(legacy))


def row_multiset_diff(x, y):
    """Max diff between x and y up to row permutation (rows may only be
    permuted within blocks and padding rows removed, so comparing the
    multiset of all rows is exact and works for non-multiple lengths).

    Rows are matched greedily by nearest neighbor: unlike sorting by a
    scalar key, this cannot mis-pair rows whose keys are within fp noise.
    """
    assert x.shape == y.shape, f"shape mismatch {x.shape} vs {y.shape}"
    Bx, H, L, d = x.shape
    worst = 0.0
    for b in range(Bx):
        for h in range(H):
            cost = torch.cdist(y[b, h], x[b, h])  # (L, L)
            used = torch.zeros(L, dtype=torch.bool)
            for i in range(L):
                c = cost[i].clone()
                c[used] = float("inf")
                j = int(c.argmin())
                used[j] = True
                worst = max(worst, (y[b, h, i] - x[b, h, j]).abs().max().item())
    return worst


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("seq", [32, 38])  # 32: no padding; 38: pads to 40
def test_roundtrip_permutation_invariant(fused, seq):
    device = torch.device("cpu")
    cache = make_cache(2, seq, device)
    snap = tuple((k.clone(), v.clone()) for k, v in cache)
    p, _ = make_protector(fused, device)
    rec = p.deobfuscate(p.obfuscate(cache))
    for (k0, v0), (k1, v1) in zip(snap, rec):
        assert k1.shape == k0.shape, "padding rows must be removed exactly"
        assert row_multiset_diff(k0, k1) < 1e-3
        assert row_multiset_diff(v0, v1) < 1e-3


@pytest.mark.parametrize("fused", [True, False])
def test_inputs_not_mutated(fused):
    device = torch.device("cpu")
    cache = make_cache(2, 32, device)
    snap = tuple((k.clone(), v.clone()) for k, v in cache)
    p, _ = make_protector(fused, device)
    obf = p.obfuscate(cache)
    assert max((a - b).abs().max().item() for (a, _), (b, _) in zip(snap, cache)) == 0.0, (
        "obfuscate must not mutate its input cache"
    )
    obf_snap = tuple((k.clone(), v.clone()) for k, v in obf)
    p.deobfuscate(obf)
    assert max((a - c).abs().max().item() for (a, _), (c, _) in zip(obf_snap, obf)) == 0.0, (
        "deobfuscate must not mutate its input cache"
    )


def test_marker_table_deterministic_and_ranged():
    device = torch.device("cpu")
    p, cfg = make_protector(True, device)
    t1 = p._marker_table(cfg[0][0][0], device)
    t2 = p._marker_table(cfg[0][0][0], device)
    assert torch.equal(t1, t2), "marker table must be cached/deterministic"
    assert t1.shape == (B + 1, D)
    assert torch.all(t1[0] == 0), "row 0 is the no-marker sentinel"
    conf = cfg[0][0][0]
    theta = conf["theta"]
    theta = theta.item() if torch.is_tensor(theta) else float(theta)
    thr = theta * conf["M_ratio"] * 1.42
    for i in range(1, B + 1):
        assert torch.all(t1[i] > 3 * i * thr) and torch.all(t1[i] < (3 * i + 0.5) * thr), (
            f"marker row {i} outside U(3i*t, (3i+0.5)t)"
        )


def test_every_row_carries_a_marker():
    device = torch.device("cpu")
    p, cfg = make_protector(True, device)
    S, M, A = p._get_SMA_obf(cfg[0][0][0], device)
    assert A.shape == (B, D)
    assert torch.all(A.abs().sum(dim=-1) > 0), "all b rows must be non-zero"
