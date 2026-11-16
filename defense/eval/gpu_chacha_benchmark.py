"""GPU ChaCha20 (stream core) benchmark vs AES, for GPU-suitability comparison.

ChaCha20 is an ARX cipher: no S-box lookups, only 32-bit add / rotate / xor,
which maps well to GPU SIMT. Correctness verified against `cryptography`
ChaCha20Poly1305 keystream (ciphertext XOR plaintext).
"""

import argparse
import time
import numpy as np
import torch

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CONSTS = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
MASK = 0xFFFFFFFF


class GpuChaCha20:
    def __init__(self, key: bytes, device: str):
        assert len(key) == 32
        self.device = device
        self.key = torch.tensor(
            [int.from_bytes(key[4 * i : 4 * i + 4], "little") for i in range(8)],
            dtype=torch.int64, device=device,
        )
        self.consts = torch.tensor(CONSTS, dtype=torch.int64, device=device)

    @staticmethod
    def _rotl(x, n):  # x: int64 tensor holding uint32 values
        return ((x << n) | (x >> (32 - n))) & MASK

    def _chacha_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """blocks: [M,16] int64 (const|key|counter|nonce), returns keystream words."""
        s = blocks.clone()
        orig = s
        s = s.reshape(-1, 4, 4)  # [M, row, col]; word w = 4*row+col
        for _ in range(10):
            # column round: quads are columns
            a, b, c_, d = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            a, b, c_, d = self._qr4(a, b, c_, d)
            s = torch.stack([a, b, c_, d], dim=1)
            # diagonal round: quads = (s[0,j], s[1,j+1], s[2,j+2], s[3,j+3])
            a = s[:, 0]
            b = torch.roll(s[:, 1], -1, dims=1)
            c_ = torch.roll(s[:, 2], -2, dims=1)
            d = torch.roll(s[:, 3], -3, dims=1)
            a, b, c_, d = self._qr4(a, b, c_, d)
            s = torch.stack(
                [a, torch.roll(b, 1, dims=1), torch.roll(c_, 2, dims=1), torch.roll(d, 3, dims=1)],
                dim=1,
            )
        return ((s.reshape(-1, 16) + orig) & MASK).reshape(-1, 1)

    def _qr4(self, a, b, c_, d):
        """Vectorized quarter-round over [M,4] quads."""
        a = (a + b) & MASK; d = self._rotl(d ^ a, 16)
        c_ = (c_ + d) & MASK; b = self._rotl(b ^ c_, 12)
        a = (a + b) & MASK; d = self._rotl(d ^ a, 8)
        c_ = (c_ + d) & MASK; b = self._rotl(b ^ c_, 7)
        return a, b, c_, d

    def keystream_xor(self, data_u8: torch.Tensor, nonce: bytes, ctr_start: int = 1) -> torch.Tensor:
        n_blocks = (data_u8.numel() + 63) // 64
        out = data_u8.to(self.device).clone()
        nonce_words = [int.from_bytes(nonce[4 * i : 4 * i + 4], "little") for i in range(3)]
        chunk = 1 << 21  # 2M blocks = 128 MB
        for off in range(0, n_blocks, chunk):
            m = min(chunk, n_blocks - off)
            ctr = torch.arange(ctr_start + off, ctr_start + off + m, dtype=torch.int64, device=self.device)
            const = self.consts.repeat(m, 1)
            key = self.key.repeat(m, 1)
            cn = torch.stack(
                [ctr] + [torch.full((m,), w, dtype=torch.int64, device=self.device) for w in nonce_words],
                dim=1,
            )
            blocks = torch.cat([const, key, cn], dim=1)
            w = self._chacha_blocks(blocks).reshape(-1, 1)
            ks = torch.cat([ (w >> (8 * i)) & 0xFF for i in range(4) ], dim=1).to(torch.uint8).reshape(-1)
            lo, hi = off * 64, min((off + m) * 64, out.numel())
            out[lo:hi] = out[lo:hi] ^ ks[: hi - lo]
        return out


def bench_gpu(c, data_bytes: bytes, nonce: bytes, trials: int):
    data = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(c.device)
    def op():
        e = c.keystream_xor(data, nonce)
        return c.keystream_xor(e, nonce)
    op()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.time()
        op()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)
    return np.mean(ts), np.std(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--sizes-gb", default="1,2")
    args = p.parse_args()

    key = bytes(range(32))
    nonce = bytes(range(12))  # chacha uses 12-byte nonce with 32-bit ctr
    pt = np.random.default_rng(0).bytes(4096)
    ref_ct = ChaCha20Poly1305(key).encrypt(nonce, pt, None)
    c = GpuChaCha20(key, args.device)
    my_ct = c.keystream_xor(torch.frombuffer(bytearray(pt), dtype=torch.uint8), nonce).cpu().numpy().tobytes()
    assert my_ct == ref_ct[: len(pt)], "ChaCha20 core mismatch!"
    print("Correctness check passed: GPU ChaCha20 keystream == OpenSSL ChaCha20-Poly1305 ciphertext")

    pat = np.random.default_rng(1).bytes(1 << 26)
    for gb in [float(x) for x in args.sizes_gb.split(",")]:
        n = int(gb * (1 << 30))
        data_bytes = (pat * (n // len(pat) + 1))[:n]
        g_mean, g_std = bench_gpu(c, data_bytes, nonce, args.trials)
        tp = n / (1 << 30) / (g_mean / 1000) / 2  # per-direction GB/s (enc+dec in one time)
        print(f"[{gb:g} GB] GPU ChaCha20 enc+dec (torch): {g_mean:8.1f} ± {g_std:6.1f} ms  (~{tp:.1f} GB/s per direction)")


if __name__ == "__main__":
    main()
