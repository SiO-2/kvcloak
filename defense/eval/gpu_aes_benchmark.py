"""GPU AES-GCM (CTR core) vs CPU AES-GCM latency benchmark.

GPU AES-GCM = AES-CTR keystream + GHASH. This benchmark implements the
vectorized AES-128-CTR core in PyTorch on GPU (GHASH excluded, so the GPU
number is a *lower bound* on a full GPU AES-GCM implementation). Correctness
of the AES core is verified against the `cryptography` (OpenSSL) AESGCM.
"""

import argparse
import time
import numpy as np
import torch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76"
    "ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d83115"
    "04c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f84"
    "53d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa8"
    "51a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d1973"
    "60814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479"
    "e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a"
    "703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df"
    "8ca1890dbfe6426841992d0fb054bb16"
)
assert len(SBOX) == 256, f"SBOX length {len(SBOX)} != 256"
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def key_expansion_128(key: bytes):
    """AES-128 key schedule -> list of 11 round keys, each 16 bytes (flat, col-major)."""
    assert len(key) == 16
    w = [list(key[4 * i : 4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        temp = list(w[i - 1])
        if i % 4 == 0:
            temp = temp[1:] + temp[:1]  # RotWord
            temp = [SBOX[b] for b in temp]  # SubWord
            temp[0] ^= RCON[i // 4 - 1]
        w.append([a ^ b for a, b in zip(w[i - 4], temp)])
    rks = []
    for r in range(11):
        rk = []
        for c in range(4):  # column-major flat: [w0, w1, w2, w3] per column
            rk.extend(w[4 * r + c])
        rks.append(rk)
    return rks


class GpuAesCtr:
    """Vectorized AES-128-CTR on GPU via torch ops."""

    def __init__(self, key: bytes, device: str):
        self.device = device
        rks = [torch.tensor(rk, dtype=torch.int32, device=device) for rk in key_expansion_128(key)]
        self.rk = torch.stack(rks)  # [11,16]
        self.sbox_t = torch.tensor(list(SBOX), dtype=torch.int32, device=device)
        # flat state index p: r = p % 4 (row), c = p // 4 (col); ShiftRows out[m,c,r]=in[m,(c+r)%4,r]
        cc, rr = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        self.col_perm = ((cc + rr) % 4).to(device).unsqueeze(0)  # [1,4,4] for dim-1 gather

    def _shift_rows(self, s):  # s: [M,4,4] dims (col,row)
        return torch.take_along_dim(s, self.col_perm.expand(s.shape[0], 4, 4), dim=1)

    @staticmethod
    def _gf2(x):  # x * 2 in GF(2^8)
        return ((x << 1) ^ ((x >> 7) * 0x1B)) & 0xFF

    def _mix_columns(self, s):  # s: [M,4,4] dims (col,row)
        a0, a1, a2, a3 = s[:, :, 0], s[:, :, 1], s[:, :, 2], s[:, :, 3]
        f2, f3 = self._gf2, lambda x: self._gf2(x) ^ x
        b = torch.stack(
            [f2(a0) ^ f3(a1) ^ a2 ^ a3,
             a0 ^ f2(a1) ^ f3(a2) ^ a3,
             a0 ^ a1 ^ f2(a2) ^ f3(a3),
             f3(a0) ^ a1 ^ a2 ^ f2(a3)],
            dim=-1,
        )
        return b

    def _aes_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """Encrypt [M,16] int32 blocks with AES-128 (ECB of counter blocks)."""
        s = blocks ^ self.rk[0]
        for r in range(1, 10):
            s = self.sbox_t[s.long()]          # SubBytes
            s = self._shift_rows(s.reshape(-1, 4, 4)).reshape(-1, 16)  # ShiftRows
            s = self._mix_columns(s.reshape(-1, 4, 4)).reshape(-1, 16)  # MixColumns
            s = s ^ self.rk[r]
        s = self.sbox_t[s.long()]
        s = self._shift_rows(s.reshape(-1, 4, 4)).reshape(-1, 16)
        return s ^ self.rk[10]

    def keystream_xor(self, data_u8: torch.Tensor, nonce: bytes, ctr_start: int = 2) -> torch.Tensor:
        """XOR data (uint8 flat tensor on device) with AES-CTR keystream."""
        n_blocks = (data_u8.numel() + 15) // 16
        out = data_u8.to(self.device).clone()
        chunk = 1 << 23  # 8M blocks = 128 MB
        prefix = torch.tensor(list(nonce) + [0, 0, 0, 0], dtype=torch.int32, device=self.device)
        for off in range(0, n_blocks, chunk):
            m = min(chunk, n_blocks - off)
            ctr = torch.arange(ctr_start + off, ctr_start + off + m, dtype=torch.int64, device=self.device)
            blocks = prefix.repeat(m, 1)
            blocks[:, 12] = (ctr >> 24) & 0xFF
            blocks[:, 13] = (ctr >> 16) & 0xFF
            blocks[:, 14] = (ctr >> 8) & 0xFF
            blocks[:, 15] = ctr & 0xFF
            ks = self._aes_blocks(blocks).to(torch.uint8).reshape(-1)
            end = (off + m) * 16
            seg = out[off * 16 : end]
            ks = ks[: seg.numel()]
            out[off * 16 : end] = seg ^ ks
        return out


def bench_gpu(gpu_ctr, data_bytes: bytes, nonce: bytes, trials: int):
    data = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(gpu_ctr.device)
    def op():
        ct = gpu_ctr.keystream_xor(data, nonce)
        return gpu_ctr.keystream_xor(ct, nonce)
    op()  # warmup
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.time()
        op()
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    return np.mean(ts), np.std(ts)


def bench_cpu(key: bytes, data_bytes: bytes, nonce: bytes, trials: int):
    aesgcm = AESGCM(key)
    chunk = 1 << 29  # cryptography AESGCM limit is 2**31-1; chunk for safety
    def op():
        for off in range(0, len(data_bytes), chunk):
            seg = data_bytes[off : off + chunk]
            aesgcm.decrypt(nonce, aesgcm.encrypt(nonce, seg, None), None)
    op()
    ts = []
    for _ in range(trials):
        t0 = time.time()
        op()
        ts.append((time.time() - t0) * 1000)
    return np.mean(ts), np.std(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--sizes-gb", default="1,2")
    args = p.parse_args()

    key = bytes(range(16))
    nonce = bytes(range(12))

    # --- correctness: GPU CTR core vs OpenSSL AES-GCM ---
    pt = np.random.default_rng(0).bytes(4096)
    ref_ct = AESGCM(key).encrypt(nonce, pt, None)
    gpu_ctr = GpuAesCtr(key, args.device)
    my_ct = gpu_ctr.keystream_xor(
        torch.frombuffer(bytearray(pt), dtype=torch.uint8), nonce
    ).cpu().numpy().tobytes()
    assert my_ct == ref_ct[: len(pt)], "AES core mismatch vs OpenSSL!"
    print("Correctness check passed: GPU AES-128-CTR keystream == OpenSSL AES-GCM ciphertext")

    pat = np.random.default_rng(1).bytes(1 << 26)  # 64 MB pattern
    for gb in [float(x) for x in args.sizes_gb.split(",")]:
        n = int(gb * (1 << 30))
        data_bytes = (pat * (n // len(pat) + 1))[:n]
        g_mean, g_std = bench_gpu(gpu_ctr, data_bytes, nonce, args.trials)
        c_mean, c_std = bench_cpu(key, data_bytes, nonce, args.trials)
        print(
            f"[{gb:g} GB] CPU AES-GCM (OpenSSL, enc+dec): {c_mean:8.1f} ± {c_std:6.1f} ms"
            f" | GPU AES-CTR enc+dec (torch): {g_mean:8.1f} ± {g_std:6.1f} ms"
            f" | speedup: {c_mean / g_mean:5.1f}x"
        )


if __name__ == "__main__":
    main()
