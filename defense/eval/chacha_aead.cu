// Full ChaCha20-Poly1305 AEAD GPU benchmark.
// - ChaCha20: as in chacha_gpu.cu (verified vs OpenSSL)
// - Poly1305: donna-style 64-bit limbs, parallelized by splitting the message
//   into per-thread chunks (Horner within chunk), then combining partials with
//   powers of r mod 2^130-5 (generic 130-bit mulmod), then finalize + pad.
// Usage: ./chacha_aead verify|bench
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

typedef unsigned __int128 u128;

#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define QR(a, b, c, d)                \
    a += b; d ^= a; d = ROTL(d, 16);  \
    c += d; b ^= c; b = ROTL(b, 12);  \
    a += b; d ^= a; d = ROTL(d, 8);   \
    c += d; b ^= c; b = ROTL(b, 7);

// ---------------- ChaCha20 ----------------
__device__ __forceinline__ void chacha_block(uint32_t out[16], const uint32_t st[16], uint32_t ctr) {
    uint32_t s[16], stc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) { stc[i] = st[i]; s[i] = st[i]; }
    s[12] += ctr; stc[12] += ctr;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        QR(s[0], s[4], s[8], s[12]) QR(s[1], s[5], s[9], s[13])
        QR(s[2], s[6], s[10], s[14]) QR(s[3], s[7], s[11], s[15])
        QR(s[0], s[5], s[10], s[15]) QR(s[1], s[6], s[11], s[12])
        QR(s[2], s[7], s[8], s[13]) QR(s[3], s[4], s[9], s[14])
    }
#pragma unroll
    for (int i = 0; i < 16; i++) out[i] = s[i] + stc[i];
}

#define GROUP 4
__global__ void chacha_xor_kernel(uint32_t *__restrict__ data, size_t n_words,
                                  const uint32_t *key_nonce) {
    uint32_t st[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                       key_nonce[0], key_nonce[1], key_nonce[2], key_nonce[3],
                       key_nonce[4], key_nonce[5], key_nonce[6], key_nonce[7],
                       0, key_nonce[8], key_nonce[9], key_nonce[10]};
    size_t n_blocks = n_words / 16;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x * GROUP;
    uint32_t ks[GROUP][16];
    for (size_t g = tid * GROUP; g < n_blocks; g += stride) {
#pragma unroll
        for (int k = 0; k < GROUP; k++) {
            size_t b = g + k;
            if (b < n_blocks) chacha_block(ks[k], st, (uint32_t)(b + 1));
        }
#pragma unroll
        for (int k = 0; k < GROUP; k++) {
            size_t b = g + k;
            if (b < n_blocks) {
                uint4 *p = reinterpret_cast<uint4 *>(data + b * 16);
                const uint4 *ksv = reinterpret_cast<const uint4 *>(ks[k]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint4 v = p[j], kk = ksv[j];
                    v.x ^= kk.x; v.y ^= kk.y; v.z ^= kk.z; v.w ^= kk.w;
                    p[j] = v;
                }
            }
        }
    }
}

// keystream block 0 -> poly1305 one-time key
__global__ void poly_key_kernel(uint32_t *out16, const uint32_t *key_nonce) {
    uint32_t st[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                       key_nonce[0], key_nonce[1], key_nonce[2], key_nonce[3],
                       key_nonce[4], key_nonce[5], key_nonce[6], key_nonce[7],
                       0, key_nonce[8], key_nonce[9], key_nonce[10]};
    uint32_t o[16];
    chacha_block(o, st, 0);
    if (threadIdx.x == 0)
#pragma unroll
        for (int i = 0; i < 16; i++) out16[i] = o[i];
}

// ---------------- Poly1305 (donna-64 limbs) ----------------
struct PolyR { uint64_t r0, r1, r2, s1, s2, pad0, pad1; };

__device__ __forceinline__ void poly_init(PolyR &pr, const uint32_t *k16 /*16 words = 64B keystream*/) {
    uint64_t t0 = (uint64_t)k16[0] | ((uint64_t)k16[1] << 32);
    uint64_t t1 = (uint64_t)k16[2] | ((uint64_t)k16[3] << 32);
    pr.r0 = t0 & 0xffc0fffffffULL;
    pr.r1 = ((t0 >> 44) | (t1 << 20)) & 0xfffffc0ffffULL;
    pr.r2 = (t1 >> 24) & 0x00ffffffc0fULL;
    pr.s1 = pr.r1 * (5 << 2); pr.s2 = pr.r2 * (5 << 2);  // donna: 20*r
    pr.pad0 = (uint64_t)k16[4] | ((uint64_t)k16[5] << 32);
    pr.pad1 = (uint64_t)k16[6] | ((uint64_t)k16[7] << 32);
}

__device__ __forceinline__ void poly_block(uint64_t &h0, uint64_t &h1, uint64_t &h2,
                                           const PolyR &pr, const uint32_t *m) {
    uint64_t t0 = (uint64_t)m[0] | ((uint64_t)m[1] << 32);
    uint64_t t1 = (uint64_t)m[2] | ((uint64_t)m[3] << 32);
    h0 += t0 & 0xfffffffffffULL;
    h1 += ((t0 >> 44) | (t1 << 20)) & 0xfffffffffffULL;
    h2 += ((t1 >> 24) & 0x3ffffffffffULL) + (1ULL << 40);  // full block -> +2^128
    u128 d0 = (u128)h0 * pr.r0 + (u128)h1 * pr.s2 + (u128)h2 * pr.s1;
    u128 d1 = (u128)h0 * pr.r1 + (u128)h1 * pr.r0 + (u128)h2 * pr.s2;
    u128 d2 = (u128)h0 * pr.r2 + (u128)h1 * pr.r1 + (u128)h2 * pr.r0;
    uint64_t c = (uint64_t)(d0 >> 44); h0 = (uint64_t)d0 & 0xfffffffffffULL;
    d1 += c; c = (uint64_t)(d1 >> 44); h1 = (uint64_t)d1 & 0xfffffffffffULL;
    d2 += c; c = (uint64_t)(d2 >> 42); h2 = (uint64_t)d2 & 0x3ffffffffffULL;
    h0 += c * 5; c = h0 >> 44; h0 &= 0xfffffffffffULL; h1 += c;
}

// normalize h to < p (no pad)
__device__ __forceinline__ void poly_finish_nopad(uint64_t &h0, uint64_t &h1, uint64_t &h2) {
    uint64_t c = h1 >> 44; h2 += c; h1 &= 0xfffffffffffULL;
    c = h2 >> 42; h2 &= 0x3ffffffffffULL; h0 += c * 5; c = h0 >> 44; h0 &= 0xfffffffffffULL; h1 += c;
    uint64_t g0 = h0 + 5; c = g0 >> 44; g0 &= 0xfffffffffffULL;
    uint64_t g1 = h1 + c; c = g1 >> 44; g1 &= 0xfffffffffffULL;
    uint64_t g2 = h2 + c - (1ULL << 42);
    uint64_t mask = (g2 >> 63) - 1;  // all-ones if g wrapped (i.e. h < p)
    h0 = (h0 & ~mask) | (g0 & mask);
    h1 = (h1 & ~mask) | (g1 & mask);
    h2 = (h2 & ~mask) | (g2 & mask);
}

// ---- generic 130-bit mulmod: value = (lo128, hi<8) ----
struct N130 { u128 lo; uint64_t hi; };  // value = lo + hi*2^128, hi < 8

__device__ __forceinline__ void add130(N130 &a, const N130 &b) {
    u128 old = a.lo;
    a.lo += b.lo;
    uint64_t c = (a.lo < old) ? 1 : 0;
    a.hi += b.hi + c;
    // keep hi < 8 lazily; folds happen in mulmod/finalize
    while (a.hi >= 4) {  // 2^130 == 5 (mod p)
        uint64_t q = a.hi >> 2; a.hi &= 3;
        u128 o = a.lo; a.lo += (u128)q * 5;
        a.hi += (a.lo < o) ? 1 : 0;
    }
}

// a, b < 2^131 (hi < 8). schoolbook 5-word x 5-word multiply, then fold mod 2^130-5.
__device__ __forceinline__ N130 mulmod130(const N130 &a, const N130 &b) {
    uint64_t A[5] = {(uint64_t)a.lo, (uint64_t)(a.lo >> 64), a.hi, 0, 0};
    uint64_t B[5] = {(uint64_t)b.lo, (uint64_t)(b.lo >> 64), b.hi, 0, 0};
    uint64_t W[10] = {0};
    for (int i = 0; i < 5; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 5; j++) {
            u128 t = (u128)A[i] * B[j] + W[i + j] + carry;
            W[i + j] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
        W[i + 5] += carry;
    }
    // product < 2^262: W[0..4] active, W[4] < 64
    // result = lo130 + 5*(product >> 130)
    uint64_t L0 = W[0], L1 = W[1], L2 = W[2] & 3;
    u128 H1 = ((u128)(W[3] >> 2)) | ((u128)W[4] << 62);  // bits 64..127 of high part
    uint64_t H0 = (W[2] >> 2) | (W[3] << 62);            // bits 0..63 of high part
    uint64_t H1_lo = (uint64_t)H1, H1_hi = (uint64_t)(H1 >> 64);  // H1_hi <= 1
    uint64_t H2 = W[4] >> 2;                              // <= 15
    u128 s0 = (u128)L0 + (u128)H0 * 5;
    uint64_t r0 = (uint64_t)s0, c = (uint64_t)(s0 >> 64);
    u128 s1 = (u128)L1 + (u128)H1_lo * 5 + c;
    uint64_t r1 = (uint64_t)s1; c = (uint64_t)(s1 >> 64);
    uint64_t rhi = L2 + H2 * 5 + c;  // H1_hi == H2; do not double count
    while (rhi >= 4) {
        uint64_t q = rhi >> 2; rhi &= 3;
        u128 o = ((u128)r1 << 64) | r0;
        u128 add = (u128)q * 5;
        u128 n = o + add;
        uint64_t carry2 = (n < o) ? 1 : 0;
        r0 = (uint64_t)n; r1 = (uint64_t)(n >> 64);
        rhi += carry2;
    }
    return N130{(u128)r0 | ((u128)r1 << 64), rhi};
}

// combine: total = sum_t P_t * r^(K*(T-1-t)) mod p ; then tag = (total + pad) mod 2^128
__global__ void poly_combine_kernel(const uint3 *__restrict__ partials_packed, size_t T, size_t K, size_t n_mac,
                                    const PolyR *pr, uint64_t *tag_out) {
    // partials stored as 3x uint64 packed into 2x uint4 per thread? For simplicity partials passed as ulonglong
    // We re-derive: partials_packed actually is (uint64*) with 3 slots per chunk.
    const uint64_t *partials = reinterpret_cast<const uint64_t *>(partials_packed);
    __shared__ N130 sh[256];
    size_t t = threadIdx.x;
    N130 r;  // r value < 2^124
    r.lo = (u128)pr->r0 + ((u128)pr->r1 << 44) + ((u128)pr->r2 << 88);
    r.hi = 0;
    N130 acc; acc.lo = 0; acc.hi = 0;
    for (size_t i = t; i < T; i += blockDim.x) {
        // P_i from limbs
        uint64_t h0 = partials[3 * i], h1 = partials[3 * i + 1], h2 = partials[3 * i + 2];
        N130 P; P.lo = (u128)h0 + ((u128)h1 << 44) + (((u128)h2 & 0xffffffffffULL) << 88);
        P.hi = h2 >> 40;  // < 4
        // exponent = number of mac blocks AFTER chunk i
        size_t e = n_mac - min((i + 1) * K, n_mac);
        // pow via repeated squaring
        N130 result; result.lo = 1; result.hi = 0;
        N130 base = r;
        while (e) {
            if (e & 1) result = mulmod130(result, base);
            base = mulmod130(base, base);
            e >>= 1;
        }
        N130 term = mulmod130(P, result);
        add130(acc, term);
        if (T == 1)
            printf("combine: i=%d e=%d P=%016llx%016llx hi=%llx term=%016llx%016llx thi=%llx acc=%016llx%016llx ahi=%llx\n",
                   (int)i, (int)(K * (T - 1 - i)),
                   (uint64_t)(P.lo >> 64), (uint64_t)P.lo, P.hi,
                   (uint64_t)(term.lo >> 64), (uint64_t)term.lo, term.hi,
                   (uint64_t)(acc.lo >> 64), (uint64_t)acc.lo, acc.hi);
    }
    sh[t] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) add130(sh[t], sh[t + s]);
        __syncthreads();
    }
    if (t == 0) {
        N130 h = sh[0];
        // final reduce: while >= p subtract p: h < 2^130 + small; fold 2^130 -> 5
        while (h.hi >= 4) {
            uint64_t q = h.hi >> 2; h.hi &= 3;
            u128 o = h.lo; h.lo += (u128)q * 5;
            h.hi += (h.lo < o) ? 1 : 0;
        }
        // conditional subtract p (2^130-5): if h >= p: h -= p
        // h = h.lo + h.hi*2^128, h.hi in {0..3}
        // approx: if (h.hi == 3 && high) or compare with p = (3*2^128 + (2^128-5))... p = 2^130-5 = hi=3, lo=2^128-5
        u128 p_lo = ~(u128)4;  // 2^128-5
        uint64_t p_hi = 3;
        if (h.hi > p_hi || (h.hi == p_hi && h.lo >= p_lo)) {
            u128 o = h.lo;
            h.lo -= p_lo;
            h.hi -= p_hi + ((h.lo > o) ? 1 : 0);  // borrow
        }
        // add pad mod 2^128
        if (T == 1)
            printf("final: h=%016llx%016llx hi=%llx pad=%016llx%016llx\n",
                   (uint64_t)(h.lo >> 64), (uint64_t)h.lo, h.hi, pr->pad1, pr->pad0);
        u128 pad = ((u128)pr->pad1 << 64) | pr->pad0;
        u128 tag = h.lo + pad;  // mod 2^128 (u128 wraps)
        tag_out[0] = (uint64_t)tag;
        tag_out[1] = (uint64_t)(tag >> 64);
    }
}

__global__ void poly_debug_kernel(const uint32_t *data, const PolyR *pr) {
    printf("r0=%011llx r1=%011llx r2=%010llx pad0=%016llx pad1=%016llx\n", pr->r0, pr->r1, pr->r2, pr->pad0, pr->pad1);
    uint64_t h0 = 0, h1 = 0, h2 = 0;
    uint32_t lb[4] = {0, 0, 128, 0};
    const uint32_t *blocks[3] = {data, data + 4, lb};
    for (int b = 0; b < 3; b++) {
        poly_block(h0, h1, h2, *pr, blocks[b]);
        printf("after block %d: h0=%011llx h1=%011llx h2=%010llx\n", b, h0, h1, h2);
    }
    poly_finish_nopad(h0, h1, h2);
    printf("finish: h0=%011llx h1=%011llx h2=%010llx\n", h0, h1, h2);
}



__device__ __forceinline__ N130 pow_r_N(const N130 &r, size_t e) {
    N130 result; result.lo = 1; result.hi = 0;
    N130 base = r;
    while (e) {
        if (e & 1) result = mulmod130(result, base);
        base = mulmod130(base, base);
        e >>= 1;
    }
    return result;
}

__global__ void pow_init_kernel(N130 *pw, size_t Kc, const PolyR *pr) {
    if (threadIdx.x != 0) return;
    N130 r; r.lo = (u128)pr->r0 + ((u128)pr->r1 << 44) + ((u128)pr->r2 << 88); r.hi = 0;
    pw[0] = pow_r_N(r, Kc);
    for (int j = 1; j < 32; j++) pw[j] = mulmod130(pw[j - 1], pw[j - 1]);
}

__device__ __forceinline__ N130 load_partial(const uint64_t *p, size_t idx, size_t T) {
    if (idx >= T) { N130 z; z.lo = 0; z.hi = 0; return z; }
    uint64_t h0 = p[3 * idx], h1 = p[3 * idx + 1], h2 = p[3 * idx + 2];
    N130 v; v.lo = (u128)h0 + ((u128)h1 << 44) + (((u128)h2 & 0xffffffffffULL) << 88);
    v.hi = h2 >> 40;
    return v;
}

// generic level: seg = Kc << level; entries hold segments of mac blocks
__device__ __forceinline__ void tree_step(const N130 &a, const N130 &b, bool has_b, size_t right_idx,
                                          size_t S, size_t seg, size_t n_mac, const N130 &r,
                                          const N130 &pw_seg, N130 &out) {
    if (!has_b) { out = a; return; }
    N130 mult;
    if (right_idx == S - 1) {  // right entry is the ragged last one
        size_t len = n_mac - right_idx * seg;
        if (len == seg) mult = pw_seg;
        else mult = pow_r_N(r, len);
    } else {
        mult = pw_seg;
    }
    N130 term = mulmod130(a, mult);
    add130(term, b);
    out = term;
}

__global__ void tree_level0(const uint64_t *__restrict__ partials, N130 *__restrict__ out,
                            size_t T, size_t n_mac, size_t Kc, const N130 *__restrict__ pw,
                            const PolyR *pr) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t pairs = (T + 1) / 2;
    if (t >= pairs) return;
    N130 r; r.lo = (u128)pr->r0 + ((u128)pr->r1 << 44) + ((u128)pr->r2 << 88); r.hi = 0;
    N130 a = load_partial(partials, 2 * t, T);
    N130 b = load_partial(partials, 2 * t + 1, T);
    N130 res;
    tree_step(a, b, (2 * t + 1) < T, 2 * t + 1, T, Kc, n_mac, r, pw[0], res);
    out[t] = res;
}

__global__ void tree_level(const N130 *__restrict__ in, N130 *__restrict__ out,
                           size_t S, size_t n_mac, size_t Kc, int level,
                           const N130 *__restrict__ pw, const PolyR *pr) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t pairs = (S + 1) / 2;
    if (t >= pairs) return;
    N130 r; r.lo = (u128)pr->r0 + ((u128)pr->r1 << 44) + ((u128)pr->r2 << 88); r.hi = 0;
    N130 a = in[2 * t], b = in[2 * t + 1];
    N130 res;
    tree_step(a, b, (2 * t + 1) < S, 2 * t + 1, S, ((size_t)Kc) << level, n_mac, r, pw[level], res);
    out[t] = res;
}

__global__ void finalize_kernel(const N130 *total, const PolyR *pr, uint64_t *tag_out) {
    N130 h = total[0];
    while (h.hi >= 4) {
        uint64_t q = h.hi >> 2; h.hi &= 3;
        u128 o = h.lo; h.lo += (u128)q * 5;
        h.hi += (h.lo < o) ? 1 : 0;
    }
    u128 p_lo = ~(u128)4;
    uint64_t p_hi = 3;
    if (h.hi > p_hi || (h.hi == p_hi && h.lo >= p_lo)) {
        u128 o = h.lo;
        h.lo -= p_lo;
        h.hi -= p_hi + ((h.lo > o) ? 1 : 0);
    }
    u128 pad = ((u128)pr->pad1 << 64) | pr->pad0;
    u128 tag = h.lo + pad;
    tag_out[0] = (uint64_t)tag;       // little-endian tag bytes
    tag_out[1] = (uint64_t)(tag >> 64);
}

// partial sums: thread t handles K consecutive MAC blocks; the FINAL MAC block
// is the length block (aad_len||ct_len). data blocks come from ciphertext.
__global__ void poly_partial2_kernel(const uint32_t *__restrict__ data, size_t n_ct_blocks,
                                     size_t K, const PolyR *pr, uint64_t *partials,
                                     uint64_t aad_len, uint64_t ct_len) {
    size_t n_mac = n_ct_blocks + 1;  // + length block
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t start = t * K, end = min(start + K, n_mac);
    if (start >= n_mac) return;
    uint64_t h0 = 0, h1 = 0, h2 = 0;
    for (size_t b = start; b < end; b++) {
        if (b < n_ct_blocks) {
            poly_block(h0, h1, h2, *pr, data + b * 4);
        } else {
            // length block: LE64(aad_len) || LE64(ct_len), full block (+2^128)
            uint32_t lb[4] = {(uint32_t)aad_len, (uint32_t)(aad_len >> 32),
                              (uint32_t)ct_len, (uint32_t)(ct_len >> 32)};
            poly_block(h0, h1, h2, *pr, lb);
        }
    }
    poly_finish_nopad(h0, h1, h2);
    partials[3 * t] = h0;
    partials[3 * t + 1] = h1;
    partials[3 * t + 2] = h2;
}

static uint32_t h_key_nonce[11];
static PolyR h_pr;
static uint32_t *d_key, *d_ks0;
static PolyR *d_pr;
static uint64_t *d_partials, *d_tag;
static N130 *d_pw, *d_tree[2];
static size_t d_tree_cap = 0;

static size_t n_mac_global = 0;
static void poly_combine_tree(size_t T, size_t Kc) {
    pow_init_kernel<<<1, 1>>>(d_pw, Kc, d_pr);
    size_t S = T;
    int buf = 0;
    if (!d_tree[0] || d_tree_cap < T) {
        if (d_tree[0]) cudaFree(d_tree[0]);
        if (d_tree[1]) cudaFree(d_tree[1]);
        cudaMalloc(&d_tree[0], T * sizeof(N130));
        cudaMalloc(&d_tree[1], T * sizeof(N130));
        d_tree_cap = T;
    }
    size_t n_mac = n_mac_global;
    size_t pairs = (S + 1) / 2;
    tree_level0<<<(pairs + 255) / 256, 256>>>(d_partials, d_tree[0], T, n_mac, Kc, d_pw, d_pr);
    S = pairs; buf = 0;
    for (int level = 1; S > 1; level++) {
        pairs = (S + 1) / 2;
        tree_level<<<(pairs + 255) / 256, 256>>>(d_tree[buf], d_tree[1 - buf], S, n_mac, Kc, level, d_pw, d_pr);
        S = pairs; buf = 1 - buf;
    }
    finalize_kernel<<<1, 1>>>(d_tree[buf], d_pr, d_tag);
}
static const size_t K = 256;  // 16-byte MAC blocks per chunk (4 KB)

static void setup() {
    for (int i = 0; i < 8; i++) h_key_nonce[i] = 0x03020100u + (uint32_t)i * 0x04040404u;
    for (int i = 0; i < 3; i++) h_key_nonce[8 + i] = 0x0b0a0908u + (uint32_t)i * 0x03030303u;
    cudaMalloc(&d_key, 11 * 4);
    cudaMemcpy(d_key, h_key_nonce, 11 * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ks0, 64);
    cudaMalloc(&d_pr, sizeof(PolyR));
    cudaMalloc(&d_tag, 16);
    cudaMalloc(&d_pw, 32 * sizeof(N130));
    poly_key_kernel<<<1, 32>>>(d_ks0, d_key);
    cudaMemcpy(&h_pr, d_ks0, 0, cudaMemcpyDeviceToHost);  // placeholder no-op
    // build PolyR on host from keystream block 0
    uint32_t ks0[16];
    cudaMemcpy(ks0, d_ks0, 64, cudaMemcpyDeviceToHost);
    uint64_t t0 = (uint64_t)ks0[0] | ((uint64_t)ks0[1] << 32);
    uint64_t t1 = (uint64_t)ks0[2] | ((uint64_t)ks0[3] << 32);
    h_pr.r0 = t0 & 0xffc0fffffffULL;
    h_pr.r1 = ((t0 >> 44) | (t1 << 20)) & 0xfffffc0ffffULL;
    h_pr.r2 = (t1 >> 24) & 0x00ffffffc0fULL;
    h_pr.s1 = h_pr.r1 * (5 << 2); h_pr.s2 = h_pr.r2 * (5 << 2);
    h_pr.pad0 = (uint64_t)ks0[4] | ((uint64_t)ks0[5] << 32);
    h_pr.pad1 = (uint64_t)ks0[6] | ((uint64_t)ks0[7] << 32);
    cudaMemcpy(d_pr, &h_pr, sizeof(PolyR), cudaMemcpyHostToDevice);
}

static float t_enc = 0, t_poly = 0, t_comb = 0;
static cudaEvent_t ev0, ev1;
static void aead_roundtrip(uint32_t *d, size_t bytes) {
    size_t n_blocks = bytes / 16;  // 16-byte MAC blocks
    size_t T = (n_blocks + 1 + K - 1) / K;  // +1 length block
    if (!ev0) { cudaEventCreate(&ev0); cudaEventCreate(&ev1); }
    cudaEventRecord(ev0);
    chacha_xor_kernel<<<4096, 256>>>(d, bytes / 4, d_key);           // encrypt
    cudaEventRecord(ev1); cudaEventSynchronize(ev1); t_enc += 0;
    { cudaError_t e = cudaGetLastError(); if (e) { printf("enc err: %s\n", cudaGetErrorString(e)); return; } }
    float mm; cudaEventElapsedTime(&mm, ev0, ev1); t_enc += mm;
    cudaEventRecord(ev0);
    poly_partial2_kernel<<<(T + 255) / 256, 256>>>(d, n_blocks, K, d_pr, d_partials, 0, bytes);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    { cudaError_t e = cudaGetLastError(); if (e) { printf("poly1 err: %s\n", cudaGetErrorString(e)); return; } }
    cudaEventElapsedTime(&mm, ev0, ev1); t_poly += mm;
    cudaEventRecord(ev0);
    n_mac_global = n_blocks + 1; poly_combine_tree(T, K);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    { cudaError_t e = cudaGetLastError(); if (e) { printf("comb1 err: %s\n", cudaGetErrorString(e)); return; } }
    cudaEventElapsedTime(&mm, ev0, ev1); t_comb += mm;
    chacha_xor_kernel<<<4096, 256>>>(d, bytes / 4, d_key);           // decrypt
    poly_partial2_kernel<<<(T + 255) / 256, 256>>>(d, n_blocks, K, d_pr, d_partials, 0, bytes);
    n_mac_global = n_blocks + 1; poly_combine_tree(T, K);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e) { printf("round2 err: %s\n", cudaGetErrorString(e)); return; } }
}

int main(int argc, char **argv) {
    setup();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const char *mode = argc > 1 ? argv[1] : "bench";

    if (!strcmp(mode, "verifybig")) {
        size_t bytes = 65536;
        uint32_t *d;
        cudaMalloc(&d, bytes);
        cudaMemset(d, 0, bytes);
        size_t n_blocks = bytes / 16;
        size_t T = (n_blocks + 1 + K - 1) / K;  // +1 length block -> ragged last chunk
        cudaMalloc(&d_partials, 3 * 8 * T);
        chacha_xor_kernel<<<4096, 256>>>(d, bytes / 4, d_key);
        poly_partial2_kernel<<<(T + 255) / 256, 256>>>(d, n_blocks, K, d_pr, d_partials, 0, bytes);
        n_mac_global = n_blocks + 1;
        // debug: run tree manually, dumping levels
        if (!d_tree[0]) { cudaMalloc(&d_tree[0], T * sizeof(N130)); cudaMalloc(&d_tree[1], T * sizeof(N130)); d_tree_cap = T; }
        pow_init_kernel<<<1, 1>>>(d_pw, K, d_pr);
        size_t S = T; int buf = 0;
        size_t pairs = (S + 1) / 2;
        tree_level0<<<(pairs + 255) / 256, 256>>>(d_partials, d_tree[0], T, n_blocks + 1, K, d_pw, d_pr);
        S = pairs;
        { N130 *h = new N130[S]; cudaMemcpy(h, d_tree[0], S * sizeof(N130), cudaMemcpyDeviceToHost);
          printf("L0 (%zu):", S); for (size_t i = 0; i < S; i++) printf(" %llx%016llx", h[i].hi, (uint64_t)h[i].lo); printf("\n"); }
        for (int level = 1; S > 1; level++) {
            pairs = (S + 1) / 2;
            tree_level<<<(pairs + 255) / 256, 256>>>(d_tree[buf], d_tree[1 - buf], S, n_blocks + 1, K, level, d_pw, d_pr);
            S = pairs; buf = 1 - buf;
            cudaDeviceSynchronize();
            { cudaError_t e = cudaGetLastError(); if (e) printf("level err: %s\n", cudaGetErrorString(e)); }
            N130 *h = new N130[S]; cudaMemcpy(h, d_tree[buf], S * sizeof(N130), cudaMemcpyDeviceToHost);
            printf("L%d (%zu):", level, S); for (size_t i = 0; i < S; i++) printf(" %llx%016llx", h[i].hi, (uint64_t)h[i].lo); printf("\n");
        }
        finalize_kernel<<<1, 1>>>(d_tree[buf], d_pr, d_tag);
        { cudaError_t e = cudaGetLastError(); if (e) printf("err: %s\n", cudaGetErrorString(e)); }
        uint64_t tag[2];
        cudaMemcpy(tag, d_tag, 16, cudaMemcpyDeviceToHost);
        printf("TAGBIG %016lx%016lx\n", tag[1], tag[0]);
        return 0;
    }

    if (!strcmp(mode, "polydebug")) {
        // poly over 3 known blocks (ct of 128 zero bytes + length block), print after each
        uint32_t *d; cudaMalloc(&d, 128);
        chacha_xor_kernel<<<1, 32>>>(d, 32, d_key);
        cudaDeviceSynchronize();
        poly_debug_kernel<<<1, 1>>>(d, d_pr);
        cudaError_t e = cudaGetLastError();
        if (e) printf("launch error: %s\n", cudaGetErrorString(e));
        cudaDeviceSynchronize();
        return 0;
    }

    if (!strcmp(mode, "verify")) {
        size_t bytes = 128;
        uint32_t *h = new uint32_t[bytes / 4];
        memset(h, 0, bytes);
        uint32_t *d;
        cudaMalloc(&d, bytes);
        cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
        size_t n_blocks = bytes / 16, T = (n_blocks + K - 1) / K;  // T=1... use T=2 for combine path
        T = 2;  // 2 chunks
        size_t K2 = 5;  // 9 mac blocks -> chunk0: 0-4, chunk1: 5-8
        if (argc > 2 && !strcmp(argv[2], "single")) { T = 1; K2 = 16; }
        if (argc > 2 && !strcmp(argv[2], "debug")) { T = 2; }
        chacha_xor_kernel<<<1, 32>>>(d, bytes / 4, d_key);
        cudaMalloc(&d_partials, 3 * 8 * T);
        poly_partial2_kernel<<<1, T>>>(d, n_blocks, K2, d_pr, d_partials, 0, (uint64_t)bytes);
        { cudaError_t e = cudaGetLastError(); if (e) printf("partial launch: %s\n", cudaGetErrorString(e)); }
        { uint64_t pp[6] = {0}; cudaMemcpy(pp, d_partials, 3 * 8 * T, cudaMemcpyDeviceToHost);
          printf("P0 %016llx %016llx %016llx\n", pp[0], pp[1], pp[2]);
          printf("P1 %016llx %016llx %016llx\n", pp[3], pp[4], pp[5]); }
        n_mac_global = n_blocks + 1; poly_combine_tree(T, K2);
        { cudaError_t e = cudaGetLastError(); if (e) printf("combine launch: %s\n", cudaGetErrorString(e)); }
        uint64_t tag[2];
        cudaMemcpy(tag, d_tag, 16, cudaMemcpyDeviceToHost);
        printf("TAG %016lx%016lx\n", tag[1], tag[0]);
        return 0;
    }

    for (double gb : {1.0, 2.0}) {
        size_t bytes = (size_t)(gb * (1 << 30));
        size_t n_blocks = bytes / 16;  // 16-byte mac blocks
        size_t T = (n_blocks + 1 + K - 1) / K;  // +1 length block
        t_enc = t_poly = t_comb = 0;
        cudaMalloc(&d_partials, 3 * 8 * T);
        uint32_t *d;
        cudaMalloc(&d, bytes);
        cudaMemset(d, 0, bytes);
        aead_roundtrip(d, bytes);  // warmup
        cudaDeviceSynchronize();
        { cudaError_t e = cudaGetLastError(); if (e) printf("warmup error: %s\n", cudaGetErrorString(e)); }
        float total = 0;
        int trials = 5;
        for (int t = 0; t < trials; t++) {
            cudaEventRecord(t0);
            aead_roundtrip(d, bytes);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            total += ms;
        }
        float mean = total / trials;
        double tput = (2.0 * bytes) / (mean / 1000.0) / 1e9;
        printf("[%.0f GB] full AEAD enc+dec: %.2f ms | %.1f GB/s total (%.1f per dir)\n",
               gb, mean, tput, tput / 2);
        t_enc = t_poly = t_comb = 0;
        // one instrumented roundtrip for the breakdown
        aead_roundtrip(d, bytes);
        cudaDeviceSynchronize();
        printf("  breakdown: enc(chacha)=%.2f ms  poly_partial=%.2f ms  combine=%.2f ms (first pass of roundtrip)\n", t_enc, t_poly, t_comb);
        cudaFree(d); cudaFree(d_partials);
    }
    return 0;
}
