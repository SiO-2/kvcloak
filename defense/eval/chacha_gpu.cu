// Standalone optimized ChaCha20 GPU benchmark.
// Each thread computes 4 independent ChaCha20 blocks (ILP), coalesced uint4 access.
// Usage: ./chacha_gpu [mode]  mode=verify -> print hex of first 128 output bytes
//                     mode=bench -> benchmark 1/2 GB encrypt+decrypt
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define QR(a, b, c, d)                \
    a += b; d ^= a; d = ROTL(d, 16);  \
    c += d; b ^= c; b = ROTL(b, 12);  \
    a += b; d ^= a; d = ROTL(d, 8);   \
    c += d; b ^= c; b = ROTL(b, 7);

__device__ __forceinline__ void chacha_block(uint32_t out[16], const uint32_t st[16], uint32_t ctr) {
    uint32_t s[16];
    uint32_t stc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) { stc[i] = st[i]; s[i] = st[i]; }
    s[12] += ctr; stc[12] += ctr;  // counter enters both the round input and the final feed-forward
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

// grid-stride over groups of GROUP blocks; thread t handles blocks with stride blockDim
#define GROUP 4
__global__ void chacha_xor_kernel(uint32_t *__restrict__ data, size_t n_words,
                                  const uint32_t st_const, const uint32_t *key_nonce) {
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
        // each block's 16 words = 4 uint4; coalesced by construction across threads
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

static uint32_t h_key_nonce[11];

int main(int argc, char **argv) {
    // key = bytes 0..31, nonce = bytes 32..43 (little-endian words)
    for (int i = 0; i < 8; i++) h_key_nonce[i] = 0x03020100u + (uint32_t)i * 0x04040404u;
    for (int i = 0; i < 3; i++) h_key_nonce[8 + i] = 0x0b0a0908u + (uint32_t)i * 0x03030303u;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    uint32_t *d_key;
    cudaMalloc(&d_key, 11 * sizeof(uint32_t));
    cudaMemcpy(d_key, h_key_nonce, 11 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    const char *mode = argc > 1 ? argv[1] : "bench";

    if (!strcmp(mode, "verify")) {
        size_t n = 128;  // bytes
        size_t words = n / 4;
        uint32_t *h = new uint32_t[words];
        memset(h, 0, n);
        uint32_t *d;
        cudaMalloc(&d, n);
        cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
        chacha_xor_kernel<<<1, 32>>>(d, words, 0, d_key);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
        printf("HEX ");
        for (size_t i = 0; i < words; i++) printf("%08x", h[i]);
        printf("\n");
        return 0;
    }

    for (double gb : {1.0, 2.0}) {
        size_t bytes = (size_t)(gb * (1 << 30));
        size_t words = bytes / 4;
        uint32_t *d;
        cudaMalloc(&d, bytes);
        cudaMemset(d, 0, bytes);
        int blocks = 4096, threads = 256;
        // warmup
        chacha_xor_kernel<<<blocks, threads>>>(d, words, 0, d_key);
        cudaDeviceSynchronize();
        float total = 0;
        int trials = 5;
        for (int t = 0; t < trials; t++) {
            cudaEventRecord(t0);
            chacha_xor_kernel<<<blocks, threads>>>(d, words, 0, d_key);  // enc
            chacha_xor_kernel<<<blocks, threads>>>(d, words, 0, d_key);  // dec
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            total += ms;
        }
        float mean_ms = total / trials;
        double tput = (2.0 * bytes) / (mean_ms / 1000.0) / 1e9;  // enc+dec bytes/s
        printf("[%.0f GB] enc+dec: %.2f ms | kernel throughput %.1f GB/s (per direction %.1f GB/s)\n",
               gb, mean_ms, tput, tput / 2);
        cudaFree(d);
    }
    return 0;
}
