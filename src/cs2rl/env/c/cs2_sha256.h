/* src/cs2rl/env/c/cs2_sha256.h — minimal SHA-256, for the StaticData layout hash only.
 *
 * WHAT: FIPS 180-4 SHA-256 over a byte stream, streaming API (init/update/final).
 *       `cs2_sha256_final_hex` writes 64 lowercase hex chars + NUL.
 *
 * WHY a hand-rolled hash rather than a library: the binding links libc and libm
 * and nothing else (build.zig), on purpose — the extension has to build from
 * source on every developer box with only `ziglang` installed. Adding OpenSSL
 * would put a system dependency in front of `pip install`, for one hash of one
 * ~3 KB string computed once per call. The only consumer is the W2 layout hash
 * (binding.static_data_layout), which tests/test_static_data_layout.py compares
 * against the digest Python's hashlib produces for the same serialisation. That
 * cross-implementation comparison IS this file's test: a padding, endianness or
 * length-field bug here cannot agree with hashlib by accident, so it surfaces as
 * a failing hash assertion. Note what that does NOT cover — only inputs of the
 * layout blob's one length are exercised, so if you ever hash something else
 * here, re-verify against a published vector first (this implementation was
 * checked against the FIPS "abc" / empty / 448-bit / multi-block vectors when it
 * was written; that check lives in the T4a task report, not in the suite).
 *
 * PITFALL: this is a plain data-integrity hash, NOT a security primitive. There
 * is no constant-time compare and no HMAC here. Do not grow it into one — if
 * something in this repo ever needs cryptographic guarantees, it needs a real
 * library, not this file.
 *
 * PITFALL: `cs2_sha256_final_hex` finalises the context destructively (it
 * appends the padding block). A context cannot be updated again afterwards.
 */
#ifndef CS2_SHA256_H
#define CS2_SHA256_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;  /* total message length in BITS, not bytes */
    uint8_t  buf[64]; /* partial block carried between update() calls */
    size_t   buflen;
} Cs2Sha256;

/* Round constants: first 32 bits of the fractional parts of the cube roots of
 * the first 64 primes (FIPS 180-4 §4.2.2). */
static const uint32_t CS2_SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

/* n is always 1..31 at every call site, so the `x << (32 - n)` half never hits
 * the undefined `<< 32` case. Kept as a function (not a macro) so the argument
 * cannot be evaluated twice. */
static uint32_t cs2_sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* One 64-byte block through the compression function. `p` is read big-endian
 * byte by byte rather than cast to uint32_t*: the input is a char buffer with
 * no alignment guarantee, and the digest must not depend on host endianness. */
static void cs2_sha256_block(Cs2Sha256* c, const uint8_t* p) {
    uint32_t w[64];
    int      i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i * 4] << 24) | ((uint32_t)p[i * 4 + 1] << 16) |
               ((uint32_t)p[i * 4 + 2] << 8) | (uint32_t)p[i * 4 + 3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 =
            cs2_sha256_rotr(w[i - 15], 7) ^ cs2_sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 =
            cs2_sha256_rotr(w[i - 2], 17) ^ cs2_sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c->state[0], b = c->state[1], cc = c->state[2], d = c->state[3];
    uint32_t e = c->state[4], f = c->state[5], g = c->state[6], h = c->state[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1  = cs2_sha256_rotr(e, 6) ^ cs2_sha256_rotr(e, 11) ^ cs2_sha256_rotr(e, 25);
        uint32_t ch  = (e & f) ^ ((~e) & g);
        uint32_t t1  = h + S1 + ch + CS2_SHA256_K[i] + w[i];
        uint32_t S0  = cs2_sha256_rotr(a, 2) ^ cs2_sha256_rotr(a, 13) ^ cs2_sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2  = S0 + maj;
        h            = g;
        g            = f;
        f            = e;
        e            = d + t1;
        d            = cc;
        cc           = b;
        b            = a;
        a            = t1 + t2;
    }
    c->state[0] += a;
    c->state[1] += b;
    c->state[2] += cc;
    c->state[3] += d;
    c->state[4] += e;
    c->state[5] += f;
    c->state[6] += g;
    c->state[7] += h;
}

/* Initial hash value: first 32 bits of the fractional parts of the square roots
 * of the first 8 primes (FIPS 180-4 §5.3.3). */
static void cs2_sha256_init(Cs2Sha256* c) {
    c->state[0] = 0x6a09e667u;
    c->state[1] = 0xbb67ae85u;
    c->state[2] = 0x3c6ef372u;
    c->state[3] = 0xa54ff53au;
    c->state[4] = 0x510e527fu;
    c->state[5] = 0x9b05688cu;
    c->state[6] = 0x1f83d9abu;
    c->state[7] = 0x5be0cd19u;
    c->bitlen   = 0;
    c->buflen   = 0;
}

static void cs2_sha256_update(Cs2Sha256* c, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    size_t         i;
    for (i = 0; i < len; i++) {
        c->buf[c->buflen++] = p[i];
        if (c->buflen == 64) {
            cs2_sha256_block(c, c->buf);
            c->bitlen += 512;
            c->buflen  = 0;
        }
    }
}

/* Append the FIPS padding (0x80, zeros, 64-bit big-endian bit length) and write
 * the digest as 64 lowercase hex chars plus a NUL — `out` needs 65 bytes.
 * Destructive: see the PITFALL at the top of the file. */
static void cs2_sha256_final_hex(Cs2Sha256* c, char* out) {
    size_t  i = c->buflen;
    uint8_t hex[16];
    /* Length is counted before padding, so add the bits still sitting in buf. */
    c->bitlen   += (uint64_t)c->buflen * 8u;
    c->buf[i++]  = 0x80u;
    /* The length field needs the last 8 bytes; if it will not fit, flush a full
     * zero-padded block first and pad a fresh one. */
    if (i > 56) {
        while (i < 64)
            c->buf[i++] = 0x00u;
        cs2_sha256_block(c, c->buf);
        i = 0;
    }
    while (i < 56)
        c->buf[i++] = 0x00u;
    for (i = 0; i < 8; i++)
        c->buf[56 + i] = (uint8_t)((c->bitlen >> (56 - 8 * i)) & 0xffu);
    cs2_sha256_block(c, c->buf);

    memcpy(hex, "0123456789abcdef", 16);
    for (i = 0; i < 8; i++) {
        uint32_t v = c->state[i];
        int      j;
        for (j = 0; j < 4; j++) {
            uint8_t byte           = (uint8_t)((v >> (24 - 8 * j)) & 0xffu);
            out[i * 8 + j * 2]     = (char)hex[byte >> 4];
            out[i * 8 + j * 2 + 1] = (char)hex[byte & 0x0fu];
        }
    }
    out[64] = '\0';
}

#endif /* CS2_SHA256_H */
