
#pragma once

inline
__device__ __host__ uint32_t murmur_hash3(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;

    return key;
}

inline
__device__ __host__ uint32_t hash_combine(uint32_t v1, uint32_t v2) {
    return v1 ^ (v2 + 0x9e3779b9 + (v1 << 6) + (v1 >> 2));
}
