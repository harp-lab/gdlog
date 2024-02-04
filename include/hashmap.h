
#pragma once

#include <cstdint>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

using hash_entry_t = struct {
    uint64_t key;
    uint32_t value;
};

struct OpenAddressHashMap {
    rmm::device_vector<hash_entry_t> table;
    uint32_t capacity;
    uint32_t size;

    float load_factor;

    OpenAddressHashMap(uint32_t capacity, float load_factor);
};
