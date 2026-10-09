#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <cstddef>
#include <stdexcept>

enum class Device {
    CPU,
    GPU,
};

enum class Dtype {
    FLOAT64,
    FLOAT32,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL,
};

// Note:
// The inline is required. types.h gets included into several .cpp files, 
// and without inline each of them compiles its own copy of the function, 
// so the linker fails with a "multiple definition" error

inline size_t dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::FLOAT64: return sizeof(double);
        case Dtype::FLOAT32: return sizeof(float);
        case Dtype::INT64:   return sizeof(int64_t);
        case Dtype::INT32:   return sizeof(int32_t);
        case Dtype::INT16:   return sizeof(int16_t);
        case Dtype::INT8:    return sizeof(int8_t);
        case Dtype::UINT8:   return sizeof(uint8_t);
        case Dtype::BOOL:    return sizeof(bool);
    }
    throw std::runtime_error("Unknown dtype");
}

#endif