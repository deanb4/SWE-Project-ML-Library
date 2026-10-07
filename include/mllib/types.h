#ifndef TYPES_H
#define TYPES_H

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

#endif