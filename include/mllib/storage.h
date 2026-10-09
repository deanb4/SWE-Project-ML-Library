#ifndef STORAGE_H
#define STORAGE_H

#include <memory>
#include <cstddef>
#include "types.h"

class Storage {
    private:
        void* buffer;
        size_t nbytes;
        Device device;

    public:
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        Storage(size_t size, Device device = Device::CPU);
        virtual ~Storage();
        std::shared_ptr<Storage> copy_to(Device target) const;
        void* get_ptr() const;
        size_t get_nbytes() const;
        Device get_device() const;

};

#endif