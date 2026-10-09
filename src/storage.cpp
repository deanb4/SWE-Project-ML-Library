#include "mllib/storage.h"
#include <stdexcept>
#include <cstring>

Storage::Storage(size_t size, Device device) : buffer(nullptr), nbytes(size), device(device) {
    // Check size and throw error upon issue
    if (device != Device::CPU) {
        throw std::runtime_error("Only CPU Storage is supported for now");
    }

    if (nbytes > 0) {
        buffer = new std::byte[nbytes];
    }
}

Storage::~Storage() {
   delete[] static_cast<std::byte*>(buffer);
}

std::shared_ptr<Storage> Storage::copy_to(Device target) const {
  if (device == Device::CPU && target == Device::CPU) {
    auto out = std::make_shared<Storage>(nbytes, target);
    if (nbytes > 0) {
        std::memcpy(out->buffer, buffer, nbytes);
    }
    return out;
  }

  throw std::runtime_error("Only CPU -> CPU copies are supported for now");
}


void* Storage::get_ptr() const {
    return buffer;
}

size_t Storage::get_nbytes() const {
    return nbytes;
}

Device Storage::get_device() const {
    return device;
}










// std::shared_ptr<Storage> Storage::copy_to(Device target) const {
//     std::shared_ptr<Storage> storage = nullptr;
//     if (this->device == target) {
//         // return this tensor
//         storage = std::make_shared<Storage>(*this);
//     } 

//     else if (target == Device::CPU) {
//         storage = std::make_shared<Storage>(*this);
//         storage->device = target;
//     }

//     else if (target == Device::GPU){
//         // pass for now TODO 
//         throw std::runtime_error("Only works on CPU for now.");
//     }

//     return storage;
// }