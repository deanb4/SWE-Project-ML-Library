#include "mllib/data.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
    //Reads a 4-byte big-endian unsigned integer from an IDX file
    uint32_t read_big_endian_uint32(std::ifstream& file, const std::string& field_name) {
        unsigned char bytes[4];

        file.read(reinterpret_cast<char*>(bytes), 4);
        if(file.gcount() != 4) {
            throw std::runtime_error("Data: unexpected end of file while reading " + field_name);
        }

        return (static_cast<uint32_t>(bytes[0]) << 24) |
               (static_cast<uint32_t>(bytes[1]) << 16) |
               (static_cast<uint32_t>(bytes[2]) << 8) |
               static_cast<uint32_t>(bytes[3]);
    }

    //Multiplies two size_t values and checks for integer overflow
    size_t checked_multiply(size_t a, size_t b, const char* message) {
         if((b != 0) && (a > std::numeric_limits<size_t>::max() / b)) {
            throw std::overflow_error(message);
        }
        return a * b;
    }

} //anonymous namespace

Tensor load_images(const std::string& path) {
    //Open the file
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("load_images: could not open file" + path);
    }

    //Read and validate the IDX magic number
    unsigned char magic[4];

    file.read(reinterpret_cast<char*>(magic), 4);
    if(file.gcount() != 4) {
        throw std::runtime_error("load_images: file is too short to contain an IDX header");
    }

    //first two bytes must be zero, the third byte (0x08) indicates unsigned byte data
    if(magic[0] != 0 || magic[1] != 0 || magic[2] != 0x08) {
        throw std::runtime_error("load_images: unsupported IDX file type, expected unsigned byte data");
    }

    //fourth byte specifies number of dimensions
    const uint8_t num_dimensions = magic[3];
    if(num_dimensions < 3) { // dataset has 3 or 4 dimensions, depending on if it is RGB or not
        throw std::runtime_error("load_images: expected and IDX image file with at least 3 dimensions");
    }

    //Read the dimensions
    //each dimension is stored as a 4-byte big-endian unsigned integer
    std::vector<uint32_t> dimensions(num_dimensions);
    for(size_t i = 0; i < num_dimensions; ++i) {
        dimensions[i] = read_big_endian_uint32(file, "dimension " + std::to_string(i));

        if(dimensions[i] = 0) {
            throw std::runtime_error("load_images: IDX file contains a zero-sized dimension");
        }
    }

    //first dimension is number of images
    const size_t num_images = dimensions[0];

    //multiply remaining dimensions to determine the number of values per image
    //i.e. for grayscale (image_height * image_width), or for RGB (image_height * image_width * 3)
    size_t image_size = 1;
    for(size_t i = 1; i < dimensions.size(); ++i) {
        image_size = checked_multiply(image_size, static_cast<size_t>(dimensions[i]), 
            "load_images: image dimensions overflow size_t");
    }

    //calculate total number of pixel values in dataset
    const size_t total_values = checked_multiply(num_images, image_size, 
        "load_images: total number of pixel values overflows size_t");
    
    //tensor shapes use int64_t dimensions
    if(image_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("load_images: flattened image size exceeds int64_t");
    }

    //Read image data
    //each pixel is stored as an unsigned byte
    if(total_values > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("load_images: image data is too large to read");
    }

    //temporarily store raw pixel values in this vector before converting to normalized floating-point values
    std::vector<uint8_t> raw_pixels(total_values);
    file.read(reinterpret_cast<char*>(raw_pixels.data()), static_cast<std::streamsize>(raw_pixels.size()));

    if(file.gcount() != static_cast<std::streamsize>(raw_pixels.size())) {
        throw std::runtime_error("load_images: IDX file does not contain enough pixel data");
    }

    //reject unexpected extra data
    //an IDX file should contain exactly the header followed by the declared number of pixel values
    if(file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("load_images: IDX file contains unexpected trailing data");
    }

    if(file.bad()) {
        throw std::runtime_error("load_images: an error occured while reading pixel data");
    }

    //Allocate storage for output tensor
    //the resulting tensor contains FLOAT32 values, so each pixel needs sizeof(float) bytes
    const size_t output_bytes = checked_multiply(total_values, sizeof(float), 
        "load_images: output tensor allocation size overflows size_t");

    auto storage = std::make_shared<Storage>(output_bytes, Device::CPU);

    float* output = static_cast<float*>(storage->get_ptr());

    //Normalize pixel values
    for(size_t i; i < total_values; ++i) {
        output[i] = static_cast<float>(raw_pixels[i]) / 255.0f;
    }

    //Construct and return tensor
    return Tensor(
        std::move(storage),
        {
            static_cast<int64_t>(num_images),
            static_cast<int64_t>(image_size)
        },
        Device::CPU,
        false,
        Dtype::FLOAT32
    );

}