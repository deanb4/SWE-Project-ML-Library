#include "mllib/data.h"
#include <cstring>
#include <fstream>
#include <iterator>
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

        if(dimensions[i] == 0) {
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
    for(size_t i = 0; i < total_values; ++i) {
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

Tensor load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error openeing file");
    }

    unsigned char magic[4];
    file.read(reinterpret_cast<char*>(magic),4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Error with magic header");
    }
    if (magic[0] != 0 || magic[1] != 0 || magic[2] != 0x08 || magic[3] != 1) {
        throw std::runtime_error("Load labels: expected an IDX label file");
    }

    // only dim is the num of labels
    const size_t num_labels = read_big_endian_uint32(file, "number of labels");

    // read label data, one unsigned byte per label
    std::vector<uint8_t> raw_labels(num_labels);
    file.read(reinterpret_cast<char*>(raw_labels.data()), static_cast<std::streamsize>(raw_labels.size()));

    if (file.gcount() != static_cast<std::streamsize>(raw_labels.size())) {
        throw std::runtime_error("load labels: IDX file does not contain enough label data");
    }

    // reject unexpected extra data
    if (file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("load labels IDC file contains unexpected trailing data");
    }

    // Copy into a int64 tensor
    auto storage = std::make_shared<Storage>(num_labels * sizeof(int64_t), Device::CPU);
    int64_t* output = static_cast<int64_t*>(storage->get_ptr());
    for (size_t i = 0; i < num_labels; i++) {
        output[i] = raw_labels[i];
    }

    return Tensor(std::move(storage), {static_cast<int64_t>(num_labels)}, Device::CPU, false, Dtype::INT64);

}

Tensor select_rows(const Tensor& t, const std::vector<int64_t>& rows) {
    const std::vector<int64_t>& shape = t.get_shape();
    if (shape.empty() || shape[0] == 0) {
        throw std::out_of_range("select rows: tensor has no rows");
    }

    // bytes in one row: 
    const size_t row_bytes = (t.num_elements() / shape[0]) * dtype_size(t.get_dtype());

    std::vector<int64_t> out_shape = shape;
    out_shape[0] = static_cast<int64_t>(rows.size());
    auto storage = std::make_shared<Storage>(rows.size() * row_bytes, Device::CPU);

    // copy each row as raw bytes
    const char* src = static_cast<const char*>(t.get_data()->get_ptr());
    char* dst = static_cast<char*>(storage->get_ptr());
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i] < 0 || rows[i] >= shape[0]) {
            throw std::out_of_range("selected rows: row " + std::to_string(rows[i]) + " is out of range");
        }
        std::memcpy(dst + i * row_bytes, src + rows[i] * row_bytes, row_bytes);
    }
    
    return Tensor(std::move(storage), out_shape, Device::CPU, false, t.get_dtype());
}

namespace {
    Tensor make_uint8_t_tensor(const std::vector<uint8_t>& values, std::vector<int64_t> shape) {
        const size_t n = values.size();
        auto storage = std::make_shared<Storage>(n * sizeof(uint8_t), Device::CPU);
        if (n > 0) {
            std::memcpy(storage->get_ptr(), values.data(), n * sizeof(uint8_t));
        }
        return Tensor(std::move(storage), shape, Device::CPU, false, Dtype::UINT8);
    }
}

TextDataset::TextDataset(const std::string& path) : char_to_id(256, -1), rng(std::random_device{}()) {

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("TextDataset: could not open file " + path);
    }

    //Assign string with contents of the text file from beginning to end
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    if (file.bad()) {
        throw std::runtime_error("TextDataset: error while reading " + path);
    }
    if (text.empty()) {
        throw std::runtime_error("TextDataset: file is empty " + path);
    }

    bool seen[256] = {};
    for (unsigned char c : text) {
        seen[c] = true;
    }
    vocab.clear();
    for (int i = 0; i < 256; ++i) {
        if (seen[i]) {
            //character associated with token id from [0, vocab_size)
            char_to_id[i] = static_cast<uint8_t>(vocab.size());
            vocab.push_back(static_cast<char>(i));
        }
    }

    encoded.resize(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        encoded[i] = char_to_id[static_cast<unsigned char>(text[i])];
    }
}

size_t TextDataset::vocab_size() const {
    return vocab.size();
}

const std::string& TextDataset::vocabulary() const {
    return vocab;
}

Tensor TextDataset::encode(const std::string& s) const {
    std::vector<uint8_t> ids(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        //check if character is in vocabulary
        const int id = char_to_id[static_cast<unsigned char>(s[i])];
        if (id < 0) {
            throw std::out_of_range("TextDataset::encode: character is not in the vocabulary");
        }
        ids[i] = id;
    }
    return make_uint8_t_tensor(ids, {static_cast<int64_t>(ids.size())});
}

std::string TextDataset::decode(const Tensor& ids) const {
    if (ids.get_dtype() != Dtype::UINT8) {
        throw std::invalid_argument("TextDataset::decode: ids must have UINT8 dtype");
    }
    if (ids.get_device() != Device::CPU) {
        throw std::invalid_argument("TextDataset::decode: only CPU tensors are supported");
    }

    const size_t n = ids.num_elements();
    const uint8_t* values = static_cast<const uint8_t*>(ids.get_data()->get_ptr());

    std::string out;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (values[i] < 0 || values[i] >= vocab.size()) {
            throw std::out_of_range("TextDataset::decode: token ID is outside the vocabulary");
        }
        out[i] = vocab[values[i]];
    }
    return out;
}

std::pair<Tensor, Tensor> TextDataset::get_batch(size_t batch_size, size_t block_size) const {
    if (batch_size == 0) {
        throw std::invalid_argument("TextDataset::get_batch: batch_size must be greater than zero");
    }
    if (block_size == 0) {
        throw std::invalid_argument("TextDataset::get_batch: block_size must be greater than zero");
    }
    if (encoded.size() < block_size + 1) {
        throw std::invalid_argument(
            "TextDataset::get_batch: text is shorter than block_size + 1");
    }

    //The latest starting point in the text to safely extract a block from
    //The block should not exceed the end of the text
    const size_t max_start = encoded.size() - block_size - 1;
    std::uniform_int_distribution<size_t> start_dist(0, max_start);

    const size_t n = batch_size * block_size;
    std::vector<uint8_t> x_values(n);
    std::vector<uint8_t> y_values(n);

    for (size_t b = 0; b < batch_size; ++b) {
        //random starting point in text
        const size_t start = start_dist(rng);
        for (size_t t = 0; t < block_size; ++t) {
            x_values[b * block_size + t] = encoded[start + t];
            y_values[b * block_size + t] = encoded[start + t + 1];
        }
    }

    const std::vector<int64_t> shape{
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(block_size)
    };
    return {
        make_uint8_t_tensor(x_values, shape),
        make_uint8_t_tensor(y_values, shape)
    };
}