#include "mllib/embedding.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
    Tensor create_embedding_table(size_t vocab_size, size_t embed_dim) {
        //Since size_t does not allow negative numbers, only need to check if the passed parameters are zero
        if(vocab_size == 0) {
            throw std::invalid_argument("Embedding: vocab_size must be greater than zero");
        }
        if(embed_dim == 0) {
            throw std::invalid_argument("Embedding: embed_dim must be greater than zero");
        }

        //Checks if entered dimensions would overflow size_t (in case of absurdly large dimensions)
        if(vocab_size > (std::numeric_limits<size_t>::max() / embed_dim) ) { //(a*b > max) == (a > max/b)
            throw std::overflow_error("Embedding: table size overflows size_t");
        }
        size_t num_values = vocab_size * embed_dim;

        //Checks if number of bytes would overflow size_t
        if(num_values > (std::numeric_limits<size_t>::max() / sizeof(float)) ) {
            throw std::overflow_error("Embedding: table byte size overflows size_t");
        }
        size_t nbytes = num_values * sizeof(float);

        //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
        auto storage = std::make_shared<Storage>(nbytes, Device::CPU);

        //Getting pointer to raw FLOAT32 memory (enough memory for exactly vocab_size * embed_dim floats)
        float* values = static_cast<float*>(storage->get_ptr());

        //Random number generater, seedable RNG to be implemented later MAKE NOTE OF THIS <---------------------------------------------------------------
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> distribution(0.0f, 0.02f);

        for(size_t i = 0; i < num_values; i++) {
            values[i] = distribution(generator);
        }

        //Create tensor that owns the embedding table
        return Tensor(
            std::move(storage),
            {static_cast<int64_t>(vocab_size), static_cast<int64_t>(vocab_size)},
            Device::CPU,
            true,
            Dtype::FLOAT32
        );
    }
} //anonymous namespace with helper functions

Embedding::Embedding(size_t vocab_size, size_t embed_dim) 
    : weight(create_embedding_table(vocab_size, embed_dim)) { }

Tensor Embedding::forward(const Tensor& ids) const {
    //
}

std::shared_ptr<Tensor> Embedding::parameters() const {
    //
}