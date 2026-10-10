#include "mllib/embedding.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
    //Initialize weight tensor
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
        float* values_ptr = static_cast<float*>(storage->get_ptr());

        //Random number generater, seedable RNG to be implemented later MAKE NOTE OF THIS <---------------------------------------------------------------
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> distribution(0.0f, 0.02f);

        for(size_t i = 0; i < num_values; i++) {
            values_ptr[i] = distribution(generator);
        }

        //Create tensor that owns the embedding table
        return Tensor(
            std::move(storage),
            {static_cast<int64_t>(vocab_size), static_cast<int64_t>(embed_dim)},
            Device::CPU,
            true,
            Dtype::FLOAT32
        );
    }

    //Copy weights for each input ID to the output tensor    
    template <typename IdType> void copy_embedding_rows(
        const IdType* ids_values_ptr,
        size_t num_ids,
        const float* weight_values_ptr,
        float* output_values_ptr,
        size_t vocab_size,
        size_t embed_dim
    ) {
        for(size_t k = 0; k < num_ids; ++k) {
            //Read token ID with correct integer type
            const int64_t token_id = static_cast<int64_t>(ids_values_ptr[k]);

            //Reject IDs that are outside the valid vocabulary range
            if((token_id < 0) || (token_id >= vocab_size)) {
                throw std::out_of_range("Embedding::forward: token ID is outside the vocabulary");
            }

            //Calculate starting position of embedding row
            const size_t weight_offset = static_cast<size_t>(token_id) * embed_dim;

            //Calculate where this row belongs in the output
            const size_t output_offset = k * embed_dim;

            //Copy the entire embedding vector
            std::memcpy(
                output_values_ptr + output_offset,
                weight_values_ptr + weight_offset,
                embed_dim * sizeof(float)
            );
        }
    }

} //Anonymous namespace with helper functions

Embedding::Embedding(size_t vocab_size, size_t embed_dim) 
                     : weight(create_embedding_table(vocab_size, embed_dim)) { }

Tensor Embedding::forward(const Tensor& ids) const {
    const std::vector<int64_t>& input_shape = ids.get_shape();

    //Only rank 1 or rank 2 ID tensors are accepted as inputs
    if(input_shape.size() != 1 && input_shape.size() != 2) {
        throw std::invalid_argument("Embedding::forward: ids must have shape (T) or (B,T)");
    }

    //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
    if(ids.get_device() != Device::CPU) {
        throw std::invalid_argument("Embedding::forward: only CPU tensors are supported");
    }

    //getting embedding table dimensions
    const std::vector<int64_t>& weight_shape = weight.get_shape();
    const size_t vocab_size = static_cast<size_t>(weight_shape[0]);
    const size_t embed_dim = static_cast<size_t>(weight_shape[1]);

    //Calculate number of token IDs: for (T) num_ids = T, for (B, T) num_ids = B*T.
    size_t num_ids = 1;
    for(int64_t dim : input_shape) {
        const size_t dimension = static_cast<size_t>(dim);

        //Check if number of IDs would overflow size_t
        if( (dimension != 0) && (num_ids > (std::numeric_limits<size_t>::max() / dimension)) ) {
            throw std::overflow_error("Embedding::forward: number of IDs overflows size_t");
        }

        num_ids *= dimension;
    }

    //Check if number of output values (num_ids * embed_dim) would overflow size_t
    if( (num_ids > 0) && (embed_dim > (std::numeric_limits<size_t>::max() / num_ids)) ) {
        throw std::overflow_error("Embedding::forward: output size overflows size_t");
    }
    const size_t num_output_values = num_ids * embed_dim;

    //Check if number of bytes of output would overflow size_t
    if(num_output_values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::overflow_error("Embedding::forward: output byte size overflows size_t");
    }
    const size_t num_output_bytes = num_output_values * sizeof(float);

    //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
    auto output_storage = std::make_shared<Storage>(num_output_bytes, Device::CPU);

    //Constructing output shape: input (T) returns (T, embed_dim), input (B, T) returns (B, T, embed_dim)
    std::vector<int64_t> output_shape = input_shape;
    output_shape.push_back(static_cast<int64_t>(embed_dim)); //appends embed_dim

    //Get pointers to the memory of the embedding table (weights), output table, and input id table
    const float* weight_values_ptr = static_cast<const float*>(weight.get_data()->get_ptr());
    float* output_values_ptr = static_cast<float*>(output_storage->get_ptr());
    const void* ids_data = ids.get_data()->get_ptr();

    //Check dtype and use correct implementation
    switch(ids.get_dtype()) {
        case Dtype::INT64:
            copy_embedding_rows(
                static_cast<const int64_t*>(ids_data),
                num_ids,
                weight_values_ptr,
                output_values_ptr,
                vocab_size,
                embed_dim
            );
            break;
        case Dtype::INT32:
            copy_embedding_rows(
                static_cast<const int32_t*>(ids_data),
                num_ids,
                weight_values_ptr,
                output_values_ptr,
                vocab_size,
                embed_dim
            );
            break;
        case Dtype::INT16:
            copy_embedding_rows(
                static_cast<const int16_t*>(ids_data),
                num_ids,
                weight_values_ptr,
                output_values_ptr,
                vocab_size,
                embed_dim
            );
            break;
        case Dtype::INT8:
            copy_embedding_rows(
                static_cast<const int8_t*>(ids_data),
                num_ids,
                weight_values_ptr,
                output_values_ptr,
                vocab_size,
                embed_dim
            );
            break;
        case Dtype::UINT8:
            copy_embedding_rows(
                static_cast<const uint8_t*>(ids_data),
                num_ids,
                weight_values_ptr,
                output_values_ptr,
                vocab_size,
                embed_dim
            );
            break;
        //Non-integer dtypes are rejected (FLOAT64, FLOAT32, BOOL, etc.)
        default:
            throw std::invalid_argument("Embedding::forward: ids must have an integer dtype");
    }

    //Construct and return the output tensor
    //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
    return Tensor(
        std::move(output_storage),
        std::move(output_shape),
        Device::CPU,
        weight.get_requires_grad(),
        Dtype::FLOAT32
    );

}

std::shared_ptr<Tensor> Embedding::parameters() const {
    return std::make_shared<Tensor>(weight);
}
