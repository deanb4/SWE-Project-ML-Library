#ifndef DATA_H
#define DATA_H


#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "tensor.h"

/**
 * @brief Loads image files (IDX format, e.g. train-images-idx3-ubyte).
 * @return FLOAT32 tensor of shape (N, image_height * image_width * channels), one image per row, 
 *         pixels normalized to 0-1.
 * @throws std::runtime_error If the file can't be opened.
 */
Tensor load_images(const std::string& path);

/**
 * @brief Loads label files (IDX format, e.g. train-labels-idx1-ubyte).
 * @return INT64 tensor of shape (N) holding the digit corresponding to the class of each image.
 * @throws std::runtime_error If the file can't be opened.
 */
Tensor load_labels(const std::string& path);

/**
 * @brief Copies the given rows (entries along the first dimension) of t into a new tensor.
 *
 * Used to build a mini-batch: select_rows(images, {5, 2, 9}) has shape (3, 784).
 *
 * @throws std::out_of_range If a row index is outside t.
 */
Tensor select_rows(const Tensor& t, const std::vector<int64_t>& rows);

/**
 * @class TextDataset
 * @brief Byte-level language-modeling dataset loaded from a .txt file.
 *
 * The file is read as raw bytes. Unique characters become a vocabulary in byte order,
 * so encode() / decode() map between text and integer token IDs in `[0, vocab_size)`.
 *
 * get_batch(B, T) draws B random windows of T tokens. Targets are the same windows
 * shifted one character to the right (next-character prediction).
 */
class TextDataset {
    private:
        std::string text;
        std::string vocab;
        std::vector<uint8_t> char_to_id;
        std::vector<uint8_t> encoded;
        mutable std::mt19937 rng;

    public:
        /**
         * @brief Reads `path` and builds a character vocabulary from its contents.
         * @throws std::runtime_error If the file cannot be opened or is empty.
         */
        explicit TextDataset(const std::string& path);

        size_t vocab_size() const;

        /** Characters in ID order: vocabulary()[id] is the character encoded as `id`. */
        const std::string& vocabulary() const;

        /** Encodes string `s` as an INT64 tensor of IDs of shape (L). Unknown characters throw. */
        Tensor encode(const std::string& s) const;

        /** Decodes an INT64 tensor of token IDs back to a string. */
        std::string decode(const Tensor& ids) const;

        /**
         * @brief Random mini-batch for next-character prediction.
         *
         * @param batch_size B, number of independent windows.
         * @param block_size T, number of tokens in each window.
         * @return `{x, y}` both INT64 of shape (B, T), with `y[b][t]` equal to the
         *         character that follows `x[b][t]` in the file.
         *
         * @throws std::invalid_argument If B or T is zero, or the text is shorter than T + 1.
         */
        std::pair<Tensor, Tensor> get_batch(size_t batch_size, size_t block_size) const;
};

#endif