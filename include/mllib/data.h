#ifndef DATA_H
#define DATA_H


#include <cstdint>
#include <string>
#include <vector>
#include "tensor.h"

/**
 * @brief Loads image files (IDX format, e.g. train-images-idx3-ubyte).
 * @return FLOAT32 tensor of shape (N, image_height * image_width), one image per row, pixels normalized to 0-1.
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

#endif