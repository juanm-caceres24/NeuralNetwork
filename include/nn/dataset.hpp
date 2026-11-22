#pragma once

#include "nn/types.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace nn {

// Tabular supervised dataset supporting CSV I/O and O(N) zero-copy index shuffling.
class Dataset {
public:
    Dataset() = default;
    Dataset(std::size_t input_size, std::size_t target_size);

    void add_sample(std::vector<Scalar> input, std::vector<Scalar> target);

    std::size_t size() const noexcept { return inputs_.size(); }
    bool empty() const noexcept { return inputs_.empty(); }
    std::size_t input_size() const noexcept { return input_size_; }
    std::size_t target_size() const noexcept { return target_size_; }

    // Access sample through the current (potentially shuffled) index permutation.
    const std::vector<Scalar>& input_at(std::size_t logical_idx) const;
    const std::vector<Scalar>& target_at(std::size_t logical_idx) const;

    // Direct access to underlying storage in original insertion order.
    const std::vector<Scalar>& raw_input(std::size_t raw_idx) const;
    const std::vector<Scalar>& raw_target(std::size_t raw_idx) const;

    const std::vector<std::size_t>& indices() const noexcept { return indices_; }

    // Randomizes sample traversal order without moving feature/target vectors in memory.
    void shuffle(std::mt19937& rng);

    // Restores traversal order to 0, 1, ..., size() - 1.
    void reset_order();

    // Splits the dataset into (train, validation).
    // Set shuffle_before_split = false for chronological time-series splits (e.g., trading).
    std::pair<Dataset, Dataset> split(Scalar validation_fraction,
                                      bool shuffle_before_split = true,
                                      std::uint32_t seed = 42) const;

    // Loads a CSV file where each row has `input_cols` followed by `target_cols` floats.
    // Skips empty lines and lines starting with '#'.
    static Dataset load_csv(const std::string& path,
                            std::size_t input_cols,
                            std::size_t target_cols,
                            bool has_header = false);

    // Writes the dataset in original order to a CSV file using exact 32-bit float precision.
    void save_csv(const std::string& path, bool write_header = false) const;

private:
    std::size_t input_size_ = 0;
    std::size_t target_size_ = 0;
    std::vector<std::vector<Scalar>> inputs_;
    std::vector<std::vector<Scalar>> targets_;
    std::vector<std::size_t> indices_;
};

// Built-in synthetic dataset generators for testing and experimentation.
namespace datasets {

Dataset make_xor();
Dataset make_and();
Dataset make_or();
Dataset make_nand();

// 2D points in [-1, 1]^2 labeled 1 if (x + y > 0) else 0.
Dataset make_linear_separable(std::size_t num_samples, std::uint32_t seed = 42);

// Non-linear 2D points in [-1, 1]^2 labeled 1 if inside radius 0.65 else 0.
Dataset make_circle_classification(std::size_t num_samples, std::uint32_t seed = 42);

// 4-class 2D quadrant classification with one-hot targets (ideal for Softmax + CCE).
Dataset make_quadrant_multiclass(std::size_t num_samples, std::uint32_t seed = 42);

}  // namespace datasets
}  // namespace nn
