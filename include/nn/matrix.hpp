#pragma once

#include "nn/types.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace nn {

// 2D matrix backed by a single contiguous row-major 1D buffer.
// Element (row, col) is located at index (row * cols_ + col).
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, Scalar init_value = Scalar{0});
    Matrix(std::size_t rows, std::size_t cols, std::vector<Scalar> data);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    // Unchecked element access for performance-critical loops.
    Scalar& operator()(std::size_t row, std::size_t col) noexcept {
        return data_[row * cols_ + col];
    }
    Scalar operator()(std::size_t row, std::size_t col) const noexcept {
        return data_[row * cols_ + col];
    }

    // Bounds-checked element access.
    Scalar& at(std::size_t row, std::size_t col);
    Scalar at(std::size_t row, std::size_t col) const;

    Scalar* data() noexcept { return data_.data(); }
    const Scalar* data() const noexcept { return data_.data(); }
    const std::vector<Scalar>& vector() const noexcept { return data_; }

    void fill(Scalar value);
    void zero();

    void randomize_uniform(Scalar min_val, Scalar max_val, std::mt19937& rng);
    void randomize_xavier(std::mt19937& rng);
    void randomize_he(std::mt19937& rng);

    // Computes out = M * x  (M is rows x cols, x has size cols, out has size rows).
    void matvec(const std::vector<Scalar>& x, std::vector<Scalar>& out) const;

    // Computes out = M^T * x  (M is rows x cols, x has size rows, out has size cols).
    void matvec_transpose(const std::vector<Scalar>& x, std::vector<Scalar>& out) const;

    // Accumulates M += u * v^T  (u has size rows, v has size cols).
    void add_outer_product(const std::vector<Scalar>& u, const std::vector<Scalar>& v);

    // Accumulates M += scale * other  (dimensions must match).
    void add_scaled(const Matrix& other, Scalar scale);

    bool approx_equal(const Matrix& other, Scalar eps = Scalar{1e-5f}) const noexcept;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<Scalar> data_;
};

}  // namespace nn
