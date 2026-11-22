#include "nn/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nn {

Matrix::Matrix(std::size_t rows, std::size_t cols, Scalar init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<Scalar> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Matrix data size does not match rows * cols");
    }
}

Scalar& Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[row * cols_ + col];
}

Scalar Matrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[row * cols_ + col];
}

void Matrix::fill(Scalar value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::zero() {
    fill(Scalar{0});
}

void Matrix::randomize_uniform(Scalar min_val, Scalar max_val, std::mt19937& rng) {
    if (min_val > max_val) {
        throw std::invalid_argument("randomize_uniform requires min_val <= max_val");
    }
    std::uniform_real_distribution<Scalar> dist(min_val, max_val);
    for (Scalar& value : data_) {
        value = dist(rng);
    }
}

void Matrix::randomize_xavier(std::mt19937& rng) {
    if (rows_ + cols_ == 0) {
        return;
    }
    const Scalar limit = std::sqrt(Scalar{6} / static_cast<Scalar>(rows_ + cols_));
    randomize_uniform(-limit, limit, rng);
}

void Matrix::randomize_he(std::mt19937& rng) {
    if (cols_ == 0) {
        return;
    }
    const Scalar stddev = std::sqrt(Scalar{2} / static_cast<Scalar>(cols_));
    std::normal_distribution<Scalar> dist(Scalar{0}, stddev);
    for (Scalar& value : data_) {
        value = dist(rng);
    }
}

void Matrix::matvec(const std::vector<Scalar>& x, std::vector<Scalar>& out) const {
    if (x.size() != cols_) {
        throw std::invalid_argument("Matrix::matvec input size mismatch");
    }
    out.resize(rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
        const Scalar* row_ptr = data_.data() + r * cols_;
        Scalar sum = Scalar{0};
        for (std::size_t c = 0; c < cols_; ++c) {
            sum += row_ptr[c] * x[c];
        }
        out[r] = sum;
    }
}

void Matrix::matvec_transpose(const std::vector<Scalar>& x, std::vector<Scalar>& out) const {
    if (x.size() != rows_) {
        throw std::invalid_argument("Matrix::matvec_transpose input size mismatch");
    }
    out.assign(cols_, Scalar{0});
    // Iterate in row-major order for sequential memory access on both M and out.
    for (std::size_t r = 0; r < rows_; ++r) {
        const Scalar* row_ptr = data_.data() + r * cols_;
        const Scalar xr = x[r];
        for (std::size_t c = 0; c < cols_; ++c) {
            out[c] += row_ptr[c] * xr;
        }
    }
}

void Matrix::add_outer_product(const std::vector<Scalar>& u, const std::vector<Scalar>& v) {
    if (u.size() != rows_ || v.size() != cols_) {
        throw std::invalid_argument("Matrix::add_outer_product dimension mismatch");
    }
    for (std::size_t r = 0; r < rows_; ++r) {
        Scalar* row_ptr = data_.data() + r * cols_;
        const Scalar ur = u[r];
        for (std::size_t c = 0; c < cols_; ++c) {
            row_ptr[c] += ur * v[c];
        }
    }
}

void Matrix::add_scaled(const Matrix& other, Scalar scale) {
    if (other.rows_ != rows_ || other.cols_ != cols_) {
        throw std::invalid_argument("Matrix::add_scaled dimension mismatch");
    }
    const std::size_t count = data_.size();
    for (std::size_t i = 0; i < count; ++i) {
        data_[i] += scale * other.data_[i];
    }
}

bool Matrix::approx_equal(const Matrix& other, Scalar eps) const noexcept {
    if (other.rows_ != rows_ || other.cols_ != cols_) {
        return false;
    }
    const std::size_t count = data_.size();
    for (std::size_t i = 0; i < count; ++i) {
        if (std::fabs(data_[i] - other.data_[i]) > eps) {
            return false;
        }
    }
    return true;
}

}  // namespace nn
