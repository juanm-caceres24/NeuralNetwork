#include "nn/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nn {

namespace {

std::string trim_copy(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string format_scalar_exact(Scalar value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(value));
    return std::string(buf);
}

}  // namespace

Dataset::Dataset(std::size_t input_size, std::size_t target_size)
    : input_size_(input_size), target_size_(target_size) {
    if (input_size_ == 0 || target_size_ == 0) {
        throw std::invalid_argument("Dataset input_size and target_size must be non-zero");
    }
}

void Dataset::add_sample(std::vector<Scalar> input, std::vector<Scalar> target) {
    if (input.empty() || target.empty()) {
        throw std::invalid_argument("Dataset sample input and target must not be empty");
    }
    if (input_size_ == 0 && target_size_ == 0) {
        input_size_ = input.size();
        target_size_ = target.size();
    } else if (input.size() != input_size_ || target.size() != target_size_) {
        throw std::invalid_argument("Dataset sample dimensions do not match dataset schema");
    }

    const std::size_t next_idx = inputs_.size();
    inputs_.push_back(std::move(input));
    targets_.push_back(std::move(target));
    indices_.push_back(next_idx);
}

const std::vector<Scalar>& Dataset::input_at(std::size_t logical_idx) const {
    return inputs_[indices_.at(logical_idx)];
}

const std::vector<Scalar>& Dataset::target_at(std::size_t logical_idx) const {
    return targets_[indices_.at(logical_idx)];
}

const std::vector<Scalar>& Dataset::raw_input(std::size_t raw_idx) const {
    return inputs_.at(raw_idx);
}

const std::vector<Scalar>& Dataset::raw_target(std::size_t raw_idx) const {
    return targets_.at(raw_idx);
}

void Dataset::shuffle(std::mt19937& rng) {
    std::shuffle(indices_.begin(), indices_.end(), rng);
}

void Dataset::reset_order() {
    std::iota(indices_.begin(), indices_.end(), static_cast<std::size_t>(0));
}

Dataset Dataset::load_csv(const std::string& path,
                          std::size_t input_cols,
                          std::size_t target_cols,
                          bool has_header) {
    Dataset ds(input_cols, target_cols);
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV dataset file: '" + path + "'");
    }

    const std::size_t expected_cols = input_cols + target_cols;
    std::string line;
    std::size_t line_num = 0;
    bool skipped_header = !has_header;

    while (std::getline(file, line)) {
        ++line_num;
        const std::string trimmed = trim_copy(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (!skipped_header) {
            skipped_header = true;
            continue;
        }

        std::vector<Scalar> values;
        values.reserve(expected_cols);
        std::stringstream ss(trimmed);
        std::string token;

        while (std::getline(ss, token, ',')) {
            const std::string cell = trim_copy(token);
            if (cell.empty()) {
                throw std::runtime_error("Empty CSV field at " + path + ":" +
                                         std::to_string(line_num));
            }
            std::size_t consumed = 0;
            Scalar parsed = Scalar{0};
            try {
                parsed = std::stof(cell, &consumed);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid numeric value '" + cell + "' at " + path + ":" +
                                         std::to_string(line_num));
            }
            if (consumed != cell.size()) {
                throw std::runtime_error("Trailing characters in CSV value '" + cell + "' at " +
                                         path + ":" + std::to_string(line_num));
            }
            values.push_back(parsed);
        }

        if (values.size() != expected_cols) {
            throw std::runtime_error("Expected " + std::to_string(expected_cols) +
                                     " columns but found " + std::to_string(values.size()) +
                                     " at " + path + ":" + std::to_string(line_num));
        }

        std::vector<Scalar> input(values.begin(),
                                  values.begin() + static_cast<std::ptrdiff_t>(input_cols));
        std::vector<Scalar> target(values.begin() + static_cast<std::ptrdiff_t>(input_cols),
                                   values.end());
        ds.add_sample(std::move(input), std::move(target));
    }

    return ds;
}

void Dataset::save_csv(const std::string& path, bool write_header) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file for writing: '" + path + "'");
    }

    if (write_header) {
        for (std::size_t i = 0; i < input_size_; ++i) {
            if (i > 0) file << ',';
            file << "x" << i;
        }
        for (std::size_t j = 0; j < target_size_; ++j) {
            file << ",y" << j;
        }
        file << '\n';
    }

    for (std::size_t row = 0; row < inputs_.size(); ++row) {
        for (std::size_t i = 0; i < input_size_; ++i) {
            if (i > 0) file << ',';
            file << format_scalar_exact(inputs_[row][i]);
        }
        for (std::size_t j = 0; j < target_size_; ++j) {
            file << ',' << format_scalar_exact(targets_[row][j]);
        }
        file << '\n';
    }
}

namespace datasets {

Dataset make_xor() {
    Dataset ds(2, 1);
    ds.add_sample({0.0f, 0.0f}, {0.0f});
    ds.add_sample({0.0f, 1.0f}, {1.0f});
    ds.add_sample({1.0f, 0.0f}, {1.0f});
    ds.add_sample({1.0f, 1.0f}, {0.0f});
    return ds;
}

Dataset make_and() {
    Dataset ds(2, 1);
    ds.add_sample({0.0f, 0.0f}, {0.0f});
    ds.add_sample({0.0f, 1.0f}, {0.0f});
    ds.add_sample({1.0f, 0.0f}, {0.0f});
    ds.add_sample({1.0f, 1.0f}, {1.0f});
    return ds;
}

Dataset make_or() {
    Dataset ds(2, 1);
    ds.add_sample({0.0f, 0.0f}, {0.0f});
    ds.add_sample({0.0f, 1.0f}, {1.0f});
    ds.add_sample({1.0f, 0.0f}, {1.0f});
    ds.add_sample({1.0f, 1.0f}, {1.0f});
    return ds;
}

Dataset make_nand() {
    Dataset ds(2, 1);
    ds.add_sample({0.0f, 0.0f}, {1.0f});
    ds.add_sample({0.0f, 1.0f}, {1.0f});
    ds.add_sample({1.0f, 0.0f}, {1.0f});
    ds.add_sample({1.0f, 1.0f}, {0.0f});
    return ds;
}

Dataset make_linear_separable(std::size_t num_samples, std::uint32_t seed) {
    Dataset ds(2, 1);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < num_samples; ++i) {
        const Scalar x = dist(rng);
        const Scalar y = dist(rng);
        const Scalar label = (x + y > Scalar{0}) ? Scalar{1} : Scalar{0};
        ds.add_sample({x, y}, {label});
    }
    return ds;
}

Dataset make_circle_classification(std::size_t num_samples, std::uint32_t seed) {
    Dataset ds(2, 1);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    constexpr Scalar kRadiusSq = Scalar{0.65f * 0.65f};
    for (std::size_t i = 0; i < num_samples; ++i) {
        const Scalar x = dist(rng);
        const Scalar y = dist(rng);
        const Scalar label = (x * x + y * y < kRadiusSq) ? Scalar{1} : Scalar{0};
        ds.add_sample({x, y}, {label});
    }
    return ds;
}

}  // namespace datasets
}  // namespace nn
