#include "nn/matrix.hpp"
#include "test_framework.hpp"

#include <random>
#include <stdexcept>
#include <vector>

using nn::Matrix;
using nn::Scalar;

TEST_CASE(matrix_construction_and_indexing) {
    Matrix m(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 3);
    CHECK(m.size() == 6);
    CHECK(!m.empty());

    CHECK_NEAR(m(0, 0), 1.0f, 1e-6f);
    CHECK_NEAR(m(0, 2), 3.0f, 1e-6f);
    CHECK_NEAR(m(1, 0), 4.0f, 1e-6f);
    CHECK_NEAR(m.at(1, 2), 6.0f, 1e-6f);

    // Row-major contiguous layout verification
    const Scalar* raw = m.data();
    CHECK_NEAR(raw[0], 1.0f, 1e-6f);
    CHECK_NEAR(raw[3], 4.0f, 1e-6f);
    CHECK_NEAR(raw[5], 6.0f, 1e-6f);

    CHECK_THROWS(m.at(2, 0), std::out_of_range);
    CHECK_THROWS(m.at(0, 3), std::out_of_range);
    CHECK_THROWS(Matrix(2, 2, {1.0f, 2.0f}), std::invalid_argument);
}

TEST_CASE(matrix_matvec_and_transpose) {
    // M = [[1, 2, 3],
    //      [4, 5, 6]]
    Matrix m(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::vector<Scalar> x = {10.0f, 20.0f, 30.0f};
    std::vector<Scalar> out;

    m.matvec(x, out);
    CHECK(out.size() == 2);
    // 1*10 + 2*20 + 3*30 = 140
    // 4*10 + 5*20 + 6*30 = 320
    CHECK_NEAR(out[0], 140.0f, 1e-5f);
    CHECK_NEAR(out[1], 320.0f, 1e-5f);

    // M^T * y where y = [2, 3]
    // [1*2 + 4*3, 2*2 + 5*3, 3*2 + 6*3] = [14, 19, 24]
    std::vector<Scalar> y = {2.0f, 3.0f};
    std::vector<Scalar> out_t;
    m.matvec_transpose(y, out_t);
    CHECK(out_t.size() == 3);
    CHECK_NEAR(out_t[0], 14.0f, 1e-5f);
    CHECK_NEAR(out_t[1], 19.0f, 1e-5f);
    CHECK_NEAR(out_t[2], 24.0f, 1e-5f);

    CHECK_THROWS(m.matvec({1.0f, 2.0f}, out), std::invalid_argument);
    CHECK_THROWS(m.matvec_transpose({1.0f, 2.0f, 3.0f}, out_t), std::invalid_argument);
}

TEST_CASE(matrix_add_outer_product_and_add_scaled) {
    Matrix grad(2, 3, 0.0f);
    grad.add_outer_product({2.0f, -1.0f}, {3.0f, 4.0f, 5.0f});
    // [[6, 8, 10], [-3, -4, -5]]
    Matrix expected(2, 3, {6.0f, 8.0f, 10.0f, -3.0f, -4.0f, -5.0f});
    CHECK(grad.approx_equal(expected, 1e-6f));

    // Accumulate a second outer product
    grad.add_outer_product({1.0f, 1.0f}, {1.0f, 1.0f, 1.0f});
    Matrix expected2(2, 3, {7.0f, 9.0f, 11.0f, -2.0f, -3.0f, -4.0f});
    CHECK(grad.approx_equal(expected2, 1e-6f));

    // SGD-style scaled update: W += (-0.5) * grad
    Matrix w(2, 3, 10.0f);
    w.add_scaled(grad, -0.5f);
    CHECK_NEAR(w(0, 0), 6.5f, 1e-5f);
    CHECK_NEAR(w(1, 2), 12.0f, 1e-5f);

    CHECK_THROWS(grad.add_outer_product({1.0f}, {1.0f, 2.0f, 3.0f}), std::invalid_argument);
    CHECK_THROWS(w.add_scaled(Matrix(3, 2), 1.0f), std::invalid_argument);
}

TEST_CASE(matrix_deterministic_seeded_initialization) {
    Matrix a(16, 16);
    Matrix b(16, 16);
    std::mt19937 rng_a(12345);
    std::mt19937 rng_b(12345);

    a.randomize_xavier(rng_a);
    b.randomize_xavier(rng_b);
    CHECK(a.approx_equal(b, 0.0f));

    const Scalar xavier_limit = std::sqrt(6.0f / 32.0f);
    for (Scalar val : a.vector()) {
        CHECK(val >= -xavier_limit && val <= xavier_limit);
    }

    a.randomize_he(rng_a);
    b.randomize_he(rng_b);
    CHECK(a.approx_equal(b, 0.0f));

    CHECK_THROWS(a.randomize_uniform(2.0f, 1.0f, rng_a), std::invalid_argument);
}

int main() {
    return nn_test::run_all_tests();
}
