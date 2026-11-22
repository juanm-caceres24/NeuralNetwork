#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        registry().push_back({std::move(name), std::move(fn)});
    }
};

inline int run_all_tests() {
    int passed = 0;
    int failed = 0;

    for (const auto& tc : registry()) {
        try {
            tc.fn();
            std::cout << "  [PASS] " << tc.name << "\n";
            ++passed;
        } catch (const std::exception& ex) {
            std::cerr << "  [FAIL] " << tc.name << ": " << ex.what() << "\n";
            ++failed;
        } catch (...) {
            std::cerr << "  [FAIL] " << tc.name << ": unknown exception\n";
            ++failed;
        }
    }

    std::cout << "Summary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}

}  // namespace nn_test

#define TEST_CASE(name)                                                           \
    static void test_fn_##name();                                                 \
    static ::nn_test::Registrar test_reg_##name(#name, test_fn_##name);           \
    static void test_fn_##name()

#define CHECK(expr)                                                               \
    do {                                                                          \
        if (!(expr)) {                                                            \
            std::ostringstream oss;                                               \
            oss << __FILE__ << ":" << __LINE__ << " CHECK failed: " << #expr;     \
            throw std::runtime_error(oss.str());                                  \
        }                                                                         \
    } while (false)

#define CHECK_NEAR(actual, expected, eps)                                         \
    do {                                                                          \
        const double a_val = static_cast<double>(actual);                         \
        const double e_val = static_cast<double>(expected);                       \
        if (!(std::fabs(a_val - e_val) <= static_cast<double>(eps))) {            \
            std::ostringstream oss;                                               \
            oss << __FILE__ << ":" << __LINE__ << " CHECK_NEAR failed: "          \
                << #actual << " (" << a_val << ") != " << #expected << " ("       \
                << e_val << "), eps=" << (eps);                                   \
            throw std::runtime_error(oss.str());                                  \
        }                                                                         \
    } while (false)

#define CHECK_THROWS(expr, exception_type)                                        \
    do {                                                                          \
        bool threw = false;                                                       \
        try {                                                                     \
            (void)(expr);                                                         \
        } catch (const exception_type&) {                                         \
            threw = true;                                                         \
        }                                                                         \
        if (!threw) {                                                             \
            std::ostringstream oss;                                               \
            oss << __FILE__ << ":" << __LINE__ << " CHECK_THROWS failed: "        \
                << #expr << " did not throw " << #exception_type;                 \
            throw std::runtime_error(oss.str());                                  \
        }                                                                         \
    } while (false)
