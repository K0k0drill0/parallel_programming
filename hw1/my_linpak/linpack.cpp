#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

using Matrix = std::vector<double>;

// Row-major storage. On return, the strict lower triangle contains L
// (its diagonal is implicitly 1), and the upper triangle contains U.
void factorize(Matrix& a, std::vector<std::size_t>& pivots, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i * n + k]) > std::abs(a[pivot * n + k])) {
                pivot = i;
            }
        }
        if (a[pivot * n + k] == 0.0) {
            throw std::runtime_error("Singular matrix");
        }
        pivots[k] = pivot;
        if (pivot != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[k * n + j], a[pivot * n + j]);
            }
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            a[i * n + k] /= a[k * n + k];
            const double multiplier = a[i * n + k];
            for (std::size_t j = k + 1; j < n; ++j) {
                a[i * n + j] -= multiplier * a[k * n + j];
            }
        }
    }
}

// Apply P to b, solve Ly = Pb, then Ux = y. Overwrite b with x.
void solve(const Matrix& lu, const std::vector<std::size_t>& pivots,
           std::vector<double>& b, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        std::swap(b[k], b[pivots[k]]);
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            b[i] -= lu[i * n + j] * b[j];
        }
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j) {
            b[i] -= lu[i * n + j] * b[j];
        }
        b[i] /= lu[i * n + i];
    }
}

std::size_t parse_size(std::string_view argument) {
    std::size_t n = 0;
    const auto result = std::from_chars(argument.data(),
                                        argument.data() + argument.size(), n);
    if (result.ec != std::errc{} || result.ptr != argument.data() + argument.size()
        || n == 0) {
        throw std::invalid_argument("N must be a positive integer");
    }
    if (n > Matrix{}.max_size() / n) {
        throw std::invalid_argument("N is too large for matrix storage");
    }
    return n;
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [N]\nDefault N: 1000\n";
            return 0;
        }
        if (argc > 2) {
            throw std::invalid_argument("Usage: linpack [N]");
        }
        const std::size_t n = argc == 2 ? parse_size(argv[1]) : 1000;
        Matrix a(n * n);
        std::vector<double> b(n);
        std::mt19937 generator(42);
        for (std::size_t i = 0; i < n; ++i) {
            long double sum = 0.0L;
            for (std::size_t j = 0; j < n; ++j) {
                a[i * n + j] = static_cast<double>(generator())
                              / static_cast<double>(std::mt19937::max()) - 0.5;
                sum += a[i * n + j];
            }
            // The intended solution is x = (1, ..., 1), up to rounding in b.
            b[i] = static_cast<double>(sum);
        }
        Matrix lu = a;
        std::vector<double> x = b;
        std::vector<std::size_t> pivots(n);

        const auto start = std::chrono::steady_clock::now();
        factorize(lu, pivots, n);
        solve(lu, pivots, x, n);
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        if (seconds <= 0.0) {
            throw std::runtime_error("Timer resolution is insufficient; increase N");
        }

        long double residual = 0.0L;
        long double a_norm = 0.0L;
        long double b_norm = 0.0L;
        long double x_norm = 0.0L;
        long double error = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i])) {
                throw std::runtime_error("Non-finite solution");
            }
            long double ax = 0.0L;
            long double row_norm = 0.0L;
            for (std::size_t j = 0; j < n; ++j) {
                ax += static_cast<long double>(a[i * n + j]) * x[j];
                row_norm += std::abs(a[i * n + j]);
            }
            residual = std::max(residual, std::abs(ax - b[i]));
            a_norm = std::max(a_norm, row_norm);
            b_norm = std::max(b_norm, std::abs(static_cast<long double>(b[i])));
            x_norm = std::max(x_norm, std::abs(static_cast<long double>(x[i])));
            error = std::max(error, std::abs(static_cast<long double>(x[i]) - 1.0L));
        }
        const long double scaled_residual = residual
            / (std::numeric_limits<double>::epsilon() * static_cast<long double>(n)
               * (a_norm * x_norm + b_norm));
        const bool passed = std::isfinite(scaled_residual) && scaled_residual < 16.0L;
        const double size = static_cast<double>(n);
        const double operations = (2.0 / 3.0) * size * size * size + 2.0 * size * size;

        std::cout << "Sequential LINPACK-style benchmark (double)\n"
                  << "N: " << n << '\n'
                  << std::fixed << std::setprecision(6)
                  << "Time (s): " << seconds << '\n'
                  << "GFLOPS: " << operations / seconds / 1e9 << '\n'
                  << std::scientific
                  << "Residual ||Ax-b||_inf: " << residual << '\n'
                  << "Error ||x-1||_inf: " << error << '\n'
                  << "Scaled residual: " << scaled_residual << '\n'
                  << "Check (scaled residual < 16): " << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
