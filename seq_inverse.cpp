// seq_inverse.cpp - Sequential matrix inversion (Gauss-Jordan, partial pivoting)
// Baseline for AMCS2103 Parallel Matrix Inversion assignment.
// Visual Studio 2022: Console App (C++), set Release + x64, run with Ctrl+F5.
//
// Usage:
//   seq_inverse.exe            -> benchmark mode: sizes 256, 512, 1024, 2048 (doubling), 5 runs each,
//                                 prints median table and writes results.csv
//   seq_inverse.exe <n> [reps] -> benchmark a single size (default 5 reps)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Result {
    int n;
    double setup_ms, pivot_ms, normalise_ms, eliminate_ms, total_inv_ms, verify_ms, residual;
};

// Diagonally dominant random matrix: always invertible, identical on every compiler
// (uses raw mt19937 output, not std::uniform_real_distribution, which differs per library).
static void generate(std::vector<double>& A, int n, unsigned seed) {
    std::mt19937 gen(seed);
    for (int i = 0; i < n; i++) {
        double rowsum = 0.0;
        for (int j = 0; j < n; j++) {
            double v = gen() / 4294967295.0 * 2.0 - 1.0;  // [-1, 1]
            A[(size_t)i * n + j] = v;
            rowsum += std::fabs(v);
        }
        A[(size_t)i * n + i] += rowsum + 1.0;
    }
}

// Returns false if the matrix is singular.
static bool run_once(int n, unsigned seed, Result& r) {
    const size_t N = (size_t)n, W = 2 * N;  // augmented width
    std::vector<double> A(N * N), aug(N * W), rowtmp(W);

    // ---- Phase 1: setup ----
    auto t0 = Clock::now();
    generate(A, n, seed);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            aug[i * W + j] = A[i * N + j];
            aug[i * W + N + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    r.setup_ms = ms_since(t0);

    // ---- Phase 2: Gauss-Jordan, timed per phase (per k-step) ----
    double pivot = 0, norm = 0, elim = 0;
    auto inv_start = Clock::now();
    for (size_t k = 0; k < N; k++) {
        // (a) pivot search + row swap
        auto t = Clock::now();
        size_t p = k;
        double maxv = std::fabs(aug[k * W + k]);
        for (size_t i = k + 1; i < N; i++) {
            double v = std::fabs(aug[i * W + k]);
            if (v > maxv) { maxv = v; p = i; }
        }
        if (maxv < 1e-12) return false;  // singular
        if (p != k) {
            std::copy(aug.begin() + k * W, aug.begin() + (k + 1) * W, rowtmp.begin());
            std::copy(aug.begin() + p * W, aug.begin() + (p + 1) * W, aug.begin() + k * W);
            std::copy(rowtmp.begin(), rowtmp.end(), aug.begin() + p * W);
        }
        pivot += ms_since(t);

        // (b) normalise pivot row
        t = Clock::now();
        const double inv = 1.0 / aug[k * W + k];
        for (size_t j = k; j < W; j++) aug[k * W + j] *= inv;
        norm += ms_since(t);

        // (c) eliminate column k from all other rows: O(n^2) per k, O(n^3) total
        t = Clock::now();
        for (size_t i = 0; i < N; i++) {
            if (i == k) continue;
            const double f = aug[i * W + k];
            if (f == 0.0) continue;
            for (size_t j = k; j < W; j++)
                aug[i * W + j] -= f * aug[k * W + j];
        }
        elim += ms_since(t);
    }
    r.total_inv_ms = ms_since(inv_start);
    r.pivot_ms = pivot; r.normalise_ms = norm; r.eliminate_ms = elim;

    // ---- Phase 3: fast O(n^2) verification: x -> y = A*x -> z = Ainv*y, check z == x ----
    t0 = Clock::now();
    std::vector<double> x(N), y(N, 0.0), z(N, 0.0);
    std::mt19937 gx(123);
    for (size_t i = 0; i < N; i++) x[i] = gx() / 4294967295.0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) y[i] += A[i * N + j] * x[j];
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) z[i] += aug[i * W + N + j] * y[j];
    double resid = 0.0;
    for (size_t i = 0; i < N; i++) resid = std::max(resid, std::fabs(z[i] - x[i]));
    r.verify_ms = ms_since(t0);
    r.residual = resid;
    r.n = n;
    return true;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Runs `reps` times, returns per-field medians.
static bool bench(int n, int reps, Result& out) {
    std::vector<double> s, p, nm, e, t, v, rs;
    for (int k = 0; k < reps; k++) {
        Result r{};
        if (!run_once(n, 42u, r)) { std::cerr << "Matrix is singular\n"; return false; }
        s.push_back(r.setup_ms);  p.push_back(r.pivot_ms);  nm.push_back(r.normalise_ms);
        e.push_back(r.eliminate_ms); t.push_back(r.total_inv_ms); v.push_back(r.verify_ms);
        rs.push_back(r.residual);
    }
    out = { n, median(s), median(p), median(nm), median(e), median(t), median(v),
            *std::max_element(rs.begin(), rs.end()) };
    return true;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    int reps = 5;
    if (argc >= 2) {
        sizes.push_back(std::atoi(argv[1]));
        if (argc >= 3) reps = std::max(1, std::atoi(argv[2]));
        if (sizes[0] <= 0) { std::cerr << "n must be positive\n"; return 1; }
    }
    else {
        // Fixed doubling progression: 256, 512, 1024, 2048, ... (each step = 8x the work)
        const int START_N = 256;
        const int MAX_N = 4096;   // set to 4096 for one more point (~8x longer; check RAM and patience)
        for (int n = START_N; n <= MAX_N; n *= 2) sizes.push_back(n);
    }

    const int base = 256;
    std::ofstream csv("results.csv");
    csv << "n,setup_ms,pivot_ms,normalise_ms,eliminate_ms,total_inv_ms,verify_ms,residual\n";

    std::printf("%6s %8s %10s %14s %14s %9s %10s\n",
        "n", "size x", "ops x(n^3)", "approx ops", "inversion ms", "elim %", "residual");
    for (int n : sizes) {
        Result r{};
        if (!bench(n, reps, r)) return 2;
        double ratio = (double)n / base;
        double ops = 1.5 * std::pow((double)n, 3.0);  // multiply-subtract operations
        std::printf("%6d %8.2f %10.3f %14.3e %14.2f %8.1f%% %10.2e\n",
            n, ratio, ratio * ratio * ratio, ops, r.total_inv_ms,
            100.0 * r.eliminate_ms / r.total_inv_ms, r.residual);
        csv << r.n << ',' << r.setup_ms << ',' << r.pivot_ms << ',' << r.normalise_ms << ','
            << r.eliminate_ms << ',' << r.total_inv_ms << ',' << r.verify_ms << ','
            << r.residual << '\n';
        if (r.residual > 1e-8) std::cerr << "WARNING: residual too large for n=" << n << "\n";
    }
    std::printf("\nWrote results.csv (medians of %d runs)\n", reps);
    return 0;
}