// omp_inverse.cpp - OpenMP parallel matrix inversion (Gauss-Jordan, partial pivoting)
// Same algorithm, input generator, verification and timing method as seq_inverse.cpp.
//
// Visual Studio 2022: Console App (C++), Release + x64.
//   Project Properties > C/C++ > Language > Open MP Support = Yes (/openmp)
//   (MSVC supports OpenMP 2.0, so every omp-for loop variable below is a signed int.)
// g++ : g++ -O2 -fopenmp omp_inverse.cpp -o omp_inverse
//
// Run seq_inverse.exe (no arguments) FIRST in the same folder, so results.csv holds the
// sequential times Ts for this machine. The speedup S = Ts/Tp and efficiency E = S/p are
// printed using those times.
//
// Usage:
//   omp_inverse.exe size [threads]     n = 256, 512, 1024, 2048, 4096 with 16 threads (or given) -> results_omp_size.csv (Table 4)
//   omp_inverse.exe threads            n = 1024, 2048 with 1..16 threads                   -> results_omp_threads.csv (Table 5)
//   omp_inverse.exe sched [threads]    n = 1024, 2048: static / dynamic / guided           -> results_omp_sched.csv   (Table 7)
//   omp_inverse.exe one <n> [threads] [reps]       one configuration
//   omp_inverse.exe selftest           compare with a sequential inverse (n = 128, 256, 512)
// Add  reps=3  as the LAST argument of size/threads/sched to use 3 runs instead of 5 (faster).

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

enum Sched { SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED };

static const char* sched_name(Sched s) {
    return s == SCHED_STATIC ? "static" : (s == SCHED_DYNAMIC ? "dynamic" : "guided");
}

// ---------------------------------------------------------------- input (identical to seq_inverse.cpp)
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

// Plain random matrix (NOT diagonally dominant): forces real row swaps. Used by selftest only.
static void generate_general(std::vector<double>& A, int n, unsigned seed) {
    std::mt19937 gen(seed);
    for (size_t i = 0; i < (size_t)n * n; i++) A[i] = gen() / 4294967295.0 * 2.0 - 1.0;
}

static void build_aug(const std::vector<double>& A, std::vector<double>& aug, int n) {
    const size_t N = n, W = 2 * N;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            aug[i * W + j] = A[i * N + j];
            aug[i * W + N + j] = (i == j) ? 1.0 : 0.0;
        }
}

// ---------------------------------------------------------------- sequential reference (for selftest)
static bool invert_seq(std::vector<double>& aug, int n) {
    const size_t N = n, W = 2 * N;
    std::vector<double> rowtmp(W);
    for (size_t k = 0; k < N; k++) {
        size_t p = k;
        double maxv = std::fabs(aug[k * W + k]);
        for (size_t i = k + 1; i < N; i++) {
            double v = std::fabs(aug[i * W + k]);
            if (v > maxv) { maxv = v; p = i; }
        }
        if (maxv < 1e-12) return false;
        if (p != k) std::swap_ranges(aug.begin() + k * W, aug.begin() + (k + 1) * W, aug.begin() + p * W);
        const double inv = 1.0 / aug[k * W + k];
        for (size_t j = k; j < W; j++) aug[k * W + j] *= inv;
        for (size_t i = 0; i < N; i++) {
            if (i == k) continue;
            const double f = aug[i * W + k];
            if (f == 0.0) continue;
            for (size_t j = k; j < W; j++) aug[i * W + j] -= f * aug[k * W + j];
        }
    }
    return true;
}

// ---------------------------------------------------------------- OpenMP inversion
// Eliminates column k from row i (same arithmetic as the sequential inner loops).
static inline void eliminate_row(double* a, int W, int k, int i) {
    if (i == k) return;
    const double f = a[(size_t)i * W + k];
    if (f == 0.0) return;
    for (int j = k; j < W; j++) a[(size_t)i * W + j] -= f * a[(size_t)k * W + j];
}

// aug = [A | I] (n rows, W = 2n columns) is inverted in place.
// One parallel region is opened ONCE and kept alive for all n pivot steps, so threads are not
// created and destroyed at every step. Each step has two barriers:
//   1. omp single : pivot search + row swap + normalisation (O(n) work, one thread)
//   2. omp for    : eliminate column k in all other rows (O(n^2) work, shared by all threads)
static bool invert_omp(std::vector<double>& aug, int n, int threads, Sched sc, int chunk) {
    const int W = 2 * n;
    double* a = aug.data();
    bool singular = false;

#pragma omp parallel num_threads(threads) shared(singular)
    {
        for (int k = 0; k < n; k++) {
#pragma omp single
            {
                int p = k;
                double maxv = std::fabs(a[(size_t)k * W + k]);
                for (int i = k + 1; i < n; i++) {
                    double v = std::fabs(a[(size_t)i * W + k]);
                    if (v > maxv) { maxv = v; p = i; }
                }
                if (maxv < 1e-12) {
                    singular = true;
                } else {
                    double* rk = a + (size_t)k * W;
                    double* rp = a + (size_t)p * W;
                    if (p != k) std::swap_ranges(rk, rk + W, rp);
                    const double inv = 1.0 / a[(size_t)k * W + k];
                    for (int j = k; j < W; j++) a[(size_t)k * W + j] *= inv;
                }
            }  // implicit barrier: every thread now sees the normalised pivot row
            if (singular) break;  // same decision in all threads

            if (sc == SCHED_STATIC) {
#pragma omp for schedule(static)
                for (int i = 0; i < n; i++) eliminate_row(a, W, k, i);
            } else if (sc == SCHED_DYNAMIC) {
#pragma omp for schedule(dynamic, chunk)
                for (int i = 0; i < n; i++) eliminate_row(a, W, k, i);
            } else {
#pragma omp for schedule(guided)
                for (int i = 0; i < n; i++) eliminate_row(a, W, k, i);
            }  // implicit barrier: column k is finished in every row before the next step
        }
    }
    return !singular;
}

// ---------------------------------------------------------------- verification (identical to seq_inverse.cpp)
static double verify(const std::vector<double>& A, const std::vector<double>& aug, int n) {
    const size_t N = n, W = 2 * N;
    std::vector<double> x(N), y(N, 0.0), z(N, 0.0);
    std::mt19937 gx(123);
    for (size_t i = 0; i < N; i++) x[i] = gx() / 4294967295.0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) y[i] += A[i * N + j] * x[j];
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) z[i] += aug[i * W + N + j] * y[j];
    double resid = 0.0;
    for (size_t i = 0; i < N; i++) resid = std::max(resid, std::fabs(z[i] - x[i]));
    return resid;
}

// ---------------------------------------------------------------- sequential times Ts from results.csv
static std::map<int, double> load_seq_ts(const char* path) {
    std::map<int, double> ts;
    std::ifstream f(path);
    std::string line;
    if (!std::getline(f, line)) return ts;  // header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> c;
        while (std::getline(ss, tok, ',')) c.push_back(tok);
        if (c.size() >= 6) ts[std::atoi(c[0].c_str())] = std::atof(c[5].c_str());  // column 6 = total_inv_ms
    }
    return ts;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Row {
    int n, threads;
    Sched sc;
    double tp_ms, ts_ms, speedup, eff, resid;
};

// Runs one configuration `reps` times and returns the median inversion time.
static bool bench(int n, int threads, Sched sc, int chunk, int reps, const std::map<int, double>& ts, Row& out) {
    std::vector<double> A((size_t)n * n), aug((size_t)n * 2 * n);
    generate(A, n, 42u);
    std::vector<double> times;
    double worst = 0.0;
    for (int r = 0; r < reps; r++) {
        build_aug(A, aug, n);                    // not timed (same as the sequential setup phase)
        auto t0 = Clock::now();
        bool ok = invert_omp(aug, n, threads, sc, chunk);
        double t = ms_since(t0);                 // only the inversion is timed
        if (!ok) { std::cerr << "Matrix is singular\n"; return false; }
        times.push_back(t);
        worst = std::max(worst, verify(A, aug, n));
    }
    out.n = n; out.threads = threads; out.sc = sc;
    out.tp_ms = median(times);
    auto it = ts.find(n);
    out.ts_ms = (it == ts.end()) ? -1.0 : it->second;
    out.speedup = out.ts_ms > 0 ? out.ts_ms / out.tp_ms : -1.0;
    out.eff = out.speedup > 0 ? out.speedup / threads : -1.0;
    out.resid = worst;
    if (worst > 1e-8) std::cerr << "WARNING: residual too large for n=" << n << "\n";
    return true;
}

static void print_header() {
    std::printf("%6s %8s %9s %12s %12s %9s %9s %10s\n", "n", "threads", "schedule", "Tp (ms)", "Ts (ms)",
                "speedup", "effic.", "residual");
}

static void print_row(const Row& r) {
    if (r.ts_ms > 0)
        std::printf("%6d %8d %9s %12.2f %12.2f %9.2f %9.3f %10.2e\n", r.n, r.threads, sched_name(r.sc), r.tp_ms,
                    r.ts_ms, r.speedup, r.eff, r.resid);
    else
        std::printf("%6d %8d %9s %12.2f %12s %9s %9s %10.2e\n", r.n, r.threads, sched_name(r.sc), r.tp_ms, "n/a",
                    "n/a", "n/a", r.resid);
    std::fflush(stdout);
}

static void write_row(std::ofstream& csv, const Row& r, int chunk) {
    csv << r.n << ',' << r.threads << ',' << sched_name(r.sc) << ',' << chunk << ',' << r.tp_ms << ',' << r.ts_ms
        << ',' << r.speedup << ',' << r.eff << ',' << r.resid << '\n';
    csv.flush();
}

static const char* CSV_HEADER = "n,threads,schedule,chunk,Tp_ms,Ts_ms,speedup,efficiency,residual\n";

// ---------------------------------------------------------------- self test
static int selftest() {
    int bad = 0;
    for (int kind = 0; kind < 2; kind++) {
        for (int n : {128, 256, 512}) {
            std::vector<double> A((size_t)n * n), ref((size_t)n * 2 * n);
            if (kind == 0) generate(A, n, 42u); else generate_general(A, n, 7u);
            build_aug(A, ref, n);
            if (!invert_seq(ref, n)) { std::printf("sequential reference failed\n"); return 1; }
            for (Sched sc : {SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED}) {
                for (int th : {1, 4}) {
                    std::vector<double> aug((size_t)n * 2 * n);
                    build_aug(A, aug, n);
                    bool ok = invert_omp(aug, n, th, sc, 8);
                    double diff = 0.0;
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            diff = std::max(diff, std::fabs(aug[(size_t)i * 2 * n + n + j] - ref[(size_t)i * 2 * n + n + j]));
                    double res = verify(A, aug, n);
                    bool pass = ok && diff < 1e-8 && res < 1e-6;
                    if (!pass) bad++;
                    std::printf("%-12s n=%4d %-8s threads=%d  max|Ainv_omp - Ainv_seq| = %.2e  residual = %.2e  %s\n",
                                kind == 0 ? "dominant" : "needs-pivot", n, sched_name(sc), th, diff, res,
                                pass ? "PASS" : "FAIL");
                }
            }
        }
    }
    if (bad == 0) std::printf("\nALL TESTS PASSED\n");
    else std::printf("\n%d TESTS FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    omp_set_dynamic(0);
    std::string mode = argc >= 2 ? argv[1] : "size";
    int reps = 5;
    // optional trailing "reps=N"
    if (argc >= 2 && std::strncmp(argv[argc - 1], "reps=", 5) == 0) {
        reps = std::max(1, std::atoi(argv[argc - 1] + 5));
        argc--;
    }
    const int CHUNK = 8;  // chunk size used for schedule(dynamic, chunk)
    std::map<int, double> ts = load_seq_ts("results.csv");
    if (ts.empty())
        std::printf("NOTE: results.csv (sequential times) not found - run seq_inverse.exe first for speedup values.\n\n");
    if (mode == "selftest") return selftest();

    if (mode == "one") {
        if (argc < 3) { std::cerr << "usage: omp_inverse one <n> [threads] [reps]\n"; return 1; }
        int n = std::atoi(argv[2]);
        int th = argc >= 4 ? std::atoi(argv[3]) : 16;
        if (argc >= 5) reps = std::max(1, std::atoi(argv[4]));
        std::printf("max threads available: %d, runs: %d (median)\n\n", omp_get_max_threads(), reps);
        Row r{};
        print_header();
        if (!bench(n, th, SCHED_STATIC, CHUNK, reps, ts, r)) return 2;
        print_row(r);
        return 0;
    }

    std::printf("max threads available: %d, runs per configuration: %d (median)\n\n", omp_get_max_threads(), reps);
    std::ofstream csv;
    print_header();
    if (mode == "size") {
        int th = argc >= 3 ? std::atoi(argv[2]) : 16;
        csv.open("results_omp_size.csv");
        csv << CSV_HEADER;
        for (int n : {256, 512, 1024, 2048, 4096}) {
            Row r{};
            if (!bench(n, th, SCHED_STATIC, CHUNK, reps, ts, r)) return 2;
            print_row(r); write_row(csv, r, 0);
        }
    } else if (mode == "threads") {
        csv.open("results_omp_threads.csv");
        csv << CSV_HEADER;
        for (int n : {1024, 2048})
            for (int th = 1; th <= 16; th++) {
                Row r{};
                if (!bench(n, th, SCHED_STATIC, CHUNK, reps, ts, r)) return 2;
                print_row(r); write_row(csv, r, 0);
            }
    } else if (mode == "sched") {
        int th = argc >= 3 ? std::atoi(argv[2]) : 16;
        csv.open("results_omp_sched.csv");
        csv << CSV_HEADER;
        for (int n : {1024, 2048})
            for (Sched sc : {SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED}) {
                Row r{};
                if (!bench(n, th, sc, CHUNK, reps, ts, r)) return 2;
                print_row(r); write_row(csv, r, sc == SCHED_DYNAMIC ? CHUNK : 0);
            }
    } else {
        std::cerr << "unknown mode '" << mode << "'. Modes: size, threads, sched, one, selftest\n";
        return 1;
    }
    std::printf("\nWrote results_omp_%s.csv (medians of %d runs)\n", mode.c_str(), reps);
    return 0;
}
