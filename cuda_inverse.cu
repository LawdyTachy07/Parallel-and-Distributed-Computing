// cuda_inverse.cu - CUDA parallel matrix inversion (Gauss-Jordan, partial pivoting)
// Same algorithm, input generator, verification and timing method as seq_inverse.cpp.
//
// Design: the augmented matrix [A | I] (n rows, W = 2n columns, row-major) lives in GPU memory.
// A is copied to the GPU once, all n pivot steps run on the GPU, and only the inverse (right half) is
// copied back. Every pivot step k launches four kernels in the same stream (no host synchronisation
// inside the loop, so the launches queue up back to back):
//   1. pivot_search    one block, parallel max-reduction of |M[i][k]| over i >= k (lowest index wins ties)
//   2. swap_normalise  swap rows k and p, divide the pivot row by the pivot (one thread per column j >= k)
//   3. save_column     copy the multipliers M[i][k] into a temporary vector d_f
//   4. eliminate       2-D grid, one thread per element (i, j): M[i][j] -= d_f[i] * M[k][j]
// The multipliers are saved first (kernel 3) because kernel 4 overwrites column k itself.
// Tp (the time reported) = copy A to the GPU + all kernels + copy A^-1 back to the host, measured with CUDA
// events. Device memory allocation and CUDA start-up are done once before timing (a warm-up run is made).
//
// Build (needs an NVIDIA GPU, the CUDA Toolkit, and Visual Studio's C++ compiler for nvcc on Windows):
//   Visual Studio 2022: New Project > "CUDA 12.x Runtime", replace kernel.cu by this file, Release + x64.
//   or, from an "x64 Native Tools Command Prompt for VS 2022":
//       nvcc -O3 -arch=native cuda_inverse.cu -o cuda_inverse.exe
//     (-arch=native needs CUDA 11.6 or newer; otherwise use -arch=sm_XX for your GPU, e.g. sm_75 for an RTX 2060)
//
// Run seq_inverse.exe (no arguments) FIRST in the same folder, so results.csv holds the sequential times Ts.
//
// Usage:
//   cuda_inverse.exe size [bx by]      n = 256, 512, 1024, 2048, 4096 (block bx x by, default 32 x 8) -> results_cuda_size.csv (Table 4)
//   cuda_inverse.exe block             n = 2048 with several thread-block shapes                  -> results_cuda_block.csv
//   cuda_inverse.exe one <n> [bx by] [reps]
//   cuda_inverse.exe selftest          compare with a sequential inverse (needs a GPU)
// Add  reps=3  as the LAST argument to use 3 runs instead of 5 (faster).

#include <cuda_runtime.h>

#include <algorithm>
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

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(3);                                                                       \
        }                                                                                       \
    } while (0)

// ================================================================ kernels
// Copies A into the left half of aug and writes the identity into the right half.
__global__ void init_aug(double* aug, const double* A, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        const size_t W = 2 * (size_t)n;
        aug[(size_t)i * W + j] = A[(size_t)i * n + j];
        aug[(size_t)i * W + n + j] = (i == j) ? 1.0 : 0.0;
    }
}

// One block of 1024 threads. Finds the row p >= k with the largest |M[p][k]| (lowest index on ties, like the
// sequential code) and stores p, the signed pivot value M[p][k], and a singular flag.
__global__ void pivot_search(const double* aug, int n, int k, int* d_piv, double* d_pivval, int* d_flag) {
    __shared__ double sv[1024];
    __shared__ int si[1024];
    const int t = threadIdx.x;
    const size_t W = 2 * (size_t)n;
    double best = -1.0;
    int bi = 0x7fffffff;
    for (int i = k + t; i < n; i += blockDim.x) {
        double v = fabs(aug[(size_t)i * W + k]);
        if (v > best) { best = v; bi = i; }
    }
    sv[t] = best;
    si[t] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            double ov = sv[t + s];
            int oi = si[t + s];
            if (ov > sv[t] || (ov == sv[t] && oi < si[t])) { sv[t] = ov; si[t] = oi; }
        }
        __syncthreads();
    }
    if (t == 0) {
        *d_piv = si[0];
        *d_pivval = aug[(size_t)si[0] * W + k];
        if (sv[0] < 1e-12) *d_flag = 1;  // singular: checked on the host after the last step
    }
}

// Thread j handles column j (k <= j < W) of rows k and p: swap them, then divide the new
// row k by the pivot. Columns j < k are never read again, so they are not swapped.
__global__ void swap_normalise(double* aug, int n, int k, const int* d_piv,
                               const double* d_pivval) {
    int j = k + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t W = 2 * (size_t)n;
    if (j >= (int)W) return;
    const int p = *d_piv;
    const double inv = 1.0 / *d_pivval;
    const double a = aug[(size_t)k * W + j];
    const double b = aug[(size_t)p * W + j];
    if (p != k) aug[(size_t)p * W + j] = a;
    aug[(size_t)k * W + j] = b * inv;
}

// d_f[i] = M[i][k] for all rows (the multipliers), taken after the swap and normalisation.
__global__ void save_column(const double* aug, int n, int k, double* d_f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_f[i] = aug[(size_t)i * 2 * (size_t)n + k];
}

// One thread per element: row i (y direction), column j >= k (x direction). Consecutive
// threads touch consecutive addresses, so the accesses are coalesced. Row k is only read.
__global__ void eliminate(double* aug, int n, int k, const double* d_f) {
    int j = k + blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t W = 2 * (size_t)n;
    if (j >= (int)W || i >= n || i == k) return;
    aug[(size_t)i * W + j] -= d_f[i] * aug[(size_t)k * W + j];
}

// ================================================================ host side
// Device buffers for one matrix size (allocated once, outside the timed region).
struct Gpu {
    int n = 0;
    double *d_A = nullptr, *d_aug = nullptr, *d_f = nullptr, *d_pivval = nullptr;
    int *d_piv = nullptr, *d_flag = nullptr;
    cudaEvent_t e0, e1, e2, e3;
    void alloc(int n_) {
        n = n_;
        CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * (size_t)n * n));
        CUDA_CHECK(cudaMalloc(&d_aug, sizeof(double) * (size_t)n * 2 * n));
        CUDA_CHECK(cudaMalloc(&d_f, sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(&d_pivval, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_piv, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
    }
    void release() {
        cudaFree(d_A); cudaFree(d_aug); cudaFree(d_f); cudaFree(d_pivval); cudaFree(d_piv); cudaFree(d_flag);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    }
};

struct Times {
    float total_ms, h2d_ms, kernel_ms, d2h_ms;
};

// Inverts A (n x n, row-major, on the host) and writes the inverse to Ainv (host). Returns false if singular.
static bool invert_cuda(Gpu& g, const std::vector<double>& A, std::vector<double>& Ainv, int bx, int by, Times& tm) {
    const int n = g.n, W = 2 * n;
    CUDA_CHECK(cudaMemset(g.d_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(g.e0));
    CUDA_CHECK(cudaMemcpy(g.d_A, A.data(), sizeof(double) * (size_t)n * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(g.e1));

    dim3 blk(bx, by);
    dim3 grid_init((n + bx - 1) / bx, (n + by - 1) / by);
    init_aug<<<grid_init, blk>>>(g.d_aug, g.d_A, n);

    const int TB = 256;  // threads per block for the 1-D kernels
    for (int k = 0; k < n; k++) {
        pivot_search<<<1, 1024>>>(g.d_aug, n, k, g.d_piv, g.d_pivval, g.d_flag);
        swap_normalise<<<(W - k + TB - 1) / TB, TB>>>(g.d_aug, n, k, g.d_piv, g.d_pivval);
        save_column<<<(n + TB - 1) / TB, TB>>>(g.d_aug, n, k, g.d_f);
        dim3 grid_el((W - k + bx - 1) / bx, (n + by - 1) / by);
        eliminate<<<grid_el, blk>>>(g.d_aug, n, k, g.d_f);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(g.e2));

    // copy only the right half (A^-1) back: n rows of n doubles, source pitch = 2n doubles
    CUDA_CHECK(cudaMemcpy2D(Ainv.data(), sizeof(double) * n, g.d_aug + n, sizeof(double) * W,
                            sizeof(double) * n, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(g.e3));
    CUDA_CHECK(cudaEventSynchronize(g.e3));

    int flag = 0;
    CUDA_CHECK(cudaMemcpy(&flag, g.d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (flag) return false;
    CUDA_CHECK(cudaEventElapsedTime(&tm.h2d_ms, g.e0, g.e1));
    CUDA_CHECK(cudaEventElapsedTime(&tm.kernel_ms, g.e1, g.e2));
    CUDA_CHECK(cudaEventElapsedTime(&tm.d2h_ms, g.e2, g.e3));
    CUDA_CHECK(cudaEventElapsedTime(&tm.total_ms, g.e0, g.e3));
    return true;
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

// Sequential reference (for selftest).
static bool invert_seq(const std::vector<double>& A, std::vector<double>& inv, int n) {
    const size_t N = n, W = 2 * N;
    std::vector<double> aug(N * W);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            aug[i * W + j] = A[i * N + j];
            aug[i * W + N + j] = (i == j) ? 1.0 : 0.0;
        }
    for (size_t k = 0; k < N; k++) {
        size_t p = k;
        double maxv = std::fabs(aug[k * W + k]);
        for (size_t i = k + 1; i < N; i++) {
            double v = std::fabs(aug[i * W + k]);
            if (v > maxv) { maxv = v; p = i; }
        }
        if (maxv < 1e-12) return false;
        if (p != k) std::swap_ranges(aug.begin() + k * W, aug.begin() + (k + 1) * W, aug.begin() + p * W);
        const double iv = 1.0 / aug[k * W + k];
        for (size_t j = k; j < W; j++) aug[k * W + j] *= iv;
        for (size_t i = 0; i < N; i++) {
            if (i == k) continue;
            const double f = aug[i * W + k];
            if (f == 0.0) continue;
            for (size_t j = k; j < W; j++) aug[i * W + j] -= f * aug[k * W + j];
        }
    }
    inv.assign(N * N, 0.0);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) inv[i * N + j] = aug[i * W + N + j];
    return true;
}

// Verification (identical to seq_inverse.cpp, but Ainv is n x n)
static double verify(const std::vector<double>& A, const std::vector<double>& Ainv, int n) {
    const size_t N = n;
    std::vector<double> x(N), y(N, 0.0), z(N, 0.0);
    std::mt19937 gx(123);
    for (size_t i = 0; i < N; i++) x[i] = gx() / 4294967295.0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) y[i] += A[i * N + j] * x[j];
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) z[i] += Ainv[i * N + j] * y[j];
    double resid = 0.0;
    for (size_t i = 0; i < N; i++) resid = std::max(resid, std::fabs(z[i] - x[i]));
    return resid;
}

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

struct Result {
    int n, bx, by;
    double tp, h2d, kernel, d2h, ts, speedup, resid;
};

static bool bench(int n, int bx, int by, int reps, const std::map<int, double>& ts, Result& res) {
    std::vector<double> A((size_t)n * n), Ainv((size_t)n * n);
    generate(A, n, 42u);
    Gpu g;
    g.alloc(n);
    std::vector<double> t_tot, t_h, t_k, t_d;
    double worst = 0.0;
    for (int r = 0; r < reps; r++) {
        Times tm{};
        if (!invert_cuda(g, A, Ainv, bx, by, tm)) { std::cerr << "Matrix is singular\n"; g.release(); return false; }
        t_tot.push_back(tm.total_ms); t_h.push_back(tm.h2d_ms); t_k.push_back(tm.kernel_ms); t_d.push_back(tm.d2h_ms);
        worst = std::max(worst, verify(A, Ainv, n));
    }
    g.release();
    res.n = n; res.bx = bx; res.by = by;
    res.tp = median(t_tot); res.h2d = median(t_h); res.kernel = median(t_k); res.d2h = median(t_d);
    auto it = ts.find(n);
    res.ts = (it == ts.end()) ? -1.0 : it->second;
    res.speedup = res.ts > 0 ? res.ts / res.tp : -1.0;
    res.resid = worst;
    if (worst > 1e-8) std::cerr << "WARNING: residual too large for n=" << n << "\n";
    return true;
}

static void print_header() {
    std::printf("%6s %8s %12s %10s %10s %10s %12s %9s %10s\n", "n", "block", "Tp (ms)", "H2D", "kernels", "D2H", "Ts (ms)",
                "speedup", "residual");
}

static void print_row(const Result& r) {
    char blk[16];
    std::snprintf(blk, sizeof blk, "%dx%d", r.bx, r.by);
    if (r.ts > 0)
        std::printf("%6d %8s %12.2f %10.2f %10.2f %10.2f %12.2f %9.2f %10.2e\n", r.n, blk, r.tp, r.h2d, r.kernel, r.d2h,
                    r.ts, r.speedup, r.resid);
    else
        std::printf("%6d %8s %12.2f %10.2f %10.2f %10.2f %12s %9s %10.2e\n", r.n, blk, r.tp, r.h2d, r.kernel, r.d2h, "n/a",
                    "n/a", r.resid);
    std::fflush(stdout);
}

static const char* CSV_HEADER = "n,block_x,block_y,Tp_ms,h2d_ms,kernels_ms,d2h_ms,Ts_ms,speedup,residual\n";

static void write_row(std::ofstream& csv, const Result& r) {
    csv << r.n << ',' << r.bx << ',' << r.by << ',' << r.tp << ',' << r.h2d << ',' << r.kernel << ',' << r.d2h << ','
        << r.ts << ',' << r.speedup << ',' << r.resid << '\n';
    csv.flush();
}

static int selftest() {
    int bad = 0;
    for (int kind = 0; kind < 2; kind++) {
        for (int n : {100, 128, 256, 512}) {  // 100: not a multiple of the block size
            std::vector<double> A((size_t)n * n), Ainv((size_t)n * n), ref;
            if (kind == 0) generate(A, n, 42u); else generate_general(A, n, 7u);
            invert_seq(A, ref, n);
            Gpu g;
            g.alloc(n);
            Times tm{};
            bool ok = invert_cuda(g, A, Ainv, 32, 8, tm);
            g.release();
            double diff = 0.0;
            for (size_t i = 0; i < (size_t)n * n; i++) diff = std::max(diff, std::fabs(Ainv[i] - ref[i]));
            double res = verify(A, Ainv, n);
            bool pass = ok && diff < 1e-8 && res < 1e-6;
            if (!pass) bad++;
            std::printf("%-12s n=%4d  max|Ainv_gpu - Ainv_seq| = %.2e  residual = %.2e  %s\n",
                        kind == 0 ? "dominant" : "needs-pivot", n, diff, res, pass ? "PASS" : "FAIL");
        }
    }
    if (bad == 0) std::printf("\nALL TESTS PASSED\n");
    else std::printf("\n%d TESTS FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string mode = argc >= 2 ? argv[1] : "size";
    int reps = 5;
    if (argc >= 2 && std::strncmp(argv[argc - 1], "reps=", 5) == 0) {
        reps = std::max(1, std::atoi(argv[argc - 1] + 5));
        argc--;
    }

    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
        std::fprintf(stderr, "No CUDA-capable GPU found.\n");
        return 4;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (compute capability %d.%d, %.1f GB), runs per configuration: %d (median)\n", prop.name,
                prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0, reps);

    // Warm-up: the first CUDA call creates the context (hundreds of ms) and must not be timed.
    {
        std::vector<double> A(128 * 128), Ainv(128 * 128);
        generate(A, 128, 42u);
        Gpu g;
        g.alloc(128);
        Times tm{};
        invert_cuda(g, A, Ainv, 32, 8, tm);
        g.release();
    }

    if (mode == "selftest") return selftest();

    std::map<int, double> ts = load_seq_ts("results.csv");
    if (ts.empty())
        std::printf("NOTE: results.csv (sequential times) not found - run seq_inverse.exe first for speedup values.\n");
    std::printf("\n");

    if (mode == "one") {
        if (argc < 3) { std::cerr << "usage: cuda_inverse one <n> [bx by] [reps]\n"; return 1; }
        int n = std::atoi(argv[2]);
        int bx = argc >= 5 ? std::atoi(argv[3]) : 32, by = argc >= 5 ? std::atoi(argv[4]) : 8;
        if (argc >= 6) reps = std::max(1, std::atoi(argv[5]));
        Result r{};
        print_header();
        if (!bench(n, bx, by, reps, ts, r)) return 2;
        print_row(r);
        return 0;
    }

    std::ofstream csv;
    print_header();
    if (mode == "size") {
        int bx = argc >= 4 ? std::atoi(argv[2]) : 32, by = argc >= 4 ? std::atoi(argv[3]) : 8;
        csv.open("results_cuda_size.csv");
        csv << CSV_HEADER;
        for (int n : {256, 512, 1024, 2048, 4096}) {
            Result r{};
            if (!bench(n, bx, by, reps, ts, r)) return 2;
            print_row(r); write_row(csv, r);
        }
    } else if (mode == "block") {
        csv.open("results_cuda_block.csv");
        csv << CSV_HEADER;
        const int shapes[][2] = {{32, 4}, {32, 8}, {32, 16}, {16, 16}, {64, 4}, {128, 2}};
        for (auto& s : shapes) {
            Result r{};
            if (!bench(2048, s[0], s[1], reps, ts, r)) return 2;
            print_row(r); write_row(csv, r);
        }
    } else {
        std::cerr << "unknown mode '" << mode << "'. Modes: size, block, one, selftest\n";
        return 1;
    }
    std::printf("\nWrote results_cuda_%s.csv (medians of %d runs)\n", mode.c_str(), reps);
    return 0;
}
