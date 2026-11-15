// mpi_inverse.cpp - MPI parallel matrix inversion (Gauss-Jordan, partial pivoting)
// Same algorithm, input generator, verification and timing method as seq_inverse.cpp.
//
// Design: the rows of the augmented matrix [A | I] are divided into contiguous blocks, one block per
// process. In every pivot step k:
//   1. each process finds its best pivot candidate in column k; MPI_Allreduce(MPI_MAXLOC) picks the global pivot
//   2. if the pivot row p and row k live on different processes, the two rows are exchanged (MPI_Sendrecv)
//   3. the owner of row k normalises it and broadcasts it to everybody (MPI_Bcast)
//   4. every process eliminates column k from its own rows (no communication)
// Tp (the time reported) = scatter of A + all n pivot steps + gather of A^-1 on process 0.
//
// Visual Studio 2022 + MS-MPI (install the MS-MPI runtime AND the MS-MPI SDK):
//   Console App (C++), Release + x64, then in Project Properties:
//     VC++ Directories > Include Directories: add  $(MSMPI_INC);$(MSMPI_INC)\x64
//     VC++ Directories > Library Directories: add  $(MSMPI_LIB64)
//     Linker > Input > Additional Dependencies: add  msmpi.lib
//   or, from an "x64 Native Tools Command Prompt for VS 2022":
//     cl /O2 /EHsc /I"%MSMPI_INC%" /I"%MSMPI_INC%\x64" mpi_inverse.cpp /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib
// Linux / Mac:  mpicxx -O2 mpi_inverse.cpp -o mpi_inverse
//
// Run seq_inverse.exe (no arguments) FIRST in the same folder, so results.csv holds the sequential times Ts.
//
// Usage (P = number of processes):
//   mpiexec -n P mpi_inverse.exe size       n = 256, 512, 1024, 2048, 4096 -> results_mpi_size.csv   (Table 4; use P = 8 or 16)
//   mpiexec -n P mpi_inverse.exe procs      n = 1024, 2048             -> results_mpi_procs.csv  (Table 6; APPENDS a row per n,
//                                                                          so run it for P = 1, 2, 4, 8, 16)
//   mpiexec -n P mpi_inverse.exe one <n> [reps]
//   mpiexec -n P mpi_inverse.exe selftest   compare with a sequential inverse
// Add  reps=3  as the LAST argument to use 3 runs instead of 5 (faster).

#include <mpi.h>

#include <algorithm>
#include <climits>
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

// ---------------------------------------------------------------- sequential reference (for selftest, rank 0)
static bool invert_seq(const std::vector<double>& A, std::vector<double>& inv, int n) {
    const size_t N = n, W = 2 * N;
    std::vector<double> aug(N * W), rowtmp(W);
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

// ---------------------------------------------------------------- block row distribution
struct Dist {
    int n, p;
    std::vector<int> start, count;
    Dist(int n_, int p_) : n(n_), p(p_), start(p_), count(p_) {
        int base = n / p, rem = n % p, s = 0;
        for (int r = 0; r < p; r++) {
            count[r] = base + (r < rem ? 1 : 0);
            start[r] = s;
            s += count[r];
        }
    }
    int owner(int i) const {
        int base = n / p, rem = n % p, cut = rem * (base + 1);
        return (i < cut) ? i / (base + 1) : rem + (i - cut) / base;
    }
};

struct Times {
    double total_ms, scatter_ms, steps_ms, gather_ms;
};

// Inverts the n x n matrix Aroot (valid on rank 0 only). The inverse is returned in Ainv (rank 0 only).
// Returns false (on every rank) if the matrix is singular.
static bool invert_mpi(const double* Aroot, double* Ainv, int n, int rank, int p, const Dist& D, Times& tm) {
    const int W = 2 * n;
    const int myrows = D.count[rank], mystart = D.start[rank];
    std::vector<double> Aloc((size_t)myrows * n), aug((size_t)myrows * W), pivrow(W), tmp(W), invloc((size_t)myrows * n);
    std::vector<int> cnt(p), disp(p);
    for (int r = 0; r < p; r++) { cnt[r] = D.count[r] * n; disp[r] = D.start[r] * n; }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    // (1) distribute the rows of A, then build the local part of [A | I]
    MPI_Scatterv(Aroot, cnt.data(), disp.data(), MPI_DOUBLE, Aloc.data(), myrows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int li = 0; li < myrows; li++) {
        for (int j = 0; j < n; j++) {
            aug[(size_t)li * W + j] = Aloc[(size_t)li * n + j];
            aug[(size_t)li * W + n + j] = 0.0;
        }
        aug[(size_t)li * W + n + (mystart + li)] = 1.0;
    }
    const double t1 = MPI_Wtime();

    // (2) the n pivot steps
    struct { double val; int idx; } loc, glob;
    bool singular = false;
    for (int k = 0; k < n; k++) {
        // (a) pivot search: best |aug[i][k]| over my rows with global index >= k
        loc.val = -1.0;
        loc.idx = INT_MAX;
        for (int li = 0; li < myrows; li++) {
            int gi = mystart + li;
            if (gi < k) continue;
            double v = std::fabs(aug[(size_t)li * W + k]);
            if (v > loc.val) { loc.val = v; loc.idx = gi; }
        }
        // global pivot: largest |value|, lowest row index on a tie (as in the sequential code)
        MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (glob.val < 1e-12) { singular = true; break; }  // same decision on all ranks
        const int p_row = glob.idx;
        const int ok = D.owner(k), op = D.owner(p_row);

        // (b) row swap (rows k and p_row)
        if (p_row != k) {
            if (ok == op) {
                if (rank == ok) {
                    double* rk = &aug[(size_t)(k - mystart) * W];
                    double* rp = &aug[(size_t)(p_row - mystart) * W];
                    std::swap_ranges(rk, rk + W, rp);
                }
            } else if (rank == ok) {
                double* rk = &aug[(size_t)(k - mystart) * W];
                MPI_Sendrecv(rk, W, MPI_DOUBLE, op, 100, tmp.data(), W, MPI_DOUBLE, op, 100,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::memcpy(rk, tmp.data(), (size_t)W * sizeof(double));
            } else if (rank == op) {
                double* rp = &aug[(size_t)(p_row - mystart) * W];
                MPI_Sendrecv(rp, W, MPI_DOUBLE, ok, 100, tmp.data(), W, MPI_DOUBLE, ok, 100,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::memcpy(rp, tmp.data(), (size_t)W * sizeof(double));
            }
        }

        // (c) the owner normalises row k and broadcasts it (only columns k..W-1 are needed)
        if (rank == ok) {
            double* rk = &aug[(size_t)(k - mystart) * W];
            const double inv = 1.0 / rk[k];
            for (int j = k; j < W; j++) rk[j] *= inv;
            std::memcpy(&pivrow[k], &rk[k], (size_t)(W - k) * sizeof(double));
        }
        MPI_Bcast(&pivrow[k], W - k, MPI_DOUBLE, ok, MPI_COMM_WORLD);

        // (d) eliminate column k from my rows (same arithmetic as the sequential code)
        for (int li = 0; li < myrows; li++) {
            if (mystart + li == k) continue;
            const double f = aug[(size_t)li * W + k];
            if (f == 0.0) continue;
            double* row = &aug[(size_t)li * W];
            for (int j = k; j < W; j++) row[j] -= f * pivrow[j];
        }
    }
    const double t2 = MPI_Wtime();
    if (singular) return false;

    // (3) collect A^-1 (right half of the augmented rows) on rank 0
    for (int li = 0; li < myrows; li++)
        std::memcpy(&invloc[(size_t)li * n], &aug[(size_t)li * W + n], (size_t)n * sizeof(double));
    MPI_Gatherv(invloc.data(), myrows * n, MPI_DOUBLE, Ainv, cnt.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double t3 = MPI_Wtime();

    tm.scatter_ms = (t1 - t0) * 1000.0;
    tm.steps_ms = (t2 - t1) * 1000.0;
    tm.gather_ms = (t3 - t2) * 1000.0;
    tm.total_ms = (t3 - t0) * 1000.0;  // measured on rank 0, which is the last one to finish (it receives the result)
    return true;
}

// ---------------------------------------------------------------- verification on rank 0 (identical to seq_inverse.cpp)
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
    int n, p;
    double tp, scatter, steps, gather, ts, speedup, eff, resid;
};

// All ranks must call this together. `res` is only filled on rank 0. Returns false if singular.
static bool bench(int n, int reps, int rank, int p, const std::map<int, double>& ts, Result& res) {
    Dist D(n, p);
    std::vector<double> A, Ainv;
    if (rank == 0) {
        A.resize((size_t)n * n);
        Ainv.resize((size_t)n * n);
        generate(A, n, 42u);
    }
    std::vector<double> t_tot, t_sc, t_st, t_ga;
    double worst = 0.0;
    for (int r = 0; r < reps; r++) {
        Times tm{};
        if (!invert_mpi(rank == 0 ? A.data() : nullptr, rank == 0 ? Ainv.data() : nullptr, n, rank, p, D, tm)) {
            if (rank == 0) std::cerr << "Matrix is singular\n";
            return false;
        }
        if (rank == 0) {
            t_tot.push_back(tm.total_ms); t_sc.push_back(tm.scatter_ms);
            t_st.push_back(tm.steps_ms);  t_ga.push_back(tm.gather_ms);
            worst = std::max(worst, verify(A, Ainv, n));
        }
    }
    if (rank == 0) {
        res.n = n; res.p = p;
        res.tp = median(t_tot); res.scatter = median(t_sc); res.steps = median(t_st); res.gather = median(t_ga);
        auto it = ts.find(n);
        res.ts = (it == ts.end()) ? -1.0 : it->second;
        res.speedup = res.ts > 0 ? res.ts / res.tp : -1.0;
        res.eff = res.speedup > 0 ? res.speedup / p : -1.0;
        res.resid = worst;
        if (worst > 1e-8) std::cerr << "WARNING: residual too large for n=" << n << "\n";
    }
    return true;
}

static void print_header() {
    std::printf("%6s %6s %12s %10s %10s %10s %12s %9s %9s %10s\n", "n", "procs", "Tp (ms)", "scatter", "steps", "gather",
                "Ts (ms)", "speedup", "effic.", "residual");
}

static void print_row(const Result& r) {
    if (r.ts > 0)
        std::printf("%6d %6d %12.2f %10.2f %10.2f %10.2f %12.2f %9.2f %9.3f %10.2e\n", r.n, r.p, r.tp, r.scatter, r.steps,
                    r.gather, r.ts, r.speedup, r.eff, r.resid);
    else
        std::printf("%6d %6d %12.2f %10.2f %10.2f %10.2f %12s %9s %9s %10.2e\n", r.n, r.p, r.tp, r.scatter, r.steps,
                    r.gather, "n/a", "n/a", "n/a", r.resid);
    std::fflush(stdout);
}

static const char* CSV_HEADER = "n,procs,Tp_ms,scatter_ms,steps_ms,gather_ms,Ts_ms,speedup,efficiency,residual\n";

static void write_row(std::ofstream& csv, const Result& r) {
    csv << r.n << ',' << r.p << ',' << r.tp << ',' << r.scatter << ',' << r.steps << ',' << r.gather << ',' << r.ts << ','
        << r.speedup << ',' << r.eff << ',' << r.resid << '\n';
    csv.flush();
}

// ---------------------------------------------------------------- self test
static int selftest(int rank, int p) {
    int bad = 0;
    for (int kind = 0; kind < 2; kind++) {
        for (int n : {128, 250, 256, 512}) {  // 250: not divisible by most process counts
            if (n < p) continue;
            Dist D(n, p);
            std::vector<double> A, Ainv, ref;
            if (rank == 0) {
                A.resize((size_t)n * n);
                Ainv.resize((size_t)n * n);
                if (kind == 0) generate(A, n, 42u); else generate_general(A, n, 7u);
                invert_seq(A, ref, n);
            }
            Times tm{};
            bool ok = invert_mpi(rank == 0 ? A.data() : nullptr, rank == 0 ? Ainv.data() : nullptr, n, rank, p, D, tm);
            if (rank == 0) {
                double diff = 0.0;
                for (size_t i = 0; i < (size_t)n * n; i++) diff = std::max(diff, std::fabs(Ainv[i] - ref[i]));
                double res = verify(A, Ainv, n);
                bool pass = ok && diff < 1e-8 && res < 1e-6;
                if (!pass) bad++;
                std::printf("%-12s n=%4d procs=%2d  max|Ainv_mpi - Ainv_seq| = %.2e  residual = %.2e  %s\n",
                            kind == 0 ? "dominant" : "needs-pivot", n, p, diff, res, pass ? "PASS" : "FAIL");
            }
        }
    }
    if (rank == 0) {
        if (bad == 0) std::printf("\nALL TESTS PASSED\n");
        else std::printf("\n%d TESTS FAILED\n", bad);
    }
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, p = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    std::string mode = argc >= 2 ? argv[1] : "size";
    int reps = 5;
    if (argc >= 2 && std::strncmp(argv[argc - 1], "reps=", 5) == 0) {
        reps = std::max(1, std::atoi(argv[argc - 1] + 5));
        argc--;
    }
    int rc = 0;
    std::map<int, double> ts;
    if (rank == 0) {
        ts = load_seq_ts("results.csv");
        if (ts.empty())
            std::printf("NOTE: results.csv (sequential times) not found - run seq_inverse.exe first for speedup values.\n\n");
    }

    if (mode == "selftest") {
        rc = selftest(rank, p);
    } else if (mode == "one") {
        if (argc < 3) {
            if (rank == 0) std::cerr << "usage: mpiexec -n P mpi_inverse one <n> [reps]\n";
            rc = 1;
        } else {
            int n = std::atoi(argv[2]);
            if (argc >= 4) reps = std::max(1, std::atoi(argv[3]));
            Result r{};
            if (rank == 0) { std::printf("processes: %d, runs: %d (median)\n\n", p, reps); print_header(); }
            if (!bench(n, reps, rank, p, ts, r)) rc = 2;
            else if (rank == 0) print_row(r);
        }
    } else if (mode == "size" || mode == "procs") {
        std::ofstream csv;
        if (rank == 0) {
            std::printf("processes: %d, runs per size: %d (median)\n\n", p, reps);
            print_header();
            if (mode == "size") {
                csv.open("results_mpi_size.csv");
                csv << CSV_HEADER;
            } else {
                std::ifstream probe("results_mpi_procs.csv");
                bool exists = probe.good();
                probe.close();
                csv.open("results_mpi_procs.csv", std::ios::app);
                if (!exists) csv << CSV_HEADER;
            }
        }
        std::vector<int> sizes = (mode == "size") ? std::vector<int>{256, 512, 1024, 2048, 4096} : std::vector<int>{1024, 2048};
        for (int n : sizes) {
            Result r{};
            if (n < p) continue;
            if (!bench(n, reps, rank, p, ts, r)) { rc = 2; break; }
            if (rank == 0) { print_row(r); write_row(csv, r); }
        }
        if (rank == 0 && rc == 0) std::printf("\nWrote results_mpi_%s.csv (medians of %d runs)\n", mode.c_str(), reps);
    } else {
        if (rank == 0) std::cerr << "unknown mode '" << mode << "'. Modes: size, procs, one, selftest\n";
        rc = 1;
    }
    MPI_Finalize();
    return rc;
}
