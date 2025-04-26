#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h> // For ceil
#include <string.h>
#include <time.h>

// Required libraries
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>
#include <fftw3.h>
#include <omp.h>   // For OpenMP

// --- Configuration ---
#define DEFAULT_ALPHA 1.23
#define PREC_GUARD 64 // Additional bits for working precision

// --- Helper Function Declarations ---
void calculate_Ik(mpc_t result, unsigned long k, mpfr_prec_t prec);

// --- Main SSP Function ---
// Returns 0 on success, -1 on error
int calculate_pi_ssp(mpfr_t pi_result, unsigned long n_bits, double alpha) {
    mpfr_prec_t work_prec = n_bits + PREC_GUARD;
    mpfr_set_default_prec(work_prec);

    printf("--- Starting SSP Algorithm (Strict Interpretation Attempt) ---\n");
    printf("Target bits (n): %lu\n", n_bits);
    printf("Alpha (α): %.2f\n", alpha);
    printf("Working precision: %ld bits\n", work_prec);
    printf("NOTE: FFTW step uses double precision, limiting overall accuracy.\n");

    // --- Timing ---
    clock_t start_time, stage_start_time;
    double cpu_time_used;
    start_time = clock();

    // --- Initialization ---
    stage_start_time = clock();

    // 1. Determine N = ceil(alpha * n)
    unsigned long N = (unsigned long)ceil(alpha * (double)n_bits);
    if (N == 0) N = 1;
    printf("Series Order (N): %lu\n", N);

    // Allocate memory (size N+1 for indices 0 to N)
    mpc_t *f_vals_mpc = (mpc_t *)malloc(sizeof(mpc_t) * (N + 1));
    mpc_t *a_k_mpc = (mpc_t *)malloc(sizeof(mpc_t) * (N + 1));
    mpc_t *I_k_mpc = (mpc_t *)malloc(sizeof(mpc_t) * (N + 1));
    // FFTW arrays (double precision)
    fftw_complex *f_vals_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N + 1));
    fftw_complex *coeffs_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N + 1)); // FFT output

    if (!f_vals_mpc || !a_k_mpc || !I_k_mpc || !f_vals_fft || !coeffs_fft) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        if (f_vals_mpc) free(f_vals_mpc);
        if (a_k_mpc) free(a_k_mpc);
        if (I_k_mpc) free(I_k_mpc);
        if (f_vals_fft) fftw_free(f_vals_fft);
        if (coeffs_fft) fftw_free(coeffs_fft);
        return -1;
    }

    // Initialize MPC variables
    mpc_t mpc_pi, mpc_i, temp_mpc, mpc_j_val, mpc_N_val, angle;
    mpfr_t mpfr_pi, temp_r, temp_i;

    mpc_init2(mpc_pi, work_prec); mpc_init2(mpc_i, work_prec);
    mpc_init2(temp_mpc, work_prec); mpc_init2(mpc_j_val, work_prec);
    mpc_init2(mpc_N_val, work_prec); mpc_init2(angle, work_prec);
    mpfr_init2(mpfr_pi, work_prec); mpfr_init2(temp_r, work_prec); mpfr_init2(temp_i, work_prec);

    mpfr_const_pi(mpfr_pi, MPFR_RNDN);          // High-precision Pi from MPFR
    mpc_set_fr(mpc_pi, mpfr_pi, MPC_RNDNN);     // Convert Pi to MPC
    mpc_set_si_si(mpc_i, 0, 1, MPC_RNDNN);      // Complex i
    mpc_set_ui(mpc_N_val, N, MPC_RNDNN);        // N as MPC variable

    // Initialize arrays
    for (unsigned long k = 0; k <= N; ++k) {
        mpc_init2(f_vals_mpc[k], work_prec);
        mpc_init2(a_k_mpc[k], work_prec);
        mpc_init2(I_k_mpc[k], work_prec);
    }

    cpu_time_used = ((double)(clock() - stage_start_time)) / CLOCKS_PER_SEC;
    printf("[%.4f s] Initialization complete.\n", cpu_time_used);

    // --- Step 2: Calculate nodes x_j and function values f_j ---
    // As discussed, we evaluate the function part *without* the weight function,
    // as the weight is part of the integral definition for a_k, implicitly
    // handled by standard Chebyshev/DCT approaches.
    // f_j = exp(i * arccos(x_j)) where x_j = cos(pi*j/N)
    // This simplifies to f_j = exp(i * pi*j/N)
    stage_start_time = clock();
    printf("Step 2: Calculating f_j = exp(i * pi*j/N) for j=0..N...\n");

    #pragma omp parallel for private(mpc_j_val, angle, temp_mpc, temp_r, temp_i) schedule(dynamic)
    for (unsigned long j = 0; j <= N; ++j) {
        // Calculate theta_j = pi * j / N
        mpc_set_ui(mpc_j_val, j, MPC_RNDNN);
        mpc_mul(angle, mpc_pi, mpc_j_val, MPC_RNDNN); // angle = pi * j
        mpc_div(angle, angle, mpc_N_val, MPC_RNDNN);  // angle = pi * j / N

        // Calculate f_j = exp(i * angle)
        mpc_mul(temp_mpc, mpc_i, angle, MPC_RNDNN); // temp = i * theta_j
        mpc_exp(f_vals_mpc[j], temp_mpc, MPC_RNDNN); // f_j = exp(i * theta_j)

        // Convert to double complex for FFTW (PRECISION LOSS HAPPENS HERE)
        mpc_real(temp_r, f_vals_mpc[j], MPFR_RNDN);
        mpc_imag(temp_i, f_vals_mpc[j], MPFR_RNDN);
        f_vals_fft[j][0] = mpfr_get_d(temp_r, MPFR_RNDN); // Real part
        f_vals_fft[j][1] = mpfr_get_d(temp_i, MPFR_RNDN); // Imaginary part
    }
    cpu_time_used = ((double)(clock() - stage_start_time)) / CLOCKS_PER_SEC;
    printf("[%.4f s] Step 2 complete.\n", cpu_time_used);

    // --- Step 3: FFT Interpolation to find coefficients {a_k} ---
    // Using complex DFT (FFTW) on f_j as a proxy for interpolation.
    // The exact relation to the theoretical a_k requires careful analysis or
    // use of a high-precision complex Chebyshev transform.
    stage_start_time = clock();
    printf("Step 3: Calculating coefficients a_k via FFT (using FFTW double precision)...\n");

    fftw_plan plan_forward = fftw_plan_dft_1d(N + 1, f_vals_fft, coeffs_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan_forward) {
        fprintf(stderr, "Error: FFTW plan creation failed.\n");
        goto cleanup;
    }
    fftw_execute(plan_forward);
    fftw_destroy_plan(plan_forward);

    // Convert FFTW output back to MPC and apply scaling.
    // Scaling: Multiply by (1/N) or similar factor, adjust endpoints.
    // This scaling relates DFT output to series coefficients (approximate).
    // The factor 2/pi from the paper's a_k formula is handled implicitly by the integral target.
    mpc_t scale_factor_mpc;
    mpc_init2(scale_factor_mpc, work_prec);

    #pragma omp parallel for private(temp_mpc, scale_factor_mpc)
    for (unsigned long k = 0; k <= N; ++k) {
        // Convert double complex back to mpc_t
        mpc_set_d_d(a_k_mpc[k], coeffs_fft[k][0], coeffs_fft[k][1], MPC_RNDNN);

        // Apply scaling: Common scaling is 1/N, with endpoints divided by 2.
        // (This relates the transform output to the coefficients).
        mpc_set_ui(scale_factor_mpc, N, MPC_RNDNN); // Denominator N

        if (k == 0 || k == N) {
             mpc_mul_ui(scale_factor_mpc, scale_factor_mpc, 2, MPC_RNDNN); // Denominator 2N for endpoints
        }

        if (N > 0) { // Avoid division by zero if N=0 (edge case)
             mpc_div(a_k_mpc[k], a_k_mpc[k], scale_factor_mpc, MPC_RNDNN);
        }
    }
    mpc_clear(scale_factor_mpc);

    cpu_time_used = ((double)(clock() - stage_start_time)) / CLOCKS_PER_SEC;
    printf("[%.4f s] Step 3 complete.\n", cpu_time_used);

    // --- Step 4: Inner Product Calculation ---
    // π_est = Im( Sum_{k=0..N} a_k * I_k )
    stage_start_time = clock();
    printf("Step 4: Calculating Inner Product Sum(a_k * I_k)...\n");

    // Calculate I_k = integral_{-1..1} T_k(x) dx
    printf("   Calculating I_k values (k=0..N)...\n");
    #pragma omp parallel for
    for (unsigned long k = 0; k <= N; ++k) {
         // Optimization: Only non-zero for even k
         if (k % 2 == 0) {
            calculate_Ik(I_k_mpc[k], k, work_prec); // I_k is real, stored in MPC
         } else {
            mpc_set_ui(I_k_mpc[k], 0, MPC_RNDNN); // Set odd k to zero explicitly
         }
    }

    // Calculate the sum: Sum_{k=0..N} a_k * I_k
    // Parallel summation of MPC is complex, do serially for correctness.
    mpc_t total_sum, term;
    mpc_init2(total_sum, work_prec);
    mpc_init2(term, work_prec);
    mpc_set_ui(total_sum, 0, MPC_RNDNN); // Initialize sum to zero

    printf("   Summing a_k * I_k terms...\n");
    for (unsigned long k = 0; k <= N; ++k) {
        // I_k is zero for odd k, so term is zero. Can skip, but explicit check is fine.
        if (mpc_is_zero(I_k_mpc[k])) {
             continue;
        }
        mpc_mul(term, a_k_mpc[k], I_k_mpc[k], MPC_RNDNN); // term = a_k * I_k
        mpc_add(total_sum, total_sum, term, MPC_RNDNN);  // total_sum += term
    }

    // Extract the imaginary part as the estimate for Pi
    // Pi = Im( Sum a_k * I_k ) according to paper's formula derivation path
    mpc_imag(pi_result, total_sum, MPFR_RNDN);

    mpc_clear(total_sum); mpc_clear(term);

    cpu_time_used = ((double)(clock() - stage_start_time)) / CLOCKS_PER_SEC;
    printf("[%.4f s] Step 4 complete.\n", cpu_time_used);

    // --- Step 5: Rounding (handled by MPFR output format) ---
    printf("Step 5: Rounding to target precision (implicit in output).\n");


    // --- Cleanup ---
cleanup:
    printf("Cleaning up resources...\n");
    mpc_clear(mpc_pi); mpc_clear(mpc_i); mpc_clear(temp_mpc);
    mpc_clear(mpc_j_val); mpc_clear(mpc_N_val); mpc_clear(angle);
    mpfr_clear(mpfr_pi); mpfr_clear(temp_r); mpfr_clear(temp_i);

    for (unsigned long k = 0; k <= N; ++k) {
        mpc_clear(f_vals_mpc[k]);
        mpc_clear(a_k_mpc[k]);
        mpc_clear(I_k_mpc[k]);
    }
    free(f_vals_mpc);
    free(a_k_mpc);
    free(I_k_mpc);
    fftw_free(f_vals_fft);
    fftw_free(coeffs_fft);

    cpu_time_used = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
    printf("--- SSP Algorithm Finished (Total Time: %.4f s) ---\n", cpu_time_used);
    return 0; // Success
}

// --- Helper Function Implementation ---

// Calculate I_k = integral_{-1..1} T_k(x) dx
// I_k = 2/(1-k^2) for k even >= 0
// I_k = 0         for k odd
// Note: For k=0, T_0(x)=1, Integral is 2. Formula 2/(1-0^2) = 2 works.
// Note: For k=1, T_1(x)=x, Integral is 0. Formula gives singularity, but k is odd.
// Note: For k=2, T_2(x)=2x^2-1, Integral is 2*(x^3/3) - x |_(-1)^1 = (2/3 - 1) - (-2/3 + 1) = -1/3 - 1/3 = -2/3.
//       Formula gives 2/(1-2^2) = 2/(1-4) = 2/-3 = -2/3. Matches.
void calculate_Ik(mpc_t result, unsigned long k, mpfr_prec_t prec) {
    // Handles odd k implicitly by check below, or explicitly set zero first
    mpc_set_ui(result, 0, MPC_RNDNN);

    if (k % 2 == 0) { // Only even k contribute
        mpfr_t num, den, k_sq, one, temp_r;
        mpfr_init2(num, prec); mpfr_init2(den, prec);
        mpfr_init2(k_sq, prec); mpfr_init2(one, prec);
        mpfr_init2(temp_r, prec);

        mpfr_set_ui(num, 2, MPFR_RNDN);         // Numerator = 2
        mpfr_set_ui(k_sq, k, MPFR_RNDN);        // k_sq = k (as unsigned long)
        mpfr_mul_ui(k_sq, k_sq, k, MPFR_RNDN);  // k_sq = k*k
        mpfr_set_ui(one, 1, MPFR_RNDN);         // one = 1
        mpfr_sub(den, one, k_sq, MPFR_RNDN);    // den = 1 - k^2 (negative or zero if k=1)

        // Check for k=1 case (although it's odd, robustness) - Denominator zero
        // This case is skipped by the k%2==0 check anyway.

        mpfr_div(temp_r, num, den, MPFR_RNDN); // result = 2 / (1 - k^2)
        mpc_set_fr(result, temp_r, MPC_RNDNN); // Store real result in mpc_t

        mpfr_clear(num); mpfr_clear(den); mpfr_clear(k_sq); mpfr_clear(one); mpfr_clear(temp_r);
    }
    // If k is odd, result remains 0
}


// --- Main Execution ---
int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <num_bits> [alpha]\n", argv[0]);
        fprintf(stderr, "  num_bits: Target number of bits for Pi\n");
        fprintf(stderr, "  alpha:    Optional constant for N = ceil(alpha * n) (default: %.2f)\n", DEFAULT_ALPHA);
        return 1;
    }

    unsigned long n_bits = strtoul(argv[1], NULL, 10);
    double alpha = DEFAULT_ALPHA;
    if (argc == 3) {
        alpha = atof(argv[2]);
        if (alpha <= 1.0) {
             fprintf(stderr, "Warning: Alpha should generally be > 1 for accuracy. Using provided value: %.2f\n", alpha);
        }
    }

    if (n_bits == 0) {
        fprintf(stderr, "Error: Number of bits must be positive.\n");
        return 1;
    }
     // Basic check: FFTW double precision limits meaningful bits
    if (n_bits > 53) {
        fprintf(stderr, "Warning: Target bits (%lu) exceed double precision (~53 bits) used by FFTW.\n", n_bits);
        fprintf(stderr, "         The result accuracy will be limited by the FFT step.\n");
    }


    // Initialize MPFR Pi result variable with target precision
    mpfr_t pi_ssp;
    mpfr_init2(pi_ssp, n_bits);

    // --- Run SSP ---
    int status = calculate_pi_ssp(pi_ssp, n_bits, alpha);

    if (status == 0) {
        // --- Output Result ---
        printf("\n--- SSP Calculation Result ---\n");
        printf("Pi (SSP, %lu bits target, FFTW limited) = ", n_bits);
        mpfr_out_str(stdout, 10, 0, pi_ssp, MPFR_RNDN); // Base 10, enough digits for precision
        printf("\n");

        // Compare with MPFR's internal Pi at the target precision
        mpfr_t mpfr_pi_ref;
        mpfr_init2(mpfr_pi_ref, n_bits);
        mpfr_const_pi(mpfr_pi_ref, MPFR_RNDN);
        printf("Pi (MPFR ref, %lu bits)                = ", n_bits);
        mpfr_out_str(stdout, 10, 0, mpfr_pi_ref, MPFR_RNDN);
        printf("\n");

        // Calculate and print difference (absolute error)
        mpfr_t diff;
        mpfr_init2(diff, n_bits);
        mpfr_sub(diff, pi_ssp, mpfr_pi_ref, MPFR_RNDN);
        mpfr_abs(diff, diff, MPFR_RNDN);
        printf("Absolute Difference                       : ");
        // Use scientific notation for small differences
        mpfr_printf("%.5Re\n", diff); // Print difference with 5 significant digits in scientific notation

        mpfr_clear(mpfr_pi_ref);
        mpfr_clear(diff);

    } else {
        fprintf(stderr, "\nSSP calculation failed.\n");
    }

    // Final cleanup
    mpfr_clear(pi_ssp);
    mpfr_free_cache(); // Free global caches used by MPFR

    return status;
}