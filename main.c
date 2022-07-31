#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

#include "hll.h"
#include "hll_empirical.h"

/*
 * measure_dense_hll_estimation_error linearly divides the range
 * [0, max_card] by this number of points.
 * Increasing this value can critically increase the execution time.
 */
const size_t N_POINTS = 20;

/*
 * Number of randomly generated sets for every cardinality
 * in measure_dense_hll_estimation_error.
 * Increasing this value can critically increase the execution time.
 */
const size_t SETS_PER_POINT = 20;

/*
 * File to dump the data that is used to measure the errors.
 */
char *OUTPUT_FILE_NAME = NULL;


double
average_sum_of(double *arr, size_t n)
{
	double sum = 0;
	for (size_t i = 0; i < n; ++i)
		sum += arr[i];
	return sum / n;
}

double
max_of(double *arr, size_t n)
{
	double max = arr[0];
	for (size_t i = 0; i < n; ++i)
		if (max < arr[i])
			max = arr[i];
	return max;
}

double
dispersion_of(double *arr, double val, size_t n)
{
	assert(n >= 1);
	double sqr_sum = 0;
	for (size_t i = 0; i < n; ++i) {
		sqr_sum += (val - arr[i]) * (val - arr[i]);
	}
	return sqrt(sqr_sum / (n - 1));
}

struct est_errors {
	/* Expected error. */
	double exp_err;
	/* Average standart error. */
	double std_err;
	/* Average max error. */
	double max_err;
};

uint64_t
phash(uint64_t val)
{
	unsigned char *bytes = (void *)&val;
	uint64_t p = 29791;
	uint64_t pi = p;
	uint64_t hash = 0;
	for (int i = 0; i < 8; ++i, pi *= p) {
		hash += pi * bytes[i];
	}
	return hash;
}

uint64_t
hash(uint64_t val)
{
	return phash(phash(val));
}

uint64_t big_rand()
{
	/*
	 * C standard rand() genarates random numbers in the range form 0 to
	 * RAND_MAX. RAND_MAX is at least 32767. This function helps to reduce
	 * the number of repeats if RAND_MAX is small.
	 */
#if RAND_MAX < (1ULL << 16)
	uint64_t r1 = rand();
	uint64_t r2 = rand();
	uint64_t r3 = rand();
	uint64_t r4 = rand();
	return r1 * r2 * r3 * r4;
#elif RAND_MAX < (1ULL << 32)
	uint64_t r1 = rand();
	uint64_t r2 = rand();
	uint64_t r3 = rand();
	return r1 * r2 * r3;
#else
	uint64_t r1 = rand();
	uint64_t r2 = rand();
	return r1 * r2;
#endif
}

#define MAYBE_PRINT(file, ...)			\
do {						\
	if (file) {				\
		fprintf(file, __VA_ARGS__);	\
	}					\
} while (0)

/*
 * The error measure occurs as follows:
 * The range [0, max_card] is linearly devided by n_points.
 * For every cardinality from the range sets_per_point HyperLogLog are created
 * and estimates the cardinality of randomly generated set of this cardinality.
 * Using the estimations, the error for this cardinality is calculated.
 * The error and other useful data can be dumped in the output file.
 * The resulting error is the average error of all cardinalities.
 */
void
measure_dense_hll_estimation_error(
	int prec, size_t max_card, size_t n_points, size_t sets_per_point,
	struct est_errors *res, FILE *output)
{
	double max_err_sum = 0;
	double std_err_sum = 0;
	const double card_step = 1.f * max_card / n_points;
	for (size_t n = 0; n < n_points; ++n) {

		size_t card = card_step * n;
		double error[sets_per_point];
		double est[sets_per_point];

		for (size_t i = 0; i < sets_per_point; ++i) {

			struct hll *hll = hll_create(prec, HLL_DENSE);

			for (size_t j = 0; j < card; ++j) {
				uint64_t val = big_rand();
				hll_add(hll, hash(val));
			}
			double this_est = hll_estimate(hll);
			double this_err = abs(this_est - card);
			est[i] = this_est;
			error[i] = this_err;

			hll_destroy(hll);
		}

		double max_err = max_of(error, sets_per_point) / (card + 1);
		max_err_sum += max_err;

		double avg_est = average_sum_of(est, sets_per_point);
		double std_err = dispersion_of(est, card, sets_per_point) /
				(card + 1);
		std_err_sum += std_err;

		MAYBE_PRINT(output,
			"%2d, %12zu, %12.2f, %12lg, %12lg\n",
			prec, card, avg_est, std_err, max_err);
	}

	double avg_std_err = std_err_sum / n_points;
	double avg_max_err = max_err_sum / n_points;

	res->std_err = avg_std_err;
	res->max_err = avg_max_err;
	res->exp_err = 1.04f / sqrt(1u << prec);
}

#define MAX(l, r)  ((l) < (r) ? (r) : (l))

/*
 * This test can dump the data that is used to measure
 * the estimation error. These data can be used for further
 * analysis and empirical based impovements of the algorithm.
 */
void test_dense_hyperloglog_error()
{
	struct est_errors errors[HLL_MAX_PRECISION + 1];

	FILE *output = NULL;
	if (OUTPUT_FILE_NAME != NULL) {
		output = fopen(OUTPUT_FILE_NAME, "w");
		assert(output && "Can't open the output file.");
	}

	MAYBE_PRINT(output,
		"prec,       card,      avg_est,      std_err,      max_err\n");

	for (int prec = HLL_MIN_PRECISION;
	     prec <= HLL_MAX_PRECISION; ++prec) {
		size_t n_regs = 1u << prec;
		size_t max_card = 10 * n_regs;
		measure_dense_hll_estimation_error(prec, max_card,
			N_POINTS, SETS_PER_POINT, errors + prec, output);
	}

	if (output)
		fflush(output);

	for (int prec = HLL_MIN_PRECISION;
	     prec <= HLL_MAX_PRECISION; ++prec) {
		MAYBE_PRINT(output,
			"prec:%d, std_err:%lg, max_err:%lg, exp_err: %lg\n",
			prec, errors[prec].std_err, errors[prec].max_err,
			errors[prec].exp_err);
	}

	for (int prec = HLL_MIN_PRECISION;
	     prec <= HLL_MAX_PRECISION; ++prec) {
		/*
		 * The error of HyperLogLog is close to 1/sqrt(n_counters),
		 * but for small cardinalities LinearCounting is used because
		 * it has better accuracy, so the resulting error must be
		 * smaller than the HyperLogLog theoretical error.
		 */
		assert(errors[prec].std_err < errors[prec].exp_err);
	}
}

void test_sparse_hll_error()
{
	/*
	 * Make sure that the sparse prepresentation provides the best accuracy.
	 */
	double max_error = 1.04 / (1 << (HLL_MAX_PRECISION / 2) + 1);
	for (int prec = HLL_MIN_PRECISION;
	     prec <= HLL_MAX_PRECISION; ++prec) {
		struct hll *hll = hll_create(prec, HLL_SPARSE);

		double card_step = (1 << prec) / 64;
		size_t card = 0;
		while (1) {
			if (hll->representation != HLL_SPARSE)
				break;

			uint64_t est = hll_estimate(hll);
			assert(abs(est - card) <= max_error * card);

			for (size_t i = 0; i < card_step; ++i) {
				uint64_t val = big_rand();
				uint64_t h = hash(val);
				hll_add(hll, h);
			}
			card += card_step;
		}

		hll_destroy(hll);
	}
}

void test_sparse_to_dense_convertion()
{
	for (int prec = HLL_MIN_PRECISION;
	     prec <= HLL_MAX_PRECISION; ++prec) {
		struct hll *sparse_hll = hll_create(prec, HLL_SPARSE);
		struct hll *dense_hll = hll_create(prec, HLL_DENSE);

		while (sparse_hll->representation == HLL_SPARSE) {
			uint64_t val = big_rand();
			uint64_t h = hash(val);
			/* Double add must not affect the estimation. */
			hll_add(sparse_hll, h);
			hll_add(sparse_hll, h);
			hll_add(dense_hll, h);
		}

		uint64_t sparse_est = hll_estimate(sparse_hll);
		uint64_t dense_est = hll_estimate(dense_hll);

		assert(sparse_est == dense_est);

		hll_destroy(sparse_hll);
		hll_destroy(dense_hll);
	}
}

int
main(int argc, char *argv[])
{
	test_dense_hyperloglog_error();
	test_sparse_hll_error();
	test_sparse_to_dense_convertion();

        return 0;
}
