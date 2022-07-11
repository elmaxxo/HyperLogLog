#include "hll.h"
#include "hll_empirical.h"

#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

/*
 * The HyperLogLog algrithm.
 * The algorithm allows to estimate cardinalities of multisets using almost no
 * memory overhead compared to other cardinality estimation algorithms.
 * This achieved by storing only the array 2^precision registers.
 * Each register stores properies of the added alements hashes.
 * The register index is the last precision bits of the element hash.
 * The register value is the number of leading zeros in the hash of the element
 * plus one. The hash function for the algorithm must equally likely to give
 * values from 0 to 2^64. The idea is that big number of leading zeros are
 * achieved by large sets. The cardinality can be estimated by using registers
 * values.
 * The estimation formula:
 * ESTIMATION := N_REGISTERS^2 * ALPHA / HARMONIC_SUM,
 * where N_REGISTERS is the number of registers and it equal to 2^precision,
 * APLHA is normalized constant for the estimation,
 * HARMONIC_SUM is the sum of 2^(-register(i)) where register(i) is the value
 * of register with idx i, i takes the values of all indexes.
 */

enum HLL_CONSTANTS {
	/*
	 * 6 bits are needed to store the number of
	 * leading zeros of 64 bit hash.
	 */
	HLL_RANK_BITS = 6,

	/* The maximum value that can be stored in HLL_RANK_BITS bits. */
	HLL_RANK_MAX = (1 << HLL_RANK_BITS) - 1,

	/* Number of bits in registers bucket. See struct reg_bucket. */
	HLL_BUCKET_BITS = 24,

	/* Precision of sparsely represented HyperLogLog. */
	HLL_SPARSE_PRECISION = 26,

	/*
	 * Grow coefficient for the sparse representation. 2 allows to grow
	 * to exact limit of the sparse representation
	 * (equals to the memory usage of the dense representation).
	 */
	HLL_SPARSE_GROW_COEF = 2,

	/*
	 * The smallest precision of HyperLogLog that starts with the sparse
	 * representation. Sparse representation for smaller precision can't
	 * store more than 100 pairs so there is no need in this representation.
	 */
	HLL_SPARSE_MIN_PRECISION = 10,

	/*
	 * Inital size of sparsely represented HyperLogLog.
	 * The initial size must contain at least a header and differ by a
	 * power of 2 from the sizes for the dense representation.
	 */
	HLL_SPARSE_INITIAL_SIZE = 48,
};

/* Check if the precision is correct. */
static int
hll_is_valid_precision(uint8_t prec)
{
	return (prec >= HLL_MIN_PRECISION && prec <= HLL_MAX_PRECISION) ||
	       (prec == HLL_SPARSE_PRECISION);
}

/* Get a number whose first n bits are equal to ones. */
static uint64_t
hll_ones(uint8_t n)
{
	assert(n < 64);
	return ((UINT64_C(1) << n) - 1);
}

/* Check whether the cached value stores the valid value of the estimation. */
static int
hll_is_valid_cache(const struct hll *hll)
{
	return hll->cached_estimation >= 0;
}

/* Mark the cached value as invalid. */
static void
hll_invalidate_cache(struct hll *hll)
{
	hll->cached_estimation = -1.f;
}

/*
 * The highest precision bits of the hash are interpreted as a register index.
 */
static uint32_t
hll_hash_register_idx(uint64_t hash, uint8_t precision)
{
	assert(hll_is_valid_precision(precision));
	return hash >> (64 - precision);
}

/*
 * Return the number of leading zeros of the first
 * (64 - precision) hash bits plus one.
 */
static uint8_t
hll_hash_rank(uint64_t hash, uint8_t precision)
{
	assert(hll_is_valid_precision(precision));
	hash |= hll_ones(precision) << (64 - precision);
	uint8_t zero_count = 0;
	uint64_t bit = 0x1;
	while ((hash & bit) == 0) {
		++zero_count;
		bit <<= 1;
	}
	uint8_t rank = zero_count + 1;
	assert(rank <= HLL_RANK_MAX);
	return rank;
}

/* Calculate the number of registers for this presision. */
static uint64_t
hll_n_registers(uint8_t precision)
{
	assert(precision < 64);
	return UINT64_C(1) << precision;
}

/* Alpha constant that HyperLogLog uses in the estimation formula. */
static double
hll_alpha(uint8_t precision)
{
	return 0.7213 / (1.0 + 1.079 / hll_n_registers(precision));
}

/* Estimate the cardinality using the LinearCounting algorithm. */
static double
linear_counting(size_t counters, size_t empty_counters)
{
	return counters * log((double)counters / empty_counters);
}

/*
 * ================================================================
 * Implementation of the dense representation.
 * Dense representation is a classical representation: there is alsways
 * allocated 2^precision number of counters so it may be wasteful for
 * small caridnalities. The dense representation should be used for
 * estimating large caridnalities.
 * ================================================================
 */

/*
 * Calculate the amount of memory reqired to store
 * the registers for dense representation.
 */
static size_t
hll_dense_reqired_memory(uint8_t precision)
{
	size_t n_registers = hll_n_registers(precision);
	return n_registers * HLL_RANK_BITS / CHAR_BIT;
}

/* Init a densely represented HyperLogLog estimator. */
struct hll *
hll_dense_init(struct hll *hll, uint8_t precision)
{
	hll->representation = HLL_DENSE;
	hll->precision = precision;
	hll->cached_estimation = 0;
	/* For the dense representation data interpreted as registers. */
	const size_t registers_size = hll_dense_reqired_memory(precision);
	hll->data = calloc(registers_size, 1);
	return hll;
}

/*
 * Dense register is represented by 6 bits so it can go out the range of one
 * byte but 4 registers occupy exactly 3 bytes so I called it a bucket.
 * The registers array can always be separated to such buckets because its size
 * in bits is devided by 24 if precision more than 2
 * (2^2 * 6 bits = 24 bits, other sizes differ by power of 2 times).
 *
 * ASCII-visualization of a bucket:
 * +----------+----------+----------+----------+
 * |0 regsiter|1 register|2 register|3 register|
 * +----------+----------+----------+----------+
 * |<---------6 bits * 4 = 24 bits------------>|
 */
struct reg_bucket {
	/* Pointer to the 3 byte bucket where the register is stored. */
	uint8_t *addr;
	/* Offset of the register in the bucket. */
	size_t offset;
};

/*
 * Init a register bucket structure that is used for
 * convenient work with registers.
 */
static void
reg_bucket_init(struct reg_bucket *bucket, uint8_t *regs, size_t reg_idx)
{
	/*
	 * ASCII-visualization of the logic of the function:
	 *
	 * regs		  1 byte	 2 byte	       3 byte	      4 byte
	 * |		  |		 |	       |	      |
	 * +----------+----------+----------+----------+----------+----------+--
	 * |0 regsiter|1 register|2 register|3 register|4 register|5 register|..
	 * +----------+----------+----------+----------+----------+----------+--
	 * |	      6		 12	    18	       |          30	     32
	 * 0 bucket				       1 bucket
	 *
	 * For instance, the 5th register is stored in (5*6 / 24 = 1) the first
	 * bucket and its offset is equal to 5*6 % 24 = 6.
	 */
	size_t bucket_size = HLL_BUCKET_BITS / CHAR_BIT;
	size_t bucket_idx = reg_idx * HLL_RANK_BITS / HLL_BUCKET_BITS;
	bucket->addr = (uint8_t *)(regs + bucket_idx * bucket_size);
	bucket->offset = reg_idx * HLL_RANK_BITS % HLL_BUCKET_BITS;
	assert(bucket->offset <= HLL_BUCKET_BITS - HLL_RANK_BITS);
}

/* Get an integer value of 3 bytes stored in the bucket. */
static uint32_t
reg_bucket_value(const struct reg_bucket *bucket)
{
	uint8_t *addr = bucket->addr;
	return addr[0] | (addr[1] << CHAR_BIT) | (addr[2] << 2 * CHAR_BIT);
}

/*
 * Get a mask that clears the register stored in the bucket and
 * saves the other boundary registers in the bucket.
 */
static uint32_t
reg_bucket_boundary_mask(const struct reg_bucket *bucket)
{
	/*
	 * |000000000000000000111111|
	 * |------------regstr------|
	 */
	uint32_t ones = hll_ones(HLL_RANK_BITS);
	/*
	 * |000000000000111111000000|
	 * |------------regstr------|
	 */
	uint32_t register_mask = ones << bucket->offset;
	/*
	 * |111111111111000000111111|
	 * |------------regstr------|
	 */
	uint32_t boundary_mask = ~register_mask;
	return boundary_mask;
}

/* Get the value of the register with the idx index. */
static uint8_t
hll_dense_register_rank(const struct hll *hll, size_t idx)
{
	struct reg_bucket bucket;
	reg_bucket_init(&bucket, hll->data, idx);
	uint32_t reg_mask = hll_ones(HLL_RANK_BITS);
	uint32_t bucket_value = reg_bucket_value(&bucket);
	uint8_t rank = (bucket_value >> bucket.offset) & reg_mask;
	assert(rank <= HLL_RANK_MAX);
	return rank;
}

/* Set a new rank for the register with the idx index. */
static void
hll_dense_set_register_rank(struct hll *hll, size_t idx, uint8_t rank)
{
	struct reg_bucket bucket;
	reg_bucket_init(&bucket, hll->data, idx);
	uint32_t boundary_mask = reg_bucket_boundary_mask(&bucket);
	uint32_t bucket_value = reg_bucket_value(&bucket);
	union {
		uint32_t value;
		uint8_t bytes[3];
	} new_bucket;
	new_bucket.value = (rank << bucket.offset) |
			   (bucket_value & boundary_mask);

	bucket.addr[0] = new_bucket.bytes[0];
	bucket.addr[1] = new_bucket.bytes[1];
	bucket.addr[2] = new_bucket.bytes[2];
}

/* Add hash to the densely represented HyperLogLog estimator. */
static void
hll_dense_add(struct hll *hll, uint64_t hash)
{
	uint8_t precision = hll->precision;
	size_t idx = hll_hash_register_idx(hash, precision);
	assert(idx < hll_n_registers(precision));
	uint8_t hash_rank = hll_hash_rank(hash, precision);
	uint8_t reg_rank = hll_dense_register_rank(hll, idx);
	if (reg_rank < hash_rank) {
		hll_dense_set_register_rank(hll, idx, hash_rank);
		hll_invalidate_cache(hll);
	}
}

/*
 * Estimate the cardinality of the densely represented HyperLogLog using the
 * estimation formula. Raw estimation can have larger relative error
 * for small cardinalities.
 */
static double
hll_dense_raw_estimate(const struct hll *hll)
{
	double sum = 0;
	const size_t n_registers = hll_n_registers(hll->precision);
	for (size_t i = 0; i < n_registers; ++i) {
		sum += pow(2, -hll_dense_register_rank(hll, i));
	}

	const double alpha = hll_alpha(hll->precision);
	return alpha * n_registers * n_registers / sum;
}

/* Count the number of registers that are zero. */
static size_t
hll_dense_count_zero_registers(const struct hll *hll)
{
	size_t count = 0;
	const size_t n_registers = hll_n_registers(hll->precision);
	for (size_t i = 0; i < n_registers; ++i) {
		if (hll_dense_register_rank(hll, i) == 0)
			++count;
	}
	return count;
}

/* Estimate the caridnality of the densely represented HyperLogLog */
static uint64_t
hll_dense_estimate(struct hll *hll)
{
	if (hll_is_valid_cache(hll)) {
		return hll->cached_estimation;
	}
	const uint8_t prec = hll->precision;
	const size_t n_registers = hll_n_registers(prec);
	double raw_estimation = hll_dense_raw_estimate(hll);

	double hll_estimation = raw_estimation;
	if (raw_estimation < 4.f * n_registers) {
		hll_estimation -=
			hll_empirical_bias_correction(prec, raw_estimation);
	}

	size_t zero_count = hll_dense_count_zero_registers(hll);
	double lc_estimation = zero_count != 0 ?
		linear_counting(n_registers, zero_count) :
		hll_estimation;

	uint64_t threshold = hll_empirical_estimation_threshold(prec);
	size_t estimation = lc_estimation < threshold ? lc_estimation :
							hll_estimation;
	hll->cached_estimation = estimation;
	return estimation;
}

/*
 * =====================================================================
 * Implementation of the sparse representation.
 * The sparse representation allocates only pairs of index and the highest rank
 * that has added for the index. It requires less memory for small cardinalities
 * than dense representation and provides better accuracy. The sparse
 * representation swithces to the dense representation if it statrs to require
 * more amount of memory that is needed for the dense representation.
 * =====================================================================
 */

/*
 * Instead of registers the sparse representation stores pairs of
 * register index and its the highest rank.
 * The pairs for the sparse representation have the following structure:
 * +----------+--------------------------------+
 * |   rank   |		   index	       |
 * +----------+--------------------------------+
 * |<-6 bits->|<-----------26 bits------------>|
 */

typedef uint32_t pair_t;

/* Make a pair with specified index and rank. */
static pair_t
hll_sparse_new_pair(size_t idx, uint8_t rank)
{
	return rank | (idx << HLL_RANK_BITS);
}

/* Get the index of the pair. */
static uint32_t
hll_sparse_pair_idx(pair_t pair)
{
	return pair >> HLL_RANK_BITS;
}

/* Get the rank of the pair. */
static uint8_t
hll_sparse_pair_rank(pair_t pair)
{
	return pair & hll_ones(HLL_RANK_BITS);
}

static uint32_t
hll_sparse_pair_dense_idx(pair_t pair, uint8_t precision)
{
	uint32_t idx = hll_sparse_pair_idx(pair);
	/*
	 * Since the sparse precision is more than any dense precisions
	 * the hash can alway be restored by discarding the extra bits.
	 * |101110101010010010010011...1011| : hash
	 * |<-------idx(26)------->|
	 * |101110101010010010010011...1011| : hash
	 * |<---idx(prec)--->|
	 */
	return idx >> (HLL_SPARSE_PRECISION - precision);
}

static uint8_t
hll_sparse_pair_dense_rank(pair_t pair)
{
	/*
	 * I make an assumption that rank for both representations is the same.
	 * But in fact, the rank of the hash with sparse precision may differ by
	 * rank with dense representation but the probability of this is
	 * less than * 0.5^(64 - 26) ~ 3.6e-12 (38 leading zeros)
	 * so such assumption will not make a big mistake.
	 */
	return hll_sparse_pair_rank(pair);
}

/*
 * Header for the sparsely represented HyperLogLog. The whole HyperLogLog
 * is stored in the data field of the hll struct.
 * The first bytes are used for the header.
 * The list of added pairs starts after the header.
 * +------------+-----------------  --+
 * |   HEADER   |  PAIRS LIST    ...  |
 * +------------+-----------------  --+
 * |<-------------size--------------->|
 */
struct hll_sparse_header {
	/* Number of pairs stored in the estimator. */
	uint32_t pairs_count;
	/*
	 * Amount of memory that is used to store the HyperLogLog
	 * (incluning the header).
	 */
	uint32_t size;
	/*
	 * Precision declared when creating HyperLogLog.
	 * The precision is used for switching to the dense representation.
	 */
	uint32_t dense_precision;
};

/* Get the header of sparsely represented HyperLogLog. */
static struct hll_sparse_header *
hll_sparse_header(const uint8_t *sparse_hll)
{
	return (struct hll_sparse_header *)sparse_hll;
}

/*
 * Get the size of sparsely represented HyperLogLog.
 */
static uint32_t
hll_sparse_size(const uint8_t *sparse_hll)
{
	struct hll_sparse_header *header = hll_sparse_header(sparse_hll);
	return header->size;
}

/* Get the maximum number of the HyeprLogLog pairs. */
static uint32_t
hll_sparse_pairs_max_count(const uint8_t *sparse_hll)
{
	uint32_t hll_size = hll_sparse_size(sparse_hll);
	uint32_t header_size = sizeof(struct hll_sparse_header);
	uint32_t pairs_size = hll_size - header_size;
	uint32_t max_count = pairs_size / sizeof(pair_t);
	return max_count;
}

/* Get the number of pairs stored in the HyperLogLog list. */
static uint32_t
hll_sparse_pairs_count(const uint8_t *sparse_hll)
{
	struct hll_sparse_header *header = hll_sparse_header(sparse_hll);
	return header->pairs_count;
}

/* Get the precision declared when creating HyperLogLog. */
static uint32_t
hll_sprase_dense_precision(const uint8_t *sparse_hll)
{
	struct hll_sparse_header *header = hll_sparse_header(sparse_hll);
	return header->dense_precision;
}

/* Get the maximum amount of memory that sparse representation can use. */
static uint32_t
hll_sparse_max_size(uint32_t precision)
{
	return hll_dense_reqired_memory(precision);
}

/* Init a sparsely represented HyperLogLog estimator. */
static void
hll_sparse_init(struct hll *hll, uint8_t precision)
{
	assert(precision >= HLL_SPARSE_MIN_PRECISION);
	hll->representation = HLL_SPARSE;
	hll->precision = precision;
	/* For the sparse representation data interpreted as a list of pairs. */
	hll->data = calloc(HLL_SPARSE_INITIAL_SIZE, 1);
	struct hll_sparse_header *header = hll_sparse_header(hll->data);
	header->size = HLL_SPARSE_INITIAL_SIZE;
	header->pairs_count = 0;
	header->dense_precision = precision;
}

/* Get the pairs stored in HyperLogLog. */
static pair_t *
hll_sparse_pairs(const uint8_t *sparse_hll)
{
	return (pair_t *)(sparse_hll + sizeof(struct hll_sparse_header));
}

/* The HyperLogLog is full if it can't add a new pair without growing. */
static int
hll_sparse_is_full(const uint8_t *sparse_hll)
{
	uint32_t count = hll_sparse_pairs_count(sparse_hll);
	uint32_t max_count = hll_sparse_pairs_max_count(sparse_hll);
	assert(count <= max_count);
	return count == max_count;
}

/*
 * Check if a size after growing is not more than
 * size of the dense representation.
 */
static int
hll_sparse_can_grow(const uint8_t *sparse_hll)
{
	uint32_t precision = hll_sprase_dense_precision(sparse_hll);
	uint32_t current_size = hll_sparse_size(sparse_hll);
	uint32_t new_size = HLL_SPARSE_GROW_COEF * current_size;
	uint32_t max_size = hll_sparse_max_size(precision);
	return new_size <= max_size;
}

/* Increas the size of HyperLogLog by HLL_SPARSE_GROW_COEF times. */
static void
hll_sparse_grow(uint8_t **sparse_hll_addr)
{
	uint8_t *sparse_hll = *sparse_hll_addr;
	uint32_t size = hll_sparse_size(sparse_hll);
	sparse_hll = realloc(sparse_hll, HLL_SPARSE_GROW_COEF * size);
	struct hll_sparse_header *header = hll_sparse_header(sparse_hll);
	header->size *= HLL_SPARSE_GROW_COEF;
	*sparse_hll_addr = sparse_hll;
}

/*
 * Find the index of pair that stores a register index idx.
 * If there is no such pair return a number of pairs.
 */
static uint32_t
hll_sparse_find_idx(const uint8_t *sparse_hll, uint32_t idx)
{
	uint32_t count = hll_sparse_pairs_count(sparse_hll);
	pair_t *pairs = hll_sparse_pairs(sparse_hll);
	for (uint32_t i = 0; i < count; ++i) {
		if (hll_sparse_pair_idx(pairs[i]) == idx) {
			return i;
		}
	}
	return count;
}

/* Insert a new pair at the end of the HyperLogLog list. */
static void
hll_sparse_insert(uint8_t *sparse_hll, pair_t pair)
{
	struct hll_sparse_header *header = hll_sparse_header(sparse_hll);
	pair_t *pairs = hll_sparse_pairs(sparse_hll);
	assert(header->pairs_count < hll_sparse_pairs_max_count(sparse_hll));
	pairs[header->pairs_count++] = pair;
}

/*
 * Convert a sparsely represented HyperLogLog to a densely represented
 * HyperLogLog. The sparsely represented HyperLogLog is freed after converting.
 */
static void
hll_sparse_to_dense(struct hll *hll)
{
	assert(hll->representation == HLL_SPARSE);
	uint8_t *sparse_hll = hll->data;
	uint8_t precision = hll_sprase_dense_precision(sparse_hll);
	hll->representation = HLL_DENSE;
	hll->precision = precision;
	hll->data = calloc(hll_dense_reqired_memory(hll->precision), 1);

	size_t pairs_count = hll_sparse_pairs_count(sparse_hll);
	pair_t *pairs = hll_sparse_pairs(sparse_hll);
	for (size_t i = 0; i < pairs_count; ++i) {
		uint8_t new_rank = hll_sparse_pair_dense_rank(pairs[i]);
		uint32_t idx = hll_sparse_pair_dense_idx(pairs[i], precision);
		uint32_t rank = hll_dense_register_rank(hll, idx);
		if (rank < new_rank)
			hll_dense_set_register_rank(hll, idx, new_rank);
	}
	hll_invalidate_cache(hll);
	free(sparse_hll);
}

/*
 * Add hash to the sparsely represented HyperLogLog estimator.
 * The representation may be changed to dense after the call.
*/
static void
hll_sparse_add(struct hll *hll, uint64_t hash)
{
	uint32_t idx = hll_hash_register_idx(hash, HLL_SPARSE_PRECISION);
	uint32_t rank = hll_hash_rank(hash, HLL_SPARSE_PRECISION);

	pair_t *pairs = hll_sparse_pairs(hll->data);
	uint32_t count = hll_sparse_pairs_count(hll->data);
	uint32_t pair_idx = hll_sparse_find_idx(hll->data, idx);
	if (pair_idx != count) {
		if (hll_sparse_pair_rank(pairs[pair_idx]) < rank) {
			pairs[pair_idx] = hll_sparse_new_pair(idx, rank);
			return;
		}
	}

	if (hll_sparse_is_full(hll->data)) {
		if (hll_sparse_can_grow(hll->data)) {
			hll_sparse_grow(&hll->data);
		} else {
			hll_sparse_to_dense(hll);
			hll_dense_add(hll, hash);
			return;
		}
	}

	pair_t new_pair = hll_sparse_new_pair(idx, rank);
	hll_sparse_insert(hll->data, new_pair);
}

/* Estimate the cardinality of the sparsely represented HyperLogLog. */
static uint64_t
hll_sparse_estimate(const uint8_t *hll_sparse)
{
	/*
	 * Since the number of pairs is low compared to linaer counting
	 * estimation treshold linear counting is always used.
	 */
	size_t n_counters = hll_n_registers(HLL_SPARSE_PRECISION);
	size_t n_pairs = hll_sparse_pairs_count(hll_sparse);
	return linear_counting(n_counters, n_counters - n_pairs);
}

void
hll_add(struct hll *hll, uint64_t hash)
{
	switch (hll->representation) {
		case HLL_SPARSE:
			hll_sparse_add(hll, hash);
			break;
		case HLL_DENSE:
			hll_dense_add(hll, hash);
			break;
		//default:
			//unreachable();
	}
}

uint64_t
hll_estimate(struct hll *hll)
{
	switch (hll->representation) {
		case HLL_SPARSE:
			return hll_sparse_estimate(hll->data);
		case HLL_DENSE:
			return hll_dense_estimate(hll);
		default:
			//unreachable();
			return 0;
	}
}

void
hll_destroy(struct hll *hll)
{
	free(hll->data);
	free(hll);
}

struct hll *
hll_create(uint8_t precision, enum HLL_REPRESENTATION representation)
{
	assert(precision >= HLL_MIN_PRECISION);
	assert(precision <= HLL_MAX_PRECISION);
	struct hll *hll = calloc(1, sizeof(*hll));
	if (representation == HLL_SPARSE &&
	    precision >= HLL_SPARSE_MIN_PRECISION) {
		hll_sparse_init(hll, precision);
	} else {
		hll_dense_init(hll, precision);
	}
	return hll;
}
