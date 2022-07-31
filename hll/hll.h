#ifndef HYPER_LOG_LOG_H_INCLUDED
#define HYPER_LOG_LOG_H_INCLUDED

#include <stdint.h>

/**
 * HyperLogLog supports sparse and dense represenations.
 * The rerpesentation determine the data storage scheme.
 */
enum HLL_REPRESENTATION {
	/**
	* Sparse representation stores only pairs of index and its the
 	* highest rank that has added. It requires less memory for
 	* small cardinalities and provides the best accuracy. Sparse
 	* representation swithces to dence representation if it starts
 	* to require more amount of memory that is needed for the dense
 	* representation.
 	*/
	HLL_SPARSE,
	/*
	 * Dense representation allocates 2^precision counters. For small
	 * cardinalities most counters are not used so this representation
	 * should be used for esimating large cardinalities.
	 */
	HLL_DENSE
};

/**
 * Estimator that is used for the HyperLogLog algorithm.
 * The algorithm allows to estimate cardinality of a multiset using fixed amount
 * of memory or even less. Memory requirements and estimation accuracy are
 * determined by the algorithm precision parameter.
 * For the dense representation the relative error is 1.04/sqrt(m) and the
 * memory capasity is m*6 bits where m is number of counters wich
 * equals to 2^precision.
 * For the sparse representation the memory usage is proportional to the number
 * of distinct elements that has added until it reaches the memory usage of the
 * dense representation, and then it switches to the dense representation with
 * fixed memory requretments. The sparse representation has the best accuracy.
 */
struct hll {
	/** See the comment to HLL_REPRESENTATION enum. */
	enum HLL_REPRESENTATION representation;
	/**
	 * Interpretation of the data depends on the representation.
	 * For dense representaion it's an array of registers of size
	 * 2^precision * 6 bits. Registers store the maximum added rank of set
	 * of hashes wich last precision bits are equal to register index.
	 * For the sparse representation it's a sparsely represented
	 * HyperLogLog. It stores only pairs of index and its the highest rank
	 * that has added so it needs less memory for small cardinalities.
	 */
	uint8_t *data;
	/**
	 * Precision is equal to the number of bits that are interpreted as
	 * a register index. Available values are from HLL_MIN_PRECISION to
	 * HLL_MAX_PRECISION (defined in hll_emprirical.h.)
	 * The larger value leads to less estimation error but larger
	 * memory requirement (2^precision * 6 bits).
	 */
	uint8_t precision;
	/**
	 * Cached value of the last estimation.
	 */
	double cached_estimation;
};

/**
 * Create a HyperLogLog estimator.
 *
 * \param precision defines the estimation error and memory requirements of
 * 		    the dense representation.
 * 		    The algorithm needs no more than 2^precision * 6 bits.
 * 		    The error formula of the dense representaion is
 * 		    1.04 / sqrt(2^precision). The valid values are defined by
 * 		    constants HLL_MIN_PRECISION and HLL_MAX_PRECISION from
 *                  hll_emprirical.h file.
 *
 * \param representation determines the data storage scheme for small
 * 			 cardinalities. For big carinalities the dense
 * 			 representation is always used.
 * 			 For more details see #HLL_REPRESENTATION.
 */
struct hll *
hll_create(uint8_t precision, enum HLL_REPRESENTATION representation);

/**
 * Add a hash of a dataset element to the hll estimator.
 */
void
hll_add(struct hll *hll, uint64_t hash);

/**
 * Estimate cardinality of the hll estimator.
 */
uint64_t
hll_estimate(struct hll *hll);

/**
 * Destroy the hll estimator.
 */
void
hll_destroy(struct hll *hll);

#endif /* HYPER_LOG_LOG_H_INCLUDED */
