#ifndef CALS_UTILS_H
#define CALS_UTILS_H

#include <string>

#include "ktensor.h"

namespace cals::utils {
/** Create a string with the modes of a tensor.
 *
 * @param modes vector containing the modes of the tensor.
 *
 * @return String with the modes of a tensor.
 */
std::string mode_string(std::vector<dim_t> const &modes);

cals::Ktensor concatenate_ktensors(std::vector<cals::Ktensor> const &ktensors);

void generate_jk_ktensors(cals::Ktensor const &reference_ktensor, std::vector<cals::Ktensor> &jk_ktensor_v);

void jk_permutation_adjustment(cals::Ktensor &ktensor, std::vector<cals::Ktensor> &jk_ktensor_v);

vector<double> calculate_jackknifing_norms(cals::Tensor const &tensor);
} // namespace cals::utils

namespace cals::ops {
/** Compute the gramian by performing an A^{T}A operation.
 *
 * @param[in] factor Factor Matrix for which to compute the gramian.
 * @param[out] gramian Matrix in which to store the gramian.
 */
void update_gramian(const cals::Matrix &factor, cals::Matrix &gramian);

/** Compute all gramians of a Ktensor by performing an A^{T}A operation.
 *
 * @param[in] Ktensor Ktensor whose gramians should be updated.
 * @param[out] gramians Vector of Matrix objects, in which to store the gramians.
 */
void update_gramians(const cals::Ktensor &ktensor, vector<Matrix> &gramians);

/** Compute the hadamard product of a set of matrices except one (which will be the output).
 *
 * Hadamard of the all matrices in \p matrices, except the Matrix in position \p mode. (all matrices are assumed of
 * same size. Store the result in position \p mode.
 *
 * @param[in, out] matrices vector containing Matrix objects to be multiplied together.
 * @param[in] mode specify the index of the matrix in \p matrices, in which the output is going to be written.
 *
 * @return Reference to Matrix used as output.
 */
Matrix &hadamard_but_one(std::vector<cals::Matrix> &matrices, dim_t mode);

/** Compute the hadamard product of all matrices and store the result in the first.
 *
 * @param[in, out] matrices vector containing Matrix objects to be multiplied together. The first matrix is used as
 * output.
 */
void hadamard_all(std::vector<cals::Matrix> &matrices);
} // namespace cals::ops

#endif // CALS_UTILS_H
