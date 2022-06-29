#include "utils/line_search.h"

#include <iostream>

#include "utils/error.h"
#include "utils/utils.h"

using cals::Ktensor;
using cals::Matrix;
using std::vector;

namespace cals::ls {

/** Perform line search extrapolation without checking for the new error.
 *
 * The ktensor is extrapolated based on the difference between the current (\p ktensor) and the previous (\p
 * ls_prev_ktensor), and the step, specified in \p params.
 *
 * @param[in, out] ktensor the Ktensor on which to perform line search.
 * @param[in] ls_prev_ktensor Ktensor object containing the Ktensor of the previous iteration.
 * @param[out] gramians Vector containing the gramian Matrix objects, to be updated.
 * @param[in] params LineSearchParams object, containing parameters related to the line search extrapolation.
 */
void line_search_no_error_checking(Ktensor &ktensor,
                                   Ktensor &ls_prev_ktensor,
                                   vector<Matrix> &gramians,
                                   LineSearchParams &params) {

  ktensor.denormalize();
  ls_prev_ktensor.denormalize();
  for (dim_t n = 0; n < ktensor.get_n_modes(); n++) {
    auto &ktf = ktensor.get_factor(n);
    auto &pktf = ls_prev_ktensor.get_factor(n);

    for (dim_t i = 0; i < ktf.get_n_elements(); i++)
      ktf[i] += params.step * (ktf[i] - pktf[i]);
  }
  ktensor.normalize();

  ktensor.set_approximation_error(std::numeric_limits<double>::max());
  ktensor.calculate_new_fit(1.0);

#if CUDA_ENABLED
  auto stream = cuda::create_stream();
#endif

  if (params.cuda) {
#if CUDA_ENABLED
    for (dim_t n = 0; n < ktensor.get_n_modes(); n++)
      ktensor.get_factor(n).send_to_device_async(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  cals::ops::update_gramians(ktensor, gramians);

  if (params.cuda) {
#if CUDA_ENABLED
    cudaStreamSynchronize(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

#if CUDA_ENABLED
  cuda::destroy_stream(stream);
#endif
}

/** Perform line search extrapolation. (only available for 3D tensors)
 *
 * The ktensor is extrapolated and the new approximation error is calculated (with respect to the target tensor). If the
 * new error decreases, substitute the ktensor with the extrapolation. Otherwise, leave the ktensor as is.
 *
 * @param[in, out] ktensor the Ktensor on which to perform line search. If the error decreases, the \p ktensor is
 * updated.
 * @param[in] ls_ktensor Ktensor object to temporarily store the extrapolated ktensor.
 * @param[out] gramians Vector containing the gramian Matrix objects, to be updated in case the \p ktensor is updated.
 * @param[in] X Target tensor, used for error calculation.
 * @param[in] X_norm L2 norm of the target tensor.
 * @param[in] params LineSearchParams object, containing parameters related to the line search extrapolation.
 */
void line_search_error_checking(Ktensor &ktensor,
                                Ktensor &ls_ktensor,
                                vector<Matrix> &gramians,
                                const Tensor &X,
                                double X_norm,
                                LineSearchParams &params) {
  auto modes = X.get_modes();
  std::sort(modes.begin(), modes.end(), std::greater<>());
  Matrix krp_workspace(modes[0] * modes[1], static_cast<dim_t>(ktensor.get_components()));
  Matrix ten_workspace(modes[0], modes[1] * modes[2]);

  for (dim_t n = 0; n < ls_ktensor.get_n_modes(); n++) {
    const auto &curr_factor = ktensor.get_factor(n);
    auto &old_factor = ls_ktensor.get_factor(n);

    krp_workspace.resize(curr_factor.get_rows(), curr_factor.get_cols());
    for (dim_t i = 0; i < old_factor.get_n_elements(); i++)
      krp_workspace[i] = curr_factor[i] - old_factor[i];

    old_factor.copy(curr_factor);
    cblas_daxpy(old_factor.get_n_elements(), params.step, krp_workspace.get_data(), 1, old_factor.get_data(), 1);
  }
  for (auto i = 0lu; i < ktensor.get_lambda().size(); i++)
    ls_ktensor.get_lambda()[i] = ktensor.get_lambda()[i];

  double error = cals::error::compute_error(X, ls_ktensor, krp_workspace, ten_workspace);

  const double old_error = ktensor.get_approximation_error();

  params.reversed = true;
  if (error < old_error) {
    params.reversed = false;
#if CUDA_ENABLED
    auto stream = cuda::create_stream();
#endif

    DEBUG(std::cout << "Fast forwarding... id: " << ktensor.get_id() << " Old: " << old_error << " New: " << error
                    << std::endl;);
    for (dim_t n = 0; n < ls_ktensor.get_n_modes(); n++) {
      ktensor.get_factor(n).copy(ls_ktensor.get_factor(n));

      if (params.cuda) {
#if CUDA_ENABLED
        ktensor.get_factor(n).send_to_device_async(stream);
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }
      cals::ops::update_gramian(ktensor.get_factor(n), gramians[n]);
    }

    ktensor.set_approximation_error(error);
    ktensor.calculate_new_fit(X_norm);

    if (params.cuda) {
#if CUDA_ENABLED
      cudaStreamSynchronize(stream);
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }
#if CUDA_ENABLED
    cuda::destroy_stream(stream);
#endif
  }
}

/** Perform line search extrapolation. This version favors concurrent calculation of line search, without high
 * memory usage. (only available for 3D tensors)
 *
 * The ktensor is extrapolated and the new approximation error is calculated (with respect to the target tensor). If the
 * new error decreases, substitute the ktensor with the extrapolation. Otherwise, leave the ktensor as is.
 *
 * @param[in, out] ktensor the Ktensor on which to perform line search. If the error decreases, the \p ktensor is
 * updated.
 * @param[in] ls_ktensor Ktensor object to temporarily store the extrapolated ktensor.
 * @param[in] ls_tr_ktensor Ktensor object to temporarily store the extrapolated ktensor.
 * @param[out] gramians Vector containing the gramian Matrix objects, to be updated in case the \p ktensor is updated.
 * @param[in] X Target tensor, used for error calculation.
 * @param[in] X_norm L2 norm of the target tensor.
 * @param[in] params LineSearchParams object, containing parameters related to the line search extrapolation.
 */
void line_search_error_checking_par(Ktensor &ktensor,
                                    Ktensor &ls_ktensor,
                                    Ktensor &ls_tr_ktensor,
                                    vector<Matrix> &gramians,
                                    const Tensor &X,
                                    double X_norm,
                                    LineSearchParams &params) {
#if CUDA_ENABLED
  auto stream = cuda::create_stream();
#endif
  for (dim_t n = 0; n < ls_ktensor.get_n_modes(); n++) {
    const auto &curr_factor = ktensor.get_factor(n);
    auto &old_factor = ls_ktensor.get_factor(n);

    for (dim_t i = 0; i < old_factor.get_n_elements(); i++)
      old_factor[i] = curr_factor[i] + params.step * (curr_factor[i] - old_factor[i]);
  }
  for (dim_t i = 0; i < ktensor.get_lambda().size(); i++)
    ls_ktensor.get_lambda()[i] = ktensor.get_lambda()[i];

  double error = cals::error::compute_error_par(X, ls_ktensor, ls_tr_ktensor);

  const double old_error = ktensor.get_approximation_error();

  if (error < old_error) {
    DEBUG(std::cout << "Fast forwarding... id: " << ktensor.get_id() << " Old: " << old_error << " New: " << error
                    << std::endl;);
    for (dim_t n = 0; n < ls_ktensor.get_n_modes(); n++) {
      ktensor.get_factor(n).copy(ls_ktensor.get_factor(n));

      if (params.cuda) {
#if CUDA_ENABLED
        ktensor.get_factor(n).send_to_device_async(stream);
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }
      cals::ops::update_gramian(ktensor.get_factor(n), gramians[n]);
    }

    ktensor.set_approximation_error(error);
    ktensor.calculate_new_fit(X_norm);

    if (params.cuda) {
#if CUDA_ENABLED
      cudaStreamSynchronize(stream);
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }
#if CUDA_ENABLED
    cuda::destroy_stream(stream);
#endif
  }
}

void line_search(cals::Ktensor &ktensor, std::vector<cals::Matrix> &gramians, cals::ls::LineSearchParams &params) {

#if CUDA_ENABLED
  auto stream = cuda::create_stream();
#endif
  params.reversed = false;
  params.extrapolated = false;
  params.iter++;
  if (params.method == NO_ERROR_CHECKING) {
    if (params.updated_last_iter) {
      params.updated_last_iter = false;
      if (params.backup_ktensor.get_approximation_error() < ktensor.get_approximation_error()) {
        params.reversed = true;
        params.iter = 0;
        ktensor.copy(params.backup_ktensor);
        if (params.cuda) {
#if CUDA_ENABLED
          for (dim_t n = 0; n < ktensor.get_n_modes(); n++)
            ktensor.get_factor(n).send_to_device_async(stream);
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        cals::ops::update_gramians(ktensor, gramians);

        if (params.cuda) {
#if CUDA_ENABLED
          cudaStreamSynchronize(stream);
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }
      }
    }
    if (params.iter == params.interval) {
      params.extrapolated = true;
      params.iter = 0;                     // Reset the line search iteration counter (to count for the next LS)
      params.updated_last_iter = true;     // Indicate to the next iteration that line search was performed.
      params.backup_ktensor.copy(ktensor); // Backup Ktensor in case the error increases and we need to revert back.
      line_search_no_error_checking(ktensor, params.prev_ktensor, gramians, params);
    }
  } else if (params.method == ERROR_CHECKING_SERIAL) {
    if (params.iter == params.interval) {
      params.extrapolated = true;
      params.iter = 0;
#pragma omp critical
      ls::line_search_error_checking(ktensor, params.prev_ktensor, gramians, *params.T, (*params.T).norm(), params);
    }
  }
#if CUDA_ENABLED
  cuda::destroy_stream(stream);
#endif
}

} // namespace cals::ls
