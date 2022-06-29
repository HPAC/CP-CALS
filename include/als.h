#ifndef CALS_ALS_H
#define CALS_ALS_H

#include <fstream>
#include <iostream>

#include "ktensor.h"
#include "timer.h"
#include "utils/line_search.h"
#include "utils/mttkrp.h"
#include "utils/update.h"
#include "utils/utils.h"

namespace cals {

struct JKTime {
  double pre_als_time{0.0};
  double als_time{0.0};
};

struct JKReport {
  JKTime jk_time{};
  vector<vector<Ktensor>> results;
};

/** struct containing information about an ALS execution.
 *
 */
struct AlsReport {
  // Target tensor details
  int tensor_rank;     /*!< Target tensor rank (if available). */
  dim_t n_modes;       /*!< Target tensor number of modes (dimensions). */
  vector<dim_t> modes; /*!< Target tensor mode sizes. */
  double X_norm;       /*!< Target tensor norm. */

  // Execution parameters
  dim_t iter = 0;           /*!< Total number of iterations performed by the ALS algorithm. */
  dim_t max_iter;           /*!< Maximum number of iterations set by the user. */
  int n_threads;            /*!< Number of threads available. */
  int ktensor_id;           /*!< ID of the Ktensor that was fitted. */
  dim_t ktensor_components; /*!< Number of components of the Ktensor that was fitted. */
  double tol;               /*!< Tolerance used to determine when a Ktensor has reached convergence. */
  bool cuda;                /*!< Whether CUDA was used. */

  update::UPDATE_METHOD update_method; /*!< Update method used for all Ktensors. */

  bool line_search;         /*!< Whether line search was used. */
  int line_search_interval; /*!< Number of iterations (per model) when line search was invoked. */
  double line_search_step;  /*!< Factor with which line search moved. */
  dim_t ls_performed{0};
  dim_t ls_failed{0};
  ls::LS_METHOD line_search_method;

  // Experiment analysis data
  uint64_t flops_per_iteration; /*!< Number of FLOPS performed in each iteration. */

  // Timers
  double total_time{0.0};
#if WITH_TIME
  Matrix als_times{};
  Matrix mode_times{};
  Matrix mttkrp_times{};
#endif

  /* Print header of CSV file, containing information about an ALS invocation.
   *
   * @param file_name Name of CSV file.
   * @param sep Optional argument, separator of CSV.
   * */
  void print_header(const std::string &file_name, const std::string &sep = ";") const {
    auto file = std::ofstream(file_name, std::ios::out);

    AlsTimers als_timers;
    ModeTimers mode_timers;
    file << "TENSOR_RANK" << sep;
    file << "TENSOR_MODES" << sep;
    file << "KTENSOR_ID" << sep;
    file << "KTENSOR_COMP" << sep;
    file << "UPDATE_METHOD" << sep;
    file << "LINE_SEARCH" << sep;
    file << "MAX_ITERS" << sep;
    file << "ITER" << sep;
    file << "NUM_THREADS" << sep;
    file << "TOTAL" << sep;
#if WITH_TIME
    file << "FLOPS" << sep;
    for (const auto &name : als_timers.names)
      file << name << sep;
    for (auto i = 0lu; i < modes.size(); i++)
      for (auto &name : mode_timers.names)
        file << "MODE_" << i << "_" << name << sep;
#endif
    file << std::endl;
  }

  /* Print contents of CSV file, containing information about an ALS invocation.
   *
   * @param file_name Name of CSV file.
   * @param sep Optional argument, separator of CSV.
   * */
  void print_to_file(const std::string &file_name, const std::string &sep = ";") const {
    auto file = std::ofstream(file_name, std::ios::app);

    file << tensor_rank << sep;
    file << cals::utils::mode_string(modes) << sep;
    file << ktensor_id << sep;
    file << ktensor_components << sep;
    file << update::update_method_names[update_method] << sep;
    file << line_search << sep;
    file << max_iter << sep;
    file << iter << sep;
    file << n_threads << sep;
    file << total_time << sep;
#if WITH_TIME
    file << flops_per_iteration << sep;
    file << std::scientific;

    auto max_it = static_cast<dim_t>(iter - 1);

    // Find the minimum value per row (min across all iterations) and write it to file
    for (dim_t i = 0; i < als_times.get_rows(); i++) {
      double min = std::numeric_limits<double>::max();
      for (dim_t j = 0; j < max_it; j++)
        if (als_times(i, j) < min)
          min = als_times(i, j);
      file << min << sep;
    }

    // Find the minimum value per row (min across all iterations) and write it to file
    for (dim_t i = 0; i < mode_times.get_rows(); i++) {
      double min = std::numeric_limits<double>::max();
      for (dim_t j = 0; j < max_it; j++)
        if (mode_times(i, j) < min)
          min = mode_times(i, j);
      file << min << sep;
    }
#endif
    file << std::endl;
  }
};

struct AlsParams {
  /*! Method for updating factor matrices. */
  update::UPDATE_METHOD update_method{update::UPDATE_METHOD::UNCONSTRAINED};

  /*! MTTKRP method to use. Look at mttkrp::MTTKRP_METHOD. */
  mttkrp::MTTKRP_METHOD mttkrp_method{mttkrp::MTTKRP_METHOD::AUTO};

  /*! Lookup table with the best variant of MTTKRP per mode. */
  cals::mttkrp::MttkrpLut mttkrp_lut{};

  dim_t max_iterations{200}; /*!< Maximum number of iterations before evicting a model. */
  double tol{1e-7};          /*!< Tolerance of fit difference between consecutive iterations. */

  bool cuda{false}; /*!< Use CUDA (make sure code is compiled with CUDA support). */

  bool line_search{false};     /*!< Use line search. */
  int line_search_interval{5}; /*!< Number of iterations when line search is invoked.*/
  double line_search_step{0};  /*!< Factor for line search. If not set (set to 0), (iteration)^(1/3) is used. */
  ls::LS_METHOD line_search_method{ls::NO_ERROR_CHECKING};

  // Internal variables (not for use by the users)
  bool cuda_no_tensor_alloc{false}; /*!< Tell ALS that the tensor has already been allocated on the GPU. */

  bool force_max_iter{false};       /*!< Force maximum iterations for every model (for experiments) */
  bool suppress_lut_warning{false}; /*!< Suppress the warning that the LookupTable was not found (for experiments) */

  /** Print contents of AlsParams to standard output.
   */
  void print() const {
    using std::cout;
    using std::endl;

    cout << "---------------------------------------" << endl;
    cout << "ALS parameters" << endl;
    cout << "---------------------------------------" << endl;
    cout << "Tolerance:        " << tol << endl;
    cout << "Max Iterations:   " << max_iterations << endl;
    cout << "Mttkrp Method:    " << mttkrp::mttkrp_method_names[mttkrp_method] << endl;
    cout << "Update Method:    " << update::update_method_names[update_method] << endl;
    cout << "Line Search:      " << ((line_search) ? "true" : "false") << endl;
    if (line_search) {
      cout << "-Line Search Interval: " << line_search_interval << " iterations" << endl;
      cout << "-Line Search Method:   " << ls::ls_method_names[line_search_method] << endl;
    }
    cout << "CUDA:             " << ((cuda) ? "true" : "false") << endl;
    cout << "---------------------------------------" << endl;
  }
};

/** Computes a single Alternating Least Squares algorithm, for the Canonical Polyadic Decomposition.
 *
 * The function fits the \p ktensor to the target tensor \p X using ALS, overwritting the \p ktensor with the result.
 *
 * @param X Target tensor to be decomposed.
 * @param ktensor A Ktensor, which needs to be fitted to the target tensor using ALS.
 * @param als_params Parameters for the algorithm.
 *
 * @return AlsReport object, containing all data from the execution.
 */
AlsReport cp_als(const Tensor &X, Ktensor &ktensor, AlsParams &als_params);

JKReport jk_cp_als(const Tensor &X, vector<Ktensor> &kt_vector, AlsParams &als_params);

/** Computes multiple Alternating Least Squares algorithms for the Canonical Polyadic Decomposition, in parallel, using
 * OpenMP.
 *
 * The function lets OpenMP create threads and distribute the computation specified in the vector \p ktensor_v.
 * Each thread invokes cp_als to fit each input Ktensor to the target tensor \p X using ALS and then overwrites
 * the Ktensor with the result.
 *
 * @param X Target tensor to be decomposed.
 * @param ktensor_v A reference to a vector of input Ktensors, which need to be fitted to the target tensor using ALS.
 * @param params Parameters for the algorithm.
 *
 * @return vector of AlsReport objects, containing data from the execution of each als algorithm.
 */
vector<AlsReport> cp_omp_als(const Tensor &X, vector<Ktensor> &ktensor, AlsParams &params);

JKReport jk_cp_omp_als(const Tensor &X, vector<Ktensor> &kt_vector, AlsParams &params);

} // namespace cals

#endif // CALS_ALS_H
