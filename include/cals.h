#ifndef CALS_CALS_H
#define CALS_CALS_H

#include <cfloat>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>

#include "als.h"
#include "ktensor.h"
#include "tensor.h"
#include "timer.h"
#include "utils/line_search.h"
#include "utils/mttkrp.h"
#include "utils/update.h"
#include "utils/utils.h"

namespace cals {
/* A queue of references to Ktensors, used as input to CALS. */
typedef std::queue<std::reference_wrapper<Ktensor>> KtensorQueue;

/** struct containing information about a CALS execution.
 *
 */
struct CalsReport {
  // Target tensor details
  int tensor_rank;     /*!< Target tensor rank (if available). */
  dim_t n_modes;       /*!< Target tensor number of modes (dimensions). */
  vector<dim_t> modes; /*!< Target tensor mode sizes. */
  double X_norm;       /*!< Target tensor norm. */

  // Execution parameters
  dim_t iter{0};                       /*!< Total number of iterations of the CALS algorithm. */
  dim_t max_iter;                      /*!< Maximum number of iterations for each model. */
  int n_threads;                       /*!< Number of threads available. */
  dim_t buffer_size;                   /*!< Max buffer size used. */
  int n_ktensors{0};                   /*!< Total number of Ktensors that were fitted to the target tensor. */
  int ktensor_comp_sum{0};             /*!< Total component sum of all Ktensors fitted to the target tensor. */
  double tol;                          /*!< Tolerance used to determine when a Ktensor had reached convergence. */
  bool cuda;                           /*!< Whether CUDA was used. */
  update::UPDATE_METHOD update_method; /*!< Update method used for all Ktensors. */
  std::string output_file_name;        /*!< Name of the output file to export data (used in experiments) */

  bool line_search;         /*!< Whether line search was used. */
  int line_search_interval; /*!< Number of iterations (per model) when line search was invoked. */
  double line_search_step;  /*!< Factor with which line search moved. */
  dim_t ls_performed{0};
  dim_t ls_failed{0};
  ls::LS_METHOD line_search_method;

  // Timers
  double total_time{0.0};
#if WITH_TIME
  Matrix als_times{};
  Matrix mode_times{};
  Matrix mttkrp_times{};

  // Experiment analysis data
  vector<uint64_t> flops_per_iteration; /*!< Number of FLOPS performed in each iteration. */
  vector<dim_t> cols;                   /*!< Number of active columns of the multi-factors for every iteration. */
#endif

  /* Print header of CSV file, containing information about an CALS invocation.
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
    file << "BUFFER_SIZE" << sep;
    file << "N_KTENSORS" << sep;
    file << "KTENSOR_COMP_SUM" << sep;
    file << "UPDATE_METHOD" << sep;
    file << "LINE_SEARCH" << sep;
    file << "MAX_ITERS" << sep;
    file << "ITER" << sep;
    file << "NUM_THREADS" << sep;
    file << "TOTAL" << sep;
#if WITH_TIME
    file << "FLOPS" << sep;
    file << "COLS" << sep;
    for (const auto &name : als_timers.names)
      file << name << sep;
    for (auto i = 0lu; i < modes.size(); i++)
      for (auto &name : mode_timers.names)
        file << "MODE_" << i << "_" << name << sep;
#endif
    file << std::endl;
  }

  /* Print contents of CSV file, containing information about an CALS invocation.
   *
   * @param file_name Name of CSV file.
   * @param sep Optional argument, separator of CSV.
   * */
  void print_to_file(const std::string &file_name, const std::string &sep = ";") const {
    auto file = std::ofstream(file_name, std::ios::app);
    auto max_it = static_cast<dim_t>(iter - 1);

    for (dim_t it = 0; it <= max_it; it++) {
      file << tensor_rank << sep;
      file << cals::utils::mode_string(modes) << sep;
      file << buffer_size << sep;
      file << n_ktensors << sep;
      file << ktensor_comp_sum << sep;
      file << update::update_method_names[update_method] << sep;
      file << line_search << sep;
      file << max_iter << sep;
      file << it + 1 << sep;
      file << n_threads << sep;
      file << total_time << sep;
#if WITH_TIME
      file << flops_per_iteration[it] << sep;
      file << cols[it] << sep;

      file << std::scientific;
      for (dim_t i = 0; i < als_times.get_rows(); i++)
        file << als_times(i, it) << sep;

      for (dim_t i = 0; i < mode_times.get_rows(); i++)
        file << mode_times(i, it) << sep;
#endif
      file << std::endl;
    }
  }
};

/** Struct containing all the execution parameters of the CALS algorithm.
 *
 */
struct CalsParams {
  /*! Method for updating factor matrices. */
  update::UPDATE_METHOD update_method{update::UPDATE_METHOD::UNCONSTRAINED};

  /*! MTTKRP method to use. Look at mttkrp::MTTKRP_METHOD. */
  mttkrp::MTTKRP_METHOD mttkrp_method{mttkrp::MTTKRP_METHOD::AUTO};

  /*! Lookup table with the best variant of MTTKRP per mode. */
  cals::mttkrp::MttkrpLut mttkrp_lut{};

  dim_t max_iterations{200}; /*!< Maximum number of iterations before evicting a model. */
  double tol{1e-7};          /*!< Tolerance of fit difference between consecutive iterations. */
  bool cuda{false};          /*!< Use CUDA (make sure code is compiled with CUDA support). */
  dim_t buffer_size{4200};   /*!< Maximum size of the multi-factor columns (sum of ranks of concurrent models) */

  bool line_search{false};     /*!< Use line search. */
  int line_search_interval{5}; /*!< Number of iterations when line search is invoked. */
  double line_search_step{0};  /*!< Factor for line search. If not set (set to 0), (iteration)^(1/3) is used. */
  ls::LS_METHOD line_search_method{ls::NO_ERROR_CHECKING};

  bool force_max_iter{false};     /*!< Force maximum iterations for every model (for experiments) */
  bool always_evict_first{false}; /*!< Always evict the first model in the buffer (for experiments) */

  /** Print contents of CalsParams to standard output.
   */
  void print() const {
    using std::cout;
    using std::endl;

    cout << "---------------------------------------" << endl;
    cout << "CALS parameters" << endl;
    cout << "---------------------------------------" << endl;
    cout << "Tol:             " << tol << endl;
    cout << "Max Iterations:  " << max_iterations << endl;
    cout << "Buffer Size:     " << buffer_size << endl;
    cout << "Mttkrp Method:   " << mttkrp::mttkrp_method_names[mttkrp_method] << endl;
    cout << "Update Method:   " << update::update_method_names[update_method] << endl;
    cout << "Line Search:     " << ((line_search) ? "true" : "false") << endl;
    if (line_search) {
      cout << "-Line Search Interval: " << line_search_interval << " iterations" << endl;
      cout << "-Line Search Method:   " << ls::ls_method_names[line_search_method] << endl;
    }
    cout << "CUDA:            " << ((cuda) ? "true" : "false") << endl;
    cout << "---------------------------------------" << endl;
  }
};

/** Computes multiple Concurrent Alternating Least Squares algorithms, for the Canonical Polyadic Decomposition.
 *
 * The function consumes Ktensors from the input queue \p kt_queue, fits them to the target tensor \p X using ALS,
 * and then overwrites each input Ktensor with the result.
 *
 * @param X Target tensor to be decomposed.
 * @param kt_queue A queue of references to the input Ktensors, which need to be fitted to the target tensor using ALS.
 * @param cals_params Parameters for the algorithm.
 *
 * @return CalsReport object, containing all data from the execution.
 */
CalsReport cp_cals(const Tensor &X, KtensorQueue &kt_queue, CalsParams &cals_params);

JKReport jk_cp_cals(const Tensor &X, vector<Ktensor> &kt_vector, CalsParams &cals_params);
} // namespace cals
#endif // CALS_CALS_H
