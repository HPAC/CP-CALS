//@HEADER
// ******************************************************************************
//
//  CP-CALS: Software for computing the Canonical Polyadic Decomposition using
//  the Concurrent Alternating Least Squares Algorithm.
//
//  Copyright (c) 2020, Christos Psarras
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ******************************************************************************
//@HEADER

#ifndef CALS_CALS_H
#define CALS_CALS_H

#include <queue>
#include <cfloat>
#include <numeric>
#include <fstream>
#include <iostream>
#include <functional>

#include "utils.h"
#include "timer.h"
#include "tensor.h"
#include "ktensor.h"

namespace cals
{
  typedef std::queue<std::reference_wrapper<Ktensor>> KtensorQueue;  /* A queue of references to Ktensors, used as input to CALS. */

  /** struct containing information about a CALS execution.
   *
   */
  struct CalsReport
  {
    // Target tensor details
    int tensor_rank;    /*!< Target tensor rank (if available). */
    int n_modes;        /*!< Target tensor number of modes (dimensions). */
    vector<int> modes;  /*!< Target tensor mode sizes. */
    double X_norm;      /*!< Target tensor norm. */

    // Execution parameters
    int iter{0};                  /*!< Total number of iterations of the CALS algorithm. */
    int max_iter;                 /*!< Maximum number of iterations for each model. */
    int n_threads;                /*!< Number of threads available. */
    int buffer_size;              /*!< Max buffer size used. */
    int n_ktensors{0};            /*!< Total number of Ktensors that were fitted to the target tensor. */
    int ktensor_rank_sum{0};      /*!< Total rank sum of all Ktensors fitted to the target tensor. */
    double tol;                   /*!< Tolerance used to determine when a Ktensor has reached convergence. */
    bool cuda;                    /*!< Whether to use CUDA (make sure that the code is compiled with CUDA support). */
    bool line_search;             /*!< Whether to use line search. */
    int line_search_interval;     /*!< Number of iterations (per model) for invocation of line search. */
    double line_search_step;      /*!< Factor with which to move during line search. */
    update::UPDATE_METHOD update_method;  /*!< Update method used for all Ktensors. */
    std::string output_file_name; /*!< Name of the output file to export data (used in experiments) */

    // Execution data
    TIME(vector<uint64_t> flops_per_iteration;)  /*!< Number of FLOPS performed in each iteration. */
    TIME(vector<int> cols;)   /*!< Number of active columns of the multi-factors for every iteration. */

    // Timers
    TIME(vector<AlsTimer> als_timer;)            /*!< Timers for anything outside of MTTKRP. */
    TIME(vector<vector<ModeTimer>> mode_timers;) /*!< Timers for MTTKRP (per mode). */

    /* Function to initialize timers
     * */
    TIME(void init_timers(int n_modes, int max_iter)
         {
           mode_timers.resize(max_iter, vector<ModeTimer>(n_modes, ModeTimer()));
           als_timer.resize(max_iter, AlsTimer());
           flops_per_iteration.resize(max_iter, 0llu);
           cols.resize(max_iter, 0);
         }
    )

    /* Print header of CSV file.
     *
     * @param file_name Name of CSV file.
     * @param sep Optional argument, separator of CSV.
     * */
    void print_header(const std::string &file_name, const std::string &sep = ";")
    {
      auto file = std::ofstream(file_name, std::ios::out);
      file << "TENSOR_RANK" << sep
           << "TENSOR_MODES" << sep
           << "BUFFER_SIZE" << sep
           << "N_KTENSORS" << sep
           << "KTENSOR_RANK_SUM" << sep
           << "UPDATE_METHOD" << sep
           << "LINE_SEARCH" << sep
           << "MAX_ITERS" << sep
           << "ITER" << sep
           << "NUM_THREADS" << sep;
      TIME(file << "FLOPS" << sep
                << "COLS" << sep;
               for (const auto &name : als_timer[0].names)
                 file << name << sep;
               for (auto i = 0lu; i < modes.size(); i++)
                 for (auto j : {ModeTimer::TOTAL_MTTKRP, ModeTimer::UPDATE})
                   file << "MODE_" << i << "_" << mode_timers[0][0].names[j] << sep;
      )
      file << std::endl;
    }

    /* Print contents of CSV file.
     *
     * @param file_name Name of CSV file.
     * @param sep Optional argument, separator of CSV.
     * */
    void print_to_file(const std::string &file_name, const std::string &sep = ";")
    {
      auto file = std::ofstream(file_name, std::ios::app);

      for (auto it = 1; it <= iter; it++)
      {
        file << tensor_rank << sep;
        for (auto m = modes.cbegin(); m < modes.cend() - 1; m++) file << *m << "-";
        file << *(modes.cend() - 1) << sep;
        file << buffer_size << sep;
        file << n_ktensors << sep;
        file << ktensor_rank_sum << sep;
        file << update::update_method_names[update_method] << sep;
        file << line_search << sep;
        file << max_iter << sep;
        file << it << sep;
        file << n_threads << sep;
        TIME(file << flops_per_iteration[it] << sep;
                 file << cols[it] << sep;
                 for (auto &timer : als_timer[it].timers)
                   file << timer.get_time() << sep;
                 for (auto &mode_timer : mode_timers[it])
                   for (auto j : {ModeTimer::TOTAL_MTTKRP, ModeTimer::UPDATE})
                     file << mode_timer.timers[j].get_min() << sep;
        )
        file << std::endl;
      }
    }
  };

  /** Struct containing all the execution parameters of the CALS algorithm.
   *
   */
  struct CalsParams
  {
    update::UPDATE_METHOD
        update_method{update::UPDATE_METHOD::UNCONSTRAINED}; /*!< Method for updating factor matrices. */
    ops::MTTKRP_METHOD
        mttkrp_method{ops::MTTKRP_METHOD::AUTO}; /*!< MTTKRP method to use. Look at ops::MTTKRP_METHOD. */
    int max_iterations{50};        /*!< Maximum number of iterations before evicting a model. */
    int buffer_size{4200};         /*!< Maximum size of the multi-factor columns (sum of ranks of concurrent models) */
    double tol{1e-4};              /*!< Tolerance of fit difference between consecutive iterations. */

    bool cuda{false};              /*!< Use CUDA (make sure code is compiled with CUDA support). */

    bool line_search{false};       /*!< Use line search. */
    int line_search_interval{5};   /*!< Number of iterations when line search is invoked.*/
    double line_search_step{1.2};  /*!< Factor for line search. */

    bool force_max_iter{false};    /*!< Force maximum iterations for every model (for experiments) */

    /** Print contents of CalsParams to standard output.
     */
    void print() const
    {
      using std::cout;
      using std::endl;

      cout << "---------------------------------------" << endl;
      cout << "CALS parameters" << endl;
      cout << "---------------------------------------" << endl;
      cout << "Tol:             " << tol << endl;
      cout << "Max Iterations:  " << max_iterations << endl;
      cout << "Buffer Size:     " << buffer_size << endl;
      cout << "Mttkrp Method:   " << mttkrp_method << endl;
      cout << "Update Method:   " << update_method << endl;
      cout << "Line Search:     " << line_search << endl;
      cout << "Line Search Int: " << line_search_interval << endl;
      cout << "CUDA:            " << cuda << endl;
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
   * @param params Parameters for the algorithm.
   *
   * @return CalsReport object, containing all data from the execution.
   */
  CalsReport cp_cals(const Tensor &X, KtensorQueue &kt_queue, CalsParams &params);
}
#endif //CALS_CALS_H
