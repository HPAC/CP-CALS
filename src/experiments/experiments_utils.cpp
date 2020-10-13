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

#include "experiments/experiments_utils.h"

#include <cmath>
#include <vector>
#include <iomanip>
#include <functional>

#include "als.h"
#include "cals.h"
#include "utils.h"
#include "cals_blas.h"

using cals::Ktensor;
using cals::Tensor;
using std::vector;
using std::cout;
using std::endl;


vector<cals::AlsReport> regular_als(const Tensor &X, cals::KtensorQueue &kt_queue, cals::CalsParams &params)
{
  vector<cals::AlsReport> reports(kt_queue.size());

  cals::AlsParams als_params;
  als_params.mttkrp_method = params.mttkrp_method;
  als_params.max_iterations = params.max_iterations;
  als_params.force_max_iter = params.force_max_iter;

  // Sweep over Ktensor models.
  int i = 0;
  cout << "Finished ALS:";
  while (!kt_queue.empty())
  {
    reports[i++] = cals::cp_als(X, kt_queue.front(), als_params);
    cout << " " << i;
    kt_queue.pop();
  }
  cout << endl;

  return reports;
}

cals::CalsReport concurrent_als(const Tensor &X, cals::KtensorQueue &kt_queue, cals::CalsParams &params)
{
  //
  // Configuration
  //

  // Fit the models using Simultaneous ALS.
  return cals::cp_cals(X, kt_queue, params);
}

void compare_als_cals(int tensor_rank, const vector<int> &modes, vector<int> &ranks, int num_threads,
                      cals::CalsParams &params, const std::basic_string<char>& file_suffix)
{
  Tensor X(tensor_rank, modes);
  compare_als_cals(X, ranks, num_threads, params, file_suffix);
}

void compare_als_cals(const Tensor &X, vector<int> &ranks, int num_threads, cals::CalsParams &params,
                      const std::basic_string<char>& file_suffix)
{
  set_threads(num_threads);

  auto const modes = X.get_modes();
  cout << "Modes: ";
  for (auto m : modes) cout << m << " ";
  cout << endl;

  vector<Ktensor> ktensor_vector(ranks.size());
  auto i = 0;
  for (auto &ktensor : ktensor_vector)
  {
    ktensor = Ktensor(ranks[i++], modes);
    ktensor.randomize();
  }
  auto als_input(ktensor_vector);
  auto cals_input(ktensor_vector);

  cals::KtensorQueue als_queue, cals_queue;

  for (auto p = 0lu; p < als_input.size(); p++)
  {
    als_queue.emplace(als_input[p]);
    cals_queue.emplace(cals_input[p]);
  }

  cout << "Starting CALS" << endl;
  auto cals_report = concurrent_als(X, cals_queue, params);
  cout << "Starting ALS" << endl;
  auto als_report = regular_als(X, als_queue, params);

  auto index = 0;
  auto const xnorm = X.norm();
  for (auto &k : cals_input)
  {
    auto model_norm_diff_pct = (als_input[index++].get_approx_error() - k.get_approx_error()) / xnorm;

    if (std::isnan(model_norm_diff_pct)
        || std::isnan(k.get_factor(0)[0])
        || std::isnan(k.get_factor(1)[0])
        || std::isnan(k.get_factor(2)[0]))
    {
      std::cerr << "NaN values in solutions" << std::endl;
      abort();
    }
    if (std::fabs(model_norm_diff_pct) > MODEL_DIFF_ACC)
    {
      std::cerr << "Ktensor " << index - 1 << " ALS v CALS: " << model_norm_diff_pct * 100 << " %" << endl;
//      abort();
    }
  }

  // Print results to file
  std::string str_mode;
  for (const auto &m : modes) str_mode += std::to_string(m) + "-";
  str_mode.pop_back();
  std::string str_n_threads = std::to_string(cals_report.n_threads);
  std::string str_blas_vendor = CALS_BACKEND;
  std::string dir = "../data/" + str_blas_vendor + "/";
  std::string suffix;
  if (!file_suffix.empty()) suffix = "_" + file_suffix;

  std::string als_file_name = dir + "ALS_" + str_blas_vendor + "_" + str_mode + "_" + str_n_threads + suffix + ".csv";
  als_report[0].print_header(als_file_name);
  for (auto &rm : als_report) rm.print_to_file(als_file_name);

  std::string cals_file_name = dir + "CALS_" + str_blas_vendor + "_" + str_mode + "_" + str_n_threads + suffix + ".csv";
  cals_report.print_header(cals_file_name);
  cals_report.print_to_file(cals_file_name);

  // Print all Ktensor ids, ranks, errors and iterations
  std::string ktensor_file_name = dir + "Ktensors_" + str_blas_vendor + "_" + str_mode + "_" + str_n_threads + suffix + ".csv";

  auto file = std::ofstream(ktensor_file_name, std::ios::app);
  file << "KTENSOR_ID;KTENSOR_RANK;ERROR;ITERS" << endl;
  for (auto &k : cals_input)
    file << k.get_id() << ";" << k.get_rank() << ";" << k.get_approx_error() << ";" << k.get_iters() << endl;
}

void run_cals(int tensor_rank, const vector<int> &modes, vector<int> &ranks, int num_threads,
                          cals::CalsParams &params, bool print_header, const std::basic_string<char>& file_suffix)
{
  Tensor X(tensor_rank, modes);
  run_cals(X, ranks, num_threads, params, print_header, file_suffix);
}

void run_cals(const Tensor &X, vector<int> &ranks, int num_threads, cals::CalsParams &params, bool print_header,
                          const std::basic_string<char>& file_suffix)
{
  set_threads(num_threads);

  cout << "============================" << endl;
  cout << "START Single CALS experiment" << endl;
  cout << "Modes: ";
  auto const modes = X.get_modes();
  for (auto m : modes) cout << m << " ";
  cout << endl;
  cout << "Threads: " << num_threads << endl;

  vector<Ktensor> ktensor_vector(ranks.size());
  auto i = 0;
  for (auto &ktensor : ktensor_vector)
  {
    ktensor = Ktensor(ranks[i++], modes);
    ktensor.randomize();
  }
  auto cals_input(ktensor_vector);

  cals::KtensorQueue cals_queue;

  for (auto & p : cals_input) cals_queue.emplace(p);

  auto cals_report = concurrent_als(X, cals_queue, params);

  // Print results to file
  std::string str_mode;
  for (const auto &m : modes) str_mode += std::to_string(m) + "-";
  str_mode.pop_back();
  std::string str_n_threads = std::to_string(cals_report.n_threads);
  std::string str_blas_vendor = CALS_BACKEND;
  std::string dir = "../data/" + str_blas_vendor + "/";
  std::string str_cuda = (params.cuda) ? "CUDA_" : "";

  std::string cals_file_name = dir + "CALS_" + str_cuda + str_blas_vendor + "_" + str_mode + "_" + str_n_threads;
  if (file_suffix.empty())
    cals_file_name += ".csv";
  else
    cals_file_name += "_" + file_suffix + ".csv";

  cals_report.output_file_name = cals_file_name;

  if (print_header)
    cals_report.print_header(cals_file_name);
  cals_report.print_to_file(cals_file_name);

  cout << cals_report.output_file_name << endl;
  cout << "END Single CALS experiment" << endl;
  cout << "==========================" << endl;

}
