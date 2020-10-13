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

#include <algorithm>
#include <iostream>
#include <random>

#include "experiments/experiments_utils.h"
#include "experiments/mttkrp_benchmarker.h"

void generate_ranks(vector<int> &ranks, int min, int max, int copies);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Not enough arguments. Give number of threads.";
    abort();
  }

  int num_threads = std::atoi(argv[1]); // NOLINT(cert-err34-c)

  // Mock run to warmup everything.
  if (true)
  {
    vector<int> ranks;
    generate_ranks(ranks, 1, 20, 20);
    std::sort(ranks.begin(), ranks.end());

    cals::CalsParams params;
    params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
    params.max_iterations = 50; // TTB default
    params.buffer_size = 4200;
    params.tol = 1e-4; // TTB default
    params.cuda = false;

    std::string file_name = "../data/fluorescence_cancer_UD.txt";
    cals::Tensor T(file_name);
    compare_als_cals(T, ranks, num_threads, params);
  }

  // Real Dataset (Fluorescence Cancer) comparison of ALS and CALS
  if (true)
  {
    vector<int> ranks;
    generate_ranks(ranks, 1, 20, 20);
    std::sort(ranks.begin(), ranks.end());

    cals::CalsParams params;
    params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
    params.max_iterations = 50; // TTB default
    params.buffer_size = 4200;
    params.tol = 1e-4; // TTB default
    params.cuda = false;

    std::string file_name = "../data/fluorescence_cancer_UD.txt";
    cals::Tensor T(file_name);
    compare_als_cals(T, ranks, num_threads, params);
  }

  // Synthetic Dataset comparison of ALS and CALS
  if (true)
  {
    vector<int> ranks;
    generate_ranks(ranks, 1, 20, 20);
    std::sort(ranks.begin(), ranks.end());

    cals::CalsParams params;
    params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
    params.force_max_iter = true;
    params.max_iterations = 50;
    params.tol = 1e-4;
    params.cuda = false;

    if (num_threads == 1)
      params.buffer_size = 90;
    else
      params.buffer_size = 4200;
    compare_als_cals(25, {100, 100, 100}, ranks, num_threads, params);
    params.buffer_size = 4200;
    compare_als_cals(25, {200, 200, 200}, ranks, num_threads, params);
    compare_als_cals(25, {300, 300, 300}, ranks, num_threads, params);
  }

  // Speedup plot ALS v CALS
  if (true)
  {
    cals::CalsParams params;
    params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
    params.force_max_iter = true;
    params.max_iterations = 50;
    params.tol = 1e-4;
    params.cuda = false;

    for (int rank = 1; rank <= 20; rank++)
    {
      vector<int> ranks;
      generate_ranks(ranks, rank, rank, 20);

      if (num_threads == 1)
        params.buffer_size = 90;
      else
        params.buffer_size = 20 * rank;
      compare_als_cals(25, {100, 100, 100}, ranks, num_threads, params, "speedup_" + std::to_string(rank));
      params.buffer_size = 20 * rank;
      compare_als_cals(25, {200, 200, 200}, ranks, num_threads, params, "speedup_" + std::to_string(rank));
      compare_als_cals(25, {300, 300, 300}, ranks, num_threads, params, "speedup_" + std::to_string(rank));
    }
  }

  // MTTKRP Benchmark for sweeping intensity
  if (true)
  {
    if (true)
    {
      auto params = cals::ops::MttkrpParams{};
      params.method = cals::ops::MTTKRP_METHOD::MTTKRP;
      cals::benchmarker::benchmark_mttkrp({100, 100, 100}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({200, 200, 200}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({300, 300, 300}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({299, 301, 41}, num_threads, params);
    }
    if (true)
    {
      auto params = cals::ops::MttkrpParams{};
      params.method = cals::ops::MTTKRP_METHOD::TWOSTEP0;
      cals::benchmarker::benchmark_mttkrp({100, 100, 100}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({200, 200, 200}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({300, 300, 300}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({299, 301, 41}, num_threads, params);
    }
    if (true)
    {
      auto params = cals::ops::MttkrpParams{};
      params.method = cals::ops::MTTKRP_METHOD::TWOSTEP1;
      cals::benchmarker::benchmark_mttkrp({100, 100, 100}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({200, 200, 200}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({300, 300, 300}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({299, 301, 41}, num_threads, params);
    }
  }
}

void generate_ranks(vector<int> &ranks, int min, int max, int copies)
{
  for (auto rank = min; rank <= max; rank++)
    for (auto cp = 0; cp < copies; cp++)
      ranks.push_back(rank);
}
