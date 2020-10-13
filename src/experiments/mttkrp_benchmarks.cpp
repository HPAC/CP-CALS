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

#include "experiments/mttkrp_benchmarker.h"

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Not enough arguments. Give number of threads.";
    abort();
  }

  int num_threads = std::atoi(argv[1]); // NOLINT(cert-err34-c)

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
    }
    if (true)
    {
      auto params = cals::ops::MttkrpParams{};
      params.method = cals::ops::MTTKRP_METHOD::TWOSTEP0;
      cals::benchmarker::benchmark_mttkrp({100, 100, 100}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({200, 200, 200}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({300, 300, 300}, num_threads, params);
    }
    if (true)
    {
      auto params = cals::ops::MttkrpParams{};
      params.method = cals::ops::MTTKRP_METHOD::TWOSTEP1;
      cals::benchmarker::benchmark_mttkrp({100, 100, 100}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({200, 200, 200}, num_threads, params);
      cals::benchmarker::benchmark_mttkrp({300, 300, 300}, num_threads, params);
    }
  }
}
