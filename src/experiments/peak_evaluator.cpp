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

#include <cmath>
#include <numeric>
#include <iostream>

#include "matrix.h"
#include "timer.h"
#include "cals_blas.h"

const int iterations = 10;

using std::cout;
using std::endl;
using std::cerr;

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cerr << "Not enough arguments." << endl;
    cerr << "Give number of threads, AVX peak freq (in MHz) for those threads, Flops/Cycle and size of GEMM."
         << endl;
    cerr << "Example: ./Evaluator 24 2000 32 1000"
         << endl
         << "for 24 threads, 2000 MHz AVX peak freq for 24 threads, 32 Flops/Cycle and matrix sizes 1000x1000."
         << endl;
    abort();
  }

  typedef long long unsigned int lint;

  const lint num_threads = std::atoi(argv[1]); // NOLINT(cert-err34-c)
  set_threads(num_threads);
  lint freq = std::atoi(argv[2]); // NOLINT(cert-err34-c)
  freq *= 1000;
  freq *= 1000;
  const lint fpc = std::atoi(argv[3]); // NOLINT(cert-err34-c)
  const lint tpp = num_threads * freq * fpc;

  lint size = std::atoi(argv[4]);  // NOLINT(cert-err34-c)
  cals::Matrix A(size, size), B(size, size), C(size, size);
  cals::Matrix cache(3000, 3000);
  vector<cals::Timer> vec_t(iterations, cals::Timer());

  A.randomize();
  B.randomize();
  C.randomize();
  const lint flops = 2llu * A.get_rows() * A.get_cols() * B.get_cols() + 2llu * C.get_rows() * C.get_cols();

  for (int i = 0; i < iterations; i++)
  {
    cache.randomize();
    auto &t = vec_t[i];
    t.start();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.get_rows(), B.get_cols(), A.get_cols(),
                1.0, A.get_data(), A.get_col_stride(), B.get_data(), B.get_col_stride(),
                0.0, C.get_data(), C.get_col_stride());
    t.stop();
  }

  vec_t.erase(vec_t.begin());

  cout << "Timings: " << endl;
  for (auto &i : vec_t)
    cout << i.get_time() << " ";
  cout << endl;

  std::vector<double> times(vec_t.size());
  for (auto i = 0lu; i < vec_t.size(); i++)
    times[i] = 1.0 * flops / vec_t[i].get_time() / tpp;

  auto sum = std::accumulate(times.begin(), times.end(), 0.0);
  auto mean = sum / times.size();
  double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / times.size() - mean * mean);
  auto max_t = *std::max_element(times.begin(), times.end());
  auto min_t = *std::min_element(times.begin(), times.end());

  std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
  auto median = times[times.size() / 2];

  cout << endl;
  cout << "=============================================================" << endl;
  cout << "=============================================================" << endl;
  cout << "FLOPS:            " << flops << endl;
  cout << "-------------------------------------------------------------" << endl
       << "Min Performance:        " << min_t << endl
       << "Max Performance:        " << max_t << endl
       << "Mean Performance:       " << mean << endl
       << "Median Performance:     " << median << endl
       << "Stdev Performance:      " << stdev << endl;
  cout << "-------------------------------------------------------------" << endl;
  cout << "Freq:    " << freq << endl;
  cout << "FPC:     " << fpc << endl;
  cout << "Size:    " << size << endl;
  cout << "Threads: " << num_threads << endl;
  cout << "{'mean': " << mean << ", 'median': " << median << ", 'std': " << stdev << "}" << endl;
  cout << "=============================================================" << endl;
  cout << "=============================================================" << endl;
  cout << endl;
}
