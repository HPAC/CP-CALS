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

#include <iostream>
#include <random>

#include "cals.h"
#include "als.h"
#include "cals_blas.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  /////////////////////////////////////////////////////////////////////////////////
  // Set the number of threads.
  /////////////////////////////////////////////////////////////////////////////////
  int num_threads = 4;
  set_threads(num_threads);

  /////////////////////////////////////////////////////////////////////////////////
  // Create target Tensor.
  /////////////////////////////////////////////////////////////////////////////////
  vector<int> modes {210, 210, 210};
  cals::Tensor X(modes);
  X.randomize();

  /////////////////////////////////////////////////////////////////////////////////
  // Create input Ktensors.
  /////////////////////////////////////////////////////////////////////////////////

  // Create a vector with all the ranks we want to run CP-ALS for.
  vector<int> ranks;
  for (auto rank = 1; rank <= 5; rank++)  // Ranks from 1 up to (and including) 5
    for (auto cp = 0; cp < 10; cp++)       // 10 copies of each rank
      ranks.push_back(rank);

  // Create a vector containing the Ktensors we want to fit to the target Tensor.
  vector<cals::Ktensor> cals_input(ranks.size());
  auto i = 0;
  for (auto &ktensor : cals_input)
  {
    ktensor = cals::Ktensor(ranks[i++], modes);
    ktensor.randomize();
  }
  cout << "Created " << ranks.size() << " Ktensors." << endl;
  cout << "Ktensor ranks: ";
  for (auto &r : ranks) cout << r << " ";
  cout << endl;

  // Copy the same Ktensors to be used as input to regular ALS.
  auto als_input(cals_input);

  /////////////////////////////////////////////////////////////////////////////////
  // Run CALS.
  /////////////////////////////////////////////////////////////////////////////////
  cout << "Running CALS..." << endl;

  // Create the input to CALS, as a Queue of references to the Ktensors created.
  cals::KtensorQueue cals_queue;
  for (auto &p : cals_input)
    cals_queue.emplace(p);

  // Set the CALS parameters (if necessary).
  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
  cals_params.force_max_iter = true;
  cals_params.buffer_size = std::accumulate(ranks.cbegin(), ranks.cend(), 0);
  cals_params.max_iterations = 50;
  cals_params.tol = 1e-4;
  cals_params.cuda = true;

  cals::Timer cals_timer;
  cals_timer.start();
  auto cals_report = cp_cals(X, cals_queue, cals_params);
  cals_timer.stop();

  /////////////////////////////////////////////////////////////////////////////////
  // Run a for-loop of regular ALS for comparison.
  /////////////////////////////////////////////////////////////////////////////////
  cout << "Running ALS..." << endl;

  // Set the ALS parameters (if necessary).
  cals::AlsParams als_params;
  als_params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
  als_params.force_max_iter = true;
  als_params.max_iterations = 50;
  als_params.tol = 1e-4;

  cals::Timer als_timer;
  als_timer.start();
  for (auto &kt : als_input)
    auto als_report = cp_als(X, kt, als_params);
  als_timer.stop();

  /////////////////////////////////////////////////////////////////////////////////
  // Print timings.
  /////////////////////////////////////////////////////////////////////////////////

  cout << "======================================================================" << endl;
  cout << "ALS time: " << als_timer.get_time() << endl;
  cout << "CALS time: " << cals_timer.get_time() << endl;
  cout << "Speedup: " << als_timer.get_time() / cals_timer.get_time() << endl;
}
