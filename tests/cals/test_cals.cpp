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

#include "gtest/gtest.h"

#include <cmath>
#include <cals.h>
#include <random>

#include "tensor.h"
#include "als.h"

#define MODEL_DIFF_ACC 1e-08

class CalsParametersTests :public ::testing::TestWithParam<bool> {};

TEST_P(CalsParametersTests, Correctness)
{
  // parameter = GetParam()
  // You can pass a CalsParams object or a tuple of values and access them with std::get<1>(GetParam())

  vector<int> ranks;
  for (auto rank = 1; rank <= 10; rank++)
    for (auto copies = 0; copies < 300; copies++)
      ranks.push_back(rank);
//  std::sort(ranks.begin(), ranks.end());
  std::shuffle(ranks.begin(), ranks.end(), std::default_random_engine(3454));

  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::ops::MTTKRP_METHOD::AUTO;
  cals_params.max_iterations = 100;
  cals_params.tol = 1e-4;
  cals_params.buffer_size = 300;
  cals_params.line_search = true;
  cals_params.line_search_interval = 5;
  cals_params.line_search_step = 0.9;
  cals_params.cuda = GetParam();

  cals::AlsParams als_params;
  als_params.mttkrp_method = cals_params.mttkrp_method;
  als_params.max_iterations = cals_params.max_iterations;
  als_params.line_search = cals_params.line_search;
  als_params.line_search_interval = cals_params.line_search_interval;
  als_params.line_search_step = cals_params.line_search_step;
  als_params.tol = cals_params.tol;
//  als_params.cuda = cals_params.cuda;

  cals::Tensor T(5, {9, 4, 3});

  auto const modes = T.get_modes();

  vector<cals::Ktensor> ktensor_vector(ranks.size());
  auto i = 0;
  for (auto &ktensor : ktensor_vector)
  {
    ktensor = cals::Ktensor(ranks[i++], modes);
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

  vector<cals::AlsReport> reports(als_queue.size());

// Sweep over Ktensor models.
  i = 0;
  while (!als_queue.empty())
  {
    reports[i++] = cals::cp_als(T, als_queue.front(), als_params);
    als_queue.pop();
  }

  cals::cp_cals(T, cals_queue, cals_params);

  auto index = 0;
  for (auto &c: cals_input)
    EXPECT_NEAR(als_input[index++].get_approx_error(), c.get_approx_error(), MODEL_DIFF_ACC);
//  {
//    if (als_input[index++].get_approx_error() - c.get_approx_error() > MODEL_DIFF_ACC)
//        std::cerr << "Model missmatch" << std::endl;
//  }
};

INSTANTIATE_TEST_CASE_P(CalsCorrectnessTests, CalsParametersTests, ::testing::Values(false));

#if CUDA_ENABLED
INSTANTIATE_TEST_CASE_P(CalsCUDACorrectnessTests, CalsParametersTests, ::testing::Values(true));
#endif

int main(int argc, char **argv)
{
  set_threads(4);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
