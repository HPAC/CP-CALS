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
#include <random>

#include "tensor.h"
#include "als.h"

#define MODEL_DIFF_ACC 1e-09

TEST(Als, ComputeCorrectResult3D)
{
  vector<int> modes = {9, 4, 2};
  cals::Tensor X(5, modes);
  cals::Ktensor k(5, modes);

  cals::ops::MTTKRP_METHOD mttkrp_variants[] = {
      cals::ops::MTTKRP_METHOD::MTTKRP,
      cals::ops::MTTKRP_METHOD::TWOSTEP0,
      cals::ops::MTTKRP_METHOD::TWOSTEP1,
      cals::ops::MTTKRP_METHOD::AUTO
  };

  for (auto p = 0; p < 20; p++)
  {
    double errors[4] = {0.0};
    auto index = 0;
    k.randomize();

    for (auto &method : mttkrp_variants)
    {
      cals::AlsParams params;
      params.max_iterations = 100;
      params.mttkrp_method = method;
      params.line_search = true;
      params.line_search_interval = 5;
      params.line_search_step = 0.9;
      cals::Ktensor k_copy(k);

      auto report = cp_als(X, k_copy, params);

      // Get the slow error calculated by reconstructing the tensor, elementwise subtracting and norming
      auto result_tensor = k_copy.to_tensor();
      for (auto i = 0; i < X.get_n_elements(); i++) result_tensor[i] = X[i] - result_tensor[i];
      auto slow_error = result_tensor.norm();
      errors[index++] = slow_error;

      EXPECT_FALSE(std::isnan(slow_error));
      EXPECT_LT(slow_error, 50);
    }
    for (auto e : errors) EXPECT_NEAR(e, errors[0], 1e-8);
  }
}

TEST(Als, ComputeCorrectResultConstrained3D)
{
  vector<int> modes = {9, 4, 2};
  cals::Tensor X(5, modes);
  cals::Ktensor k(5, modes);

  cals::ops::MTTKRP_METHOD mttkrp_variants[] = {
      cals::ops::MTTKRP_METHOD::MTTKRP,
      cals::ops::MTTKRP_METHOD::TWOSTEP0,
      cals::ops::MTTKRP_METHOD::TWOSTEP1,
      cals::ops::MTTKRP_METHOD::AUTO
  };

  for (auto p = 0; p < 20; p++)
  {
    double errors[4] = {0.0};
    auto index = 0;
    k.randomize();

    for (auto &method : mttkrp_variants)
    {
      cals::AlsParams params;
      params.max_iterations = 100;
      params.mttkrp_method = method;
      params.update_method = cals::update::UPDATE_METHOD::NNLS;
      cals::Ktensor k_copy(k);

      auto report = cp_als(X, k_copy, params);

      // Get the slow error calculated by reconstructing the tensor, elementwise subtracting and norming
      auto result_tensor = k_copy.to_tensor();
      for (auto i = 0; i < X.get_n_elements(); i++) result_tensor[i] = X[i] - result_tensor[i];
      auto slow_error = result_tensor.norm();
      errors[index++] = slow_error;

      EXPECT_FALSE(std::isnan(slow_error));
      EXPECT_LT(slow_error, 10);
    }
    for (auto e : errors) EXPECT_NEAR(e, errors[0], 1);
  }
}

TEST(Als, ComputeCorrectResult4D)
{
  cals::Tensor X(5, {3, 3, 3, 3});
  cals::Ktensor k(7, {3, 3, 3, 3});
  k.randomize();

  cals::AlsParams params;
  params.max_iterations = 100;
  auto report = cp_als(X, k, params);

  // Get the slow error calculated by reconstructing the tensor, elementwise subtracting and norming
  auto result_tensor = k.to_tensor();
  for (auto i = 0; i < X.get_n_elements(); i++) result_tensor[i] = X[i] - result_tensor[i];
  auto slow_error = result_tensor.norm();

  EXPECT_FALSE(std::isnan(slow_error));
  EXPECT_LT(slow_error, 1e-1);
}

TEST(Als, ComputeCorrectError)
{
  cals::Tensor X(5, {9, 3, 2});
  cals::Ktensor k(5, {9, 3, 2});
  k.randomize();

  cals::AlsParams params;
  params.max_iterations = 3;
  auto report = cp_als(X, k, params);

  // Get the fast error reported by the algorithm
  auto fast_error = k.get_approx_error();

  // Get the slow error calculated by reconstructing the tensor, elementwise subtracting and norming
  auto result_tensor = k.to_tensor();
  for (auto i = 0; i < X.get_n_elements(); i++) result_tensor[i] = X[i] - result_tensor[i];
  auto slow_error = result_tensor.norm();

  EXPECT_NEAR(fast_error, slow_error, 1e-10);
}

int main(int argc, char **argv)
{
  set_threads(4);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}