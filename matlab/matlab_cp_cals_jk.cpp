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
//
//  ***************************************************************************************************
//
//  This file includes modified versions of code found in Genten,
//  which is covered by the following license:
//
//      Genten: Software for Generalized Tensor Decompositions
//      by Sandia National Laboratories
//
//  Sandia National Laboratories is a multimission laboratory managed
//  and operated by National Technology and Engineering Solutions of Sandia,
//  LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
//  U.S. Department of Energy's National Nuclear Security Administration under
//  contract DE-NA0003525.
//
//  Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
//  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
//  Government retains certain rights in this software.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//  1. Redistributions of source code must retain the above copyright
//  notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//  notice, this list of conditions and the following disclaimer in the
//  documentation and/or other materials provided with the distribution.
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
//
// ***************************************************************************************************
//@HEADER

#include "cals.h"
#include "matlab.h"
#include "matlab_parsing.h"
#include "timer.h"
#include <execution>
#include <rectangular_lsap/rectangular_lsap.h>

using cals::Ktensor;
using cals::Tensor;
using std::cout;
using std::endl;

extern "C" {

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  try {
    if (nrhs < 2) {
      cout << "Expected at least 2 command line arguments" << endl;
      return;
    }

    bool debug = false;
    DEBUG(debug = true;)

    // Parse inputs
    auto args = mxBuildArgList(nrhs, 2, prhs);
    cals::CalsParams cals_params;
    cals::matlab::parsing::parse(cals_params, args);
    cals_params.print();
    // TODO Implement the check_print_unused_args function
    //    if (cals_params.check_and_print_unused_args(args, cout))
    //    {
    //      cals_params.print_help(cout);
    //      throw std::string("Invalid command line arguments.");
    //    }

    // Get tensor
    Tensor tensor = mxGetTensor(prhs[0], debug);

    // Get input init ktensors
    const mxArray *arg_ktensors = prhs[1];
    dim_t n_arg_ktensors = mxGetNumberOfElements(arg_ktensors);
    vector<Ktensor> init_ktensors(n_arg_ktensors);

    if (!mxIsCell(arg_ktensors))
      throw std::string("Invalid type for initial ktensors (expecting cell array of ktensors).");

    auto idx = 0;
    for (auto &ktensor : init_ktensors)
      ktensor = Ktensor(mxGetKtensor(mxGetCell(arg_ktensors, (mwIndex)idx++), debug));

    auto fitted_ktensors(init_ktensors);

    for (auto &ktensor : fitted_ktensors) {
      ktensor.denormalize();
      ktensor.normalize();
    }

    vector<vector<Ktensor>> cals_jk_input(n_arg_ktensors);
    dim_t tensor_mode_0 = tensor.get_modes()[0];
    idx = 0;
    for (auto &ktensor : fitted_ktensors) {
      cals_jk_input.reserve(n_arg_ktensors * tensor_mode_0);
      cals::utils::generate_jk_ktensors(ktensor, cals_jk_input[idx++]);
    }

    // Create a queue of references to be fed to cp_cals
    cals::KtensorQueue cals_queue;
    for (auto &k : cals_jk_input)
      for (auto &m : k)
        cals_queue.emplace(m);

    // Call driver
    cout << "OpenMP threads: " << get_threads() << endl;
    auto report = cals::cp_cals(tensor, cals_queue, cals_params);

    for (auto &k : cals_jk_input)
      for (auto &m : k) {
        m.set_jk_fiber(0.0);
        m.denormalize();
        m.normalize();
        m.set_jk_fiber(NAN);
      }

    for (dim_t i = 0; i < fitted_ktensors.size(); i++) {
      auto &kt = fitted_ktensors[i];
      auto &jk_input = cals_jk_input[i];
      auto &Bov = kt.get_factor(1);
      auto &Cov = kt.get_factor(2);

      for (dim_t m = 0; m < tensor.get_modes()[0]; m++) {
        auto &kt_jk = jk_input[m];

        auto const comp = kt.get_components();

        auto M = cals::Matrix(comp, comp);
        auto Mt = cals::Matrix(comp, comp);

        auto &Bm = kt_jk.get_factor(1);
        auto &Cm = kt_jk.get_factor(2);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, comp, tensor.get_modes()[1], 1.0, Bov.get_data(),
                    Bov.get_col_stride(), Bm.get_data(), Bm.get_col_stride(), 0.0, M.get_data(), M.get_col_stride());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, comp, tensor.get_modes()[2], 1.0, Cov.get_data(),
                    Cov.get_col_stride(), Cm.get_data(), Cm.get_col_stride(), 0.0, Mt.get_data(), Mt.get_col_stride());
        for (dim_t ii = 0; ii < M.get_n_elements(); ii++)
          M[ii] += Mt[ii];

        vector<int64_t> init_v(comp);
        vector<int64_t> solved_v(comp);

        solve_rectangular_linear_sum_assignment(comp, comp, M.get_data(), true, init_v.data(), solved_v.data());

        for (dim_t mode = 0; mode < kt.get_n_modes(); mode++) {
          auto &factor = kt_jk.get_factor(mode);
          auto copy_factor = cals::Matrix(factor.get_rows(), factor.get_cols());
          copy_factor.copy(factor);
          auto const stride = factor.get_col_stride();

          auto curr_col_id = 0;
          for (auto &swap_col_id : solved_v) {
            if (swap_col_id != curr_col_id) {
              cals::Matrix(factor.get_rows(), 1, factor.get_data() + curr_col_id * stride)
                  .copy(cals::Matrix(copy_factor.get_rows(), 1, copy_factor.get_data() + swap_col_id * stride));
            }
            curr_col_id++;
          }
        }
      }
    }

    // Return results
    // Result structure
    // U ---> init_ktensor_0 ---> {init_ktensor}
    //  |                   |---> {fitted_ktensor}
    //  |                   |---> {JK_fitted_ktensors} --- not fixed for scale etc.
    //  |                   |---> {stdB, stdC} --- placeholder
    //  |---> init_ktensor_1 ---> {init_ktensor}
    //  |                   |---> {fitted_ktensor}
    //  |                   |---> {JK_fitted_ktensors} --- not fixed for scale etc.
    //  |                   |---> {stdB, stdC} --- placeholder
    auto const fields_per_output = 3;
    mxArray *cell_array_ptr = mxCreateCellMatrix((mwSize)n_arg_ktensors, (mwSize)1);
    for (int k = 0; k < n_arg_ktensors; k++) {
      mxArray *k_array_ptr = mxCreateCellMatrix((mwSize)fields_per_output, (mwSize)1);
      for (int f = 0; f < fields_per_output; f++) {
        mxArray *mat_ptr = nullptr;
        if (f == 0) // fitted_ktensors
          mat_ptr = mxSetKtensor(fitted_ktensors[k]);
        else if (f == 1) // JK_fitted_ktensors
          mat_ptr = mxSetKtensor(cals::utils::concatenate_ktensors(cals_jk_input[k]));
        else if (f == 2) // stdB, stdC
          mat_ptr = mxCreateCellMatrix((mwSize)tensor.get_n_modes(), (mwSize)1);
        mxSetCell(k_array_ptr, (mwIndex)f, mat_ptr);
      }
      mxSetCell(cell_array_ptr, (mwIndex)k, k_array_ptr);
    }
    plhs[0] = cell_array_ptr;

    cout << "Done." << endl;
  } catch (std::string &sExc) {
    cout << "Call to CALS threw an exception:" << endl;
    cout << "  " << sExc << endl;
  }
}
}
