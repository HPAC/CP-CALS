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

#include "ktensor.h"

#include <random>
#include <iostream>

#if CUDA_ENABLED
#include "cuda_utils.h"
#endif

namespace cals
{
  Ktensor &Ktensor::randomize()
  {
    for (auto &f : factors) f.randomize();
    normalize();
    return *this;
  }

  void Ktensor::rec_to_tensor(vector<int> &modes, int &index, double *const x_data, int const level, int *const dim_ind)
  {
    if (level >= 1)
      for (auto i = 0; i < modes[level - 1]; i++)
      {
        dim_ind[level - 1] = i;
        rec_to_tensor(modes, index, x_data, level - 1, dim_ind);
      }
    else if (level == 0)
    {
      double s = 0;
      for (auto r = 0; r < rank; r++)
      {
        double m = 1.0;
        for (auto f = 0; f < get_n_modes(); f++)
          m *= factors[f].at(dim_ind[f], r);
        s += lambda[r] * m;
      }
      x_data[index++] = s;
    }
  }

  Tensor Ktensor::to_tensor()
  {
    vector<int> modes(get_n_modes());
    for (auto i = 0; i < get_n_modes(); i++) modes[i] = factors[i].get_rows();

    Tensor X(modes);
    int index = 0;
    double *x_data = X.get_data();
    auto *dim_ind = new int[get_n_modes()];
    rec_to_tensor(modes, index, x_data, get_n_modes(), dim_ind);
    delete[] dim_ind;
    return X;
  }

  Ktensor &Ktensor::normalize(int mode, int iteration)
  {
#pragma omp parallel for // NOLINT(openmp-use-default-none)
    for (auto col = 0; col < factors[mode].get_cols(); col++)
    {
      auto &factor = factors[mode];
      auto *data_ptr = factor.get_data() + col * factor.get_col_stride();

      if (iteration == 1)
        lambda[col] = cblas_dnrm2(factor.get_rows(), data_ptr, 1);
      else
      {
        auto index = cblas_idamax(factor.get_rows(), data_ptr, 1);
        assert(index >= 0 && index < factor.get_rows());
        lambda[col] = data_ptr[index];
      }
      if (lambda[col] != 0)
        cblas_dscal(factor.get_rows(), 1 / lambda[col], data_ptr, 1);
    }
    return *this;
  }

  Ktensor &Ktensor::normalize()
  {
    for (auto &l : lambda) l = 1.0;

    for (auto n = 0; n < get_n_modes(); n++)
    {
      auto &factor = factors[n];
      for (auto col = 0; col < get_rank(); col++)
      {
        auto coeff = cblas_dnrm2(factor.get_rows(), factor.get_data() + col * factor.get_col_stride(), 1);
        cblas_dscal(factor.get_rows(), 1 / coeff, factor.get_data() + col * factor.get_col_stride(), 1);
        lambda[col] *= coeff;
      }
    }
    normalized = true;
    return *this;
  }

  Ktensor &Ktensor::denormalize()
  {
    auto &factor = factors[0];
    for (auto col = 0; col < get_rank(); col++)
      cblas_dscal(factor.get_rows(), lambda[col], factor.get_data() + col * factor.get_col_stride(), 1);
    normalized = false;
    return *this;
  }

  Ktensor &Ktensor::attach(vector<double *> &data_ptrs, bool multi_thread)
  {
    assert(data_ptrs.size() == get_n_modes());

    auto th = get_threads();
    if (!multi_thread)
      set_threads(1);

    auto index = 0;
    for (auto &f : factors)
    {
      Matrix(f.get_rows(), f.get_cols(), data_ptrs[index]).copy(f);
      f.attach(data_ptrs[index++]);
    }

    if (!multi_thread)
      set_threads(th);
    return *this;
  }

  Ktensor &Ktensor::detach()
  {
    for (auto &f : factors)
    {
      auto *old_data = f.get_data();
      f.detach();
      f.copy(Matrix(f.get_rows(), f.get_cols(), old_data));
      Matrix(f.get_rows(), f.get_cols(), old_data).zero();
    }
    return *this;
  }

  void Ktensor::print(const std::basic_string<char>& text) const
  {
    using std::cout;
    using std::endl;

    cout << "----------------------------------------" << endl;
    cout << "Ktensor:";
    cout << "----------------------------------------" << endl;

    cout << "Rank: " << get_rank() << endl;
    cout << "Num Modes: " << get_n_modes() << endl;
    cout << "Modes: [ ";
    for (const auto &f : get_factors()) cout << f.get_rows() << " ";
    cout << "]" << endl;

    cout << "Weights: [";
    for (const auto &l : lambda) cout << l << " ";
    cout << " ] " << endl;

    std::string txt = "factor";
    for (auto const &f : get_factors()) f.print(txt);

    cout << "----------------------------------------" << endl;
  }

#if CUDA_ENABLED
  Ktensor &Ktensor::cuattach(vector<double *> &cudata_ptrs)
  {
    assert(cudata_ptrs.size() == get_n_modes());

    auto index = 0;
    for (auto &f : factors)
    {
      f.cuattach(cudata_ptrs[index++]);
      f.send_to_device_async(custream);
    }

    return *this;
  }

  Ktensor &Ktensor::cudetach()
  {
    for (auto &f : factors)
    {
      cudaMemsetAsync(f.get_cudata(), 0, f.get_n_elements(), custream);
      f.cudetach();
    }
    return *this;
  }
#endif

} // namespace cals
