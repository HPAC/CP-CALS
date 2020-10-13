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

#ifndef CALS_KTENSOR_H
#define CALS_KTENSOR_H

#include <vector>
#include <cfloat>

#include "matrix.h"
#include "cals_blas.h"

using std::vector;
using std::multiplies;

static int universal_ktensor_id = 1;

namespace cals
{
  class Ktensor
  {
    int id{-1};
    int rank{-1};
    int iters{-1};

    double approx_error{0};
    double approx_error_diff{0};

    double fit{0};
    double fit_diff{0};

    bool normalized{false};

    vector<vector<vector<bool>>> active_set;

    vector<double> lambda{};
    vector<Matrix> factors{};

    void rec_to_tensor(vector<int> &modes, int &index, double *x_data, int level, int *dim_ind);

  public:
    Ktensor() = default;

    /** Allocates a Ktensor (no initialization).
     *
     * This constructor allocates a Ktensor of a specific \p rank and sizes of \p modes. The
     * contents of the Ktensor created are not initialized. One can use the randomize member function
     * to fill in the Ktensor with random values.
     *
     * @param rank The rank of the Ktensor.
     * @param vector A vector containing the sizes of each mode of the Ktensor.
     */
    Ktensor(const int rank, const vector<int> &modes)
        : id{universal_ktensor_id++},
          rank{rank},
          active_set{vector<vector<vector<bool>>>(modes.size(), vector<vector<bool>>())},
          factors{vector<Matrix>(modes.size())}
    {
      for (auto n = 0lu; n < modes.size(); n++)
      {
        factors[n] = Matrix{modes[n], rank};
        active_set[n] = vector<vector<bool>>(modes[n], vector<bool>(rank, true));
      }

      // For each mode allocate a coefficients vector
      lambda.resize(get_rank());
    }

    Ktensor &operator=(Ktensor &&rhs) = default;

    Ktensor(Ktensor &&rhs) = default;

    Ktensor(const Ktensor &rhs)
        : id{universal_ktensor_id++},
          rank{rhs.rank},
          active_set{vector<vector<vector<bool>>>(rhs.get_n_modes(), vector<vector<bool>>())},
          lambda{vector<double>(rhs.rank)},
          factors{vector<Matrix>(rhs.get_n_modes())}
    {
      for (auto n = 0; n < rhs.get_n_modes(); n++)
      {
        factors[n] = rhs.get_factor(n);
        active_set[n] = vector<vector<bool>>(get_factor(n).get_rows(), vector<bool>(rank, true));
      }
      lambda = rhs.get_lambda();
    }

    Ktensor &operator=(const Ktensor &rhs)
    {
      if (this == &rhs) // Properly handle self assignment
        return *this;

      id = rhs.id;
      rank = rhs.rank;
      lambda = rhs.get_lambda();

      factors = vector<Matrix>(rhs.get_n_modes());
      for (auto n = 0; n < rhs.get_n_modes(); n++) factors[n] = rhs.get_factor(n);
      return *this;
    }

    ~Ktensor() = default;

    // TODO clean this. Rank should either not exist or be updated properly when adjusting the dimensions of a Ktensor.
    inline int get_rank() const noexcept
    { return factors[0].get_cols(); }

    inline int get_iters() const noexcept
    { return iters; }

    inline void set_iters(int new_iters) noexcept
    { iters = new_iters; }

    inline int get_id() const noexcept
    { return id; }

    inline double get_approx_error_diff() const noexcept
    { return approx_error_diff; }

    inline void set_approx_error_diff(double new_error_diff) noexcept
    { approx_error_diff = new_error_diff; }

    inline double get_approx_error() const noexcept
    { return approx_error; }

    inline void set_approx_error(double new_error) noexcept
    { approx_error = new_error; }

    inline double get_fit_diff() const noexcept
    { return fit_diff; }

    inline void set_fit_diff(double new_fit_diff) noexcept
    { fit_diff = new_fit_diff; }

    inline double get_fit() const noexcept
    { return fit; }

    inline void set_fit(double new_fit) noexcept
    { fit = new_fit; }

    inline int get_n_modes() const noexcept
    { return factors.size(); }

    inline Matrix const &get_last_factor() const noexcept
    { return factors.back(); }

    inline Matrix const &get_factor(const int mode) const noexcept
    { return factors.at(mode); }

    inline vector<vector<bool>> &get_active_set(const int mode) noexcept
    { return active_set.at(mode); }

    inline Matrix &get_factor(const int mode) noexcept
    { return factors.at(mode); }

    inline vector<Matrix> const &get_factors() const noexcept
    { return factors; }

    inline vector<Matrix> &get_factors() noexcept
    { return factors; }

    inline vector<double> const &get_lambda() const noexcept
    { return lambda; }

    inline vector<double> &get_lambda() noexcept
    { return lambda; }

    inline void set_factor(int index, double* data) noexcept
    {
      auto &target = get_factor(index);
      cblas_dcopy(target.get_n_elements(), data, 1, target.get_data(), 1);
    }

    inline void set_lambda(double const * data) noexcept
    {
      auto i = 0;
      for (auto &l : lambda) l = data[i++];
    }

    void print(const std::basic_string<char>& text = "Ktensor") const;

    Ktensor &attach(vector<double *> &data_ptrs, bool multi_thread = true);

    Ktensor &detach();

    Ktensor &normalize();

    Ktensor &normalize(int mode, int iteration = 1);

    Ktensor &denormalize();

    Ktensor &randomize();

    Tensor to_tensor();

#if CUDA_ENABLED
    Ktensor &cuattach(vector<double *> &cudata_ptrs);

    Ktensor &cudetach();
#endif
  };

} // namespace cals
#endif // CALS_KTENSOR_H
