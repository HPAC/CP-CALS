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

#ifndef CALS_MATRIX_H
#define CALS_MATRIX_H

#include "tensor.h"

namespace cals
{
  class Matrix : public Tensor
  {
    int rows{};
    int cols{};
    int row_stride{};
    int col_stride{};

  public:
    // Constructors and Destructor
    Matrix() = default;

    ~Matrix() = default;

    Matrix(int dim0, int dim1);

    Matrix(int dim0, int dim1, double *view_data);

    Matrix(int dim0, int dim1, int row_stride, int col_stride, double *view_data = nullptr);

    // Move and Copy Constructors
    Matrix(Matrix &&rhs) = default;

    Matrix &operator=(Matrix &&rhs) = default;

    Matrix(const Matrix &rhs) = default;

    Matrix &operator=(const Matrix &rhs) = default;

    // Getters
    inline int get_rows() const noexcept
    { return rows; };

    inline int get_cols() const noexcept
    { return cols; };

    inline int get_row_stride() const noexcept
    { return row_stride; };

    inline int get_col_stride() const noexcept
    { return col_stride; };

    inline double &at(int row, int col)
    { return get_data()[row * row_stride + col * col_stride]; };

    inline double at(int row, int col) const noexcept
    { return get_data()[row * row_stride + col * col_stride]; };

    Matrix &resize(int new_rows, int new_cols) noexcept
    {
      vector<int> modes = {new_rows, new_cols};
      Tensor::resize(new_rows * new_cols, modes);
      rows = new_rows;
      cols = new_cols;
      col_stride = new_rows;
//      row_stride = 1;
      return *this;
    };

    Matrix &hadamard(const Matrix &mat);

    Matrix &read(const Matrix &m);

    inline void attach(double *data)
    { set_data(data); }

    inline void detach()
    { reset_data(); }

    void print(const std::basic_string<char>& text = "Matrix") const;

    void info() const;

    double one_norm() const
    {
      auto max = -DBL_MAX;
      for (auto col = 0; col < get_cols(); col++)
      {
        auto one_norm = cblas_dasum(get_rows(), get_data() + col * get_col_stride(), 1);
        if (one_norm > max) max = one_norm;
      }
      return max;
    };
#if CUDA_ENABLED
    inline void cuattach(double *cudata)
    { set_cudata(cudata); }

    inline void cudetach()
    { reset_cudata(); }
#endif

  };
} // namespace cals

#endif // CALS_MATRIX_H
