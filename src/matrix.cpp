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

#include "matrix.h"

#include <iostream>

namespace cals
{
  Matrix::Matrix(const int dim0, const int dim1)
      : Tensor{dim0, dim1},
        rows{dim0},
        cols{dim1},
        row_stride{1}, // All matrices stored in col major format by default
        col_stride{dim0}
  {}

  Matrix::Matrix(int dim0, int dim1, double *view_data) : Matrix(dim0, dim1, 1, dim0, view_data)
  {}

  Matrix::Matrix(int dim0, int dim1, int row_stride, int col_stride, double *view_data)
      : Tensor{dim0, dim1, view_data},
        rows{dim0},
        cols{dim1},
        row_stride{row_stride},
        col_stride{col_stride}
  {}

  Matrix &Matrix::hadamard(const Matrix &mat)
  {
    assert(!is_view() && !mat.is_view());

    // TODO improve with element-wise multiplication in MKL
    for (auto i = 0; i < get_n_elements(); i++)
      get_data()[i] *= mat[i];
    return *this;
  }

  void Matrix::print(const std::basic_string<char>& text) const
  {
    using std::cout;
    using std::endl;

    cout << "----------------------------------------" << endl;
    cout << text << endl;
    cout << "----------------------------------------" << endl;
    cout << "Rows: " << rows << ", Cols: " << cols << endl;
    cout << "[ " << endl;
    for (auto row = 0; row < rows; row++)
    {
      for (auto col = 0; col < cols; col++)
        cout << "  " << get_data()[row * row_stride + col * col_stride] << "  ";
      cout << std::endl;
    }
    cout << "]" << std::endl;
    cout << "----------------------------------------" << endl;
  }

  Matrix &Matrix::read(const Matrix &mat)
  {
    if (this == &mat) // Properly handle self assignment
      return *this;

    assert(get_n_elements() == mat.get_n_elements());
    assert(get_rows() == mat.get_rows() && get_cols() == mat.get_cols());

    if (!is_view() && !mat.is_view())
      cblas_dcopy(get_n_elements(), mat.get_data(), 1, get_data(), 1);
    else
      for (auto row = 0; row < mat.get_rows(); row++)
        for (auto col = 0; col < mat.get_cols(); col++)
          get_data()[get_col_stride() * col + get_row_stride() * row] =
              mat.get_data()[mat.get_col_stride() * col + mat.get_row_stride() * row];
    return *this;
  }

  void Matrix::info() const
  {
    std::cout << "nRows: " << get_rows() << ", nCols: " << get_cols() << ", nElements: " << get_n_elements()
              << ", maxNElements: " << get_max_n_elements() << std::endl;
  }


} // namespace cals
