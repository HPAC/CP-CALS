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

#include "tensor.h"
#include "ktensor.h"

#include <iostream>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>

using std::accumulate;
using std::make_unique;
using std::multiplies;
using std::cout;
using std::endl;

namespace cals
{
  Tensor::Tensor(const vector<int> &modes)
      : n_elements{accumulate(modes.cbegin(), modes.cend(), static_cast<int>(1), multiplies<>())},
        max_n_elements{n_elements},
        modes{modes}
  {
    double *host_ptr = (double *) operator new(n_elements * sizeof(double));
    data_up = unique_ptr<double, Dopnew>(host_ptr, Dopnew());
    data = data_up.get();
  }

  // Constructor for Matlab, to not copy the tensor twice.
  Tensor::Tensor(const vector<int> &modes, double *view_data)
      : n_elements{accumulate(modes.cbegin(), modes.cend(), static_cast<int>(1), multiplies<>())},
        max_n_elements{n_elements},
        modes{modes},
        data_up{nullptr},
        data{view_data}
  {}

  Tensor::Tensor(const std::string &file_name)
  {
    vector<int> read_modes;

    auto file = std::ifstream(file_name, std::ios::in);

    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);

    while (ss.good())
    {
      std::string substr;
      getline(ss, substr, ' ');
      read_modes.push_back(std::atoi(substr.c_str()));
    }

    n_elements = accumulate(read_modes.cbegin(), read_modes.cend(), static_cast<int>(1), multiplies<>());
    max_n_elements = n_elements;
    modes = read_modes;

    double *host_ptr = (double *) operator new(n_elements * sizeof(double));
    data_up = unique_ptr<double, Dopnew>(host_ptr, Dopnew());
    data = data_up.get();

    double val = 0;
    int index = 0;
    while (file >> val)
      data[index++] = val;

    assert(index == n_elements);
  }


  Tensor::Tensor(const int mode0, const int mode1, double *view_data)
      : n_elements{mode0 * mode1},
        max_n_elements{n_elements},
        modes{vector<int>{mode0, mode1}}
  {
    if (view_data == nullptr)
    {
      double *host_ptr = (double *) operator new(n_elements * sizeof(double));
      data_up = unique_ptr<double, Dopnew>(host_ptr, Dopnew());
      data = data_up.get();
    } else
    {
      data_up = nullptr;
      data = view_data;
    }
  }

  Tensor::Tensor(const int rank, const vector<int> &modes)
  {
    Ktensor P(rank, modes);
    P.randomize();

    *this = P.to_tensor();
    this->rank = rank;
  }

  Tensor::Tensor(const Tensor &rhs)
      : n_elements{rhs.n_elements},
        modes{rhs.modes}
  {
    if (!rhs.is_view())  // If copying views, don't allocate new memory and don't copy over data
    {
      double *host_ptr = (double *) operator new(n_elements * sizeof(double));
      data_up = unique_ptr<double, Dopnew>(host_ptr, Dopnew());
      data = data_up.get();
      cblas_dcopy(n_elements, rhs.get_data(), 1, get_data(), 1);
    } else
    {
      data_up = nullptr;
      data = rhs.get_data();
    }
    cout << "WARNING: Performed copy constructor =============================" << endl;
  }

  Tensor &Tensor::operator=(const Tensor &rhs)
  {
    if (this == &rhs) // Properly handle self assignment
      return *this;

    n_elements = rhs.n_elements;
    modes = rhs.modes;

    if (!rhs.is_view())  // If copying views, don't allocate new memory and don't copy over data
    {
      double *host_ptr = (double *) operator new(n_elements * sizeof(double));
      data_up = unique_ptr<double, Dopnew>(host_ptr, Dopnew());
      data = data_up.get();
      cblas_dcopy(n_elements, rhs.get_data(), 1, get_data(), 1);
    } else
    {
      data_up = nullptr;
      data = rhs.get_data();
    }

    return *this;
  }

  Tensor &Tensor::randomize()
  {
    std::random_device device;
//    std::srand(0);
    std::mt19937 generator(device());
//    std::mt19937 generator(rand());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    fill([&dist, &generator]() -> double
         { return dist(generator); });
    return *this;
  }

  Tensor &Tensor::zero()
  {
    fill([]() -> double
         { return static_cast<double>(0.0); });
    return *this;
  }

  Tensor &Tensor::fill(const std::function<double()> &&generator)
  {
    for (auto i = 0; i < n_elements; i++) data[i] = generator();
    return *this;
  }

  Unfolding Tensor::implicit_unfold(const int mode) const
  {
    Unfolding unfolding{};

    auto prod_before = [this](int mode) -> int
    {
      int prod = 1;
      for (auto n = 0; n < mode; n++)
        prod *= modes[n];
      return prod;
    };
    auto prod_after = [this](int mode) -> int
    {
      int prod = 1;
      for (auto n = mode + 1; n < get_n_modes(); n++)
        prod *= modes[n];
      return prod;
    };

    if (mode == 0)
    {
      unfolding.n_blocks = 1;
      unfolding.block_offset = 0;
      unfolding.rows = modes[mode];
      unfolding.cols = prod_after(mode);
      unfolding.stride = modes[mode];
    } else if (mode == get_n_modes() - 1)
    {
      unfolding.n_blocks = 1;
      unfolding.block_offset = 0;
      unfolding.rows = modes[mode];
      unfolding.cols = prod_before(mode);
      unfolding.stride = prod_before(mode);
    } else //(mode > 0 && mode < n_dims - 1)
    {
      unfolding.n_blocks = prod_after(mode);
      unfolding.block_offset = prod_before(mode) * modes[mode]; // include mode
      unfolding.rows = modes[mode];
      unfolding.cols = prod_before(mode);
      unfolding.stride = prod_before(mode);
    }
    return unfolding;
  }

  void Tensor::print(const std::basic_string<char>& text) const
  {
    cout << "----------------------------------------" << endl;
    cout << "Modes: ";
    for (auto const &m : modes) cout << m << " ";
    cout << endl;
    cout << "data = [ ";
    for (auto i = 0; i < n_elements; i++) cout << get_data()[i] << "  ";
    cout << "]" << endl;
    cout << "----------------------------------------" << endl;
  }


} // namespace cals
