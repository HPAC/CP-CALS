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

#ifndef CALS_TENSOR_H
#define CALS_TENSOR_H

#include <cassert>
#include <functional>
#include <memory>
#include <cfloat>

#include "definitions.h"
#include "cals_blas.h"

#if CUDA_ENABLED
#include "cuda_utils.h"
#endif

using std::vector;
using std::unique_ptr;
using std::function;

namespace cals
{
#if CUDA_ENABLED
  struct DDev {
    void operator() (double* &dev_ptr) {
      cuda::deallocate(dev_ptr);
    }
  };
#endif
  struct Dopnew {
    void operator() (double* &host_ptr) {
      operator delete(host_ptr);
    }
  };

  struct Unfolding
  {
    int n_blocks;     // The number of matrix blocks.
    int block_offset; // The offset (# entries) between blocks in the tensor.
    int rows;         // The rows of one block.
    int cols;         // The cols of one block.
    int stride;       // The (row- or column-) stride of each block.
  };

  // TODO Create separate class for "data_up" to support an iterator, if it does not inhibit performance.
  //  or implement iterator for the tensor class to simplify certain C-like loops
  class Tensor
  {
    int n_elements{};
    int max_n_elements{};
    vector<int> modes;

#if CUDA_ENABLED
    mutable unique_ptr<double, DDev> cudata_up{nullptr, DDev()};
    mutable double *cudata{};
#endif

    unique_ptr<double, Dopnew> data_up{nullptr, Dopnew()};
    double *data{};

    int rank{0};

  public:
    // Constructors and Destructor
    Tensor() = default;

    ~Tensor() = default;

    /** Generic constructor that allocates a Tensor (no initialization).
     *
     * This constructor allocates a Tensor with specific sizes of \p modes.
     * The contents of the Tensor created are not initialized. One can use the randomize member function
     * to fill in the Tensor with random values.
     *
     * @param modes A vector containing the sizes of each mode of the Ktensor.
     */
    explicit Tensor(const vector<int> &modes);

    /** Constructor that uses pre-existing Tensor.
     *
     * This constructor creates a Tensor object which points to data (does not own it) already existing in memory.
     * It is meant to be used, for example, with Matlab, to view a Tensor already initialized in Matlab.
     *
     * @param modes A vector containing the sizes of each mode of the Ktensor.
     * @param view_data Pointer to the location of the Tensor data.
     */
    explicit Tensor(const vector<int> &modes, double *view_data);

    /** Constructor that reads a Tensor from a file.
     *
     * This constructor creates a Tensor, whose contents are read from a file.
     * The first line of the file must contain the dimensions, separated by spaces.
     * The rest of the file should contain the elements of the Tensor, separated by the newline character.
     *
     * @param file_name The name of the file to read the Tensor from.
     */
    explicit Tensor(const std::string &file_name);

    /** Constructor that simplifies the creation of Matrices (2D tensors).
     *
     * @param mode0 Number of rows of a Matrix.
     * @param mode1 Number of columns of a Matrix.
     * @param view_data (Optional) If set, the Matrix created points to data already existing in memory (does not
     * own that data). Otherwise, it allocates an (un-initialized) Matrix.
     */
    Tensor(int mode0, int mode1, double *view_data = nullptr);  // Constructor to simplify creation of matrices

    /** Creates a Tensor of a specific rank.
     *
     * This constructor creates a Tensor of a specific rank by first creating a randomly initialized Ktensor
     * of a specific \p rank, and then transforms that Ktensor into a full Tensor.
     *
     * @param rank The desired rank of the Tensor.
     * @param modes A vector containing the sizes of each mode of the Ktensor.
     */
    Tensor(int rank, const vector<int> &modes);

    // Move and Copy Constructors
    Tensor(Tensor &&rhs) = default;

    Tensor &operator=(Tensor &&rhs) = default;

    Tensor(const Tensor &);

    Tensor &operator=(const Tensor &rhs);

    // Getters
    inline int get_n_elements() const noexcept
    { return n_elements; };

    inline int get_max_n_elements() const noexcept
    { return max_n_elements; };

    inline int get_mode(const int mode) const noexcept
    { return modes[mode]; };

    inline int get_n_modes() const noexcept
    { return modes.size(); };

    inline vector<int> get_modes() const noexcept
    { return modes; };

    inline double *get_data() const noexcept
    { return data; };

    inline void set_data(double *new_data) noexcept
    { data = new_data; };

    inline Tensor &reset_data() noexcept
    {
      data = data_up.get();
      return *this;
    };

    inline int get_rank() const noexcept
    { return rank; };

    inline bool is_view() const noexcept
    { return data_up == nullptr; };

    inline void reset() noexcept
    {
      data_up.reset();
      data = nullptr;
    }

    inline double const &operator[](int index) const noexcept
    { return data[index]; }

    inline double &operator[](int index) noexcept
    { return data[index]; }

    void resize(int new_n_elements, vector<int> &new_modes)
    {
      assert(new_n_elements <= max_n_elements);
      assert(new_modes.size() == modes.size());

      n_elements = new_n_elements;
      modes = std::move(new_modes);
    }

    double norm() const
    { return cblas_dnrm2(n_elements, data, 1); };

    Tensor &fill(const function<double()> &&f);

    Tensor &zero();

    Tensor &randomize();

    inline void copy(const Tensor &ten) noexcept
    { cblas_dcopy(ten.get_n_elements(), ten.get_data(), 1, data, 1); };

    inline long int max_id() noexcept
    { return cblas_idamax(n_elements, data, 1); };

    inline double max() noexcept
    { return data[max_id()]; };

    inline long int max_id(vector<bool> &R) noexcept
    {
      auto max_id = -1;
      auto max = -DBL_MAX;
      assert(R.size() == n_elements);

      for (auto i = 0lu; i < R.size(); i++)
        if (R[i] == true && data[i] > max)
        {
          max = data[i];
          max_id = i;
        }
      assert(max_id != -1);
      return max_id;
    };

    inline double max(vector<bool> &R) noexcept
    {
      return data[max_id(R)];
    };

    inline double min() noexcept
    { return *std::min_element(data, data + n_elements); };

    void print(const std::basic_string<char>& text = "Tensor") const;

    Unfolding implicit_unfold(int mode) const;

#if CUDA_ENABLED
    inline double * const & get_cudata() const noexcept
    { return cudata; };

    inline double *& get_cudata() noexcept
    { return cudata; };

    inline void allocate_cudata(int size) const noexcept
    {
      cuda::allocate(cudata, size);
      cudata_up = unique_ptr<double, DDev>(cudata, DDev());
    };

    inline void send_to_device() const noexcept
    { cuda::send_to_device(data, cudata, n_elements); };

    inline void receive_from_device() noexcept
    { cuda::receive_from_device(cudata, data, n_elements); };

    inline void send_to_device_async(cudaStream_t &stream) const noexcept
    { cuda::send_to_device_async(data, cudata, n_elements, stream); };

    inline void receive_from_device_async(cudaStream_t &stream) noexcept
    { cuda::receive_from_device_async(cudata, data, n_elements, stream); };

    inline void set_cudata(double *new_cudata) noexcept
    { cudata = new_cudata; };

    inline Tensor &reset_cudata() noexcept
    {
      cudata = cudata_up.get();
      return *this;
    };
#endif

  };

} // namespace cals

#endif // CALS_TENSOR_H
