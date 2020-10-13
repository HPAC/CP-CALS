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

#include "multi_ktensor.h"

#include "utils.h"

namespace cals
{
  MultiKtensor::MultiKtensor(vector<int> &modes, int buffer_size)
      : Ktensor(buffer_size, modes),
        occupancy(0),
        modes(modes),
        occupancy_vec(buffer_size)
  {
    for (auto &f : get_factors()) f.resize(f.get_rows(), 0);
  }

  int MultiKtensor::check_availability(Ktensor &ktensor)
  {
    auto pos_index = -1;
    auto rank_counter = 0;
    auto prev_occ = -1;

    for (auto i = 0lu; i < occupancy_vec.size(); i++)
    {
      if (rank_counter == ktensor.get_rank())
        break;

      if (occupancy_vec[i] == 0 && prev_occ != 0)
      {
        pos_index = i;
        rank_counter++;
      } else if (occupancy_vec[i] == 0 && prev_occ == 0)
        rank_counter++;
      else
        rank_counter = 0;

      prev_occ = occupancy_vec[i];
    }

    if (pos_index == -1 || rank_counter != ktensor.get_rank())
      throw BufferFull();

    return pos_index;
  }

  MultiKtensor &MultiKtensor::add(Ktensor &ktensor)
  {
    int pos_index{0};
    try
    {
      pos_index = check_availability(ktensor);
    }
    catch (BufferFull &e)
    {
      throw e;
    }

    // Determine address for each factor matrix
    vector<double *> pos_ptrs(ktensor.get_n_modes());

    auto index = 0;
    for (auto &f : get_factors())
      pos_ptrs[index++] = f.reset_data().get_data() + pos_index * f.get_col_stride();

    // Attach Ktensor
    ktensor.attach(pos_ptrs);


    if (cuda)
    {
#if CUDA_ENABLED
      vector<double *> cupos_ptrs(ktensor.get_n_modes());
      auto cuindex = 0;
      for (auto &f : get_factors())
        cupos_ptrs[cuindex++] = f.reset_cudata().get_cudata() + pos_index * f.get_col_stride();
      ktensor.cuattach(cupos_ptrs);
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
    }

    // Adjust occupancy vector
    for (auto i = 0; i < ktensor.get_rank(); i++) occupancy_vec[pos_index + i] = ktensor.get_id();

    // adjust occupancy_pct
    occupancy += ktensor.get_rank();

    // Create gramians
    vector<Matrix> gramians(ktensor.get_n_modes());
    for (auto &g : gramians) g = Matrix(ktensor.get_rank(), ktensor.get_rank());
    index = 0;
    for (const auto &f : ktensor.get_factors()) ops::update_gramian(f, gramians[index++]);
    ktensor.set_iters(1);

    RegistryEntry entry{ktensor, std::move(gramians), pos_index};

    if (line_search)
      entry.ls_ktensor = Ktensor(entry.ktensor.get_rank(), modes);

    // Update registry
    registry.insert(std::pair(ktensor.get_id(), std::move(entry)));

    adjust_edges();

    return *this;
  }

  MultiKtensor &MultiKtensor::remove(int ktensor_id)
  {
    auto &entry = registry.at(ktensor_id);
    auto &ktensor = entry.ktensor;

    // Remove ktensor from the factor matrices and copy over their last value
    ktensor.detach();

    if (cuda)
    {
#if CUDA_ENABLED
      ktensor.cudetach();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
    }

    // Update the occupancy vector and counter
    for (auto &el : occupancy_vec)
      if (el == ktensor_id)
        el = 0;
    occupancy -= ktensor.get_rank();

    // Update registry
    registry.erase(ktensor_id);

    adjust_edges();

    return *this;
  }

  MultiKtensor &MultiKtensor::adjust_edges()
  {
    start = 0;
    int end = occupancy_vec.size();

//    for (auto i = 0; i < end; i++)
//      if (occupancy_vec[i] == 0) start++;
//      else break;
    for (auto i = end - 1; i > start; i--) // end is equal to the size (so -1 to get the last element)
      if (occupancy_vec[i] == 0) end--;
      else break;

    int active_cols = end - start;
    this->end = end;
    for (auto &f : get_factors())
    {
      f.set_data(f.reset_data().get_data() + start * f.get_col_stride());  // Adjust data to point to start
      f.resize(f.get_rows(), active_cols);
    }

    return *this;
  }

  MultiKtensor &MultiKtensor::compress()
  {
    if (1.0 * occupancy / (end - start) < 0.99)
    {
//      std::cout << "Occupancy percent: " << 1.0 * occupancy / (end - start) << " Size: " << end << " Compression...";

      int col_offset = 0;
      int added = -1;
      // move_requests is an ordered list with the ids of ktensors and the steps to the left they need to move (offset).
      std::vector<std::tuple<int, int>> move_requests;
      move_requests.reserve(occupancy);

      for (auto cell : occupancy_vec)
        if (cell == added)
          continue;
        else if (cell == 0)
          col_offset++;
        else if (col_offset != 0)
        {
          move_requests.emplace_back(std::make_tuple(cell, col_offset));
          added = cell;
        }

      vector<double *> new_data(get_n_modes());
      for (auto &[key, offset]: move_requests)
      {
        auto index = 0;
        auto &registry_entry = registry.at(key);
        auto &ktensor = registry_entry.ktensor;

        for (auto &factor : ktensor.get_factors())
          new_data.at(index++) = factor.get_data() - offset * factor.get_col_stride();

        if (ktensor.get_rank() < offset)
          registry_entry.ktensor.attach(new_data, true);
        else
          registry_entry.ktensor.attach(new_data, false);

        for (auto i = registry_entry.col; i < registry_entry.col + ktensor.get_rank(); i++)
          std::swap(occupancy_vec[i - offset], occupancy_vec[i]);
        registry.at(key).col -= offset;

        if (cuda)
        {
#if CUDA_ENABLED
          for (auto &factor : ktensor.get_factors())
          {
            auto *new_cudata = factor.get_cudata() - offset * factor.get_col_stride();
            cudaMemcpyAsync(new_cudata, factor.get_cudata(), factor.get_n_elements() * sizeof(double),
                            cudaMemcpyDeviceToDevice, custream);
            factor.set_cudata(new_cudata);
          }
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }
      }

      adjust_edges();

      if (cuda)
      {
#if CUDA_ENABLED
        cudaDeviceSynchronize();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
      }

    }
    return *this;
  }
}

