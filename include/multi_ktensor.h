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

#ifndef CALS_MULTI_KTENSOR_H
#define CALS_MULTI_KTENSOR_H

#include <map>
#include "definitions.h"
#include "ktensor.h"

namespace cals
{
  struct RegistryEntry
  {
    Ktensor &ktensor;
    vector<Matrix> gramians;
    int col;
    Ktensor ls_ktensor{};
  };

  typedef std::map<int, RegistryEntry> Registry;

  class MultiKtensor : public Ktensor
  {
    int occupancy{0};  // Number of occupied cols.
    int start{0};
    int end{0};
    bool cuda{false};
    bool line_search{false};
    vector<int> modes;
    vector<int> occupancy_vec;  // Occupancy of the available buffer.
    Registry registry;  // Keeps track of Ktensors currently in the buffer.

    int check_availability(Ktensor &ktensor);

    MultiKtensor &adjust_edges();

  public:
    MultiKtensor() = default;

    ~MultiKtensor() = default;

    MultiKtensor &operator=(MultiKtensor &&mk) = default;

    explicit MultiKtensor(vector<int> &modes, int buffer_size);

    MultiKtensor &add(Ktensor &ktensor);

    MultiKtensor &remove(int ktensor_id);

    Registry &get_registry()
    { return registry; }

    inline int get_start() const noexcept
    { return start; }

    MultiKtensor &compress();

    inline void set_cuda(bool value)
    { cuda = value; };

    inline void set_line_search(bool value)
    { line_search = value; };

    inline vector<int> &get_modes()
    { return modes; };
  };

  struct BufferFull : public std::exception
  {
    const char *what() const noexcept override
    {
      return "Buffer is full, wait until some ktensors converge.";
    }
  };
}

#endif //CALS_MULTI_KTENSOR_H
