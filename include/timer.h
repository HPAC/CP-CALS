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

#ifndef CALS_TIMER_H
#define CALS_TIMER_H

#include <chrono>
#include <iostream>
#include <cfloat>

using std::vector;
using time_point = std::chrono::high_resolution_clock::time_point;

namespace cals
{
  class Timer
  {
    time_point t0{};
    time_point t1{};
    double t{FLT_MAX};
    double min{FLT_MAX};

  public:
    Timer() = default;

    inline void start()
    { t0 = std::chrono::high_resolution_clock::now(); };

    inline void stop()
    {
      t1 = std::chrono::high_resolution_clock::now();
      t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
      if (t < min) min = t;
    };

    inline double get_time() const
    { return (t == FLT_MAX) ? 0.0 : t; };

    inline double get_min() const
    { return (min == FLT_MAX) ? 0.0 : min; };
  };

  struct ModeTimer
  {
    enum MODE_TIMERS
    {
      TOTAL_MTTKRP = 0,
      MTTKRP_KRP,
      MTTKRP_GEMM,
      TWOSTEP_GEMM,
      TWOSTEP_GEMV,
      UPDATE,
      LENGTH
    };
    std::string names[MODE_TIMERS::LENGTH] =
        {
            "TOTAL_MTTKRP",
            "MTTKRP_KRP",
            "MTTKRP_GEMM",
            "TWOSTEP_GEMM",
            "TWOSTEP_GEMV",
            "UPDATE"
        };
    Timer timers[MODE_TIMERS::LENGTH];
  };

  struct AlsTimer
  {
    enum ALS_TIMERS
    {
      TOTAL = 0,
      ITERATION,
      ERROR,
      G_COPY,
      LENGTH
    };
    std::string names[ALS_TIMERS::LENGTH] =
        {
            "TOTAL",
            "ITERATION",
            "ERROR",
            "G_COPY"
        };
    Timer timers[ALS_TIMERS::LENGTH];
  };

} // namespace cals

#endif // CALS_TIMER_H
