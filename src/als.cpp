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

#include "als.h"

#include <cmath>

#include "utils.h"

#if CUDA_ENABLED

#include "cuda_utils.h"

#endif

using std::cout;
using std::endl;
using ALS_TIMERS = cals::AlsTimer::ALS_TIMERS;
using MODE_TIMERS = cals::ModeTimer::MODE_TIMERS;

namespace cals
{
  AlsReport cp_als(const Tensor &X, Ktensor &ktensor, AlsParams &params)
  {
    // TODO investigate TTV library for fancier way to do this print.
    DEBUG(
        cout << "START method=Regular_ALS" << endl;
        cout << "Tensor size: [ ";
        for (auto mode : X.get_modes()) cout << mode << " ";
        cout << "]" << endl;
        cout << "Ktensor rank= " << ktensor.get_rank() << endl;
    )

    AlsReport rep;

    rep.tensor_rank = X.get_rank();
    rep.n_modes = X.get_n_modes();
    rep.modes = X.get_modes();
    rep.X_norm = X.norm();

    rep.iter = 0;
    rep.max_iter = params.max_iterations;
    rep.n_threads = get_threads();
    rep.ktensor_id = ktensor.get_id();
    rep.ktensor_rank = ktensor.get_rank();
    rep.tol = params.tol;
    rep.cuda = params.cuda;
    rep.update_method = params.update_method;
    rep.line_search = params.line_search;
    rep.line_search_interval = params.line_search_interval;
    rep.line_search_step = params.line_search_step;

    TIME(rep.init_timers();)
    TIME(rep.als_timer.timers[cals::AlsTimer::TOTAL].start();)

    assert(rep.modes.size() >= 3);

    Ktensor ls_ktensor{};
    Matrix ten_workspace{};
    cals::ops::LineSearchParams ls_params{};
    if (rep.line_search)
    {
      if (rep.modes.size() == 3)
      {
        ls_ktensor = Ktensor(rep.ktensor_rank, rep.modes);
        ten_workspace = Matrix(rep.modes[0], rep.modes[1] * rep.modes[2]);
        ls_params.cuda = rep.cuda;
        ls_params.step = rep.line_search_step;
      } else
      {
        std::cerr << "Line search supported only for 3D tensors." << std::endl;
        abort();
      }
    }

    // Allocate and Initialize Gramians vector
    auto gramians = vector<Matrix>(rep.n_modes);
    for (auto n = 0; n < rep.n_modes; n++) gramians[n] = Matrix(ktensor.get_rank(), ktensor.get_rank());
    for (auto n = 0; n < rep.n_modes; n++) cals::ops::update_gramian(ktensor.get_factor(n), gramians[n]);

    // Create empty matrix to store last G for error calculation
    Matrix G_last(rep.modes[rep.n_modes - 1], ktensor.get_rank());

    // Calculate and Allocate Workspace for intermediate KRP and Twostep
    vector<Matrix> workspace;
    vector<int> modes = rep.modes;  // Create a copy to sort it
    std::sort(modes.begin(), modes.end(), std::greater<>());
    vector<int> sizes(std::ceil(static_cast<float>(modes.size() - 1) / 2.0),
                      0);  // sizes holds the nrows of each workspace matrix for KRP
    sizes[0] = modes[0] * modes[1];  // Two largest modes (Can be used as a workspace for twostep)
    for (auto i = 1lu; i < sizes.size(); i++) sizes[i] = modes[i + 1] * sizes[i - 1];

    workspace.resize(sizes.size());
    for (auto i = 0lu; i < sizes.size(); i++) workspace[i] = Matrix(sizes[i], ktensor.get_rank());

    // Create and initialize MTTKRP params
    cals::ops::MttkrpParams mttkrp_params;
    mttkrp_params.method = params.mttkrp_method;
    mttkrp_params.cuda = params.cuda;
    mttkrp_params.krp_params.cuda = params.cuda;

    Timer lut_timer;
    lut_timer.start();
    mttkrp_params.lut = cals::utils::read_lookup_table(rep.modes, get_threads());

    if (!mttkrp_params.lut.empty())
    {
      vector<int> mttkrp_lut_keys;
      mttkrp_lut_keys.reserve(mttkrp_params.lut.size());
      for (auto &[key, val] : mttkrp_params.lut[0])
        mttkrp_lut_keys.emplace_back(key);
      mttkrp_params.lut_keys = std::move(mttkrp_lut_keys);
    }
    lut_timer.stop();
//    std::cout << "Lookup table creation took " << lut_timer.get_time() << "seconds." << std::endl;

    if (params.cuda)
    {
#if CUDA_ENABLED
      X.allocate_cudata(X.get_n_elements());
      X.send_to_device();

      for (int i = 0; i < ktensor.get_n_modes(); i++)
      {
        ktensor.get_factor(i).allocate_cudata(ktensor.get_factor(i).get_n_elements());
        ktensor.get_factor(i).send_to_device();
      }

      for (auto &w : workspace)
        w.allocate_cudata(w.get_n_elements());
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }

    bool converged{false};
    rep.iter = 0;
    do
    {
      rep.iter++;
      rep.flops_per_iteration = 0;

      TIME(rep.als_timer.timers[ALS_TIMERS::ITERATION].start();)

      if (rep.line_search && !((rep.iter + 1) % rep.line_search_interval))
      {
        for (auto i = 0; i < rep.n_modes; i++)
          ls_ktensor.get_factor(i).copy(ktensor.get_factor(i));
      }

      // Loop over the modes
      for (auto n = 0; n < rep.n_modes; n++)
      {
        auto &factor = ktensor.get_factor(n);

        TIME(rep.mode_timers[n].timers[MODE_TIMERS::TOTAL_MTTKRP].start();)
        TIME(cals::utils::set_mttkrp_timers(rep, mttkrp_params, n);)

        auto &G = ops::mttkrp(X, ktensor, workspace, n, mttkrp_params);

        TIME(rep.mode_timers[n].timers[MODE_TIMERS::TOTAL_MTTKRP].stop();)
        TIME(rep.flops_per_iteration += mttkrp_params.flops;)

        auto &H = cals::ops::hadamard_but_one(gramians, n);

        if (params.cuda)
        {
#if CUDA_ENABLED
          cudaDeviceSynchronize();
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        // Save the last G matrix for error calculation
        if (n == rep.n_modes - 1)
        {
          TIME(rep.als_timer.timers[ALS_TIMERS::G_COPY].start();)
          G_last.read(G);
          TIME(rep.als_timer.timers[ALS_TIMERS::G_COPY].stop();)
        }

        TIME(rep.mode_timers[n].timers[MODE_TIMERS::UPDATE].start();)

        if (params.update_method == cals::update::UPDATE_METHOD::UNCONSTRAINED)
          cals::update::update_factor_unconstrained(factor, H);
        else
          cals::update::update_factor_non_negative_constrained(factor, H, ktensor.get_active_set(n));

        ktensor.normalize(n, rep.iter);

        if (params.cuda)
        {
#if CUDA_ENABLED
          ktensor.get_factor(n).send_to_device_async(mttkrp_params.cuda_params.streams[0]);
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        cals::ops::update_gramian(factor, gramians[n]);

        if (params.cuda)
        {
#if CUDA_ENABLED
          cudaDeviceSynchronize();
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        TIME(rep.mode_timers[n].timers[MODE_TIMERS::UPDATE].stop();)
      } // for mode

      // Compute the approximation error
      TIME(rep.als_timer.timers[ALS_TIMERS::ERROR].start();)

      ktensor.set_iters(rep.iter);

      const auto old_error = ktensor.get_approx_error();

      cals::ops::hadamard_all(gramians);
      const auto error = cals::ops::compute_fast_error(rep.X_norm, ktensor.get_lambda(), ktensor.get_last_factor(),
                                                       G_last, gramians[0]);

      ktensor.set_approx_error(error);
      ktensor.set_approx_error_diff(std::fabs(old_error - error));

      if (!((rep.iter == 1) || (old_error - error >= -1e-4)))
        std::cerr << "error not decr Ktensor: " << ktensor.get_id() << " Error: " << ktensor.get_approx_error()
                  << " Old Error: " << old_error << endl;  // Ensure error always gets smaller

      const auto old_fit = ktensor.get_fit();
      const auto fit = 1 - std::fabs(ktensor.get_approx_error()) / rep.X_norm;

      ktensor.set_fit(fit);
      ktensor.set_fit_diff(std::fabs(fit - old_fit));

      TIME(rep.als_timer.timers[ALS_TIMERS::ERROR].stop();)

      if (rep.line_search && !(rep.iter % rep.line_search_interval))
        cals::ops::line_search(ktensor, ls_ktensor, workspace[0], ten_workspace, gramians, X, rep.X_norm, ls_params);

      TIME(rep.als_timer.timers[ALS_TIMERS::ITERATION].stop();)

      DEBUG(cout << "CONVERGENCE " << rep.iter << " " << ktensor.get_approx_error() << endl;)

      if (!params.force_max_iter)
        converged = (ktensor.get_fit_diff() < rep.tol) || (rep.iter >= rep.max_iter);
      else
        converged = rep.iter >= rep.max_iter;

      if (params.force_max_iter) converged = (rep.iter >= rep.max_iter);
    } while (!converged);

    TIME(rep.als_timer.timers[AlsTimer::TOTAL].stop();)
    DEBUG(cout << "done." << endl;)

    return rep;
  }
}
