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

#include "cals.h"

#include <cmath>

#include "utils.h"
#include <multi_ktensor.h>

#if CUDA_ENABLED

#include "cuda_utils.h"

#endif

using std::cout;
using std::endl;
using ALS_TIMERS = cals::AlsTimer::ALS_TIMERS;
using MODE_TIMERS = cals::ModeTimer::MODE_TIMERS;

namespace cals
{
  CalsReport cp_cals(const Tensor &X, KtensorQueue &kt_queue, CalsParams &params)
  {
    // TODO investigate TTV library for fancier way to do this print.
    DEBUG(
        cout << "START method=Concurrent_ALS" << endl;
        cout << "Tensor size: [ ";
        for (const auto &mode: X.get_modes()) cout << mode << " ";
        cout << "]" << endl;
    )

    Timer total_time;
    Timer init_time;
    CalsReport rep;

    rep.tensor_rank = X.get_rank();
    rep.n_modes = X.get_n_modes();
    rep.modes = X.get_modes();
    rep.X_norm = X.norm();

    rep.iter = 0;
    rep.max_iter = params.max_iterations;
    rep.n_threads = get_threads();
    rep.buffer_size = params.buffer_size;
    rep.n_ktensors = 0;
    rep.ktensor_rank_sum = 0;
    rep.tol = params.tol;
    rep.cuda = params.cuda;
    rep.update_method = params.update_method;
    rep.line_search = params.line_search;
    rep.line_search_interval = params.line_search_interval;
    rep.line_search_step = params.line_search_step;

    TIME(auto timer_size = std::ceil(20.0 * kt_queue.size() / rep.buffer_size) * (rep.max_iter + 1););
    TIME(rep.init_timers(rep.modes.size(), timer_size);)
    TIME(for (auto &timer : rep.als_timer) timer.timers[cals::AlsTimer::TOTAL].start();)

    assert(rep.modes.size() >= 3);
    total_time.start();
    init_time.start();

    Matrix ten_workspace{};
    cals::ops::LineSearchParams ls_params{};
    if (rep.line_search)
    {
      if (rep.modes.size() == 3)
      {
        ten_workspace = Matrix(rep.modes[0], rep.modes[1] * rep.modes[2]);
        ls_params.cuda = rep.cuda;
        ls_params.step = rep.line_search_step;
      } else
      {
        std::cerr << "Line search supported only for 3D tensors." << std::endl;
        abort();
      }
    }

    MultiKtensor mkt(rep.modes, params.buffer_size);
    mkt.set_cuda(rep.cuda);
    mkt.set_line_search(rep.line_search);

    auto &registry = mkt.get_registry();

    // Vector needed to hold the keys in the registry, so that multithreaded update & error can happen.
    // OpenMP (4.5) does not yet support parallalelizing non range-based loops.
    // Check OpenMP 5.0 (gcc 9) for possible improvement.
    vector<int> keys_v;
    keys_v.reserve(rep.buffer_size);

    // Create empty matrix to store last G for error calculation
    Matrix G_last(rep.modes[rep.n_modes - 1], params.buffer_size);

    // Calculate and Allocate Workspace for intermediate KRP and Twostep
    vector<Matrix> workspace;
    vector<int> modes = rep.modes;  // Create a copy to sort it
    std::sort(modes.begin(), modes.end(), std::greater<>());
    vector<int> sizes(std::ceil(static_cast<float>(modes.size() - 1) / 2.0),
                      0);  // sizes holds the nrows of each workspace matrix for KRP
    sizes[0] = modes[0] * modes[1];  // Two largest modes (Can be used as a workspace for twostep)
    for (auto i = 1lu; i < sizes.size(); i++) sizes[i] = modes[i + 1] * sizes[i - 1];

    workspace.resize(sizes.size());
    for (auto i = 0lu; i < sizes.size(); i++) workspace[i] = Matrix(sizes[i], params.buffer_size);

    // Vector needed to hold the keys of ktensors to be removed from the registry.
    // Can't traverse the map and remove at the same time safely.
    vector<int> remove_ids;
    remove_ids.reserve(100);

    // Create and initialize MTTKRP params
    cals::ops::MttkrpParams mttkrp_params;
    mttkrp_params.method = params.mttkrp_method;
    mttkrp_params.cuda = params.cuda;
    mttkrp_params.krp_params.cuda = params.cuda;

    init_time.stop();

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
    std::cout << "Lookup table creation took " << lut_timer.get_time() << " seconds." << std::endl;
    std::cout << "Initialization took " << init_time.get_time() << " seconds." << std::endl;

    if (params.cuda)
    {
#if CUDA_ENABLED
      X.allocate_cudata(X.get_n_elements());
      X.send_to_device_async(mttkrp_params.cuda_params.streams[0]);

      for (auto &mf : mkt.get_factors())
        mf.allocate_cudata(mf.get_max_n_elements());

      for (auto &w : workspace)
        w.allocate_cudata(w.get_n_elements());

      cuda::init_custream();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }

    bool converged{false};
    mkt.set_iters(0);
    do
    {
      mkt.set_iters(mkt.get_iters() + 1);
      rep.iter = mkt.get_iters();
      cout << "Iteration: " << rep.iter << endl;

      TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::ITERATION].start();)

      while (!kt_queue.empty())
      {
        try
        {
          Ktensor &ktensor = kt_queue.front();
          mkt.add(ktensor);
          rep.n_ktensors++;
          rep.ktensor_rank_sum += ktensor.get_rank();
          kt_queue.pop();
        }
        catch (BufferFull &e)
        {
          break;
        }
      }
      keys_v.clear();
      for (auto &[key, val] : registry)
        keys_v.emplace_back(key);

#pragma omp parallel for  // NOLINT(openmp-use-default-none)
      for (auto i = 0lu; i < keys_v.size(); i++)
      {
        auto &val = registry.at(keys_v[i]);
        if (rep.line_search && !((val.ktensor.get_iters() + 1) % rep.line_search_interval))
        {
          for (auto j = 0; j < rep.n_modes; j++)
            val.ls_ktensor.get_factor(j).copy(val.ktensor.get_factor(j));
        }

      }

      if (params.cuda)
      {
#if CUDA_ENABLED
        cudaDeviceSynchronize();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }

      TIME(rep.cols[rep.iter] = mkt.get_factor(0).get_cols();)

      // Loop over the modes
      for (auto n = 0; n < rep.n_modes; n++)
      {
        TIME(rep.mode_timers[rep.iter][n].timers[MODE_TIMERS::TOTAL_MTTKRP].start();)
        TIME(cals::utils::set_mttkrp_timers(rep, mttkrp_params, n, rep.iter);)

        auto &G = ops::mttkrp(X, mkt, workspace, n, mttkrp_params);

        TIME(rep.mode_timers[rep.iter][n].timers[MODE_TIMERS::TOTAL_MTTKRP].stop();)
        TIME(rep.flops_per_iteration[rep.iter] += mttkrp_params.flops;)

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
          TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::G_COPY].start();)
          G_last.resize(G.get_rows(), G.get_cols()).copy(G);
          TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::G_COPY].stop();)
        }

        TIME(rep.mode_timers[rep.iter][n].timers[MODE_TIMERS::UPDATE].start();)

#pragma omp parallel for  // NOLINT(openmp-use-default-none)
        for (auto i = 0lu; i < keys_v.size(); i++)
        {
          auto &val = registry.at(keys_v[i]);
          ops::hadamard_but_one(val.gramians, n);

          if (params.update_method == update::UPDATE_METHOD::UNCONSTRAINED)
            update::update_factor_unconstrained(val.ktensor.get_factor(n), val.gramians[n]);
          else
            update::update_factor_non_negative_constrained(val.ktensor.get_factor(n), val.gramians[n],
                                                           val.ktensor.get_active_set(n));

          val.ktensor.normalize(n, rep.iter);

          ops::update_gramian(val.ktensor.get_factor(n), val.gramians[n]);
        }

        if (params.cuda)
        {
#if CUDA_ENABLED
          mkt.get_factor(n).send_to_device();
          cudaDeviceSynchronize();
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        TIME(rep.mode_timers[rep.iter][n].timers[MODE_TIMERS::UPDATE].stop();)
      } // for mode

      // Compute the approximation error
      TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::ERROR].start();)

#pragma omp parallel for  // NOLINT(openmp-use-default-none)
      for (auto i = 0lu; i < keys_v.size(); i++)
      {
        auto &val = registry.at(keys_v[i]);

        const auto old_error = val.ktensor.get_approx_error();

        const auto &ktensor_G_last = Matrix(val.ktensor.get_last_factor().get_rows(), val.ktensor.get_rank(),
                                            G_last.get_data() + (val.col - mkt.get_start()) * G_last.get_col_stride());
        cals::ops::hadamard_all(val.gramians);
        const auto error = ops::compute_fast_error(rep.X_norm, val.ktensor.get_lambda(), val.ktensor.get_last_factor(),
                                                   ktensor_G_last, val.gramians[0]);
        val.ktensor.set_approx_error(error);
        val.ktensor.set_approx_error_diff(std::fabs(old_error - error));
        assert((val.ktensor.get_iters() == 1) || (old_error - error >= -1e-5));  // Make sure error decreases

        const auto old_fit = val.ktensor.get_fit();
        const auto fit = 1 - std::fabs(val.ktensor.get_approx_error()) / rep.X_norm;
        val.ktensor.set_fit(fit);
        val.ktensor.set_fit_diff(std::fabs(old_fit - fit));
      }

      // Don't parallelize until you figure out way to overcome ten_workspace limitation
      for (auto i = 0lu; i < keys_v.size(); i++)
      {
        auto &val = registry.at(keys_v[i]);
        if (rep.line_search && !(val.ktensor.get_iters() % (rep.line_search_interval)))
          cals::ops::line_search(val.ktensor, val.ls_ktensor, workspace[0], ten_workspace, val.gramians, X, rep.X_norm,
                                 ls_params);
      }
      TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::ERROR].stop();)

      remove_ids.clear();
      for (auto &[key, val] : registry)
        if (!params.force_max_iter)
        {
          if (val.ktensor.get_fit_diff() < rep.tol || val.ktensor.get_iters() >= rep.max_iter)
            remove_ids.push_back(key);
          else
            val.ktensor.set_iters(val.ktensor.get_iters() + 1);
        }
        else if (val.ktensor.get_iters() >= rep.max_iter)
          remove_ids.push_back(key);
        else
          val.ktensor.set_iters(val.ktensor.get_iters() + 1);

      for (auto &key : remove_ids)
        mkt.remove(key);

      if (params.cuda)
      {
#if CUDA_ENABLED
        cudaDeviceSynchronize();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }
      // Compression
      mkt.compress();

      TIME(rep.als_timer[rep.iter].timers[ALS_TIMERS::ITERATION].stop();)

      DEBUG(
          cout << "CONVERGENCE " << rep.iter << endl;
          for (auto &[key, val]: registry) cout << val.ktensor.get_approx_error() << " ";
          cout << endl;
      )

      if (kt_queue.empty() && registry.empty()) converged = true;

    } while (!converged);

    // Remove any model that has not converged (and copy it back to the original ktensor)
    if (!registry.empty())
    {
      remove_ids.clear();
      for (auto &[key, val] : registry)
        remove_ids.push_back(key);
      for (auto &key : remove_ids)
        mkt.remove(key);
    }

    if (params.cuda)
    {
#if CUDA_ENABLED
      cudaDeviceSynchronize();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }

    TIME(for (auto &timer : rep.als_timer) timer.timers[cals::AlsTimer::TOTAL].stop();)

    total_time.stop();

    cout << "Computation time: " << total_time.get_time() << endl;

    if (params.cuda)
    {
#if CUDA_ENABLED
      cuda::destroy_custream();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }

    DEBUG(cout << "done." << endl;)

    return rep;
  }
}