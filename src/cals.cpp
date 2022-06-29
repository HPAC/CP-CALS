#include "cals.h"

#include "multi_ktensor.h"
#include "utils/error.h"

#if CUDA_ENABLED

#include "cuda_utils.h"

#endif

using std::cout;
using std::endl;
using ALS_TIMERS = cals::AlsTimers::TIMERS;
using MODE_TIMERS = cals::ModeTimers::TIMERS;
using MTTKRP_TIMERS = cals::MttkrpTimers::TIMERS;

namespace cals {
CalsReport cp_cals(const Tensor &X, KtensorQueue &kt_queue, CalsParams &cals_params) {

#ifndef NDEBUG
  cout << "START method=Concurrent_ALS" << endl;
  cout << "Tensor size: ";
  for (const auto &mode : X.get_modes())
    cout << mode << " ";
  cout << endl;
#endif

  CalsReport rep;
  DEBUG(cout << "OMP_NUM_THREADS: " << omp_get_max_threads() << endl;)
  DEBUG(cout << "Threads set: " << get_threads() << endl;)

  rep.tensor_rank = X.get_rank();
  rep.n_modes = X.get_n_modes();
  rep.modes = X.get_modes();
  rep.X_norm = X.norm();

  rep.iter = 0;
  rep.max_iter = cals_params.max_iterations;
  rep.n_threads = get_threads();
  rep.buffer_size = cals_params.buffer_size;
  rep.n_ktensors = 0;
  rep.ktensor_comp_sum = 0;
  rep.tol = cals_params.tol;
  rep.cuda = cals_params.cuda;
  rep.update_method = cals_params.update_method;
  rep.line_search = cals_params.line_search;
  rep.line_search_interval = cals_params.line_search_interval;
  rep.line_search_step = cals_params.line_search_step;
  rep.line_search_method = cals_params.line_search_method;

  assert(rep.modes.size() >= 3);

#if WITH_TIME
  auto timer_size = static_cast<dim_t>(std::ceil(20.0 * kt_queue.size() / rep.buffer_size) * rep.max_iter);
  rep.als_times = Matrix(ALS_TIMERS::LENGTH, timer_size);
  rep.mode_times = Matrix(MODE_TIMERS::LENGTH * rep.n_modes, timer_size);
  rep.mttkrp_times = Matrix(MTTKRP_TIMERS::LENGTH * rep.n_modes, timer_size);
  auto &als_times = rep.als_times;
  auto &mode_times = rep.mode_times;
  auto &mttkrp_times = rep.mttkrp_times;
  rep.flops_per_iteration.resize(timer_size, 0llu);
  rep.cols.resize(timer_size, 0);
#endif

  AlsTimers als_timers;
  ModeTimers mode_timers;

  Timer total_time;
  Timer init_time;
  total_time.start();
  init_time.start();

  // Create empty matrix to store last G for error calculation.
  Matrix G_last(rep.modes[rep.n_modes - 1], rep.buffer_size);

  // Calculate and Allocate Workspaces for intermediate KRP and Twostep
  auto sorted_modes = rep.modes; // Create a copy to sort it
  std::sort(sorted_modes.begin(), sorted_modes.end(), std::greater<>());
  auto n_half_modes = static_cast<size_t>(std::ceil((rep.modes.size() - 1.0) / 2.0));
  vector<dim_t> ws_sizes(n_half_modes);            // ws_sizes holds the nrows of each workspaces matrix for KRP
  ws_sizes[0] = sorted_modes[0] * sorted_modes[1]; // Two largest modes (Can be used as a workspaces for twostep)
  for (auto i = 1lu; i < ws_sizes.size(); i++)
    ws_sizes[i] = sorted_modes[i + 1] * ws_sizes[i - 1];

  vector<Matrix> workspaces(ws_sizes.size());
  for (auto i = 0lu; i < ws_sizes.size(); i++)
    workspaces[i] = Matrix(ws_sizes[i], rep.buffer_size);

  // Create and initialize MTTKRP parameters
  mttkrp::MttkrpParams mttkrp_params;
  mttkrp_params.method = cals_params.mttkrp_method;
  mttkrp_params.lut = cals_params.mttkrp_lut;
  mttkrp_params.krp_params.cuda = rep.cuda;
  mttkrp_params.cuda = rep.cuda;

  if (mttkrp_params.lut.keys_v.empty()) {
    Timer lut_timer;
    lut_timer.start();
    std::cerr << "LUT empty" << endl;
    mttkrp_params.lut = mttkrp::read_lookup_table(rep.modes, get_threads(), rep.cuda, false);
    lut_timer.stop();
    // std::cout << "Lookup table creation took " << lut_timer.get_time() << " seconds." << std::endl;
  }

  // Create and initialize Line Search parameters
  ls::LineSearchParams ls_params{};
  if (rep.line_search) {
    ls_params.method = rep.line_search_method;
    ls_params.interval = rep.line_search_interval;
    ls_params.step = rep.line_search_step;
    ls_params.cuda = rep.cuda;
    ls_params.T = &X;
  }

  // Create MultiKtensor (buffers)
  MultiKtensor mkt(rep.modes, rep.buffer_size);
  mkt.set_cuda(rep.cuda);
  mkt.set_line_search(rep.line_search);
  mkt.set_line_search_params(ls_params);

  auto &registry = mkt.get_registry();

  // Vector needed to hold the keys in the registry, so that multi-threaded update & error can happen.
  // OpenMP (4.5) does not yet support parallalelizing non range-based loops.
  // TODO Check OpenMP 5.0 (gcc 9) for possible improvement.
  vector<int> keys_v;
  keys_v.reserve(rep.buffer_size);

  // Vector needed to hold the keys of ktensors to be removed from the registry.
  // Can't traverse the map and remove at the same time safely.
  vector<int> remove_ids;
  remove_ids.reserve(100);

  // Jackknifing norms of X for different missing samples.
  vector<double> X_norms_jk{};

#if CUDA_ENABLED
  auto stream = cuda::create_stream();
#endif

  if (rep.cuda) {
#if CUDA_ENABLED
    if (X.get_cudata() == nullptr) {
      X.allocate_cudata(X.get_n_elements());
      X.send_to_device_async(stream);
    }

    for (auto &mf : mkt.get_factors())
      mf.allocate_cudata(mf.get_max_n_elements());

    for (auto &w : workspaces)
      w.allocate_cudata(w.get_n_elements());
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  if (rep.cuda) {
#if CUDA_ENABLED
    cudaStreamSynchronize(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  init_time.stop();
  // std::cout << "Initialization took " << init_time.get_time() << " seconds." << std::endl;

  bool converged{false};
  mkt.set_iters(0);
  do {
    mkt.set_iters(mkt.get_iters() + 1);
    rep.iter = mkt.get_iters();
    // cout << "Iteration: " << rep.iter << endl;

    als_timers[ALS_TIMERS::ITERATION].start();

    // Add new models to the registry if they are waiting in the queue and they fit.
    while (!kt_queue.empty()) {
      try {
        Ktensor &ktensor = kt_queue.front();
        mkt.add(ktensor);
        rep.n_ktensors++;
        rep.ktensor_comp_sum += ktensor.get_components();
        kt_queue.pop();
      } catch (BufferFull &e) {
        break;
      }
    }
    // Update the vector containing the keys of all models currently in the registry.
    keys_v.clear();
    for (auto &[key, val] : registry)
      keys_v.emplace_back(key);

    if (mkt.get_flag_jk())
      if (X_norms_jk.empty())
        X_norms_jk = cals::utils::calculate_jackknifing_norms(X);

    // If line search is enabled, check to see if it is time to store the previous version of a model.
    if (rep.line_search) {
#pragma omp parallel for // NOLINT(openmp-use-default-none)
      for (auto i = 0lu; i < keys_v.size(); i++) {
        auto &val = registry.at(keys_v[i]);
        if (val.ls_params.iter == (val.ls_params.interval - 1)) {
          val.ls_params.prev_ktensor.copy(val.ktensor);
        }
      }
    }

#if WITH_TIME
    auto iter_index = static_cast<dim_t>(rep.iter - 1);
    rep.cols[iter_index] = mkt.get_factor(0).get_cols();
    rep.flops_per_iteration[rep.iter - 1] = 0;
#endif

    // Loop over the modes
    for (dim_t n = 0; n < rep.n_modes; n++) {

      // Compute the MTTKRP for the current mode.
      mode_timers[MODE_TIMERS::MTTKRP].start();

      auto &G = mttkrp::mttkrp(X, mkt, workspaces, n, mttkrp_params);

      mode_timers[MODE_TIMERS::MTTKRP].stop();

      // Save the last G matrix for error calculation
      if (n == rep.n_modes - 1) {
        als_timers[ALS_TIMERS::G_COPY].start();
        G_last.resize(G.get_rows(), G.get_cols()).copy(G);
        als_timers[ALS_TIMERS::G_COPY].stop();
      }

      // Update the factor matrix for the current mode.
      mode_timers[MODE_TIMERS::UPDATE].start();

#pragma omp parallel for
      for (auto i = 0lu; i < keys_v.size(); i++) {
        auto &val = registry.at(keys_v[i]);
        ops::hadamard_but_one(val.gramians, n);

        if (cals_params.update_method == update::UPDATE_METHOD::UNCONSTRAINED)
          update::update_factor_unconstrained(val.ktensor.get_factor(n), val.gramians[n]);
        else
          update::update_factor_non_negative_constrained(val.ktensor.get_factor(n), val.gramians[n],
                                                         val.ktensor.get_active_set(n));

        if (val.ktensor.is_jk() && val.ktensor.get_jk_mode() == n)
          val.ktensor.set_jk_fiber(0.0);

        val.ktensor.normalize(n, val.ktensor.get_iters());

        ops::update_gramian(val.ktensor.get_factor(n), val.gramians[n]);
      }

      if (rep.cuda) {
#if CUDA_ENABLED
        mkt.get_factor(n).send_to_device();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }

      mode_timers[MODE_TIMERS::UPDATE].stop();

#if WITH_TIME
      rep.flops_per_iteration[rep.iter - 1] += mttkrp_params.flops;
      for (auto i = 0; i < MODE_TIMERS::LENGTH; i++)
        mode_times(n * MODE_TIMERS::LENGTH + i, iter_index) = mode_timers[i].get_time();
      for (auto i = 0; i < MTTKRP_TIMERS::LENGTH; i++)
        mttkrp_times(n * MTTKRP_TIMERS::LENGTH + i, iter_index) = mttkrp_params.mttkrp_timers[i].get_time();
#endif
    } // end for mode

    // Compute the approximation error for each model in the registry.
    als_timers[ALS_TIMERS::ERROR].start();

#pragma omp parallel for // NOLINT(openmp-use-default-none)
    for (auto i = 0lu; i < keys_v.size(); i++) {
      auto &val = registry.at(keys_v[i]);

      cals::ops::hadamard_all(val.gramians);

      // Create a view in the buffer for the last G matrix of the specific Ktensor
      const auto &kt_G_last = Matrix(val.ktensor.get_last_factor().get_rows(), val.ktensor.get_components(),
                                     G_last.get_data() + (val.col - mkt.get_start()) * G_last.get_col_stride());

      auto X_norm_kt = rep.X_norm;
      if (val.ktensor.is_jk())
        X_norm_kt = X_norms_jk[val.ktensor.get_jk_fiber()];

      const auto error = cals::error::compute_fast_error(X_norm_kt, val.ktensor.get_lambda(),
                                                         val.ktensor.get_last_factor(), kt_G_last, val.gramians[0]);

      // Ensure error always decreases (first iteration excluded due to undefined initial error)
      assert((val.ktensor.get_iters() == 1) || (val.ktensor.get_approximation_error() - error >= -1e-4));

      val.ktensor.set_approximation_error(error);
      val.ktensor.calculate_new_fit(rep.X_norm);
    }

    als_timers[ALS_TIMERS::ERROR].stop();

    // If line search is enabled, compute the line search extrapolation for each model in the registry.
    als_timers[ALS_TIMERS::LINE_SEARCH].start();

    if (rep.line_search) {
#pragma omp parallel for
      for (auto i = 0lu; i < keys_v.size(); i++) {
        auto &val = registry.at(keys_v[i]);
        // Make sure extrapolation doesn't happen right before a Ktensor is evicted (only if error is not going to be
        // checked).
        if (!(val.ls_params.method == ls::NO_ERROR_CHECKING && val.ktensor.get_iters() >= rep.max_iter)) {
          if (cals_params.line_search_step == 0)
            val.ls_params.step = std::cbrt(val.ktensor.get_iters());
          ls::line_search(val.ktensor, val.gramians, val.ls_params);

          if (val.ls_params.extrapolated) {
#pragma omp atomic
            rep.ls_performed++;
          }
          if (val.ls_params.reversed) {
#pragma omp atomic
            rep.ls_failed++;
          }
        }
      }
    }

    als_timers[ALS_TIMERS::LINE_SEARCH].stop();

    // Figure out which models should get evicted in this iteration and add them to remove_ids.
    remove_ids.clear();
    for (auto &[key, val] : registry)
      if (!cals_params.always_evict_first) {
        if (!cals_params.force_max_iter) {
          if (val.ktensor.get_fit_diff() < rep.tol || val.ktensor.get_iters() >= rep.max_iter)
            remove_ids.push_back(key);
          else
            val.ktensor.set_iters(val.ktensor.get_iters() + 1);
        } else if (val.ktensor.get_iters() >= rep.max_iter)
          remove_ids.push_back(key);
        else
          val.ktensor.set_iters(val.ktensor.get_iters() + 1);
      } else // Evict leftmost model for experiments
      {
        int id = mkt.get_leftmost_id();
        if (id != -1)
          remove_ids.push_back(id);
        break;
      }

    // Remove the IDs of the models in remove_ids from the registry.
    for (auto &key : remove_ids)
      mkt.remove(key);

    // Compress the buffers, since models might have been removed from them and they might be fragmented.
    als_timers.timers[ALS_TIMERS::DEFRAGMENTATION].start();
    mkt.compress();
    als_timers.timers[ALS_TIMERS::DEFRAGMENTATION].stop();

    als_timers[ALS_TIMERS::ITERATION].stop();

#if WITH_TIME
    for (dim_t i = 0; i < ALS_TIMERS::LENGTH; i++)
      als_times(i, rep.iter - 1) = als_timers[i].get_time();
#endif

#ifndef NDEBUG
    cout << "CONVERGENCE " << rep.iter << endl;
    for (auto &[key, val] : registry)
      cout << val.ktensor.get_approximation_error() << " ";
    cout << endl;
#endif

    if (kt_queue.empty() && registry.empty())
      converged = true;

  } while (!converged);

#if CUDA_ENABLED
  cuda::destroy_stream(stream);
#endif

  total_time.stop();
  rep.total_time = total_time.get_time();
  // cout << "Computation time: " << total_time.get_time() << endl;

  DEBUG(cout << "done." << endl;)

  return rep;
}

JKReport jk_cp_cals(const Tensor &X, vector<Ktensor> &kt_vector, CalsParams &cals_params) {

  auto ktensors(kt_vector);
  auto n_ktensors = ktensors.size();

  for (auto &ktensor : ktensors) {
    ktensor.denormalize();
    ktensor.normalize();
  }

  Timer pre_cals_time;
  pre_cals_time.start();
  vector<vector<Ktensor>> jk_input(n_ktensors);
  dim_t tensor_mode_0 = X.get_modes()[0];
  auto idx = 0;
  for (auto &ktensor : ktensors) {
    jk_input.reserve(n_ktensors * tensor_mode_0);
    cals::utils::generate_jk_ktensors(ktensor, jk_input[idx++]);
  }

  // Create a queue of references to be fed to cp_cals
  cals::KtensorQueue cals_queue;
  for (auto &k : jk_input)
    for (auto &m : k)
      cals_queue.emplace(m);
  pre_cals_time.stop();

  // Call driver
  //  cout << "OpenMP threads: " << get_threads() << endl;
  Timer cals_time;
  cals_time.start();
  auto report = cals::cp_cals(X, cals_queue, cals_params);
  cals_time.stop();

  for (auto &k : jk_input)
    for (auto &m : k) {
      m.set_jk_fiber(0.0);
      m.denormalize();
      m.normalize();
      m.set_jk_fiber(NAN);
    }

  idx = 0;
  for (auto &ktensor : ktensors)
    utils::jk_permutation_adjustment(ktensor, jk_input[idx++]);

  JKTime jk_time{pre_cals_time.get_time(), cals_time.get_time()};
  JKReport jk_report{jk_time, std::move(jk_input)};
  return jk_report;
}

} // namespace cals