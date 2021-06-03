#include "als.h"
#include <iomanip>

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
AlsReport cp_als(const Tensor &X, Ktensor &ktensor, AlsParams &als_params) {

#ifndef NDEBUG
  cout << "START method=Regular_ALS" << endl;
  cout << "Tensor size: ";
  for (auto mode : X.get_modes())
    cout << mode << " ";
  cout << endl;
  cout << "Ktensor components= " << ktensor.get_components() << endl;
#endif

  AlsReport rep;

  rep.tensor_rank = X.get_rank();
  rep.n_modes = X.get_n_modes();
  rep.modes = X.get_modes();
  rep.X_norm = X.norm();

  rep.iter = 0;
  rep.max_iter = als_params.max_iterations;
  rep.n_threads = get_threads();
  rep.ktensor_id = ktensor.get_id();
  rep.ktensor_components = static_cast<dim_t>(ktensor.get_components());
  rep.tol = als_params.tol;
  rep.cuda = als_params.cuda;
  rep.update_method = als_params.update_method;
  rep.line_search = als_params.line_search;
  rep.line_search_interval = als_params.line_search_interval;
  rep.line_search_step = als_params.line_search_step;
  rep.line_search_method = als_params.line_search_method;

  assert(rep.modes.size() >= 3);

#if WITH_TIME
  rep.als_times = Matrix(ALS_TIMERS::LENGTH, rep.max_iter);
  rep.mode_times = Matrix(MODE_TIMERS::LENGTH * rep.n_modes, rep.max_iter);
  rep.mttkrp_times = Matrix(MTTKRP_TIMERS::LENGTH * rep.n_modes, rep.max_iter);
  auto &als_times = rep.als_times;
  auto &mode_times = rep.mode_times;
  auto &mttkrp_times = rep.mttkrp_times;
#endif

  Timer total_time;
  AlsTimers als_timers;
  ModeTimers mode_timers;

  using ALS_TIMERS = cals::AlsTimers::TIMERS;

  total_time.start();

  // Create empty matrix to store last G for error calculation
  Matrix G_last(rep.modes[rep.n_modes - 1], ktensor.get_components());

  // Calculate and Allocate Workspaces for intermediate KRP and Twostep
  vector<dim_t> sorted_modes = rep.modes; // Create a copy to sort it
  std::sort(sorted_modes.begin(), sorted_modes.end(), std::greater<>());
  auto n_half_modes = static_cast<size_t>(std::ceil((rep.modes.size() - 1.0) / 2.0));
  vector<dim_t> ws_sizes(n_half_modes);            // ws_sizes holds the nrows of each workspaces matrix for KRP
  ws_sizes[0] = sorted_modes[0] * sorted_modes[1]; // Two largest sorted_modes (Can be used as a workspaces for twostep)
  for (auto i = 1lu; i < ws_sizes.size(); i++)
    ws_sizes[i] = sorted_modes[i + 1] * ws_sizes[i - 1];

  vector<Matrix> workspaces(ws_sizes.size());
  for (auto i = 0lu; i < ws_sizes.size(); i++)
    workspaces[i] = Matrix(ws_sizes[i], rep.ktensor_components);

  // Create and initialize MTTKRP parameters
  cals::mttkrp::MttkrpParams mttkrp_params;
  mttkrp_params.method = als_params.mttkrp_method;
  mttkrp_params.lut = als_params.mttkrp_lut;
  mttkrp_params.krp_params.cuda = rep.cuda;
  mttkrp_params.cuda = rep.cuda;

  if (mttkrp_params.lut.keys_v.empty()) {
    Timer lut_timer;
    lut_timer.start();
    if (!als_params.suppress_lut_warning)
      std::cerr << "LUT empty" << endl;
    mttkrp_params.lut = mttkrp::read_lookup_table(rep.modes, get_threads(), rep.cuda, als_params.suppress_lut_warning);
    lut_timer.stop();
    //    std::cout << "Lookup table creation took " << lut_timer.get_time() << "seconds." << std::endl;
  }

  // Create and initialize Line Search parameters
  ls::LineSearchParams ls_params{};
  if (rep.line_search) {
    ls_params.prev_ktensor = Ktensor(static_cast<int>(rep.ktensor_components), rep.modes);
    ls_params.backup_ktensor = Ktensor(static_cast<int>(rep.ktensor_components), rep.modes);
    ls_params.method = rep.line_search_method;
    ls_params.interval = rep.line_search_interval;
    ls_params.step = rep.line_search_step;
    ls_params.cuda = rep.cuda;
    ls_params.T = &X;
  }

  // Allocate and Initialize Gramians vector
  auto gramians = vector<Matrix>(rep.n_modes);
  for (dim_t n = 0; n < rep.n_modes; n++)
    gramians[n] = Matrix(rep.ktensor_components, rep.ktensor_components);
  for (dim_t n = 0; n < rep.n_modes; n++)
    cals::ops::update_gramian(ktensor.get_factor(n), gramians[n]);

#if CUDA_ENABLED
  auto stream = cuda::create_stream();
#endif

  if (rep.cuda) {
#if CUDA_ENABLED
    if (!als_params.cuda_no_tensor_alloc) {
      if (X.get_cudata() == nullptr) {
        X.allocate_cudata(X.get_n_elements());
        X.send_to_device();
      }
    }

    for (int i = 0; i < ktensor.get_n_modes(); i++) {
      ktensor.get_factor(i).allocate_cudata(ktensor.get_factor(i).get_n_elements());
      ktensor.get_factor(i).send_to_device();
    }

    for (auto &w : workspaces)
      w.allocate_cudata(w.get_n_elements());
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  bool converged{false};
  rep.iter = 0;
  ktensor.set_iters(0);
  do {
    rep.iter++;
    ktensor.set_iters(ktensor.get_iters() + 1);
    rep.flops_per_iteration = 0;

    als_timers[ALS_TIMERS::ITERATION].start();

    // If line search is enabled, check to see if it is time to store the previous version of a model.
    if (rep.line_search && ls_params.iter == (ls_params.interval - 1))
      ls_params.prev_ktensor.copy(ktensor);

    // Loop over the modes
    for (dim_t n = 0; n < rep.n_modes; n++) {
      auto &factor = ktensor.get_factor(n);

      // Compute the MTTKRP for the current mode.
      mode_timers[MODE_TIMERS::MTTKRP].start();

      auto &G = mttkrp::mttkrp(X, ktensor, workspaces, n, mttkrp_params);

      mode_timers[MODE_TIMERS::MTTKRP].stop();

      // Save the last G matrix for error calculation
      if (n == rep.n_modes - 1) {
        als_timers[ALS_TIMERS::G_COPY].start();
        G_last.copy(G);
        als_timers[ALS_TIMERS::G_COPY].stop();
      }

      // Update the factor matrix for the current mode.
      mode_timers[MODE_TIMERS::UPDATE].start();

      auto &H = cals::ops::hadamard_but_one(gramians, n);

      if (rep.update_method == cals::update::UPDATE_METHOD::UNCONSTRAINED)
        cals::update::update_factor_unconstrained(factor, H);
      else
        cals::update::update_factor_non_negative_constrained(factor, H, ktensor.get_active_set(n));

      if (ktensor.is_jk() && ktensor.get_jk_mode() == n)
        ktensor.set_jk_fiber(0.0);

      ktensor.normalize(n, ktensor.get_iters());

      if (rep.cuda) {
#if CUDA_ENABLED
        ktensor.get_factor(n).send_to_device_async(stream);
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }

      cals::ops::update_gramian(factor, gramians[n]);

      if (rep.cuda) {
#if CUDA_ENABLED
        cudaStreamSynchronize(stream);
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }

      mode_timers[MODE_TIMERS::UPDATE].stop();

#if WITH_TIME
      rep.flops_per_iteration += mttkrp_params.flops;
      auto iter_index = static_cast<dim_t>(rep.iter - 1);
      for (auto i = 0; i < MODE_TIMERS::LENGTH; i++)
        mode_times(n * MODE_TIMERS::LENGTH + i, iter_index) = mode_timers[i].get_time();
      for (auto i = 0; i < MTTKRP_TIMERS::LENGTH; i++)
        mttkrp_times(n * MTTKRP_TIMERS::LENGTH + i, iter_index) = mttkrp_params.mttkrp_timers[i].get_time();
#endif
    } // end for mode

    // Compute the approximation error
    als_timers[ALS_TIMERS::ERROR].start();

    cals::ops::hadamard_all(gramians);
    const auto error = cals::error::compute_fast_error(rep.X_norm, ktensor.get_lambda(), ktensor.get_last_factor(),
                                                       G_last, gramians[0]);

    // Ensure error always decreases (first iteration excluded due to undefined initial error)
    if (rep.iter != 1 && (ktensor.get_approximation_error() - error < -1e-4))
      std::cerr << std::scientific << "error incr Ktensor: " << ktensor.get_id() << " Iter: " << std::setw(3)
                << ktensor.get_iters() << " || Error: " << std::setw(5) << error << " || Old Error: " << std::setw(5)
                << ktensor.get_approximation_error() << " || Diff: " << std::setw(5)
                << ktensor.get_approximation_error() - error << endl;

    ktensor.set_approximation_error(error);
    ktensor.calculate_new_fit(rep.X_norm);

    als_timers[ALS_TIMERS::ERROR].stop();

    // If line search is enabled, compute the line search extrapolation for each model in the registry.
    als_timers[ALS_TIMERS::LINE_SEARCH].start();

    if (rep.line_search) {
      // Make sure extrapolation doesn't happen right before the Ktensor is evicted, if error is not going to be
      // checked.
      if (!(ls_params.method == ls::NO_ERROR_CHECKING && ktensor.get_iters() >= rep.max_iter)) {
        if (als_params.line_search_step == 0)
          ls_params.step = std::cbrt(ktensor.get_iters());
        ls::line_search(ktensor, gramians, ls_params);
        if (ls_params.extrapolated)
          rep.ls_performed++;
        if (ls_params.reversed)
          rep.ls_failed++;
      }
    }
    als_timers[ALS_TIMERS::LINE_SEARCH].stop();

    als_timers[ALS_TIMERS::ITERATION].stop();

#if WITH_TIME
    auto iter_index = static_cast<dim_t>(rep.iter - 1);
    for (dim_t i = 0; i < ALS_TIMERS::LENGTH; i++)
      als_times(i, iter_index) = als_timers[i].get_time();
#endif

    DEBUG(cout << "CONVERGENCE " << rep.iter << " " << ktensor.get_approximation_error() << endl;)

    if (!als_params.force_max_iter)
      converged = (ktensor.get_fit_diff() < rep.tol) || (ktensor.get_iters() >= rep.max_iter);
    else
      converged = ktensor.get_iters() >= rep.max_iter;
  } while (!converged);

#if CUDA_ENABLED
  cuda::destroy_stream(stream);
#endif

  total_time.stop();
  rep.total_time = total_time.get_time();

  DEBUG(cout << "done." << endl;)

  return rep;
}

JKReport jk_cp_als(const Tensor &X, vector<Ktensor> &kt_vector, AlsParams &als_params) {

  auto modes = X.get_modes();
  auto X_0 = cals::Matrix(modes[0], modes[1] * modes[2], X.get_data());

  auto ktensors(kt_vector);
  auto n_ktensors = ktensors.size();

  auto jk_modes(modes);
  jk_modes[0] -= 1;

  dim_t tensor_mode_0 = X.get_modes()[0];

  for (auto &ktensor : ktensors) {
    ktensor.denormalize();
    ktensor.normalize();
  }

  vector<vector<JKTime>> jk_time_v(n_ktensors);
  for (auto &k : jk_time_v)
    k.resize(tensor_mode_0);

  vector<vector<Ktensor>> jk_input(n_ktensors);
  for (auto &k : jk_input)
    k.resize(tensor_mode_0);

  // Create new versions of the original tensor and the corresponding JK ktensors
  for (auto i_kt = 0; i_kt < n_ktensors; i_kt++) {
    auto &ktensor = ktensors[i_kt];
    auto components = ktensor.get_components();
    for (auto i_jk = 0; i_jk < tensor_mode_0; i_jk++) {
      Timer pre_als_timer;
      pre_als_timer.start();

      auto ktensor_jk = cals::Ktensor(components, jk_modes);

      ktensor_jk.get_lambda() = ktensor.get_lambda();
      for (dim_t f = 0; f < ktensor.get_n_modes(); f++) {
        auto &factor_source = ktensor.get_factor(f);
        auto &factor_destin = ktensor_jk.get_factor(f);
        if (f == 0) {
          for (dim_t jj = 0; jj < factor_source.get_cols(); jj++)
            for (dim_t ii = 0; ii < factor_source.get_rows(); ii++)
              if (ii < i_jk)
                factor_destin(ii, jj) = factor_source(ii, jj);
              else if (ii > i_jk)
                factor_destin(ii - 1, jj) = factor_source(ii, jj);
        } else
          factor_destin.copy(factor_source);
      }
      jk_input[i_kt][i_jk] = std::move(ktensor_jk);

      // Create tensor subsample
      auto X_jk = cals::Tensor(jk_modes);
      auto X_jk_0 = cals::Matrix(jk_modes[0], jk_modes[1] * jk_modes[2], X_jk.get_data());
      for (dim_t jj = 0; jj < X_0.get_cols(); jj++)
        for (dim_t ii = 0; ii < X_0.get_rows(); ii++)
          if (ii < i_jk)
            X_jk_0(ii, jj) = X_0(ii, jj);
          else if (ii > i_jk)
            X_jk_0(ii - 1, jj) = X_0(ii, jj);

      pre_als_timer.stop();
      auto pre_als_time = pre_als_timer.get_time();

      Timer als_timer;
      als_timer.start();
      auto report = cp_als(X_jk, jk_input[i_kt][i_jk], als_params);
      als_timer.stop();
      auto als_time = als_timer.get_time();

      jk_time_v[i_kt][i_jk] = {pre_als_time, als_time};
    };
  }

  for (auto &k : jk_input)
    for (auto &m : k) {
      m.denormalize();
      m.normalize();
    }

  auto idx = 0;
  for (auto &ktensor : ktensors)
    utils::jk_permutation_adjustment(ktensor, jk_input[idx++]);

  double pre_als_time = 0.0;
  double als_time = 0.0;
  for (auto &k : jk_time_v)
    for (auto &[pre_als_t, als_t] : k) {
      pre_als_time += pre_als_t;
      als_time += als_t;
    }

  JKTime jk_time{pre_als_time, als_time};
  JKReport jk_report{jk_time, std::move(jk_input)};
  return jk_report;
}
std::vector<AlsReport> cp_omp_als(const Tensor &X, std::vector<Ktensor> &ktensor_v, AlsParams &params) {
  cals::Timer als_omp_total_time;
  als_omp_total_time.start();

#if CUDA_ENABLED
  X.allocate_cudata(X.get_n_elements());
  X.send_to_device();
  auto old_cuda_no_tensor_alloc = params.cuda_no_tensor_alloc;
  params.cuda_no_tensor_alloc = true;
#endif

  vector<AlsReport> reports(ktensor_v.size());
#pragma omp parallel for
  for (auto i = 0lu; i < ktensor_v.size(); i++)
    reports[i] = cals::cp_als(X, ktensor_v[i], params);

#if CUDA_ENABLED
  params.cuda_no_tensor_alloc = old_cuda_no_tensor_alloc;
#endif

  als_omp_total_time.stop();
  for (auto &rep : reports)
    rep.total_time = als_omp_total_time.get_time();

  return reports;
}

JKReport jk_cp_omp_als(const Tensor &X, vector<Ktensor> &kt_vector, AlsParams &als_params) {

  auto modes = X.get_modes();
  auto X_0 = cals::Matrix(modes[0], modes[1] * modes[2], X.get_data());

  auto ktensors(kt_vector);
  auto n_ktensors = ktensors.size();

  auto jk_modes(modes);
  jk_modes[0] -= 1;

  dim_t tensor_mode_0 = X.get_modes()[0];

  for (auto &ktensor : ktensors) {
    ktensor.denormalize();
    ktensor.normalize();
  }

  vector<vector<Ktensor>> jk_input(n_ktensors);
  for (auto &k : jk_input)
    k.resize(tensor_mode_0);

  // auto mal = omp_get_max_active_levels();
  // std::cout << "Max active levels: "<< mal << std::endl;
  // auto nest = omp_get_nested();
  // std::cout << "Nested: "<< nest << std::endl;
  // omp_set_nested(true);
  // omp_set_max_active_levels(3);

  Timer als_omp_timer;
  als_omp_timer.start();
  // Create new versions of the original tensor and the corresponding JK ktensors
//#pragma omp parallel for  // Nested openmp leads to worse performance. Disabling.
  for (auto i_kt = 0; i_kt < n_ktensors; i_kt++) {
    auto &ktensor = ktensors[i_kt];
    auto components = ktensor.get_components();
#pragma omp parallel for
    for (auto i_jk = 0; i_jk < tensor_mode_0; i_jk++) {
      auto ktensor_jk = cals::Ktensor(components, jk_modes);

      ktensor_jk.get_lambda() = ktensor.get_lambda();
      for (dim_t f = 0; f < ktensor.get_n_modes(); f++) {
        auto &factor_source = ktensor.get_factor(f);
        auto &factor_destin = ktensor_jk.get_factor(f);
        if (f == 0) {
          for (dim_t jj = 0; jj < factor_source.get_cols(); jj++)
            for (dim_t ii = 0; ii < factor_source.get_rows(); ii++)
              if (ii < i_jk)
                factor_destin(ii, jj) = factor_source(ii, jj);
              else if (ii > i_jk)
                factor_destin(ii - 1, jj) = factor_source(ii, jj);
        } else
          factor_destin.copy(factor_source);
      }
      jk_input[i_kt][i_jk] = std::move(ktensor_jk);

      // Create tensor subsample
      auto X_jk = cals::Tensor(jk_modes);
      auto X_jk_0 = cals::Matrix(jk_modes[0], jk_modes[1] * jk_modes[2], X_jk.get_data());
      for (dim_t jj = 0; jj < X_0.get_cols(); jj++)
        for (dim_t ii = 0; ii < X_0.get_rows(); ii++)
          if (ii < i_jk)
            X_jk_0(ii, jj) = X_0(ii, jj);
          else if (ii > i_jk)
            X_jk_0(ii - 1, jj) = X_0(ii, jj);

      auto report = cp_als(X_jk, jk_input[i_kt][i_jk], als_params);
    };
  }
  als_omp_timer.stop();
  // omp_set_nested(false);
  // omp_set_max_active_levels(mal);

  for (auto &k : jk_input)
    for (auto &m : k) {
      m.denormalize();
      m.normalize();
    }

  auto idx = 0;
  for (auto &ktensor : ktensors)
    utils::jk_permutation_adjustment(ktensor, jk_input[idx++]);

  JKTime jk_time{NAN, als_omp_timer.get_time()};
  JKReport jk_report{jk_time, std::move(jk_input)};
  return jk_report;
}
} // namespace cals
