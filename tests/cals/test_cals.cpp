#include "gtest/gtest.h"

#include "als.h"
#include "cals.h"
#include <iostream>

#define MODEL_DIFF_ACC 1e-11

class CalsGeneralTests : public ::testing::TestWithParam<bool> {};
class CalsLineSearchTests : public ::testing::TestWithParam<std::tuple<bool, cals::ls::LS_METHOD>> {};
class CalsJackknifingTests : public ::testing::TestWithParam<bool> {};

TEST_P(CalsGeneralTests, SimpleCorrectness) {
  // parameter = GetParam()
  // You can pass a CalsParams object or a tuple of values and access them with std::get<1>(GetParam())

  vector<int> ranks;
  for (auto rank = 1; rank <= 12; rank++)
    for (auto copies = 0; copies < 30; copies++)
      ranks.push_back(rank);
  //  std::sort(ranks.begin(), ranks.end());
  std::default_random_engine generator;
  std::shuffle(ranks.begin(), ranks.end(), generator);

  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::mttkrp::AUTO;
  cals_params.max_iterations = 1000;
  cals_params.tol = 1e-5;
  cals_params.buffer_size = 30;
  cals_params.line_search = false;
  cals_params.cuda = GetParam();

  cals::AlsParams als_params;
  als_params.mttkrp_method = cals_params.mttkrp_method;
  als_params.max_iterations = cals_params.max_iterations;
  als_params.tol = cals_params.tol;
  als_params.line_search = cals_params.line_search;
  als_params.cuda = cals_params.cuda;
  als_params.suppress_lut_warning = true;

  std::mt19937 reproducible_generator(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  cals::Ktensor P(10, {13, 12, 11});
  P.fill((function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  cals::Tensor T = P.to_tensor();

  auto const modes = T.get_modes();

  auto n_ktensors = ranks.size();
  vector<cals::Ktensor> ktensor_vector(n_ktensors);

  auto i = 0;
  for (auto &ktensor : ktensor_vector) {
    ktensor = cals::Ktensor(ranks[i++], modes);

    ktensor.fill(
        (function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  }
  auto als_input(ktensor_vector);
  auto als_omp_input(ktensor_vector);
  auto cals_input(ktensor_vector);

  cals::KtensorQueue cals_queue;
  for (auto p = 0lu; p < n_ktensors; p++)
    cals_queue.emplace(cals_input[p]);
  auto crep = cals::cp_cals(T, cals_queue, cals_params);

  vector<cals::AlsReport> alr(n_ktensors);
  for (auto p = 0lu; p < n_ktensors; p++)
    alr[p] = cals::cp_als(T, als_input[p], als_params);

  cals::cp_omp_als(T, als_omp_input, als_params);

  for (auto p = 0lu; p < n_ktensors; p++) {
    auto t1 = als_input[p].to_tensor();
    auto t2 = cals_input[p].to_tensor();
    auto t3 = als_omp_input[p].to_tensor();
    for (dim_t e = 0; e < t1.get_n_elements(); e++)
      t1[e] -= t2[e];
    for (dim_t e = 0; e < t1.get_n_elements(); e++)
      t2[e] -= t3[e];
    EXPECT_NEAR(t1.norm(), 0, MODEL_DIFF_ACC);
    EXPECT_NEAR(t2.norm(), 0, MODEL_DIFF_ACC);
  }
};

TEST_P(CalsLineSearchTests, LineSearchCorrectness) {
  // parameter = GetParam()
  // You can pass a CalsParams object or a tuple of values and access them with std::get<1>(GetParam())

  vector<int> ranks;
  for (auto rank = 1; rank <= 12; rank++)
    for (auto copies = 0; copies < 30; copies++)
      ranks.push_back(rank);
  //  std::sort(ranks.begin(), ranks.end());
  std::default_random_engine generator;
  std::shuffle(ranks.begin(), ranks.end(), generator);

  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::mttkrp::AUTO;
  cals_params.max_iterations = 1000;
  cals_params.tol = 1e-5;
  cals_params.buffer_size = 30;
  cals_params.line_search = true;
  cals_params.line_search_interval = 10;
  cals_params.line_search_step = 0; // Use step = (iteration)^(1/3)
  cals_params.line_search_method = std::get<1>(GetParam());
  cals_params.cuda = std::get<0>(GetParam());

  cals::AlsParams als_params;
  als_params.mttkrp_method = cals_params.mttkrp_method;
  als_params.max_iterations = cals_params.max_iterations;
  als_params.tol = cals_params.tol;
  als_params.line_search = cals_params.line_search;
  als_params.line_search_interval = cals_params.line_search_interval;
  als_params.line_search_step = cals_params.line_search_step;
  als_params.line_search_method = cals_params.line_search_method;
  als_params.cuda = cals_params.cuda;
  als_params.suppress_lut_warning = true;

  std::mt19937 reproducible_generator(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  cals::Ktensor P(10, {13, 12, 11});
  P.fill((function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  cals::Tensor T = P.to_tensor();

  auto const modes = T.get_modes();

  auto n_ktensors = ranks.size();
  vector<cals::Ktensor> ktensor_vector(n_ktensors);

  auto i = 0;
  for (auto &ktensor : ktensor_vector) {
    ktensor = cals::Ktensor(ranks[i++], modes);

    ktensor.fill(
        (function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  }
  auto als_input(ktensor_vector);
  auto als_omp_input(ktensor_vector);
  auto cals_input(ktensor_vector);

  cals::KtensorQueue cals_queue;
  for (auto p = 0lu; p < n_ktensors; p++)
    cals_queue.emplace(cals_input[p]);
  auto crep = cals::cp_cals(T, cals_queue, cals_params);

  vector<cals::AlsReport> alr(n_ktensors);
  for (auto p = 0lu; p < n_ktensors; p++)
    alr[p] = cals::cp_als(T, als_input[p], als_params);

  cals::cp_omp_als(T, als_omp_input, als_params);

  // Line Search Details
  dim_t perf = 0;
  dim_t fail = 0;
  for (auto p = 0lu; p < n_ktensors; p++) {
    perf += alr[p].ls_performed;
    fail += alr[p].ls_failed;
  }
  std::cout << " ALS LS perf: " << perf << " Failed: " << fail << " Ratio: " << (double)fail / perf << std::endl;
  std::cout << "CALS LS perf: " << crep.ls_performed << " Failed: " << crep.ls_failed
            << " Ratio: " << (double)crep.ls_failed / crep.ls_performed << std::endl;

  for (auto p = 0lu; p < n_ktensors; p++) {
    auto t1 = als_input[p].to_tensor();
    auto t2 = cals_input[p].to_tensor();
    auto t3 = als_omp_input[p].to_tensor();
    for (dim_t e = 0; e < t1.get_n_elements(); e++)
      t1[e] -= t2[e];
    for (dim_t e = 0; e < t1.get_n_elements(); e++)
      t2[e] -= t3[e];
    EXPECT_NEAR(t1.norm(), 0, MODEL_DIFF_ACC);
    EXPECT_NEAR(t2.norm(), 0, MODEL_DIFF_ACC);
  }
  //  std::cout << count << " " << (double) count / 600 << " " << (double) comp / count << std::endl;
};

TEST_P(CalsJackknifingTests, LogicCorrectness) {
  // parameter = GetParam()
  // You can pass a CalsParams object or a tuple of values and access them with std::get<1>(GetParam())
  std::mt19937 reproducible_generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  dim_t n_ktensors = 20;
  dim_t components = 5;
  vector<dim_t> modes = {20, 9, 12};

  cals::Ktensor P(components, modes);
  P.fill((function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  cals::Tensor T = P.to_tensor();
  auto T_0 = cals::Matrix(modes[0], modes[1] * modes[2], T.get_data());

  auto jk_modes(modes);
  jk_modes[0] -= 1;

  auto ktensor_ref = cals::Ktensor(components, modes);
  ktensor_ref.fill(
      (function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });

  // Create new versions of the original tensor and the corresponding JK ktensors
  vector<cals::Tensor> tensor_als_vector(n_ktensors);
  vector<cals::Ktensor> ktensor_als_vector(n_ktensors);
  dim_t i_jk = 0;
  for (auto &ktensor : ktensor_als_vector) {
    ktensor = cals::Ktensor(components, jk_modes);

    ktensor.get_lambda() = ktensor_ref.get_lambda();
    for (dim_t f = 0; f < ktensor.get_n_modes(); f++) {
      auto &factor_source = ktensor_ref.get_factor(f);
      auto &factor_destin = ktensor.get_factor(f);
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

    auto &T_jk = tensor_als_vector[i_jk];
    T_jk = cals::Tensor(jk_modes);
    auto T_jk_0 = cals::Matrix(jk_modes[0], jk_modes[1] * jk_modes[2], T_jk.get_data());

    for (dim_t jj = 0; jj < T_0.get_cols(); jj++)
      for (dim_t ii = 0; ii < T_0.get_rows(); ii++)
        if (ii < i_jk)
          T_jk_0(ii, jj) = T_0(ii, jj);
        else if (ii > i_jk)
          T_jk_0(ii - 1, jj) = T_0(ii, jj);
    i_jk++;
  }

  // Create the different JK ktensors for CALS (no tensor copies required in this case, obviously)
  i_jk = 0;
  vector<cals::Ktensor> ktensor_cals_vector(n_ktensors);
  for (auto &ktensor : ktensor_cals_vector) {
    ktensor = cals::Ktensor(components, modes);
    ktensor.copy(ktensor_ref);
    ktensor.to_jk(0, i_jk);
    ktensor.set_jk_fiber(0.0);
    i_jk++;
  }

  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::mttkrp::AUTO;
  cals_params.max_iterations = 1000;
  cals_params.tol = 1e-4;
  cals_params.buffer_size = 18;
  cals_params.line_search = false;
  cals_params.force_max_iter = true;
  cals_params.cuda = GetParam();

  cals::AlsParams als_params;
  als_params.mttkrp_method = cals_params.mttkrp_method;
  als_params.max_iterations = cals_params.max_iterations;
  als_params.tol = cals_params.tol;
  als_params.line_search = cals_params.line_search;
  als_params.cuda = cals_params.cuda;
  als_params.force_max_iter = cals_params.force_max_iter;
  als_params.suppress_lut_warning = true;

  auto ktensor_ref_als = cals::Ktensor(ktensor_ref);
  auto ktensor_ref_cals = cals::Ktensor(ktensor_ref);

  cals::KtensorQueue cals_queue;
  for (auto p = 0lu; p < n_ktensors; p++)
    cals_queue.emplace(ktensor_cals_vector[p]);
  cals_queue.emplace(ktensor_ref_cals);
  auto crep = cals::cp_cals(T, cals_queue, cals_params);

  vector<cals::AlsReport> alr(n_ktensors);
  for (auto p = 0lu; p < n_ktensors; p++)
    alr[p] = cals::cp_als(tensor_als_vector[p], ktensor_als_vector[p], als_params);
  auto alr_ref = cals::cp_als(T, ktensor_ref_als, als_params);

  for (auto p = 0lu; p < n_ktensors; p++) {
    auto t1 = ktensor_cals_vector[p].to_regular().to_tensor();
    auto t2 = ktensor_als_vector[p].to_tensor();
    for (dim_t e = 0; e < t1.get_n_elements(); e++)
      t1[e] -= t2[e];
    //    if (td::fabs(t1.norm()) > MODEL_DIFF_ACC)
    //      std::cout << ktensor_als_vector[p].get_iters() << " " << ktensor_cals_vector[p].get_iters() << std::endl;
    EXPECT_NEAR(t1.norm(), 0.0, MODEL_DIFF_ACC);
  }
  auto t1 = ktensor_ref_als.to_tensor();
  auto t2 = ktensor_ref_cals.to_tensor();
  for (dim_t e = 0; e < t1.get_n_elements(); e++)
    t1[e] -= t2[e];
  //    if (td::fabs(t1.norm()) > MODEL_DIFF_ACC)
  //      std::cout << ktensor_als_vector[p].get_iters() << " " << ktensor_cals_vector[p].get_iters() << std::endl;
  EXPECT_NEAR(t1.norm(), 0.0, MODEL_DIFF_ACC);
};

TEST_P(CalsJackknifingTests, FunctionCorrectness) {
  // parameter = GetParam()
  // You can pass a CalsParams object or a tuple of values and access them with std::get<1>(GetParam())
  std::mt19937 reproducible_generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  dim_t n_ktensors = 4;
  vector<dim_t> modes = {10, 21, 20};
  dim_t components = 5;

  cals::Ktensor P(components, modes);
  P.fill((function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  cals::Tensor T = P.to_tensor();
  auto T_0 = cals::Matrix(modes[0], modes[1] * modes[2], T.get_data());

  vector<cals::Ktensor> ktensor_ref_v(n_ktensors);
  for (auto &kt : ktensor_ref_v) {
    kt = cals::Ktensor(components, modes);
    kt.fill((function<double()> &&)[&dist, &reproducible_generator]()->double { return dist(reproducible_generator); });
  }

  cals::CalsParams cals_params;
  cals_params.mttkrp_method = cals::mttkrp::AUTO;
  cals_params.max_iterations = 1000;
  cals_params.tol = 1e-4;
  cals_params.buffer_size = 18;
  cals_params.line_search = false;
  cals_params.force_max_iter = true;
  cals_params.cuda = GetParam();

  cals::AlsParams als_params;
  als_params.mttkrp_method = cals_params.mttkrp_method;
  als_params.max_iterations = cals_params.max_iterations;
  als_params.tol = cals_params.tol;
  als_params.line_search = cals_params.line_search;
  als_params.cuda = cals_params.cuda;
  als_params.force_max_iter = cals_params.force_max_iter;
  als_params.suppress_lut_warning = true;

  for (auto &kt : ktensor_ref_v)
    auto first_rep = cals::cp_als(T, kt, als_params);

  auto cals_rep = jk_cp_cals(T, ktensor_ref_v, cals_params);
  auto als_rep = jk_cp_als(T, ktensor_ref_v, als_params);
  auto als_omp_rep = jk_cp_omp_als(T, ktensor_ref_v, als_params);

  std::cout << cals_rep.jk_time.pre_als_time << " " << als_omp_rep.jk_time.pre_als_time << " " << als_rep.jk_time.pre_als_time << std::endl;
  std::cout << cals_rep.jk_time.als_time << " " << als_omp_rep.jk_time.als_time << " " << als_rep.jk_time.als_time << std::endl;
  for (auto p = 0lu; p < n_ktensors; p++)
    for (auto m = 0; m < T.get_modes()[0]; m++)
    {
      auto t1 = cals_rep.results[p][m].to_regular().to_tensor();
      auto t2 = als_rep.results[p][m].to_tensor();
      auto t3 = als_omp_rep.results[p][m].to_tensor();
      for (dim_t e = 0; e < t1.get_n_elements(); e++)
        t1[e] -= t2[e];
      for (dim_t e = 0; e < t2.get_n_elements(); e++)
        t2[e] -= t3[e];
      //    if (td::fabs(t1.norm()) > MODEL_DIFF_ACC)
      //      std::cout << ktensor_als_vector[p].get_iters() << " " << ktensor_cals_vector[p].get_iters() << std::endl;
      EXPECT_NEAR(t1.norm(), 0.0, MODEL_DIFF_ACC);
      EXPECT_NEAR(t2.norm(), 0.0, MODEL_DIFF_ACC);
    }
};

INSTANTIATE_TEST_SUITE_P(CalsGeneralTest, CalsGeneralTests, ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(CalsLineSearchTestNoErr,
                         CalsLineSearchTests,
                         ::testing::Values(std::make_tuple(false, cals::ls::NO_ERROR_CHECKING)));
INSTANTIATE_TEST_SUITE_P(CalsLineSearchTestErr,
                         CalsLineSearchTests,
                         ::testing::Values(std::make_tuple(false, cals::ls::ERROR_CHECKING_SERIAL)));
INSTANTIATE_TEST_SUITE_P(CalsJackknifingTest, CalsJackknifingTests, ::testing::Values(false));

#if CUDA_ENABLED
INSTANTIATE_TEST_SUITE_P(CalsCUDAGeneralTest, CalsGeneralTests, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(CalsCUDALineSearchTestNoErr,
                         CalsLineSearchTests,
                         ::testing::Values(std::make_tuple(true, cals::ls::NO_ERROR_CHECKING)));
INSTANTIATE_TEST_SUITE_P(CalsCUDALineSearchTestErr,
                         CalsLineSearchTests,
                         ::testing::Values(std::make_tuple(true, cals::ls::ERROR_CHECKING_SERIAL)));
INSTANTIATE_TEST_SUITE_P(CalsCUDAJackknifingTest, CalsJackknifingTests, ::testing::Values(true));
#endif

int main(int argc, char **argv) {
  set_threads(4);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
