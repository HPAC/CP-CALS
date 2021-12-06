#include "utils/utils.h"

#include <iostream>
#include <numeric>

#include <rectangular_lsap/rectangular_lsap.h>

namespace cals::utils {
std::string mode_string(vector<dim_t> const &modes) {
  std::string m_string;
  for (auto const &m : modes)
    m_string += std::to_string(m) + '-';
  m_string.pop_back();

  return m_string;
}

Ktensor concatenate_ktensors(vector<Ktensor> const &ktensors) {

  auto const n_ktensors = ktensors.size();
  auto const components = ktensors[0].get_components();
  auto concatenated_ktensor = Ktensor(n_ktensors * components, ktensors[0].get_modes());

  dim_t index = 0;
  for (auto const &kt : ktensors) {
    for (dim_t i = 0; i < kt.get_components(); i++)
      concatenated_ktensor.get_lambda()[index * kt.get_components() + i] = kt.get_lambda()[i];
    for (dim_t m = 0; m < kt.get_n_modes(); m++) {
      auto const &ktensor_fac = kt.get_factor(m);
      auto &concat_ktensor_fac = concatenated_ktensor.get_factor(m);
      auto *const ckt_pos = concat_ktensor_fac.get_data() + (index * components) * concat_ktensor_fac.get_col_stride();
      Matrix(ktensor_fac.get_rows(), ktensor_fac.get_cols(), ckt_pos).copy(ktensor_fac);
    }
    index++;
  }

  return concatenated_ktensor;
}

void generate_jk_ktensors(cals::Ktensor const &reference_ktensor, std::vector<cals::Ktensor> &jk_ktensor_v) {

  auto const tensor_mode_0 = reference_ktensor.get_modes()[0];
  if (tensor_mode_0 <= 1)
    throw std::string("Can't do Jack-knife with just one sample.");

  for (dim_t i = 0; i < tensor_mode_0; i++) {
    Ktensor ktensor_copy(reference_ktensor);
    auto &jk_ktensor = ktensor_copy.to_jk(0, i);
    jk_ktensor_v.push_back(std::move(jk_ktensor));
  }
}

void jk_permutation_adjustment(cals::Ktensor &ktensor, std::vector<cals::Ktensor> &jk_ktensor_v) {

  auto modes = ktensor.get_modes();
  auto mode_0 = modes[0];
  auto mode_1 = modes[1];
  auto mode_2 = modes[2];
  auto &Bov = ktensor.get_factor(1);
  auto &Cov = ktensor.get_factor(2);

  for (dim_t m = 0; m < mode_0; m++) {
    auto &kt_jk = jk_ktensor_v[m];

    auto const comp = ktensor.get_components();

    auto M = cals::Matrix(comp, comp);
    auto Mt = cals::Matrix(comp, comp);

    auto &Bm = kt_jk.get_factor(1);
    auto &Cm = kt_jk.get_factor(2);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, comp, mode_1, 1.0, Bov.get_data(), Bov.get_col_stride(),
                Bm.get_data(), Bm.get_col_stride(), 0.0, M.get_data(), M.get_col_stride());
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, comp, comp, mode_2, 1.0, Cov.get_data(), Cov.get_col_stride(),
                Cm.get_data(), Cm.get_col_stride(), 0.0, Mt.get_data(), Mt.get_col_stride());
    for (dim_t ii = 0; ii < M.get_n_elements(); ii++)
      M[ii] += Mt[ii];

    vector<int64_t> init_v(comp);
    vector<int64_t> solved_v(comp);

    solve_rectangular_linear_sum_assignment(comp, comp, M.get_data(), true, init_v.data(), solved_v.data());

    for (dim_t mode = 0; mode < ktensor.get_n_modes(); mode++) {
      auto &factor = kt_jk.get_factor(mode);
      auto copy_factor = cals::Matrix(factor.get_rows(), factor.get_cols());
      copy_factor.copy(factor);
      auto const stride = factor.get_col_stride();

      auto curr_col_id = 0;
      for (auto &swap_col_id : solved_v) {
        if (swap_col_id != curr_col_id) {
          cals::Matrix(factor.get_rows(), 1, factor.get_data() + curr_col_id * stride)
              .copy(cals::Matrix(copy_factor.get_rows(), 1, copy_factor.get_data() + swap_col_id * stride));
        }
        curr_col_id++;
      }
    }
  }
}

vector<double> calculate_jackknifing_norms(cals::Tensor const &tensor) {

  auto modes = tensor.get_modes();
  auto cols = std::accumulate(modes.cbegin() + 1, modes.cend(), (dim_t)1, std::multiplies<>());
  auto tensor_0 = cals::Matrix(modes[0], cols, tensor.get_data());
  auto max_threads = omp_get_max_threads();

  vector<double> sum_of_squares(modes[0] * max_threads, 0.0);

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto max_tid = omp_get_num_threads() - 1;

    auto *soq_global = &(sum_of_squares[0]);
    auto *soq_local = &(sum_of_squares[tid * modes[0]]);

#pragma omp for
    for (dim_t j = 0; j < tensor_0.get_cols(); j++) {
      for (dim_t i = 0; i < tensor_0.get_rows(); i++)
        soq_local[i] += tensor_0(i, j) * tensor_0(i, j);
    }

#pragma omp barrier

    for (auto pair = 1; pair <= max_tid; pair++) {
      if (tid == 0)
        for (dim_t i = 0; i < tensor_0.get_rows(); i++)
          soq_local[i] += soq_global[(tid + pair) * modes[0] + i];
    }

    //      for (auto pair_index = 1; pair_index < (max_tid << 1); pair_index = pair_index << 1) {
    //      if (omp_get_thread_num() % (pair_index << 1) == 0 && tid < max_tid) {
    //        for (dim_t i = 0; i < tensor_0.get_rows(); i++)
    //          soq_local[i] += soq_global[(tid + pair_index) * modes[0] + i];
    //      }
  }

  sum_of_squares.resize(modes[0]);

  auto sum0 = std::accumulate(sum_of_squares.cbegin(), sum_of_squares.cend(), (double)0.0);

#pragma omp parallel for
  for (dim_t i = 0; i < tensor_0.get_rows(); i++) {
    sum_of_squares[i] = sum0 - sum_of_squares[i];
    sum_of_squares[i] = std::sqrt(sum_of_squares[i]);
  }
  // std::cout << "First sum of squares: "<< sum_of_squares[0] << std::endl;
  return sum_of_squares;
}
} // namespace cals::utils

namespace cals::ops {
void hadamard_all(vector<Matrix> &matrices) {
  for (auto i = 1lu; i < matrices.size(); i++) // skip first gramian
    matrices[0].hadamard(matrices[i]);
}

Matrix &hadamard_but_one(vector<Matrix> &matrices, dim_t mode) {
  // Initialize target matrix
  matrices[mode].fill((function<double()> const &&)[]()->double { return static_cast<double>(1.0); });

  for (auto i = 0lu; i < matrices.size(); i++) {
    if (i == static_cast<unsigned long int>(mode)) // ...except mode...
      continue;
    else // ATA[n] := ATA[n] .* ATA[k]
      matrices[mode].hadamard(matrices[i]);
  }
  return matrices[mode];
}

void update_gramian(const Matrix &factor, Matrix &gramian) {
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, gramian.get_rows(), gramian.get_cols(), factor.get_rows(), 1.0,
              factor.get_data(), factor.get_col_stride(), factor.get_data(), factor.get_col_stride(), 0.0,
              gramian.get_data(), gramian.get_col_stride());
}

void update_gramians(const cals::Ktensor &ktensor, vector<Matrix> &gramians) {
  for (dim_t i = 0; i < ktensor.get_n_modes(); i++)
    update_gramian(ktensor.get_factor(i), gramians[i]);
}

} // namespace cals::ops
