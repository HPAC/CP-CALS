#ifndef CALS_KTENSOR_H
#define CALS_KTENSOR_H

#include <cfloat>
#include <cmath>
#include <vector>

#include "cals_blas.h"
#include "matrix.h"

using std::multiplies;
using std::vector;

static int universal_ktensor_id = 1;

namespace cals {

struct JackKniffing {
  bool enabled{false};
  dim_t fiber{0};
  dim_t mode{0};
};

class Ktensor {
  int id{-1};          /*< Unique ID of the Ktensor */
  dim_t components{0}; /*< Number of components of the Ktensor */
  dim_t iters{0};      /*< Number of iterations spent in ALS */

  double fit{0.0};          /*< The fit to the target tensor */
  double old_fit{0.0};      /*< The fit in the previous iteration */
  double approx_error{0.0}; /*< The approximation error of the Ktensor to the target Tensor */

  bool normalized{false}; /*< Boolean indicating whether the Ktensor is normalized or not */

  JackKniffing jk{false, 0, 0};

  vector<vector<vector<bool>>> active_set; /*< Active set for each factor matrix of last iteration (Used in NNLS) */

  vector<dim_t> modes{};

  vector<double> lambda{};  /*< Vector containing the lambda normalization factors per column of the factor matrices */
  vector<Matrix> factors{}; /*< Vector containing the factor matrices of the Ktensor */

  /** Recursive function that is used to compute the reconstruction of the Ktensor into a Tensor.
   */
  void rec_to_tensor(vector<dim_t> &modes, int &index, double *x_data, int level, dim_t *dim_ind);

public:
  Ktensor() = default;

  /** Allocates a Ktensor (no initialization).
   *
   * This constructor allocates a Ktensor of a specific \p components and sizes of \p modes. The
   * contents of the Ktensor created are not initialized. One can use the randomize member function
   * to fill in the Ktensor with random values.
   *
   * @param components The components of the Ktensor.
   * @param vector A vector containing the sizes of each mode of the Ktensor.
   */
  Ktensor(dim_t components, const vector<dim_t> &modes)
      : id{universal_ktensor_id++},
        components{components},
        active_set{vector<vector<vector<bool>>>(modes.size(), vector<vector<bool>>())},
        modes(modes),
        factors{vector<Matrix>(modes.size())} {
    assert(components > 0);
    for (dim_t n = 0; n < modes.size(); n++) {
      factors[n] = Matrix{modes[n], static_cast<dim_t>(components)};
      active_set[n] = vector<vector<bool>>(modes[n], vector<bool>(components, true));
    }

    // For each mode allocate a coefficients vector
    lambda.resize(get_components());
  }

  /** Allocates a Jack-Kniffed Ktensor (no initialization).
   *
   * This constructor allocates a Ktensor of a specific \p components and sizes of \p modes. The
   * contents of the Ktensor created are not initialized. One can use the randomize member function
   * to fill in the Ktensor with random values.
   *
   * @param components The components of the Ktensor.
   * @param vector A vector containing the sizes of each mode of the Ktensor.
   * @param fiber
   * @param mode
   */
  Ktensor(dim_t components, const vector<dim_t> &modes, dim_t jk_fiber, dim_t jk_mode = 0)
      : Ktensor(components, modes) {
    jk.enabled = true;
    jk.fiber = jk_fiber;
    jk.mode = jk_mode;
  };

  Ktensor &operator=(Ktensor &&rhs) = default;

  Ktensor(Ktensor &&rhs) = default;

  Ktensor(const Ktensor &rhs)
      : id{universal_ktensor_id++},
        components{rhs.components},
        jk{rhs.jk},
        active_set{vector<vector<vector<bool>>>(rhs.get_n_modes(), vector<vector<bool>>())},
        modes(rhs.modes),
        lambda{vector<double>(static_cast<size_t>(rhs.components))},
        factors{vector<Matrix>(rhs.get_n_modes())} {
    for (dim_t n = 0; n < rhs.get_n_modes(); n++) {
      factors[n] = rhs.get_factor(n);
      active_set[n] = vector<vector<bool>>(get_factor(n).get_rows(), vector<bool>(components, true));
    }
    lambda = rhs.get_lambda();
  }

  Ktensor &operator=(const Ktensor &rhs) {
    if (this == &rhs) // Properly handle self assignment
      return *this;

    id = universal_ktensor_id++;
    components = rhs.components;
    lambda = rhs.get_lambda();
    jk = rhs.jk;
    modes = rhs.modes;

    factors = vector<Matrix>(rhs.get_n_modes());
    for (dim_t n = 0; n < rhs.get_n_modes(); n++)
      factors[n] = rhs.get_factor(n);
    return *this;
  }

  ~Ktensor() = default;

  // Getters
  // TODO clean this. Components should either not exist or be updated properly when adjusting the dimensions of a
  // Ktensor.
  [[nodiscard]] inline dim_t get_components() const noexcept { return factors[0].get_cols(); }

  [[nodiscard]] inline dim_t get_iters() const noexcept { return iters; }

  [[nodiscard]] inline int get_id() const noexcept { return id; }

  [[nodiscard]] inline bool is_jk() const noexcept { return jk.enabled; }

  [[nodiscard]] inline dim_t get_jk_mode() const noexcept { return jk.mode; }

  [[nodiscard]] inline dim_t get_jk_fiber() const noexcept { return jk.fiber; }

  [[nodiscard]] inline double get_approximation_error() const noexcept { return approx_error; }

  [[nodiscard]] inline vector<dim_t> const &get_modes() const noexcept { return modes; }

  inline vector<Matrix> &get_factors() noexcept { return factors; }

  [[nodiscard]] inline vector<double> const &get_lambda() const noexcept { return lambda; }

  inline vector<double> &get_lambda() noexcept { return lambda; }

  // Setters
  inline void set_iters(dim_t new_iters) noexcept { iters = new_iters; }

  inline void set_approximation_error(double new_error) noexcept { approx_error = new_error; }

  inline void set_factor(int index, double *data) noexcept {
    auto &target = get_factor(static_cast<dim_t>(index));
    cblas_dcopy(target.get_n_elements(), data, 1, target.get_data(), 1);
  }

  inline void set_lambda(double const *data) noexcept {
    auto i = 0;
    for (auto &l : lambda)
      l = data[i++];
  }

  /** Calculate the fit to the target Tensor.
   *
   * @param X_norm L2-Norm of the target Tensor.
   *
   * @return The fit to the target Tensor.
   */
  inline double calculate_new_fit(double X_norm) noexcept {
    assert(X_norm != 0);
    old_fit = fit;
    fit = 1 - std::fabs(approx_error) / X_norm;
    return fit;
  }

  /** Calculate the difference in fit between the last iterations.
   *
   * @return The difference in fit.
   */
  [[nodiscard]] inline double get_fit_diff() const noexcept { return std::fabs(old_fit - fit); }

  /** Get the number of modes of the Ktensor.
   */
  inline dim_t get_n_modes() const noexcept { return static_cast<dim_t>(factors.size()); }

  /** Get a reference to the last factor Matrix.
   */
  inline Matrix const &get_last_factor() const noexcept { return factors.back(); }

  /** Get a reference to a specific factor Matrix.
   *
   * @param mode The mode for which to get the factor Matrix.
   */
  inline Matrix const &get_factor(dim_t mode) const noexcept { return factors.at(mode); }

  inline Matrix &get_factor(dim_t mode) noexcept { return factors.at(mode); }

  inline vector<vector<bool>> &get_active_set(const dim_t mode) noexcept { return active_set.at(mode); }

  [[nodiscard]] inline vector<Matrix> const &get_factors() const noexcept { return factors; }

  /** Print the contents of the Ktensor, together with some optional text.
   *
   * @param text (Optional) Text to display along with the contents of the Ktensor.
   */
  void print(const std::string &&text = "Ktensor") const;

  /** Attach every factor matrix to pointers (and copy their contents to the new locations).
   *
   * @param data_ptrs vector containing the pointers to which the factor matrices should point (must be the same length
   * as components)
   * @param multi_thread (Optional) Whether it should be performed using multiple threads or not (to avoid parallel copy
   * of overlaping memory regions).
   *
   * @return Reference to self.
   */
  Ktensor &attach(vector<double *> &data_ptrs, bool multi_thread = true);

  /** Reset the factor matrices to point to the data they own.
   */
  Ktensor &detach();

  /** Normalize the Ktensor.
   */
  Ktensor &normalize();

  /** Normalize a specific factor matrix.
   *
   * @param mode Specify the factor matrix to normalize.
   * @param iteration Depending on the iteration a different normalization function is used.
   *
   * @return Reference to self.
   */
  Ktensor &normalize(dim_t mode, dim_t iteration = 1);

  /** Remove normalization of the Ktensor.
   *
   * @return Reference to self.
   */
  Ktensor &denormalize();

  /** Randomize the Ktensor.
   *
   * @param r (Optional) Whether to use the global pre-seeded generator to create a reproducible set of values for
   * testing/experiments.
   *
   * @return Reference to self.
   */
  Ktensor &randomize();

  /** Fill the Ktensor with values.
   *
   * @param f function that returns a double every time it is invoked.
   *
   * @return Reference to self.
   */
  Ktensor &fill(function<double()> &&func);

  /** Convert the Ktensor to a Tensor.
   *
   * @return The reconstructed tensor.
   */
  Tensor to_tensor();

  Ktensor &copy(Ktensor &rhs);

  inline Ktensor &to_jk(dim_t mode, dim_t fiber) {
    jk.enabled = true;
    jk.mode = mode;
    jk.fiber = fiber;

    return *this;
  }

  inline Ktensor to_regular() {
    if (jk.enabled) {
      vector<dim_t> reg_modes{};
      reg_modes.reserve(modes.size());

      for (dim_t i = 0; i < modes.size(); i++)
        if (i != jk.mode)
          reg_modes.push_back(modes[i]);
        else
          reg_modes.push_back(modes[i] - 1);

      auto reg_ktensor = Ktensor(components, reg_modes);
      for (dim_t f = 0; f < modes.size(); f++)
        if (f != jk.mode)
          reg_ktensor.get_factor(f).copy(this->get_factor(f));
        else {
          auto &t_factor = reg_ktensor.get_factor(f);
          auto &s_factor = this->get_factor(f);
          for (dim_t ii = 0; ii < s_factor.get_rows(); ii++)
            for (dim_t jj = 0; jj < s_factor.get_cols(); jj++)
              if (ii < jk.fiber)
                t_factor(ii, jj) = s_factor(ii, jj);
              else if (ii > jk.fiber)
                t_factor(ii - 1, jj) = s_factor(ii, jj);
        }
      reg_ktensor.get_lambda() = this->get_lambda();

      return reg_ktensor;
    } else
      return *this;
  }

  inline void set_jk_fiber(double value) noexcept {
    if (jk.enabled) {
      auto &jk_factor = get_factor(jk.mode);
      if (!std::isnan(value))
        cblas_dscal(get_components(), value, jk_factor.get_data() + jk.fiber, jk_factor.get_col_stride());
      else
        for (dim_t j = 0; j < jk_factor.get_cols(); j++)
          jk_factor(jk.fiber, j) = NAN;
    }
  }

#if CUDA_ENABLED
  /** Attach every factor matrix to pointers (for GPU) (and copy their contents to the new locations).
   *
   * @param data_ptrs vector containing the pointers to which the factor matrices should point (must be the same length
   * as components)
   *
   * @return Reference to self.
   */
  Ktensor &cuattach(vector<double *> &cudata_ptrs, cudaStream_t &stream);

  /** Reset the factor matrices to point to the data they own.
   */
  Ktensor &cudetach();
#endif
};

} // namespace cals
#endif // CALS_KTENSOR_H
