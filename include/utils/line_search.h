#ifndef CP_CALS_LINE_SEARCH_H
#define CP_CALS_LINE_SEARCH_H

#include "ktensor.h"

namespace cals::ls {

enum LS_METHOD { NO_ERROR_CHECKING = 0, ERROR_CHECKING_SERIAL, ERROR_CHECKING_PARALLEL, LENGTH };

static const std::string ls_method_names[LS_METHOD::LENGTH] = {"no-error-checking", "error-checking-serial",
                                                               "error-checking-parallel"};

struct LineSearchParams {
  int iter{};
  int interval{};
  bool updated_last_iter{};

  LS_METHOD method{NO_ERROR_CHECKING};
  Ktensor prev_ktensor{};
  double step{0.0}; /*< Factor with which to extrapolate. */
  bool cuda{false}; /*< Indicate whether cuda is used (to stream results back to the GPU when done). */

  bool extrapolated{false};
  bool reversed{false};

  // NO_ERROR_CHECKING
  Ktensor backup_ktensor{};

  // ERROR_CHECKING_SERIAL or ERROR_CHECKING_PARALLEL
  Tensor const *T{nullptr};
};

/** Perform line search extrapolation.
 *
 * The Ktensor is extrapolated in the direction dictated by the position vector of the current and the previous
 * iterations.
 *
 * @param[in, out] ktensor the Ktensor on which to perform line search.
 * @param[in] prev_ktensor Ktensor object holding the Ktensor of the previous iteration.
 * @param[out] gramians Vector containing the gramian Matrix objects.
 * @param[in, out] params LineSearchParams object, containing parameters related to the line search extrapolation.
 */
void line_search(cals::Ktensor &ktensor, std::vector<cals::Matrix> &gramians, cals::ls::LineSearchParams &params);

} // namespace cals::ls

#endif // CP_CALS_LINE_SEARCH_H
