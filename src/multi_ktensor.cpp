#include "multi_ktensor.h"

#include <iostream>

#include "utils/utils.h"

namespace cals {
MultiKtensor::MultiKtensor(vector<dim_t> &modes, dim_t buffer_size)
    : Ktensor(buffer_size, modes), occupancy(0), occupancy_vec(buffer_size), modes(modes) {
  for (auto &f : get_factors())
    f.resize(f.get_rows(), 0);
}

int MultiKtensor::check_availability(Ktensor &ktensor) {
  dim_t comp_counter = 0;
  auto pos_index = -1;
  auto prev_occ = -1;
  auto n_cells = static_cast<int>(occupancy_vec.size());

  for (auto i = 0; i < n_cells; i++) {
    if (comp_counter == ktensor.get_components())
      break;

    if (occupancy_vec[i] == 0 && prev_occ != 0) {
      pos_index = i;
      comp_counter++;
    } else if (occupancy_vec[i] == 0 && prev_occ == 0)
      comp_counter++;
    else
      comp_counter = 0;

    prev_occ = occupancy_vec[i];
  }

  if (pos_index == -1 || comp_counter != ktensor.get_components())
    throw BufferFull();

  return pos_index;
}

MultiKtensor &MultiKtensor::add(Ktensor &ktensor) {
  if (cuda) {
#if CUDA_ENABLED
    cuda::initialize_stream(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }
  int pos_index;
  try {
    pos_index = check_availability(ktensor);
  } catch (BufferFull &e) {
    throw e;
  }

  // Determine address for each factor matrix
  vector<double *> pos_ptrs(ktensor.get_n_modes());

  auto index = 0;
  for (auto &f : get_factors())
    pos_ptrs[index++] = f.reset_data().get_data() + pos_index * f.get_col_stride();

  // Attach Ktensor
  ktensor.attach(pos_ptrs);

  if (cuda) {
#if CUDA_ENABLED
    vector<double *> cupos_ptrs(ktensor.get_n_modes());
    auto cuindex = 0;
    for (auto &f : get_factors())
      cupos_ptrs[cuindex++] = f.reset_cudata().get_cudata() + pos_index * f.get_col_stride();
    ktensor.cuattach(cupos_ptrs, stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  auto unique_id = unique_kt_id++;

  // Adjust occupancy vector
  for (dim_t i = 0; i < ktensor.get_components(); i++)
    occupancy_vec[pos_index + i] = unique_id;

  // adjust occupancy_pct
  occupancy += ktensor.get_components();

  // Create gramians
  vector<Matrix> gramians(ktensor.get_n_modes());
  for (auto &g : gramians)
    g = Matrix(ktensor.get_components(), ktensor.get_components());
  index = 0;
  for (const auto &f : ktensor.get_factors())
    cals::ops::update_gramian(f, gramians[index++]);
  ktensor.set_iters(1);

  // If the added model is a JK model, ensure that all X_norms are calculated
  if (ktensor.is_jk())
    flag_jk = true;

  RegistryEntry entry{ktensor, std::move(gramians), pos_index, unique_id};

  if (line_search) {
    entry.ls_params.prev_ktensor = Ktensor(entry.ktensor.get_components(), modes);
    entry.ls_params.backup_ktensor = Ktensor(entry.ktensor.get_components(), modes);
    entry.ls_params.cuda = cuda;
    entry.ls_params.interval = ls_params.interval;
    entry.ls_params.step = ls_params.step;
    entry.ls_params.method = ls_params.method;
    entry.ls_params.T = ls_params.T;
  }

  // Update registry
  registry.insert(std::pair(unique_id, std::move(entry)));

  adjust_edges();

  if (cuda) {
#if CUDA_ENABLED
    cudaStreamSynchronize(stream);
    cuda::destroy_stream(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  return *this;
}

MultiKtensor &MultiKtensor::remove(dim_t ktensor_id) {
  auto &entry = registry.at(ktensor_id);
  auto &ktensor = entry.ktensor;

  //  if (ktensor.is_jk())
  //    ktensor.set_jk_fiber(NAN);

  // Remove ktensor from the factor matrices and copy over their last value
  ktensor.detach();

  if (cuda) {
#if CUDA_ENABLED
    ktensor.cudetach();
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }

  // Update the occupancy vector and counter
  for (auto &el : occupancy_vec)
    if (el == ktensor_id)
      el = 0;
  occupancy -= ktensor.get_components();

  // Update registry
  registry.erase(ktensor_id);

  adjust_edges();

  return *this;
}

MultiKtensor &MultiKtensor::adjust_edges() {
  start = 0;
  int end = static_cast<int>(occupancy_vec.size());

  //    for (auto i = 0; i < end; i++)
  //      if (occupancy_vec[i] == 0) start++;
  //      else break;
  for (auto i = end - 1; i > start; i--) // end is equal to the size (so -1 to get the last element)
    if (occupancy_vec[i] == 0)
      end--;
    else
      break;

  int active_cols = end - start;
  this->end = end;
  for (auto &f : get_factors()) {
    f.set_data(f.reset_data().get_data() + start * f.get_col_stride()); // Adjust data to point to start
    f.resize(f.get_rows(), static_cast<dim_t>(active_cols));
  }

  return *this;
}

MultiKtensor &MultiKtensor::compress() {
  //    if (1.0 * occupancy / (end - start) < 0.99)
  //    {
  if (cuda) {
#if CUDA_ENABLED
    cuda::initialize_stream(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }
  // std::cout << "Occupancy percent: " << 1.0 * occupancy / (end - start) << " Size: " << end << "Compression...";

  int col_offset = 0;
  ptrdiff_t added = -1;
  // move_requests is an ordered list with the ids of ktensors and the steps to the left they need to move (offset).
  std::vector<std::tuple<int, int>> move_requests;
  move_requests.reserve(static_cast<unsigned long>(occupancy));

  for (auto cell : occupancy_vec)
    if (cell == added)
      continue;
    else if (cell == 0)
      col_offset++;
    else if (col_offset != 0) {
      move_requests.emplace_back(std::make_tuple(cell, col_offset));
      added = cell;
    }

  vector<double *> new_data(get_n_modes());
  for (auto &[key, offset] : move_requests) {
    dim_t index = 0;
    auto &registry_entry = registry.at(key);
    auto &ktensor = registry_entry.ktensor;

    for (auto &factor : ktensor.get_factors())
      new_data.at(index++) = factor.get_data() - offset * factor.get_col_stride();

    if (static_cast<int>(ktensor.get_components()) < offset)
      registry_entry.ktensor.attach(new_data, true);
    else
      registry_entry.ktensor.attach(new_data, false);

    for (auto i = static_cast<dim_t>(registry_entry.col); i < registry_entry.col + ktensor.get_components(); i++)
      std::swap(occupancy_vec[i - offset], occupancy_vec[i]);

    registry.at(key).col -= offset;

    if (cuda) {
#if CUDA_ENABLED
      for (auto &factor : ktensor.get_factors()) {
        auto *new_cudata = factor.get_cudata() - offset * factor.get_col_stride();
        cudaMemcpyAsync(new_cudata, factor.get_cudata(), factor.get_n_elements() * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
        factor.set_cudata(new_cudata);
      }
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }
  }

  adjust_edges();

  if (cuda) {
#if CUDA_ENABLED
    cudaStreamSynchronize(stream);
    cuda::destroy_stream(stream);
#else
    std::cerr << "Not compiled with CUDA support" << std::endl;
    exit(EXIT_FAILURE);
#endif
  }
  //    }
  return *this;
}
} // namespace cals
