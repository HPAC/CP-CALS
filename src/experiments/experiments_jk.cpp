#include "experiments/experiments_utils.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

int main(int argc, char *argv[]) {
  cout << "=============================================================================" << endl;
  cout << "This executable performs the experiments in the paper." << endl;
  cout << "The corresponding source file is located in `src/experiments/experiments.cpp`" << endl;
  cout << "The output is writen in CSV files in the `data` folder of the project." << endl;
  cout << "This executable accepts only one (mandatory) argument, the number of threads." << endl;
  cout << "=============================================================================" << endl;
  cout << endl;

  if (argc != 2) {
    cerr << "Not enough arguments. Give number of threads." << endl;
    cout << "USAGE: " << argv[0] << " <num_threads>" << endl;
    abort();
  } else {
    std::string arg = argv[1];
    if ((arg == "-h") || (arg == "--help")) {
      cout << "USAGE: " << argv[0] << " <num_threads>" << endl;
      return 0;
    }
  }

  auto num_threads = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));

  bool experiment_5_1_and_5_2 = false;
  bool experiment_5_3 = true;

  if (experiment_5_1_and_5_2) {
    auto components = vector<dim_t>({3, 5, 7, 9});

    // Set the parameters of the execution
    cals::CalsParams params;
    params.mttkrp_method = cals::mttkrp::MTTKRP_METHOD::AUTO; // Let the Lookup tables decide the best method.
    params.force_max_iter = true;                             // Force all models to reach max_iterations.
    params.max_iterations = 100;                              // Set maximum number of iterations.
#if CUDA_ENABLED
    params.cuda = true;
#else
    params.cuda = false;
#endif

    vector<vector<dim_t>> modes_v;
    modes_v.push_back({50, 100, 100});
    modes_v.push_back({50, 200, 200});
    modes_v.push_back({50, 400, 400});
    // modes_v.push_back({50, 800, 800});

    for (auto &modes : modes_v) {
      params.buffer_size = 1500; // Else, select the largest buffer size to fit all models
      int nt = (num_threads == 1) ? 1 : 24;
      params.mttkrp_lut = cals::mttkrp::read_lookup_table(modes, nt, params.cuda);
      cals::Tensor T(modes);
      T.randomize();
      compare_jk_als_cals(T, components, num_threads, params);
    }
  }

  if (experiment_5_3) {

    // Set the parameters of the execution
    cals::CalsParams params;
    params.mttkrp_method = cals::mttkrp::MTTKRP_METHOD::AUTO; // Let the Lookup tables decide the best method.
    params.force_max_iter = false;                            // Force all models to reach max_iterations.
    params.max_iterations = 1000;                             // Set maximum number of iterations.
    params.tol = 1e-6;
#if CUDA_ENABLED
    params.cuda = true;
#else
    params.cuda = false;
#endif

    {
      string file_name = string(SOURCE_DIR) + "/data/stjohns.txt";
      cals::Tensor T(file_name);
      int nt = (num_threads == 1) ? 1 : 24;
      params.mttkrp_lut = cals::mttkrp::read_lookup_table(T.get_modes(), nt, params.cuda);
      params.buffer_size = 1335;
      auto components = vector<dim_t>({4, 5, 6});
      compare_jk_als_cals_real(T, components, num_threads, params);
    }

    {
      string file_name = string(SOURCE_DIR) + "/data/wine.txt";
      cals::Tensor T(file_name);
      int nt = (num_threads == 1) ? 1 : 24;
      params.mttkrp_lut = cals::mttkrp::read_lookup_table(T.get_modes(), nt, params.cuda);
      params.buffer_size = (num_threads == 1) ? 2640 : 500;
      params.buffer_size = (params.cuda) ? 1000 : params.buffer_size;
      auto components = vector<dim_t>({20, 20, 20});
      compare_jk_als_cals_real(T, components, num_threads, params);
    }
  }
}
