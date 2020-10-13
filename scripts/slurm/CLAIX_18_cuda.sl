#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=CALS-CUDA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --output=cuda-%j.out
#SBATCH --error=cuda-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:2
#SBATCH --account=rwth0575
#SBATCH --exclusive

module purge
module load DEVELOP
module load gcc/8
module load LIBRARIES
module load intelmkl/2019
module load cmake/3.13.2
module load cuda/102

source ~/.zshrc.local
cd ${CALS_DIR} || exit

export GOMP_CPU_AFFINITY="0 1 2 6 7 8 12 13 14 18 19 20 3 4 5 9 10 11 15 16 17 21 22 23"  # CLAIX18 xeon platinum 8160

if [ -d "build_cuda" ]; then
  rm -rvf build_cuda/*
  rm -rv build_cuda
  mkdir build_cuda
else
  mkdir build_cuda
fi

cd build_cuda || exit
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_TESTS=Off -DWITH_MKL=On -DWITH_OPENBLAS=Off -DWITH_BLIS=Off -DWITH_CUBLAS=On ..
make -j 24 CUDA_Experiment_MKL

./CUDA_Experiment_MKL 24

# make clean
