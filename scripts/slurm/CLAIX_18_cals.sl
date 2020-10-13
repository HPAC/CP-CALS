#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=CALS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=CALS-%j.out
#SBATCH --error=CALS-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=c18m
#SBATCH --account=rwth0575
#SBATCH --exclusive

module purge
module load DEVELOP
module load gcc/8
module load LIBRARIES
module load intelmkl/2019
module load cmake/3.13.2

source ~/.zshrc.local
cd ${CALS_DIR} || exit
export GOMP_CPU_AFFINITY="0 1 2 6 7 8 12 13 14 18 19 20 3 4 5 9 10 11 15 16 17 21 22 23"  # CLAIX18 xeon platinum 8160

if [ -d "build" ]; then
  rm -rvf build/*
  rm -rvf build
  mkdir build
  touch build/.gitkeep
else
  mkdir build
fi

cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_TESTS=Off -DWITH_MKL=On -DWITH_OPENBLAS=Off -DWITH_BLIS=Off -DWITH_CUBLAS=Off ..
make -j 48
./Experiment_MKL 1
./Experiment_MKL 12
./Experiment_MKL 24
#./Experiment_MKL 48

#./Experiment_OPENBLAS 1
#./Experiment_OPENBLAS 12

#./Experiment_BLIS 1
#./Experiment_BLIS 12
make clean
