# CP-CALS

Software for computing the Canonical Polyadic Decomposition (CPD), also known as PARAllel FACtors (PARAFAC), using the Concurrent Alternating Least Squares Algorithm (CALS).

[![Build Status](https://travis-ci.com/ChrisPsa/CP-CALS.svg?token=RsRp8LsqHqUm5bMEckfD&branch=master)](https://travis-ci.com/ChrisPsa/CP-CALS)

## Requirements

### Mandatory
* CMake 3.13.5 or higher
* GNU g++ 8.0 (or any C++ compiler with C++17 support)
* OpenMP
* Intel MKL (not required for MATLAB MEX generation)

### Optional
* CUDA 11
* MATLAB 2019b

## Compilation

Clone the CP-CALS repo using:

```bash
git clone https://github.com/HPAC/CP-CALS.git
```

Use the following commands to compile the MKL version.

```bash
cd CP-CALS/build
cmake  \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_MKL=ON ..

make -j 8 all
```

Use the following commands to compile the MKL version with CUDA enabled.

```bash
cd CP-CALS/build
cmake  \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_MKL=ON \
-DWITH_CUBLAS=ON ..

make -j 8 all
```

To compile the MEX files for Matlab use:

```bash
cd CP-CALS/build
cmake  \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_MATLAB=ON \
-DMATLAB_PATH=/path/to/Matlab ..

make -j 8 all
```

## Executing example code

After compiling, you can run `driver_MKL` for the CPU driver, or `cuda_driver_MKL` for the GPU driver (provided you compiled with CUDA enabled).
