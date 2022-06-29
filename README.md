# CP-CALS

Software for computing the Canonical Polyadic Decomposition (CPD), also known as PARAllel FACtors (PARAFAC), using the Concurrent Alternating Least Squares Algorithm (CALS).

[![Build Status](https://travis-ci.com/ChrisPsa/CP-CALS.svg?token=RsRp8LsqHqUm5bMEckfD&branch=master)](https://travis-ci.com/ChrisPsa/CP-CALS)

## Requirements

### Mandatory
* CMake 3.17.5 or higher.
  A bash script is provided, to help install CMake 3.17.5 (for Linux) in the `extern` folder and updates the `PATH` environment variable to point to it. Try using it by navigating to the directory where you cloned CALS and running: `source scripts/environment_setup.sh`. Then, using the same terminal session, running `cmake --version` should return version 3.17.5.
* OpenMP
* BLAS/LAPACK (not required for MATLAB MEX generation). Either of the following libraries has been tested to work:
  * Intel MKL
  * OpenBLAS

### Optional
* CUDA 11
* MATLAB 2019b

CALS has been tested to compile with g++-8, g++-10 and clang-10.

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

or the following commands to compile the OpenBLAS version.

```bash
cd CP-CALS/build
cmake  \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_OPENBLAS=ON ..

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

`src/examples/driver` contains a demonstration of how to use CALS (The -h flag is supported for a look at possible input arguments).

### MATLAB examples

After compiling the MATLAB MEX, one can execute file `matlab/matlab_src/TTB_vs_CALS.m` in MATLAB. This executable performs a comparisson of TensorToolbox and CALS. The user first needs to point MATLAB to the CALS MEX and the Tensor Toolbox source code by editing the first two lines of the file.

## Related Publications

* [Algorithm XXX: Concurrent Alternating Least Squares for multiple simultaneous Canonical Polyadic Decompositions](https://dl.acm.org/doi/10.1145/3519383)

```bibtex
@article{10.1145/3519383,
	title        = {Algorithm XXX: Concurrent Alternating Least Squares for Multiple Simultaneous Canonical Polyadic Decompositions},
	author       = {Psarras, Christos and Karlsson, Lars and Bro, Rasmus and Bientinesi, Paolo},
	year         = 2022,
	month        = {feb},
	journal      = {ACM Trans. Math. Softw.},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	doi          = {10.1145/3519383},
	issn         = {0098-3500},
	url          = {https://doi.org/10.1145/3519383},
	note         = {Just Accepted},
	keywords     = {PARAFAC, high-performance, Tensor, CP, decomposition}
}
```

* [Accelerating jackknife resampling for the Canonical Polyadic Decomposition](https://www.frontiersin.org/articles/10.3389/fams.2022.830270/full)

```bibtex
@article{10.3389/fams.2022.830270,
	title        = {Accelerating Jackknife Resampling for the Canonical Polyadic Decomposition},
	author       = {Psarras, Christos and Karlsson, Lars and Bro, Rasmus and Bientinesi, Paolo},
	year         = 2022,
	journal      = {Frontiers in Applied Mathematics and Statistics},
	volume       = 8,
	doi          = {10.3389/fams.2022.830270},
	issn         = {2297-4687},
	url          = {https://www.frontiersin.org/article/10.3389/fams.2022.830270},
}
```


