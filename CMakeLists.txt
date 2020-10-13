#@HEADER
# ******************************************************************************
#
#  CP-CALS: Software for computing the Canonical Polyadic Decomposition using
#  the Concurrent Alternating Least Squares Algorithm.
#
#  Copyright (c) 2020, Christos Psarras
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************
#@HEADER

CMAKE_MINIMUM_REQUIRED(VERSION 3.12.4 FATAL_ERROR)
PROJECT(CALS)
include(CheckCXXCompilerFlag)

OPTION(WITH_MKL "Create MKL Library version" Off)
OPTION(WITH_CUBLAS "Create cuBLAS Library version" Off)
OPTION(WITH_BLIS "Create BLIS Library version" Off)
OPTION(WITH_OPENBLAS "Create OpenBLAS Library version" Off)
OPTION(WITH_MATLAB "Enable MATLAB support" Off)
OPTION(WITH_TESTS "Enable Tests" Off)
OPTION(WITH_TRAVIS "Travis mode" Off)  # There is a bug with cmake on travis, disable specifying BLAS version for now

#########################################################################################
# Find MKL
#########################################################################################
if (${WITH_MKL})
    if (NOT ${WITH_TRAVIS})
      set(BLA_VENDOR Intel10_64lp)
    endif ()
    FIND_PACKAGE(BLAS REQUIRED)
    FIND_PACKAGE(LAPACK REQUIRED)

    FIND_PATH(CALS_INCLUDE_DIRS mkl.h $ENV{MKLROOT}/include)
    SET(CALS_LINK_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} m)

    SET(BLA_VEN MKL)
endif ()

#########################################################################################
# Find OpenMP
#########################################################################################
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

#########################################################################################
# Find CUDA
#########################################################################################
if (${WITH_CUBLAS})
    ENABLE_LANGUAGE(CUDA)
    SET(CALS_INCLUDE_DIRS ${CALS_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    SET(CALS_LINK_LIBRARIES ${CALS_LINK_LIBRARIES} -lcublas -lcudart)
    IF (EXISTS "/usr/local/cuda/lib64")
      SET (CALS_LINK_DIRECTORIES "/usr/local/cuda/lib64")
    ELSE ()
      SET (CALS_LINK_DIRECTORIES "$ENV{CUDA_ROOT}/lib64" "$ENV{CUDA_ROOT}/lib64/lib64")
    ENDIF ()
    MESSAGE (STATUS CALS_LINK_DIRECTORIES: ${CALS_LINK_DIRECTORIES})
endif ()

#########################################################################################
# Find BLIS
#########################################################################################
if (${WITH_BLIS})
    if(NOT ${WITH_MKL})
        MESSAGE(FATAL_ERROR "!! BLIS does not provide LAPACK functions. Please use -DWITH_MKL=On to provide LAPACK from MKL.")
    endif ()
    MESSAGE("BLIS Install dir: " "$ENV{BLIS_INSTALL_DIR}")
    SET(BLIS_LIBRARIES $ENV{BLIS_INSTALL_DIR}/lib/libblis.so)

    SET(CALS_INCLUDE_DIRS $ENV{BLIS_INSTALL_DIR}/include/blis ${CALS_INCLUDE_DIRS})
    SET(CALS_LINK_LIBRARIES ${BLIS_LIBRARIES} ${LAPACK_LIBRARIES} m)

    SET(BLA_VEN BLIS)
endif ()

#########################################################################################
# Find OpenBLAS
#########################################################################################
if (${WITH_OPENBLAS})
    if(NOT ${WITH_MKL})
        MESSAGE(FATAL_ERROR "!! Please use -DWITH_MKL=On to provide LAPACK from MKL (for OpenBLAS due to Bug).")
    endif ()
    MESSAGE("OPENBLAS Install dir: " "$ENV{OPENBLAS_INSTALL_DIR}")
    SET(OPENBLAS_LIBRARIES $ENV{OPENBLAS_INSTALL_DIR}/lib/libopenblas.so)

    SET(CALS_INCLUDE_DIRS $ENV{OPENBLAS_INSTALL_DIR}/include ${CALS_INCLUDE_DIRS})
    SET(CALS_LINK_LIBRARIES ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES} m)

    SET(BLA_VEN OPENBLAS)
endif ()

################################
# Find Matlab
################################
IF (WITH_MATLAB)
    OPTION(MATLAB_PATH "Path to MATLAB installation")
    IF (MATLAB_PATH)
        SET(Matlab_ROOT_DIR "${MATLAB_PATH}")
    ELSE ()
        SET(Matlab_ROOT_DIR "$ENV{HOME}/opt/Matlab2018b")
    ENDIF ()

    MESSAGE("MATLAB_ROOT_DIR: " ${MATLAB_PATH})
    FIND_PACKAGE(Matlab COMPONENTS MX_LIBRARY REQUIRED)

    SET(CALS_LINK_LIBRARIES ${CALS_LINK_LIBRARIES} ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})

    SET(HAVE_MATLAB ON)
    SET(BLA_VEN Matlab)
ENDIF ()

#########################################################################################
# General Configuration
#########################################################################################
SET(CMAKE_VERBOSE_MAKEFILE 1)
SET(CMAKE_CXX_STANDARD 17)

SET(CALS_FILES
        include/timer.h
        include/definitions.h
        src/tensor.cpp include/tensor.h
        src/matrix.cpp include/matrix.h
        src/ktensor.cpp include/ktensor.h
        src/als.cpp include/als.h
        src/cals.cpp include/cals.h
        src/cuda_utils.cpp include/cuda_utils.h
        src/utils.cpp include/utils.h
        src/multi_ktensor.cpp include/multi_ktensor.h
        include/cals_blas.h src/cals_blas.cpp)

SET(CALS_EXPERIMENT_FILES
        src/experiments/cpu_experiments.cpp src/experiments/experiments_utils.cpp include/experiments/experiments_utils.h
        src/experiments/mttkrp_benchmarker.cpp include/experiments/mttkrp_benchmarker.h)

SET(CALS_MTTKRP_BENCHMARK_FILES
        src/experiments/mttkrp_benchmarks.cpp include/experiments/mttkrp_benchmarker.h src/experiments/mttkrp_benchmarker.cpp)

SET(CALS_EVALUATOR_FILES
        src/experiments/peak_evaluator.cpp)

SET(CALS_CUDA_EXPERIMENT_FILES
        src/experiments/cuda_experiment.cpp include/experiments/experiments_utils.h src/experiments/experiments_utils.cpp)

SET(CALS_DRIVER_FILES
        src/examples/driver.cpp)

SET(CALS_CUDA_DRIVER_FILES
        src/examples/cuda_driver.cpp)

# CALS Library
SET(CALS_NAME CALS_${BLA_VEN})
ADD_LIBRARY(${CALS_NAME} SHARED ${CALS_FILES})

SET_TARGET_PROPERTIES(${CALS_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

foreach (cld ${CALS_LINK_DIRECTORIES})
  TARGET_LINK_DIRECTORIES(${CALS_NAME} PUBLIC ${cld})
endforeach ()
TARGET_LINK_LIBRARIES(${CALS_NAME} ${CALS_LINK_LIBRARIES})

TARGET_INCLUDE_DIRECTORIES(${CALS_NAME} PUBLIC include ${CALS_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

TARGET_COMPILE_DEFINITIONS(${CALS_NAME} PUBLIC ${CALS_NAME}=1)
TARGET_COMPILE_OPTIONS(${CALS_NAME} PUBLIC -fPIC)

if (${BLA_VEN} STREQUAL "Matlab")
    TARGET_COMPILE_DEFINITIONS(${CALS_NAME} PUBLIC WITH_TIME=0)
    TARGET_INCLUDE_DIRECTORIES(${CALS_NAME} PRIVATE ${Matlab_INCLUDE_DIRS})
else ()
    TARGET_COMPILE_DEFINITIONS(${CALS_NAME} PUBLIC WITH_TIME=1)
endif ()

if (${WITH_CUBLAS})
    TARGET_COMPILE_DEFINITIONS(${CALS_NAME} PUBLIC CUDA_ENABLED=1)
endif ()

# Other Libs and Executables
if (NOT (${BLA_VEN} STREQUAL "Matlab"))
    ADD_EXECUTABLE(Experiment_${BLA_VEN} ${CALS_EXPERIMENT_FILES})
    TARGET_LINK_LIBRARIES(Experiment_${BLA_VEN} ${CALS_NAME})

    ADD_EXECUTABLE(mttkrp_benchmark_${BLA_VEN} ${CALS_MTTKRP_BENCHMARK_FILES})
    TARGET_LINK_LIBRARIES(mttkrp_benchmark_${BLA_VEN} ${CALS_NAME})

    ADD_EXECUTABLE(Evaluator_${BLA_VEN} ${CALS_EVALUATOR_FILES})
    TARGET_LINK_LIBRARIES(Evaluator_${BLA_VEN} ${CALS_NAME})

    ADD_EXECUTABLE(CUDA_Experiment_${BLA_VEN} ${CALS_CUDA_EXPERIMENT_FILES})
    TARGET_LINK_LIBRARIES(CUDA_Experiment_${BLA_VEN} ${CALS_NAME})

    ADD_EXECUTABLE(driver_${BLA_VEN} ${CALS_DRIVER_FILES})
    TARGET_LINK_LIBRARIES(driver_${BLA_VEN} ${CALS_NAME})

    ADD_EXECUTABLE(cuda_driver_${BLA_VEN} ${CALS_CUDA_DRIVER_FILES})
    TARGET_LINK_LIBRARIES(cuda_driver_${BLA_VEN} ${CALS_NAME})
endif ()

################################
# GTest
################################
if (${WITH_TESTS})

    FIND_PACKAGE(GTest REQUIRED)
    enable_testing()

    INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
    SET(GTEST_LINK_FLAGS ${GTEST_BOTH_LIBRARIES} -lpthread)

    ADD_SUBDIRECTORY(tests)
endif (${WITH_TESTS})

################################
# Matlab
################################

IF (HAVE_MATLAB)

    ADD_LIBRARY(
            cals_matlab_interface_library SHARED
            ${CALS_SOURCE_DIR}/matlab/matlab.h
            ${CALS_SOURCE_DIR}/matlab/matlab.cpp
            ${CALS_SOURCE_DIR}/matlab/matlab_parsing.h
            ${CALS_SOURCE_DIR}/matlab/matlab_parsing.cpp)
    TARGET_INCLUDE_DIRECTORIES(cals_matlab_interface_library PRIVATE ${Matlab_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(cals_matlab_interface_library CALS_Matlab ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})
    SET(CALS_MATLAB_LINK_LIBS cals_matlab_interface_library CALS_Matlab)

    # Newer versions of CMake have an API version argument to MATLAB_ADD_MEX
    # which needs to be set to the right value since it defaults to an incorrect
    # value
    SET(CALS_MATLAB_API_VERSION "")
    IF (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14.0")
        SET(CALS_MATLAB_API_VERSION "R2018a")
    ENDIF ()

    MATLAB_ADD_MEX(
            NAME cp_cals_driver
            SRC ${CALS_SOURCE_DIR}/matlab/matlab_cp_cals.cpp
            ${CALS_MATLAB_API_VERSION})

    TARGET_LINK_LIBRARIES(cp_cals_driver ${CALS_MATLAB_LINK_LIBS})
ENDIF ()


# Optimization Flags
if (CMAKE_BUILD_TYPE MATCHES Release)
    SET(COMP_FLAGS -Ofast -march=native -Wall -Wno-unused-function)  # -Wno-unused-function for BLIS lib warnings

    foreach (flag ${COMP_FLAGS})
        CHECK_CXX_COMPILER_FLAG(${flag} COMPILER_OPT_SUPPORTED${flag})
        if (COMPILER_OPT_SUPPORTED${flag})
            MESSAGE(STATUS "Adding ${flag} flag")
            TARGET_COMPILE_OPTIONS(${CALS_NAME} PUBLIC ${flag})
            if (${WITH_MATLAB})
                TARGET_COMPILE_OPTIONS(cals_matlab_interface_library PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(cp_cals_driver PUBLIC ${flag})
            else()
                TARGET_COMPILE_OPTIONS(Experiment_${BLA_VEN} PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(mttkrp_benchmark_${BLA_VEN} PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(Evaluator_${BLA_VEN} PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(CUDA_Experiment_${BLA_VEN} PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(driver_${BLA_VEN} PUBLIC ${flag})
                TARGET_COMPILE_OPTIONS(cuda_driver_${BLA_VEN} PUBLIC ${flag})
            endif ()
        else ()
            MESSAGE(STATUS "The compiler does not support ${flag} flag")
        endif ()
    endforeach ()
endif ()
