#!/usr/bin/env bash

module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
module load cmake

export NVCC_WRAPPER_DEFAULT_COMPILER=CC

dir=gcc_avx2_a100
rm -rf $dir

cmake -S cmake -B $dir -C cmake/presets/kokkos-cuda.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_FLAGS="-march=native -fno-math-errno" \
  -DMPI_CXX_COMPILER=CC \
  -DLAMMPS_LONGLONG_TO_LONG=yes \
  -DBUILD_MPI=yes \
  -DBUILD_OMP=no \
  -DKokkos_ARCH_AMPERE80=yes \
  -DPKG_EXTRA-FIX=yes \
  -DPKG_MANYBODY=yes \
  -DPKG_MEAM=yes \
  -DPKG_ML-SNAP=yes \
  -DPKG_ML-PACE=yes \
  -DPKG_REPLICA=yes

cmake --build $dir -j4
