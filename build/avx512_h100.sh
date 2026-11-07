#!/usr/bin/env bash

module reset
module load clang/19.1.3-magic
module load cuda/13.1.1
module load cmake/3.30.5

export NVCC_WRAPPER_DEFAULT_COMPILER=clang++

dir=avx512_h100
rm -rf $dir

# Matrix defines int64_t as long so map LAMMPS bigint MPI to MPI_LONG
cmake -S cmake -B $dir -C cmake/presets/kokkos-cuda.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_FLAGS="-march=native -mprefer-vector-width=512 -fno-math-errno -fveclib=libmvec -Wno-constant-logical-operand" \
  -DMPI_CXX_COMPILER=mpicxx \
  -DLAMMPS_LONGLONG_TO_LONG=yes \
  -DBUILD_MPI=yes \
  -DBUILD_OMP=no \
  -DKokkos_ARCH_HOPPER90=yes \
  -DPKG_EXTRA-FIX=yes \
  -DPKG_MANYBODY=yes \
  -DPKG_MEAM=yes \
  -DPKG_ML-SNAP=yes \
  -DPKG_ML-PACE=yes \
  -DPKG_REPLICA=yes

cmake --build $dir -j4
