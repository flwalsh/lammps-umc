#!/usr/bin/env bash

module reset
module load rocm/7.2.1
module load rocmcc/7.2.1-magic

dir=mi300a
rm -rf $dir

cmake -S cmake -B $dir \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=hipcc \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_CXX_FLAGS="-march=native -mprefer-vector-width=512 -fno-math-errno -Xarch_host -fveclib=libmvec" \
  -DBUILD_MPI=yes \
  -DBUILD_OMP=no \
  -DKokkos_ARCH_NATIVE=yes \
  -DKokkos_ARCH_AMD_GFX942_APU=yes \
  -DKokkos_ENABLE_HIP=yes \
  -DPKG_KOKKOS=yes \
  -DPKG_EXTRA-FIX=yes \
  -DPKG_MANYBODY=yes \
  -DPKG_MEAM=yes \
  -DPKG_ML-SNAP=yes \
  -DPKG_ML-PACE=yes \
  -DPKG_REPLICA=yes

cmake --build $dir -j4
