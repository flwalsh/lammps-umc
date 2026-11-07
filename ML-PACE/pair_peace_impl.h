/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Flynn Walsh
   Shared by pair_peace and pair_peace_kokkos
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_PEACE_IMPL_H
#define LMP_PAIR_PEACE_IMPL_H

#include "pair_peace.h"
#include "platform.h"

#include <cmath>

#ifdef KOKKOS_INLINE_FUNCTION
#define PEACE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define PEACE_INLINE inline
#endif

namespace LAMMPS_NS {
namespace PEACE {

static constexpr int MAX_L = 5;
static constexpr int MAX_MS = 2 * MAX_L + 1;
static constexpr int MAX_ORDER = 4;
static constexpr int MAX_NRAD = 32; // 1 CUDA warp

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
static constexpr int SIMD = 1;
static constexpr int TEAM_SIZE = 256;
static constexpr int MIX_TILE = 16;
static constexpr int NEIGH_THREADS = 16;
static constexpr int N_PER_THREAD = 2;
#ifdef KOKKOS_ENABLE_HIP
static constexpr int NL_PER_THREAD = 1;
#else
static constexpr int NL_PER_THREAD = 2;
#endif
#ifdef KOKKOS_ENABLE_CUDA
static constexpr int DB_STRIDE = TEAM_SIZE;
#else
static constexpr int DB_STRIDE = 0;
#endif
#else
#if defined(__AVX512F__)
static constexpr int SIMD = 8;
static constexpr int REGISTERS = 32;
#elif defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
static constexpr int SIMD = __ARM_FEATURE_SVE_BITS / 64;
static constexpr int REGISTERS = 32;
#elif defined(__AVX__)
static constexpr int SIMD = 4;
static constexpr int REGISTERS = 16;
#elif defined(__aarch64__)
static constexpr int SIMD = 2;
static constexpr int REGISTERS = 32;
#else
static constexpr int SIMD = 2;
static constexpr int REGISTERS = 16;
#endif
static constexpr int NEIGH_SIMD = 2 * SIMD;
static constexpr int MIX_TILE = REGISTERS / 2;
#endif

static PEACE_INLINE constexpr int lm_size(const int lmax)
{
  return (lmax + 1) * (lmax + 1);
}

static inline void stamp(int timing, double *times, int kernel, double &last)
{
  if (!timing) return;
#ifdef KOKKOS_INLINE_FUNCTION
  Kokkos::fence();
#endif
  const double now = platform::walltime();
  times[kernel] += now - last;
  last = now;
}

static PEACE_INLINE double spline_value(const double *coeff, const int stride, const double w)
{
  return ((coeff[3 * stride] * w + coeff[2 * stride]) * w + coeff[stride]) * w + coeff[0];
}

static PEACE_INLINE double
spline_deriv(const double *coeff, const int stride, const double w, const double dr_inv)
{
  return ((3.0 * coeff[3 * stride] * w + 2.0 * coeff[2 * stride]) * w + coeff[stride]) * dr_inv;
}

static PEACE_INLINE int spline_bin(const double r, const double dr_inv, const int nbin, double &w)
{
  const double xb = r * dr_inv;
  const int bin = MIN((int) xb, nbin - 1);
  w = xb - (double) bin;
  return bin;
}

// recursion originally from Numerical Recipes 3rd Ed.
template <int lmax, bool deriv, bool ldep = true>
static PEACE_INLINE void
compute_ylm(const double x, const double y, const double z, const int stride, double *out, const double *R = nullptr, const double *P = nullptr, const double *Q = nullptr, const double rinv = 0.0)
{
  double pmm = 1.0, c = 1.0, s = 0.0;
  double rad = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
  for (int m = 0; m <= lmax; m++) {
    const double c1 = c, s1 = s;
    if (m > 0) {
      c = x * c1 - y * s1;
      s = y * c1 + x * s1;
      pmm *= m == 1 ? sqrt(3.0) : sqrt(1.0 + 0.5 / m);
    }
    double p1 = 0.0, p2 = 0.0, dp1 = 0.0, dp2 = 0.0;
    double Ppos = 0.0;
    double Pneg = 0.0;
    double Qpos = 0.0;
    double Qneg = 0.0;
    double dQpos = 0.0;
    double dQneg = 0.0;
#pragma unroll
    for (int l = m; l <= lmax; l++) {
      double p, dp = 0.0;
      if (l == m) {
        p = pmm;
      } else if (l == m + 1) {
        p = sqrt(2.0 * m + 3.0) * z * pmm;
        if (deriv)
          dp = sqrt(2.0 * m + 3.0) * pmm;
      } else {
        const double l1sq = (l - 1) * (l - 1);
        const double a = sqrt((4.0 * l * l - 1.0) / ((double) (l * l) - m * m));
        const double b = -sqrt((l1sq - m * m) / (4.0 * l1sq - 1.0));
        p = a * (z * p1 + b * p2);
        if (deriv)
          dp = a * (p1 + z * dp1 + b * dp2);
      }
      const int pos = (l * (l + 1) + m) * stride;
      const int neg = (l * (l + 1) - m) * stride;
      // force path 
      if (deriv) {
        Ppos += P[pos] * p;
        Qpos += Q[pos] * p;
        dQpos += Q[pos] * dp;
        if (m > 0) {
          Pneg += P[neg] * p;
          Qneg += Q[neg] * p;
          dQneg += Q[neg] * dp;
        }
      } 
      // Ai GPU path: R passed in
      else if (R != nullptr) {
        out[pos] += c * p * R[ldep ? l : 0];
        if (m > 0)
          out[neg] += s * p * R[ldep ? l : 0];
      } 
      // Ai CPU path
      else {
        out[pos] = c * p;
        if (m > 0)
          out[neg] = s * p;
      }
      p2 = p1;
      p1 = p;
      dp2 = dp1;
      dp1 = dp;
    }
    if (deriv) {
      rad += c * Ppos + s * Pneg;
      gz += c * dQpos + s * dQneg;
      if (m > 0) {
        gx += m * (c1 * Qpos + s1 * Qneg);
        gy += m * (c1 * Qneg - s1 * Qpos);
      }
    }
  }
  if (deriv) {
    const double rg = x * gx + y * gy + z * gz;
    out[0] = rad * x + rinv * (gx - rg * x);
    out[stride] = rad * y + rinv * (gy - rg * y);
    out[2 * stride] = rad * z + rinv * (gz - rg * z);
  }
}

static PEACE_INLINE void
evaluate_polynomial_1(double *a, const int a_stride, const double *cgs, const int *ms, double *B)
{
  double *dB_da = a;
  const int im1 = ms[0];
  const double cg = cgs[0];
  for (int ii = 0; ii < SIMD; ii++) {
    B[ii] = cg * a[im1 * a_stride + ii];
    dB_da[im1 * a_stride + ii] = cg;
  }
}

static PEACE_INLINE void
evaluate_polynomial_2(double *a, const int a_stride, const double *cgs, const int *ms, const int nterms, double *B)
{
  double *dB_da = a;
  for (int ii = 0; ii < SIMD; ii++)
    B[ii] = 0.0;
  for (int term = 0; term < nterms; term++) {
    const int im1 = ms[MAX_ORDER * term + 0];
    const int im2 = ms[MAX_ORDER * term + 1];
    const double cg = cgs[term];
    for (int ii = 0; ii < SIMD; ii++) {
      const double a1 = a[im1 * a_stride + ii];
      const double a2 = a[im2 * a_stride + ii];
      dB_da[im1 * a_stride + ii] = cg * a2;
      dB_da[im2 * a_stride + ii] = cg * a1;
      B[ii] += cg * a1 * a2;
    }
  }
}

static PEACE_INLINE void
evaluate_polynomial_3(double *a, const int a_stride, const int *a_first, const int *ls, const double *cgs, const int *ms, const int nterms, double *B, double *dB_da_shared = nullptr)
{
  int first[3];
  int l[3];
#pragma unroll
  for (int t = 0; t < 3; t++) {
    first[t] = a_first[t];
    l[t] = ls[t];
  }
  double *dB_da = a;
  for (int ii = 0; ii < SIMD; ii++)
    B[ii] = 0.0;
// apparently on CUDA runtime indexing goes to L1/L2
// workaround: dB_da on faster shared storage
// the alignas(64) were empirically found to give more consistent CPU times
#if defined(__CUDA_ARCH__) && defined(KOKKOS_ENABLE_CUDA)
  constexpr int dB_stride = TEAM_SIZE;
  double *dB_da1 = dB_da_shared;
  double *dB_da2 = dB_da_shared + (2 * l[0] + 1) * dB_stride;
#else
  constexpr int dB_stride = SIMD;
  alignas(64) double dB_da1[MAX_MS * SIMD] = {};
  alignas(64) double dB_da2[MAX_MS * SIMD] = {};
#endif
  int im3_prev = first[2] - 1;
  double a3[SIMD];
  double dB_da3[SIMD];
  for (int term = 0; term < nterms; term++) {
    const int im1 = ms[MAX_ORDER * term + 0];
    const int im2 = ms[MAX_ORDER * term + 1];
    const int im3 = ms[MAX_ORDER * term + 2];
    if (im3 != im3_prev) {
      if (im3_prev >= first[2])
        for (int ii = 0; ii < SIMD; ii++) {
          B[ii] += a3[ii] * dB_da3[ii];
          dB_da[im3_prev * a_stride + ii] = dB_da3[ii];
        }
      for (int im = im3_prev + 1; im < im3; im++)
        for (int ii = 0; ii < SIMD; ii++)
          dB_da[im * a_stride + ii] = 0.0;
      im3_prev = im3;
      for (int ii = 0; ii < SIMD; ii++) {
        a3[ii] = a[im3 * a_stride + ii];
        dB_da3[ii] = 0.0;
      }
    }
    const double cg = cgs[term];
    for (int ii = 0; ii < SIMD; ii++) {
      const double a1 = a[im1 * a_stride + ii];
      const double a2 = a[im2 * a_stride + ii];
      const double a12 = a1 * a2;
      dB_da1[(im1 - first[0]) * dB_stride + ii] += cg * a2 * a3[ii];
      dB_da2[(im2 - first[1]) * dB_stride + ii] += cg * a1 * a3[ii];
      dB_da3[ii] += cg * a12;
    }
  }
  for (int ii = 0; ii < SIMD; ii++) {
    B[ii] += a3[ii] * dB_da3[ii];
    dB_da[im3_prev * a_stride + ii] = dB_da3[ii];
  }
  for (int im = im3_prev + 1; im < first[2] + 2 * l[2] + 1; im++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[im * a_stride + ii] = 0.0;
  for (int m = 0; m < 2 * l[0] + 1; m++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[(first[0] + m) * a_stride + ii] = dB_da1[m * dB_stride + ii];
  for (int m = 0; m < 2 * l[1] + 1; m++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[(first[1] + m) * a_stride + ii] = dB_da2[m * dB_stride + ii];
}

// theoretically could be merged with 3 but in practice that slows down
static PEACE_INLINE void
evaluate_polynomial_4(double *a, const int a_stride, const int *a_first, const int *ls, const double *cgs, const int *ms, const int nterms, double *B, double *dB_da_shared = nullptr)
{
  int first[4];
  int l[4];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    first[t] = a_first[t];
    l[t] = ls[t];
  }
  double *dB_da = a;
  for (int ii = 0; ii < SIMD; ii++)
    B[ii] = 0.0;
#if defined(__CUDA_ARCH__) && defined(KOKKOS_ENABLE_CUDA)
  constexpr int dB_stride = TEAM_SIZE;
  double *dB_da2 = dB_da_shared;
  double *dB_da3 = dB_da_shared + (2 * l[1] + 1) * dB_stride;
  double *dB_da4 = dB_da_shared + (2 * l[1] + 2 * l[2] + 2) * dB_stride;
#else
  constexpr int dB_stride = SIMD;
  alignas(64) double dB_da2[MAX_MS * SIMD] = {};
  alignas(64) double dB_da3[MAX_MS * SIMD] = {};
  alignas(64) double dB_da4[MAX_MS * SIMD] = {};
#endif
  int im1_prev = first[0] - 1;
  double a1[SIMD];
  double dB_da1[SIMD];
  for (int term = 0; term < nterms; term++) {
    const int im1 = ms[MAX_ORDER * term + 0];
    const int im2 = ms[MAX_ORDER * term + 1];
    const int im3 = ms[MAX_ORDER * term + 2];
    const int im4 = ms[MAX_ORDER * term + 3];
    if (im1 != im1_prev) {
      if (im1_prev >= first[0])
        for (int ii = 0; ii < SIMD; ii++) {
          B[ii] += a1[ii] * dB_da1[ii];
          dB_da[im1_prev * a_stride + ii] = dB_da1[ii];
        }
      im1_prev = im1;
      for (int ii = 0; ii < SIMD; ii++) {
        a1[ii] = a[im1 * a_stride + ii];
        dB_da1[ii] = 0.0;
      }
    }
    const double cg = cgs[term];
    for (int ii = 0; ii < SIMD; ii++) {
      const double a2 = a[im2 * a_stride + ii];
      const double a3 = a[im3 * a_stride + ii];
      const double a4 = a[im4 * a_stride + ii];
      const double a12 = a1[ii] * a2;
      const double a34 = a3 * a4;
      dB_da1[ii] += cg * a2 * a34;
      dB_da2[(im2 - first[1]) * dB_stride + ii] += cg * a1[ii] * a34;
      dB_da3[(im3 - first[2]) * dB_stride + ii] += cg * a12 * a4;
      dB_da4[(im4 - first[3]) * dB_stride + ii] += cg * a12 * a3;
    }
  }
  for (int ii = 0; ii < SIMD; ii++) {
    B[ii] += a1[ii] * dB_da1[ii];
    dB_da[im1_prev * a_stride + ii] = dB_da1[ii];
  }
  for (int m = 0; m < 2 * l[1] + 1; m++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[(first[1] + m) * a_stride + ii] = dB_da2[m * dB_stride + ii];
  for (int m = 0; m < 2 * l[2] + 1; m++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[(first[2] + m) * a_stride + ii] = dB_da3[m * dB_stride + ii];
  for (int m = 0; m < 2 * l[3] + 1; m++)
    for (int ii = 0; ii < SIMD; ii++)
      dB_da[(first[3] + m) * a_stride + ii] = dB_da4[m * dB_stride + ii];
}

template <int tile>
static PEACE_INLINE void
forward_mix(double *a, const int a_stride, const double *A, const int A_stride, const double *W, const int nrad_A, const int al_count)
{
  double a_tile[tile * SIMD] = {};
#pragma GCC unroll 1
  for (int n = 0; n < nrad_A; n++) {
    double An[SIMD];
    for (int ii = 0; ii < SIMD; ii++)
      An[ii] = A[n * A_stride + ii];
#pragma GCC unroll 32
    for (int ia = 0; ia < tile; ia++)
      for (int ii = 0; ii < SIMD; ii++)
        a_tile[ia * SIMD + ii] += W[ia * nrad_A + n] * An[ii];
  }
#pragma GCC unroll 32
  for (int ia = 0; ia < tile; ia++)
    if (ia < al_count)
      for (int ii = 0; ii < SIMD; ii++)
        a[ia * a_stride + ii] = a_tile[ia * SIMD + ii];
}

template <int nrad>
static PEACE_INLINE void
backward_mix(double *dEdA_tile, const double *dBda, const int dBda_stride, const double *dEdB, const int dEdB_stride, const int *a_ipoly, const double *W, const int W_stride, const int al_count)
{
  for (int ia = 0; ia < al_count; ia++) {
    double dEda[SIMD];
    for (int ii = 0; ii < SIMD; ii++)
      dEda[ii] = dBda[ia * dBda_stride + ii] * dEdB[a_ipoly[ia] * dEdB_stride + ii];
#pragma GCC unroll 32
    for (int n = 0; n < nrad; n++)
      for (int ii = 0; ii < SIMD; ii++)
        dEdA_tile[n * SIMD + ii] += dEda[ii] * W[ia * W_stride + n];
  }
}

static PEACE_INLINE double
zbl_pair(const double *table, const int nelements, const int mui, const int muj, const int nbin, const double rnorm, const double dr_inv, double &evdwl)
{
  double w;
  const int bin = spline_bin(rnorm, dr_inv, nbin, w);
  const int a = MAX(mui, muj), b = MIN(mui, muj);
  const double *coeff = &table[(bin * (nelements * (nelements + 1) / 2) + a * (a + 1) / 2 + b) * 4];
  evdwl = spline_value(coeff, 1, w);
  return spline_deriv(coeff, 1, w, dr_inv);
}

static PEACE_INLINE void
fs_embedding(const double *params, const int ndensity, double *rho, const int stride, double *e)
{
  for (int p = 0; p < ndensity; p++) {
    const double mexp = params[4 * p];
    const double inv_mexp = params[4 * p + 1];
    const double xoff0 = params[4 * p + 2];
    const double weight = params[4 * p + 3];
    double *rhop = rho + p * stride;
#pragma clang loop unroll(disable) vectorize(enable)
    for (int ii = 0; ii < SIMD; ii++) {
      const double abs_rho = fabs(rhop[ii]);
      const double xoff = xoff0 * exp(-abs_rho);
      const double yoff = inv_mexp * xoff;
      const double base_pow = exp((mexp - 1.0) * log(xoff + abs_rho));
      e[ii] += weight * (rhop[ii] < 0.0 ? -1.0 : 1.0) * ((xoff + abs_rho) * base_pow - yoff);
      rhop[ii] = weight * (yoff + mexp * (1.0 - xoff) * base_pow);
    }
  }
}

}    // namespace PEACE
}    // namespace LAMMPS_NS

#undef PEACE_INLINE

#endif
