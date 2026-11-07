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
   CPU adaptation of pair_peace_kokkos
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(peace,PairPEACE);
// clang-format on
#else

#ifndef LMP_PAIR_PEACE_H
#define LMP_PAIR_PEACE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPEACE : public Pair {
 public:
  PairPEACE(class LAMMPS *);
  ~PairPEACE() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

 protected:
  virtual void allocate();
  void destroy_arrays();
  void resize_work_arrays(int max_jnum, int inum);
  int ComputeNeigh(int, int);
  template <int LMAX> void ComputeAi(int, int);
  void ComputeForwardMix();
  template <int NRAD> void ComputeBackwardMix(int);
  template <int LMAX> void ComputeForce(int, int, int, int, double);
  void print_kernel_times(const double *) const;

  int *mu_group;
  int ngroups;
  int *group_first;
  int lmax;
  int nrad_A;
  bool ldep;
  int nrad;
  int ndensity;
  double cutoff;
  double **fs_params;
  double *E0;

  double *dEdB;
  int max_npolys;

  double **scale;

  int chunksize; // 'chunksize' keyword (default 65536): atoms per kokkos chunk

  int print_kernel_timing; // 'timing' keyword (default 0): 1 to log per-kernel times every step

  bool subtract_zbl;
  double ***zbl_splines;

  int *a_first;
  int **a_ipoly;
  int *al_first;
  double **W;

  int nbins;
  double rln_dr_inv;
  double ***rln_splines;
  double **z_ln;

  int **nearest;
  int **neigh_mu;
  double **rnorms;
  double ***rhats;
  double *Ai;
  double *neigh_scratch;
  double **Rij_cache;
  double *A, *dEdA;
  double *a, *dBda;
  double *rho, *dEdrho;
  int *ilist_sorted;
  int **element_batches;
  int *element_natoms;
  int maxneigh;
  int alloc_inum;

  int npolys;
  int *poly_order;
  int *poly_term_first;
  int **poly_a_first;
  int **poly_ls;
  int **poly_ms;
  double *poly_cgs;
  double ***coeffs;
};
}    // namespace LAMMPS_NS

#endif
#endif
