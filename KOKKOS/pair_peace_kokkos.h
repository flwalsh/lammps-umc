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

#ifdef PAIR_CLASS
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// clang-format off
PairStyle(peace/kk,PairPEACEKokkos<LMPDeviceType>);
PairStyle(peace/kk/device,PairPEACEKokkos<LMPDeviceType>);
// clang-format on
#endif
#else

#ifndef LMP_PAIR_PEACE_KOKKOS_H
#define LMP_PAIR_PEACE_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "pair_peace.h"

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
namespace LAMMPS_NS {

template <class DeviceType> class PairPEACEKokkos : public PairPEACE {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  using ScratchViewType = Kokkos::View<KK_FLOAT *, typename DeviceType::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using IntScratchViewType = Kokkos::View<int *, typename DeviceType::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type team_member;
  template <class Tag, class... Props> using Range = Kokkos::RangePolicy<DeviceType, Tag, Props...>;

  struct TagPairPEACEComputeNeigh {};
  template <int LMAX, bool LDEP> struct TagPairPEACEComputeAi {};
  struct TagPairPEACEComputeForwardMix {};
  struct TagPairPEACEComputeDensities {};
  struct TagPairPEACEComputeEmbedding {};
  struct TagPairPEACEComputeWeights {};
  template <int NRAD> struct TagPairPEACEComputeBackwardMix {};
  template <int LMAX, bool LDEP, int EVFLAG> struct TagPairPEACEComputeForce {};

  template <int LMAX, bool LDEP> void launch_Ai();
  template <int LMAX, bool LDEP> void launch_force(EV_FLOAT &);
  template <int NRAD> void launch_backward_mix();

  PairPEACEKokkos(class LAMMPS *);
  ~PairPEACEKokkos() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  int chunk_size, chunk_offset, nthreads;
  int max_terms, max_dB;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d d_ilist_sorted;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  DAT::ttransform_kkacc_1d_9 k_cvatom;
  typename AT::t_kkacc_1d_9 d_cvatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  Kokkos::DualView<KK_FLOAT **, DeviceType> k_scale;
  Kokkos::View<KK_FLOAT **, DeviceType> d_scale;

  typename AT::t_int_1d d_map;

  friend void pair_virial_fdotr_compute<PairPEACEKokkos>(PairPEACEKokkos *);

  void grow();
  void allocate() override;
  double memory_usage() override;

  Kokkos::View<KK_FLOAT *, DeviceType> d_pool;

  Kokkos::View<KK_FLOAT ***, Kokkos::LayoutLeft, DeviceType> d_A, d_dEdA;
  Kokkos::View<KK_FLOAT **, Kokkos::LayoutLeft, DeviceType> d_a, d_dBda;
  Kokkos::View<KK_FLOAT **, Kokkos::LayoutLeft, DeviceType> d_dEdB;

  Kokkos::View<KK_FLOAT *, DeviceType> d_e_embed;
  Kokkos::View<KK_FLOAT **, Kokkos::LayoutLeft, DeviceType> d_rho, d_dEdrho;

  Kokkos::View<int *, DeviceType> d_jcount;
  Kokkos::View<int *, DeviceType> d_mu;
  Kokkos::View<int *, DeviceType> d_mu_group;
  Kokkos::View<KK_FLOAT **, Kokkos::LayoutRight, DeviceType> d_rnorms;
  Kokkos::View<KK_FLOAT **[3], Kokkos::LayoutRight, DeviceType> d_rhats;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_nearest;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_neigh_mu;

  Kokkos::View<KK_FLOAT ***, Kokkos::LayoutRight, DeviceType> d_rln_splines;
  Kokkos::View<KK_FLOAT **, Kokkos::LayoutRight, DeviceType> d_z_ln;

  Kokkos::View<KK_FLOAT **, Kokkos::LayoutRight, DeviceType> d_fs_params;

  Kokkos::View<KK_FLOAT *, DeviceType> d_E0;
  Kokkos::View<KK_FLOAT **[4], Kokkos::LayoutRight, DeviceType> d_zbl_splines;

  Kokkos::View<int *, Kokkos::LayoutRight, DeviceType> d_group_first;
  Kokkos::View<int *, Kokkos::LayoutRight, DeviceType> d_poly_order;
  Kokkos::View<int *, Kokkos::LayoutRight, DeviceType> d_poly_term_first;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_poly_ms;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_a_ipoly;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_poly_a_first;
  Kokkos::View<int **, Kokkos::LayoutRight, DeviceType> d_poly_ls;
  Kokkos::View<KK_FLOAT *, Kokkos::LayoutRight, DeviceType> d_poly_cgs;
  Kokkos::View<KK_FLOAT ***, Kokkos::LayoutLeft, DeviceType> d_coeffs;

  Kokkos::View<KK_FLOAT **, Kokkos::LayoutRight, DeviceType> d_W;
  Kokkos::View<int *, DeviceType> d_al_first;
  Kokkos::View<int *, DeviceType> d_a_first;

 public:
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeNeigh, const team_member &team) const;

  template <int LMAX, bool LDEP>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeAi<LMAX, LDEP>, const int &idx) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeForwardMix, const team_member &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeDensities, const team_member &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeEmbedding, const int &ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeWeights, const int &idx) const;

  template <int NRAD>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeBackwardMix<NRAD>, const team_member &team) const;

  template <int LMAX, bool LDEP, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeForce<LMAX, LDEP, EVFLAG>, const int &idx, EV_FLOAT &ev) const;

  template <int LMAX, bool LDEP, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairPEACEComputeForce<LMAX, LDEP, EVFLAG>, const int &idx) const
  {
    EV_FLOAT ev;
    operator()(TagPairPEACEComputeForce<LMAX, LDEP, EVFLAG>(), idx, ev);
  }
};
}    // namespace LAMMPS_NS
#endif

#endif
#endif
