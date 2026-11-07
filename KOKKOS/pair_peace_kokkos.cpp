/* ----------------------------------------------------------------------
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
   Originally adapted from pair_pace_kokkos by Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_peace_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "pair_peace_impl.h"
#include "platform.h"

#include <Kokkos_Sort.hpp>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

using namespace LAMMPS_NS;
using namespace PEACE;

template <class DeviceType>
PairPEACEKokkos<DeviceType>::PairPEACEKokkos(LAMMPS *lmp) : PairPEACE(lmp), max_terms(0), max_dB(0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

template <class DeviceType> PairPEACEKokkos<DeviceType>::~PairPEACEKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  memoryKK->destroy_kokkos(k_cvatom, cvatom);
}

template <class DeviceType> void PairPEACEKokkos<DeviceType>::grow()
{
  if ((int) d_A.extent(0) < chunk_size) {
    MemKK::realloc_kokkos(d_pool, "peace:pool", (int64_t) chunk_size * (a_first[al_first[lmax + 1]] + lm_size(lmax) * nrad_A + ndensity + 1));
    KK_FLOAT *pool = d_pool.data();

    d_a = decltype(d_a)(pool, chunk_size, a_first[al_first[lmax + 1]]);
    pool += d_a.span();
    d_A = decltype(d_A)(pool, chunk_size, lm_size(lmax), nrad_A);
    pool += d_A.span();
    d_rho = decltype(d_rho)(pool, chunk_size, ndensity);
    pool += d_rho.span();
    d_e_embed = decltype(d_e_embed)(pool, chunk_size);
    d_dEdA = d_A;
    d_dBda = d_a;
    d_dEdrho = d_rho;

    MemKK::realloc_kokkos(d_dEdB, "peace:dEdB", chunk_size, max_npolys + 1);
    Kokkos::deep_copy(Kokkos::subview(d_dEdB, Kokkos::ALL, max_npolys), 0.0); // padding
    MemKK::realloc_kokkos(d_jcount, "peace:jcount", chunk_size);
    MemKK::realloc_kokkos(d_mu, "peace:mu", chunk_size);
  }

  const int maxneigh = d_neighbors.extent(1);
  if (((int) d_rnorms.extent(0) < chunk_size) || ((int) d_rnorms.extent(1) < maxneigh)) {
    MemKK::realloc_kokkos(d_rhats, "peace:rhats", chunk_size, maxneigh);
    MemKK::realloc_kokkos(d_rnorms, "peace:rnorms", chunk_size, maxneigh);
    MemKK::realloc_kokkos(d_nearest, "peace:nearest", chunk_size, maxneigh);
    MemKK::realloc_kokkos(d_neigh_mu, "peace:neigh_mu", chunk_size, maxneigh);
  }
}

template <class DeviceType>
template <int LMAX, bool LDEP>
void PairPEACEKokkos<DeviceType>::launch_Ai()
{
  Kokkos::parallel_for("ComputeAi", Range<TagPairPEACEComputeAi<LMAX, LDEP>, Kokkos::LaunchBounds<TEAM_SIZE, 1>>(0, chunk_size * nrad_A), *this);
}

template <class DeviceType>
template <int LMAX, bool LDEP>
void PairPEACEKokkos<DeviceType>::launch_force(EV_FLOAT &ev)
{
  const int n = (chunk_size * nthreads + 31) / 32 * 32;
  const Kokkos::ChunkSize align(TEAM_SIZE);
  if (evflag)
    Kokkos::parallel_reduce("ComputeForce", Range<TagPairPEACEComputeForce<LMAX, LDEP, 1>, Kokkos::LaunchBounds<TEAM_SIZE, 1>>(0, n, align), *this, ev);
  else
    Kokkos::parallel_for("ComputeForce", Range<TagPairPEACEComputeForce<LMAX, LDEP, 0>, Kokkos::LaunchBounds<TEAM_SIZE, 1>>(0, n, align), *this);
}

template <class DeviceType>
template <int NRAD>
void PairPEACEKokkos<DeviceType>::launch_backward_mix()
{
  typedef TagPairPEACEComputeBackwardMix<NRAD> Tag;
  const int teams = (chunk_size + TEAM_SIZE - 1) / TEAM_SIZE;
  Kokkos::TeamPolicy<DeviceType, Tag> policy(lm_size(lmax) * teams, TEAM_SIZE);
  policy.set_scratch_size(0, Kokkos::PerTeam(ScratchViewType::shmem_size(MIX_TILE * NRAD) +
                                         IntScratchViewType::shmem_size(ngroups * MIX_TILE)));
  Kokkos::parallel_for("ComputeBackwardMix", policy, *this);
}

template <class DeviceType> void PairPEACEKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  copymode = 1;

  ev_init(eflag_in, vflag_in, 0);

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.template view<DeviceType>();
  }
  if (cvflag_atom) {
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
    memoryKK->create_kokkos(k_cvatom, cvatom, maxcvatom, "pair:cvatom");
    d_cvatom = k_cvatom.template view<DeviceType>();
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  k_scale.sync_device();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  const int inum = list->inum;

  if (ngroups > 1) {
    if ((int) d_ilist_sorted.extent(0) < inum)
      MemKK::realloc_kokkos(d_ilist_sorted, "peace:ilist_sorted", inum);
    Kokkos::parallel_for("SortKey", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_CLASS_LAMBDA(const int ii) {
      d_ilist_sorted(ii) = d_mu_group(d_map(type(d_ilist[ii]))) * inum + ii;
    });
    Kokkos::sort(DeviceType(), Kokkos::subview(d_ilist_sorted, Kokkos::make_pair(0, inum)));
    Kokkos::parallel_for("SortOrder", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_CLASS_LAMBDA(const int ii) {
      d_ilist_sorted(ii) = d_ilist[d_ilist_sorted(ii) % inum];
    });
    d_ilist = d_ilist_sorted;
  }

  chunk_size = MIN(chunksize, inum);

  grow();

  EV_FLOAT ev;
  double times[9] = {};
  const double t_start = print_kernel_timing ? platform::walltime() : 0.0;
  double t_last = t_start;

  for (chunk_offset = 0; chunk_offset < inum; chunk_offset += chunk_size) {
    chunk_size = MIN(chunksize, inum - chunk_offset);

    EV_FLOAT ev_tmp;
    const int teams = (chunk_size + TEAM_SIZE - 1) / TEAM_SIZE;

    constexpr int atoms_per_team = TEAM_SIZE / NEIGH_THREADS;
    Kokkos::TeamPolicy<DeviceType, TagPairPEACEComputeNeigh> neigh_policy((chunk_size + atoms_per_team - 1) / atoms_per_team, atoms_per_team, NEIGH_THREADS);
    Kokkos::parallel_for("ComputeNeigh", neigh_policy, *this);
    stamp(print_kernel_timing, times, 1, t_last);

    switch (lmax) {
      case 2: ldep ? launch_Ai<2, true>() : launch_Ai<2, false>(); break;
      case 3: ldep ? launch_Ai<3, true>() : launch_Ai<3, false>(); break;
      case 4: ldep ? launch_Ai<4, true>() : launch_Ai<4, false>(); break;
      case 5: ldep ? launch_Ai<5, true>() : launch_Ai<5, false>(); break;
    }
    stamp(print_kernel_timing, times, 2, t_last);

    Kokkos::TeamPolicy<DeviceType, TagPairPEACEComputeForwardMix> mix_policy(lm_size(lmax) * teams, TEAM_SIZE);
    mix_policy.set_scratch_size(0, Kokkos::PerTeam(ScratchViewType::shmem_size(MIX_TILE * nrad_A)));
    Kokkos::parallel_for("ComputeForwardMix", mix_policy, *this);
    stamp(print_kernel_timing, times, 3, t_last);

    Kokkos::deep_copy(d_rho, 0.0);
    Kokkos::TeamPolicy<DeviceType, TagPairPEACEComputeDensities> density_policy(npolys * teams, TEAM_SIZE);
    density_policy.set_scratch_size(
        0, Kokkos::PerTeam(ScratchViewType::shmem_size(max_dB * DB_STRIDE) +
                           ScratchViewType::shmem_size(max_terms) +
                           IntScratchViewType::shmem_size(max_terms * MAX_ORDER)));
    Kokkos::parallel_for("ComputeDensities", density_policy, *this);
    stamp(print_kernel_timing, times, 4, t_last);

    Kokkos::parallel_for("ComputeEmbedding", Range<TagPairPEACEComputeEmbedding>(0, chunk_size), *this);
    stamp(print_kernel_timing, times, 5, t_last);

    Kokkos::parallel_for("ComputeWeights", Range<TagPairPEACEComputeWeights>(0, chunk_size * max_npolys), *this);
    stamp(print_kernel_timing, times, 6, t_last);

    switch ((nrad_A + 3) / 4) {
      case 1: launch_backward_mix<4>(); break;
      case 2: launch_backward_mix<8>(); break;
      case 3: launch_backward_mix<12>(); break;
      case 4: launch_backward_mix<16>(); break;
      case 5: launch_backward_mix<20>(); break;
      case 6: launch_backward_mix<24>(); break;
      case 7: launch_backward_mix<28>(); break;
      case 8: launch_backward_mix<32>(); break;
    }
    stamp(print_kernel_timing, times, 7, t_last);

    switch (lmax) {
      case 2: ldep ? launch_force<2, true>(ev_tmp) : launch_force<2, false>(ev_tmp); break;
      case 3: ldep ? launch_force<3, true>(ev_tmp) : launch_force<3, false>(ev_tmp); break;
      case 4: ldep ? launch_force<4, true>(ev_tmp) : launch_force<4, false>(ev_tmp); break;
      case 5: ldep ? launch_force<5, true>(ev_tmp) : launch_force<5, false>(ev_tmp); break;
    }
    stamp(print_kernel_timing, times, 8, t_last);

    ev += ev_tmp;
  }

  if (print_kernel_timing) {
    times[0] = platform::walltime() - t_start;
    print_kernel_times(times);
  }

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global)
    for (int k = 0; k < 6; k++) virial[k] += ev.v[k];

  if (eflag_atom) {
    k_eatom.modify_device();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.modify_device();
    k_vatom.sync_host();
  }

  if (cvflag_atom) {
    k_cvatom.modify_device();
    k_cvatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  atomKK->modified(execution_space, F_MASK);

  copymode = 0;
}

template <class DeviceType> void PairPEACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairPEACE::coeff(narg, arg);

  auto to_device = [](auto &d_view, auto *data, auto... dims) {
    // On APUs host_mirror_type would alias without copying so explicit HostSpace
    typedef std::decay_t<decltype(d_view)> DeviceView;
    typedef Kokkos::View<typename DeviceView::data_type, typename DeviceView::array_layout, Kokkos::HostSpace> HostView;
    d_view = Kokkos::create_mirror_view_and_copy(DeviceType(), HostView(data, dims...));
  };
  to_device(d_map, map, atom->ntypes + 1);
  to_device(d_fs_params, &fs_params[0][0], ndensity, 4);
  to_device(d_E0, E0, nelements);
  to_device(d_rln_splines, &rln_splines[0][0][0], nbins, 4, nrad);
  to_device(d_z_ln, &z_ln[0][0], nelements, nrad);
  if (subtract_zbl) to_device(d_zbl_splines, &zbl_splines[0][0][0], nbins, nelements * (nelements + 1) / 2);
  to_device(d_mu_group, mu_group, nelements);
  to_device(d_group_first, group_first, ngroups + 1);
  to_device(d_poly_order, poly_order, npolys);
  to_device(d_poly_term_first, poly_term_first, npolys + 1);
  to_device(d_poly_a_first, &poly_a_first[0][0], npolys, MAX_ORDER);
  to_device(d_poly_ls, &poly_ls[0][0], npolys, MAX_ORDER);
  to_device(d_poly_ms, &poly_ms[0][0], poly_term_first[npolys], MAX_ORDER);
  to_device(d_poly_cgs, poly_cgs, poly_term_first[npolys]);
  to_device(d_a_ipoly, &a_ipoly[0][0], ngroups, al_first[lmax + 1]);
  to_device(d_W, &W[0][0], al_first[lmax + 1], nrad_A);
  to_device(d_al_first, al_first, lmax + 2);
  to_device(d_a_first, a_first, al_first[lmax + 1] + 1);

  MemKK::realloc_kokkos(d_coeffs, "peace:coeffs", nelements, max_npolys, ndensity);
  auto h_coeffs = Kokkos::create_mirror_view(d_coeffs);
  Kokkos::deep_copy(h_coeffs, Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace>(&coeffs[0][0][0], nelements, max_npolys, ndensity));
  Kokkos::deep_copy(d_coeffs, h_coeffs);

  max_terms = max_dB = 0;
  for (int ipoly = 0; ipoly < npolys; ipoly++) {
    max_terms = MAX(max_terms, poly_term_first[ipoly + 1] - poly_term_first[ipoly]);
    const int order = poly_order[ipoly];
    const int *ls = poly_ls[ipoly];
    const int dB_size = order == 3 ? 2 * (ls[0] + ls[1]) + 2
        : order == 4 ? 2 * (ls[1] + ls[2] + ls[3]) + 3 : 0;
    max_dB = MAX(max_dB, dB_size);
  }

  nthreads = 1;
  while (nthreads * (ldep ? NL_PER_THREAD : N_PER_THREAD) < nrad_A)
    nthreads *= 2;

  d_A = decltype(d_A)();
}

template <class DeviceType> double PairPEACEKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairPEACE::init_one(i, j);
  auto h_scale = k_scale.view_host();

  h_scale(i, j) = h_scale(j, i) = scale[i][j];
  k_scale.modify_host();

  return cutone;
}

template <class DeviceType> void PairPEACEKokkos<DeviceType>::init_style()
{
  PairPEACE::init_style();
  neighbor->find_request(this)->set_kokkos_device(1);
}

template <class DeviceType> void PairPEACEKokkos<DeviceType>::allocate()
{
  PairPEACE::allocate();

  int n = atom->ntypes + 1;

  k_scale = decltype(k_scale)("pair:scale", n, n);
  d_scale = k_scale.template view<DeviceType>();
}

template <class DeviceType> double PairPEACEKokkos<DeviceType>::memory_usage()
{
  double bytes = PairPEACE::memory_usage();

  const size_t real_values = d_pool.span() + d_rnorms.span() +
      d_rhats.span() + d_z_ln.span() + d_rln_splines.span() + d_fs_params.span() +
      d_E0.span() + d_zbl_splines.span() + d_scale.span() +
      d_poly_cgs.span() + d_coeffs.span() + d_W.span() + d_dEdB.span();
  const size_t int_values = d_ilist_sorted.span() + d_jcount.span() + d_mu.span() + d_mu_group.span() +
      d_nearest.span() + d_map.span() + d_neigh_mu.span() + d_poly_order.span() +
      d_poly_term_first.span() + d_poly_ms.span() + d_a_ipoly.span() + d_poly_a_first.span() +
      d_poly_ls.span() + d_group_first.span() + d_al_first.span() + d_a_first.span();
  bytes += real_values * sizeof(KK_FLOAT) + int_values * sizeof(int);

  return bytes;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeNeigh, const team_member &team) const
{
  const int ii = team.league_rank() * team.team_size() + team.team_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const KK_FLOAT xtmp = x(i, 0);
  const KK_FLOAT ytmp = x(i, 1);
  const KK_FLOAT ztmp = x(i, 2);
  const int jnum = d_numneigh(i);
  const KK_FLOAT rcutsq = cutoff * cutoff;

  int jcount = 0;
  Kokkos::parallel_scan(
      Kokkos::ThreadVectorRange(team, jnum),
      [&](const int jj, int &offset, const bool final) {
        const int j = d_neighbors(i, jj) & NEIGHMASK;
        const KK_FLOAT delx = xtmp - x(j, 0);
        const KK_FLOAT dely = ytmp - x(j, 1);
        const KK_FLOAT delz = ztmp - x(j, 2);
        const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
        if (rsq >= rcutsq) return;

        if (final) {
          const KK_FLOAT rij = sqrt(rsq);
          const KK_FLOAT rinv = 1.0 / rij;
          d_nearest(ii, offset) = j;
          d_neigh_mu(ii, offset) = d_map(type(j));
          d_rhats(ii, offset, 0) = -delx * rinv;
          d_rhats(ii, offset, 1) = -dely * rinv;
          d_rhats(ii, offset, 2) = -delz * rinv;
          d_rnorms(ii, offset) = rij;
        }
        ++offset;
      },
      jcount);
  d_jcount(ii) = jcount;
  d_mu(ii) = d_map(type(i));
}

template <class DeviceType>
template <int LMAX, bool LDEP>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeAi<LMAX, LDEP>, const int &idx) const
{
  constexpr int lm_count = lm_size(LMAX);
  const int n = idx % nrad_A;
  const int ii = idx / nrad_A;
  KK_FLOAT accum[lm_count] = {};

  const int jcount = d_jcount(ii);
  for (int jj = 0; jj < jcount; jj++) {
    const KK_FLOAT rij = d_rnorms(ii, jj);
    const int muj = d_neigh_mu(ii, jj);
    const KK_FLOAT *rhat = &d_rhats(ii, jj, 0);
    KK_FLOAT R[LDEP ? LMAX + 1 : 1];
    KK_FLOAT w;
    const int bin = spline_bin(rij, rln_dr_inv, nbins, w);
#pragma unroll
    for (int l = 0; l <= (LDEP ? LMAX : 0); l++) {
      const int ln = l * nrad_A + n;
      const KK_FLOAT *coeff = &d_rln_splines(bin, 0, ln);
      const KK_FLOAT zj = d_z_ln(muj, ln);
      R[l] = spline_value(coeff, nrad, w) * zj;
    }
    compute_ylm<LMAX, false, LDEP>(rhat[0], rhat[1], rhat[2], 1, accum, R);
  }

#pragma unroll
  for (int lm = 0; lm < lm_count; lm++)
    d_A(ii, lm, n) = accum[lm];
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeForwardMix, const team_member &team) const
{
  const int lm = team.league_rank() % lm_size(lmax);
  const int ii = (team.league_rank() / lm_size(lmax)) * TEAM_SIZE + team.team_rank();
  const int l = (int) sqrt((double) lm);
  const int m = lm - l * l;
  const int ia_lo = d_al_first(l);
  const int ia_hi = d_al_first(l + 1);
  ScratchViewType W_tile(team.team_scratch(0), MIX_TILE * nrad_A);

  for (int ia0 = ia_lo; ia0 < ia_hi; ia0 += MIX_TILE) {
    for (int itile = team.team_rank(); itile < MIX_TILE * nrad_A; itile += team.team_size()) {
      const int ia = ia0 + itile / nrad_A;
      W_tile(itile) = ia < ia_hi ? d_W(ia, itile % nrad_A) : 0.0;
    }
    team.team_barrier();
    if (ii < chunk_size)
      forward_mix<MIX_TILE>(&d_a(ii, d_a_first(ia0) + m), (2 * l + 1) * d_a.stride(1), &d_A(ii, lm, 0), d_A.stride(2), W_tile.data(), nrad_A, ia_hi - ia0);
    team.team_barrier();
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeDensities, const team_member &team) const
{
  const int teams = (chunk_size + TEAM_SIZE - 1) / TEAM_SIZE;
  const int ipoly = team.league_rank() / teams;
  const int ii = (team.league_rank() % teams) * TEAM_SIZE + team.team_rank();
  const int order = d_poly_order(ipoly);
  const int term_first = d_poly_term_first(ipoly);
  const int nterms = d_poly_term_first(ipoly + 1) - term_first;
  ScratchViewType dB_da_shared(team.team_scratch(0), max_dB * DB_STRIDE);
  ScratchViewType cgs(team.team_scratch(0), max_terms);
  IntScratchViewType ms(team.team_scratch(0), max_terms * MAX_ORDER);

  for (int term = team.team_rank(); term < nterms; term += team.team_size()) {
    cgs(term) = d_poly_cgs(term_first + term);
    for (int t = 0; t < MAX_ORDER; t++)
      ms(term * MAX_ORDER + t) = d_poly_ms(term_first + term, t);
  }
  team.team_barrier();
  if (ii >= chunk_size) return;

  const int mu = d_mu(ii);
  const int igroup = d_mu_group(mu);
  if (ipoly < d_group_first(igroup) || ipoly >= d_group_first(igroup + 1)) return;
  const int *first = &d_poly_a_first(ipoly, 0);
  const int *ls = &d_poly_ls(ipoly, 0);
  KK_FLOAT *dB_da = dB_da_shared.data() + team.team_rank();
  // technically inefficient but simpler and doesn't seem to hurt
  for (int k = 0; k < max_dB * DB_STRIDE; k += DB_STRIDE)
    dB_da[k] = 0.0;

  KK_FLOAT B;
  switch (order) {
    case 1: evaluate_polynomial_1(&d_a(ii, 0), d_a.stride(1), cgs.data(), ms.data(), &B); break;
    case 2: evaluate_polynomial_2(&d_a(ii, 0), d_a.stride(1), cgs.data(), ms.data(), nterms, &B); break;
    case 3: evaluate_polynomial_3(&d_a(ii, 0), d_a.stride(1), first, ls, cgs.data(), ms.data(), nterms, &B, dB_da); break;
    case 4: evaluate_polynomial_4(&d_a(ii, 0), d_a.stride(1), first, ls, cgs.data(), ms.data(), nterms, &B, dB_da); break;
  }
  const int ipoly_local = ipoly - d_group_first(igroup);
  for (int p = 0; p < ndensity; p++)
    Kokkos::atomic_add(&d_rho(ii, p), B * d_coeffs(mu, ipoly_local, p));
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeEmbedding, const int &ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  KK_FLOAT evdwl = d_E0(d_mu(ii));
  fs_embedding(d_fs_params.data(), ndensity, &d_rho(ii, 0), d_rho.stride(1), &evdwl);
  const KK_FLOAT scale = d_scale(type(i), type(i));
  d_e_embed(ii) = evdwl * scale;
  for (int p = 0; p < ndensity; p++)
    d_dEdrho(ii, p) *= scale;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeWeights, const int &idx) const
{
  const int ipoly = idx / chunk_size;
  const int ii = idx % chunk_size;
  const int mu = d_mu(ii);

  KK_FLOAT dEdBi = 0.0;
  for (int p = 0; p < ndensity; p++)
    dEdBi += d_dEdrho(ii, p) * d_coeffs(mu, ipoly, p);
  d_dEdB(ii, ipoly) = dEdBi;
}

template <class DeviceType>
template <int NRAD>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeBackwardMix<NRAD>, const team_member &team) const
{
  const int lm = team.league_rank() % lm_size(lmax);
  const int ii = (team.league_rank() / lm_size(lmax)) * TEAM_SIZE + team.team_rank();
  const int l = (int) sqrt((double) lm);
  const int m = lm - l * l;
  const int ia_lo = d_al_first(l);
  const int ia_hi = d_al_first(l + 1);
  const int igroup = ii < chunk_size ? d_mu_group(d_mu(ii)) : 0;
  ScratchViewType W_tile(team.team_scratch(0), MIX_TILE * NRAD);
  IntScratchViewType a_ipoly_tile(team.team_scratch(0), ngroups * MIX_TILE);

  KK_FLOAT dEdA_tile[NRAD] = {};
  for (int ia0 = ia_lo; ia0 < ia_hi; ia0 += MIX_TILE) {
    for (int itile = team.team_rank(); itile < MIX_TILE * NRAD; itile += team.team_size()) {
      const int ia = ia0 + itile / NRAD;
      W_tile(itile) = ia < ia_hi && itile % NRAD < nrad_A ? d_W(ia, itile % NRAD) : 0.0;
    }
    // this staging seems important for A100 unfortunately
    for (int itile = team.team_rank(); itile < ngroups * MIX_TILE; itile += team.team_size()) {
      const int group = itile / MIX_TILE;
      const int ia = ia0 + itile % MIX_TILE;
      a_ipoly_tile(itile) = ia < ia_hi ? d_a_ipoly(group, ia) : max_npolys;
    }
    team.team_barrier();
    if (ii < chunk_size)
      backward_mix<NRAD>(dEdA_tile, &d_dBda(ii, d_a_first(ia0) + m), (2 * l + 1) * d_dBda.stride(1),
                         &d_dEdB(ii, 0), d_dEdB.stride(1), &a_ipoly_tile(igroup * MIX_TILE),
                         W_tile.data(), NRAD, MIN(MIX_TILE, ia_hi - ia0));
    team.team_barrier();
  }
  if (ii < chunk_size)
    for (int n = 0; n < NRAD; n++)
      if (n < nrad_A)
        d_dEdA(ii, lm, n) = dEdA_tile[n];
}

template <class DeviceType>
template <int LMAX, bool LDEP, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairPEACEKokkos<DeviceType>::operator()(TagPairPEACEComputeForce<LMAX, LDEP, EVFLAG>, const int &idx, EV_FLOAT &ev) const
{
  constexpr int lm_count = lm_size(LMAX);
  const int n0 = idx % nthreads;
  const bool pad = idx >= chunk_size * nthreads;
  const int ii = MIN(idx / nthreads, chunk_size - 1);
  const int i = d_ilist[ii + chunk_offset];
  KK_FLOAT fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0, evdwl = 0.0;
  KK_FLOAT vi[6] = {};

  KK_FLOAT dEdAi[lm_count][LDEP ? NL_PER_THREAD : N_PER_THREAD];
#pragma unroll
  for (int lm = 0; lm < lm_count; lm++)
#pragma unroll
    for (int nn = 0; nn < (LDEP ? NL_PER_THREAD : N_PER_THREAD); nn++) {
      const int n = n0 + nn * nthreads;
      dEdAi[lm][nn] = n < nrad_A ? d_dEdA(ii, lm, n) : 0.0;
    }

  const int mu = d_mu(ii);
  const int jcount = pad ? 0 : d_jcount(ii);
  const int jlast = MAX(jcount - 1, 0);
  int jcount_warp = jcount;
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    const int other = Kokkos::shfl_down(jcount_warp, offset, 32);
    jcount_warp = MAX(jcount_warp, other);
  }
  jcount_warp = Kokkos::shfl(jcount_warp, 0, 32);
#endif
  KK_FLOAT rij_next = d_rnorms(ii, 0);
  int muj_next = d_neigh_mu(ii, 0);
  for (int jj = 0; jj < jcount_warp; jj++) {
    const KK_FLOAT rij = rij_next;
    const int muj = muj_next;
    rij_next = d_rnorms(ii, MIN(jj + 1, jlast));
    muj_next = d_neigh_mu(ii, MIN(jj + 1, jlast));
    const int jc = MIN(jj, jlast);
    const KK_FLOAT *rhat = &d_rhats(ii, jc, 0);
    KK_FLOAT R[LDEP ? LMAX + 1 : 1][LDEP ? NL_PER_THREAD : N_PER_THREAD];
    KK_FLOAT dR[LDEP ? LMAX + 1 : 1][LDEP ? NL_PER_THREAD : N_PER_THREAD];
    KK_FLOAT w;
    // re-evaluate R/dR now sinced not cached
    const int bin = spline_bin(rij, rln_dr_inv, nbins, w);
#pragma unroll
    for (int l = 0; l <= (LDEP ? LMAX : 0); l++)
#pragma unroll
      for (int nn = 0; nn < (LDEP ? NL_PER_THREAD : N_PER_THREAD); nn++) {
        const int n = MIN(n0 + nn * nthreads, nrad_A - 1);
        const int ln = l * nrad_A + n;
        const KK_FLOAT *coeff = &d_rln_splines(bin, 0, ln);
        const KK_FLOAT zj = d_z_ln(muj, ln);
        R[l][nn] = spline_value(coeff, nrad, w) * zj;
        dR[l][nn] = spline_deriv(coeff, nrad, w, rln_dr_inv) * zj;
      }

    // now the first step in the chain rule P = dE/dA dR/dr, Q = dE/dA R
    KK_FLOAT P[lm_count], Q[lm_count];
#pragma unroll
    for (int lm = 0, l = 0; lm < lm_count; lm++) {
      // I would prefer sqrt for l but that slows down CUDA
      if (lm == (l + 1) * (l + 1)) ++l;
      const KK_FLOAT *Rl = R[LDEP ? l : 0];
      const KK_FLOAT *dRl = dR[LDEP ? l : 0];
      const KK_FLOAT *dEdAlm = dEdAi[lm];
      KK_FLOAT Plm = 0.0, Qlm = 0.0;
#pragma unroll
      for (int nn = 0; nn < (LDEP ? NL_PER_THREAD : N_PER_THREAD); nn++) {
        Plm += dEdAlm[nn] * dRl[nn];
        Qlm += dEdAlm[nn] * Rl[nn];
      }
      P[lm] = Plm;
      Q[lm] = Qlm;
    }
    KK_FLOAT fij[3];
    compute_ylm<LMAX, true>(rhat[0], rhat[1], rhat[2], 1, fij, nullptr, P, Q, (KK_FLOAT) 1.0 / rij);

    // now we need an all-to-one reduction to thread 0, which sets forces
    // this is basically a hand-rolled parallel reduce
    // note must be power of 2
    // naive Kokkos team policy is slow
    // only works with HIP/CUDA so no host
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
    for (int offset = nthreads / 2; offset > 0; offset /= 2) {
      fij[0] += Kokkos::shfl_down(fij[0], offset, nthreads);
      fij[1] += Kokkos::shfl_down(fij[1], offset, nthreads);
      fij[2] += Kokkos::shfl_down(fij[2], offset, nthreads);
    }
#endif

    // thread 0 does the work of a PACE force kernel while fij is loaded
    if (n0 == 0 && jj < jcount) {
      const int j = d_nearest(ii, jj);

      if (subtract_zbl) {
        KK_FLOAT ezbl;
        const KK_FLOAT dEdr = zbl_pair(d_zbl_splines.data(), nelements, mu, muj, nbins, rij, rln_dr_inv, ezbl);
        fij[0] += dEdr * rhat[0];
        fij[1] += dEdr * rhat[1];
        fij[2] += dEdr * rhat[2];
        if (EVFLAG && eflag_either) evdwl += ezbl;
      }

      fxtmp += fij[0];
      fytmp += fij[1];
      fztmp += fij[2];
      Kokkos::atomic_add(&f(j, 0), -fij[0]);
      Kokkos::atomic_add(&f(j, 1), -fij[1]);
      Kokkos::atomic_add(&f(j, 2), -fij[2]);

      if (EVFLAG && vflag_either) {
        const KK_FLOAT delx = -rhat[0] * rij;
        const KK_FLOAT dely = -rhat[1] * rij;
        const KK_FLOAT delz = -rhat[2] * rij;
        KK_FLOAT v[9];
        v[0] = delx * fij[0];
        v[1] = dely * fij[1];
        v[2] = delz * fij[2];
        v[3] = delx * fij[1];
        v[4] = delx * fij[2];
        v[5] = dely * fij[2];
        if (vflag_global)
          for (int k = 0; k < 6; k++) vi[k] += v[k];
        if (cvflag_atom) {
          v[6] = dely * fij[0];
          v[7] = delz * fij[0];
          v[8] = delz * fij[1];
          for (int k = 0; k < 9; k++) {
            Kokkos::atomic_add(&d_cvatom(i, k), 0.5 * v[k]);
            Kokkos::atomic_add(&d_cvatom(j, k), 0.5 * v[k]);
          }
        }
        if (vflag_atom)
          for (int k = 0; k < 6; k++) {
            Kokkos::atomic_add(&d_vatom(i, k), 0.5 * v[k]);
            Kokkos::atomic_add(&d_vatom(j, k), 0.5 * v[k]);
          }
      }
    }
  }

  if (n0 == 0 && !pad) {
    Kokkos::atomic_add(&f(i, 0), fxtmp);
    Kokkos::atomic_add(&f(i, 1), fytmp);
    Kokkos::atomic_add(&f(i, 2), fztmp);
    if (EVFLAG && eflag_either) {
      evdwl += d_e_embed(ii);
      if (eflag_global) ev.evdwl += evdwl;
      if (eflag_atom) d_eatom(i) += evdwl;
    }
    if (EVFLAG && vflag_global)
      for (int k = 0; k < 6; k++) ev.v[k] += vi[k];
  }
}

namespace LAMMPS_NS {
template class PairPEACEKokkos<LMPDeviceType>;
}
#endif
