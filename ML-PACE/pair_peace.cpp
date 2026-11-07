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

/* -------------------------------------------------------------------------
   Contributing author: Flynn Walsh
   CPU adaptation of pair_peace_kokkos
------------------------------------------------------------------------- */

#include "pair_peace.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair_peace_impl.h"
#include "platform.h"
#include "update.h"

#include "yaml-cpp/yaml.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace PEACE;


static const double CB_BETAS_20[] = {
    4.730040744862704,  7.853204624095826,  10.995607838001661, 14.137165491257454,
    17.278759657399469, 20.420352245626049, 23.561944902040445, 26.703537555508177,
    29.845130209103242, 32.986722862692808, 36.128315516282612, 39.269908169872402,
    42.411500823462198, 45.553093477051988, 48.694686130641784, 51.836278784231581,
    54.977871437821371, 58.119464091411167, 61.261056745000957, 64.402649398590754};

static void compute_clamped_basis(double r, double rcut, int nradbase, double *gr, double *dgr)
{
  const double u = r / rcut;

  for (int k = 0; k < nradbase; k++) {
    const double beta = k < 20 ? CB_BETAS_20[k] : (2.0 * k + 3.0) * MY_PI / 2.0;
    const double sigma = (cosh(beta) - cos(beta)) / (sinh(beta) - sin(beta));
    const double delta = (-exp(-beta) + cos(beta) - sin(beta)) / (sinh(beta) - sin(beta));
    const double bu = beta * u;

    const double y = exp(-bu) + delta * sinh(bu) - cos(bu) + sigma * sin(bu);
    const double dydu = beta * (-exp(-bu) + delta * cosh(bu) + sin(bu) + sigma * cos(bu));
    gr[k] = y;
    dgr[k] = dydu / rcut;
  }
}

static void sbessel_aux(double r, double rc, int n, double &g, double &dg)
{
  const double pref = ((n % 2 == 0) ? 1.0 : -1.0) * sqrt(2.0) * MY_PI / pow(rc, 1.5) * (n + 1) *
      (n + 2) / sqrt((double) ((n + 1) * (n + 1) + (n + 2) * (n + 2)));
  const double a1 = (n + 1) * MY_PI / rc, a2 = (n + 2) * MY_PI / rc;
  const double x1 = a1 * r, x2 = a2 * r;
  g = pref * (sin(x1) / x1 + sin(x2) / x2);
  dg = pref *
      (a1 * (x1 * cos(x1) - sin(x1)) / (x1 * x1) + a2 * (x2 * cos(x2) - sin(x2)) / (x2 * x2));
}

static void compute_sbessel_basis(double r, double rcut, int nradbase, double *gr, double *dgr)
{
  if (r <= 0.0) {
    for (int k = 0; k < nradbase; k++) {
      gr[k] = dgr[k] = 0.0;
    }
    return;
  }
  sbessel_aux(r, rcut, 0, gr[0], dgr[0]);
  double d_prev = 1.0;
  for (int n = 1; n < nradbase; n++) {
    const double en =
        (double) (n * n) * (double) ((n + 2) * (n + 2)) / (4.0 * pow(n + 1.0, 4) + 1.0);
    const double dn = 1.0 - en / d_prev;
    const double s = sqrt(en / d_prev);
    sbessel_aux(r, rcut, n, gr[n], dgr[n]);
    gr[n] = (gr[n] + s * gr[n - 1]) / sqrt(dn);
    dgr[n] = (dgr[n] + s * dgr[n - 1]) / sqrt(dn);
    d_prev = dn;
  }
}

static void zbl_e_de(double r, double ainv, double zze, double &e, double &de)
{
  const double rinv = 1.0 / r;
  double phi = 0, dphi = 0;
  const double c[4] = {0.02817, 0.28022, 0.50986, 0.18175};
  const double d[4] = {0.20162, 0.40290, 0.94229, 3.19980};
  for (int k = 0; k < 4; k++) {
    double ex = exp(-d[k] * ainv * r);
    phi += c[k] * ex;
    dphi -= c[k] * d[k] * ainv * ex;
  }
  e = zze * phi * rinv;
  de = zze * (dphi - phi * rinv) * rinv;
}

static const char *const PEACE_ELEMENTS[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

static int peace_atomic_number_by_name(const std::string &elemname)
{
  const int nelem = sizeof(PEACE_ELEMENTS) / sizeof(const char *);
  for (int i = 1; i < nelem; i++)
    if (elemname == PEACE_ELEMENTS[i]) return i;
  return -1;
}

PairPEACE::PairPEACE(LAMMPS *lmp) :
    Pair(lmp), mu_group(nullptr), ngroups(0), group_first(nullptr), lmax(0), nrad_A(0), ldep(false), nrad(0),
    ndensity(0), cutoff(0.0), fs_params(nullptr), E0(nullptr), dEdB(nullptr), max_npolys(0), scale(nullptr),
    chunksize(65536), print_kernel_timing(0), subtract_zbl(false), zbl_splines(nullptr), a_first(nullptr),
    a_ipoly(nullptr), al_first(nullptr), W(nullptr), nbins(0), rln_dr_inv(0.0), rln_splines(nullptr), z_ln(nullptr),
    nearest(nullptr), neigh_mu(nullptr), rnorms(nullptr), rhats(nullptr), Ai(nullptr), neigh_scratch(nullptr),
    Rij_cache(nullptr), A(nullptr), dEdA(nullptr), a(nullptr), dBda(nullptr), rho(nullptr), dEdrho(nullptr),
    ilist_sorted(nullptr), element_batches(nullptr), element_natoms(nullptr), maxneigh(0), alloc_inum(0), npolys(0),
    poly_order(nullptr), poly_term_first(nullptr), poly_a_first(nullptr), poly_ls(nullptr), poly_ms(nullptr),
    poly_cgs(nullptr), coeffs(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  centroidstressflag = CENTROID_AVAIL;
}

PairPEACE::~PairPEACE()
{
  if (copymode) return;

  destroy_arrays();

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairPEACE::destroy_arrays()
{
  memory->destroy(W);
  memory->destroy(fs_params);
  memory->destroy(E0);
  memory->destroy(al_first);
  memory->destroy(a_ipoly);
  memory->destroy(poly_a_first);
  memory->destroy(a_first);
  memory->destroy(poly_ls);
  memory->destroy(nearest);
  memory->destroy(neigh_mu);
  memory->destroy(rnorms);
  memory->destroy(rhats);
  memory->destroy(neigh_scratch);
  memory->destroy(Rij_cache);
  memory->destroy(Ai);
  memory->destroy(A);
  memory->destroy(a);
  memory->destroy(rho);
  memory->destroy(ilist_sorted);
  memory->destroy(element_batches);
  memory->destroy(element_natoms);
  memory->destroy(poly_order);
  memory->destroy(poly_term_first);
  memory->destroy(mu_group);
  memory->destroy(group_first);
  memory->destroy(poly_ms);
  memory->destroy(poly_cgs);
  memory->destroy(coeffs);
  memory->destroy(dEdB);
  memory->destroy(rln_splines);
  memory->destroy(z_ln);
  memory->destroy(zbl_splines);
}

int PairPEACE::ComputeNeigh(const int ii, const int i)
{
  double **x = atom->x;
  const int *type = atom->type;
  const int *jlist = list->firstneigh[i];
  const int jnum = list->numneigh[i];
  const double rcutsq = cutoff * cutoff;
  const double xtmp = x[i][0];
  const double ytmp = x[i][1];
  const double ztmp = x[i][2];

  int *inearest = nearest[ii];
  int *neigh_mui = neigh_mu[ii];
  double *rinorms = rnorms[ii];
  double **rhati = rhats[ii];

  int jcount = 0;
  for (int jj = 0; jj < jnum; jj++) {
    const int j = jlist[jj] & NEIGHMASK;
    const double delx = xtmp - x[j][0];
    const double dely = ytmp - x[j][1];
    const double delz = ztmp - x[j][2];
    const double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < rcutsq) {
      const double rij = sqrt(rsq);
      const double rinv = 1.0 / rij;
      inearest[jcount] = j;
      neigh_mui[jcount] = map[type[j]];
      rhati[0][jcount] = -delx * rinv;
      rhati[1][jcount] = -dely * rinv;
      rhati[2][jcount] = -delz * rinv;
      rinorms[jcount] = rij;
      jcount++;
    }
  }
  return jcount;
}

template <int LMAX> void PairPEACE::ComputeAi(const int ii, const int jcount)
{
  const int nrad_A_pad = (nrad_A + SIMD - 1) / SIMD * SIMD;
  const int *neigh_mui = neigh_mu[ii];
  const double *rinorms = rnorms[ii];
  const double *const *rhati = rhats[ii];
  double *Ri = Rij_cache[ii];
  double *Rji = neigh_scratch;
  double *Y = Rji + NEIGH_SIMD * nrad;
  const double dr_inv = rln_dr_inv; // local needed to prevent reloads here

  memset(Ai, 0, lm_size(LMAX) * nrad_A_pad * sizeof(double));
  for (int jj0 = 0; jj0 < jcount; jj0 += NEIGH_SIMD) {
    double *R = Ri + 2 * nrad * jj0;
    double *dR = R + NEIGH_SIMD * nrad;
    for (int jj = 0; jj < NEIGH_SIMD; jj++) {
      if (jj0 + jj >= jcount) {
        for (int ln = 0; ln < nrad; ln++) {
          Rji[jj * nrad + ln] = 0.0;
          R[ln * NEIGH_SIMD + jj] = 0.0;
          dR[ln * NEIGH_SIMD + jj] = 0.0;
        }
        continue;
      }
      double w;
      const int bin = spline_bin(rinorms[jj0 + jj], dr_inv, nbins, w);
      const double *coeff = &rln_splines[bin][0][0];
      const double *zj = &z_ln[neigh_mui[jj0 + jj]][0];
      for (int ln = 0; ln < nrad; ln++) {
        const double r = spline_value(coeff + ln, nrad, w) * zj[ln];
        Rji[jj * nrad + ln] = r;
        R[ln * NEIGH_SIMD + jj] = r;
        dR[ln * NEIGH_SIMD + jj] = spline_deriv(coeff + ln, nrad, w, dr_inv) * zj[ln];
      }
    }
    for (int jj = 0; jj < NEIGH_SIMD; jj++)
      compute_ylm<LMAX, false>(rhati[0][jj0 + jj], rhati[1][jj0 + jj], rhati[2][jj0 + jj], NEIGH_SIMD, Y + jj);
    // prefetch splines
#if defined(__GNUC__)
    for (int jj = NEIGH_SIMD; jj < 2 * NEIGH_SIMD && jj0 + jj < jcount; jj++) {
      double w;
      const char *row = (const char *) &rln_splines[spline_bin(rinorms[jj0 + jj], dr_inv, nbins, w)][0][0];
      for (int off = 0; off < 4 * nrad * (int) sizeof(double); off += 64) __builtin_prefetch(row + off, 0, 3);
    }
#endif
    // unrolls are all needed for ComputeAi to work as a function vs. inline
#pragma GCC unroll 6
    for (int l = 0; l <= LMAX; l++) {
      const double *Yl = Y + l * l * NEIGH_SIMD;
      for (int n0 = 0; n0 < nrad_A_pad; n0 += SIMD) {
        double *Ail = Ai + l * l * nrad_A_pad + n0;
        const double *Rl = Rji + (ldep ? l : 0) * nrad_A + n0;
        double accum[2 * LMAX + 1][SIMD];
#pragma GCC unroll 11
        for (int m = 0; m < 2 * l + 1; m++)
#pragma GCC unroll 8
          for (int n = 0; n < SIMD; n++)
            accum[m][n] = Ail[m * nrad_A_pad + n];
        for (int jj = 0; jj < NEIGH_SIMD; jj++)
#pragma GCC unroll 11
          for (int m = 0; m < 2 * l + 1; m++)
#pragma GCC unroll 8
            for (int n = 0; n < SIMD; n++)
              accum[m][n] += Yl[m * NEIGH_SIMD + jj] * Rl[jj * nrad + n];
#pragma GCC unroll 11
        for (int m = 0; m < 2 * l + 1; m++)
#pragma GCC unroll 8
          for (int n = 0; n < SIMD; n++)
            Ail[m * nrad_A_pad + n] = accum[m][n];
      }
    }
  }
  for (int lm = 0; lm < lm_size(LMAX); lm++)
    for (int n = 0; n < nrad_A; n++)
      A[(lm * nrad_A + n) * SIMD + ii] = Ai[lm * nrad_A_pad + n];
}

template <int LMAX>
void PairPEACE::ComputeForce(const int ii, const int i, const int jcount, const int mu, double evdwl)
{
  double **f = atom->f;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;
  const int *inearest = nearest[ii];
  const int *neigh_mui = neigh_mu[ii];
  const double *rinorms = rnorms[ii];
  const double *const *rhati = rhats[ii];
  const double *Ri = Rij_cache[ii];
  double *P = neigh_scratch;
  double *Q = neigh_scratch + lm_size(LMAX) * NEIGH_SIMD;

  double fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0;
  for (int jj0 = 0; jj0 < jcount; jj0 += NEIGH_SIMD) {
    const double *R = Ri + 2 * nrad * jj0;
    const double *dR = R + NEIGH_SIMD * nrad;

    for (int l = 0; l <= LMAX; l++) {
      const double *Rl = &R[(ldep ? l : 0) * nrad_A * NEIGH_SIMD];
      const double *dRl = &dR[(ldep ? l : 0) * nrad_A * NEIGH_SIMD];
      for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
        const double *dEdAlm = &dEdA[lm * nrad_A * SIMD + ii];
        double Plm[NEIGH_SIMD] = {}, Qlm[NEIGH_SIMD] = {};
        for (int n = 0; n < nrad_A; n++) {
          const double dEdAn = dEdAlm[n * SIMD];
          const double *Rn = Rl + n * NEIGH_SIMD;
          const double *dRn = dRl + n * NEIGH_SIMD;
          for (int jj = 0; jj < NEIGH_SIMD; jj++) {
            Plm[jj] += dEdAn * dRn[jj];
            Qlm[jj] += dEdAn * Rn[jj];
          }
        }
        for (int jj = 0; jj < NEIGH_SIMD; jj++)
          P[lm * NEIGH_SIMD + jj] = Plm[jj];
        for (int jj = 0; jj < NEIGH_SIMD; jj++)
          Q[lm * NEIGH_SIMD + jj] = Qlm[jj];
      }
    }
    double fij[3 * NEIGH_SIMD];
    for (int jj = 0; jj < NEIGH_SIMD; jj++)
      compute_ylm<LMAX, true>(rhati[0][jj0 + jj], rhati[1][jj0 + jj], rhati[2][jj0 + jj], NEIGH_SIMD, fij + jj, nullptr, P + jj, Q + jj, 1.0 / rinorms[jj0 + jj]);
    if (subtract_zbl) {
#pragma clang loop vectorize(enable)
      for (int jj = 0; jj < NEIGH_SIMD; jj++) {
        const double rij = rinorms[jj0 + jj];
        const int muj = neigh_mui[jj0 + jj];
        double ezbl;
        const double dEdr = zbl_pair(&zbl_splines[0][0][0], nelements, mu, muj, nbins, rij, rln_dr_inv, ezbl);
        fij[jj] += dEdr * rhati[0][jj0 + jj];
        fij[NEIGH_SIMD + jj] += dEdr * rhati[1][jj0 + jj];
        fij[2 * NEIGH_SIMD + jj] += dEdr * rhati[2][jj0 + jj];
        evdwl += jj0 + jj < jcount ? ezbl : 0.0;
      }
    }
    for (int jj = 0; jj < NEIGH_SIMD && jj0 + jj < jcount; jj++) {
      const int j = inearest[jj0 + jj];
      const double fx = fij[jj];
      const double fy = fij[NEIGH_SIMD + jj];
      const double fz = fij[2 * NEIGH_SIMD + jj];
      fxtmp += fx;
      fytmp += fy;
      fztmp += fz;
      f[j][0] -= fx;
      f[j][1] -= fy;
      f[j][2] -= fz;

      if (vflag_either) {
        const double delx = -rhati[0][jj0 + jj] * rinorms[jj0 + jj];
        const double dely = -rhati[1][jj0 + jj] * rinorms[jj0 + jj];
        const double delz = -rhati[2][jj0 + jj] * rinorms[jj0 + jj];
        ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fx, fy, fz, delx, dely, delz);

        if (cvflag_atom) {
          const double v[9] = {delx * fx, dely * fy, delz * fz, delx * fy, delx * fz, dely * fz, dely * fx, delz * fx, delz * fy};
          for (int k = 0; k < 9; k++) {
            cvatom[i][k] += 0.5 * v[k];
            cvatom[j][k] += 0.5 * v[k];
          }
        }
      }
    }
  }

  f[i][0] += fxtmp;
  f[i][1] += fytmp;
  f[i][2] += fztmp;
  if (eflag_global) eng_vdwl += evdwl;
  if (eflag_atom) eatom[i] += evdwl;
}

void PairPEACE::ComputeForwardMix()
{
  for (int lm = 0; lm < lm_size(lmax); lm++) {
    const int l = (int) sqrt((double) lm);
    const int m = lm - l * l;
    const int ia_lo = al_first[l];
    const int ia_hi = al_first[l + 1];
    for (int ia0 = ia_lo; ia0 < ia_hi; ia0 += MIX_TILE)
      forward_mix<MIX_TILE>(&a[(a_first[ia0] + m) * SIMD], (2 * l + 1) * SIMD, &A[lm * nrad_A * SIMD], SIMD, W[ia0], nrad_A, ia_hi - ia0);
  }
}

template <int NRAD> void PairPEACE::ComputeBackwardMix(const int igroup)
{
  for (int lm = 0; lm < lm_size(lmax); lm++) {
    const int l = (int) sqrt((double) lm);
    const int m = lm - l * l;
    const int ia_lo = al_first[l];
    const int ia_hi = al_first[l + 1];
    double *dEdAlm = &dEdA[lm * nrad_A * SIMD];
    for (int n0 = 0; n0 < nrad_A; n0 += NRAD) {
      double dEdA_tile[NRAD * SIMD] = {};
      backward_mix<NRAD>(dEdA_tile, &dBda[(a_first[ia_lo] + m) * SIMD], (2 * l + 1) * SIMD,
                         dEdB, SIMD, &a_ipoly[igroup][ia_lo],
                         &W[ia_lo][n0], nrad_A, ia_hi - ia_lo);
#pragma GCC unroll 32
      for (int n = 0; n < NRAD; n++)
        if (n0 + n < nrad_A)
          for (int ii = 0; ii < SIMD; ii++)
            dEdAlm[(n0 + n) * SIMD + ii] = dEdA_tile[n * SIMD + ii];
    }
  }
}

void PairPEACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int *type = atom->type;
  const int inum = list->inum;
  int *ilist = list->ilist;
  const int *numneigh = list->numneigh;

  int max_jnum = 0;
  for (int ii = 0; ii < inum; ii++)
    max_jnum = MAX(max_jnum, numneigh[ilist[ii]]);

  resize_work_arrays(max_jnum, inum);
  // group atoms so we can vectorize in weights phase
  memset(element_natoms, 0, nelements * sizeof(int));
  int nordered = 0;
  for (int ii = 0; ii < inum; ii++) {
    const int mu = map[type[ilist[ii]]];
    element_batches[mu][element_natoms[mu]++] = ilist[ii];
    if (element_natoms[mu] == SIMD) {
      for (int k = 0; k < SIMD; k++)
        ilist_sorted[nordered++] = element_batches[mu][k];
      element_natoms[mu] = 0;
    }
  }
  for (int mu = 0; mu < nelements; mu++)
    if (element_natoms[mu])
      for (int k = 0; k < SIMD; k++)
        ilist_sorted[nordered++] = k < element_natoms[mu] ? element_batches[mu][k] : -1;
  ilist = ilist_sorted;

  int jcounti[SIMD];
  double evdwl[SIMD];
  const int passes = (nrad_A + MIX_TILE - 1) / MIX_TILE;
  const int nrad_per_pass = ((nrad_A + passes - 1) / passes + 3) / 4 * 4;

  double times[9] = {};
  const double t_start = print_kernel_timing ? platform::walltime() : 0.0;
  double t_last = t_start;

  for (int ii0 = 0; ii0 < nordered; ii0 += SIMD) {
    const int mu = map[type[ilist[ii0]]];
    const int igroup = mu_group[mu];

    for (int ii = 0; ii < SIMD; ii++) {
      const int i = ilist[ii0 + ii];
      // <--------- Neigh Phase ---------->
      jcounti[ii] = i >= 0 ? ComputeNeigh(ii, i) : 0;
      stamp(print_kernel_timing, times, 1, t_last);

      // <--------- Ai Phase ---------->
      switch (lmax) {
        case 2: ComputeAi<2>(ii, jcounti[ii]); break;
        case 3: ComputeAi<3>(ii, jcounti[ii]); break;
        case 4: ComputeAi<4>(ii, jcounti[ii]); break;
        case 5: ComputeAi<5>(ii, jcounti[ii]); break;
      }
      stamp(print_kernel_timing, times, 2, t_last);
    }

    // <--------- Forward Mix Phase ---------->
    ComputeForwardMix();
    stamp(print_kernel_timing, times, 3, t_last);

    // <--------- Densities Phase ---------->
    memset(rho, 0, ndensity * SIMD * sizeof(double));
    for (int ipoly = group_first[igroup]; ipoly < group_first[igroup + 1]; ipoly++) {
      const int order = poly_order[ipoly];
      const int term_first = poly_term_first[ipoly];
      const int nterms = poly_term_first[ipoly + 1] - term_first;
      const int *first = poly_a_first[ipoly];
      const int *ls = poly_ls[ipoly];
      const double *cgs = &poly_cgs[term_first];
      const int *ms = poly_ms[term_first];
      double B[SIMD];
      switch (order) {
        case 1: evaluate_polynomial_1(a, SIMD, cgs, ms, B); break;
        case 2: evaluate_polynomial_2(a, SIMD, cgs, ms, nterms, B); break;
        case 3: evaluate_polynomial_3(a, SIMD, first, ls, cgs, ms, nterms, B); break;
        case 4: evaluate_polynomial_4(a, SIMD, first, ls, cgs, ms, nterms, B); break;
      }

      const int ipoly_local = ipoly - group_first[igroup];
      const double *icoeffs = coeffs[mu][ipoly_local];
#pragma clang loop vectorize(disable)
      for (int p = 0; p < ndensity; p++) {
        const double coeff = icoeffs[p];
        for (int ii = 0; ii < SIMD; ii++)
          rho[p * SIMD + ii] += B[ii] * coeff;
      }
    }
    stamp(print_kernel_timing, times, 4, t_last);

    // <---------- Embedding Phase ---------->
    for (int ii = 0; ii < SIMD; ii++)
      evdwl[ii] = E0[mu];
    fs_embedding(&fs_params[0][0], ndensity, rho, SIMD, evdwl);
    for (int ii = 0; ii < SIMD; ii++) {
      const int i = ilist[ii0 + ii];
      if (i >= 0) { // skip padded atoms
        const double iscale = scale[type[i]][type[i]];
        evdwl[ii] *= iscale;
        for (int p = 0; p < ndensity; p++)
          dEdrho[p * SIMD + ii] *= iscale;
      }
    }
    stamp(print_kernel_timing, times, 5, t_last);

    // <---------- Weights Phase ---------->
    for (int ipoly = 0; ipoly < group_first[igroup + 1] - group_first[igroup]; ipoly++) {
      const double *icoeffs = coeffs[mu][ipoly];
      double dEdBi[SIMD] = {};
      for (int p = 0; p < ndensity; p++)
        for (int ii = 0; ii < SIMD; ii++)
          dEdBi[ii] += dEdrho[p * SIMD + ii] * icoeffs[p];
      for (int ii = 0; ii < SIMD; ii++)
        dEdB[ipoly * SIMD + ii] = dEdBi[ii];
    }
    stamp(print_kernel_timing, times, 6, t_last);

    // <---------- Backward Mixing Phase ---------->
    switch (nrad_per_pass) {
      case 4: ComputeBackwardMix<4>(igroup); break;
      case 8: ComputeBackwardMix<8>(igroup); break;
      case 12: ComputeBackwardMix<12>(igroup); break;
      case 16: ComputeBackwardMix<16>(igroup); break;
    }
    stamp(print_kernel_timing, times, 7, t_last);

    // <---------- Force Phase ---------->
    for (int ii = 0; ii < SIMD; ii++) {
      const int i = ilist[ii0 + ii];
      if (i < 0) continue;
      switch (lmax) {
        case 2: ComputeForce<2>(ii, i, jcounti[ii], mu, evdwl[ii]); break;
        case 3: ComputeForce<3>(ii, i, jcounti[ii], mu, evdwl[ii]); break;
        case 4: ComputeForce<4>(ii, i, jcounti[ii], mu, evdwl[ii]); break;
        case 5: ComputeForce<5>(ii, i, jcounti[ii], mu, evdwl[ii]); break;
      }
      stamp(print_kernel_timing, times, 8, t_last);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  if (print_kernel_timing) {
    times[0] = platform::walltime() - t_start;
    print_kernel_times(times);
  }
}

void PairPEACE::print_kernel_times(const double *t) const
{
  if (comm->me != 0) return;
  utils::logmesg(lmp,
                 "[PEACE] kernel times(s) total={:.6g} neigh={:.6g} Ai={:.6g} fmix={:.6g} "
                 "densities={:.6g} embed={:.6g} weights={:.6g} bmix={:.6g} force={:.6g}\n",
                 t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
}

void PairPEACE::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(scale, n, n, "pair:scale");
  map = new int[n];
}

void PairPEACE::settings(int narg, char **arg)
{
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "PEACE potentials require 'metal' units");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "chunksize") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style peace chunksize", error);
      chunksize = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
    } else if (strcmp(arg[iarg], "timing") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style peace timing", error);
      print_kernel_timing = utils::logical(FLERR, arg[iarg + 1], false, lmp);
    } else
      error->all(FLERR, "Unknown pair_style peace keyword: {}", arg[iarg]);
    iarg += 2;
  }
}

void PairPEACE::coeff(int narg, char **arg)
{
  if (narg - 3 != atom->ntypes)
    error->all(FLERR, "Number of element to type mappings does not match number of atom types");

  if (!allocated) allocate();

  auto potential_file_name = utils::get_potential_file_path(arg[2]);

  destroy_arrays();
  maxneigh = 0;
  alloc_inum = 0;

  // read grace yaml
  YAML_PACE::Node root = YAML_PACE::LoadFile(potential_file_name);

  const std::vector<std::string> element_names = root["elements"].as<std::vector<std::string>>();
  nelements = (int) element_names.size();
  const double nnorm = root["nnorm"].as<double>(1.0);

  YAML_PACE::Node radial_basis = root["radial_basis"];
  const std::string radbasename = radial_basis["radbasename"].as<std::string>();
  if (radbasename != "ClampedBeam" && radbasename != "SBessel")
    error->all(FLERR, "PEACE does not support radbasename {}", radbasename);
  cutoff = radial_basis["cutoff"].as<double>();
  const std::vector<double> crad = radial_basis["crad"].as<std::vector<double>>();

  std::vector<int> crad_shape = radial_basis["crad_shape"].as<std::vector<int>>();
  nrad_A = crad_shape[0];
  if (nrad_A > MAX_NRAD)
    error->all(FLERR, "PEACE supports nmax <= {}, not {}", MAX_NRAD, nrad_A);
  ldep = (crad_shape[1] != 1);
  nrad = nrad_A * crad_shape[1];
  lmax = -1;
  const int nradbase = crad_shape[2];

  subtract_zbl = radial_basis["subtract_zbl"].as<bool>(false);
  const double deltaSplineBins = radial_basis["deltaSplineBins"].as<double>(0.001);

  // build chemistry
  const std::vector<double> Z = radial_basis["Z"].as<std::vector<double>>();
  const bool z_is_nl = radial_basis["Z_shape"].as<std::vector<int>>().size() == 3;
  const size_t z_row = (size_t) nelements * (z_is_nl ? crad_shape[1] : 1);
  const int z_nrad = (int) (Z.size() / z_row);
  memory->create(z_ln, nelements, nrad, "peace:z_ln");
  for (int mu = 0; mu < nelements; mu++)
    for (int n = 0; n < nrad_A; n++)
      for (int l = 0; l < crad_shape[1]; l++)
        z_ln[mu][l * nrad_A + n] =
            (z_is_nl ? Z[(mu * z_nrad + n) * crad_shape[1] + l] : Z[mu * z_nrad + n]) * nnorm;

  std::vector<double> emb_params = root["emb_spec"]["params"].as<std::vector<double>>();

  YAML_PACE::Node functions = root["functions"];

  // count polynomials
  max_npolys = 0;
  int total_polys = 0;
  int total_terms = 0;
  ndensity = (int) functions[0][0]["coeff"].size();
  for (int mu = 0; mu < nelements; mu++) {
    for (auto poly_node : functions[mu])
      total_terms += (int) poly_node["gen_cgs"].size();
    max_npolys = MAX(max_npolys, (int) functions[mu].size());
    total_polys += (int) functions[mu].size();
  }

  // create initial arrays for all polynomials (possibly redundant)
  memory->create(group_first, nelements + 1, "peace:group_first");
  memory->create(poly_order, total_polys, "peace:poly_order");
  memory->create(poly_term_first, total_polys + 1, "peace:poly_term_first");
  int **poly_ias;
  memory->create(poly_ias, total_polys, MAX_ORDER, "peace:poly_ias");
  memory->create(poly_ls, total_polys, MAX_ORDER, "peace:poly_ls");
  memory->create(poly_ms, total_terms, MAX_ORDER, "peace:poly_ms");
  memory->create(poly_cgs, total_terms, "peace:poly_cgs");
  memory->create(coeffs, nelements, max_npolys, ndensity, "peace:coeffs");

  // lambda to determine if groups of polynomials (elements) are identical
  auto same_basis = [&](int group1, int group2) {
    // 1 check number of polynomials in group
    const int first1 = group_first[group1], first2 = group_first[group2];
    const int count = group_first[group1 + 1] - first1;
    if (count != group_first[group2 + 1] - first2)
      return false;
    // 2 check number of terms
    const int term1 = poly_term_first[first1], term2 = poly_term_first[first2];
    const int nterms = poly_term_first[first1 + count] - term1;
    if (nterms != poly_term_first[first2 + count] - term2)
      return false;
    // 3 compare all entries
    return !memcmp(poly_order + first1, poly_order + first2, count * sizeof(int)) &&
        !memcmp(poly_ias[first1], poly_ias[first2], count * MAX_ORDER * sizeof(int)) &&
        !memcmp(poly_ls[first1], poly_ls[first2], count * MAX_ORDER * sizeof(int)) &&
        !memcmp(poly_ms[term1], poly_ms[term2], nterms * MAX_ORDER * sizeof(int)) &&
        !memcmp(poly_cgs + term1, poly_cgs + term2, nterms * sizeof(double));
  };

  // read polynomials and create mu (element) groups
  memory->create(mu_group, nelements, "peace:mu_group");
  ngroups = npolys = 0;
  poly_term_first[0] = 0;
  group_first[0] = 0;
  for (int mu = 0; mu < nelements; mu++) {
    memset(coeffs[mu][0], 0, max_npolys * ndensity * sizeof(double));
    int ipoly = npolys;
    int iterm = poly_term_first[npolys];
    for (auto poly_node : functions[mu]) {
      std::vector<int> ls = poly_node["ls"].as<std::vector<int>>();
      std::vector<int> ns = poly_node["ns"].IsSequence()
          ? poly_node["ns"].as<std::vector<int>>()
          : std::vector<int>(ls.size(), poly_node["ns"].as<int>());
      std::vector<int> ms = poly_node["ms_combs"].as<std::vector<int>>();
      std::vector<double> cgs = poly_node["gen_cgs"].as<std::vector<double>>();
      std::vector<double> coeff = poly_node["coeff"].as<std::vector<double>>();
      int order = poly_node["rank"].as<int>((int) ls.size());

      poly_order[ipoly] = order;
      memset(poly_ias[ipoly], 0, MAX_ORDER * sizeof(int));
      memset(poly_ls[ipoly], 0, MAX_ORDER * sizeof(int));
      for (int t = 0; t < order; t++) {
        poly_ls[ipoly][t] = ls[t];
        poly_ias[ipoly][t] = ns[t] - 1;
        lmax = MAX(lmax, ls[t]);
      }
      // sort order 3/4 ms by a so we can use once and overwrite
      for (int m1 = -MAX_L; m1 <= MAX_L; m1++)
        for (int m2 = -MAX_L; m2 <= MAX_L; m2++)
          for (int term = 0; term < (int) cgs.size(); term++) {
            const int *m = &ms[term * order];
            if (order == 4   ? m[0] != m1 || m[1] != m2
                : order == 3 ? m[2] != m1 || m2 != 0
                             : m1 != 0 || m2 != 0)
              continue;
            memset(poly_ms[iterm], 0, MAX_ORDER * sizeof(int));
            for (int t = 0; t < order; t++)
              poly_ms[iterm][t] = m[t] + ls[t];
            poly_cgs[iterm++] = cgs[term];
          }
      memcpy(coeffs[mu][ipoly - npolys], coeff.data(), ndensity * sizeof(double));
      poly_term_first[++ipoly] = iterm;
    }
    group_first[ngroups + 1] = ipoly;
    mu_group[mu] = 0;
    while (mu_group[mu] < ngroups && !same_basis(mu_group[mu], ngroups))
      mu_group[mu]++;
    if (mu_group[mu] == ngroups) {
      ngroups++;
      npolys = ipoly;
    }
  }
  // resize to actual unique sizes
  memory->grow(group_first, ngroups + 1, "peace:group_first");
  memory->grow(poly_order, npolys, "peace:poly_order");
  memory->grow(poly_term_first, npolys + 1, "peace:poly_term_first");
  memory->grow(poly_ls, npolys, MAX_ORDER, "peace:poly_ls");
  memory->grow(poly_ms, poly_term_first[npolys], MAX_ORDER, "peace:poly_ms");
  memory->grow(poly_cgs, poly_term_first[npolys], "peace:poly_cgs");
  if (lmax < 2 || lmax > MAX_L)
    error->all(FLERR, "PEACE supports lmax 2 to {}, model has {}", MAX_L, lmax);

  // map a to W
  memory->create(al_first, lmax + 2, "peace:al_first");
  al_first[0] = 0;
  std::vector<double> wrad;
  if (radial_basis["wrad"]) {
    std::vector<int> wrad_shape = radial_basis["wrad_shape"].as<std::vector<int>>();
    wrad = radial_basis["wrad"].as<std::vector<double>>();
    const std::vector<int> wrad_lcount = radial_basis["wrad_lcount"].as<std::vector<int>>();
    for (int l = 0; l <= lmax; l++)
      al_first[l + 1] = al_first[l] + wrad_lcount[l];
    if (wrad_shape[1] != nrad_A)
      error->all(FLERR, "PEACE wrad mixes {} radials but crad has {}", wrad_shape[1], nrad_A);
    if (wrad.size() != (size_t) al_first[lmax + 1] * nrad_A)
      error->all(FLERR, "PEACE wrad has {} entries but wrad_lcount needs {}", wrad.size(), al_first[lmax + 1] * nrad_A);
  }
  // make dummy mixing matrix for GRACE FS models
  else {
    std::vector<int> a_n;
    for (int l = 0; l <= lmax; l++) {
      for (int ipoly = 0; ipoly < npolys; ipoly++)
        for (int t = 0; t < poly_order[ipoly]; t++)
          if (poly_ls[ipoly][t] == l) {
            a_n.push_back(poly_ias[ipoly][t]);
            poly_ias[ipoly][t] = a_n.size() - 1;
          }
      al_first[l + 1] = a_n.size();
    }
    wrad.assign(a_n.size() * nrad_A, 0.0);
    for (int ia = 0; ia < (int) a_n.size(); ia++)
      wrad[ia * nrad_A + a_n[ia]] = 1.0;
  }
  // pad W
  memory->create(W, al_first[lmax + 1] + MIX_TILE - 1, nrad_A, "peace:W");
  memcpy(W[0], wrad.data(), wrad.size() * sizeof(double));
  memset(W[al_first[lmax + 1]], 0, (MIX_TILE - 1) * nrad_A * sizeof(double));

  // map each a entry to its polynomial
  memory->create(a_ipoly, ngroups, al_first[lmax + 1], "peace:a_ipoly");
  for (int group = 0; group < ngroups; group++)
    for (int ia = 0; ia < al_first[lmax + 1]; ia++)
      a_ipoly[group][ia] = max_npolys;
  for (int group = 0; group < ngroups; group++)
    for (int ipoly = group_first[group]; ipoly < group_first[group + 1]; ipoly++)
      for (int t = 0; t < poly_order[ipoly]; t++) {
        const int ia = poly_ias[ipoly][t];
        a_ipoly[group][ia] = ipoly - group_first[group];
      }

  // record the start of each a entry
  // a is flat due to different entry lengths (2l+1)
  memory->create(a_first, al_first[lmax + 1] + 1, "peace:a_first");
  a_first[0] = 0;
  for (int l = 0; l <= lmax; l++)
    for (int ia = al_first[l]; ia < al_first[l + 1]; ia++)
      a_first[ia + 1] = a_first[ia] + 2 * l + 1;
  memory->create(poly_a_first, npolys, MAX_ORDER, "peace:poly_a_first");
  for (int ipoly = 0; ipoly < npolys; ipoly++)
    for (int t = 0; t < MAX_ORDER; t++) {
      poly_a_first[ipoly][t] = a_first[poly_ias[ipoly][t]];
      for (int term = poly_term_first[ipoly]; term < poly_term_first[ipoly + 1]; term++)
        poly_ms[term][t] += poly_a_first[ipoly][t];
    }
  memory->destroy(poly_ias);

  memory->create(E0, nelements, "peace:E0");
  // fold any global shift into E0
  const double shift = root["shift"].as<double>(0.0);
  for (int mu = 0; mu < nelements; mu++)
    E0[mu] = (root["E0"] ? root["E0"][mu].as<double>() : 0.0) + shift;
  const double global_scale = root["scale"].as<double>(1.0);
  memory->create(fs_params, ndensity, 4, "peace:fs_params");
  for (int p = 0; p < ndensity; p++) {
    const double mexp = emb_params[2 * p + 1];
    fs_params[p][0] = mexp;
    fs_params[p][1] = 1.0 / mexp;
    fs_params[p][2] = mexp == 1.0 ? exp(-1.0) : exp(-log1p(mexp - 1.0) / (mexp - 1.0));
    fs_params[p][3] = emb_params[2 * p] * global_scale;
  }

  const bool clamped = (radbasename == "ClampedBeam");

  nbins = (int) std::lround(cutoff / deltaSplineBins);
  rln_dr_inv = (double) nbins / cutoff;

  memory->create(rln_splines, nbins, 4, nrad, "peace:rln_splines");

  std::vector<double> gk(nradbase);
  std::vector<double> dgk(nradbase);

  // splines part 1: evaluate radial functions at bin edges
  std::vector<double> values((nbins + 1) * nrad);
  std::vector<double> derivatives((nbins + 1) * nrad);
  for (int bin = 0; bin <= nbins; bin++) {
    const double r = bin / rln_dr_inv;
    if (clamped)
      compute_clamped_basis(r, cutoff, nradbase, gk.data(), dgk.data());
    else
      compute_sbessel_basis(r, cutoff, nradbase, gk.data(), dgk.data());
    for (int n = 0; n < nrad_A; n++) {
      for (int l = 0; l < crad_shape[1]; l++) {
        double R = 0.0;
        double dR = 0.0;
        const int base = (n * crad_shape[1] + l) * nradbase;
        for (int kb = 0; kb < nradbase; kb++) {
          const double c = crad[base + kb];
          R += c * gk[kb];
          dR += c * dgk[kb];
        }
        values[bin * nrad + l * nrad_A + n] = R;
        derivatives[bin * nrad + l * nrad_A + n] = dR;
      }
    }
  }
  // part 2 compute bin cubic coefficients
  for (int bin = 0; bin < nbins; bin++)
    for (int ln = 0; ln < nrad; ln++) {
      const double v0 = values[bin * nrad + ln];
      const double v1 = values[(bin + 1) * nrad + ln];
      const double d0 = derivatives[bin * nrad + ln] / rln_dr_inv;
      const double d1 = derivatives[(bin + 1) * nrad + ln] / rln_dr_inv;
      rln_splines[bin][0][ln] = v0;
      rln_splines[bin][1][ln] = d0;
      rln_splines[bin][2][ln] = 3.0 * (v1 - v0) - 2.0 * d0 - d1;
      rln_splines[bin][3][ln] = 2.0 * (v0 - v1) + d0 + d1;
    }
  // same splines but for zbl
  if (subtract_zbl) {
    const int npair = nelements * (nelements + 1) / 2;
    const double r_first = 1.0 / rln_dr_inv;
    std::vector<double> e(nbins + 1);
    std::vector<double> de(nbins + 1);
    memory->create(zbl_splines, nbins, npair, 4, "peace:zbl_splines");
    for (int mu = 0; mu < nelements; mu++)
      if (peace_atomic_number_by_name(element_names[mu]) < 0)
        error->all(FLERR, "PEACE ZBL does not know element {}", element_names[mu]);
    for (int mui = 0; mui < nelements; mui++)
      for (int muj = 0; muj <= mui; muj++) {
        const int ipair = mui * (mui + 1) / 2 + muj;
        const double zi = peace_atomic_number_by_name(element_names[mui]);
        const double zj = peace_atomic_number_by_name(element_names[muj]);
        const double ainv = (pow(zi, 0.23) + pow(zj, 0.23)) / 0.46850;
        const double zze = zi * zj * force->qqr2e;
        double E_c, dE_c;
        zbl_e_de(cutoff, ainv, zze, E_c, dE_c);
        for (int bin = 0; bin <= nbins; bin++) {
          double r = bin / rln_dr_inv;
          if (r < r_first) r = r_first;
          double E, dE;
          zbl_e_de(r, ainv, zze, E, dE);
          e[bin] = 0.5 * (E - E_c - (r - cutoff) * dE_c);
          de[bin] = 0.5 * (dE - dE_c);
        }
        for (int bin = 0; bin < nbins; bin++) {
          const double d0 = de[bin] / rln_dr_inv;
          const double d1 = de[bin + 1] / rln_dr_inv;
          zbl_splines[bin][ipair][0] = e[bin];
          zbl_splines[bin][ipair][1] = d0;
          zbl_splines[bin][ipair][2] = 3.0 * (e[bin + 1] - e[bin]) - 2.0 * d0 - d1;
          zbl_splines[bin][ipair][3] = 2.0 * (e[bin] - e[bin + 1]) + d0 + d1;
        }
      }
  }

  const int nrad_A_pad = (nrad_A + SIMD - 1) / SIMD * SIMD;
  memory->create(Ai, lm_size(lmax) * nrad_A_pad, "peace:Ai");
  memory->create(A, SIMD * lm_size(lmax) * nrad_A, "peace:A");
  memory->create(a, SIMD * a_first[al_first[lmax + 1]], "peace:a");
  memory->create(rho, SIMD * ndensity, "peace:rho");
  dEdA = A;
  dBda = a;
  dEdrho = rho;
  const int neigh_scratch_size = MAX(nrad + lm_size(lmax), 2 * lm_size(lmax)) * NEIGH_SIMD;
  memory->create(neigh_scratch, neigh_scratch_size, "peace:neigh_scratch");
  memory->create(dEdB, (max_npolys + 1) * SIMD, "peace:dEdB");
  memory->create(element_batches, nelements, SIMD, "peace:element_batches");
  memory->create(element_natoms, nelements, "peace:element_natoms");
  memset(dEdB + max_npolys * SIMD, 0, SIMD * sizeof(double));

  // standard type/element map
  const int n = atom->ntypes;
  for (int itype = 1; itype <= n; itype++) {
    char *elemname = arg[2 + itype];
    if (strcmp(elemname, "NULL") == 0) {
      map[itype] = -1;
      continue;
    }

    int mu = 0;
    while (mu < nelements && element_names[mu] != elemname) mu++;
    if (mu == nelements)
      error->all(FLERR, "PEACE model {} has no element {}", potential_file_name, elemname);

    map[itype] = mu;
  }

  for (int itype = 1; itype <= n; itype++)
    for (int jtype = itype; jtype <= n; jtype++) {
      scale[itype][jtype] = 1.0;
      setflag[itype][jtype] = map[itype] >= 0 && map[jtype] >= 0;
    }
}

void PairPEACE::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style peace requires newton pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairPEACE::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  scale[j][i] = scale[i][j];
  return cutoff;
}

void *PairPEACE::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "scale") == 0) return (void *) scale;
  return nullptr;
}

void PairPEACE::resize_work_arrays(int max_jnum, int inum)
{
  max_jnum = (max_jnum + NEIGH_SIMD - 1) / NEIGH_SIMD * NEIGH_SIMD;

  if (maxneigh < max_jnum) {
    memory->grow(nearest, SIMD, max_jnum, "peace:nearest");
    memory->grow(neigh_mu, SIMD, max_jnum, "peace:neigh_mu");
    memory->grow(rnorms, SIMD, max_jnum, "peace:rnorms");
    memory->grow(rhats, SIMD, 3, max_jnum, "peace:rhats");
    memory->grow(Rij_cache, SIMD, 2 * nrad * max_jnum, "peace:Rij_cache");
    maxneigh = max_jnum;
    for (int ii = 0; ii < SIMD; ii++)
      for (int jj = 0; jj < maxneigh; jj++) {
        neigh_mu[ii][jj] = 0;
        rnorms[ii][jj] = 1.0;
        rhats[ii][0][jj] = rhats[ii][1][jj] = rhats[ii][2][jj] = 0.0;
      }
  }
  inum += nelements * (SIMD - 1);
  if (alloc_inum < inum) {
    memory->grow(ilist_sorted, inum, "peace:ilist_sorted");
    alloc_inum = inum;
  }
}
