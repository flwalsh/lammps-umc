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

#ifdef FIX_CLASS
// clang-format off
FixStyle(hgcmc,FixHGCMC);
// clang-format on
#else

#ifndef LMP_FIX_HGCMC_H
#define LMP_FIX_HGCMC_H

#include "fix.h"
#include "atom.h"

namespace LAMMPS_NS {

class FixHGCMC : public Fix {
 public:
  FixHGCMC(class LAMMPS *, int, char **);
  ~FixHGCMC() override;
  
  int setmask() override;
  void init() override;
  void setup(int) override; 
  void initial_integrate(int) override; 
  void final_integrate() override; 
  void post_integrate() override; 
  void end_of_step() override; 
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  //double compute_array(int,int) override;
  double memory_usage() override;
  //void write_restart(FILE *) override;
  //void restart(char *) override;

 private:

  double dtv, dtf; 

  int nlocal_stored;
  int nmax_stored;
  int *type_stored, *mask_stored;
  tagint *tag_stored;
  imageint *image_stored;
  double *x_stored;
  //double *mass_stored;

  void reneighbor();
  void recompute_peatom();
  void recompute_peatom_swap();
  void store_atom();
  void restore_atom();
  void activate_vacancies();
  void deactivate_vacancies();
  //void store_atom2d(double **&, double *&, int);
  //void restore2d(double **&, double *&, int, int);
  //void delete_stored();

  int iswap, otype, ftype; //, idswap;
  int fswaptype;

  char *excludestring;
  char *groupname, *fictgroupname, *caplogroupname, *caphigroupname;
  int iincludegroup, ifictgroup, icaplogroup, icaphigroup;
  int fictgroupbit, movebit;
  int *nreal_diff; // for each swaptype
  double capmass, fictmass;

  //int nevery, seed;
  int seed;
  int nswap;          // # of all swap atoms on all procs; necessary for picking
  int nswapreal;          // # of real swap atoms on all procs
  
  //char *swap_region_id;
  //class Region *swap_region;

  // Referencing
  char *ref_region_id;
  class Region *ref_region;
  int *nrefs_local;
  int *nrefs;
  void update_ref_count();
  double gce_stored; // grand canonical energy
  double gbarea;
  double gbe_stored; // grain boundary energy

  int nmd_trials, nmd_accepts, ninsertion_trials, ninsertion_accepts, ndeletion_trials, ndeletion_accepts, niso_trials, niso_accepts, nxycap_trials, nxycap_accepts, nzcap_trials, nzcap_accepts;

  bigint laststep; // last step of MC trial

  int nrealtypes;
  int nswaptypes;
  int *swaptypes;
  int swaptype;
  bigint *typecount;
  double *mu, *lambda3;

  void print_scale();
  int nmdsteps, ngcmcsteps, nsgcmc;
  double pmd, pgcmc, psgcmc, piso, pxycap, pzcap;
  double isomax, xycapmax, zcapmax;
  double press;

  //double boxlo[3], h_inv[6];
  
  int i_MC_V; // variable index for volume
  int cap, volume;
  double zcaplo, zcaphi;

  bool unequal_cutoffs;

  //int revtestid;
  int revtest;
  int realpot;
  int atom_swap_nmax;
  double beta, beta_insert, beta_delete, beta_md;
  double *qtype;
  double pereal_stored, pecaphi_stored, pecaplo_stored;
  double ke_stored, kereal_stored, kefict_stored, kecap_stored;
  double **sqrt_mass_ratio;
  /* this is a bit inelegant  but I believe is desirable for canonical swaps
  as if we select among ALL swappable atoms by ID we can frequently end up with
  the same type and we can't know this without all-to-all communications */
  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_pe;
  class Compute *c_peatom;
  class Compute *c_pereal;
  class Compute *c_pecaphi;
  class Compute *c_pecaplo;
  
  class Compute *c_ke;
  class Compute *c_keatom;
  class Compute *c_kereal;
  class Compute *c_kefict;
  class Compute *c_kecap;

  class Compute *c_types;
  class Compute *c_patom;
  class Compute *c_preal;

  void options(int, char **);

  char *mcprintid;
  class FixPrint *mcprint;
  class PairHybridScaled *hybrid;

  char *mcdumpid;
  int imcdump;
  int onlydumpgc;

  double trial_forward;
  double trial_reverse;
  int idswap;
  int gcmc; // -1, 0, 1 for number of particles
  double dfrac;
  void bias_forward();
  void bias_reverse();
  double *plocal;

  //void initiate_cmc();
  void initiate_gcmc();
  void initiate_velocities();
  void store_pe();
  void store_ke();

  int oldtype, newtype;
  void trial_sgcmc();
  int nsgc_trials, nsgc_accepts;
  
  void trial_iso();
  void trial_xycap();
  void trial_zcap();
  
  void evaluate_move();
};

}    // namespace LAMMPS_NS

#endif
#endif
