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
FixStyle(umc,FixUMC);
// clang-format on
#else

#ifndef LMP_FIX_UMC_H
#define LMP_FIX_UMC_H

#include "fix.h"
#include "atom.h"

namespace LAMMPS_NS {

class FixUMC : public Fix {
 public:
  FixUMC(class LAMMPS *, int, char **);
  ~FixUMC() override;
  
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_run() override;
  void initial_integrate(int) override; 
  void final_integrate() override; 
  void post_integrate() override; 
  void end_of_step() override; 
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  double compute_array(int, int) override;
  double memory_usage() override;

 private:

  double dtv, dtf; 

  int nlocal_stored;
  int nmax_stored;
  int *type_stored, *mask_stored;
  tagint *tag_stored;
  imageint *image_stored;
  double *x_stored;

  void reneighbor();
  void recompute_peatom(int);
  void dump_pefict();
  void store_atom();
  void restore_atom();

  char *groupname, *fictgroupname, *caplogroupname, *caphigroupname;
  char *pair_style_name;
  int lambda_var;
  int ifictgroup, icaplogroup, icaphigroup;
  int fictbit, movebit;
  
  int nmd_trials, nmd_accepts, ninsertion_trials, ninsertion_accepts, ndeletion_trials, ndeletion_accepts, niso_trials, niso_accepts, naniso_trials, naniso_accepts, nxycap_trials, nxycap_accepts, nzcap_trials, nzcap_accepts;

  bigint laststep; // last step of MC trial

  int nrealtypes;
  int nswaptypes;
  int *swaptypes;
  int **typegrid;   // typegrid[s1][s2] = the LAMMPS type encoding species pair (s1->s2), 0 = absent
  int bridgetype;
  bigint *typecount;
  double *mu, *lambda3;
  double *mu0;        // the mu list as parsed -- the ramp anchor, captured once in options()
  double *muend;      // per-species ramp endpoint, NAN = no ramp for that species
  double mustep;      // fraction of mu0 -> muend traversed per trial
  int ntrialsmax;     // `trials` keyword: stop the run here, 0 = run to the step budget
  bigint ntrialsdone; // trials this run; ntrialsdone*mustep is the fraction reached
  void update_mu();   // latch mu[] for the trial being armed; once per trial

  int tc, tgc, tsgc;
  bigint tmax;      // longest a newly armed move of any type can run; set at the end of options()
  double pc, pgc, psgc, phop, piso, paniso, pxycap, pzcap;
  double isomax, anisomax, xycapmax, zcapmax;
  double press;

  int i_MC_V; // variable index for volume
  int cap;
  double zcaplo, zcaphi;

  int revtest, reverse;
  int atom_swap_nmax;
  double beta, beta_insert, beta_delete, beta_md, beta_sg;

  // force-biased velocities; tvbias = 0 disables
  double tvbias, rvbias, tbiascap, bias_logfwd;
  double pereal_stored;
  double virial_stored;    // trace of the real-group virial, captured when it is tallied
  double virialz_stored;   // its zz component
  double kereal_stored;
  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_peatom;
  class Compute *c_pereal;
  class Compute *c_keatom;
  class Compute *c_kereal;

  class Compute *c_patom;
  class Compute *c_preal;

  void options(int, char **);
  void print();

  int print_nevery;
  bigint print_count;
  FILE *print_fp;
  FILE *revtest_fp;
  char *print_text, *print_copy, *print_work;
  int print_maxcopy, print_maxwork;

  char *mcdumpid;
  int imcdump;
  int onlydumpgc;
  int npefict_dumps;

  double trial_forward;
  double trial_reverse;
  double zswap;
  int idswap;
  int trial; // -1, 0, 1, 2 for delete, md, insert, gradual sgc
  void bias_forward();
  void bias_reverse();
  double *plocal;

  void initiate_deletion();
  void initiate_insertion();
  void initiate_sgcmc();
  void set_hybrid_lambda(double new_lambda);
  void initiate_velocities();
  double velocity_bias(int draw);
  void store_pe();
  void store_ke();

  int oldtype, newtype;
  void trial_sgcmc();
  int nsgc_trials, nsgc_accepts;
  void trial_box(int);
  void trial_xycap();
  void trial_zcap();
};

}    // namespace LAMMPS_NS

#endif
#endif
