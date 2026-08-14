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
   Contributing authors: Flynn Walsh
------------------------------------------------------------------------- */

#include "fix_umc.h"
#include "atom.h"
#include "input.h" // for variable
#include "integrate.h" // for ev_set
#include "variable.h" // for variable
#include "output.h" // for write_dump
#include "dump.h" // for id
#include "atom_vec.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_hybrid_scaled.h"
#include "random_park.h"
#include "timer.h"
#include "update.h"

#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>

#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixUMC::FixUMC(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg),
    type_stored(nullptr), mask_stored(nullptr), tag_stored(nullptr), image_stored(nullptr), x_stored(nullptr),
    groupname(nullptr), fictgroupname(nullptr), caplogroupname(nullptr), caphigroupname(nullptr),
    pair_style_name(nullptr), lambda_var(-1),
    swaptypes(nullptr), typegrid(nullptr), typecount(nullptr), mu(nullptr), lambda3(nullptr),
    random_equal(nullptr), random_unequal(nullptr),
    c_peatom(nullptr), c_pereal(nullptr), c_keatom(nullptr),
    print_nevery(0), print_count(0), print_fp(nullptr), revtest_fp(nullptr), print_text(nullptr), print_copy(nullptr),
    print_work(nullptr), print_maxcopy(0), print_maxwork(0), mcdumpid(nullptr),
    plocal(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal fix UMC command nargs");

  dynamic_group_allow = 1;
  time_integrate = 1;

  vector_flag = 1;
  size_vector = 8; // for MC statistics outputs
  global_freq = 1;
  extvector = 0;
 
  nrealtypes = -1;
  tgc = -1;
  tsgc = -1;
  ntrialsmax = 0;   // 0 = run to the step budget, ramp on the expected trial count
  zswap = 0.0;
  bridgetype = -1;
  memory->create(mu, atom->ntypes+1, "hmc:mu"); // +1 b/c 1-indexed
  memory->create(mu0, atom->ntypes+1, "hmc:mu0");
  memory->create(muend, atom->ntypes+1, "hmc:muend");
  for (int itype = 0; itype <= atom->ntypes; itype++) muend[itype] = NAN;
  memory->create(swaptypes, atom->ntypes, "hmc:swaptypes"); // 0-indexed

  fictbit = 0;

  // force-biased velocities; tvbias = 0 is a no-op.
  tvbias = 0.0;
  rvbias = 3.0;
  tbiascap = 0.0;
  bias_logfwd = 0.0;

  groupname = utils::strdup(arg[1]); // fix group (igroup, groupbit) = real atoms
  // create *move* group, i.e. particles with velocities
  group->assign(fmt::format("MC_move union {}",groupname));
  int imovegroup = group->find("MC_move");
  movebit = group->bitmask[imovegroup];

  double temperature = utils::numeric(FLERR, arg[3], false, lmp);
  int seed = utils::inumeric(FLERR, arg[4], false, lmp);
  tc = utils::inumeric(FLERR, arg[5], false, lmp);

  if (seed <= 0) error->all(FLERR, "Illegal fix UMC seed");
  if (temperature <= 0.0) error->all(FLERR, "Illegal fix UMC temperature");

  beta = 1.0 / (force->boltz * temperature);
  beta_insert = beta;
  beta_delete = beta;
  beta_md = beta;
  beta_sg = 0;

  options(narg - 6, &arg[6]);

  // Declare which box parameters the MC moves can change.  Domain::init() ORs this over every fix
  // into domain->box_change, which gates the reset_box/comm->setup/setup_bins block in Verlet::run
  // AND in reneighbor() below.  Without it those never fire, so the neighbor binning keeps the
  // ORIGINAL box's bbox forever: NBin::coord2bin does NOT clamp outside that bbox, and binhead is
  // only mbins long, so once accepted box moves push atoms past subhi+cutghost, bin_atoms() writes
  // past the end of binhead -- heap corruption that surfaces as a segfault much later in the run.
  // comm->setup() (cutghost, CommBrick maxneed) is stale for the same reason.
  if (piso > 0. || paniso > 0.) box_change |= Fix::BOX_CHANGE_SIZE;   // iso/aniso resize x,y,z
  if (pzcap > 0.) box_change |= Fix::BOX_CHANGE_Z;                    // zcap stretches z only

  // global array: one row per real species, col 1 = N, col 2 = mu.  Backs the MC_N<i>/MC_mu<i>
  // variables (setup()), which read it live, so N follows typecount and mu follows update_mu()
  // with no update path of its own.  Set here rather than with the flags above because
  // nrealtypes is only final once options() has parsed the mu list.  extarray is one flag for
  // the whole array, so it stays 0 (N is really extensive) -- it only affects thermo's replica
  // handling, and nothing here uses that.
  // Cols 3-6 are the stored PE, virial and real-atom count that back MC_U/MC_P/MC_Pz.  They are
  // scalars, so they repeat down the rows -- inelegant, but the global vector is MC_A and
  // appending to it would change what ${MC_A} prints.  Reading them here rather than invoking
  // c_pereal/c_preal is what lets a trial be logged mid-burst: an instant move tallies energy
  // but not the per-atom virial, so evaluating compute stress/atom there fails outright.
  array_flag = 1;
  size_array_rows = nrealtypes;
  size_array_cols = 6;
  extarray = 0;

  // Validate everything that does not need the computes BEFORE creating them.  Modify::add_fix
  // deletes the old fix and nulls its slot before running this constructor, so a throw from here
  // leaves modify->fix[] holding a null inside [0,nfix) and the next command to walk the fix list
  // dereferences it.  Failing before add_compute() at least leaves the compute IDs free, so the
  // corrected command can be re-issued instead of dying on "Reuse of compute ID keatom".
  if (tgc > 0 || tsgc > 0) {
    PairHybridScaled *hybrid = dynamic_cast<PairHybridScaled *>(force->pair);
    if (!hybrid) error->all(FLERR,"pair_style hybrid/scaled required for gradual gc/sgc");
    lambda_var = input->variable->find("lambda");
    if (lambda_var < 0)
      error->all(FLERR, "Variable lambda must exist for gradual gc/sgc in fix UMC");
    if (!input->variable->internalstyle(lambda_var))
      error->all(FLERR, "Variable lambda must be internal-style for gradual gc/sgc in fix UMC");
  }
  if (fictgroupname && strcmp(groupname,"all") == 0)
    error->all(FLERR, "UMC fix group cannot be 'all' with fictitious particles.");

  // create computes for real group (fix group) before any dependent computes
  c_keatom = modify->add_compute("keatom all ke/atom");
  c_kereal = modify->add_compute(fmt::format("kereal {} reduce sum c_keatom",groupname));

  c_peatom = modify->add_compute("peatom all pe/atom");
  c_pereal = modify->add_compute(fmt::format("pereal {} reduce sum c_peatom",groupname));

  c_patom  = modify->add_compute(fmt::format("patom {} stress/atom NULL pair",groupname));
  c_preal  = modify->add_compute(fmt::format("preal {} reduce sum c_patom[1] c_patom[2] c_patom[3]",groupname));

  // lots of extra initialization for gcmc
  if (fictgroupname) {
    // fict group settings
    ifictgroup = group->find(fictgroupname);
    fictbit = group->bitmask[ifictgroup];

    // add fictitious particle to move group
    group->assign(fmt::format("MC_move union MC_move {}",fictgroupname));

    // neigh_modify include to avoid calculating energies for fictitious particles when not necessary
    if (atom->firstgroupname) error->warning(FLERR, "UMC must replace {} as include group for internal use.",atom->firstgroupname);
    input->one(fmt::format("atom_modify first {}",groupname));
    input->one(fmt::format("neigh_modify include {}",groupname));
    
    // also exclude fict-fict interactions when calculating fictitious energies
    input->one(fmt::format("neigh_modify exclude group {} {}",fictgroupname,fictgroupname));
  }
  if (cap) {
    // subtract cap particles from move group
    group->assign(fmt::format("MC_move subtract MC_move {} {}",caplogroupname,caphigroupname));
  }
  
  // create \Lambda^3 for each species
  if (pgc > 0. || psgc > 0.) {
    memory->create(lambda3, nrealtypes+1, "hmc:lambda3"); // +1 b/c 1-indexed
    double lambda;
    for (int itype=1; itype<=nrealtypes; itype++) {
      lambda = sqrt( force->hplanck * force->hplanck * beta / (2.0*MathConst::MY_PI * atom->mass[itype] * force->mvv2e) );
      lambda3[itype] = lambda * lambda * lambda;
      if (comm->me == 0) utils::logmesg(lmp,"itype = {}, mass = {}, lambda = {}, lambda3 = {}\n",itype,atom->mass[itype],lambda,lambda3[itype]);
    }
  }
 
  // create type counter - easier to track than repeatedly invoking compute->count_types
  memory->create(typecount, nrealtypes+1, "hmc:typecount"); // +1 b/c 0 is fict
  bigint *typecountlocal;
  memory->create(typecountlocal, nrealtypes+1, "hmc:typecountlocal"); // +1 b/c 0 is fict
  for (int itype=0; itype<=nrealtypes; itype++) {
    typecountlocal[itype] = 0;
  }
  // count types.  Real species are the leading diagonal, types 1..nrealtypes; the fict tier is the
  // contiguous (0->j) block, whose bounds are read off the grid rather than assumed, so with no gc
  // the range is empty and every non-real type is correctly reported as a bridge.
  int fictlo = (tgc > -1) ? typegrid[0][1] : 0;
  int ficthi = (tgc > -1) ? typegrid[0][nrealtypes] : -1;
  int itype;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & movebit) { // count both real and fict
      itype = atom->type[i];
      if (itype <= nrealtypes) typecountlocal[itype]++;
      else if (itype >= fictlo && itype <= ficthi) typecountlocal[0]++;
      else error->all(FLERR, "Bridge types are not allowed in the initial configuration for fix_umc");
    }
  }
  MPI_Allreduce(typecountlocal,typecount,nrealtypes+1,MPI_LMP_BIGINT,MPI_SUM,world);
  memory->destroy(typecountlocal);

  if (tgc > -1 && typecount[0] == 0)
    error->all(FLERR, "GC requires fictitious particles in fictitious types for fix_umc");
    
  for (int itype=0; itype<=nrealtypes; itype++) {
    if (comm->me == 0) utils::logmesg(lmp,"itype = {}, typecount = {}\n",itype,typecount[itype]);
  }
  
  // random number generator, same for all procs
  random_equal = new RanPark(lmp, seed);
  random_unequal = new RanPark(lmp, seed+comm->me);

  pereal_stored = virial_stored = virialz_stored = 0.0;   // real values land in setup()

  nmax_stored = 0;
  atom_swap_nmax = 0;
  
  laststep = update->ntimestep;

  // set comm size needed by this Fix
  if (atom->q_flag) error->all(FLERR, "Charge not yet supported by UMC.");
  if (atom->q_flag) comm_forward = 2;
  else comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixUMC::~FixUMC()
{
  memory->destroy(mu);
  memory->destroy(mu0);
  memory->destroy(muend);
  memory->destroy(swaptypes);
  memory->destroy(typegrid);
  memory->destroy(typecount);
  memory->destroy(lambda3);
 
  delete random_equal;
  delete random_unequal;

  memory->sfree(image_stored);
  memory->sfree(mask_stored);
  memory->sfree(tag_stored);
  memory->sfree(type_stored);
  memory->sfree(x_stored);
  memory->sfree(plocal);

  modify->delete_compute("keatom");
  modify->delete_compute("kereal");
  modify->delete_compute("peatom"); 
  modify->delete_compute("pereal");
  modify->delete_compute("patom");
  modify->delete_compute("preal");
  
  // Undo the neighbor settings the ctor installed for the fict tier.  Left in place they silently
  // change every later run: includegroup keeps the pair build restricted to the real group and
  // firstgroupname keeps the atom ordering constraint.  Poke the objects directly instead of
  // issuing commands -- LAMMPS::destroy() deletes neighbor/output/group BEFORE modify, so a fix
  // destructor running at teardown would drive input->one() into freed state.  The null tests are
  // exactly that teardown check: Pointers holds references to the LAMMPS members.
  if (fictgroupname) {
    if (neighbor) {
      neighbor->includegroup = 0;
      // drop only the fict-fict pair the ctor excluded; any the user set stay
      for (int i = 0; i < neighbor->nex_group; i++) {
        if (neighbor->ex1_group[i] != ifictgroup || neighbor->ex2_group[i] != ifictgroup) continue;
        for (int j = i + 1; j < neighbor->nex_group; j++) {
          neighbor->ex1_group[j-1] = neighbor->ex1_group[j];
          neighbor->ex2_group[j-1] = neighbor->ex2_group[j];
        }
        neighbor->nex_group--;
        break;
      }
    }
    if (atom) {
      delete[] atom->firstgroupname;
      atom->firstgroupname = nullptr;
    }
  }

  // delete[] and fclose both accept the null a disabled option leaves behind, and print_fp /
  // revtest_fp are only ever assigned on rank 0, so none of these need a guard of their own
  delete[] groupname;
  delete[] fictgroupname;
  delete[] pair_style_name;
  delete[] caphigroupname;
  delete[] caplogroupname;
  delete[] mcdumpid;
  delete[] print_text;
  memory->sfree(print_copy);
  memory->sfree(print_work);
  if (print_fp) fclose(print_fp);
  if (revtest_fp) fclose(revtest_fp);
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */
void FixUMC::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix UMC command");

  pgc = 0.; psgc = 0; phop = 0; pxycap = 0; pzcap = 0.; piso = 0; paniso = 0;
  nswaptypes = 0;
  
  revtest = 0;
  reverse = 0;
  trial = 0;
  imcdump = -1;
  onlydumpgc = 0;
  npefict_dumps = 0;
  cap = 0;
  press = 0.;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "pair_style") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style in fix_umc");
      if (pair_style_name) error->all(FLERR, "Pair style already set in fix_umc");
      pair_style_name = utils::strdup(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "mu") == 0) {
      // mu <mu1> ... <mun>, one value (or NULL) per REAL species.  The list length defines
      // nrealtypes: consume args while they are numbers or NULL, since every other keyword
      // is a non-numeric word.  That is what frees the keywords from any ordering rule --
      // the type grid, which needs to know which tiers exist, is built at the END of
      // options() and only CHECKED against atom->ntypes there.
      if (nrealtypes != -1) error->all(FLERR, "mu given twice in fix_umc");
      int n = 0;
      while (iarg + 1 + n < narg &&
             (strcmp(arg[iarg+1+n],"NULL") == 0 || utils::is_double(arg[iarg+1+n]))) n++;
      if (n < 1) error->all(FLERR, "Illegal mu arg length in fix_umc");
      if (n > atom->ntypes) error->all(FLERR, "More mu values than atom types in fix_umc");
      nrealtypes = n;
      for ( int itype=1; itype <= nrealtypes; itype++) {
        if (strcmp(arg[iarg+itype],"NULL") == 0) mu[itype] = NAN;
        else {
          mu[itype] = utils::numeric(FLERR, arg[iarg+itype], false, lmp);
          swaptypes[nswaptypes] = itype;
          nswaptypes++;
        }
      }
      iarg += nrealtypes + 1;
    }
    else if (strcmp(arg[iarg], "muend") == 0) {
      // muend <e1> ... <en>, one per REAL species, NULL = that species does not ramp.  mu is
      // linear in TIME from mu0 to muend across the run, re-latched once per TRIAL by
      // update_mu().  Scanned like mu, so the keywords stay order-free.
      int n = 0;
      while (iarg + 1 + n < narg &&
             (strcmp(arg[iarg+1+n],"NULL") == 0 || utils::is_double(arg[iarg+1+n]))) n++;
      if (n > atom->ntypes) error->all(FLERR, "More muend values than atom types in fix_umc");
      for (int itype = 1; itype <= n; itype++)
        if (strcmp(arg[iarg+itype],"NULL") != 0)
          muend[itype] = utils::numeric(FLERR, arg[iarg+itype], false, lmp);
      iarg += n + 1;
    }
    else if (strcmp(arg[iarg], "trials") == 0) {
      // trials <N>: end the run after exactly N trials and ramp mu over those N, so two legs
      // run the identical mu(k).  The run must hold N trials; N*(longest move) always does.
      if (iarg + 2 > narg) error->all(FLERR, "Illegal trials in fix_umc");
      ntrialsmax = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      if (ntrialsmax <= 0) error->all(FLERR, "Illegal trials count in fix_umc");
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "gc") == 0) {
      if (iarg + 6 > narg) error->all(FLERR, "Illegal gc in fix_umc");
      fictgroupname = utils::strdup(arg[iarg+1]);
      pgc = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tgc = utils::inumeric(FLERR, arg[iarg+3], false, lmp);
      double T_insert = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      beta_insert = 1. / (force->boltz * T_insert);
      double T_delete = utils::numeric(FLERR, arg[iarg+5], false, lmp);
      beta_delete = 1. / (force->boltz * T_delete);
      iarg += 6;
    }
    else if (strcmp(arg[iarg], "sgc") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal sgc in fix_umc");
      psgc = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      tsgc = utils::inumeric(FLERR, arg[iarg+2], false, lmp);
      if (strcmp(arg[iarg+3],"NULL") == 0) beta_sg = 0.;
      else beta_sg = 1. / (force->boltz * utils::numeric(FLERR, arg[iarg+3], false, lmp) );
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "hop") == 0) {
      error->all(FLERR, "hop moves are currently disabled in fix UMC");
    }
    else if (strcmp(arg[iarg], "cap") == 0) {
      if (iarg + 8 > narg) error->all(FLERR, "Illegal cap in fix_umc");
      cap = 1;
      caplogroupname = utils::strdup(arg[iarg+1]);
      icaplogroup = group->find(caplogroupname);
      caphigroupname = utils::strdup(arg[iarg+2]);
      icaphigroup = group->find(caphigroupname);
      pxycap = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      xycapmax = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      pzcap = utils::numeric(FLERR, arg[iarg+5], false, lmp);
      zcapmax = utils::numeric(FLERR, arg[iarg+6], false, lmp);
      press  = utils::numeric(FLERR, arg[iarg+7], false, lmp);
      iarg += 8;
    }
    else if (strcmp(arg[iarg], "iso") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal iso in fix_umc");
      piso   = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      isomax = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      press  = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "aniso") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal aniso in fix_umc");
      paniso   = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      anisomax = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      press  = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "print") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal print in fix_umc");
      print_nevery = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      if (print_nevery <= 0) error->all(FLERR, "Illegal print frequency in fix_umc");

      if (comm->me == 0) {
        print_fp = fopen(arg[iarg+2], "w");
        if (print_fp == nullptr)
          error->one(FLERR, "Cannot open fix UMC print file {}: {}", arg[iarg+2],
                     utils::getsyserror());
      }

      print_text = utils::strdup(arg[iarg+3]);
      print_maxcopy = strlen(print_text) + 1;
      print_maxwork = print_maxcopy;
      print_copy = (char *) memory->smalloc(print_maxcopy * sizeof(char), "UMC:print_copy");
      print_work = (char *) memory->smalloc(print_maxwork * sizeof(char), "UMC:print_work");
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "dump") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal dump in fix_umc");
      mcdumpid = utils::strdup(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "revtest") == 0) {
      revtest = 1;
      iarg++;
    }
    else if (strcmp(arg[iarg], "bias") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal bias in fix_umc");
      // force-biased velocities: bias <t> <r0> <Tcap>  (t=0 disables)
      tvbias  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      rvbias = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tbiascap = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      if (rvbias <= 0.0) error->all(FLERR, "fix UMC bias r0 must be > 0");
      if (tbiascap <= 0.0) error->all(FLERR, "fix UMC bias Tcap must be > 0");
      iarg += 4;
    }
    else {
      if (comm->me == 0) utils::logmesg(lmp,"args = {}",arg[iarg]); 
      error->all(FLERR, "Illegal fix_umc command (else)");
    }
  }

  if (nrealtypes == -1) nrealtypes = atom->ntypes;   // no mu: no swap moves, every type is a species

  // Capture the ramp anchor ONCE, here rather than in init(): init() runs per `run`, and by the
  // second one mu has already been ramped, so anchoring there would let mu0 drift.  A species
  // with no muend gets muend = mu0, which makes the update mu0 + f*0 -- exactly constant, so
  // the ramp arithmetic needs no per-species test at all.  A NULL species is NAN in both and
  // stays NAN.
  for (int itype = 0; itype <= atom->ntypes; itype++) {
    mu0[itype] = mu[itype];
    if (std::isnan(muend[itype])) muend[itype] = mu[itype];
  }


  // build the species-pair type grid.  species 0 = NULL, 1..nrealtypes = real; a LAMMPS type
  // encodes an ordered pair (s1->s2): s1 = the identity under hybrid sub-style 1 (scaled 1-lambda),
  // s2 = under sub-style 2 (scaled lambda).  lambda is one global variable, so resting bulk atoms
  // must sit on the lambda-invariant diagonal (i->i) = data-file types 1..nrealtypes.
  // Layout [real | fict | sgc], compact -- a tier occupies indices only when it is active, and the
  // sgc tier skips its own diagonal since (i->i) is already the real type.  There is no (i->0)
  // delete tier: a deletion is a (0->j) fict type read at lambda = 1 with lambda ramping DOWN.
  // Built HERE, at the end of options(), because it needs to know which tiers exist -- that is what
  // lets gc/sgc appear on either side of mu.  ntypes is only checked, never solved: nrealtypes came
  // from the mu list length.
  // Every call site looks types up in typegrid[][], so no per-tier base offsets are kept.
  {
    int n = nrealtypes;
    memory->create(typegrid, n+1, n+1, "hmc:typegrid");
    for (int a = 0; a <= n; a++)
      for (int b = 0; b <= n; b++) typegrid[a][b] = 0;

    int t = 0;
    // real home diagonal (i->i): data-file types 1..nrealtypes (lambda-invariant)
    for (int i = 1; i <= n; i++) typegrid[i][i] = ++t;
    // fict (0->j): insertion reservoir tier (gc)
    if (tgc > -1) for (int j = 1; j <= n; j++) typegrid[0][j] = ++t;
    // sgc off-diagonal (i->j, i!=j): morph bridges.  Gradual only -- an instant sgc swap retypes
    // real -> real in one step and never sits on a bridge, so tsgc = 0 asks for no tier.
    if (tsgc > 0)
      for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
          if (i != j) typegrid[i][j] = ++t;

    if (t != atom->ntypes)
      error->all(FLERR, "fix_umc type grid size {} != ntypes {} (nreal {} from mu, tiers: real{}{})",
                 t, atom->ntypes, n, tgc > -1 ? " fict" : "", tsgc > 0 ? " sgc" : "");

    // log the grid in type order (scan typegrid for each type; startup only, so the
    // inverse type -> (s1,s2) is recomputed here rather than stored)
    if (comm->me == 0) {
      utils::logmesg(lmp, "UMC type grid (nreal={}, ntypes={}): ", n, atom->ntypes);
      for (int tt = 1; tt <= atom->ntypes; tt++)
        for (int a = 0; a <= n; a++)
          for (int b = 0; b <= n; b++)
            if (typegrid[a][b] == tt) utils::logmesg(lmp, "{}=({}->{}) ", tt, a, b);
      utils::logmesg(lmp, "\n");
    }
    // No pair_coeff 'none' plumbing: with no (i->0) tier, every cell of the grid has a non-NULL
    // species in at least one sub-style, so PairHybrid::init_one covers them all.
  }

  if ((tgc > 0 || tsgc > 0) && !pair_style_name)
    error->all(FLERR, "pair_style must be set for gradual gc/sgc in fix_umc");
      
  if ((pgc > 0. || psgc > 0. || phop > 0. ) && nswaptypes == 0) error->all(FLERR, "Illegal gcmc in fix_umc; mu must be defined");

  // an sgc move redraws newtype until it differs from oldtype, so a single swap species spins
  // forever on the first trial.  Unlike an emptied species -- which the non-empty redraw and
  // species_ratio already handle -- this is a config error, so catch it once here.
  if (psgc > 0. && nswaptypes < 2)
    error->all(FLERR, "fix UMC sgc requires at least two species with a mu value");
 
  tmax = tc;
  if (tsgc > tmax) tmax = tsgc;
  if (tgc > tmax) tmax = tgc;

  pc = 1. - pgc - psgc - phop - pxycap - pzcap - piso - paniso;

  if (comm->me == 0) utils::logmesg(lmp,"pc = {}, sgcmc = {}, pgc = {}, phop = {}, pxycap = {}, pzcap = {}, piso = {}, paniso = {} \n",pc,psgc,pgc,phop,pxycap,pzcap,piso,paniso);
  // probably need to make this more floating-point tolerant
  if (pgc < 0. || psgc < 0. || phop < 0. || pc < 0. || pxycap < 0. || pzcap < 0. ) error->all(FLERR, "Illegal move probabilities < 0");
}

/* ---------------------------------------------------------------------- */

int FixUMC::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE; // for NVE
  mask |= POST_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

// before each MD run: set internal flags
void FixUMC::init()
{
  // so modify->setup calculates energies that can be stored
  update->eflag_global = update->ntimestep;
  update->eflag_atom = update->ntimestep;
  
  // from FixNVE
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (tgc > 0 || tsgc > 0) {
    // re-resolve lambda EVERY run rather than trusting the index cached in the ctor.
    // Variable::remove() shifts every higher index down, so one `variable <x> delete` between runs
    // silently repoints this: internal_set() then writes some other variable, and compute_equal()
    // evaluates data[ivar][0], which past nvar is a freed pointer -> segfault.
    lambda_var = input->variable->find("lambda");
    if (lambda_var < 0)
      error->all(FLERR, "Variable lambda must exist for gradual gc/sgc in fix UMC");
    if (!input->variable->internalstyle(lambda_var))
      error->all(FLERR, "Variable lambda must be internal-style for gradual gc/sgc in fix UMC");

    set_hybrid_lambda(0.0);
  }
 
  // from FixAtomSwap::init; includes fict/bridge rows, which must match the
  // real rows for the scaled sub-style to reproduce real interactions at
  // lambda = 1 (ktype stays <= nrealtypes: fict-fict pairs are excluded)
  double **cutsq = force->pair->cutsq;
  for (int itype = 1; itype <= atom->ntypes; itype++)
    for (int jtype = 1; jtype <= atom->ntypes; jtype++)
      for (int ktype = 1; ktype <= nrealtypes; ktype++)
        if (cutsq[itype][ktype] != cutsq[jtype][ktype])
          error->all(FLERR, "Unequal cutoffs not yet supported");
}

// before each MD run: calculate forces, etc.
void FixUMC::setup(int flag)
{
  nmd_trials = nmd_accepts = ninsertion_trials = ninsertion_accepts = ndeletion_trials = ndeletion_accepts = nsgc_trials = nsgc_accepts = niso_trials = niso_accepts = naniso_trials = naniso_accepts = nxycap_trials = nxycap_accepts = nzcap_trials = nzcap_accepts = 0;
  npefict_dumps = 0;
  
  if (!atom->mass) error->all(FLERR, "Fix UMC requires per atom type masses");

  // if GCMC
  if (tgc > 0 || tsgc > 0) {
    set_hybrid_lambda(0.0);
  }
  
  if (cap == 1) {
    double zcaplo_local = domain->boxlo[2];
    double zcaphi_local = domain->boxhi[2];
    
    // find local group high/low
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->mask[i] & group->bitmask[icaplogroup] && atom->x[i][2] > zcaplo_local)  zcaplo_local = atom->x[i][2];
      if (atom->mask[i] & group->bitmask[icaphigroup] && atom->x[i][2] < zcaphi_local)  zcaphi_local = atom->x[i][2];
    }
    
    MPI_Allreduce(&zcaplo_local,&zcaplo,1,MPI_DOUBLE,MPI_MAX,world);
    MPI_Allreduce(&zcaphi_local,&zcaphi,1,MPI_DOUBLE,MPI_MIN,world);

    if (comm->me == 0) utils::logmesg(lmp,"MPI{}: zcaplo = {}, zcaphi = {}\n",comm->me,zcaplo,zcaphi);
  }

  // set internal variables now
  double volume;
  if (cap == 1) volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
  else volume = domain->xprd * domain->yprd * domain->zprd;
  input->one("variable MC_V internal 0.");
  i_MC_V = input->variable->find("MC_V");
  input->variable->internal_set(i_MC_V,volume);
  
  // create-once: redefining a dynamic vector-style variable leaves Variable::data[ivar][1]
  // dangling (freed, not nulled) in patch_30Mar2026; the next retrieve() (print keyword) then
  // double-frees it -> heap corruption.  equal/internal redefinition is safe, so only this one
  // is guarded.  It pins the id of the first UMC instance: reuse the same fix ID when replacing
  // the fix, or f_<id> lookup fails cleanly.
  if (input->variable->find("MC_A") < 0)
    input->one(fmt::format("variable MC_A vector f_{}",id));

  // per-species N and mu, one equal-style variable each.  Not a vector variable: LAMMPS
  // evaluates those once per timestep, so every row of a burst of instant trials would repeat
  // the first.  Equal-style is re-evaluated on every substitution, and redefining one is safe,
  // so these need no create-once guard.
  for (int itype = 1; itype <= nrealtypes; itype++) {
    input->one(fmt::format("variable MC_N{} equal f_{}[{}][1]",itype,id,itype));
    input->one(fmt::format("variable MC_mu{} equal f_{}[{}][2]",itype,id,itype));
  }

  // all three read the fix's stored PE/virial rather than re-invoking c_pereal/c_preal: the
  // trial that just ran already reduced the energy, and the virial only exists on the steps
  // that tallied it, so evaluating the computes here would be redundant at best and fatal
  // mid-burst at worst.
  input->one(fmt::format("variable MC_U equal f_{}[1][3]",id));

  input->one(fmt::format("variable MC_P equal -f_{}[1][4]/(3*v_MC_V)+f_{}[1][6]/(v_MC_V)*{}",id,id,force->nktv2p/beta));

  input->one(fmt::format("variable MC_Pz equal -f_{}[1][5]/v_MC_V+f_{}[1][6]/(v_MC_V)*{}",id,id,force->nktv2p/beta));

  // setup dumping
  if (mcdumpid) {
    // clear before re-searching: options() runs once per fix instance but setup() runs once per
    // `run`, so a dump deleted or redefined in between would leave the previous run's index here,
    // sail past the -1 check below, and index output->dump[]/every_dump[] out of bounds.
    imcdump = -1;
    for (int idump=0; idump<output->ndump; idump++) {
      if (strcmp(mcdumpid,output->dump[idump]->id) == 0) {
        imcdump = idump;
        break;
      }
    }
    if (imcdump == -1) error->all(FLERR,"dump index not found");
    if (comm->me == 0) utils::logmesg(lmp,"dump[{}] = {}, ndump = {}\n",imcdump,output->dump[imcdump]->id,output->ndump);
  }

  // re-anchor to this run: moves arm at laststep+1, and a run that ended on the `trials` timeout
  // left laststep past that run's end, so reusing one fix across `run` commands would arm
  // nothing.  A stale trial would likewise keep ramping lambda until it left [0,1].
  laststep = update->ntimestep;
  trial = 0;
  reverse = 0;

  // fresh ramp for this run block: the anchor is mu0, captured once in options()
  ntrialsdone = 0;
  for (int itype = 1; itype <= nrealtypes; itype++) mu[itype] = mu0[itype];

  // with `trials` the first trial is at mu0 and the Nth exactly at muend; without it, only the
  // trials the step budget is EXPECTED to hold are known, so the ramp lands wherever it lands
  if (ntrialsmax > 1) mustep = 1. / (double) (ntrialsmax - 1);
  else if (update->endstep > update->beginstep)   // else `run 0`: inf would make every mu NaN
    mustep = (pc*tc + psgc*(tsgc > 0 ? tsgc : 0) + pgc*(tgc > 0 ? tgc : 0))
             / (double) (update->endstep - update->beginstep);
  else mustep = 0.;

  // storing before next move so on same step as energy computed
  store_pe();
  store_atom();
}

/* ----------------------------------------------------------------------
   latch mu for the trial about to be armed.  Once per TRIAL, not per timestep, so a move that
   spans timesteps is judged with the mu it was selected under -- selection bias, density ratio
   and e_diff all read one value.  A species with muend == mu0, or a NULL one, sits at mu0 with
   no special case.
------------------------------------------------------------------------- */
void FixUMC::update_mu()
{
  double f = ntrialsdone * mustep;
  if (f > 1.) f = 1.;   // only reachable without `trials`, where mustep is an estimate

  for (int itype = 1; itype <= nrealtypes; itype++)
    mu[itype] = mu0[itype] + f * (muend[itype] - mu0[itype]);

  ntrialsdone++;
}

/* ----------------------------------------------------------------------
   `trials` was given but the run ended first, so the ramp never reached muend -- the work
   from such a leg is meaningless, so refuse rather than let it be analysed.
------------------------------------------------------------------------- */
void FixUMC::post_run()
{
  if (ntrialsmax == 0) return;

  // force_timeout() zeroes the timer for good, so without this every later run exits at once
  timer->reset_timeout();

  if (ntrialsdone < ntrialsmax)
    error->all(FLERR, "fix UMC ran out of timesteps after {} of {} trials",
               ntrialsdone, ntrialsmax);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void FixUMC::set_hybrid_lambda(double new_lambda)
{
  constexpr double eps = 1.0e-12;

  if (new_lambda < -eps || new_lambda > 1.0 + eps)
    error->all(FLERR, "Fix UMC lambda must be between 0 and 1");

  // read the current lambda instead of taking it on trust from the caller.  Every call site used
  // to assert what lambda "should" be, and the early-out below acts on that assertion, so a single
  // wrong claim would skip the internal_set and leave the variable out of step with the pair style.
  double old_lambda = input->variable->compute_equal(lambda_var);

  if (fabs(old_lambda) < eps) old_lambda = 0.0;
  else if (fabs(old_lambda - 1.0) < eps) old_lambda = 1.0;
  if (fabs(new_lambda) < eps) new_lambda = 0.0;
  else if (fabs(new_lambda - 1.0) < eps) new_lambda = 1.0;

  if (new_lambda == old_lambda) return;

  if (new_lambda == 0.0 || old_lambda == 0.0) {
    char pair[] = "pair", one[] = "1", two[] = "2", compute[] = "compute", yes[] = "yes", no[] = "no";
    char *args1[] = {pair, pair_style_name, one, compute, yes};
    char *args2[] = {pair, pair_style_name, two, compute, new_lambda == 0.0 ? no : yes};
    force->pair->modify_params(5,args1);
    force->pair->modify_params(5,args2);
  }

  input->variable->internal_set(lambda_var, new_lambda);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void FixUMC::initial_integrate(int /*vflag*/)
{
  bigint step = update->ntimestep;

  // the longest a newly armed move of any type can run
  // time for a new move
  if (step == laststep + 1) {
    if (ntrialsmax > 0 && ntrialsdone >= ntrialsmax) {
      // trial budget spent, and nothing is in flight -- end the run here.  Verlet tests the
      // timeout at the top of the NEXT step, so this one still runs in full: return before the
      // arming tail below so it neither resamples velocities nor advances lambda.  trial must
      // be cleared first -- it still names the move end_of_step just judged, and the
      // `if (trial != 0)` block below would advance that move's ramp once more from a lambda
      // end_of_step already returned to 0 (for a deletion, straight out of [0,1]).
      // laststep is pushed past the run so end_of_step stays quiet for this step.
      timer->force_timeout();
      trial = 0;
      laststep = update->laststep + 1;
      return;
    }
    else if (revtest && trial != 0 && !reverse) {
      // revtest: reverse the forward trajectory move (GC or SGC) just completed; same atom
      reverse = 1;
      if (comm->me == 0) utils::logmesg(lmp,"performing reverse for revtest (trial = {})\n",trial);
      if (trial == 2) {
        initiate_sgcmc();
        laststep = step + tsgc - 1;
      } else {
        trial *= -1;
        if (trial == -1) initiate_deletion();
        else if (trial == 1) initiate_insertion();
        else error->all(FLERR,"trial != +/-1");
        laststep = step + tgc - 1;
      }
    } else if (update->laststep - step + 1 < tmax) {
      // Tail of the run: a move armed here could outlast it, and end_of_step only fires at
      // laststep, so it would never be accepted or rejected -- leaving an atom on a bridge or
      // fict type with lambda stranded mid-ramp.  Fill the tail with ordinary canonical NVE
      // trials instead (trial = 0: momentum refresh, NVE leg, accept on dH -- judged like any
      // other move), each of the usual tc length, the last shortened to land exactly on the
      // run's final step.  One tc-length trial per arm, NOT a single trial spanning the whole
      // tail: an NVE leg that long drifts far enough that it would never be accepted.
      trial = 0;
      laststep = step + tc - 1;
      if (laststep > update->laststep) laststep = update->laststep;
      nmd_trials++;

      // these are trials too, and they never reach the selection loop, so re-arm mu here as well
      update_mu();
    } else {
    reverse = 0;

    // initially try non-MD moves within loop
    double p = -1.;
    double pnow = pxycap + pzcap + piso + paniso + psgc + phop;
    
    while (p < pnow) {
      // end the burst on trial N exactly.  Ending the run here rather than breaking: p still
      // holds the finished trial's draw, which passes the `p < pnow + pc` test below and
      // would arm an extra NVE trial outside the budget and outside update_mu().
      if (ntrialsmax > 0 && ntrialsdone >= ntrialsmax) {
        timer->force_timeout();
        trial = 0;
        laststep = update->laststep + 1;
        return;
      }

      // re-arm mu at the START of each trial.  One loop iteration IS one trial of ANY kind -- an
      // iteration that draws below pnow runs an instant move or arms the gradual sgc, and the one
      // that draws above pnow exits and its p then selects the canonical NVE or gc move.  So this
      // covers every trial, not just the instant ones, and a gradual move keeps the mu it was
      // armed with for its whole ramp (bias, e_diff, logged Omega).
      update_mu();

      p = random_equal->uniform();
      trial = 0; // need to set as default
      if (comm->me == 0) utils::logmesg(lmp,"\nt = {}, p = {:.3f}\n",step,p);

      double pthresh = 0;
      if (p < (pthresh += pxycap) ) trial_xycap();
      else if (p < (pthresh += pzcap) ) trial_zcap();
      else if (p < (pthresh += piso) ) trial_box(-1);
      else if (p < (pthresh += paniso/3) ) trial_box(0);
      else if (p < (pthresh += paniso/3) ) trial_box(1);
      else if (p < (pthresh += paniso/3) ) trial_box(2);
      else if (p < (pthresh += psgc) ) {
        if (tsgc > 0) {
          trial = 2;
          initiate_sgcmc();
          p = pnow;
        } else {
          trial_sgcmc();
          store_atom();
        }
      }
      else if (p < (pthresh += phop) ) {
        // hop moves removed pending rework of their type conventions (see git history)
        error->all(FLERR, "hop moves are currently disabled in fix UMC");
      }

      if (p < pnow) print();   // an instant move is a completed trial; end_of_step logs the rest

      if (step == update->laststep) break; // exit while loop if SGCMC run 0
    }
      
    // The loop only exits with p >= pnow: the gradual-sgc arm sets p = pnow and is caught by the
    // trial == 2 test first, and the run-0 break below leaves nothing in flight.  Select over the
    // remaining [pnow, pnow+pc+pgc) band from pnow directly rather than reusing pthresh, whose
    // value depends on how far the short-circuited chain above got before a branch matched.
    if (trial == 2) {
      laststep = step + tsgc-1;
    }
    else if (p < pnow + pc) {
      laststep = step + tc - 1;
      nmd_trials++;
    }
    else if (p < pnow + pc + pgc) {
      if (p < pnow + pc + pgc/2) trial = 1;
      else trial = -1;
      if (trial == -1) initiate_deletion();
      else if (trial == 1) initiate_insertion();
      else error->all(FLERR,"trial != +/-1");
      laststep = step + tgc-1;
    }
    else error->all(FLERR, "Move selection probability messed up");
    } // end normal selection (else of revtest reverse)

    // we assume updated energies, neighbor lists at this point
    initiate_velocities();
    store_ke();
 
    modify->clearstep_compute(); // sets invoked flag to 0
    c_peatom->addstep(laststep);
  }

  // update insertion/deletion Hamiltonian
  // Verlet needs a_f = a_0, a_f-1 = a_1, etc.
  // lambda comes from the position in the trial, NOT from accumulating += 1/T each step: repeated
  // addition drifts by ~T*2^-53, so past T ~ 9000 the final step overshoots the +/-1e-12 band
  // set_hybrid_lambda accepts and aborts the run mid-ramp.  g runs 1/T on the trial's first step
  // to exactly 1 on laststep; a deletion ramps the other way.  This also drops one variable
  // evaluation per timestep.
  if (trial != 0) {
    bigint tramp = (trial == 2) ? tsgc : tgc;
    double g = (double) (step - laststep + tramp) / (double) tramp;
    set_hybrid_lambda(trial == -1 ? 1.0 - g : g);
  }
    
  // FixNVE::initial_integrate | easier than managing order of separate fix
  double dtfm;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
 
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & movebit) {
      if (mask[i] & groupbit) {

        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
  
}

// if cap, fix wall/reflect behavior
void FixUMC::post_integrate()
{
  if (cap != 1) return;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & movebit) {
      if (x[i][2] < zcaplo) {
        x[i][2] = zcaplo + (zcaplo - x[i][2]);
        v[i][2] *= -1;
      } else if (x[i][2] > zcaphi) {
        x[i][2] = zcaphi - (x[i][2] - zcaphi);
        v[i][2] *= -1;
      }
    }
  }
}

void FixUMC::final_integrate()
{
  // FixNVE::final_integrate
  double dtfm;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ( (mask[i] & movebit) && (mask[i] & groupbit) ) {
      dtfm = dtf / mass[type[i]];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
    }
  }

  // revtest: per-step switching-trajectory log (forward and reverse legs).
  // PE is tallied on this step's normal force eval (scheduled in initial_integrate);
  // KE is velocity-based so computed on demand. compute_scalar() is collective.
  if (revtest && trial != 0) {
    recompute_peatom(VIRIAL_ATOM);   // fresh per-atom PE this step; Verlet's eflag isn't set here
    c_keatom->compute_peratom();
    double lambda  = input->variable->compute_equal(lambda_var);
    double pereal  = c_pereal->compute_scalar();
    double kereal  = c_kereal->compute_scalar();
    // Omega = H - mu_ramp - kT ln(density_ratio) is the acceptance-exponent quantity
    // (beta_md == beta), ~conserved on a reversible leg; fict endpoints carry no mu.
    // Credit mu_old at the REACTANT and mu_new at the product: g is progress from the
    // reactant, NOT lambda, because a deletion ramps lambda downward (reactant at lambda = 1)
    // so a raw (1-lambda) puts mu_old on the product end -> inverted mu term.
    double mu_old  = (oldtype <= nrealtypes) ? mu[oldtype] : 0.0;
    double mu_new  = (newtype <= nrealtypes) ? mu[newtype] : 0.0;
    double g = (trial == -1) ? 1.0 - lambda : lambda;   // 0 at reactant, 1 at product

    // Fold in the ideal-gas reservoir factor so Omega is the FULL acceptance exponent and not
    // just PE+KE+mu: pmove = density_ratio * ... * exp(-beta*e_diff), so Omega gains
    // -kT ln(density_ratio) at the product, ramped by g like mu -- a per-move constant would
    // otherwise cancel against the row-0 reactant reference.  typecount[0]/volume/lambda3 are
    // fixed across the ramp, so these three reproduce end_of_step's density_ratio, cap-aware
    // volume included: KEEP THEM IN STEP.  sgc conserves number but changes species, so its
    // ratio is the Lambda^3 ratio, not 1.
    double volume;
    if (cap) volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
    else volume = domain->xprd * domain->yprd * domain->zprd;
    double dens = 1.0;
    if      (trial == -1) dens = (typecount[0]+1) * lambda3[oldtype] / volume;
    else if (trial ==  1) dens = volume / lambda3[newtype] / typecount[0];
    else if (trial ==  2) dens = lambda3[oldtype] / lambda3[newtype];

    double H = pereal + kereal;
    double Omega = H - ((1.0 - g)*mu_old + g*mu_new) - g*log(dens)/beta;
    if (comm->me == 0) {
      if (!revtest_fp) {
        revtest_fp = fopen("revtest.dat","a");
        if (!revtest_fp) error->one(FLERR,"Cannot open revtest.dat");
        fseek(revtest_fp,0,SEEK_END);
        if (ftell(revtest_fp) == 0)   // header only on a freshly-created/empty file
          fmt::print(revtest_fp,"# t trial oldtype newtype lambda pereal kereal H Omega\n");
      }
      fmt::print(revtest_fp,"{} {} {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n",
                 update->ntimestep, trial, oldtype, newtype, lambda, pereal, kereal, H, Omega);
      fflush(revtest_fp);
    }
  }
}

void FixUMC::print()
{
  if (!print_text) return;

  print_count++;
  if (print_count % print_nevery) return;

  strncpy(print_copy, print_text, print_maxcopy);
  input->substitute(print_copy, print_work, print_maxcopy, print_maxwork, 0);

  if (comm->me == 0 && print_fp) {
    fmt::print(print_fp, "{}\n", print_copy);
    fflush(print_fp);
  }
}

void FixUMC::restore_atom() 
{
  int n = nlocal_stored;
  if (n > atom->nmax) error->all(FLERR,"n>nmax restore not supported");

  atom->nlocal = n;
  atom->nghost = 0;
  
  if (n == 0) return;

  memcpy( atom->x[0], x_stored,n*3*sizeof(double));
  memcpy( atom->type, type_stored, n*sizeof(int));
  memcpy( atom->mask, mask_stored, n*sizeof(int));
  memcpy( atom->tag, tag_stored, n*sizeof(tagint));
  memcpy( atom->image, image_stored, n*sizeof(imageint));
  
  atom->avec->force_clear(0,sizeof(double)*nlocal_stored);
}

// after initiate_gcmc so transforming particle in igroup
void FixUMC::initiate_velocities()
{
  double sigma;

  double **v = atom->v;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // revtest reverse leg: flip the forward-leg velocities and DO NOT resample,
  // so the trajectory retraces under velocity-Verlet
  if (revtest && reverse) {
    if (comm->me == 0) utils::logmesg(lmp,"reversing velocities for revtest\n");
    for (int i = 0; i < nlocal; i++)
      for (int x = 0; x < 3; x++) v[i][x] *= -1.;
    return;
  }

  // force-biased velocities: jump lambda to the sudden-transform endpoint, draw
  // against those relaxation forces, then restore the start-state forces.
  // bias_logfwd carries the forward proposal correction to end_of_step.
  if (tvbias > 0.0 && trial != 0) {
    double lam0 = input->variable->compute_equal(lambda_var); // trajectory start lambda
    double lam1 = (trial == -1) ? 0.0 : 1.0;                  // sudden-transform endpoint
    set_hybrid_lambda(lam1);  recompute_peatom(0);
    bias_logfwd = velocity_bias(1);
    set_hybrid_lambda(lam0);  recompute_peatom(VIRIAL_ATOM);
    return;
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & movebit) {
      // every move-group atom (real, transforming, or fictitious) carries a
      // well-defined type here; sample velocities at its own type's mass so the
      // sampling mass matches the integration mass in initial_integrate.
      sigma = sqrt( beta_md * mass[type[i]] * force->mvv2e);
      v[i][0] = random_unequal->gaussian() / sigma;
      v[i][1] = random_unequal->gaussian() / sigma;
      v[i][2] = random_unequal->gaussian() / sigma;
    }
    // need to explicitly set to 0 bc cap/base can become finite after restoring
    else {
      v[i][0] = 0.;
      v[i][1] = 0.;
      v[i][2] = 0.;
    }
  }
}

// velocity bias for GC moves: bias initial velocities along the forces that
// appear when the transforming atom (idswap) is suddenly switched, with a
// strength decaying as exp(-r/rvbias) from it:
//   mean_i = c dt w(r_i) ftm2v F_i/m_i,
// so c is measured in timesteps (the velocity a full force kick imparts over
// c steps). The mean is smoothly capped component-wise at the thermal speed
// corresponding to tbiascap*T_md.
// only real (non-fict) move atoms are biased; their KE is kereal, already in
// the acceptance, and ghosts always carry fictbit so the biased set is the
// same on the forward and reverse legs.
//
// draw = 1: sample v_i = mean_i + Maxwell-Boltzmann noise for move atoms
//           (zeroing the rest) and return the forward log-correction
//             sum_i sigma_i^2 (+0.5 mean_i^2 - v_i.mean_i)
// draw = 0: leave velocities alone and return the reverse log-correction
//             sum_i sigma_i^2 (-0.5 mean_i^2 - v_i.mean_i)
//
// the acceptance uses exp(logfwd + logrev). atom->f must hold the
// sudden-transform (endpoint) forces on entry. w = 0 atoms contribute zero
// to the corrections, so one loop serves both directions.
double FixUMC::velocity_bias(int draw)
{
  // broadcast the transforming atom's position via the owner-only sum trick
  double xs[3], loc[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->tag[i] == idswap) {
      loc[0] = atom->x[i][0]; loc[1] = atom->x[i][1]; loc[2] = atom->x[i][2];
      break;
    }
  MPI_Allreduce(loc, xs, 3, MPI_DOUBLE, MPI_SUM, world);

  double **v = atom->v, **x = atom->x, **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type, *mask = atom->mask;

  double clocal = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & movebit)) {
      if (draw) v[i][0] = v[i][1] = v[i][2] = 0.0;
      continue;
    }
    double sigma2 = beta_md * mass[type[i]] * force->mvv2e;
    double w = 0.0;
    if ((mask[i] & groupbit) && !(mask[i] & fictbit)) {
      double dx = x[i][0]-xs[0], dy = x[i][1]-xs[1], dz = x[i][2]-xs[2];
      domain->minimum_image(FLERR, dx, dy, dz);
      w = exp(-sqrt(dx*dx + dy*dy + dz*dz) / rvbias);
    }
    for (int a = 0; a < 3; a++) {
      double mean = tvbias * dtv * w * force->ftm2v * f[i][a] / mass[type[i]];
      double meancap = sqrt(tbiascap / sigma2);
      mean = meancap * tanh(mean / meancap);
      if (draw) v[i][a] = mean + random_unequal->gaussian() / sqrt(sigma2);
      clocal += sigma2 * ((draw ? 0.5 : -0.5) * mean*mean - v[i][a]*mean);
    }
  }

  double c;
  MPI_Allreduce(&clocal, &c, 1, MPI_DOUBLE, MPI_SUM, world);
  if (comm->me == 0)
    utils::logmesg(lmp,"velocity bias log-correction ({}) = {:.4f}\n", draw ? "fwd" : "rev", c);
  return c;
}


// assumes atomic energies are up to date
void FixUMC::store_pe() {
  pereal_stored = c_pereal->compute_scalar();
}

void FixUMC::store_ke() {
  c_keatom->compute_peratom();
  kereal_stored = c_kereal->compute_scalar();
}

void FixUMC::store_atom() {
  // add charge if qflag
  
  int n = atom->nlocal;
  nlocal_stored = n;
  if (n == 0) return;

  if (n > nmax_stored ) {
    memory->sfree(x_stored);
    memory->sfree(type_stored);
    memory->sfree(mask_stored);
    memory->sfree(tag_stored);
    memory->sfree(image_stored);
    
    x_stored = (double *) memory->smalloc(n*3*sizeof(double),"hmc:x_stored");
    type_stored = (int *) memory->smalloc(n*sizeof(int),"hmc:type_stored");
    mask_stored = (int *) memory->smalloc(n*sizeof(int),"hmc:mask_stored");
    tag_stored = (tagint *) memory->smalloc(n*sizeof(tagint),"hmc:tag_stored");
    image_stored = (imageint *) memory->smalloc(n*sizeof(imageint),"hmc:image_stored");
    
    nmax_stored = n;
  }

  memcpy( x_stored, atom->x[0], 3*n*sizeof(double));
  memcpy( type_stored, atom->type, n*sizeof(int));
  memcpy( mask_stored, atom->mask, n*sizeof(int));
  memcpy( tag_stored, atom->tag, n*sizeof(tagint));
  memcpy( image_stored, atom->image, n*sizeof(imageint));
}

void FixUMC::trial_xycap()
{
  nxycap_trials++;
  if (comm->me == 0) utils::logmesg(lmp,"starting xycap MC\n");

  double xshift = random_equal->gaussian() * xycapmax;
  double yshift = random_equal->gaussian() * xycapmax;

  // shift the upper cap rigidly in x/y; the same code runs the move (sign = +1) and the
  // rejection undo (sign = -1), as in trial_zcap and trial_box
  auto shift = [&](double sign) {
    for (int i=0; i<atom->nlocal; i++) {
      if (atom->mask[i] & group->bitmask[icaphigroup]) {
        atom->x[i][0] += sign * xshift;
        atom->x[i][1] += sign * yshift;
      }
    }
    comm->forward_comm();
    recompute_peatom(0);
  };

  shift(1.);

  double pe_diff = c_pereal->compute_scalar() - pereal_stored;

  double pmove = exp(-beta * pe_diff);

  if (random_equal->uniform() < pmove ) { // accept
    nxycap_accepts++;
    if (comm->me == 0) utils::logmesg(lmp,"t = {} dx = {} dy = {} xycap accepted ({}/{})\n",update->ntimestep,xshift,yshift,nxycap_accepts,nxycap_trials);
      
    store_pe();
    store_atom();
  }
  else { // reject
    if (comm->me == 0) utils::logmesg(lmp,"xycap rejected ({}/{})\n",nxycap_accepts,nxycap_trials);

    shift(-1.);

    domain->print_box("restored recalculated ");
   
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal restored = {:.3f} stored = {:.3f}\n",pereal_restored,pereal_stored);
    
  }
}

void FixUMC::trial_zcap()
{
  nzcap_trials++;
  double volume_stored = domain->xprd * domain->yprd * (zcaphi - zcaplo);

  double zshift = random_equal->gaussian() * zcapmax;

  if (comm->me == 0) utils::logmesg(lmp,"group hi = {}, lo = {}, shift = {}\n",zcaphi,zcaplo,zshift);

  // gaussian() is unbounded, so a tail draw can invert the slab or the box.  Reject out-of-support
  // proposals before touching the box -- an inverted box crashes in set_global_box()/binning long
  // before the acceptance test could reject it, and pow(V'/V, N) on a negative V is just NaN.
  if (zcaphi + zshift - zcaplo <= neighbor->cutneighmax ||
      domain->boxhi[2] + zshift - domain->boxlo[2] <= neighbor->cutneighmax) {
    if (comm->me == 0)
      utils::logmesg(lmp,"zcap rejected: slab would fall below the neighbor cutoff ({}/{})\n",
                     nzcap_accepts,nzcap_trials);
    return;
  }

  // stretch the slab [zcaplo, zcaphi] by sign*zshift, shifting the upper cap
  // rigidly; the same code runs the move (sign = +1) and the rejection undo (-1)
  auto shift = [&](double sign) {
    double dz = sign * zshift;
    domain->boxhi[2] += dz;
    domain->set_global_box();
    domain->set_local_box();

    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->mask[i] & movebit) {
        double zfrac = (atom->x[i][2] - zcaplo) / (zcaphi - zcaplo);
        atom->x[i][2] = zcaplo + zfrac * (zcaphi + dz - zcaplo);
      } else if (atom->mask[i] & group->bitmask[icaphigroup]) {
        atom->x[i][2] += dz;
      }
    }
    zcaphi += dz;

    // full reneighbor, not just forward_comm: set_local_box() above moved the subdomain bounds and
    // the atoms were rescaled, so owners and the neighbor list are both stale.  This also picks up
    // the reset_box/comm->setup/setup_bins block, which is what keeps the bins in step with the
    // box (see the box_change declaration in the ctor).  trial_box does the same after resize().
    reneighbor();
    recompute_peatom(0);
  };

  domain->print_box("initial ");
  shift(1.);
  domain->print_box("recalculated ");

  double pe_diff = c_pereal->compute_scalar() - pereal_stored;

  // scaled real atoms only: excludes fict (uniform-normalized, no Jacobian
  // contribution) and cap atoms (shifted rigidly, not scaled)
  bigint nreal = 0;
  for (int itype = 1; itype <= nrealtypes; itype++) nreal += typecount[itype];
  double volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
  double volume_ratio = pow( (volume / volume_stored), nreal);
  double volume_diff = volume - volume_stored;
  if (comm->me == 0) utils::logmesg(lmp,"zshift = {}, V0 = {}, V' = {}, N = {}, volume_ratio = {}\n",zshift,volume_stored,volume,nreal,volume_ratio);

  double pmove = volume_ratio * exp( -beta * (pe_diff + press * volume_diff / force->nktv2p) );

  if (random_equal->uniform() < pmove ) { // accept
    nzcap_accepts++;
    input->variable->internal_set(i_MC_V,volume);
    if (comm->me == 0) utils::logmesg(lmp,"zcap accepted ({}/{})\n",nzcap_accepts,nzcap_trials);

    store_pe();
    store_atom();
  }
  else { // reject: undo the shift
    if (comm->me == 0) utils::logmesg(lmp,"zcap rejected ({}/{})\n",nzcap_accepts,nzcap_trials);

    shift(-1.);

    domain->print_box("restored recalculated ");
  }
}


// isotropic (axis = -1) or single-axis (axis = 0,1,2) box-length trial move
void FixUMC::trial_box(int axis)
{
  const int iso = (axis < 0);
  const char *name = iso ? "iso" : "aniso";
  int &ntrials  = iso ? niso_trials  : naniso_trials;
  int &naccepts = iso ? niso_accepts : naniso_accepts;
  ntrials++;

  double volume_stored = domain->xprd * domain->yprd * domain->zprd;

  if (comm->me == 0) utils::logmesg(lmp,"{} box MC: axis = {}, storing volume = {}, pereal = {}\n",name,axis,volume_stored,pereal_stored);

  // proposal: iso = uniform strain on all lengths, aniso = uniform length change
  double delta = (random_equal->uniform() - 0.5) * 2 * (iso ? isomax : anisomax);
  double dl[3] = {0., 0., 0.};
  if (iso) {
    double l0 = (domain->xprd + domain->yprd + domain->zprd) / 3;
    dl[0] = delta * domain->xprd / l0;
    dl[1] = delta * domain->yprd / l0;
    dl[2] = delta * domain->zprd / l0;
  }
  else dl[axis] = delta;

  // Reject any proposal that would leave a box edge non-positive or shorter than the neighbor
  // cutoff, BEFORE touching the box.  Such a box cannot be binned or minimum-imaged, so it would
  // crash inside set_global_box()/setup_bins() long before the acceptance test below could turn it
  // down.  Rejecting out-of-support proposals up front is proper Metropolis: the move is counted
  // as a trial and the state is left untouched.
  for (int d = 0; d < 3; d++) {
    if (domain->boxhi[d] + dl[d] - domain->boxlo[d] <= neighbor->cutneighmax) {
      if (comm->me == 0)
        utils::logmesg(lmp,"{} box trial rejected: edge {} would fall below the neighbor cutoff ({}/{})\n",
                       name,d,naccepts,ntrials);
      return;
    }
  }

  // shift boxhi by sign*dl and remap move-group atoms into the resized box:
  // to fractional coordinates in the old box, back to cartesian in the new one.
  // the same code runs the move (sign = +1) and the rejection undo (sign = -1)
  auto resize = [&](double sign) {
    double boxlo[3], h_inv[6];
    for (int d = 0; d < 3; d++) boxlo[d] = domain->boxlo[d];
    for (int i = 0; i < 6; i++) h_inv[i] = domain->h_inv[i];

    for (int d = 0; d < 3; d++) domain->boxhi[d] += sign * dl[d];
    domain->set_global_box();
    domain->set_local_box();

    double **x = atom->x;
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->mask[i] & movebit) {
        domain->x2lamda(x[i],x[i],boxlo,h_inv);
        domain->lamda2x(x[i],x[i]);
      }
    }
  };

  domain->print_box("initial ");
  resize(1.);
  reneighbor();
  recompute_peatom(0);
  domain->print_box("recalculated ");

  double pe_diff = c_pereal->compute_scalar() - pereal_stored;

  if (comm->me == 0) utils::logmesg(lmp,"pe_diff = {:.3f}\n",pe_diff);

  // scaled real atoms only: excludes fict (uniform-normalized, no Jacobian
  // contribution) and cap atoms (shifted rigidly, not scaled)
  bigint nreal = 0;
  for (int itype = 1; itype <= nrealtypes; itype++) nreal += typecount[itype];
  double volume = domain->xprd * domain->yprd * domain->zprd;
  double volume_ratio = pow( (volume / volume_stored), nreal);
  double volume_diff = volume - volume_stored;
  if (comm->me == 0) utils::logmesg(lmp,"delta = {}, V0 = {}, V' = {}, N = {}, volume_ratio = {}\n",delta,volume_stored,volume,nreal,volume_ratio);

  // iso Hastings factor: uniform strain proposes a length scale s = 1 + delta/l0,
  // whose proposal density in V is asymmetric by s^2 = (V'/V)^(2/3);
  // the additive aniso proposal is symmetric and needs no factor
  double hastings = iso ? pow(volume / volume_stored, 2./3.) : 1.;

  double pmove = hastings * volume_ratio
    * exp( -beta * (pe_diff + press * volume_diff / force->nktv2p) );

  if (random_equal->uniform() < pmove ) { // accept
    naccepts++;
    input->variable->internal_set(i_MC_V,volume);
    if (comm->me == 0) utils::logmesg(lmp,"{} box trial accepted ({}/{})\n",name,naccepts,ntrials);

    store_pe();
    store_atom();
  }
  else { // reject: undo the resize
    if (comm->me == 0) utils::logmesg(lmp,"{} box trial rejected ({}/{})\n",name,naccepts,ntrials);

    resize(-1.);
    reneighbor();
    recompute_peatom(0);

    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal restored = {:.3f} stored = {:.3f}\n",pereal_restored,pereal_stored);
  }
}

void FixUMC::trial_sgcmc()
{
  nsgc_trials++;

  // pick species
  int noldtype = 0;
  while (noldtype == 0) {
    oldtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
    noldtype = typecount[oldtype];
  }
  
  newtype = oldtype;
  while (newtype == oldtype) {
    newtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
  }
  
  if (comm->me == 0) utils::logmesg(lmp,"SGCMC, oldtype = {} ({}), newtype = {} ({})\n",oldtype,typecount[oldtype],newtype,typecount[newtype]);

  // pick swap atom based on energy
  // find idswap and change to new type
  bias_forward();

  comm->forward_comm(this);

  recompute_peatom(0);
  
  bias_reverse();

  double pe_diff = c_pereal->compute_scalar() - pereal_stored;
  
  double e_diff = pe_diff + mu[oldtype] - mu[newtype];

  double trial_ratio = trial_reverse / trial_forward;
  double density_ratio = lambda3[oldtype] / lambda3[newtype]; // (1/Lambda) / (1/Lambda)

  // species-pick asymmetry of the non-empty re-draw (see end_of_step);
  // 1 unless this swap empties or repopulates a species
  int npresent = 0;
  for (int i = 0; i < nswaptypes; i++)
    if (typecount[swaptypes[i]] > 0) npresent++;
  double species_ratio = (double) npresent /
    (npresent - (typecount[oldtype] == 1 ? 1 : 0) + (typecount[newtype] == 0 ? 1 : 0));
  
  if (comm->me == 0) utils::logmesg(lmp,"pereal_diff = {:.3f}, mu_diff = {:.3f}, trial_ratio = {:.3f}, density_ratio = {:.3f}\n",pe_diff,mu[oldtype]-mu[newtype],trial_ratio,density_ratio);
  
  double pmove = density_ratio * species_ratio * trial_ratio * exp( -beta * e_diff);

  // accept
  if (random_equal->uniform() < pmove ) {
    nsgc_accepts++; 
    if (comm->me == 0) utils::logmesg(lmp,"e_diff = {:.3f}, SGCMC accepted ({}/{})\n",e_diff,nsgc_accepts,nsgc_trials);
    
    typecount[oldtype]--;
    typecount[newtype]++;
    
    store_pe();
  }
  //reject 
  else {
    if (comm->me == 0) utils::logmesg(lmp,"e_diff = {:.3f}, SGCMC rejected ({}/{})\n",e_diff,nsgc_accepts,nsgc_trials);
    //for (int i=0; i<(atom->nlocal+atom->nghost); i++) {
    for (int i=0; i<atom->nlocal; i++) {
      if (atom->tag[i] == idswap)  {
        atom->type[i] = oldtype;
      }
    }
    comm->forward_comm(this);
    recompute_peatom(0);
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal restored = {:.3f} stored = {:.3f}\n",pereal_restored,pereal_stored);
  }
}

void FixUMC::initiate_deletion()
{
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (revtest && reverse) {
    // reverse of a forward insertion: delete the SAME atom (idswap), no re-biasing
    int rtmp = oldtype; oldtype = newtype; newtype = rtmp;  // oldtype=real, newtype=fict
    if (comm->me == 0) utils::logmesg(lmp,"reverse deletion, idswap = {}, oldtype = {}, newtype = {}\n",idswap,oldtype,newtype);
    for (int i = 0; i < nlocal; i++)
      if (atom->tag[i] == idswap) type[i] = newtype;  // real -> fict type, stays in real group
    ndeletion_trials++;
    set_hybrid_lambda(1.0);   // start fully interacting (lambda=1), then fade out
    reneighbor();
    recompute_peatom(0);
    return;
  }

  int noldtype = 0;
  while (noldtype == 0) {
    oldtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
    noldtype = typecount[oldtype];
  }
  newtype = typegrid[0][oldtype];   // real oldtype -> its (0->oldtype) fict type

  if (comm->me == 0) utils::logmesg(lmp,"trial = {}, oldtype = {}, newtype = {}\n",trial,oldtype,newtype);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & fictbit)
      type[i] = newtype;
  }

  ndeletion_trials++;

  // bias with default H(x^N)
  bias_forward();

  // start from H(x^N+1)
  set_hybrid_lambda(1.0);

  // need to comm mask/type but new energy might not be necessary
  reneighbor();
  recompute_peatom(0);
}

void FixUMC::initiate_insertion()
{
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (revtest && reverse) {
    // reverse of a forward deletion: re-insert the SAME atom (idswap), no re-biasing
    int rtmp = oldtype; oldtype = newtype; newtype = rtmp;  // oldtype=fict, newtype=real
    if (comm->me == 0) utils::logmesg(lmp,"reverse insertion, idswap = {}, oldtype = {}, newtype = {}\n",idswap,oldtype,newtype);
    for (int i = 0; i < nlocal; i++)
      if (atom->tag[i] == idswap) { mask[i] |= groupbit; mask[i] &= ~fictbit; }  // back into real group, keeps fict type
    ninsertion_trials++;
    // lambda is already 0 (atom absent); grow it in as lambda 0->1
    reneighbor();
    recompute_peatom(0);
    return;
  }

  newtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
  oldtype = typegrid[0][newtype];   // real newtype -> its (0->newtype) fict type

  if (comm->me == 0) utils::logmesg(lmp,"trial = {}, oldtype = {}, newtype = {}\n",trial,oldtype,newtype);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & fictbit)
      type[i] = oldtype;
  }

  ninsertion_trials++;

  // bias with H(x^N+1), fictitious particles on
  set_hybrid_lambda(1.0);
  // turn on fictitious particles
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & fictbit) mask[i] |= groupbit;
  }

  reneighbor();
  recompute_peatom(0);
  bias_forward();

  // reneighbor() may reorder and reallocate the per-atom arrays.
  mask = atom->mask;
  type = atom->type;
  nlocal = atom->nlocal;

  // restore from H(x^N), turn off fictitious particles
  set_hybrid_lambda(0.0);
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & fictbit) mask[i] &= ~groupbit;
  }

  // restores energy/forces w/o fictitious particles
  reneighbor();
  recompute_peatom(VIRIAL_ATOM);
}

void FixUMC::initiate_sgcmc()
{
  nsgc_trials++;

  if (revtest && reverse) {
    // reverse leg: reuse the previous move's idswap, swap species, no re-biasing
    int tmp = oldtype; oldtype = newtype; newtype = tmp;
    bridgetype = typegrid[oldtype][newtype];   // (oldtype->newtype) sgc off-diagonal type
    if (comm->me == 0)
      utils::logmesg(lmp,"reverse gradual SGCMC, oldtype = {}, newtype = {}, bridgetype = {}, idswap = {}\n",
                     oldtype,newtype,bridgetype,idswap);
    for (int i = 0; i < atom->nlocal; i++)
      if (atom->tag[i] == idswap) atom->type[i] = bridgetype;
    comm->forward_comm(this);
    recompute_peatom(0);
    return;
  }

  int noldtype = 0;
  while (noldtype == 0) {
    oldtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
    noldtype = typecount[oldtype];
  }

  newtype = oldtype;
  while (newtype == oldtype) {
    newtype = swaptypes[(int) (nswaptypes * random_equal->uniform())];
  }

  bridgetype = typegrid[oldtype][newtype];   // (oldtype->newtype) sgc off-diagonal type

  if (comm->me == 0)
    utils::logmesg(lmp,"gradual SGCMC, oldtype = {} ({}), newtype = {} ({}), bridgetype = {}\n",
                   oldtype,typecount[oldtype],newtype,typecount[newtype],bridgetype);

  bias_forward();
  comm->forward_comm(this);
  recompute_peatom(0);
}


void FixUMC::bias_forward() {
  
  // variables
  double plocal_sum = 0;
  double Ei, pi;

  int *mask = atom->mask;
  int *type = atom->type;

  if (atom->nlocal > atom_swap_nmax) {
    atom_swap_nmax = atom->nlocal;
    memory->sfree(plocal);
    plocal = (double *) memory->smalloc(atom_swap_nmax * sizeof(double),"hmc:plocal");
  }
 
  for (int i = 0; i < atom->nlocal; i++) {
    if (trial == -1 &&
        (mask[i] & groupbit) && (mask[i] & movebit) && 
        type[i] == oldtype) 
    {
      Ei = c_peatom->vector_atom[i] - mu[oldtype];
      pi = exp(beta_delete * Ei);
      plocal_sum += pi;
      plocal[i] = pi;
    }
    else if (trial == 1 &&
           mask[i] & fictbit && 
           type[i] == oldtype) 
    {
      Ei = c_peatom->vector_atom[i] - mu[newtype];
      pi = exp(-beta_insert * Ei);
      plocal_sum += pi;
      plocal[i] = pi;
    }
    else if ((trial == 0 || trial == 2) &&
           (mask[i] & groupbit) && (mask[i] & movebit) && 
           type[i] == oldtype) 
    {
      Ei = c_peatom->vector_atom[i] - mu[oldtype];
      pi = exp(beta_sg * Ei);
      plocal_sum += pi;
      plocal[i] = pi;
    }
    // non-candidate: zero weight so the selection scan below can run condition-free
    else plocal[i] = 0.;
  }

  if (trial == 1) dump_pefict();
  
  double pglobal_sum, pbefore_sum;
  MPI_Allreduce(&plocal_sum,&pglobal_sum,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Scan(&plocal_sum,&pbefore_sum,1,MPI_DOUBLE,MPI_SUM,world);
  
  pbefore_sum -= plocal_sum;
  
  trial_forward = 0.;
  idswap = -1;
  zswap = 0.0;

  // find particle with probability p
  double pmove = pglobal_sum * random_equal->uniform();

  // if p points to this processor, scan atoms in weight-loop order;
  // plocal is zero for non-candidates so no per-branch conditions are needed
  if ((pmove >= pbefore_sum) && (pmove < pbefore_sum + plocal_sum)) {
    for (int i = 0; i < atom->nlocal; i++) {
      pbefore_sum += plocal[i];
      if (pbefore_sum > pmove) {
        idswap = atom->tag[i];
        trial_forward = plocal[i] / pglobal_sum;
        zswap = atom->x[i][2];

        utils::logmesg(lmp,"selected Ei = {:.3f} xi = {:.1f},{:.1f},{:.1f}\n",
                       c_peatom->vector_atom[i],atom->x[i][0],atom->x[i][1],atom->x[i][2]);

        // transform the selected atom
        if (trial == -1) type[i] = newtype;        // fict type, stays in real group
        else if (trial == 1) mask[i] &= ~fictbit;  // keeps fict type, joins real group
        else if (trial == 2) type[i] = bridgetype;
        else type[i] = newtype;                    // trial == 0: instant semi-grand
        break;
      }
    }
  }

  struct{ double trial; int id; } sendbuff, recvbuff;
  sendbuff.trial = trial_forward;
  sendbuff.id = idswap;

  MPI_Allreduce(&sendbuff,&recvbuff,1,MPI_DOUBLE_INT,MPI_MAXLOC,world);

  trial_forward = recvbuff.trial;
  idswap = recvbuff.id;

  // no atom selected: the weights underflowed to 0 or overflowed to inf, so the scan above never
  // fired.  Every caller would then transform nothing and the move would be a silent no-op.
  if (idswap < 0)
    error->all(FLERR, "fix UMC selection bias picked no atom (trial {}, oldtype {})", trial, oldtype);
  double zlocal = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->tag[i] == idswap) {
      zlocal = atom->x[i][2];
      break;
    }
  }
  MPI_Allreduce(&zlocal,&zswap,1,MPI_DOUBLE,MPI_SUM,world);
  
  if (comm->me == 0) utils::logmesg(lmp,"trial_forward = {:.8f}/{:.8f} = {:.8f}, idswap = {}, zswap = {:.6f}\n",trial_forward*pglobal_sum,pglobal_sum,trial_forward,idswap,zswap);
}

// debug dump for fictitious-particle PE when bias_forward/reverse has just
// computed the corresponding per-atom energies
void FixUMC::dump_pefict()
{
  int naccepts = ninsertion_accepts + ndeletion_accepts + nmd_accepts + nsgc_accepts +
    niso_accepts + naniso_accepts + nxycap_accepts + nzcap_accepts;
  if (imcdump < 0 || naccepts <= npefict_dumps) return;
  // every_dump is 0 when the dump's frequency is variable-driven (dump_modify every v_foo),
  // which would make the modulus below an integer divide by zero -> SIGFPE
  int every = output->every_dump[imcdump];
  if (every <= 0 || naccepts % every || output->dump[imcdump]->igroup != 0) return;

  // only fill in a step Output would miss -- see end_of_step for why we must not set last_dump
  if (output->next_dump[imcdump] != update->ntimestep) {
    modify->clearstep_compute();
    output->dump[imcdump]->write();
    modify->addstep_compute(output->next_dump[imcdump]);
  }
  npefict_dumps = naccepts;
}

    
// hybrid->scaleval = [0,1]
// the transforming particle is in real group with a fictitious or bridge type
void FixUMC::bias_reverse() {
  
  // reallocate list of swaptypes atoms if needed
  if (atom->nlocal > atom_swap_nmax) {
    atom_swap_nmax = atom->nlocal;
    memory->sfree(plocal);
    plocal = (double *) memory->smalloc(atom_swap_nmax * sizeof(double),"hmc:plocal");
  }
  
  double plocal_sum = 0, plocal_reverse = 0;
  double Ei, pi;

  int *mask = atom->mask;
  int *type = atom->type;
 
  // include transforming particle: real group, fict type
  for (int i = 0; i < atom->nlocal; i++) {
    // insertion: reversal is deletion
    // transforming particle still has fict type
    if (trial == 1 && 
        (mask[i] & groupbit) && 
        (mask[i] & movebit) &&
        (type[i] == oldtype || type[i] == newtype) ) 
    {
      Ei = c_peatom->vector_atom[i] - mu[newtype];
      pi = exp(beta_delete * Ei );
      plocal_sum += pi;
      plocal[i] = pi;
    }
    else if (trial == -1 &&
            mask[i] & groupbit &&
            type[i] == newtype) 
    {
      Ei = c_peatom->vector_atom[i] - mu[oldtype];
      pi = exp(-beta_insert*Ei );
      plocal_sum += pi;
      plocal[i] = pi;
    }
    else if ((trial == 0 || trial == 2) && 
             mask[i] & groupbit && 
             mask[i] & movebit && 
             (type[i] == newtype || type[i] == bridgetype))
    {
      Ei = c_peatom->vector_atom[i] - mu[newtype];
      pi = exp( beta_sg * Ei );
      plocal_sum += pi;
      plocal[i] = pi;
    }

    // non-candidate: keep plocal defined so the idswap read below is safe
    else plocal[i] = 0.;

    // find p_i
    if (atom->tag[i] == idswap) {
      plocal_reverse = plocal[i];
    }
  }

  if (trial == -1) dump_pefict();
  
  double pglobal_reverse, pglobal_sum;
  MPI_Allreduce(&plocal_reverse,&pglobal_reverse,1,MPI_DOUBLE,MPI_MAX,world); // should change to one-to-all
  MPI_Allreduce(&plocal_sum,&pglobal_sum,1,MPI_DOUBLE,MPI_SUM,world);

  trial_reverse = pglobal_reverse/pglobal_sum;
  
  if (comm->me == 0) utils::logmesg(lmp,"trial_reverse = {:.8f}/{:.8f} = {:.8f}\n",pglobal_reverse,pglobal_sum,trial_reverse);

}

void FixUMC::end_of_step()
{
  if (update->ntimestep != laststep) return;
  
  if (comm->me == 0) utils::logmesg(lmp,"t = {}, evaluating trial = {}\n",update->ntimestep,trial);

  // energy should be up to date

  if (update->eflag_atom != update->ntimestep) {
    error->warning(FLERR,"Atomic energies not computed. Overwritten by external output (step {})? Recomputing.",update->eflag_atom);
    recompute_peatom(VIRIAL_ATOM);
  }

  double pereal_after = c_pereal->compute_scalar();
  double pereal_diff = pereal_after - pereal_stored;
  double e_diff = pereal_diff;
  
  c_keatom->compute_peratom();
  double kereal_after = c_kereal->compute_scalar();
  double kereal_diff = kereal_after - kereal_stored;

  double density_ratio = 1.;
  double trial_ratio = 1.;
  double vbias_ratio = 1.;   // velocity-bias proposal ratio (1 = no bias)
  double species_ratio = 1.; // species-pick proposal ratio (1 unless a swap species is empty)

  if (trial != 0) {
    
    double volume;
    if (cap) volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
    else volume = domain->xprd * domain->yprd * domain->zprd;

    // deletion/sgc re-draw the species until non-empty (pick prob 1/npresent)
    // while insertion picks over all nswaptypes; the asymmetry must enter the
    // acceptance whenever a swap species is empty on either side of the move
    int npresent = 0;
    for (int i = 0; i < nswaptypes; i++)
      if (typecount[swaptypes[i]] > 0) npresent++;
   
    // deletion -> reverse is insertion, need to turn on fictitious particles
    // must loop over all of nlocal: fict atoms sit beyond nfirst (atom_modify first)
    if (trial == -1) {
      
      set_hybrid_lambda(1.0);
      // velocity bias needs forces w/ lambda=1, no fict
      if (tvbias > 0.0) {
        recompute_peatom(0);
        vbias_ratio = exp(bias_logfwd + velocity_bias(0));
      }

      // then for reverse selection bias
      // deleted -> fict
      // all fict -> group
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->tag[i] == idswap) atom->mask[i] |= fictbit;
        if (atom->mask[i] & fictbit) atom->mask[i] |= groupbit;
      }

      reneighbor();
      recompute_peatom(0);
      bias_reverse();

      e_diff += mu[oldtype];
      species_ratio = (double) npresent / nswaptypes;
      density_ratio = (typecount[0]+1) * lambda3[oldtype] / volume; // (M-N)+1 / V/L^3 
    }
    // insertion: already hybrid->scaleval = [0,1], which is correct for bias_reverse
    else if (trial == 1) {
      // insertion -> reverse is deletion
      // swap atom in real group so ready to reverse
      bias_reverse();
      
      e_diff -= mu[newtype];
      species_ratio = (double) nswaptypes / (npresent + (typecount[newtype] == 0 ? 1 : 0));
      density_ratio = volume / lambda3[newtype] / typecount[0]; // V/L^3 / (M-N)
    }
    else if (trial == 2) {
      bias_reverse();

      e_diff += mu[oldtype] - mu[newtype];
      species_ratio = (double) npresent /
        (npresent - (typecount[oldtype] == 1 ? 1 : 0) + (typecount[newtype] == 0 ? 1 : 0));
      density_ratio = lambda3[oldtype] / lambda3[newtype];
    }

    // Insertion and gradual sgc take the reverse velocity-bias draw identically: the reverse
    // move's forces at lambda = 0 -- the atom removed for an insertion, the bridge type read as
    // oldtype for an sgc -- then restore lambda = 1.  Deletion is deliberately NOT here: it has
    // to draw before the fict group is switched on, so it stays inline in its branch above.
    if (tvbias > 0.0 && trial != -1) {
      set_hybrid_lambda(0.0);
      recompute_peatom(0);
      vbias_ratio = exp(bias_logfwd + velocity_bias(0));
      set_hybrid_lambda(1.0);
      recompute_peatom(0);
    }

    trial_ratio = trial_reverse / trial_forward;

    if (comm->me == 0) utils::logmesg(lmp,"density_ratio = {:.3f}, trial_ratio = {:.6f}, vbias_ratio = {:.4e}\n",density_ratio,trial_ratio,vbias_ratio);
  }
 
  if (comm->me == 0) utils::logmesg(lmp,"e_diff = {:.3f}, ke_diff = {:.3f}, total_diff = {:.3f}\n",e_diff,kereal_diff,e_diff+kereal_diff);
  
  double pmove = density_ratio * species_ratio * trial_ratio * vbias_ratio
    * exp(-beta_md * kereal_diff - beta * e_diff);
 
  // accept move
  if (random_equal->uniform() < pmove || revtest) {
    // displacement
    if (trial == 0) {
      nmd_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"MD CMC accepted ({}/{})\n",nmd_accepts,nmd_trials);
    }
    // deletion
    else if (trial == -1) {
      ndeletion_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"deletion GCMC accepted ({}/{})\n",ndeletion_accepts,ndeletion_trials);
      typecount[0]++;
      typecount[oldtype]--;

      for (int i=0; i<atom->nlocal; i++) {
        // add deleted particle to fict group
        if (atom->tag[i] == idswap) {
          atom->mask[i] |= fictbit;
        }
        // turn off vacancies, left on by bias_reverse
        // also remove deleted particle from real group
        if (atom->mask[i] & fictbit) atom->mask[i] &= ~groupbit;
      }
      // last energies from bias_reverse with vacancies
    }
    // insertion
    else if (trial == 1) {
      ninsertion_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"insertion GCMC accepted ({}/{})\n",ninsertion_accepts,ninsertion_trials);
      typecount[0]--;
      typecount[newtype]++;

      // give idswap real type; groups should be correct from bias_reverse
      for (int i=0; i<atom->nlocal; i++) {
        if (atom->tag[i] == idswap) {
          atom->type[i] = newtype;
          break;
        }
      }

      // report the tracked counts: group->count() is an O(N) scan plus an MPI_Allreduce, and two
      // of them ran on every accepted insertion just to fill this line.  The species that grew and
      // what is left in the reservoir are the useful numbers here, and both are O(1).  Note this is
      // now a report rather than a cross-check -- it reads the counters incremented just above, so
      // it can no longer catch typecount drifting away from actual group membership.
      if (comm->me == 0)
        utils::logmesg(lmp,"N[{}] = {}, fict = {}\n",newtype,typecount[newtype],typecount[0]);
      // energies should be current but need to reneighbor w/ new type
      // unless I can just change types for < nmax
      // scaleval = [0,1] from trajectory, however
    }
    else if (trial == 2) {
      nsgc_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"gradual SGCMC accepted ({}/{})\n",nsgc_accepts,nsgc_trials);
      typecount[oldtype]--;
      typecount[newtype]++;

      for (int i=0; i<atom->nlocal; i++) {
        if (atom->tag[i] == idswap) {
          atom->type[i] = newtype;
          break;
        }
      }
    }
    // store new states on step we calculated energy
    if (pair_style_name && trial != 0) {
      set_hybrid_lambda(0.0);
    }
    reneighbor();
    recompute_peatom(VIRIAL_ATOM);
    store_pe();
    store_atom();

    // dump output: write the snapshot directly, the way dump_pefict() does, instead of reaching
    // into Output's schedule.  The arming step used to force next_dump[imcdump] = 0, which
    // propagates into next_dump_any and then into output->next -- the value Verlet compares
    // against ntimestep -- so once it reached 0 no dump, no thermo and no addstep_compute() ran
    // again until an accepted move happened to revive it.
    int naccepts = ninsertion_accepts + ndeletion_accepts;
    if (!onlydumpgc) naccepts += nmd_accepts + nsgc_accepts + niso_accepts + naniso_accepts + nxycap_accepts + nzcap_accepts;
    // every_dump <= 0 (variable-driven dump frequency) would make this modulus a divide by zero
    int every = (imcdump > -1) ? output->every_dump[imcdump] : 0;
    if (every > 0 && naccepts % every == 0 && (!onlydumpgc || trial != 0) &&
        output->next_dump[imcdump] != update->ntimestep) {
      // Only fill in the steps Output would miss; when its own schedule lands here too, let it do
      // the write.  We cannot mark one ourselves: calculate_next_dump() is private, so setting
      // last_dump would send Output down its `continue` path, which skips BOTH the schedule
      // advance and the next_dump_any = MIN(...) below it -- dropping this dump for good.
      // clearstep/addstep bracket the write the same way Output brackets its own.
      modify->clearstep_compute();
      output->dump[imcdump]->write();
      modify->addstep_compute(output->next_dump[imcdump]);
    }
  }
  // reject move
  else {
    if (comm->me == 0) utils::logmesg(lmp,"Hamiltonian trajectory rejected\n");
    restore_atom();
    if (pair_style_name && trial != 0) {
      set_hybrid_lambda(0.0);
    }
    // need to recompute neighbor lists entirely
    reneighbor();
    recompute_peatom(VIRIAL_ATOM);
    
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal restored = {:.3f} stored = {:.3f}\n",pereal_restored,pereal_stored);
  }

  print();
}

void FixUMC::reneighbor() {
  if (modify->n_pre_exchange) {
    timer->stamp();
    modify->pre_exchange();
    timer->stamp(Timer::MODIFY);
  }
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  // mirrors Verlet::run's reneighbor block.  domain->box_change is only nonzero because the ctor
  // declares box_change for whichever box moves are enabled -- without that declaration this is
  // dead code and the bins/comm cutoffs stay pinned to the box as it was at setup().
  if (domain->box_change) {
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
  }
  timer->stamp();
  comm->exchange();
  if (atom->sortfreq == 1) atom->sort(); // only sort if would every step
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  timer->stamp(Timer::COMM);
  if (modify->n_pre_neighbor) {
    modify->pre_neighbor();
    timer->stamp(Timer::MODIFY);
  }
  neighbor->build(1);
  timer->stamp(Timer::NEIGH);
  if (modify->n_post_neighbor) {
    modify->post_neighbor();
    timer->stamp(Timer::MODIFY);
  }
}

void FixUMC::recompute_peatom(int vflag)
{
  // eflag/vflag usually set based on computes in protected integrate::ev_set(timestep)
  int eflag = ENERGY_ATOM;

  // update-> flags tell compute that energies/virials will be current
  update->eflag_atom = update->ntimestep;
  if (vflag) update->vflag_atom = update->ntimestep;

  update->integrate->force_clear();

  timer->stamp();

  if (modify->n_pre_force) {
    modify->pre_force(vflag);
    timer->stamp(Timer::MODIFY);
  }
  force->pair->compute(eflag,vflag);
  timer->stamp(Timer::PAIR);
 
  if (modify->n_pre_reverse) {
    modify->pre_reverse(eflag,vflag);
    timer->stamp(Timer::MODIFY);
  }
  if (force->newton) {
    comm->reverse_comm();
    timer->stamp(Timer::COMM);
  }
 
  if (modify->n_post_force_any) modify->post_force(vflag);
  timer->stamp(Timer::MODIFY);

  c_peatom->compute_peratom();

  // capture the virial whenever it is actually tallied -- the only moment compute stress/atom
  // is valid.  MC_P/MC_Pz read these (compute_array), so they hold the last tallied value
  // across a burst of instant moves instead of erroring on a stale per-atom virial.
  if (vflag) {
    c_preal->compute_vector();
    virial_stored  = c_preal->vector[0] + c_preal->vector[1] + c_preal->vector[2];
    virialz_stored = c_preal->vector[2];
  }
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */

double FixUMC::compute_vector(int j)
{
  if (j == 0) return nmd_trials;
  if (j == 1) return nmd_accepts;
  if (j == 2) return ndeletion_trials + ninsertion_trials;
  if (j == 3) return ndeletion_accepts + ninsertion_accepts;
  if (j == 4) return nsgc_trials;
  if (j == 5) return nsgc_accepts;
  if (j == 6) return niso_trials + naniso_trials + nxycap_trials + nzcap_trials;
  if (j == 7) return niso_accepts + naniso_accepts + nxycap_accepts + nzcap_accepts;
  return 0.0;
}

/* ----------------------------------------------------------------------
  per-species state: row i = real species i+1, col 0 = N, col 1 = mu.
  read live on every variable evaluation -- see the MC_N<i>/MC_mu<i> definitions
  in setup().  mu is the value latched for the current trial; a NULL species
  reports its mu as NaN, exactly as it is stored.
------------------------------------------------------------------------- */

double FixUMC::compute_array(int i, int j)
{
  if (!typecount) return 0.0;   // the array exists from the ctor, typecount only from init()
  if (j == 0) return (double) typecount[i+1];
  if (j == 1) return mu[i+1];
  if (j == 2) return pereal_stored;
  if (j == 3) return virial_stored;   // trace, for MC_P
  if (j == 4) return virialz_stored;  // zz, for MC_Pz
  if (j == 5) {                       // real atoms, the ideal-gas term of MC_P/MC_Pz
    bigint n = 0;
    for (int itype = 1; itype <= nrealtypes; itype++) n += typecount[itype];
    return (double) n;
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixUMC::memory_usage()
{
  // plocal (selection weights, atom_swap_nmax) plus the stored configuration the
  // MD leg is rolled back to (nmax_stored)
  double bytes = (double) atom_swap_nmax * sizeof(double);
  bytes += (double) nmax_stored * (3*sizeof(double) + 2*sizeof(int) + sizeof(tagint) + sizeof(imageint));
  return bytes;
}

// from fix atom/swap
int FixUMC::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

void FixUMC::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int>(buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++) type[i] = static_cast<int>(buf[m++]);
  }
}
