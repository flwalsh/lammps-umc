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

#include "fix_hgcmc.h"
#include "angle.h"
#include "atom.h"
#include "input.h" // for variable?
#include "integrate.h" // for ev_set
#include "variable.h" // for variable?
#include "output.h" // for write_dump
#include "dump.h" // for id
#include "fix_print.h" // used for output
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_nh.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_hybrid_scaled.h"
#include "random_park.h"
#include "region.h"
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

FixHGCMC::FixHGCMC(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), ref_region(nullptr), ref_region_id(nullptr),
    mu(nullptr), swaptypes(nullptr), typecount(nullptr), lambda3(nullptr),
    groupname(nullptr), fictgroupname(nullptr), caplogroupname(nullptr), caphigroupname(nullptr),
    qtype(nullptr), sqrt_mass_ratio(nullptr),
    plocal(nullptr),
    random_equal(nullptr), random_unequal(nullptr),
    c_keatom(nullptr), c_peatom(nullptr), c_pereal(nullptr), c_pecaphi(nullptr), c_pecaplo(nullptr),
    nrefs(nullptr), nrefs_local(nullptr),
    nreal_diff(nullptr),
    tag_stored(nullptr), type_stored(nullptr), mask_stored(nullptr), image_stored(nullptr), x_stored(nullptr),
    mcdumpid(nullptr), mcprintid(nullptr), mcprint(nullptr)
{
  // ID group-ID gcmc seed temp [ up to 3x: [move] [prob] [nsteps] ]
  if (narg < 3) error->all(FLERR, "Illegal fix hgcmc command nargs");

  dynamic_group_allow = 1;
  time_integrate = 1;

  vector_flag = 1;
  size_vector = 8; // for MC statistics outputs
  //size_array_rows = 2;
  //size_array_cols = MAX(nrealtypes,7); //
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  //time_depend = 1; // I don't think this is necsesary
 
  // 1-indexed types
  // this is not good for sgcmc!
  nrealtypes = (int) atom->ntypes/2;
  memory->create(mu, nrealtypes+1, "hgcmc:mu"); // +1 b/c 1-indexed
  memory->create(swaptypes, nrealtypes, "hgcmc:swaptypes"); // 0-indexed

  // fix: igroup, groupbit
  groupname = utils::strdup(arg[1]);
  // what gets velocities applied, real + fict if gcmc
  movebit = groupbit;
            
  double temperature = utils::numeric(FLERR, arg[3], false, lmp);
  int seed = utils::inumeric(FLERR, arg[4], false, lmp);
  nmdsteps = utils::inumeric(FLERR, arg[5], false, lmp);

  if (seed <= 0) error->all(FLERR, "Illegal fix hgcmc seed");
  if (temperature <= 0.0) error->all(FLERR, "Illegal fix hgcmc temperature");

  beta = 1.0 / (force->boltz * temperature);
  beta_insert = beta;
  beta_delete = beta;
  beta_md = beta;

  options(narg - 6, &arg[6]);
   
  // create include group for calculations, though not used unless GCMC
  if (cap == 1) group->assign(fmt::format("MC_include union {} {} {}",groupname,caplogroupname,caphigroupname));
  else group->assign(fmt::format("MC_include union {}",groupname));
  iincludegroup = group->find("MC_include");
    
  // lots of extra initialization for GCMC
  if (fictgroupname) {
    // fict group settings
    ifictgroup = group->find(fictgroupname);
    fictgroupbit = group->bitmask[ifictgroup];
    movebit |= fictgroupbit; // integrate fict positions, even if not part of real group

    // include group settings
    if (strcmp(groupname,"all") == 0) error->all(FLERR, "HGCMC fix group cannot be 'all' with fictitious particles.");
    if (atom->firstgroupname) error->warning(FLERR, "HGCMC must replace {} as include group for internal use.",atom->firstgroupname);

    input->one("atom_modify first MC_include");
    input->one("neigh_modify include MC_include");
    input->one(fmt::format("neigh_modify exclude group {} {}",fictgroupname,fictgroupname));
 
    // dual Hamiltonian/interatomic potential settings
    // TODO add check here
    hybrid = (PairHybridScaled *)force->pair;
    
    // somewhat arbitrarily, make fict mass heaviest real mass
    fictmass = 0.;
    for (int itype=1; itype<=nrealtypes; itype++) {
      if (atom->mass[itype] > fictmass) fictmass = atom->mass[itype];
    }
  }
  
  // create \Lambda^3 for each species
  if (pgcmc > 0. || psgcmc > 0.) {
    memory->create(lambda3, nrealtypes+1, "hgcmc:lambda3"); // +1 b/c 1-indexed
    double lambda;
    for (int itype=1; itype<=nrealtypes; itype++) {
      lambda = sqrt( force->hplanck * force->hplanck * beta / (2.0*MathConst::MY_PI * atom->mass[itype] * force->mvv2e) );
      lambda3[itype] = lambda * lambda * lambda;
      if (comm->me == 0) utils::logmesg(lmp,"itype = {}, mass = {}, lambda = {}, lambda3 = {}\n",itype,atom->mass[itype],lambda,lambda3[itype]);
    }
  }
 
  // create type counter - easier to track than repeatedly invoking compute->count_types
  memory->create(typecount, nrealtypes+1, "hgcmc:typecount"); // +1 b/c 0 is fict
  // create and zero out local counter
  bigint *typecountlocal;
  memory->create(typecountlocal, nrealtypes+1, "hgcmc:typecountlocal"); // +1 b/c 0 is fict
  for (int itype=0; itype<=nrealtypes; itype++) {
    typecountlocal[itype] = 0;
  }
  // count types
  int itype;
  for (int i=0; i<atom->nlocal; i++) {
    if (atom->mask[i] & movebit) { // count both real and fict
      itype = atom->type[i];
      if (itype > nrealtypes) typecountlocal[0]++;
      else typecountlocal[itype]++;
    }
  }
  MPI_Allreduce(typecountlocal,typecount,nrealtypes+1,MPI_LMP_BIGINT,MPI_SUM,world);
  memory->destroy(typecountlocal);
    
  for (int itype=0; itype<=nrealtypes; itype++) {
    if (comm->me == 0) utils::logmesg(lmp,"itype = {}, typecount = {}\n",itype,typecount[itype]);
  }

  // create computes for real group (fix group)
  c_keatom = modify->add_compute("keatom all ke/atom");
  c_kereal = modify->add_compute(fmt::format("kereal {} reduce sum c_keatom",groupname));
  
  c_peatom = modify->add_compute("peatom all pe/atom");
  c_pereal = modify->add_compute(fmt::format("pereal {} reduce sum c_peatom",groupname));

  if (cap) {
    c_pecaphi = modify->add_compute(fmt::format("pecaphi {} reduce sum c_peatom",caphigroupname));
    c_pecaplo = modify->add_compute(fmt::format("pecaplo {} reduce sum c_peatom",caplogroupname));
  }
  
  c_patom  = modify->add_compute(fmt::format("patom {} stress/atom NULL pair",groupname));
  c_preal  = modify->add_compute(fmt::format("preal {} reduce sum c_patom[1] c_patom[2] c_patom[3]",groupname));
  
  Compute *c_MC_Ns = modify->add_compute(fmt::format("MC_Ns {} count/type atom",groupname));
  
  // note computes must be up here

  // random number generator, same for all procs
  random_equal = new RanPark(lmp, seed);
  random_unequal = new RanPark(lmp, seed+comm->me);

  nmax_stored = 0;
  atom_swap_nmax = 0;
  
  plocal = nullptr;
  x_stored = nullptr;
  image_stored = nullptr;
  mask_stored = nullptr;
  tag_stored = nullptr;
  type_stored = nullptr;

  // could be useful in future but not need yet
  // was messing things up maybe?
  //force_reneighbor = 1;
  //next_reneighbor = update->ntimestep; // placeholder
  laststep = update->ntimestep;

  // set comm size needed by this Fix
  if (atom->q_flag) error->all(FLERR, "Charge not yet supported by hgcmc.");
  if (atom->q_flag) comm_forward = 2;
  else comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixHGCMC::~FixHGCMC()
{
  /*
  memory->destroy(nreal_diff);
  memory->destroy(nrefs_local);
  memory->destroy(nrefs);
  delete[] ref_region_id;
  */

  // causes segfault, maybe should undo manually
  //input->one("atom_modify first all");
  //input->one("neigh_modify include all");
  //input->one("group MC_include delete");

  delete[] groupname;
  if (gcmc != 0) delete[] fictgroupname;

  memory->destroy(mu);
  memory->destroy(swaptypes);
  memory->destroy(typecount);
  memory->destroy(lambda3);
  //memory->destroy(qtype);
 
  delete random_equal;
  delete random_unequal;

  memory->sfree(image_stored);
  memory->sfree(mask_stored);
  memory->sfree(tag_stored);
  memory->sfree(type_stored);
  memory->sfree(x_stored);

  modify->delete_compute("keatom");
  modify->delete_compute("kereal");
  modify->delete_compute("peatom"); 
  modify->delete_compute("pereal");
  modify->delete_compute("patom");
  modify->delete_compute("preal");
  modify->delete_compute("MC_Ns");
  
  if (cap) {
    delete[] caphigroupname;
    delete[] caplogroupname;
    modify->delete_compute("pecaphi");
    modify->delete_compute("pecaplo") ;
  }

  if (mcdumpid) delete[] mcdumpid;
  if (mcprintid) delete[] mcprintid;
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */
void FixHGCMC::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix hgcmc command");

  pgcmc = 0.; psgcmc = 0; pxycap = 0; pzcap = 0.; piso = 0;
  nswaptypes = 0;
  
  revtest = 0;
  realpot = 0;
  imcdump = -1;
  onlydumpgc = 0; // should be set as option
  cap = 0;
  press = 0.;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + nrealtypes + 1 > narg) error->all(FLERR, "Illegal mu arg length in fix_hgcmc");
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
    else if (strcmp(arg[iarg], "gcmc") == 0) {
      if (iarg + 6 > narg) error->all(FLERR, "Illegal gcmc in fix_hgcmc");
      fictgroupname = utils::strdup(arg[iarg+1]);
      pgcmc = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      ngcmcsteps = utils::inumeric(FLERR, arg[iarg+3], false, lmp);
      double T_insert = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      beta_insert = 1. / (force->boltz * T_insert);
      double T_delete = utils::numeric(FLERR, arg[iarg+5], false, lmp);
      beta_delete = 1. / (force->boltz * T_delete);
      //if (comm->me == 0) utils::logmesg(lmp,"nswaptypes = {}, T_ins = {}, T_del = {}\n",nswaptypes,T_insert,T_delete);
      iarg += 6;
    }
    else if (strcmp(arg[iarg], "sgcmc") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal sgcmc in fix_hgcmc");
      psgcmc = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      nsgcmc = utils::inumeric(FLERR, arg[iarg+2], false, lmp);
      iarg += 3;
    }
    else if (strcmp(arg[iarg], "cap") == 0) {
      if (iarg + 8 > narg) error->all(FLERR, "Illegal cap in fix_hgcmc");
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
      if (iarg + 4 > narg) error->all(FLERR, "Illegal volume in fix_hgcmc");
      volume = 1;
      piso   = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      isomax = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      press  = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "print") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix hgcmc print command");
      mcprintid = utils::strdup(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "dump") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix hgcmc dump command");
      mcdumpid = utils::strdup(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "revtest") == 0) {
      revtest = 1;
      iarg++;
    }
    else if (strcmp(arg[iarg], "realpot") == 0) {
      realpot = 1;
      iarg++;
    }
    else {
      if (comm->me == 0) utils::logmesg(lmp,"args = {}",arg[iarg]); 
      error->all(FLERR, "Illegal fix_hgcmc command (else)");
    }
  }
      
  if ((pgcmc > 0. || psgcmc > 0.) && nswaptypes == 0) error->all(FLERR, "Illegal gcmc in fix_hgcmc; mu must be defined");
 
  pmd = 1. - pgcmc - psgcmc - pxycap - pzcap - piso;
  if (comm->me == 0) utils::logmesg(lmp,"pmd = {}, sgcmc = {}, pgcmc = {}, pxycap = {}, pzcap = {}, piso = {} \n",pmd,pgcmc,psgcmc,pxycap,pzcap,piso);
  // probably need to make this floating-point tolerant
  if (pgcmc < 0. || psgcmc < 0. || pmd < 0. || pxycap < 0. || pzcap < 0. ) error->all(FLERR, "Illegal move probabilities < 0");
}

/* ---------------------------------------------------------------------- */

int FixHGCMC::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE; // NVE
  mask |= POST_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

// before each MD run: set internal flags
void FixHGCMC::init()
{
  // so modify->setup calculates energies that can be stored
  update->eflag_global = update->ntimestep;
  update->eflag_atom = update->ntimestep;
  
  // from FixNVE 
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
   
  // so neigh->setup() computes atomic energies
  // doesn't seem to work though 
  if (fictgroupname) {
    hybrid->scaleval[0] = 1;
    hybrid->scaleval[1] = 0;
  }
 
  // from FixAtomSwap::init
  double **cutsq = force->pair->cutsq;
  bool unequal_cutoffs = false;
  for (int itype = 1; itype <= nrealtypes; itype++)
    for (int jtype = 1; itype <= nrealtypes; itype++)
      for (int ktype = 1; ktype <= nrealtypes; ktype++)
        if (cutsq[itype][ktype] != cutsq[jtype][ktype]) {
          unequal_cutoffs = true;
          error->all(FLERR, "Unequal cutoffs not yet supported");
      }
}

// before each MD run: calculate forces, etc.
void FixHGCMC::setup(int flag)
{
  nmd_trials = nmd_accepts = ninsertion_trials = ninsertion_accepts = ndeletion_trials = ndeletion_accepts = nsgc_trials = nsgc_accepts  = niso_trials = niso_accepts = nxycap_trials = nxycap_accepts = nzcap_trials = nzcap_accepts = 0;
  
  if (!atom->mass) error->all(FLERR, "Fix hgcmc requires per atom type masses");
  if (atom->rmass_flag && (comm->me == 0))
    error->warning(FLERR, "Fix hgcmc will use per-type masses for velocity rescaling");

  // This should be moved up to somewhere else
  // TODO error if fictgroup has atoms with non-fict types
  // TODO error if pair_style not hybrid

  // if GCMC
  if (fictgroupname) {
    hybrid->scaleval[0] = 1;
    hybrid->scaleval[1] = 0;
  }
  
  if (cap == 1) {
    double zcaplo_local = domain->boxlo[2]; // placeholder values
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
  if (cap == 1) volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
  else volume = domain->xprd * domain->yprd * domain->zprd;
  input->one("variable MC_V internal 0.");
  i_MC_V = input->variable->find("MC_V");
  input->variable->internal_set(i_MC_V,volume);
  
  input->one("variable MC_Ns vector c_MC_Ns");
 
  input->one(fmt::format("variable MC_P equal -(c_preal[1]+c_preal[2]+c_preal[3])/(3*v_MC_V)+sum(v_MC_Ns)/(v_MC_V)*{}",force->nktv2p/beta));
  input->one(fmt::format("variable MC_Pz equal -(_preal[3])/(v_MC_V)+sum(v_MC_Ns)/(v_MC_V)*{}",force->nktv2p/beta));
  //input->one(fmt::format("variable MC_Pz equal -c_preal[3]/(v_MC_V)+sum(v_MC_Ns)/(v_MC_V)*{}",force->nktv2p/beta));

  // setup printing
  if (mcprintid) { // not nullptr
    if (comm->me == 0) utils::logmesg(lmp,"looking up print id {}\n",mcprintid);
    mcprint = (FixPrint *)modify->get_fix_by_id(mcprintid);
    if (comm->me == 0) utils::logmesg(lmp,"mcprint id = {} nevery = {}\n",mcprint->id,mcprint->nevery);
  }
  
  // setup dumping
  if (mcdumpid) {
    // get dump index; not sure if better way
    for (int idump=0; idump<output->ndump; idump++) { 
      if (strcmp(mcdumpid,output->dump[idump]->id) == 0) imcdump = idump; break;
    }
    if (imcdump == -1) error->all(FLERR,"MCDUMP index not found");
    if (comm->me == 0) utils::logmesg(lmp,"dump[{}] = {}, ndump = {}\n",imcdump,output->dump[imcdump]->id,output->ndump);
  }

  //recompute_peatom(1);
  // storing before next move so on same step as energy computed
  store_pe();
  store_atom();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void FixHGCMC::initial_integrate(int /*vflag*/)
{
  bigint step = update->ntimestep;

  // time for a new move
  if (step == laststep + 1) {
    // initially try non-MD moves within loop
    double p = -1.;
    gcmc = 0;
    while (p < pxycap + pzcap + piso + psgcmc) {
      p = random_equal->uniform();
      if (comm->me == 0) utils::logmesg(lmp,"\nt = {}, p = {}\n",step,p);

      if (p < pxycap) trial_xycap();
      else if (p < pxycap + pzcap) trial_zcap();
      else if (p < pxycap + pzcap + piso) trial_iso();
      else if (p < pxycap + pzcap + piso + psgcmc) {
        for (int n=0; n<nsgcmc; n++) trial_sgcmc();
        // we don't need to store or recompute within sgcmc rejections
        reneighbor();
        recompute_peatom();
        store_atom();
      }
      // so we can exit loop if just doing sgcmc
      if (step == update->laststep) break;
    }
      
    if (revtest && (step - 1) % (2*ngcmcsteps) == ngcmcsteps) {
      if (comm->me == 0) utils::logmesg(lmp,"performing reverse for revtest\n");
      gcmc *= -1;
      initiate_gcmc();
      laststep = step + ngcmcsteps-1;
    }
    else if ((p -= pxycap + pzcap + piso + psgcmc) < pmd) {
      laststep = step + nmdsteps - 1;
      nmd_trials++;
    }
    else if ((p -= pmd) < pgcmc) {
      if (p < pgcmc/2) gcmc = 1;
      else gcmc = -1;
      initiate_gcmc();
      laststep = step + ngcmcsteps-1;
    }
    else error->all(FLERR, "Move selection probability messed up");
   
    //reneighbor();
    //recompute_peatom();
    //double pereal0 = c_pereal->compute_scalar();
    //if (comm->me == 0) utils::logmesg(lmp,"new pereal = {}\n",pereal0);

    initiate_velocities();
    store_ke();
 
    // this is very much necssary
    modify->clearstep_compute(); // sets invoked flag to 0
    c_peatom->addstep(laststep);
    
    if (mcprint != nullptr) {
      modify->addstep_compute_all(laststep);
      mcprint->next_print = laststep;
    }
 
    if (imcdump > -1) output->next_dump[imcdump] = 0; // only changed if accept
  }
  // based on FixPrint, note that this breaks thermo, and thus revtest

  // update insertion/deletion Hamiltonian
  // Verlet needs a_f = a_0, a_f-1 = a_1, etc.
  if (gcmc != 0) {
    hybrid->scaleval[0] -= dfrac; // H(x^N)
    hybrid->scaleval[1] += dfrac; // H(x^N+1)
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
    if (mask[i] & groupbit) {
      dtfm = dtf / mass[type[i]];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
    }
    if (mask[i] & movebit ) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   choose insertion or deletion
   pick a vacsancy or atom
   change vacsancy/atom to 
   compare before/after energy and accept/reject the swap
------------------------------------------------------------------------- */

// if cap, fix wall/reflect behavior
void FixHGCMC::post_integrate()
{
  if (cap != 1) return;

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & movebit) {
      if (atom->x[i][2] < zcaplo) {
        atom->x[i][2] = zcaplo + (zcaplo - atom->x[i][2]);
        atom->v[i][2] *= -1;
      } else if (atom->x[i][2] > zcaphi) {
        atom->x[i][2] = zcaphi - (atom->x[i][2] - zcaphi);
        atom->v[i][2] *= -1;
      }
    }
  }
}

void FixHGCMC::final_integrate()
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
    if (mask[i] & groupbit) {
      dtfm = dtf / mass[type[i]];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
    }
  }
}


void FixHGCMC::end_of_step()
{
  if (update->ntimestep == laststep) evaluate_move();
}

void FixHGCMC::restore_atom() 
{
  int n = nlocal_stored;
  if (n > atom->nmax) error->all(FLERR,"not implemented");

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

// after initiate_gcmc so transforming particle in includegroup
void FixHGCMC::initiate_velocities()
{
  double sigma, mass;

  int reverse = 0;
  if (revtest && update->ntimestep % (2*ngcmcsteps) == ngcmcsteps + 1) { 
    utils::logmesg(lmp,"MPI{}: reversing velocities\n",comm->me);
    reverse = 1;
    for (int i=0; i<atom->nlocal; i++) {
      for (int x=0; x<3; x++) atom->v[i][x] *= -1.;
    }
  }

  for (int i=0; i<atom->nlocal; i++) {
    if (atom->mask[i] & movebit) {
      // real mass if in include group: real or transforming particle
      if (atom->mask[i] & group->bitmask[iincludegroup]) mass = atom->mass[atom->type[i]];
      else mass = fictmass;
      sigma = sqrt( beta_md * mass * force->mvv2e);
      atom->v[i][0] = random_unequal->gaussian() / sigma;
      atom->v[i][1] = random_unequal->gaussian() / sigma;
      atom->v[i][2] = random_unequal->gaussian() / sigma;
    }
    // need to explicitly set to 0 bc cap/base can become finite after restoring
    else {
      atom->v[i][0] = 0.;
      atom->v[i][1] = 0.;
      atom->v[i][2] = 0.;
    }
  }
}

// assumes atomic energies are up to date
void FixHGCMC::store_pe() {
  pereal_stored = c_pereal->compute_scalar();
  if (comm->me == 0) utils::logmesg(lmp,"stored PE = {}\n",pereal_stored);

  if (cap) {
    pecaphi_stored = c_pecaphi->compute_scalar();
    pecaplo_stored = c_pecaplo->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"stored PE cap = {}, base = {}\n",pecaphi_stored,pecaplo_stored);
  }
}

void FixHGCMC::store_ke() {
  c_keatom->compute_peratom();
  kereal_stored = c_kereal->compute_scalar();
  if (comm->me == 0) utils::logmesg(lmp,"stored KE = {}\n",kereal_stored);
}

void FixHGCMC::store_atom() {
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
    
    x_stored = (double *) memory->smalloc(n*3*sizeof(double),"hgcmc:x_stored");
    type_stored = (int *) memory->smalloc(n*sizeof(int),"hgcmc:type_stored");
    mask_stored = (int *) memory->smalloc(n*sizeof(int),"hgcmc:mask_stored");
    tag_stored = (tagint *) memory->smalloc(n*sizeof(tagint),"hgcmc:tag_stored");
    image_stored = (imageint *) memory->smalloc(n*sizeof(imageint),"hgcmc:image_stored");
    
    nmax_stored = n;
  }

  memcpy( x_stored, atom->x[0], 3*n*sizeof(double));
  memcpy( type_stored, atom->type, n*sizeof(int));
  memcpy( mask_stored, atom->mask, n*sizeof(int));
  memcpy( tag_stored, atom->tag, n*sizeof(tagint));
  memcpy( image_stored, atom->image, n*sizeof(imageint));
}

void FixHGCMC::trial_xycap()
{
  nxycap_trials++;
  if (comm->me == 0) utils::logmesg(lmp,"starting xycap MC\n");
  if (comm->me == 0) utils::logmesg(lmp,"before PE real = {}, cap = {}, base = {}\n",pereal_stored,pecaphi_stored,pecaplo_stored);

  double xshift = (random_equal->uniform() - 0.5) * 2 * xycapmax;
  double yshift = (random_equal->uniform() - 0.5) * 2 * xycapmax;

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & group->bitmask[icaphigroup]) {
      atom->x[i][0] += xshift;
      atom->x[i][1] += yshift;
    }
  }
  //int nflag = neighbor->decide();
  //recompute_peatom(nflag);
  comm->forward_comm();
  recompute_peatom();

  double pereal_diff = c_pereal->compute_scalar() - pereal_stored;
  double pecaphi_diff = c_pecaphi->compute_scalar() - pecaphi_stored;
  double pecaplo_diff = c_pecaplo->compute_scalar() - pecaplo_stored;
  double pe_diff = pereal_diff + pecaphi_diff + pecaplo_diff;
  
  if (comm->me == 0) utils::logmesg(lmp,"pe_diff = {}, real = {}, cap = {}, base = {}\n",pe_diff, pereal_diff, pecaphi_diff, pecaplo_diff);
  
  double pmove = exp(-beta * pe_diff);

  if (random_equal->uniform() < pmove ) { // accept
    nxycap_accepts++;
    if (comm->me == 0) utils::logmesg(lmp,"t = {} dx = {} dy = {} xycap accepted ({}/{})\n",update->ntimestep,xshift,yshift,nxycap_accepts,nxycap_trials);
      
    store_pe();
    store_atom();
  }
  else { // reject
    if (comm->me == 0) utils::logmesg(lmp,"xycap rejected ({}/{})\n",nxycap_accepts,nxycap_trials);
  
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->mask[i] & group->bitmask[icaphigroup]) {
        atom->x[i][0] -= xshift;
        atom->x[i][1] -= yshift;
      }
    }
    comm->forward_comm();
    recompute_peatom();
    
    domain->print_box("restored recalculated ");
   
    double pereal_restored = c_pereal->compute_scalar();
    double pecaphi_restored = c_pecaphi->compute_scalar();
    double pecaplo_restored = c_pecaplo->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"MPI{}: restored PE real = {}, cap = {}, base = {}\n",comm->me,pereal_restored,pecaphi_restored,pecaplo_restored);
    
  }
}

void FixHGCMC::trial_zcap()
{
  nzcap_trials++;
  if (comm->me == 0) utils::logmesg(lmp,"starting zcap MC\n");
 
  double volume_stored = domain->xprd * domain->yprd * (zcaphi - zcaplo);
  if (comm->me == 0) utils::logmesg(lmp,"storing volume {}\n",volume_stored);
  if (comm->me == 0) utils::logmesg(lmp,"before PE real = {}, cap = {}, base = {}\n",pereal_stored,pecaphi_stored,pecaplo_stored);

  double zshift = (random_equal->uniform() - 0.5) * 2 * zcapmax;

  if (comm->me == 0) utils::logmesg(lmp,"group hi = {}, lo = {}, shift = {}\n",zcaphi,zcaplo,zshift);

  domain->print_box("initial ");

  double zfrac;

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & movebit ) {
      zfrac = (atom->x[i][2] - zcaplo) / (zcaphi - zcaplo);
      atom->x[i][2] = zcaplo + zfrac * (zcaphi + zshift - zcaplo);
    } else if (atom->mask[i] & group->bitmask[icaphigroup]) {
      atom->x[i][2] += zshift;
    }
  }
  zcaphi += zshift;

  comm->forward_comm();
  recompute_peatom();

  domain->print_box("recalculated ");

  double pereal_diff = c_pereal->compute_scalar() - pereal_stored;
  double pecaphi_diff = c_pecaphi->compute_scalar() - pecaphi_stored;
  double pecaplo_diff = c_pecaplo->compute_scalar() - pecaplo_stored;
  double pe_diff = pereal_diff + pecaphi_diff + pecaplo_diff;
  
  if (comm->me == 0) utils::logmesg(lmp,"pe_diff = {}, real = {}, cap = {}, base = {}\n",pe_diff, pereal_diff, pecaphi_diff, pecaplo_diff);
  
  int nreal = atom->natoms - typecount[0];
  double volume = domain->xprd * domain->yprd * (zcaphi - zcaplo); //domain->zprd;
  double volume_ratio = pow( (volume / volume_stored), nreal);
  double volume_diff = volume - volume_stored;
  if (comm->me == 0) utils::logmesg(lmp,"zshift = {}, V0 = {}, V' = {}, N = {}, volume_ratio = {}\n",zshift,volume_stored,volume,nreal,volume_ratio);
  
  double pmove = volume_ratio * exp( -beta * (pe_diff + press * volume_diff / force->nktv2p) );

  if (random_equal->uniform() < pmove ) { // accept
    nzcap_accepts++;
    input->variable->internal_set(i_MC_V,volume);
    if (comm->me == 0) utils::logmesg(lmp,"t = {} dz = {} zcap accepted ({}/{})\n",update->ntimestep,zshift,nzcap_accepts,nzcap_trials);
    
    store_pe();
    store_atom();
  }
  else { // reject
    if (comm->me == 0) utils::logmesg(lmp,"zcap rejected ({}/{})\n",nzcap_accepts,nzcap_trials);
  
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->mask[i] & movebit) {
        zfrac = (atom->x[i][2] - zcaplo) / (zcaphi - zcaplo);
        atom->x[i][2] = zcaplo + zfrac * (zcaphi - zshift - zcaplo);
      } else if (atom->mask[i] & group->bitmask[icaphigroup]) {
        atom->x[i][2] -= zshift;
      }
    }

    zcaphi -= zshift;

    comm->forward_comm();
    recompute_peatom();
    
    domain->print_box("restored recalculated ");
   
    double pereal_restored = c_pereal->compute_scalar();
    double pecaphi_restored = c_pecaphi->compute_scalar();
    double pecaplo_restored = c_pecaplo->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"MPI{}: restored PE real = {}, cap = {}, base = {}\n",comm->me,pereal_restored,pecaphi_restored,pecaplo_restored);
  }
}

void FixHGCMC::trial_iso()
{
  niso_trials++;
  double volume_stored = domain->xprd * domain->yprd * domain->zprd;

  if (comm->me == 0) utils::logmesg(lmp,"iso box MC: storing volume = {}, pereal = {}\n",volume_stored,pereal_stored);

  double delta = (random_equal->uniform() - 0.5) * 2 * isomax;

  domain->print_box("initial ");

  //save old box state
  double boxlo[3];
  double h_inv[6];
  boxlo[0] = domain->boxlo[0];
  boxlo[1] = domain->boxlo[1];
  boxlo[2] = domain->boxlo[2];
  for (int i = 0; i < 6; i++)
    h_inv[i] = domain->h_inv[i];

  domain->boxhi[0] += delta;
  domain->boxhi[1] += delta;
  domain->boxhi[2] += delta;

  //domain->set_initial_box(); // error checks
  domain->set_global_box();
  domain->set_local_box();

  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->mask[i] & movebit)
      domain->x2lamda(x[i],x[i],boxlo,h_inv);

  for (int i = 0; i < atom->nlocal; i++)
    if (atom->mask[i] & movebit)
      domain->lamda2x(x[i],x[i]);
 
  comm->forward_comm();
  recompute_peatom();
  //recompute_peatom(0);
  
  //int nflag = neighbor->decide();
  //if (comm->me == 0) utils::logmesg(lmp,"nflag = {}\n",nflag);
  //recompute_peatom(nflag);

  domain->print_box("recalculated ");

  double pereal_after = c_pereal->compute_scalar();
  
  //utils::logmesg(lmp,"MPI{}: after PE real = {}, cap = {}, base = {}\n",comm->me,pereal_after,pecaphi_after,pecaplo_after);

  double pe_diff = pereal_after - pereal_stored;
  
  if (comm->me == 0) utils::logmesg(lmp,"pe_diff = {}\n",pe_diff);
  
  int nreal = atom->natoms - typecount[0];
  double volume = domain->xprd * domain->yprd * domain->zprd;
  double volume_ratio = pow( (volume / volume_stored), nreal);
  double volume_diff = volume - volume_stored;
  if (comm->me == 0) utils::logmesg(lmp,"delta = {}, V0 = {}, V' = {}, N = {}, volume_ratio = {}\n",delta, volume_stored,volume,nreal,volume_ratio);
  
  double pmove = volume_ratio * exp( -beta * (pe_diff + press * volume_diff / force->nktv2p) );

  if (random_equal->uniform() < pmove ) { // accept
    niso_accepts++;
    input->variable->internal_set(i_MC_V,volume);
    if (comm->me == 0) utils::logmesg(lmp,"iso box trial accepted ({}/{})\n",niso_accepts,niso_trials);

    store_pe();
    store_atom();
  }
  else { // reject
    if (comm->me == 0) utils::logmesg(lmp,"iso box trial rejected ({}/{})\n",niso_accepts,niso_trials);
    //store changed box
    boxlo[0] = domain->boxlo[0];
    boxlo[1] = domain->boxlo[1];
    boxlo[2] = domain->boxlo[2];
    for (int i = 0; i < 6; i++)
      h_inv[i] = domain->h_inv[i];

    domain->boxhi[0] -= delta;
    domain->boxhi[1] -= delta;
    domain->boxhi[2] -= delta;

    //domain->set_initial_box(); // error checks
    domain->set_global_box();
    domain->set_local_box();

    for (int i = 0; i < atom->nlocal; i++)
      if (atom->mask[i] & movebit)
        domain->x2lamda(x[i],x[i],boxlo,h_inv);

    for (int i = 0; i < atom->nlocal; i++)
      if (atom->mask[i] & movebit)
        domain->lamda2x(x[i],x[i]);
   
    comm->forward_comm();
    recompute_peatom();
    //recompute_peatom(0);
    //recompute_peatom(nflag);
   
    //domain->print_box("restored recalculated ");
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"real PE restored = {}, stored = {}\n",pereal_restored,pereal_stored);
  }
}

void FixHGCMC::trial_sgcmc()
{
  nsgc_trials++;

  // pick swaptype
  int typeindex;
  int noldtype = 0;
  while (noldtype == 0) {
    typeindex = (int) nswaptypes * random_equal->uniform();
    oldtype = swaptypes[typeindex];
    noldtype = typecount[oldtype];
  }
  
  newtype = oldtype;
  while (newtype == oldtype) {
    typeindex = (int) nswaptypes * random_equal->uniform();
    newtype = swaptypes[typeindex];
  }
  
  if (comm->me == 0) utils::logmesg(lmp,"sgcmc, oldtype = {} ({}), newtype = {} ({})\n",oldtype,typecount[oldtype],newtype,typecount[newtype]);

  // pick swap atom based on energy
  // find idswap and change to new type
  bias_forward();

  //comm->forward_comm(this);

  // explicitly update ghost atoms based on idswap?
  for (int i=atom->nlocal; i<atom->nghost; i++) {
    if (atom->tag[i] == idswap)  {
      atom->type[i] = newtype;
    }
  }
  
  recompute_peatom_swap();
  
  bias_reverse();
  //trial_reverse = 1. / typecount[newtype];

  double pe_diff = c_pereal->compute_scalar() - pereal_stored;
  double e_diff = pe_diff + mu[oldtype] - mu[newtype];

  //utils::logmesg(lmp,"me = {}, pereal_diff + mu_diff = {}\n",comm->me,e_diff);

  if (cap) {
   e_diff += c_pecaplo->compute_scalar() - pecaplo_stored;
   e_diff += c_pecaphi->compute_scalar() - pecaphi_stored;
   //utils::logmesg(lmp,"me = {}, cap = {}, pecaplo = {}\n",comm->me,cap,c_pecaplo->compute_scalar());
  }

  double trial_ratio = trial_reverse / trial_forward;
  double density_ratio = lambda3[oldtype] / lambda3[newtype]; // (1/Lambda) / (1/Lambda)
  
  if (comm->me == 0) utils::logmesg(lmp,"pereal_diff = {}, mu_diff = {}, trial_ratio = {}, density_ratio = {}\n",pe_diff,mu[oldtype]-mu[newtype],trial_ratio,density_ratio);
  
  double pmove = density_ratio * trial_ratio * exp( -beta * e_diff);

  // accept
  if (random_equal->uniform() < pmove ) {
    nsgc_accepts++; 
    //printf("me = %i, trial_ratio = {}, pmove = %f \n",comm->me,trial_reverse/trial_forward,pmove);
    if (comm->me == 0) utils::logmesg(lmp,"e_diff = {}, SGCMC accepted ({}/{})\n",e_diff,nsgc_accepts,nsgc_trials);
    
    typecount[oldtype]--;
    typecount[newtype]++;
    
    store_pe();
  }
  //reject 
  else {
    if (comm->me == 0) utils::logmesg(lmp,"e_diff = {}, SGCMC rejected ({}/{})\n",e_diff,nsgc_accepts,nsgc_trials);
    //for (int i=0; i<atom->nlocal; i++) {
    for (int i=0; i<(atom->nlocal+atom->nghost); i++) {
      if (atom->tag[i] == idswap)  {
        atom->type[i] = oldtype;
      }
    }
    //comm->forward_comm(this);

    // we don't actually need to recompute at all here if we trust our restore
    /*
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal successfully restored = {}, stored = {}\n",pereal_restored,pereal_stored);
    */
  }
  //if (comm->me == 0) utils::logmesg(lmp,"SGCMC completed\n");
}

void FixHGCMC::initiate_gcmc()
{
  // pick swaptype
  int swaptypeindex = (int) nswaptypes * random_equal->uniform();
  swaptype = swaptypes[swaptypeindex];

  if (comm->me == 0) utils::logmesg(lmp,"gcmc = {}, swaptype = {}\n",gcmc,swaptype);
    
  // make all fictitious particles swaptype
  // needed for reversibility and bias_forward/reverse calculations
  // has no effect of fictitious motion, which are initiated with velocity based on fict mass
  for  (int i=0; i<atom->nlocal; i++) {
    if (atom->mask[i] & fictgroupbit) atom->type[i] = nrealtypes + swaptype;
  }

  // deletion
  if (gcmc == -1) {
    ndeletion_trials++;
  
    bias_forward();
    
    // start from H(x^N+1)
    hybrid->scaleval[0] = 0;
    hybrid->scaleval[1] = 1;
    dfrac = -1./ngcmcsteps;
 
    reneighbor();
    recompute_peatom(); // might not be necessary? need to reneighbor at least
  }
  // insertion
  else if (gcmc == 1) {
    ninsertion_trials++;
   
    // turn on fictitious particles
    for (int i=0; i<atom->nlocal; i++) {
      if (atom->mask[i] & fictgroupbit) atom->mask[i] |= group->bitmask[iincludegroup];
    }
    hybrid->scaleval[0] = 0.;
    hybrid->scaleval[1] = 1.;
    
    reneighbor();
    recompute_peatom();

    bias_forward();
    
    // turn off fictitious particles
    for (int i=0; i<atom->nlocal; i++) {
      if (atom->mask[i] & fictgroupbit) atom->mask[i] &= group->inversemask[iincludegroup];
    }

    // start from H(x^N})
    hybrid->scaleval[0] = 1.;
    hybrid->scaleval[1] = 0.;
    dfrac = 1./ngcmcsteps;
    // need energy w/o fictitious particles
    reneighbor();
    recompute_peatom();
  }
  else error->all(FLERR,"gcmc != +/-1");

  // revtest: undo previous moves
  if (revtest && (update->ntimestep - 1) % (2*ngcmcsteps) == ngcmcsteps) {
    if (comm->me == 0) utils::logmesg(lmp,"keeping idswap = {}\n",idswap);
    trial_forward=1;
    for (int i=0; i<atom->nlocal; i++) {
      if (atom->tag[i] == idswap) {
        if (gcmc == 1) {
          atom->mask[i] &= group->inversemask[ifictgroup];
          atom->mask[i] |= group->bitmask[igroup];
        }
        else if (gcmc == -1) {
          atom->type[i] += nrealtypes;
        }
      }
    }
    reneighbor();
    recompute_peatom();
    return;
  }
}


void FixHGCMC::bias_forward() {
  
  int nlocal = atom->nlocal;
  if (atom->firstgroup > -1) nlocal = atom->nfirst;
  
  // variables
  double plocal_sum = 0;
  double Ei, pi;
  
  if (nlocal > atom_swap_nmax) {
    atom_swap_nmax = nlocal;
    memory->sfree(plocal);
    plocal = (double *) memory->smalloc(atom_swap_nmax * sizeof(double),"hgcmc:plocal");
  }
 
  // deletion
  if (gcmc == -1) {
    for (int i=0; i<nlocal; i++) {
      if (atom->mask[i] & groupbit && atom->type[i] == swaptype) {
        Ei = c_peatom->vector_atom[i] - mu[swaptype];
        pi = exp(beta_delete * Ei);
        plocal_sum += pi;
        plocal[i] = pi;
      }
    }
  }
  // insertion
  else if (gcmc == 1) {
    for (int i=0; i<nlocal; i++) {
      if (atom->mask[i] & fictgroupbit) {
        Ei = c_peatom->vector_atom[i] - mu[swaptype];
        pi = exp(-beta_insert * Ei);
        plocal_sum += pi;
        plocal[i] = pi;
      }
    }
  }
  // semi-grand
  else if (gcmc == 0) {
    for (int i=0; i<nlocal; i++) {
      if (atom->mask[i] & groupbit && atom->type[i] == oldtype) {
        //Ei = c_peatom->vector_atom[i] - mu[swaptype];
        //pi = exp(-beta_insert * Ei);
        pi = 1.;
        plocal_sum += pi;
        plocal[i] = pi;
      }
    }
  }
  
  double pglobal_sum, pbefore_sum;
  //pglobal_sum = typecount[oldtype]; // no benefit
  MPI_Allreduce(&plocal_sum,&pglobal_sum,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Scan(&plocal_sum,&pbefore_sum,1,MPI_DOUBLE,MPI_SUM,world);
  
  pbefore_sum -= plocal_sum;
  
  trial_forward = 0.;
  idswap = -1;

  //double zbuff;

  // find particle with probability p
  double pmove = pglobal_sum * random_equal->uniform();

  // if p points to this processor
  if ((pmove >= pbefore_sum) && (pmove < pbefore_sum + plocal_sum)) {
    // deletion
    if (gcmc == -1) {
      for (int i=0; i<nlocal; i++) {
        if (atom->mask[i] & groupbit && atom->type[i] == swaptype) {
          pbefore_sum += plocal[i];
          if (pbefore_sum > pmove) {
            idswap = atom->tag[i];
            trial_forward = plocal[i] / pglobal_sum;

            atom->type[i] += nrealtypes;
            // must be in fict group for bias_reverse interactions
            atom->mask[i] |= group->bitmask[ifictgroup];
            break;
          }
        }
      }
    }
    // insertion
    else if (gcmc == 1) {
      for (int i=0; i<nlocal; i++) {
        if (atom->mask[i] & fictgroupbit) {
          pbefore_sum += plocal[i];
          if (pbefore_sum > pmove) {
            idswap = atom->tag[i];
            trial_forward = plocal[i] / pglobal_sum;
            
            atom->mask[i] |= group->bitmask[igroup] | group->bitmask[iincludegroup];
            // must be out of fict group for bias_reverse interactions
            atom->mask[i] &= group->inversemask[ifictgroup];
            break;
          }
        }
      }
    }
    // semi-grand
    else if (gcmc == 0) {
      for (int i=0; i<nlocal; i++) {
        if (atom->mask[i] & groupbit && atom->type[i] == oldtype) {
          pbefore_sum += plocal[i];
          if (pbefore_sum > pmove) {
            idswap = atom->tag[i];
            trial_forward = plocal[i] / pglobal_sum;
           
            atom->type[i] = newtype;
            break;
          }
        }
      }
    }
  }

  
  struct{ double trial; int id; } sendbuff, recvbuff;
  sendbuff.trial = trial_forward;
  sendbuff.id = idswap;

  MPI_Allreduce(&sendbuff,&recvbuff,1,MPI_DOUBLE_INT,MPI_MAXLOC,world);

  trial_forward = recvbuff.trial;
  idswap = recvbuff.id;
  
  //trial_forward = 1. / typecount[oldtype];
  
  if (comm->me == 0) utils::logmesg(lmp,"idswap = {}, trial_forward = {}\n",idswap,trial_forward);
  
  /*
  if (cap && comm->me == 0) {
    MPI_Recv(&zbuff,1,MPI_DOUBLE,MPI_ANY_SOURCE,0,world,MPI_STATUS_IGNORE);
    utils::logmesg(lmp,"gcmc = {} zfrac = {} p = {} ({} times) \n",gcmc,(zbuff-zcaplo)/(zcaphi-zcaplo),trial_forward,trial_forward / trialrandom );
  }
  */
}

    
// hybrid->scaleval = [0,1]
// the transforming particle in real group, fict type (2nd Hamiltonian)
void FixHGCMC::bias_reverse() {
 
  int nlocal = atom->nlocal;
  if (atom->firstgroup > -1) nlocal = atom->nfirst;

  // reallocate list of swaptypes atoms if needed
  if (nlocal > atom_swap_nmax) {
    atom_swap_nmax = nlocal;
    memory->sfree(plocal);
    plocal = (double *) memory->smalloc(atom_swap_nmax * sizeof(double),"hgcmc:plocal");
  }
  
  double plocal_sum = 0, plocal_reverse = 0;
  double Ei, pi; 
 
  // include transforming particle: real group, fict type
  for (int i=0; i<nlocal; i++) {
    //utils::logmesg(lmp,"id = {}, type = {}, real? = {}, E = {}\n",atom->tag[i],atom->type[i],atom->mask[i] & groupbit,c_peatom->vector_atom[i],plocal[i]);
    // reversal is deletion
    if (gcmc == 1 && atom->mask[i] & groupbit &&
        (atom->type[i] == swaptype || atom->type[i] == swaptype + nrealtypes) ) {
      Ei = c_peatom->vector_atom[i] - mu[swaptype];
      pi = exp(beta_delete * Ei );
      //utils::logmesg(lmp,"MPI{}: rev real {} type = {} PE = {}, E = {}, p = {}\n",comm->me,atom->tag[i],itype,c_peatom->vector_atom[i],Ei,pi);
      plocal_sum += pi;
      plocal[i] = pi;
    }
    // reversal is insertion: deleted particle real group, but type changed
    else if (gcmc == -1 && atom->type[i] > nrealtypes) {
      Ei = c_peatom->vector_atom[i] - mu[swaptype];
      pi = exp(-beta_insert*Ei );
      plocal_sum += pi;
      plocal[i] = pi;
      //utils::logmesg(lmp,"MPI{}: rev vac id = {} type = {} PE = {} mu = {} E = {} p = {}\n",comm->me,atom->tag[i],itype,c_peatom->vector_atom[i],mu[otype],Ei,pi);
    }
    else if (gcmc == 0 && atom->mask[i] & groupbit && atom->type[i] == newtype) {
      //Ei = c_peatom->vector_atom[i] - mu[swaptype];
      //pi = exp(-beta_insert*Ei );
      pi = 1.;
      plocal_sum += pi;
      plocal[i] = pi;
      //utils::logmesg(lmp,"MPI{}: rev vac id = {} type = {} PE = {} mu = {} E = {} p = {}\n",comm->me,atom->tag[i],itype,c_peatom->vector_atom[i],mu[otype],Ei,pi);
    }
    // find p_i
    if (atom->tag[i] == idswap) {
      plocal_reverse = plocal[i];
      utils::logmesg(lmp,"MPI{}: reverse idswap = {}, E = {}, p = {}\n",comm->me,idswap,c_peatom->vector_atom[i],plocal[i]);
    }
  }
  
  double pglobal_reverse, pglobal_sum;
  MPI_Allreduce(&plocal_reverse,&pglobal_reverse,1,MPI_DOUBLE,MPI_MAX,world);
  MPI_Allreduce(&plocal_sum,&pglobal_sum,1,MPI_DOUBLE,MPI_SUM,world);

  trial_reverse = pglobal_reverse/pglobal_sum;
  
  if (comm->me == 0) utils::logmesg(lmp,"trial_reverse = {}/{} = {}\n",pglobal_reverse,pglobal_sum,trial_reverse);
}

void FixHGCMC::evaluate_move() {
  
  if (comm->me == 0) utils::logmesg(lmp,"t = {}, evaluating gcmc = {}\n",update->ntimestep,gcmc);
  
  int nlocal = atom->nlocal;
  if (atom->firstgroup > -1) nlocal = atom->nfirst;
 
  // energy should be up to date

  if (update->eflag_atom != update->ntimestep) {
    error->warning(FLERR,"Atomic energies not computed. Overwritten by external output (step {})? Recomputing.",update->eflag_atom);
    recompute_peatom();
  }

  double pereal_after = c_pereal->compute_scalar();
  double pereal_diff = pereal_after - pereal_stored;
  double e_diff = pereal_diff;
  
  c_keatom->compute_peratom();
  double kereal_after = c_kereal->compute_scalar();
  double kereal_diff = kereal_after - kereal_stored;
  
  if (cap) {
    double pecaphi_diff = c_pecaphi->compute_scalar() - pecaphi_stored;
    double pecaplo_diff = c_pecaplo->compute_scalar() - pecaplo_stored;
    e_diff += pecaphi_diff + pecaplo_diff;
  }

  double density_ratio = 1.;
  double volume_ratio = 1.;
  double trial_ratio = 1.;

  if (gcmc != 0) {
    
    double volume;
    if (cap) volume = domain->xprd * domain->yprd * (zcaphi - zcaplo);
    else volume = domain->xprd * domain->yprd * domain->zprd;
   
    // deletion
    if (gcmc == -1) { 
      // reverse is insertion, need to turn on fictitious particles
      for (int i=0; i<atom->nlocal; i++) {
        if (atom->mask[i] & fictgroupbit) atom->mask[i] |= group->bitmask[iincludegroup];
      }
      hybrid->scaleval[0] = 0.;
      hybrid->scaleval[1] = 1.;
     
      reneighbor();
      recompute_peatom();
      
      bias_reverse();
      
      e_diff += mu[swaptype];
      density_ratio = (typecount[0]+1) * lambda3[swaptype] / volume; // (M-N)+1 / V/L^3 
    }
    // insertion: already hybrid->scaleval = [0,1], which is correct for bias_reverse
    else if (gcmc == 1) {
      bias_reverse();
      
      e_diff -= mu[swaptype];
      density_ratio = volume / lambda3[swaptype] / typecount[0]; // V/L^3 / (M-N)
    }
 
    trial_ratio = trial_reverse / trial_forward;
  
    if (comm->me == 0) utils::logmesg(lmp,"density_ratio = {}, trial_ratio = {} , volume_ratio = {}\n",density_ratio,trial_ratio,volume_ratio);
  }
 
  //
  if (comm->me == 0) utils::logmesg(lmp,"e_diff = {}, ke_diff = {}, total_diff = {}\n",e_diff,kereal_diff,e_diff+kereal_diff);
  
  double pmove = density_ratio * trial_ratio * volume_ratio * exp(-beta_md * kereal_diff - beta * e_diff);
 
  // accept move
  if (random_equal->uniform() < pmove || revtest) {
    // displacement
    if (gcmc == 0) {
      nmd_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"MD CMC accepted ({}/{})\n",nmd_accepts,nmd_trials);
    }
    // deletion
    else if (gcmc == -1) {
      ndeletion_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"deletion GCMC accepted ({}/{})\n",ndeletion_accepts,ndeletion_trials);
      typecount[0]++;
      typecount[swaptype]--;

      for (int i=0; i<atom->nlocal; i++) {
        // remove deleted particle from real group
        if (atom->tag[i] == idswap) atom->mask[i] &= group->inversemask[igroup]; 
        // turn off vacancies, left on by bias_reverse
        // also removes deleted particle from includegroup
        if (atom->mask[i] & fictgroupbit) atom->mask[i] &= group->inversemask[iincludegroup];
      }
      // last energies from bias_reverse with vacancies
    }
    // insertion
    else if (gcmc == 1) {
      ninsertion_accepts++;
      if (comm->me == 0) utils::logmesg(lmp,"insertion GCMC accepted ({}/{})\n",ninsertion_accepts,ninsertion_trials);
      typecount[0]--;
      typecount[swaptype]++;

      // give idswap real type
      // should already be non-fictgroup b/c needed fict interactions for bias_reverse deletion
      for (int i=0; i<atom->nlocal; i++) {
        if (atom->tag[i] == idswap) {
          atom->type[i] = swaptype;
          break;
        }
      }
      // energies should be current but need to reneighbor w/ new type
      // unless I can just change types for < nmax
      // scaleval = [0,1] from trajectory, however
    }
    // store new states on step we calculated energy
    if (fictgroupname) {
      hybrid->scaleval[0] = 1.;
      hybrid->scaleval[1] = 0.;
    }
    reneighbor();
    recompute_peatom();
    store_pe();
    store_atom();

    // dump output
    int naccepts = ninsertion_accepts + ndeletion_accepts;
    if (!onlydumpgc) naccepts += nmd_accepts + nsgc_accepts + niso_accepts + nxycap_accepts + nzcap_accepts;
    if (imcdump > -1 && naccepts % output->every_dump[imcdump] == 0 && (!onlydumpgc || gcmc != 0)) {
      bigint t = update->ntimestep;
      output->next = t;
      output->next_dump_any = t;
      output->next_dump[imcdump] = t;
    }
  }
  // reject move
  else {
    if (comm->me == 0) utils::logmesg(lmp,"Hamiltonian trajectory rejected\n");
    restore_atom();
    // need to recompute neighbor lists entirely
    reneighbor();
    recompute_peatom();
    
    double pereal_restored = c_pereal->compute_scalar();
    if (comm->me == 0) utils::logmesg(lmp,"pereal restored = {}, stored = {}\n",pereal_restored,pereal_stored);
  }
}

void FixHGCMC::reneighbor() {
  if (modify->n_pre_exchange) {
    timer->stamp();
    modify->pre_exchange();
    timer->stamp(Timer::MODIFY);
  }
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
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

void FixHGCMC::recompute_peatom()
{
  // eflag/vflag usually set based on computes in protected integrate::ev_set(timestep)
  int eflag = ENERGY_ATOM; //+ ENERGY_GLOBAL;
  int vflag = VIRIAL_ATOM; //+ VIRIAL_PAIR;
  // TODO what is VIRIAL_PAIR ?

  // update-> flags tell compute that energies/virials are current
  update->eflag_atom = update->ntimestep;
  update->vflag_atom = update->ntimestep;
  
  // following Verlet::run
  //if (nflag == 0) {
    // I don't think I actually have anything to forward_comm here
    //timer->stamp();
    //comm->forward_comm();
    //timer->stamp(Timer::COMM);
  // compute new forces/energies
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
}

void FixHGCMC::recompute_peatom_swap()
{
  int eflag = ENERGY_ATOM;
  int vflag = 0; 

  update->eflag_atom = update->ntimestep;
  
  timer->stamp();

  if (modify->n_pre_force) {
    modify->pre_force(vflag);
    timer->stamp(Timer::MODIFY);
  }
  force->pair->compute(eflag,vflag);
  timer->stamp(Timer::PAIR);
 
  if (modify->n_post_force_any) modify->post_force(vflag);
  timer->stamp(Timer::MODIFY);
  
  c_peatom->compute_peratom();
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */

//double FixHGCMC::compute_array(int i, int j)
double FixHGCMC::compute_vector(int j)
{
  /*
  if (i == 0) {
    if (j > nrealtypes) return 0.0;
    else if (j == nrealtypes) {
      int ntotal = 0;
      for (int itype=1; itype<=nrealtypes; itype++) {
        ntotal += typecount[itype];
      }
      return ntotal;
    }
    else {
      return typecount[j];
    }
  }
  else if (i == 1) {
  */
    if (j == 0) return nmd_trials;
    if (j == 1) return nmd_accepts;
    //if (j == 3) return ninsertion_trials;
    if (j == 2) return ndeletion_trials + ninsertion_trials;
    //if (j == 4) return ninsertion_accepts;
    if (j == 3) return ndeletion_accepts + ninsertion_accepts;
    //if (j == 5) return ndeletion_trials;
    //if (j == 6) return ndeletion_accepts;
    //if (j == 7) return nsgc_trials;
    //if (j == 8) return nsgc_accepts;
    if (j == 4) return nsgc_trials;
    if (j == 5) return nsgc_accepts;
    if (j == 6) return niso_trials + nxycap_trials + nzcap_trials;
    if (j == 7) return niso_accepts + nxycap_accepts + nzcap_accepts;
  //}
    else return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixHGCMC::memory_usage()
{
  double bytes = (double) atom_swap_nmax * sizeof(int);
  return bytes;
}

int FixHGCMC::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  if (comm->me == 0) utils::logmesg(lmp,"packing forward comm...\n");
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

void FixHGCMC::unpack_forward_comm(int n, int first, double *buf)
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
