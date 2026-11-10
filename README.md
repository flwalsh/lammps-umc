This is the public release of my lammps contributions across multiple projects.

Directories can be recursively copied into the src/ directory of a lammps release from https://github.com/lammps/lammps.

# 1. Unconstrained Monte Carlo

See "Monte Carlo Simulations of Crystal Defects in Open Ensembles" (https://arxiv.org/abs/2506.12985v2) and "Finite-temperature grain boundary phases, transformations, and diagrams in alloys" (https://arxiv.org/abs/2609.34131).

The "MC" folder contains the actively developed version of unconstrained MC as a standalone "fix umc."  The code was last tested with the 30Mar2026 release. Documentation is in progress. The original paper version is in the "paper" branch.

# 2. Performant and expressive atomic cluster expansions (PEACE)

See "Overcoming speed-accuracy trade-offs with medium-sized interatomic potentials" (https://arxiv.org/abs/2611.11544)

ML-PACE/ contain the PEACE CPU implementation, with some shared code in pair_peace_impl.h. This folder is used to borrow PACE's yaml library configuration.

KOKKOS/ contains the PEACE Kokkos implementation (GPU only).

Example cmake configurations for an AMD MI300A, NVIDIA H100 (LLNL, Clang, AVX512), and NVIDIA A100 (NERSC, GCC, AVX2) are provided in build/.

PEACE (pair_style peace) runs the GRACE FS yaml format with optional additions like polynomial-specific radial functions. The two optional arguments are the chunksize and timing, which displays kernel/phase timings at every step.

## peacemaker
An example script for generating new models is provided in python/peacemaker.py, which works like gracemaker with pytorch replacing tensorflow and similar other dependencies (ase, numpy, scipy, pandas 2, matscipy, and pyyaml).

An example input.yaml and parallel fitting jobs are provided in python/. In addition to "radial: linear_max" mixing and learned FS embeddings (fs_parameters: 8), the script can provide a ZBL baseline with a clamped beam basis. It also parallelizes full BFGS over an arbitrary number of GPUs with minimal precision loss while keeping data on device for much faster fits. Export to lammps as "peacemaker.py export path/to/checkpoint".
