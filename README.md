This is the public release of my lammps contributions across multiple projects.

Files should be recursively copied into the src/ of a lammps release.

# 1. Unconstrained Monte Carlo

See "Monte Carlo Simulations of Crystal Defects in Open Ensembles" (https://arxiv.org/abs/2506.12985v2) and "Finite-temperature grain boundary phases, transformations, and diagrams in alloys" (https://arxiv.org/abs/2609.34131).

The "MC" folder contains the actively developed version of unconstrained MC as a standalone "fix umc."  The code was last tested with the 30Mar2026 release. Documentation is in progress.

# 2. Performant and expressive atomic cluster expansions (PEACE)

See "Designing medium-sized interatomic potentials for performance and expressivity" (https://arxiv.org/abs/XXXXXX)

ML-PACE/ contain the PEACE CPU implementation including shared code. This folder is used to borrow PACE's yaml configuration.

KOKKOS/ contains the PEACE Kokkos implementation (GPU only).

Maximizing performance requires some care when compiling. Example build scripts for an AMD MI300A and NVIDIA H100 (LLNL, Clang, AVX512) and NVIDIA A100 (NERSC, GCC, AVX2) are provided in build/.

## peacemaker
PEACE runs GRACE FS yaml files with optional additions like polynomial-specific radial functions. A pytorch-based example script is provided as python/peacemaker.py. This script calls grace-tensorpotential (0.6.0+), which is distributed elsewhere under the Academic Software License. It assumes an equivalent python environment with torch and torchvision in place of tensorflow. An example input.yaml is provided in python/example.

In addition to "radial: linear_max" mixing and learned FS embeddings (fs_parameters: 8), the script can provide a zbl baseline with a clamped beam basis. It also parallelizes full BFGS over an arbitrary number of GPUs with very little precision loss while keeping data on device for much faster fits. Export to lammps as peacemaker.py export path/to/checkpointpath/to/checkpoint
