#!/usr/bin/env python
# ----------------------------------------------------------------------
#   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
#   https://www.lammps.org/ Sandia National Laboratories
#   LAMMPS Development team: developers@lammps.org
#
#   Copyright (2003) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under
#   the GNU General Public License.
#
#   See the README file in the top-level LAMMPS directory.
# -------------------------------------------------------------------------
#   Contributing author: Flynn Walsh
# -------------------------------------------------------------------------
"""peacemaker input.yaml [-r | -rl] -- fit, then export the last checkpoint to saved_model.yaml
peacemaker export CHECKPOINT [OUTPUT] -- write CHECKPOINT out for LAMMPS, to saved_model.yaml by default"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field, fields
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from ase.data import atomic_numbers
from scipy.optimize import brentq
from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
from torch import func
from torch.nn import functional

# donated buffers interact badly with the func.grad path under compile
import torch._functorch.config
torch._functorch.config.donated_buffer = False

DTYPE = torch.float64

ROOT = Path(find_spec("tensorpotential").submodule_search_locations[0])


def call_grace_no_tf(path):
    spec = spec_from_file_location("_grace", ROOT / path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


real_coupling_metainformation = call_grace_no_tf(
    "functions/couplings.py").real_coupling_metainformation


def parity_sign(parity):
    return "+" if parity > 0 else "-"


def drop_swaps(df, A, B):
    cols = ["hist", "l", "parity", "sum_of_ls"]
    pairings = {}
    for h1, l1, p1, n1 in set(map(tuple, A[cols].itertuples(index=False))):
        for h2, l2, p2, n2 in set(map(tuple, B[cols].itertuples(index=False))):
            if l1 == l2:
                pairings[f"({h1}{l1}{parity_sign(p1)},{h2}{l2}{parity_sign(p2)})"] = (
                    (h1, p1, n1), (h2, p2, n2))

    df = df[df["l2"] <= df["l1"]]
    keep = []
    for row in df.itertuples():
        if row.l1 == row.l2:
            (h1, p1, n1), (h2, p2, n2) = pairings[row.hist]
            swap = f"({h2}{row.l1}{parity_sign(p2)},{h1}{row.l1}{parity_sign(p1)})"
            if (h1, p1, n1) > (h2, p2, n2) and swap in pairings:
                continue
        keep.append(row.Index)

    return df.loc[keep].reset_index(drop=True)


@dataclass
class Potential:
    element_map: dict
    device: torch.device = None
    basis_type: str = "SBessel"
    radial: str = "ln"
    chemistry: str = None
    n_rad_base: int | tuple = 20
    n_rad_max: tuple = (20, 15, 10, 5)
    lmax: tuple = (5, 5, 4, 3)
    lmax_hist: tuple = None
    max_sum_l: tuple = (None, None, None, None)
    n_mlp_dens: int = None
    activation: str = "silu"
    Lmax: tuple = (None, 4, 0, 0)
    Lmin: tuple = (None, 0, 0, 0)      # 1 drops invariants that are products of lower ranks
    max_order: int = 4
    rcut: float = 6.0
    rclamp: float = 0.0
    embedding_size: int = 64
    mlp_hidden: tuple = None           # None is one layer of n_mlp_dens
    fs_parameters: tuple = ((1.0, 1.0), (1.0, 0.5), (1.0, 2), (1.0, 0.75))
    avg_n_neigh: float = 1.0
    out_scale: float = 1.0
    out_shift: float = 0.0
    parity: str = "real"
    init: str = "deterministic"
    seed: int = None
    central_rank: int = 1
    zbl: bool = False
    params: dict = field(init=False, default_factory=dict)
    wrad: dict = field(init=False, default_factory=dict)
    keep_parity: list = field(init=False, default=None)


def learn(p, name, *shape, scale=1.0, fill=None):
    value = (torch.randn(*shape, dtype=DTYPE, device=p.device) * scale
             if fill is None else
             torch.full(shape, fill, dtype=DTYPE, device=p.device))
    p.params[name] = value.requires_grad_(True)
    return p.params[name]


def learn_diagonal(p, name, *shape, fill=0.0):
    w = learn(p, name, *shape, fill=fill)
    i = torch.arange(min(shape[-2], shape[-1]), device=p.device)
    with torch.no_grad():
        w[..., i, i] += 1.0
    return w


def learn_wrad(p, name, slot, *shape):
    # cyclic shifts by slot, so AA's factors do not get equal gradients forever
    nout, nin = shape[-2], shape[-1]
    if p.init != "deterministic":
        return learn(p, name, *shape, scale=nin ** -0.5)

    w = learn(p, name, *shape, fill=0.0)
    with torch.no_grad():
        for b, start in enumerate(range(0, nout, nin)):
            k = torch.arange(start, min(start + nin, nout), device=p.device)
            w[..., k, (k - start + slot - 1) % nin] += 0.5 ** b
    return w


def central_mixture(p, name, shared):
    embedding = p.params.get("central.embedding")
    if embedding is None:
        embedding = learn(p, "central.embedding", len(p.element_map),
                          p.central_rank, scale=p.central_rank ** -0.5)
        with torch.no_grad():
            embedding[:, 0] = 1.0
    components = learn(p, name, p.central_rank, *shared.shape, fill=0.0)
    with torch.no_grad():
        components[0].copy_(shared)
    return lambda: torch.einsum("ir,r...->i...", embedding, components)


# --- radial basis ---

def build_clamped_beam(n, rcut, rclamp, device=None):
    betas, sigmas, deltas = [], [], []
    for k in range(1, n + 1):
        c = (2 * k + 1) * np.pi / 2
        b = brentq(lambda x: np.cosh(x) * np.cos(x) - 1.0, c - np.pi / 2, c + np.pi / 2)
        denom = np.sinh(b) - np.sin(b)
        betas.append(b)
        sigmas.append((np.cosh(b) - np.cos(b)) / denom)
        deltas.append((-np.exp(-b) + np.cos(b) - np.sin(b)) / denom)

    betas = torch.tensor(betas, dtype=DTYPE, device=device)
    sigmas = torch.tensor(sigmas, dtype=DTYPE, device=device)
    deltas = torch.tensor(deltas, dtype=DTYPE, device=device)
    rcut = torch.tensor(rcut, dtype=DTYPE, device=device)
    rclamp = torch.tensor(rclamp, dtype=DTYPE, device=device)
    span = rcut - rclamp

    def clamped_beam(r):
        # clamped first: 0*inf reaches the where backward as NaN otherwise
        u = torch.clamp((r - rclamp) / span, 0.0, 1.0)
        bu = betas * u
        out = (torch.exp(-bu) + deltas * torch.sinh(bu)
               - torch.cos(bu) + sigmas * torch.sin(bu))
        return torch.where((r >= rclamp) & (r <= rcut), out, 0.0)

    return clamped_beam


def build_sbessel(n, rcut):
    gain, prev, d = [1.0], [0.0], 1.0
    for i in range(1, n):
        en = i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1)
        d_next = 1.0 - en / d
        gain.append(1.0 / math.sqrt(d_next))
        prev.append(math.sqrt(en / d) / math.sqrt(d_next))
        d = d_next

    amp = [g * (-1) ** k * math.sqrt(2.0) * math.pi / rcut ** 1.5
           * (k + 1) * (k + 2) / math.sqrt((k + 1) ** 2 + (k + 2) ** 2)
           for k, g in enumerate(gain)]

    def sbessel(r):
        s = [torch.sinc(r * (m / rcut)) for m in range(1, n + 2)]
        sbf = [amp[0] * (s[0] + s[1])]
        for i in range(1, n):
            sbf.append(amp[i] * (s[i] + s[i + 1]) + prev[i] * sbf[i - 1])
        return torch.where(r > rcut, 0.0, torch.cat(sbf, dim=1))

    return sbessel


def build_zbl(p):
    z = torch.tensor([atomic_numbers[s] for s in sorted(p.element_map, key=p.element_map.get)],
                     dtype=DTYPE, device=p.device)
    c = torch.tensor([0.18175, 0.50986, 0.28022, 0.02817], dtype=DTYPE, device=p.device)
    d = torch.tensor([-3.19980, -0.94229, -0.40290, -0.20162], dtype=DTYPE, device=p.device)

    def zbl(batch):
        bv, mu = batch["bond_vector"], batch["atomic_mu_i"]
        r = torch.sqrt((bv * bv).sum(1, keepdim=True))
        z_i, z_j = z[mu[batch["ind_i"]]][:, None], z[mu[batch["ind_j"]]][:, None]
        a = 0.46850 / (z_i**0.23 + z_j**0.23)
        k = 14.399645 * z_i * z_j   # eV*A
        phi_c = c * torch.exp(d * p.rcut / a)
        e_c = k * phi_c.sum(1, keepdim=True) / p.rcut
        de_c = k * (-phi_c.sum(1, keepdim=True) / p.rcut**2
                    + (phi_c * d).sum(1, keepdim=True) / (a * p.rcut))
        e = k * (c * torch.exp(d * r / a)).sum(1, keepdim=True) / r
        # pad bonds are 2*rcut where the shifted form is not zero
        e = torch.where(r < p.rcut, e - e_c - (r - p.rcut) * de_c, 0.0)
        # each pair is seen from both ends
        return e.new_zeros((mu.shape[0], 1)).index_add_(0, batch["ind_i"], 0.5 * e)

    return zbl


# --- angular basis ---

def spherical_harmonics(lmax, dtype, device):
    alm, blm = [0.0], [0.0]
    for l in range(1, lmax + 1):
        lsq, l2 = l * l, (l - 1) ** 2
        for m in range(0, l + 1):
            alm += [-math.sqrt(1.0 + 0.5 / m) if l == m
                    else math.sqrt((4 * lsq - 1) / (lsq - m * m))]
            blm += [0.0 if l == m
                    else -math.sqrt((l2 - m * m) / (4 * l2 - 1))]

    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    factor4pi, y00 = torch.sqrt(4 * pi), torch.rsqrt(4.0 * pi)
    sq3o4pi, sq3o8pi = torch.sqrt(3.0 / (4.0 * pi)), -torch.sqrt(3.0 / (8.0 * pi))

    def lm1d(l, m):
        return m + l * (l + 1) // 2

    def legendre(x):
        plm = [torch.zeros_like(x) + y00]
        if lmax > 0:
            plm += [sq3o4pi * x, torch.zeros_like(x) + sq3o8pi]
            for l in range(2, lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        plm += [x * math.sqrt(2.0 * m + 3.0) * plm[lm1d(l - 1, l - 1)]]
                    elif m == l:
                        plm += [alm[lm1d(l, l)] * plm[lm1d(l - 1, l - 1)]]
                    else:
                        plm += [alm[lm1d(l, m)] * (x * plm[lm1d(l - 1, m)]
                                + blm[lm1d(l, m)] * plm[lm1d(l - 2, m)])]
        return torch.stack(plm)

    nylm, take, scale = (lmax + 1) * (lmax + 2) // 2, [], []
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            ma = abs(m)
            take += [nylm * (m < 0) + l + ma * lmax - ma * (ma - 1) // 2]
            scale += [1.0 if m == 0 else math.sqrt(2.0) * (1.0 - 2.0 * (m % 2))]
    take = torch.tensor(take, dtype=torch.int64, device=device)
    scale = torch.tensor(scale, dtype=dtype, device=device)[:, None] * factor4pi

    def sph(rhat):
        rx, ry, rz = rhat[:, 0], rhat[:, 1], rhat[:, 2]
        plm = legendre(rz)
        ylm_r = [plm[lm1d(l, 0)] for l in range(0, lmax + 1)]
        ylm_i = [torch.zeros_like(plm[0])] * (lmax + 1)
        ylm_r += [rx * plm[lm1d(l, 1)] for l in range(1, lmax + 1)]
        ylm_i += [ry * plm[lm1d(l, 1)] for l in range(1, lmax + 1)]
        ph_r, ph_i = rx, ry
        for m in range(2, lmax + 1):
            ph_r, ph_i = ph_r * rx - ph_i * ry, ph_r * ry + ph_i * rx
            ylm_r += [ph_r * plm[lm1d(l, m)] for l in range(m, lmax + 1)]
            ylm_i += [ph_i * plm[lm1d(l, m)] for l in range(m, lmax + 1)]
        zlm = torch.stack(ylm_r + ylm_i)[take] * scale
        return torch.transpose(zlm, 0, 1)

    return sph


# --- coupling tables ---

def collect(table, max_l, l_p=None, numbering=("parity", "l", "hist")):
    # radial transforms number channels (l, parity, hist), densities (parity, l, hist)
    # order kept is always (parity, l, hist) whatever numbering says
    groups = table.groupby(["parity", "l", "hist"]).indices
    kept = np.concatenate(
        [v for (p, l, *_), v in groups.items()
         if l <= max_l and (l_p is None or [l, p] in l_p)])

    channels = table.iloc[kept].reset_index(drop=True)
    numbered = channels.groupby(list(numbering)).indices
    weight_of = np.zeros(len(channels), dtype=np.int64)
    for i, v in enumerate(numbered.values()):
        weight_of[v] = i

    return kept, weight_of, len(numbered), channels


# --- the pieces the model is made of ---

def build_chemistry(p, l_tile, nrad_A):
    nelements, nl = len(p.element_map), max(p.lmax) + 1
    fill = 1.0 if p.init == "deterministic" else None
    if p.radial == "linear_max" and p.central_rank > 1:
        w = central_mixture(
            p, "z", learn(p, "z", nelements, nrad_A, fill=fill))
        return lambda mu_i, mu_j: w()[mu_i, mu_j][:, :, None]
    if p.chemistry == "Z":
        w = learn(p, "z", nelements, fill=fill)
        return lambda mu_i, mu_j: w[mu_j][:, None, None]
    if p.chemistry == "Zn":
        w = learn(p, "z", nelements, nrad_A, fill=fill)
        return lambda mu_i, mu_j: w[mu_j][:, :, None]
    if p.chemistry == "Znl":
        w = learn(p, "z", nelements, nrad_A, nl, fill=fill)
        return lambda mu_i, mu_j: w[mu_j][:, :, l_tile]
    if p.chemistry == "Zinl":
        w = learn(p, "z", nelements, nelements, nrad_A, nl, fill=fill)
        return lambda mu_i, mu_j: w[mu_i, mu_j][:, :, l_tile]
    if p.chemistry == "Zembed":
        w = learn(p, "z", nelements, p.embedding_size, fill=fill)
        proj = (learn_diagonal(p, "z_proj", p.embedding_size, nrad_A)
                if p.init == "deterministic" else
                learn(p, "z_proj", p.embedding_size, nrad_A))
        return lambda mu_i, mu_j: (w @ (proj * proj.shape[0] ** -0.5))[mu_j][:, :, None]


# --- the model ---

def build_wrad(p, name, rank, slot, operand, parent=None, side=None):
    rows, radial_index, nradials, table = operand
    remapped = None
    if parent is not None:
        l_of = table["l"].to_numpy()
        first = {l: int(np.flatnonzero(l_of == l)[0]) for l in np.unique(l_of)}

        l_col = f"l{side + 1}"
        key = [l_col, "parity", "l", "hist"] + (
            ["invariant"] if "invariant" in parent else [])
        radial_index = parent.groupby(key).ngroup().to_numpy()
        radial_l = np.zeros(radial_index.max() + 1, dtype=np.int64)
        radial_l[radial_index] = parent[l_col].to_numpy()

        positions, indices, block, start = [], [], [], 0
        for group, l in enumerate(radial_l):
            block.append(start)
            positions.extend(range(first[l], first[l] + 2 * l + 1))
            indices.extend([group] * (2 * l + 1))
            start += 2 * l + 1

        remapped = [np.asarray(ind) - first[radial_l[group]] + block[group]
                    for ind, group in zip(parent[("left_inds", "right_inds")[side]],
                                          radial_index)]
        table = table.iloc[positions].reset_index(drop=True)
        rows, radial_index, nradials = rows[positions], indices, len(radial_l)
    else:
        table = table.copy()

    nout = p.n_rad_max[rank - 1]
    radial_index = np.asarray(radial_index)
    table["radial"] = radial_index * nout
    rows = torch.tensor(rows, dtype=torch.int64, device=p.device)
    key = f"wrad.{name}"
    if p.radial == "linear":
        radial_index = np.zeros_like(radial_index)
        nradials = 1
        key = f"wrad.{slot}"
    radial_index = torch.tensor(radial_index, dtype=torch.int64, device=p.device)

    if key not in p.params:
        shape = (nradials, nout, p.n_rad_base[1])
        learn_wrad(p, key, slot, *shape)
    w = p.params[key]
    if p.radial == "linear_max" and p.central_rank > 1:
        expanded = central_mixture(p, key, w)
        p.wrad[name] = expanded().detach()[..., :nradials, :nout, :]

        def transform(A, mu_i):
            w = expanded()[..., :nradials, :nout, :]
            return torch.einsum("aekn,ane->ake", w[:, radial_index][mu_i],
                                A[:, :, rows])

        return transform, remapped, table

    w = w[..., :nradials, :nout, :]
    p.wrad[name] = w

    if w.ndim == 4:
        def transform(A, mu_i):
            return torch.einsum("aekn,ane->ake", w[:, radial_index][mu_i],
                                A[:, :, rows])

        return transform, remapped, table

    def transform(A, mu_i):
        return torch.einsum("ekn,ane->ake", w[radial_index], A[:, :, rows])

    return transform, remapped, table


def coupling_table(p, left, right, L_out, rank, equal=False, canonical=False):
    channels = real_coupling_metainformation(
        A=left, B=right, lmax=p.lmax[rank - 1],
        Lmax=L_out,
        lmax_hist=(p.lmax if p.lmax_hist is None else p.lmax_hist)[rank - 1], max_sum_l=p.max_sum_l[rank - 1],
        keep_parity=p.keep_parity, is_A_B_equal=equal, normalize=True,
    )
    if canonical and not equal:
        channels = drop_swaps(channels, left, right)
    # an l=0 node below the root is a scalar, coupling it is multiplying by it
    below = np.array([
        min(operand.get("lowest", operand["l"]).to_numpy()[np.asarray(inds)].min()
            for operand, inds in ((left, li), (right, ri)))
        for li, ri in zip(channels["left_inds"], channels["right_inds"])],
        dtype=np.int64)
    keep = below >= p.Lmin[rank - 1]
    channels = channels[keep].reset_index(drop=True)
    channels["lowest"] = np.minimum(below[keep], channels["l"].to_numpy())
    return channels


def crop(table, readers=(), density=True):
    # readers are the (parent, column) pairs indexing table, renumbered to the rows kept
    keep = set(collect(table, max_l=0, l_p=[[0, 1]])[0].tolist()) if density else set()
    for parent, col in readers:
        keep |= set(np.concatenate(list(parent[col])).tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    where = np.full(len(table), -1)
    where[keep] = np.arange(len(keep))
    for parent, col in readers:
        parent[col] = [where[np.asarray(v)] for v in parent[col]]
    return table.iloc[keep].reset_index(drop=True)


def build_coupling(p, channels, rank, left=None, right=None):
    cg_list = channels["cg_list"]
    left = channels["left_inds"] if left is None else left
    right = channels["right_inds"] if right is None else right
    left_inds = torch.tensor(np.concatenate(list(left)), dtype=torch.int64, device=p.device)
    right_inds = torch.tensor(np.concatenate(list(right)), dtype=torch.int64, device=p.device)
    cg = torch.tensor(np.concatenate(cg_list), dtype=DTYPE, device=p.device).reshape(1, 1, -1)
    scatter = torch.tensor(np.concatenate([[i] * len(c) for i, c in enumerate(cg_list)]), dtype=torch.int64, device=p.device)
    nchannels, ncrop = len(channels), p.n_rad_max[rank - 1]

    def coupling(x, y):
        v = (x[:, :ncrop, left_inds] * y[:, :ncrop, right_inds] * cg).permute(2, 0, 1)
        out = v.new_zeros((nchannels, *v.shape[1:]))
        return out.index_add_(0, scatter, v).permute(1, 2, 0)

    return coupling, (left_inds, right_inds, cg, scatter)


def build_density(p, name, table, rank):
    kept, _, nweights, _ = collect(table, max_l=0, l_p=[[0, 1]])
    kept = torch.tensor(kept, dtype=torch.int64, device=p.device)
    if p.n_mlp_dens is None:
        ndens = (p.fs_parameters if isinstance(p.fs_parameters, int)
                 else len(p.fs_parameters))
    else:
        ndens = p.n_mlp_dens + 1
    c = learn(p, f"density.{name}", len(p.element_map), ndens,
              p.n_rad_max[rank - 1], nweights,
              fill=None if p.init == "grace" else 0.0)
    scale = p.n_rad_max[rank - 1] ** -0.5

    def density(B, mu_i):
        return torch.einsum("aknv,anv->ak", c[mu_i], B[:, :, kept]) * scale

    return density


def build_radial(p, basis, l_tile, nrad_A):
    n_rad_base = p.n_rad_base[0]
    has_l = p.radial == "ln"
    shape = (nrad_A, n_rad_base)
    if has_l:
        shape = (nrad_A, max(p.lmax) + 1, n_rad_base)
    crad = (learn(p, "crad", *shape, fill=0.0) if p.init == "deterministic"
            else learn(p, "crad", *shape,
                       scale=np.sqrt(2 / (nrad_A + n_rad_base))))
    crad = crad.view(nrad_A, -1, n_rad_base)
    if p.init == "deterministic":
        with torch.no_grad():
            for b, start in enumerate(range(0, nrad_A, n_rad_base)):
                n = torch.arange(start, min(start + n_rad_base, nrad_A),
                                 device=p.device)
                crad[..., n, :, n - start] = 0.5 ** b
    if has_l:
        r_tile = l_tile
    else:
        r_tile = torch.zeros_like(l_tile)

    if p.radial == "linear_max" and p.central_rank > 1:
        crad = central_mixture(p, "crad", crad)

        def radial(r, mu_i):
            return torch.einsum("bnlk,bk->bnl", crad()[mu_i],
                                basis(r))[:, :, r_tile]

        return radial

    def radial(r, mu_i):
        return torch.einsum("nlk,bk->bnl", crad, basis(r))[:, :, r_tile]

    return radial


def build_A(p, basis):
    L = max(p.lmax)
    table = pd.DataFrame(
        [[l, m, "", 1 if l % 2 == 0 else -1, l]
         for l in range(L + 1) for m in range(-l, l + 1)],
        columns=["l", "m", "hist", "parity", "sum_of_ls"])
    sph = spherical_harmonics(L, DTYPE, p.device)
    l_tile = torch.tensor(table["l"].to_numpy(), dtype=torch.int64, device=p.device)
    inv_navg = 1.0 / p.avg_n_neigh
    nrad_A = p.n_rad_base[1] if "linear" in p.radial else p.n_rad_max[0]
    radial = build_radial(p, basis, l_tile, nrad_A)
    chem = build_chemistry(p, l_tile, nrad_A)

    def A(batch):
        bv = batch["bond_vector"]
        r = torch.sqrt((bv * bv).sum(1, keepdim=True))
        mu = batch["atomic_mu_i"]
        mu_i, mu_j = mu[batch["ind_i"]], mu[batch["ind_j"]]
        R = radial(r, mu_i)
        Y = sph(bv / r)
        out = R.new_zeros((mu.shape[0], *R.shape[1:]))
        return out.index_add_(0, batch["ind_i"],
                              R * Y[:, None, :] * chem(mu_i, mu_j)) * inv_navg

    return A, table


def build_B_basis(p, A, A_table):
    AA = AAA = AAAA = None
    AA_node = AAA_node = AAAA_node = None

    if p.max_order > 1:
        AA = coupling_table(p, A_table, A_table, p.Lmax[1], 2, equal=True)
    if p.max_order > 2:
        AAA = coupling_table(p, AA, A_table, p.Lmax[2], 3)
    if p.max_order > 3:
        AAAA = coupling_table(p, AA, AA, p.Lmax[3], 4, equal=True)

    readers = []
    if p.max_order > 3:
        AAAA = crop(AAAA)
        readers += [(AAAA, "left_inds"), (AAAA, "right_inds")]
    if p.max_order > 2:
        AAA = crop(AAA)
        readers += [(AAA, "left_inds")]
    if p.max_order > 1:
        AA = crop(AA, readers)

    A_node = ("A", A_table)
    if p.max_order > 1:
        aa, indices = build_coupling(p, AA, 2)
        AA_node = ("AA", AA, indices, A_node, A_node)
    if p.max_order > 2:
        aaa, indices = build_coupling(p, AAA, 3)
        AAA_node = ("AAA", AAA, indices, AA_node, A_node)
    if p.max_order > 3:
        aaaa, indices = build_coupling(p, AAAA, 4)
        AAAA_node = ("AAAA", AAAA, indices, AA_node, AA_node)

    def B(batch):
        A_nlm = A(batch)
        polys = [A_nlm[:, :p.n_rad_max[0]]]
        if p.max_order > 1:
            AA = aa(A_nlm, A_nlm)
            polys.append(AA)
        if p.max_order > 2:
            polys.append(aaa(AA, A_nlm))
        if p.max_order > 3:
            polys.append(aaaa(AA, AA))
        return polys

    return B, [A_node, AA_node, AAA_node, AAAA_node][:p.max_order]


def build_wrad_B_basis(p, A, A_table):
    by_channel = p.radial == "linear_max"
    transforms = {}
    AA = AAA = AAAA = None
    AA_node = AAA_node = AAAA_node = None
    operands = {rank: collect(A_table, max_l=p.lmax[rank - 1],
                              numbering=("l", "parity", "hist"))
                for rank in range(2, p.max_order + 1)}

    def factors(rank, parent, *slots):
        rows, leaves = [], []
        for side, factor in enumerate(slots):
            if factor is None:
                rows.append(None)
                leaves.append(None)
                continue
            name, slot = factor
            transforms[name], inds, table = build_wrad(
                p, name, rank, slot, operands[rank],
                parent if by_channel else None, side)
            rows.append(inds)
            leaves.append((name, table))
        return rows, leaves

    if p.max_order > 1:
        A2 = operands[2][3]
        AA = coupling_table(p, A2, A2, 0, 2, canonical=True)
    if p.max_order > 2:
        A3 = operands[3][3]
        AA3 = coupling_table(p, A3, A3, p.lmax[2], 3, canonical=True)
        AAA = coupling_table(p, AA3, A3, 0, 3, canonical=True)
    if p.max_order > 3:
        A4 = operands[4][3]
        AA4 = coupling_table(p, A4, A4, p.lmax[3], 4, canonical=True)
        AAAA = coupling_table(p, AA4, AA4, 0, 4, canonical=True)

    if p.max_order > 3:
        AAAA = crop(AAAA)
        AA4 = crop(AA4, [(AAAA, "left_inds"), (AAAA, "right_inds")], density=False)
    if p.max_order > 2:
        AAA = crop(AAA)
        AA3 = crop(AA3, [(AAA, "left_inds")], density=False)
    if p.max_order > 1:
        AA = crop(AA)

    if p.max_order > 1:
        rows, (left, right) = factors(2, AA, ("A2_1", 1), ("A2_2", 2))
        aa, indices = build_coupling(p, AA, 2, *rows)
        AA_node = ("AA", AA, indices, left, right)
    if p.max_order > 2:
        rows, (left, right) = factors(3, AA3, ("A3_1", 1), ("A3_2", 2))
        aa3, indices = build_coupling(p, AA3, 3, *rows)
        AA3_node = ("AA3", AA3, indices, left, right)
        rows, (_, right) = factors(3, AAA, None, ("A3_3", 3))
        aaa, indices = build_coupling(p, AAA, 3, *rows)
        AAA_node = ("AAA", AAA, indices, AA3_node, right)
    if p.max_order > 3:
        halves = [AA4, AA4]
        if by_channel:
            # each invariant gets its own copy of the AA4 rows it reads, so build_wrad keys on the invariant too
            for side, col in enumerate(("left_inds", "right_inds")):
                copies, reads, start = [], [], 0
                for c, rows in enumerate(AAAA[col]):
                    copies.append(AA4.iloc[rows].assign(invariant=c))
                    reads.append(np.arange(start, start + len(rows)))
                    start += len(rows)
                halves[side] = pd.concat(copies, ignore_index=True)
                AAAA[col] = reads
        rows, (left, right) = factors(4, halves[0], ("A4_1", 1), ("A4_2", 2))
        aa4_1, indices = build_coupling(p, halves[0], 4, *rows)
        half_1 = ("AA4_1", halves[0], indices, left, right)
        rows, (left, right) = factors(4, halves[1], ("A4_3", 3), ("A4_4", 4))
        aa4_2, indices = build_coupling(p, halves[1], 4, *rows)
        half_2 = ("AA4_2", halves[1], indices, left, right)
        aaaa, indices = build_coupling(p, AAAA, 4)
        AAAA_node = ("AAAA", AAAA, indices, half_1, half_2)

    def B(batch):
        mu_i = batch["atomic_mu_i"]
        A_nlm = A(batch)
        radial = {name: transform(A_nlm, mu_i)
                  for name, transform in transforms.items()}
        polys = [A_nlm[:, :p.n_rad_max[0]]]
        if p.max_order > 1:
            polys.append(aa(radial["A2_1"], radial["A2_2"]))
        if p.max_order > 2:
            polys.append(aaa(aa3(radial["A3_1"], radial["A3_2"]),
                             radial["A3_3"]))
        if p.max_order > 3:
            polys.append(aaaa(aa4_1(radial["A4_1"], radial["A4_2"]),
                              aa4_2(radial["A4_3"], radial["A4_4"])))
        return polys

    return B, [("A", A_table), AA_node, AAA_node, AAAA_node][:p.max_order]


def build_fs_embedding(p, B, densities):
    learned = isinstance(p.fs_parameters, int)
    if learned:
        n = p.fs_parameters
        a = torch.ones(n, dtype=DTYPE, device=p.device)
        nonlinear = torch.arange(n, device=p.device) > 0
        linear_slot = torch.full((1,), 2.0, dtype=DTYPE, device=p.device)
        logm = learn(p, "fs.logm", n - 1, fill=0.0)
        with torch.no_grad():
            logm.copy_(torch.log(torch.tensor(np.geomspace(0.3, 2.0, n - 1))))
    else:
        fs = torch.tensor(p.fs_parameters, dtype=DTYPE, device=p.device)
        a, mexp = fs[:, 0], fs[:, 1]
        nonlinear = (mexp - 1.0).abs() >= 1e-10
        m0 = torch.where(nonlinear, mexp, 2.0)

    def embedding(batch):
        m = torch.cat([linear_slot, torch.exp(logm)]) if learned else m0
        m = torch.where((m - 1.0).abs() < 1e-6, 1.0 + 1e-6, m)
        nx = 1.0 / m
        xgain, ygain = nx ** (nx / (1.0 - nx)), nx ** (1.0 / (1.0 - nx))
        blocks = B(batch)
        mu_i = batch["atomic_mu_i"]
        rho = sum(density(block, mu_i)
                  for density, block in zip(densities, blocks))
        r = torch.where(nonlinear, rho, 0.0)
        sign = torch.where(r >= 0, 1.0, -1.0)
        arho = sign * r
        decay = torch.exp(-arho)
        shifted = sign * ((xgain * decay + arho) ** m - ygain * decay)
        return (torch.where(nonlinear, shifted, rho) * a).sum(1, keepdim=True)

    return embedding


def build_mlp_embedding(p, B, densities):
    activate = getattr(torch.nn.functional, p.activation)
    widths = [p.n_mlp_dens, *p.mlp_hidden, 1]
    layers = []
    for i, (nin, nout) in enumerate(zip(widths, widths[1:])):
        name = f"mlp.{i}"
        if p.init != "deterministic":
            w = learn(p, name, nin, nout, scale=nin ** -0.5)
        elif i == len(widths) - 2:
            w = learn(p, name, nin, nout, fill=1.0)
        else:
            w = learn_diagonal(p, name, nin, nout)
        layers.append(w)

    def embedding(batch):
        blocks = B(batch)
        mu_i = batch["atomic_mu_i"]
        rho = sum(density(block, mu_i)
                  for density, block in zip(densities, blocks))
        x = rho[:, 1:]
        for w in layers[:-1]:
            x = activate(x @ w)
        return x @ layers[-1] + rho[:, :1]

    return embedding


def build_energy(p):
    if p.seed is not None:
        torch.manual_seed(p.seed)
    p.keep_parity = [[l, q] for l in range(max(p.lmax) + 1)
                     for q in (((-1) ** l,) if p.parity == "real" else (-1, 1))]

    basis = (build_clamped_beam(p.n_rad_base[0], p.rcut, p.rclamp, p.device)
             if p.basis_type == "ClampedBeam" else
             build_sbessel(p.n_rad_base[0], p.rcut))

    A, A_table = build_A(p, basis)

    if "linear" in p.radial:
        B, roots = build_wrad_B_basis(p, A, A_table)
    else:
        B, roots = build_B_basis(p, A, A_table)

    densities = [build_density(p, name, node[1], rank) for rank, (name, node)
                 in enumerate(zip(("A", "AA", "AAA", "AAAA"), roots), start=1)]
    if p.n_mlp_dens is None:
        embedding = build_fs_embedding(p, B, densities)
    else:
        embedding = build_mlp_embedding(p, B, densities)
    if p.zbl:
        zbl = build_zbl(p)

    def energy(batch):
        e = embedding(batch) * p.out_scale + p.out_shift
        return e + zbl(batch) if p.zbl else e

    return energy, roots


def energy_forces(energy, batch):
    # func.grad because AOTAutograd refuses the double backward autograd.grad needs
    def energy_scalar(bond_vector):
        atomic_energy = energy({**batch, "bond_vector": bond_vector})
        return atomic_energy.sum(), atomic_energy

    energy_grad, atomic_energy = func.grad(energy_scalar, has_aux=True)(
        batch["bond_vector"])
    total_energy = atomic_energy.new_zeros(
        (batch["n_struct_total"], 1)
    ).index_add_(0, batch["map_atoms_to_structure"], atomic_energy)

    total_forces = energy_grad.new_zeros(
        (batch["atomic_mu_i"].shape[0], 3)
    ).index_add_(0, batch["ind_i"], energy_grad
    ).index_add_(0, batch["ind_j"], energy_grad, alpha=-1)

    return total_energy, total_forces


# --- loading ---

def weight_df(train_df, test_df, DE, DF):
    datasets = [train_df] + ([] if test_df is None else [test_df])
    if DE is not None:
        df = pd.concat(datasets, ignore_index=True)
        df["energy_corrected_per_atom"] = df["energy_corrected"] / df["ase_atoms"].map(len)
        call_grace_no_tf("data/process_df.py").compute_convexhull_dist(
            df, energy_per_atom_column="energy_corrected_per_atom",
            verbose=False, max_allowed_elements=None)
        distances = df["e_chull_dist_per_atom"].to_numpy()
        distances = (np.split(distances, [len(train_df)]) if test_df is not None
                     else [distances])
    else:
        distances = [np.zeros(len(dataset)) for dataset in datasets]
    for dataset, distance in zip(datasets, distances):
        energy_weight = (np.ones(len(dataset)) if DE is None
                         else 1 / (distance + abs(DE))**2)
        dataset["energy_weight"] = energy_weight * len(dataset) / energy_weight.sum()
        dataset["force_weight"] = [ew / (np.sum(np.asarray(f)**2, axis=1) + abs(DF))
                                   for ew, f in zip(dataset["energy_weight"], dataset["forces"])]
        dataset["force_weight"] *= (sum(map(len, dataset["ase_atoms"]))
                                    / sum(map(np.sum, dataset["force_weight"])))


def batch_indices(nbonds, natoms, batch_size, rank, nreplicas):
    nbatches = max(1, math.ceil(math.ceil(len(natoms) / nreplicas) / batch_size))
    order = np.arange(len(natoms))[rank::nreplicas]
    rank_batches = max(1, math.ceil(len(order) / batch_size))
    by_size = order[np.lexsort((-natoms[order], -nbonds[order]))]
    plan = [by_size[i::rank_batches] for i in range(rank_batches)]
    return plan + [np.zeros(0, dtype=np.int64)] * (nbatches - len(plan))


def load_batches(df, rcut, element_map, batch_size, rank, nreplicas, device,
                 report, tag):
    from matscipy import neighbours

    bond_vector, ind_i, ind_j, mu_i = [], [], [], []
    for atoms in df["ase_atoms"]:
        if (atoms.get_pbc() == 0).all():
            atoms = atoms.copy()
            pos = atoms.get_positions()
            max_d = np.max(np.linalg.norm(pos - pos[0], axis=1))
            atoms.set_cell(np.eye(3) * ((max_d + rcut) * 2))
            atoms.center()
            atoms.set_pbc(True)
        i, j, d = neighbours.neighbour_list("ijD", atoms, cutoff=rcut)
        bond_vector.append(d)
        ind_i.append(i)
        ind_j.append(j)
        mu_i.append([element_map[x] for x in atoms.get_chemical_symbols()])
    natoms = np.array([len(m) for m in mu_i])
    nbonds = np.array([len(i) for i in ind_i])
    true_energy = np.asarray(df["energy_corrected"], dtype=np.float64)
    true_force = list(df["forces"])
    energy_weight = (df["energy_weight"].to_numpy() if "energy_weight" in df
                     else np.ones(len(df)))
    force_weight = (list(df["force_weight"]) if "force_weight" in df
                    else [np.ones(n) for n in natoms])

    plans = [batch_indices(nbonds, natoms, batch_size, r, nreplicas)
             for r in range(nreplicas)]
    max_bonds = max(int(nbonds[idx].sum()) for plan in plans for idx in plan)
    max_atoms = max(int(natoms[idx].sum()) for plan in plans for idx in plan)
    max_struct = batch_size + 1
    nbatches = len(plans[rank]) * nreplicas
    report("[%s] batch capacity: %d bonds, %d atoms, %d structures"
           % (tag, max_bonds, max_atoms, max_struct))
    report("[%s] dataset stats: num. batches: %d | " % (tag, nbatches) + " | ".join(
        "num. real %s: %d (%+.2f%%)"
        % (what, real, 100 * (padded * nbatches / float(real) - 1))
        for what, real, padded in (("structures", len(natoms), max_struct - 1),
                                   ("atoms", natoms.sum(), max_atoms),
                                   ("neighbours", nbonds.sum(), max_bonds))))

    batches, counts = [], []
    for idx in plans[rank]:
        slot_natoms = natoms[idx]
        nstruct, nat, nb = len(idx), int(slot_natoms.sum()), int(nbonds[idx].sum())
        offset = np.cumsum(slot_natoms) - slot_natoms
        # pad bonds sit at 2*rcut off the last atom slot, pad atoms in the last structure slot
        batch = {
            "bond_vector": np.tile([2.0 * rcut, 0.0, 0.0], (max_bonds, 1)),
            "ind_i": np.full(max_bonds, max_atoms - 1, dtype=np.int64),
            "ind_j": np.full(max_bonds, max_atoms - 1, dtype=np.int64),
            "atomic_mu_i": np.zeros(max_atoms, dtype=np.int64),
            "map_atoms_to_structure": np.full(max_atoms, max_struct - 1, dtype=np.int64),
            "true_energy": np.zeros((max_struct, 1)),
            "true_force": np.zeros((max_atoms, 3)),
            "energy_weight": np.zeros((max_struct, 1)),
            "force_weight": np.zeros((max_atoms, 1)),
        }
        if nstruct:
            batch["bond_vector"][:nb] = np.concatenate([bond_vector[s] for s in idx])
            batch["ind_i"][:nb] = np.concatenate([ind_i[s] + o for s, o in zip(idx, offset)])
            batch["ind_j"][:nb] = np.concatenate([ind_j[s] + o for s, o in zip(idx, offset)])
            batch["atomic_mu_i"][:nat] = np.concatenate([mu_i[s] for s in idx])
            batch["map_atoms_to_structure"][:nat] = np.repeat(np.arange(nstruct), slot_natoms)
            batch["true_energy"][:nstruct, 0] = true_energy[idx]
            batch["true_force"][:nat] = np.concatenate([true_force[s] for s in idx])
            batch["energy_weight"][:nstruct, 0] = energy_weight[idx]
            batch["force_weight"][:nat, 0] = np.concatenate([force_weight[s] for s in idx])
        batches.append({"n_struct_total": max_struct, **{
            key: torch.tensor(value, device=device) for key, value in batch.items()}})
        counts.append((nstruct, nat))
    totals = all_reduce(torch.tensor(np.array(counts).sum(0), dtype=DTYPE, device=device))
    # Keep this a Python float so torch.compile does not put a NumPy scalar on CPU.
    return batches, totals, float(nbonds.sum() / natoms.sum())


def all_reduce(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


# --- the compiled batch loss ---

METRIC_KEYS = ("loss/energy", "loss/forces", "loss/l2",
               "abs/depa", "sqr/depa", "abs/df", "sqr/df")


def build_batch_loss(energy, params, device, energy_delta, forces_delta, l2_reg):
    energy_delta = torch.tensor(energy_delta, dtype=DTYPE, device=device)
    forces_delta = torch.tensor(forces_delta, dtype=DTYPE, device=device)
    l2_reg = torch.tensor(l2_reg, dtype=DTYPE, device=device)
    unregularized = {"fs.logm"} | {name for name in params if name.startswith("mlp.") and name != "mlp.0"}
    regularized = [w for name, w in params.items() if name not in unregularized]

    def batch_loss(batch, nstruct, natoms, energy_weight, forces_weight,
                   l2_scale):
        total_energy, total_forces = energy_forces(energy, batch)

        # the last structure slot carries the padding's constant shift, so its row is dropped
        nat_per_struct = torch.zeros_like(total_energy).index_add_(
            0, batch["map_atoms_to_structure"], torch.ones_like(batch["true_force"][:, :1]))
        depa = ((total_energy - batch["true_energy"])
                / nat_per_struct.clamp(min=1.0))[:-1]
        df = total_forces - batch["true_force"]
        huber_depa = functional.huber_loss(depa, torch.zeros_like(depa),
                                           delta=energy_delta, reduction="none")
        huber_df = functional.huber_loss(df, torch.zeros_like(df),
                                         delta=forces_delta, reduction="none")

        metrics = torch.stack([
            energy_weight * (batch["energy_weight"][:-1] * huber_depa).sum()
                / nstruct,
            forces_weight * (batch["force_weight"] * huber_df).sum()
                / (natoms * 3.0),
            sum((w * w).sum() for w in regularized) * l2_reg * l2_scale
                * (energy_weight + forces_weight),
            depa.abs().sum(),
            (depa * depa).sum(),
            df.abs().sum(),
            (df * df).sum(),
        ])
        return metrics[:3].sum(), metrics

    return batch_loss


# --- BFGS ---

def get_flat_trainable_variables(params):
    return torch.cat([w.detach().reshape(-1) for w in params.values()])


def set_flat_trainable_variables(params, flat):
    offset = 0
    with torch.no_grad():
        for w in params.values():
            w.copy_(flat[offset:offset + w.numel()].view_as(w))
            offset += w.numel()


# --- export ---

def flat(a):
    return [float(v) for v in np.asarray(a).reshape(-1)]


def monomials(node, channel, cache):
    # one channel of a tree node as [(factors, coeff)], a factor being (leaf, l, m, row, radial)
    key = (id(node), channel)
    if key not in cache:
        if len(node) == 2:
            name, table = node
            row = table.iloc[channel]
            cache[key] = [(((name, int(row["l"]), int(row["m"]), channel,
                             row.get("radial")),), 1.0)]
        else:
            _, _, indices, left, right = node
            if ("triplets", id(node)) not in cache:
                triplets = {}
                for i, j, c, s in zip(*(t.reshape(-1).tolist() for t in indices)):
                    triplets.setdefault(s, []).append((i, j, c))
                cache[("triplets", id(node))] = triplets
            cache[key] = [(fl + fr, cg * cl * cr)
                          for i, j, cg in cache[("triplets", id(node))][channel]
                          for fl, cl in monomials(left, i, cache)
                          for fr, cr in monomials(right, j, cache)]
    return cache[key]


def basis_functions(p, roots):
    coefficients = [w.detach() for name, w in p.params.items()
                    if name.startswith("density.")]
    cache, out = {}, []
    for node, w in zip(roots, coefficients):
        invariants = collect(node[1], max_l=0, l_p=[[0, 1]])[0]
        for column, channel in enumerate(invariants.tolist()):
            expansion = [(sorted(f), c) for f, c in monomials(node, channel, cache)]
            (ls,) = {tuple(l for _, l, _, _, _ in f) for f, _ in expansion}
            radials = ()
            if p.wrad and len(ls) > 1:
                (radials,) = {tuple((name, l, int(radial))
                                    for name, l, _, _, radial in factors)
                              for factors, _ in expansion}
            ms = [m for f, _ in expansion for _, _, m, _, _ in f]
            cgs = [c for _, c in expansion]
            out += [(list(ls), ms, cgs, w, column, n, list(radials))
                    for n in range(w.shape[2])]
    out.sort(key=lambda x: (len(x[0]), x[5], x[0]))
    return out


def header(p):
    crad = p.params["crad"].detach().numpy()
    crad = crad.reshape(crad.shape[0], -1, crad.shape[-1])

    z = p.params["z"].detach().numpy()
    if "z_proj" in p.params:      # a factored Zn, fold the projection in
        proj = p.params["z_proj"].detach().numpy()
        Z = z @ (proj * proj.shape[0] ** -0.5)
    elif z.ndim == 1:             # one weight per element, all radials alike
        Z = np.repeat(z[:, None], crad.shape[0], axis=1)
    else:
        Z = z                     # already indexed by radial channel and by l

    mlp = [w.detach().numpy() for name, w in p.params.items()
           if name.startswith("mlp.")]
    if mlp:
        widths = [w.shape[-2] for w in mlp] + [mlp[-1].shape[-1]]
        emb = {"type": "EmbeddedLinMLP", "activation": p.activation, "rms_norm": False,
               "layers_config": [int(v) for v in widths],
               "params": [v for w in mlp for v in flat(w)]}
    else:
        fs = p.fs_parameters
        if "fs.logm" in p.params:
            m = np.exp(p.params["fs.logm"].detach().numpy())
            m = np.where(np.abs(m - 1.0) < 1e-6, 1.0 + 1e-6, m)
            fs = [[1.0, 1.0]] + [[1.0, v] for v in m]
        emb = {"type": "FinnisSinclairShiftedScaled", "params": flat(fs)}

    radial = {
        "radbasename": p.basis_type, "n_rad_base": int(crad.shape[-1]),
        "nradmax": max(int(w.shape[2]) for name, w in p.params.items()
                       if name.startswith("density.")),
        "cutoff": float(p.rcut),
        "crad": flat(crad), "crad_shape": [int(s) for s in crad.shape],
        "Z": flat(Z), "Z_shape": [int(s) for s in Z.shape],
    }
    if p.zbl:
        radial["subtract_zbl"] = True
    if p.basis_type == "ClampedBeam":
        radial["rclamp"] = float(p.rclamp)
    return {
        "elements": [e for e, _ in sorted(p.element_map.items(),
                                          key=lambda kv: kv[1])],
        "nnorm": float(1.0 / p.avg_n_neigh),
        "emb_spec": emb,
        "radial_basis": radial,
    }


def export_to_yaml(p, roots, filename):
    # the format has no central element axis, so it would read a wrong model
    if p.central_rank > 1 or p.chemistry == "Zinl":
        raise SystemExit(f"radial={p.radial!r} chemistry={p.chemistry!r} "
                         f"has a central element axis the format cannot store")

    top = header(p)
    functions = basis_functions(p, roots)

    # a model without mixing reads raw radials, ns = n + 1 in every slot
    row_number = {(idx_poly, factor_idx): n + 1
                  for idx_poly, (ls, _, _, _, _, n, _) in enumerate(functions)
                  for factor_idx in range(len(ls))}
    if p.wrad:
        wrad = {}
        for name, weight in p.wrad.items():
            weight = weight.detach().numpy()
            wrad[name] = weight.reshape(-1, weight.shape[-1])
        nrad_A = top["radial_basis"]["crad_shape"][0]
        owned_rows = []
        for idx_poly, (ls, _, _, _, _, n, radials) in enumerate(functions):
            for factor_idx, l in enumerate(ls):
                if radials:
                    name, _, base = radials[factor_idx]
                    row = base + n
                    if p.radial == "linear":
                        row %= len(wrad[name])
                    values = wrad[name][row]
                else:
                    values = np.zeros(nrad_A)
                    values[n] = 1.0
                owned_rows.append((l, idx_poly, factor_idx, values))
        owned_rows.sort(key=lambda item: item[0])
        row_number = {(idx_poly, factor_idx): idx + 1
                      for idx, (_, idx_poly, factor_idx, _) in enumerate(owned_rows)}
        rows = [values for _, _, _, values in owned_rows]
        top["radial_basis"]["wrad"] = flat(rows)
        top["radial_basis"]["wrad_shape"] = list(np.shape(rows))
        lcount = [0] * (owned_rows[-1][0] + 1)
        for l, _, _, _ in owned_rows:
            lcount[l] += 1
        top["radial_basis"]["wrad_lcount"] = lcount
    top["shift"] = float(p.out_shift)
    top["scale"] = float(p.out_scale)

    # only mu0 and coeff differ by element, so the rest is built once
    polynomials = []
    for idx_poly, (ls, ms, cgs, w, column, n, radials) in enumerate(functions):
        ns = [row_number[(idx_poly, factor_idx)]
              for factor_idx in range(len(ls))]
        fn = {"rank": len(ls), "ls": ls, "ns": ns, "ms_combs": ms,
              "gen_cgs": flat(cgs), "ndensity": w.shape[1]}
        polynomials.append((fn, w, column, n))

    dump = lambda v: yaml.safe_dump(v, default_flow_style=True, sort_keys=False,
                                    width=999999).removesuffix("...\n")
    with open(filename, "w") as f:
        for key, value in top.items():
            f.write(f"{key}: " + dump(value))
        f.write("functions:\n")
        for el in range(len(top["elements"])):
            f.write(f"    {el}:\n")
            for fn, w, column, n in polynomials:
                f.write("     - " + dump({"mu0": el, **fn, "coeff": flat(
                    w[el, :, n, column] * w.shape[2] ** -0.5)}) + "\n")


def export(checkpoint_file, filename):
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    p = Potential(**checkpoint["potential"], device=torch.device("cpu"))
    _, roots = build_energy(p)
    with torch.no_grad():
        for w, saved in zip(p.params.values(), checkpoint["model"].values()):
            w.copy_(saved)
    export_to_yaml(p, roots, filename)


# --- driver ---

def rendezvous(host_command, job_id):
    master_addr = subprocess.check_output(
        host_command, text=True).splitlines()[0]
    master_port = 20000 + zlib.crc32(job_id.encode()) % 20000
    return f"tcp://{master_addr}:{master_port}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "export":
        export(argv[1], argv[2] if len(argv) > 2 else "saved_model.yaml")
        return 0
    restart = False
    if "-rl" in argv:
        restart = "checkpoint"
    elif "-r" in argv:
        restart = "checkpoint.best_test_loss"
    filenames = [a for a in argv if a not in ("-r", "-rl")]
    if len(filenames) != 1:
        print(__doc__.strip())
        return 1

    rank, world_size, local_rank, init_method = 0, 1, 0, None
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size > 1:
            init_method = (f"tcp://{os.environ['MASTER_ADDR']}:"
                           f"{os.environ['MASTER_PORT']}")
    elif "FLUX_TASK_RANK" in os.environ:
        rank = int(os.environ["FLUX_TASK_RANK"])
        world_size = int(os.environ["FLUX_JOB_SIZE"])
        if world_size > 1:
            init_method = rendezvous(
                ["flux", "hostlist", "-n", "0", "local"],
                os.environ.get("FLUX_JOB_ID", "0"))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        if world_size > 1:
            init_method = rendezvous(
                ["scontrol", "show", "hostnames",
                 os.environ["SLURM_JOB_NODELIST"]],
                os.environ.get("SLURM_JOB_ID", "0") + ":" +
                os.environ.get("SLURM_STEP_ID", "0"))

    with open(filenames[0]) as f:
        args_yaml = yaml.safe_load(f)

    data_config = args_yaml["data"]
    fit_config = args_yaml.get("fit", {})
    potential_kwargs = dict(args_yaml.get("potential", {}).get("kwargs", {}))
    loss = fit_config.get("loss", {})
    switch = loss.get("switch", {})
    weighting = fit_config.get("weighting")
    if weighting is not None and weighting.get("type") not in ("energy_based", "force_based"):
        raise ValueError(f"unknown weighting {weighting['type']!r}")
    if fit_config.get("optimizer", "BFGS").lower() != "bfgs":
        raise SystemExit("peacemaker is BFGS only")
    test_filename = data_config.get("test_filename")
    rcut = args_yaml.get("cutoff", 6.0)
    subtract_zbl = data_config.get("subtract_zbl", False)
    seed = args_yaml.get("seed", 1)
    batch_size = fit_config.get("batch_size", 16)
    test_batch_size = fit_config.get("test_batch_size", batch_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend, device = "nccl", torch.device(f"cuda:{local_rank}")
    else:
        backend, device = "gloo", torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world_size, init_method=init_method)

    output_dir = "."
    if potential_kwargs.get("init") == "grace":
        output_dir = os.path.join(output_dir, "seed", str(seed))
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    def report(message):
        if rank != 0:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} I - {message}"
        print(line, flush=True)
        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            print(line, file=f)

    if restart and not os.path.exists(
            os.path.join(output_dir, "checkpoints", restart)):
        report(f"restart: nothing to restart from in {output_dir}/checkpoints, "
               f"starting from scratch")
        restart = False

    report(f"Device: {device}")
    report(f"Number of replicas: {world_size}")
    report(f"Loading training data from {data_config['filename']}")
    if subtract_zbl:
        report(f"ZBL core repulsion in the model: cutoff={rcut}")
    train_df = pd.read_pickle(data_config["filename"])
    test_df = pd.read_pickle(test_filename) if test_filename else None
    if weighting is not None:
        weight_df(train_df, test_df,
                  weighting.get("DE", 1.0) if weighting["type"] == "energy_based" else None,
                  weighting.get("DF", 1.0))
    report(f"Train Dataset size: {len(train_df)}")
    if test_df is not None:
        report(f"Test Dataset size: {len(test_df)}")
    symbols = sorted({s for a in train_df["ase_atoms"] for s in a.get_chemical_symbols()})
    element_map = {s: i for i, s in enumerate(symbols)}
    report(f"Elements mapping: {element_map}")
    train, train_totals, avg_n_neigh = load_batches(
        train_df, rcut, element_map, batch_size, rank, world_size, device,
        report, "TRAIN")
    report(f"Average number of neighbors (computed): {avg_n_neigh}")
    test, test_totals, _ = (None, None, None) if test_df is None else load_batches(
        test_df, rcut, element_map, test_batch_size, rank, world_size, device,
        report, "TEST")

    model = Potential(
        element_map, **potential_kwargs,
        rcut=rcut, seed=seed, avg_n_neigh=avg_n_neigh, zbl=subtract_zbl,
        device=device)
    if isinstance(model.n_rad_base, int):
        model.n_rad_base = (model.n_rad_base, model.n_rad_base)
    if model.chemistry is None:
        model.chemistry = "Zn" if "linear" in model.radial else "Zembed"
    if model.mlp_hidden is None and model.n_mlp_dens is not None:
        model.mlp_hidden = (model.n_mlp_dens,)
    (energy, _), params = build_energy(model), model.params

    batch_loss = build_batch_loss(
        energy, params, device, loss.get("energy", {}).get("delta", 0.01),
        loss.get("forces", {}).get("delta", 0.01), loss.get("l2_reg", 0.0))
    # Inductor miscompiles the backward on CPU, so CPU runs eager
    if fit_config.get("jit_compile", False) and device.type == "cuda":
        batch_loss = torch.compile(batch_loss, dynamic=False,
                                   mode="reduce-overhead")
        report("JIT compilation: True (mode=reduce-overhead)")

    energy_weight = torch.tensor(loss.get("energy", {}).get("weight", 1.0),
                                 dtype=DTYPE, device=device)
    forces_weight = torch.tensor(loss.get("forces", {}).get("weight", 100.0),
                                 dtype=DTYPE, device=device)
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 100)
    switch_after = switch.get("after_iter")
    switch_energy_weight = switch.get("energy", {}).get("weight")
    switch_forces_weight = switch.get("forces", {}).get("weight")
    best_test_loss, switched = math.inf, False

    def run_pass(batches, totals, gradient):
        nbatches = len(batches)
        denominator = totals.clamp(min=1.0) / nbatches
        nstruct, natoms = (float(x) for x in totals)
        l2_scale = torch.tensor(1.0 / world_size if gradient else 0.0,
                                dtype=DTYPE, device=device)
        acc_metrics = torch.zeros(len(METRIC_KEYS), dtype=DTYPE, device=device)
        grad = None
        t_start = time.perf_counter()

        for batch in batches:
            torch.compiler.cudagraph_mark_step_begin()
            total_loss, metrics = batch_loss(
                batch, denominator[0], denominator[1],
                energy_weight, forces_weight, l2_scale)
            if gradient:
                total_loss.backward()
                # the cat copies each gradient out of the cudagraph pool
                g = torch.cat([w.grad.reshape(-1) for w in params.values()])
                grad = g if grad is None else grad + g
                for w in params.values():
                    w.grad = None
            acc_metrics += metrics.detach()
            del total_loss, metrics

        all_reduce(acc_metrics)
        summed = dict(zip(METRIC_KEYS, acc_metrics.detach().cpu().numpy()))
        losses = {key: float(summed[key]) / nbatches for key in METRIC_KEYS[:3]}
        return {
            **losses,
            "total_loss": sum(losses.values()),
            "mae/depa": summed["abs/depa"] / nstruct,
            "rmse/depa": math.sqrt(summed["sqr/depa"] / nstruct),
            "mae/f_comp": summed["abs/df"] / natoms / 3,
            "rmse/f_comp": math.sqrt(summed["sqr/df"] / natoms / 3),
            "total_time/per_atom": (time.perf_counter() - t_start) / max(natoms, 1),
            "n_batches": nbatches,
        }, None if grad is None else all_reduce(grad / nbatches)

    def save_checkpoint(name, epoch):
        if rank:
            return
        torch.save({
            "model": {k: v.detach().cpu() for k, v in params.items()},
            "potential": {f.name: getattr(model, f.name) for f in fields(model)
                          if f.init and f.name != "device"},
            "epoch": epoch,
            "best_test_loss": best_test_loss,
            "switched": switched,
        }, os.path.join(output_dir, "checkpoints", name))

    def load_checkpoint(name):
        nonlocal best_test_loss, switched
        if world_size > 1:
            dist.barrier()  # rank 0 may have just written it
        filename = os.path.join(output_dir, "checkpoints", name)
        if not os.path.exists(filename):
            return None
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        with torch.no_grad():
            for w, saved in zip(params.values(), checkpoint["model"].values()):
                w.copy_(saved.to(w.device))
        best_test_loss, switched = checkpoint["best_test_loss"], checkpoint["switched"]
        return checkpoint["epoch"]

    def epoch_end(epoch, train_metrics):
        nonlocal best_test_loss
        with torch.no_grad():
            test_metrics = (run_pass(test, test_totals, False)[0]
                            if test is not None else None)

        both = lambda key, scale=1.0, fmt=".3e": (
            f"{train_metrics[key] * scale:{fmt}}" if test_metrics is None else
            f"{train_metrics[key] * scale:{fmt}} ({test_metrics[key] * scale:{fmt}})")
        report(f"Iteration #{epoch}/{epochs} TRAIN(TEST): " + " ".join(
            f"{key}: {both(key)}" for key in
            ("total_loss", "mae/depa", "rmse/depa", "mae/f_comp", "rmse/f_comp")
        ) + f" Time(mcs/at): {both('total_time/per_atom', 1e6, '.0f')}")

        if rank == 0:
            for name, metrics in (("train", train_metrics), ("test", test_metrics)):
                if metrics is None:
                    continue
                filename = os.path.join(output_dir, f"{name}_metrics.yaml")
                with open(filename, "at") as f:
                    print("-", json.dumps({**{k: float(v) for k, v in metrics.items()},
                                           "epoch": epoch}), file=f)

        if test_metrics is not None and test_metrics["total_loss"] < best_test_loss:
            best_test_loss = test_metrics["total_loss"]
            report(f"New best test loss found ({best_test_loss:.5e}), checkpointing")
            save_checkpoint("checkpoint.best_test_loss", epoch)
        if epoch % checkpoint_freq == 0:
            report("Regular checkpointing")
            save_checkpoint("checkpoint", epoch)

    def switch_loss(epoch=None):
        nonlocal best_test_loss
        if epoch is not None:
            if load_checkpoint("checkpoint.best_test_loss") is not None:
                report("Loading best test loss model")
            save_checkpoint("checkpoint.stage1.best_test_loss", epoch)
            best_test_loss = math.inf  # bests under the old loss are not comparable
        if switch_energy_weight is not None:
            energy_weight.fill_(float(switch_energy_weight))
        if switch_forces_weight is not None:
            forces_weight.fill_(float(switch_forces_weight))
        report(f"Switching loss weights to energy={float(energy_weight)}, "
               f"forces={float(forces_weight)}")

    report("Start BFGS optimization")
    opt_params = fit_config.get("opt_params", {})
    gtol = opt_params.get("gtol", 1e-8)
    c1, c2 = opt_params.get("c1", 1e-4), opt_params.get("c2", 0.9)
    cache = {"x": None, "fval": None, "jac": None, "metrics": None, "neval": 0}
    on_device = lambda a: torch.as_tensor(a, dtype=DTYPE, device=device)

    def eval_at(x):
        if cache["x"] is None or not np.array_equal(x, cache["x"]):
            set_flat_trainable_variables(params, on_device(x))
            metrics, grad = run_pass(train, train_totals, True)
            cache.update(x=x.copy(), metrics=metrics, neval=cache["neval"] + 1,
                         fval=metrics["total_loss"], jac=grad)
        return cache["fval"], cache["jac"]

    f_scipy = lambda x: eval_at(x)[0]
    g_scipy = lambda x: eval_at(x)[1].detach().cpu().numpy()

    epoch = 0
    if restart:
        epoch = load_checkpoint(restart)
        report(f"Restarting from epoch {epoch}")
        # the loss weights come back from input.yaml at stage 1
        if switched:
            switch_loss()     # weights only, the loop banked stage 1 already
    xk = get_flat_trainable_variables(params).cpu().numpy().copy()
    row0 = xk.size * rank // world_size
    row1 = xk.size * (rank + 1) // world_size

    def restart_at(x, H=None):
        cache["x"] = None
        fval, g = eval_at(x)
        g = g.clone()
        # makes the first direction invariant to the loss's overall scale
        grad_l2 = float(torch.linalg.norm(g))
        h0_scale = 1.0 / grad_l2 if grad_l2 else 1.0
        if H is None:
            H = torch.zeros((row1 - row0, x.size), dtype=DTYPE, device=device)
        else:
            H.zero_()
        H.diagonal(row0).fill_(h0_scale)
        return H, fval, fval + float(torch.linalg.norm(g)) / 2, g, h0_scale

    Hk, old_fval, old_old_fval, gfk, gscale = restart_at(xk)
    report(f"Number of trainable parameters: {xk.size}")

    while float(gfk.abs().max()) * gscale > gtol and epoch < epochs:
        if not switched and epoch == switch_after:
            switch_loss(epoch)
            switched = True
            xk = get_flat_trainable_variables(params).cpu().numpy().copy()
            Hk, old_fval, old_old_fval, gfk, gscale = restart_at(xk, Hk)

        pk = torch.zeros_like(gfk)
        pk[row0:row1] = -Hk.mv(gfk)
        all_reduce(pk)
        pk_np, gfk_np = pk.cpu().numpy(), gfk.cpu().numpy()

        ret = line_search_wolfe1(f_scipy, g_scipy, xk, pk_np, gfk_np, old_fval,
                                 old_old_fval, c1=c1, c2=c2,
                                 amin=1e-100, amax=1e100)
        if ret[0] is None:
            ret = line_search_wolfe2(f_scipy, g_scipy, xk, pk_np, gfk_np,
                                     old_fval, old_old_fval, c1=c1, c2=c2,
                                     amax=1e100)
        if ret[0] is None:
            report("Desired error not necessarily achieved due to precision loss.")
            break

        alpha_k, old_fval, old_old_fval = ret[0], ret[3], ret[4]
        sk = alpha_k * pk
        xk = xk + alpha_k * pk_np
        gfkp1 = (on_device(ret[5]) if ret[5] is not None
                 else eval_at(xk)[1].clone())
        yk, gfk = gfkp1 - gfk, gfkp1
        epoch += 1

        if not np.array_equal(cache["x"], xk):
            eval_at(xk)
        epoch_end(epoch, cache["metrics"])
        if float(gfk.abs().max()) * gscale <= gtol:
            break

        rhok_inv = float(yk.dot(sk))
        rhok = 1000.0 if rhok_inv == 0.0 else 1.0 / rhok_inv
        Hyk = torch.zeros_like(yk)
        Hyk[row0:row1] = Hk.mv(yk)
        all_reduce(Hyk)
        Hk.addr_(sk[row0:row1], Hyk, alpha=-rhok)
        Hk.addr_(Hyk[row0:row1], sk, alpha=-rhok)
        Hk.addr_(sk[row0:row1], sk, alpha=rhok * rhok * float(yk.dot(Hyk)) + rhok)

    report(f"BFGS optimization results: iterations={epoch}, "
           f"function/gradient evaluations={cache['neval']}")
    save_checkpoint("checkpoint", epoch)
    if rank == 0:
        report("Exporting checkpoints/checkpoint to saved_model.yaml")
        export(os.path.join(output_dir, "checkpoints", "checkpoint"),
               os.path.join(output_dir, "saved_model.yaml"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
