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

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from ase.data import atomic_numbers
from scipy.optimize import brentq
from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
from scipy.spatial import ConvexHull
from torch import func
from torch.nn import functional

# donated buffers interact badly with the func.grad path under compile
import torch._functorch.config
torch._functorch.config.donated_buffer = False

DTYPE = torch.float64

# couplings (l1, l2, L) couple two l or two couplings to L, invariants to 0, listed by rank and largest l
COUPLINGS = {
    "base": {1: [0],
             2: [(0, 0, 0),
                 (1, 1, 0),
                 (2, 2, 0),
                 (3, 3, 0),
                 (4, 4, 0),
                 (5, 5, 0)],
             3: [((0, 0, 0), 0, 0),
                 ((1, 1, 0), 0, 0),
                 ((2, 2, 0), 0, 0),
                 ((2, 1, 1), 1, 0),
                 ((2, 2, 2), 2, 0),
                 ((3, 3, 0), 0, 0),
                 ((3, 2, 1), 1, 0),
                 ((3, 3, 2), 2, 0),
                 ((4, 4, 0), 0, 0),
                 ((4, 3, 1), 1, 0),
                 ((4, 2, 2), 2, 0),
                 ((4, 4, 2), 2, 0),
                 ((4, 3, 3), 3, 0),
                 ((4, 4, 4), 4, 0)],
             4: [((0, 0, 0), (0, 0, 0), 0),
                 ((0, 0, 0), (1, 1, 0), 0),
                 ((1, 1, 0), (1, 1, 0), 0),
                 ((0, 0, 0), (2, 2, 0), 0),
                 ((1, 0, 1), (2, 1, 1), 0),
                 ((2, 0, 2), (2, 2, 2), 0),
                 ((1, 1, 0), (2, 2, 0), 0), ((2, 1, 1), (2, 1, 1), 0),
                 ((2, 2, 0), (2, 2, 0), 0), ((2, 2, 2), (2, 2, 2), 0),
                 ((0, 0, 0), (3, 3, 0), 0),
                 ((1, 0, 1), (3, 2, 1), 0),
                 ((2, 0, 2), (3, 3, 2), 0),
                 ((1, 1, 2), (3, 1, 2), 0),
                 ((1, 1, 0), (3, 3, 0), 0), ((3, 1, 2), (3, 1, 2), 0),
                 ((2, 1, 1), (3, 2, 1), 0), ((2, 2, 2), (3, 1, 2), 0),
                 ((3, 1, 2), (3, 3, 2), 0),
                 ((2, 2, 0), (3, 3, 0), 0), ((3, 2, 1), (3, 2, 1), 0), ((3, 2, 3), (3, 2, 3), 0),
                 ((3, 3, 0), (3, 3, 0), 0), ((3, 3, 2), (3, 3, 2), 0),
                 ((0, 0, 0), (4, 4, 0), 0),
                 ((1, 0, 1), (4, 3, 1), 0),
                 ((2, 0, 2), (4, 2, 2), 0),
                 ((2, 0, 2), (4, 4, 2), 0),
                 ((3, 0, 3), (4, 3, 3), 0),
                 ((4, 0, 4), (4, 4, 4), 0),
                 ((1, 1, 2), (4, 2, 2), 0),
                 ((1, 1, 0), (4, 4, 0), 0), ((4, 1, 3), (4, 1, 3), 0),
                 ((2, 1, 1), (4, 3, 1), 0), ((3, 1, 2), (4, 2, 2), 0), ((3, 2, 3), (4, 1, 3), 0),
                 ((3, 1, 2), (4, 4, 2), 0), ((4, 1, 3), (4, 3, 3), 0),
                 ((2, 2, 2), (4, 2, 2), 0),
                 ((2, 2, 0), (4, 4, 0), 0), ((4, 2, 2), (4, 2, 2), 0), ((4, 2, 4), (4, 2, 4), 0),
                 ((3, 2, 1), (4, 3, 1), 0), ((3, 2, 3), (4, 3, 3), 0), ((3, 3, 4), (4, 2, 4), 0),
                 ((4, 2, 2), (4, 4, 2), 0), ((4, 2, 4), (4, 4, 4), 0),
                 ((3, 3, 0), (4, 4, 0), 0), ((3, 3, 2), (4, 4, 2), 0), ((4, 3, 1), (4, 3, 1), 0), ((4, 3, 3), (4, 3, 3), 0),
                 ((4, 4, 0), (4, 4, 0), 0), ((4, 4, 2), (4, 4, 2), 0), ((4, 4, 4), (4, 4, 4), 0)],
             5: [(((0, 0, 0), 0, 0), (0, 0, 0), 0),
                 (((0, 0, 0), 0, 0), (1, 1, 0), 0),
                 (((1, 0, 1), 1, 0), (1, 1, 0), 0),
                 (((0, 0, 0), 0, 0), (2, 2, 0), 0),
                 (((1, 0, 1), 0, 1), (2, 1, 1), 0),
                 (((2, 0, 2), 0, 2), (2, 2, 2), 0),
                 (((1, 0, 1), 1, 0), (2, 2, 0), 0), (((2, 0, 2), 1, 1), (2, 1, 1), 0),
                 (((2, 0, 2), 2, 0), (2, 2, 0), 0), (((2, 0, 2), 2, 2), (2, 2, 2), 0),
                 (((1, 1, 2), 2, 0), (1, 1, 0), 0),
                 (((1, 1, 2), 2, 0), (2, 2, 0), 0), (((2, 1, 1), 1, 2), (2, 2, 2), 0),
                 (((2, 2, 2), 2, 2), (2, 2, 2), 0)]},
    "ln_exclude": {4: [((2, 2, 2), (2, 2, 2), 0),
                       ((4, 2, 4), (4, 4, 4), 0),
                       ((4, 4, 4), (4, 4, 4), 0)],
                   5: [(((2, 0, 2), 2, 2), (2, 2, 2), 0)]},
    "grace_exclude": {4: [((2, 2, 2), (2, 2, 2), 0),
                          ((4, 2, 4), (4, 4, 4), 0)]},
    "grace_include": {4: [((1, 1, 2), (2, 2, 2), 0), ((2, 1, 3), (2, 1, 3), 0),
                          ((1, 1, 2), (3, 3, 2), 0), ((3, 1, 4), (3, 1, 4), 0),
                          ((2, 1, 3), (3, 2, 3), 0), ((2, 2, 4), (3, 1, 4), 0),
                          ((2, 2, 2), (3, 3, 2), 0), ((2, 2, 4), (3, 3, 4), 0),
                          ((3, 3, 4), (3, 3, 4), 0),
                          ((1, 1, 2), (4, 4, 2), 0),
                          ((2, 1, 3), (4, 3, 3), 0), ((3, 1, 4), (4, 2, 4), 0),
                          ((3, 1, 4), (4, 4, 4), 0),
                          ((2, 2, 2), (4, 4, 2), 0), ((2, 2, 4), (4, 4, 4), 0),
                          ((3, 3, 2), (4, 2, 2), 0),
                          ((3, 3, 4), (4, 4, 4), 0)]},
}


@dataclass
class Potential:
    element_map: dict
    device: torch.device = None
    basis_type: str = "SBessel"
    radial: str = "ln"
    chemistry: str = None
    n_rad_base: int | tuple = 20
    n_rad_max: tuple = (16, 16, 8, 4, 2)
    lmax: tuple = (5, 5, 4, 3, 2)
    n_mlp_dens: int = None
    activation: str = "silu"
    max_order: int = 4
    rcut: float = 6.0
    rclamp: float = 0.0
    embedding_size: int = 64
    mlp_hidden: tuple = None           # None is one layer of n_mlp_dens
    fs_parameters: tuple = ((1.0, 1.0), (1.0, 0.5), (1.0, 2), (1.0, 0.75))
    avg_n_neigh: float = 1.0
    out_scale: float = 1.0
    out_shift: float = 0.0
    init: str = "deterministic"
    seed: int = None
    central_rank: int = 1
    zbl: bool = False
    params: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.n_rad_base, int):
            self.n_rad_base = (self.n_rad_base, self.n_rad_base)
        if self.chemistry is None:
            self.chemistry = "Zn" if "linear" in self.radial else "Zembed"
        if self.mlp_hidden is None and self.n_mlp_dens is not None:
            self.mlp_hidden = (self.n_mlp_dens,)


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


def learn_wrad(p, name, slot, nrows, nout, nin):
    # cyclic shifts by slot, so AA's factors do not get equal gradients forever
    if p.init != "deterministic":
        return learn(p, name, nrows, nout, nin, scale=nin ** -0.5)
    w = learn(p, name, nrows, nout, nin, fill=0.0)
    k = torch.arange(nout, device=p.device)
    with torch.no_grad():
        w[:, k, (k + slot - 1) % nin] = 0.5 ** (k // nin).to(DTYPE)
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


# --- the one-particle basis ---

def build_radial(p, basis, nrad_A):
    # R[n, l](r) = sum_k crad[n, l, k] g_k(r), with a single l unless radial is ln
    nl, nk = max(p.lmax) + 1 if p.radial == "ln" else 1, p.n_rad_base[0]
    if p.init == "deterministic":
        crad = learn(p, "crad", nrad_A, nl, nk, fill=0.0)
        n = torch.arange(nrad_A, device=p.device)
        with torch.no_grad():
            crad[n, :, n % nk] = 0.5 ** (n // nk).to(DTYPE)[:, None]
    else:
        crad = learn(p, "crad", nrad_A, nl, nk, scale=np.sqrt(2 / (nrad_A + nk)))
    if p.radial == "linear_max" and p.central_rank > 1:
        crad = central_mixture(p, "crad", crad)
        return lambda r, mu_i: torch.einsum("bnlk,bk->bnl", crad()[mu_i], basis(r))
    return lambda r, mu_i: torch.einsum("nlk,bk->bnl", crad, basis(r))


def build_chemistry(p, nrad_A):
    # Z[mu_i, mu_j, n, l], with size-1 axes for what the chemistry does not depend on
    ne, nl = len(p.element_map), max(p.lmax) + 1
    fill = 1.0 if p.init == "deterministic" else None
    if p.chemistry == "Zembed":
        w = learn(p, "z", ne, p.embedding_size, fill=fill)
        proj = (learn_diagonal(p, "z_proj", p.embedding_size, nrad_A) if p.init == "deterministic"
                else learn(p, "z_proj", p.embedding_size, nrad_A))
        return lambda: (w @ (proj * p.embedding_size ** -0.5))[None, :, :, None]
    if p.radial == "linear_max" and p.central_rank > 1:
        w = central_mixture(p, "z", learn(p, "z", ne, nrad_A, fill=fill))
        return lambda: w()[:, :, :, None]
    stored, full = {"Z": ((ne,), (1, ne, 1, 1)),
                    "Zn": ((ne, nrad_A), (1, ne, nrad_A, 1)),
                    "Znl": ((ne, nrad_A, nl), (1, ne, nrad_A, nl)),
                    "Zinl": ((ne, ne, nrad_A, nl), (ne, ne, nrad_A, nl))}[p.chemistry]
    w = learn(p, "z", *stored, fill=fill)
    return lambda: w.view(full)


def build_A(p, basis):
    # A[i, n, lm] = sum_j R[n, l](r_ij) Y[lm](rhat_ij) Z[mu_i, mu_j, n, l] / n_avg
    lmax = max(p.lmax)
    l_of = torch.tensor([l for l in range(lmax + 1) for _ in range(2 * l + 1)], device=p.device)
    nrad_A = p.n_rad_base[1] if "linear" in p.radial else p.n_rad_max[0]
    Y = spherical_harmonics(lmax, DTYPE, p.device)
    R = build_radial(p, basis, nrad_A)
    Z = build_chemistry(p, nrad_A)

    def per_lm(x):
        return x[:, :, l_of] if x.shape[2] > 1 else x

    def A(batch):
        bv, mu, i, j = batch["bond_vector"], batch["atomic_mu_i"], batch["ind_i"], batch["ind_j"]
        r = torch.sqrt((bv * bv).sum(1, keepdim=True))
        z = Z()
        z = z[mu[i] if z.shape[0] > 1 else 0, mu[j]]
        terms = per_lm(R(r, mu[i])) * Y(bv / r)[:, None, :] * per_lm(z)
        return terms.new_zeros((mu.shape[0], *terms.shape[1:])).index_add_(0, i, terms) * (1.0 / p.avg_n_neigh)

    return A


# --- the many-body basis ---

def build_B(p, A):
    # B[rank][c] = sum over terms of cg a_1[l1 m1] ... a_k[lk mk], as LAMMPS evaluates it, a_f being A after factor f's radial mixing
    lmax = max(l for l, _, _ in COUPLINGS["base"][2])

    # a grid on the sphere exact for products of three harmonics up to lmax
    cos_theta, w_theta = np.polynomial.legendre.leggauss(3 * lmax // 2 + 1)
    phi = np.linspace(0, 2 * np.pi, 3 * lmax + 1, endpoint=False)
    ct, ph = np.meshgrid(cos_theta, phi, indexing="ij")
    st = np.sqrt(1 - ct * ct)
    rhat = np.stack([st * np.cos(ph), st * np.sin(ph), ct], -1).reshape(-1, 3)
    weight = np.repeat(w_theta / 2 / len(phi), len(phi))
    y = spherical_harmonics(lmax, DTYPE, "cpu")(torch.tensor(rhat)).numpy()

    # nice way to get CG
    def gaunt(l1, l2, L):
        # g[M, m1, m2] = <y_LM y_l1m1 y_l2m2> / sum_m1m2 <y_L0 y_l1m1 y_l2m2>^2, <...> the average over the sphere
        # <y_LM y_l1m1 y_l2m2> = sqrt(4 pi) (-1)^(M+m1+m2) real_gaunt(L, l1, l2, M, m1, m2) in sympy.physics.wigner
        g = np.einsum("p,pM,pi,pj->Mij", weight, y[:, L * L:(L + 1) ** 2], y[:, l1 * l1:(l1 + 1) ** 2], y[:, l2 * l2:(l2 + 1) ** 2])
        return g / (g[0] ** 2).sum()

    def coupling_tensor(coupling):
        # C[M, m1, ..., mk] coupling k harmonics to L, g contracted pair by pair
        if isinstance(coupling, int):
            return np.eye(2 * coupling + 1)
        left, right, L = coupling_tensor(coupling[0]), coupling_tensor(coupling[1]), coupling[2]
        g = gaunt(len(left) // 2, len(right) // 2, L)
        return np.tensordot(np.tensordot(g, left, (1, 0)), right, (1, 0))

    ranks, layouts = [], []
    for rank in range(1, p.max_order + 1):
        # the listed couplings
        excluded = COUPLINGS["ln_exclude"].get(rank, []) if p.radial == "ln" else []
        listed = [coupling_tensor(c)[0] for c in COUPLINGS["base"][rank]]
        listed_lmax = max(max(C.shape) for C in listed) // 2
        if rank > 1 and p.lmax[rank - 1] > listed_lmax:
            raise SystemExit(f"rank {rank} couplings are listed up to lmax {listed_lmax}")
        tensors = []
        for c, C in zip(COUPLINGS["base"][rank], listed):
            if max(C.shape) // 2 <= p.lmax[rank - 1] and c not in excluded:
                tensors.append(C)

        # their terms, {((l1, m1), ..., (lk, mk)): cg} for each invariant
        terms = []
        for C in tensors:
            ls = [n // 2 for n in C.shape]
            terms.append({tuple((l, int(m) - l) for l, m in zip(ls, ms)): float(C[tuple(ms)])
                          for ms in np.argwhere(np.abs(C) > 1e-12)})
        invariant, lms, cg = zip(*[(c, lms, g) for c, t in enumerate(terms) for lms, g in t.items()])

        # how each factor f gets a_f from A, and which entry of a_f each term reads
        nout, nin = p.n_rad_max[rank - 1], p.n_rad_base[1]
        weights, cols, rows, term_index = [None] * rank, [None] * rank, [None] * rank, [None] * rank
        for f in range(rank):
            term_index[f] = torch.tensor([l * l + l + m for t in lms for l, m in [t[f]]], device=p.device)
            if p.radial == "linear" and rank > 1:
                # one radial mixing per factor, shared by every invariant and rank
                weights[f] = f"wrad.{f + 1}"
                if weights[f] not in p.params:
                    learn_wrad(p, weights[f], f + 1, 1, nout, nin)
            elif p.radial == "linear_max" and rank > 1:
                # each invariant mixes its own radials, so only the block of A at its l is mixed
                weights[f] = f"wrad.{rank}.{f + 1}"
                learn_wrad(p, weights[f], f + 1, len(terms), nout, nin)
                if p.central_rank > 1:
                    central_mixture(p, weights[f], p.params[weights[f]])
                lf = [C.shape[f] // 2 for C in tensors]
                start = np.cumsum([0] + [2 * l + 1 for l in lf])
                cols[f] = torch.tensor([l * l + m for l in lf for m in range(2 * l + 1)], device=p.device)
                rows[f] = torch.tensor([c for c, l in enumerate(lf) for _ in range(2 * l + 1)], device=p.device)
                term_index[f] = torch.tensor([start[c] + l + m for c, t in zip(invariant, lms) for l, m in [t[f]]],
                                             device=p.device)
        ranks.append((nout, len(terms), torch.tensor(invariant, device=p.device),
                      torch.tensor(cg, dtype=DTYPE, device=p.device), weights, cols, rows, term_index))
        layouts.append((terms, weights))

    def B(batch):
        A_nlm, mu_i = A(batch), batch["atomic_mu_i"]
        out = []
        for nout, n, invariant, cg, weights, cols, rows, term_index in ranks:
            v = cg
            for name, col, row, index in zip(weights, cols, rows, term_index):
                if name is None:
                    a = A_nlm[:, :nout]
                elif p.radial == "linear":
                    a = torch.einsum("ki,ail->akl", p.params[name][0, :nout], A_nlm)
                elif p.central_rank > 1:
                    w = torch.einsum("er,r...->e...", p.params["central.embedding"], p.params[name])
                    a = torch.einsum("acki,aic->akc", w[:, row][mu_i], A_nlm[:, :, col])
                else:
                    a = torch.einsum("cki,aic->akc", p.params[name][row], A_nlm[:, :, col])
                v = v * a[:, :, index]
            out.append(v.new_zeros((v.shape[0], nout, n)).index_add_(2, invariant, v))
        return out

    return B, layouts


def build_fs_embedding(p):
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

    def embedding(rho):
        m = torch.cat([linear_slot, torch.exp(logm)]) if learned else m0
        m = torch.where((m - 1.0).abs() < 1e-6, 1.0 + 1e-6, m)
        nx = 1.0 / m
        xgain, ygain = nx ** (nx / (1.0 - nx)), nx ** (1.0 / (1.0 - nx))
        r = torch.where(nonlinear, rho, 0.0)
        sign = torch.where(r >= 0, 1.0, -1.0)
        arho = sign * r
        decay = torch.exp(-arho)
        shifted = sign * ((xgain * decay + arho) ** m - ygain * decay)
        return (torch.where(nonlinear, shifted, rho) * a).sum(1, keepdim=True)

    return embedding


def build_mlp_embedding(p):
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

    def embedding(rho):
        x = rho[:, 1:]
        for w in layers[:-1]:
            x = activate(x @ w)
        return x @ layers[-1] + rho[:, :1]

    return embedding


def build_energy(p):
    if p.seed is not None:
        torch.manual_seed(p.seed)

    basis = (build_clamped_beam(p.n_rad_base[0], p.rcut, p.rclamp, p.device)
             if p.basis_type == "ClampedBeam" else
             build_sbessel(p.n_rad_base[0], p.rcut))

    B, layouts = build_B(p, build_A(p, basis))
    # rho[i, d] = sum over ranks of c[mu_i, d, n, invariant] B[i, n, invariant] / sqrt(n_rad_max)
    if p.n_mlp_dens is None:
        ndens = p.fs_parameters if isinstance(p.fs_parameters, int) else len(p.fs_parameters)
    else:
        ndens = p.n_mlp_dens + 1
    densities = [learn(p, f"density.{rank}", len(p.element_map), ndens, p.n_rad_max[rank - 1], len(terms),
                       fill=None if p.init == "grace" else 0.0) for rank, (terms, _) in enumerate(layouts, start=1)]
    embedding = build_fs_embedding(p) if p.n_mlp_dens is None else build_mlp_embedding(p)
    if p.zbl:
        zbl = build_zbl(p)

    def energy(batch):
        mu_i = batch["atomic_mu_i"]
        rho = sum(torch.einsum("aknv,anv->ak", c[mu_i], b) * c.shape[2] ** -0.5 for c, b in zip(densities, B(batch)))
        e = embedding(rho) * p.out_scale + p.out_shift
        return e + zbl(batch) if p.zbl else e

    return energy, layouts


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
        # distance per atom above the lower convex hull of formation energies, pure elements at their lowest energy
        atoms = [a for dataset in datasets for a in dataset["ase_atoms"]]
        e = np.concatenate([dataset["energy_corrected"].to_numpy() / dataset["ase_atoms"].map(len).to_numpy()
                            for dataset in datasets])
        elements = sorted({s for a in atoms for s in a.get_chemical_symbols()})
        c = np.array([[a.get_chemical_symbols().count(el) / len(a) for el in elements] for a in atoms])
        pure = np.zeros(len(elements))
        for k in range(len(elements)):
            if (c[:, k] == 1).any():
                pure[k] = e[c[:, k] == 1].min()
        e = e - c @ pure
        if len(np.unique(c, axis=0)) > 1:
            lower = ConvexHull(np.column_stack([c[:, :-1], e])).equations
            lower = lower[lower[:, -2] < 0]
            e = e - np.max(-(lower[:, -1] + c[:, :-1] @ lower[:, :-2].T) / lower[:, -2], axis=1)
        else:
            e = e - e.min()
        distances = np.split(e, [len(train_df)])[:len(datasets)]
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

def build_batch_loss(energy, device, energy_delta, forces_delta):
    energy_delta = torch.tensor(energy_delta, dtype=DTYPE, device=device)
    forces_delta = torch.tensor(forces_delta, dtype=DTYPE, device=device)

    def batch_loss(batch):
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

        return torch.stack([
            (batch["energy_weight"][:-1] * huber_depa).sum(),
            (batch["force_weight"] * huber_df).sum(),
            depa.abs().sum(),
            (depa * depa).sum(),
            df.abs().sum(),
            (df * df).sum(),
        ])

    return batch_loss


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
    energy, _ = build_energy(model)
    params = model.params

    batch_loss = build_batch_loss(
        energy, device, loss.get("energy", {}).get("delta", 0.01),
        loss.get("forces", {}).get("delta", 0.01))
    # Inductor miscompiles the backward on CPU, so CPU runs eager
    if fit_config.get("jit_compile", False) and device.type == "cuda":
        batch_loss = torch.compile(batch_loss, dynamic=False,
                                   mode="reduce-overhead")
        report("JIT compilation: True (mode=reduce-overhead)")

    energy_weight = loss.get("energy", {}).get("weight", 1.0)
    forces_weight = loss.get("forces", {}).get("weight", 100.0)
    stages = [(energy_weight, forces_weight),
              (switch.get("energy", {}).get("weight", energy_weight), switch.get("forces", {}).get("weight", forces_weight))]
    l2_reg = loss.get("l2_reg", 0.0)
    l2_mask = []
    for name, w in params.items():
        unregularized = name == "fs.logm" or (name.startswith("mlp.") and name != "mlp.0")
        l2_mask.append(torch.full((w.numel(),), 0.0 if unregularized else 1.0, dtype=DTYPE, device=device))
    l2_mask = torch.cat(l2_mask)
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 100)
    switch_after = switch.get("after_iter")
    best_test_loss, switched = math.inf, False

    def run_pass(batches, totals, gradient):
        # loss = ew sum huber(depa) / nstruct + fw sum huber(df) / (3 natoms) + l2_reg (ew + fw) sum w^2
        nstruct, natoms = totals.clamp(min=1.0).tolist()
        ew, fw = stages[switched]
        scale = torch.tensor([ew / nstruct, fw / (3 * natoms)], dtype=DTYPE, device=device)
        sums, grad = torch.zeros(6, dtype=DTYPE, device=device), 0.0
        t_start = time.perf_counter()
        for batch in batches:
            torch.compiler.cudagraph_mark_step_begin()
            metrics = batch_loss(batch)
            if gradient:
                (scale * metrics[:2]).sum().backward()
                # the cat copies each gradient out of the cudagraph pool
                grad = grad + torch.cat([w.grad.reshape(-1) for w in params.values()])
                for w in params.values():
                    w.grad = None
            sums += metrics.detach()
            del metrics
        e, f, abs_depa, sqr_depa, abs_df, sqr_df = all_reduce(sums).tolist()

        l2 = 0.0
        if gradient:
            x = torch.cat([w.detach().reshape(-1) for w in params.values()])
            l2 = l2_reg * (ew + fw) * float((l2_mask * x * x).sum())
            grad = all_reduce(grad) + 2 * l2_reg * (ew + fw) * l2_mask * x
        losses = {"loss/energy": e * ew / nstruct, "loss/forces": f * fw / (3 * natoms), "loss/l2": l2}
        return {
            **losses,
            "total_loss": sum(losses.values()),
            "mae/depa": abs_depa / nstruct,
            "rmse/depa": math.sqrt(sqr_depa / nstruct),
            "mae/f_comp": abs_df / natoms / 3,
            "rmse/f_comp": math.sqrt(sqr_df / natoms / 3),
            "total_time/per_atom": (time.perf_counter() - t_start) / natoms,
            "n_batches": len(batches),
        }, grad if gradient else None

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

    report("Start BFGS optimization")
    opt_params = fit_config.get("opt_params", {})
    gtol = opt_params.get("gtol", 1e-8)
    c1, c2 = opt_params.get("c1", 1e-4), opt_params.get("c2", 0.9)
    on_device = lambda a: torch.as_tensor(a, dtype=DTYPE, device=device)
    last ={"x": None, "neval": 0}

    def eval_at(x):
        if not np.array_equal(x, last["x"]):
            with torch.no_grad():
                for w, v in zip(params.values(), on_device(x).split([w.numel() for w in params.values()])):
                    w.copy_(v.view_as(w))
            last["metrics"], last["grad"] = run_pass(train, train_totals, True)
            last.update(x=x.copy(), neval=last["neval"] + 1)
        return last["metrics"]["total_loss"], last["grad"]

    f_scipy = lambda x: eval_at(x)[0]
    g_scipy = lambda x: eval_at(x)[1].detach().cpu().numpy()

    epoch = 0
    if restart:
        epoch = load_checkpoint(restart)
        report(f"Restarting from epoch {epoch}")
    xk = torch.cat([w.detach().reshape(-1) for w in params.values()]).cpu().numpy()
    row0 = xk.size * rank // world_size
    row1 = xk.size * (rank + 1) // world_size

    def restart_at(x, H=None):
        last["x"] = None
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
            if load_checkpoint("checkpoint.best_test_loss") is not None:
                report("Loading best test loss model")
            save_checkpoint("checkpoint.stage1.best_test_loss", epoch)
            # bests under the old loss are not comparable
            best_test_loss, switched = math.inf, True
            report(f"Switching loss weights to energy={stages[1][0]}, forces={stages[1][1]}")
            xk =torch.cat([w.detach().reshape(-1) for w in params.values()]).cpu().numpy()
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

        eval_at(xk)
        train_metrics = last["metrics"]
        with torch.no_grad():
            test_metrics = None if test is None else run_pass(test, test_totals, False)[0]
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
                with open(os.path.join(output_dir, f"{name}_metrics.yaml"), "at") as f:
                    print("-", json.dumps({**{k: float(v) for k, v in metrics.items()}, "epoch": epoch}), file=f)
        if test_metrics is not None and test_metrics["total_loss"] < best_test_loss:
            best_test_loss = test_metrics["total_loss"]
            report(f"New best test loss found ({best_test_loss:.5e}), checkpointing")
            save_checkpoint("checkpoint.best_test_loss", epoch)
        if epoch % checkpoint_freq == 0:
            report("Regular checkpointing")
            save_checkpoint("checkpoint", epoch)
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
           f"function/gradient evaluations={last['neval']}")
    save_checkpoint("checkpoint", epoch)
    if rank == 0:
        report("Exporting checkpoints/checkpoint to saved_model.yaml")
        export(os.path.join(output_dir, "checkpoints", "checkpoint"),
               os.path.join(output_dir, "saved_model.yaml"))
    return 0


# --- export ---

def export(checkpoint_file, filename):
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    p = Potential(**checkpoint["potential"], device=torch.device("cpu"))
    _, layouts = build_energy(p)
    with torch.no_grad():
        for w, saved in zip(p.params.values(), checkpoint["model"].values()):
            w.copy_(saved)
    # the format has no central element axis, so it would read a wrong model
    if p.central_rank > 1 or p.chemistry == "Zinl":
        raise SystemExit(f"radial={p.radial!r} chemistry={p.chemistry!r} "
                         f"has a central element axis the format cannot store")

    def flat(a):
        return [float(v) for v in np.asarray(a).reshape(-1)]

    # header
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
    mlp = [w.detach().numpy() for name, w in p.params.items() if name.startswith("mlp.")]
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
    coefficients = [w.detach() for name, w in p.params.items() if name.startswith("density.")]
    radial = {
        "radbasename": p.basis_type, "n_rad_base": int(crad.shape[-1]),
        "nradmax": max(int(w.shape[2]) for w in coefficients),
        "cutoff": float(p.rcut),
        "crad": flat(crad), "crad_shape": [int(s) for s in crad.shape],
        "Z": flat(Z), "Z_shape": [int(s) for s in Z.shape],
    }
    if p.zbl:
        radial["subtract_zbl"] = True
    if p.basis_type == "ClampedBeam":
        radial["rclamp"] = float(p.rclamp)
    top = {
        "elements": [e for e, _ in sorted(p.element_map.items(), key=lambda kv: kv[1])],
        "nnorm": float(1.0 / p.avg_n_neigh),
        "emb_spec": emb,
        "radial_basis": radial,
        "shift": float(p.out_shift),
        "scale": float(p.out_scale),
    }

    # basis functions, one per invariant and radial channel n
    functions = []
    for (terms, weights), w in zip(layouts, coefficients):
        for column, expansion in enumerate(terms):
            merged = {}
            for lms, c in expansion.items():
                lms = lms if weights[0] else tuple(sorted(lms))
                merged[lms] = merged.get(lms, 0.0) + c
            merged = [(lms, c) for lms, c in merged.items() if abs(c) > 1e-12]
            ls = [l for l, _ in merged[0][0]]
            ms = [m for lms, _ in merged for _, m in lms]
            functions += [(ls, ms, [c for _, c in merged], w, column, n, weights) for n in range(w.shape[2])]
    functions.sort(key=lambda x: (len(x[0]), x[5], x[0]))

    # radial rows, each factor of each function owns one, numbered from 1 in order of l
    if "linear" in p.radial:
        wrad = {name: w.detach().numpy() for name, w in p.params.items() if name.startswith("wrad.")}
        rows, place = {}, []
        for ls, _, _, _, column, n, weights in functions:
            place.append([])
            for l, name in zip(ls, weights):
                place[-1].append((l, len(rows.setdefault(l, []))))
                if name is None:
                    rows[l].append(np.eye(crad.shape[0])[n])
                else:
                    rows[l].append(wrad[name][column if p.radial == "linear_max" else 0, n])
        before, count = {}, 0
        for l in sorted(rows):
            before[l], count = count, count + len(rows[l])
        ns = [[before[l] + k + 1 for l, k in fn] for fn in place]
        stacked = [v for l in sorted(rows) for v in rows[l]]
        radial["wrad"] = flat(stacked)
        radial["wrad_shape"] = list(np.shape(stacked))
        radial["wrad_lcount"] = [len(rows.get(l, [])) for l in range(max(rows) + 1)]
    else:
        # a model without mixing reads raw radials, ns = n + 1 in every slot
        ns = [[n + 1] * len(ls) for ls, _, _, _, _, n, _ in functions]

    # write, only mu0 and coeff differ by element, so the rest is built once
    polynomials = [({"rank": len(ls), "ls": ls, "ns": fn_ns, "ms_combs": ms, "gen_cgs": flat(cgs), "ndensity": w.shape[1]},
                    w, column, n) for (ls, ms, cgs, w, column, n, _), fn_ns in zip(functions, ns)]
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


if __name__ == "__main__":
    sys.exit(main())
