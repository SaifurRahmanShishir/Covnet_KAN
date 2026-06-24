#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from scipy import special
import itertools


# --------------------------------------------------
# Location/grid functions
# --------------------------------------------------

def locations(K, d):
    gr = np.arange(1, K + 1) / (K + 1)
    grid = list(itertools.product(gr, repeat=d))
    grid = np.asarray(grid)
    return grid


def locations_unif(M, d):
    loc = np.random.uniform(low=0., high=1., size=[M, d])
    return loc


def pixelate_single(s, K):
    for j in np.arange(1, K):
        if s < (j + 0.5) / (K + 1):
            return j - 1
    return K - 1


def pixelate(u, K):
    idx = []
    for s in u:
        idx.append(pixelate_single(s, K))
    return idx


def indices(u, K, d):
    idx = np.apply_along_axis(pixelate, 1, u, K)
    c = np.zeros(idx.shape[0], dtype=int)

    for i in range(d):
        c += idx[:, i] * K ** (d - 1 - i)

    return c


def print_indices(u, v, K, d, file="indices.dat"):
    idx1 = indices(u, K, d)
    idx2 = indices(v, K, d)
    idx = np.vstack((idx1, idx2))
    np.savetxt(file, idx, fmt="%d")


def print_locations(K, d, file="locations.dat"):
    u = locations(K, d)
    np.savetxt(file, u, fmt="%.10f")


# --------------------------------------------------
# Covariance functions
# --------------------------------------------------

def bessel(s, t, nu):
    r = abs(s - t)

    if r == 0:
        return 1.0

    c = 2 ** nu * special.gamma(nu + 1) * r ** (-nu) * special.jv(nu, r)
    return c


def BM(s, t):
    if s <= t and s > 0.:
        return s
    elif s > t and t > 0.:
        return t
    elif s < 0. and t < 0.:
        return BM(-s, -t)
    else:
        return 0.


def fBM(s, t, alpha):
    c = 0.5 * (
        abs(t) ** (2 * alpha)
        + abs(s) ** (2 * alpha)
        - abs(t - s) ** (2 * alpha)
    )
    return c


def fBM2(s, t, H):
    alpha = 2 * H

    if s <= t and s > 0.:
        return 0.5 * (t ** alpha + s ** alpha - (t - s) ** alpha)
    elif s > t and t > 0.:
        return 0.5 * (t ** alpha + s ** alpha - (s - t) ** alpha)
    elif s < 0. and t < 0.:
        return fBM2(-s, -t, H)
    else:
        return 0.


def iBM(s, t):
    if s <= t and s > 0.:
        return s ** 2 * t / 2 - s ** 3 / 6
    elif s > t and t > 0.:
        return t ** 2 * s / 2 - t ** 3 / 6
    elif s < 0. and t < 0.:
        return iBM(-s, -t)
    else:
        return 0.


def Bbridge(s, t, T=1.):
    if s <= t and s > 0.:
        return s * (T - t) / T
    elif s > t and t > 0.:
        return t * (T - s) / T
    elif s < 0. and t < 0.:
        return Bbridge(-s, -t, T)
    else:
        return 0.


def cauchy(s, t, gamma):
    r = abs(s - t)
    return 1. / (1 + r ** 2) ** gamma


def dampedcos(s, t, lam):
    r = abs(s - t)
    c = np.exp(-lam * r) * np.cos(r)
    return c


def fractgauss(s, t, alpha):
    r = abs(s - t)
    c = 0.5 * ((r + 1) ** alpha + abs(r - 1) ** alpha - 2 * r ** alpha)
    return c


def matern(s, t, nu, rho=1.):
    r = abs(s - t) / rho

    if r == 0:
        r = np.finfo(float).eps

    if nu == 0.5:
        c = np.exp(-r)
    elif nu == 1.5:
        c = r * np.sqrt(3)
        c = (1. + c) * np.exp(-c)
    elif nu == 2.5:
        c = r * np.sqrt(5)
        c = (1. + c + c ** 2 / 3.) * np.exp(-c)
    elif nu == np.inf:
        c = np.exp(-r ** 2 / 2.)
    else:
        tmp = np.sqrt(2 * nu) * r
        c = 2 ** (1. - nu) / special.gamma(nu)
        c *= tmp ** nu
        c *= special.kv(nu, tmp)

    return c


# --------------------------------------------------
# Rotation helper
# --------------------------------------------------

def rotation_matrix_2d(theta = np.pi / 4):
    """
    2D rotation matrix.

    theta = pi/4 means 45-degree rotation.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def resolve_rotation(rotation = False, theta = np.pi / 4, d=2):
    """
    Handles rotation argument.

    rotation=False or None -> identity matrix
    rotation=True          -> 2D rotation matrix
    rotation=np.ndarray    -> use given matrix
    """

    if rotation is False or rotation is None:
        return None

    if isinstance(rotation, np.ndarray):
        return rotation

    if rotation is True:
        if d != 2:
            raise ValueError("rotation=True is currently implemented only for d=2.")
        return rotation_matrix_2d(theta)

    raise ValueError("rotation must be False, True, None, or a numpy rotation matrix.")


# --------------------------------------------------
# Covariance matrix construction
# --------------------------------------------------

def cov_mat(K, d, method, O=None):
    """
    Builds full covariance matrix on K^d grid points.

    This version is specifically written for d=2.
    """

    if d != 2:
        raise ValueError("This datagen_nonseparable2.py version expects d=2.")

    if O is None:
        O = np.eye(d)

    u = locations(K, d)
    u_ = np.matmul(u, O.T)

    D = int(K ** d)
    C = np.zeros([D, D], dtype=float)

    for i in range(D):
        s1, t1 = u_[i, :]

        for j in range(i, D):
            s2, t2 = u_[j, :]

            C[i, j] = method(s1, s2) * method(t1, t2)
            C[j, i] = C[i, j]

    return C


def cross_cov(u, v, method, O=None):
    """
    Computes true covariance values for random test pairs.

    u shape: (M,2)
    v shape: (M,2)
    output shape: (M,)
    """

    if O is None:
        d = u.shape[1]
        O = np.eye(d)

    if u.shape != v.shape:
        raise ValueError("Mismatch in shapes of u and v.")

    if u.shape[1] != 2:
        raise ValueError("This cross_cov function expects d=2.")

    M = u.shape[0]

    u_ = np.matmul(u, O.T)
    v_ = np.matmul(v, O.T)

    C = np.zeros(M, dtype=float)

    for i in range(M):
        s1, t1 = u_[i, :]
        s2, t2 = v_[i, :]

        C[i] = method(s1, s2) * method(t1, t2)

    return C


def sqrt_mat(C):
    """
    Eigen-decomposition square-root factorization.

    If C = E diag(lambda) E^T,
    then samples are generated as E sqrt(lambda) z.
    """

    lam, E = np.linalg.eigh(C)

    idx = lam > 0.
    lam = np.sqrt(lam[idx])
    E = E[:, idx]

    return lam, E


# --------------------------------------------------
# Main data-generation function
# --------------------------------------------------

def datagen_and_print(
    N,
    K,
    d,
    replicates=1,
    method=None,
    cov_name="covariance",
    rotation=False,
    theta=np.pi / 4,
    M=10000,
    base_dir="Simulation",
    save_full_cov=False,
    seed=None
):
    """
    Generate CovNet-style data and save into:

        Simulation/cov_name/

    Files saved:
        locations1.dat
        Example1.dat
        True_locations1.dat
        indices1.dat
        True_cov1.dat

    For each replicate.
    """

    if seed is not None:
        np.random.seed(seed)

    if method is None:
        raise ValueError("You must provide a covariance method, e.g. method=iBM.")

    if d != 2:
        raise ValueError("This script currently supports d=2 only.")

    O = resolve_rotation(rotation=rotation, theta=theta, d=d)

    if rotation is True:
        folder_name = cov_name + "_rotated"
    else:
        folder_name = cov_name

    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Saving files to:", output_dir)
    print("Basis generation started")

    C_full = cov_mat(K, d, method, O)
    lam, E = sqrt_mat(C_full)

    print("Basis generated")
    print("Full covariance shape:", C_full.shape)

    if save_full_cov:
        filename = os.path.join(output_dir, "Full_cov_matrix.dat")
        np.savetxt(filename, C_full, fmt="%.10f")

    for repl in range(replicates):
        print("Replicate " + str(repl + 1))

        # Save regular grid locations
        filename = os.path.join(output_dir, "locations" + str(repl + 1) + ".dat")
        print_locations(K, d, filename)

        # Generate and save N sample surfaces
        filename = os.path.join(output_dir, "Example" + str(repl + 1) + ".dat")

        with open(filename, "w") as f:
            for n in range(N):
                z = np.random.normal(loc=0., scale=1., size=len(lam))
                x = np.sum(E * lam * z, axis=1)
                x = x.reshape(1, -1)

                np.savetxt(f, x, fmt="%.10f")

        # Generate random test pairs
        u = locations_unif(M, d)
        v = locations_unif(M, d)

        # Save test locations
        filename = os.path.join(output_dir, "True_locations" + str(repl + 1) + ".dat")
        np.savetxt(filename, np.hstack((u, v)), fmt="%.10f")

        # Save grid indices for test locations
        filename = os.path.join(output_dir, "indices" + str(repl + 1) + ".dat")
        print_indices(u, v, K, d, filename)

        # Save true covariance values for test pairs
        C = cross_cov(u, v, method, O)

        filename = os.path.join(output_dir, "True_cov" + str(repl + 1) + ".dat")
        np.savetxt(filename, C, fmt="%.10f")

        del C, u, v

    print("Data generation completed.")


# --------------------------------------------------
# Optional in-memory generator
# --------------------------------------------------

def cov_data_gen(N, K, d, method=None, rotation=False, theta=np.pi / 4):
    """
    Generate covariance matrix C and data matrix X in memory.

    Returns:
        C shape: (K^d, K^d)
        X shape: (N, K^d)
    """

    if method is None:
        raise ValueError("You must provide a covariance method.")

    O = resolve_rotation(rotation=rotation, theta=theta, d=d)

    C = cov_mat(K, d, method, O)
    lam, E = sqrt_mat(C)

    D = int(K ** d)
    X = np.zeros([N, D], dtype=float)

    for n in range(N):
        z = np.random.normal(loc=0., scale=1., size=len(lam))
        x = np.sum(E * lam * z, axis=1)
        X[n, :] = x

    return C, X




