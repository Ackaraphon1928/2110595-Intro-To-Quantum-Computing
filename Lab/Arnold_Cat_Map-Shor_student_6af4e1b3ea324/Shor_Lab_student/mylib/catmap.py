"""Arnold's cat map on an N x N image, and the 2 x 2 matrix tools of Part B.

Pixels: x is the column (0 .. N-1, left to right), y is the row (0 .. N-1, top
to bottom).  One step of the cat map moves the pixel at (x, y) to

    (x', y') = (2x + y, x + y) mod N,    i.e.  p' = M p  with  M = [[2, 1], [1, 1]].

A 2 x 2 matrix [[a, b], [c, d]] is stored as the tuple (a, b, c, d).
"""
from __future__ import annotations

import math
import os

import numpy as np

M = (2, 1, 1, 1)                                  # the cat map
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ----------------------------------------------------------------------------
# 2 x 2 matrices mod N
# ----------------------------------------------------------------------------
def identity(N):
    """The identity matrix mod N."""
    return (1 % N, 0, 0, 1 % N)


def matmul(A, B, N):
    """A * B mod N."""
    a, b, c, d = A
    e, f, g, h = B
    return ((a * e + b * g) % N, (a * f + b * h) % N, (c * e + d * g) % N, (c * f + d * h) % N)


def apply(A, p, N):
    """A * p mod N for a pixel p = (x, y)."""
    return ((A[0] * p[0] + A[1] * p[1]) % N, (A[2] * p[0] + A[3] * p[1]) % N)


def reference_matpow(A, k, N):
    """A^k mod N by repeated squaring (the reference for your matpow)."""
    R, B = identity(N), tuple(v % N for v in A)
    while k > 0:
        if k % 2 == 1:
            R = matmul(R, B, N)
        B = matmul(B, B, N)
        k //= 2
    return R


def reference_period(N, A=M):
    """Smallest k >= 1 with A^k = I (mod N), by brute force."""
    P, k = tuple(v % N for v in A), 1
    while P != identity(N):
        P, k = matmul(P, A, N), k + 1
    return k


def orbit_length(p, N, A=M):
    """Number of steps until the pixel p comes back to its place."""
    q, r = apply(A, p, N), 1
    while q != tuple(v % N for v in p):
        q, r = apply(A, q, N), r + 1
    return r


def orbit_lengths(N, A=M):
    """Orbit length of every pixel, as an N x N array indexed [y, x]."""
    out = np.zeros((N, N), dtype=np.int64)
    for y0 in range(N):
        for x0 in range(N):
            if out[y0, x0]:
                continue
            cycle, p = [], (x0, y0)
            while True:
                cycle.append(p)
                p = apply(A, p, N)
                if p == (x0, y0):
                    break
            for (x, y) in cycle:
                out[y, x] = len(cycle)
    return out


def pisano(N):
    """Period of the Fibonacci numbers mod N."""
    a, b, k = 0, 1, 0
    while True:
        a, b, k = b, (a + b) % N, k + 1
        if (a, b) == (0, 1 % N):
            return k


# ----------------------------------------------------------------------------
# shears: x <- x + k*y  (target 0)  and  y <- y + k*x  (target 1)
# ----------------------------------------------------------------------------
def shears(A, N):
    """Write A (determinant 1 mod N) as a list of shears [(target, k), ...],
    applied from left to right.  target 0: x <- x + k*y, target 1: y <- y + k*x.

    If c is invertible mod N:
        A = U(t) L(c) U(s),  s = (d - 1) / c,  t = (a - 1) / c   (mod N)
    with U(k) = [[1, k], [0, 1]] and L(k) = [[1, 0], [k, 1]]: first U(s), then L(c), then U(t).
    If not, the same with the roles of x and y swapped (b invertible), or one
    extra shear first makes c invertible."""
    a, b, c, d = (v % N for v in A)
    if (a * d - b * c) % N != 1 % N:
        raise ValueError(f"determinant of {A} is not 1 mod {N}")
    if math.gcd(c, N) == 1:
        ci = pow(c, -1, N)
        return [(0, (d - 1) * ci % N), (1, c), (0, (a - 1) * ci % N)]
    if math.gcd(b, N) == 1:
        bi = pow(b, -1, N)
        return [(1, (a - 1) * bi % N), (0, b), (1, (d - 1) * bi % N)]
    if math.gcd(a + c, N) == 1:                   # A = L(-1) (L(1) A)
        return shears((a, b, (a + c) % N, (b + d) % N), N) + [(1, N - 1)]
    if math.gcd(b + d, N) == 1:                   # A = U(-1) (U(1) A)
        return shears(((a + c) % N, (b + d) % N, c, d), N) + [(0, N - 1)]
    raise ValueError(f"no shear decomposition found for {A} mod {N}")


def apply_shears(S, p, N):
    """Apply the shears S to the pixel p (classically)."""
    x, y = p
    for target, k in S:
        if target == 0:
            x = (x + k * y) % N
        else:
            y = (y + k * x) % N
    return (x, y)


# ----------------------------------------------------------------------------
# the picture
# ----------------------------------------------------------------------------
def load_cat(N=256):
    """Chelsea the cat (a scikit-image test photo, CC0), as an N x N x 3 array
    with values in [0, 1].  N must divide 256, or any N for a nearest-pixel copy."""
    import matplotlib.image as mpimg
    img = mpimg.imread(os.path.join(DATA, "chelsea_256.png"))[..., :3].astype(np.float32)
    if N == 256:
        return img
    if 256 % N == 0:
        f = 256 // N
        return img.reshape(N, f, N, f, 3).mean(axis=(1, 3))
    idx = (np.arange(N) * 256) // N
    return img[idx][:, idx]


def scramble(img, k, A=M):
    """The image after k steps of the map A (default: the cat map)."""
    N = img.shape[0]
    a, b, c, d = reference_matpow(A, k, N)
    y, x = np.mgrid[0:N, 0:N]
    xn, yn = (a * x + b * y) % N, (c * x + d * y) % N
    out = np.empty_like(img)
    out[yn, xn] = img[y, x]
    return out


def show_frames(img, steps, A=M, size=2.0, title=None):
    """Show the image after each number of steps in `steps`."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(steps), figsize=(size * len(steps), size + 0.4))
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, steps):
        ax.imshow(scramble(img, s, A), interpolation="nearest")
        ax.set_title(f"step {s}", fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# cost of the phase-estimation circuit of Part B
# ----------------------------------------------------------------------------
def qpe_gate_count(n, m, shears_per_power=3):
    """Gates of cat_qpe for N = 2^n and m control qubits (start-pixel X gates
    and measurements not counted).  One controlled shear = QFT + n(n+1)/2
    doubly-controlled rotations + inverse QFT; a QFT on n qubits has n H and
    n(n-1)/2 controlled rotations."""
    qft = n + n * (n - 1) // 2
    shear = 2 * qft + n * (n + 1) // 2
    return m + m * shears_per_power * shear + (m + m * (m - 1) // 2)
