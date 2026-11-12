"""Two ways to run phase estimation of the cat map for one start pixel p.

CircuitSampler -- runs your cat_qpe kernel gate by gate with CUDA-Q
                  (on the GPU with the target 'nvidia').
IdealSampler   -- samples the exact output distribution of that same
                  noiseless circuit in closed form.  If the pixel's orbit has
                  length r, the eigenphases are s/r (s = 0 .. r-1, all equally
                  likely), and with Q = 2^m
                      P(k) = (1/r) * sum_s F(k/Q - s/r),
                      F(d) = sin^2(pi Q d) / (Q^2 sin^2(pi d))   (F = 1 if d is an integer).
                  It gives the same statistics, much faster, so big pictures
                  take seconds.  It needs the classical orbit length r to
                  imitate the circuit; that work is never counted.

Both have counts(N, p, tg, ks, off, m, shots) and return {bit string: count},
read as in Part A:  k = int(bits, 2).
"""
from __future__ import annotations

import functools
import subprocess
import time

import cudaq
import numpy as np

from .catmap import orbit_length


def control_qubits(N):
    """m with Q = 2^m > (3N)^2.  Dyson and Falk (1992): every orbit length is
    at most 3N, so continued fractions always recover s/r from the closest k."""
    return (9 * N * N).bit_length()


class CircuitSampler:
    """Runs your kernel cat_qpe(n, m, x0, y0, tg, ks, off) with CUDA-Q."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0
        self.seconds = 0.0

    def counts(self, N, p, tg, ks, off, m, shots=2000):
        n = N.bit_length() - 1
        if 2 ** n != N:
            raise ValueError("the circuit works for N = 2^n only (see challenge 2)")
        t0 = time.time()
        res = cudaq.sample(self.kernel, n, m, int(p[0]), int(p[1]), tg, ks, off, shots_count=shots)
        self.seconds += time.time() - t0
        self.calls += 1
        return {bits: count for bits, count in res.items()}


@functools.lru_cache(maxsize=16)
def qpe_distribution(r, m):
    """P(k), k = 0 .. 2^m - 1, for an orbit of length r (see the module text)."""
    Q = 2 ** m
    k = np.arange(Q, dtype=np.float64)
    prob = np.zeros(Q)
    for s in range(r):
        d = k / Q - s / r
        num = np.sin(np.pi * Q * d) ** 2
        den = (Q * np.sin(np.pi * d)) ** 2
        exact = np.abs(den) < 1e-18                # d is an integer: F = 1
        prob += np.where(exact, 1.0, num / np.where(exact, 1.0, den))
    prob /= prob.sum()
    return prob


class IdealSampler:
    """Samples the exact output distribution of the noiseless cat_qpe circuit."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def counts(self, N, p, tg=None, ks=None, off=None, m=None, shots=2000):
        m = m or control_qubits(N)
        r = orbit_length(tuple(p), N)              # the classical answer, only to imitate the circuit
        k = self.rng.choice(2 ** m, size=shots, p=qpe_distribution(r, m))
        values, numbers = np.unique(k, return_counts=True)
        self.calls += 1
        return {format(int(v), f"0{m}b"): int(c) for v, c in zip(values, numbers)}


def info():
    """Print the CUDA-Q version, target and GPU."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu = ""
    print(f"CUDA-Q {cudaq.__version__.split()[2] if len(cudaq.__version__.split()) > 2 else cudaq.__version__}, "
          f"target '{cudaq.get_target().name}', GPU: {gpu or 'none'}")
