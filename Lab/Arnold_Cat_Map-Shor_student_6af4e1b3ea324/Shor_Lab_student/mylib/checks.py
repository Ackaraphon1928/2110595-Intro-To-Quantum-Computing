"""Checks for the FILL CODE cells of Part B.  Each check prints what it
tested and returns True (passed) or False.  An error in your code counts as a
failed check (the error is printed)."""
from __future__ import annotations

import math
import random

import numpy as np

from .catmap import M, apply, orbit_length, reference_matpow, reference_period
from .samplers import IdealSampler, control_qubits


def _report(ok, good, bad):
    print(("✅ " + good) if ok else ("❌ " + bad))
    return bool(ok)


def _call(f, *args):
    try:
        return f(*args), None
    except Exception as e:                                    # show the student's error, don't crash
        return None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
def check_cat_period_classical(f):
    """f(N) -> the period of the cat map on an N x N image."""
    cases = [2, 3, 4, 5, 8, 10, 16, 32, 124, 128, 150]
    for N in cases:
        got, err = _call(f, N)
        ref = reference_period(N)
        if err or got != ref:
            return _report(False, "", f"cat_period_classical({N}) gave {err or got}, expected {ref}")
    return _report(True, f"cat_period_classical is right for N = {cases}", "")


def check_matpow(f):
    """f(A, k, N) -> A^k mod N."""
    rng = random.Random(1)
    cases = [(M, 0, 32), (M, 1, 32), (M, 24, 32), (M, 96, 128), (M, 10**15, 1000003)]
    for _ in range(300):
        N = rng.randint(2, 10**6)
        cases.append((tuple(rng.randrange(N) for _ in range(4)), rng.randint(0, 10**9), N))
    for A, k, N in cases:
        got, err = _call(f, A, k, N)
        ref = reference_matpow(A, k, N)
        if err or got is None or tuple(got) != ref:
            return _report(False, "", f"matpow({A}, {k}, {N}) gave {err or got}, expected {ref}")
    return _report(True, f"matpow is right on {len(cases)} cases (exponents up to 10^15)", "")


def check_shear(run):
    """run(n, x0, y0, k) -> (x, y) measured after shear(px, py, k) on |x0>|y0>."""
    rng = random.Random(2)
    cases = [(3, x0, y0, k) for x0, y0, k in [(0, 0, 1), (7, 7, 7), (5, 3, 0), (1, 1, 6)]]
    cases += [(3, rng.randrange(8), rng.randrange(8), rng.randrange(8)) for _ in range(30)]
    cases += [(4, rng.randrange(16), rng.randrange(16), rng.randrange(16)) for _ in range(12)]
    for n, x0, y0, k in cases:
        N = 2 ** n
        got, err = _call(run, n, x0, y0, k)
        ref = ((x0 + k * y0) % N, y0)
        if err or got is None or tuple(got) != ref:
            return _report(False, "", f"shear with n = {n}, x = {x0}, y = {y0}, k = {k} gave {err or got}, "
                                      f"expected {ref}")
    return _report(True, f"shear adds k*y to x (mod 2^n) on {len(cases)} basis states", "")


def check_cat_qpe(counts, N, p):
    """counts: {bits: count} of cat_qpe for N = 2^n, start pixel p."""
    if not counts:
        return _report(False, "", "no measurement results")
    m = len(next(iter(counts)))
    Q, r = 2 ** m, orbit_length(tuple(p), N)
    shots = sum(counts.values())
    near = 0
    for bits, c in counts.items():
        k = int(bits, 2)
        s = round(k * r / Q) % r                               # nearest peak s Q / r
        dist = abs(k - s * Q / r)
        dist = min(dist, Q - dist)
        if dist <= 1.0:
            near += c
    frac = near / shots
    return _report(frac > 0.8, f"{100 * frac:.0f} % of the shots are within 1 of a peak k = s Q / r "
                               f"(r = {r}, Q = 2^{m})",
                   f"only {100 * frac:.0f} % of the shots are near a peak s Q / r with r = {r}, Q = 2^{m}: "
                   f"check the controlled shears and the inverse QFT")


def check_orbit_from_measurement(f):
    """f(k, Q, p, N) -> orbit length of p, or None."""
    N = 32
    m = control_qubits(N)
    Q = 2 ** m
    for p in [(1, 0), (0, 1), (3, 5)]:
        r = orbit_length(p, N)
        for s in range(r):
            k = round(s * Q / r)
            got, err = _call(f, k, Q, p, N)
            want = r if math.gcd(s, r) == 1 else None
            if err or got != want:
                return _report(False, "", f"orbit_from_measurement(k = {k}, Q = 2^{m}, p = {p}, N = {N}) "
                                          f"gave {err or got}, expected {want} (k/Q is close to {s}/{r})")
    return _report(True, "orbit_from_measurement finds r when gcd(s, r) = 1, and returns None otherwise", "")


def check_cat_period_quantum(f):
    """f(N, sampler) -> the period, from phase estimation (checked with IdealSampler)."""
    for N in [16, 32, 64, 128, 256]:
        got, err = _call(f, N, IdealSampler(seed=N))
        ref = reference_period(N)
        if err or got != ref:
            return _report(False, "", f"cat_period_quantum({N}, IdealSampler) gave {err or got}, expected {ref}")
    return _report(True, "cat_period_quantum is right for N = 16, 32, 64, 128, 256 (IdealSampler)", "")
