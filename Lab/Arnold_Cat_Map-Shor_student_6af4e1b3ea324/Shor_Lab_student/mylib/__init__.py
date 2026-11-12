"""mylib -- helper code for Part B of the lab: order finding for Arnold's
cat map with CUDA-Q (>= 0.14).

  catmap    the cat map on a picture, 2 x 2 matrices mod N, orbit lengths,
            shears (x <- x + k*y), the picture of Chelsea the cat (CC0)
  samplers  run your phase-estimation kernel (CircuitSampler) or sample its
            exact output distribution (IdealSampler)
  checks    tests for the FILL CODE cells
"""
from . import catmap, checks, samplers
from .samplers import info

__all__ = ["catmap", "checks", "samplers", "info"]
