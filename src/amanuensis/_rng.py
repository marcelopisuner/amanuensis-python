"""R's random draw, reproduced: ``set.seed(seed, kind = "Mersenne-Twister",
sample.kind = "Rejection")`` followed by ``sample.int(n)``.

benchmark() draws its pages as a seeded permutation of the file names. The
seed should come from the source itself (its year, a shelfmark number),
which a reader can check; reproducing R's generator means the same seed
draws the same pages in the R package and here, and the draw can be
re-checked with base R alone. Python's own ``random`` module is a Mersenne
Twister too, but it is seeded and sampled differently.

The C it follows is R's src/main/RNG.c (RNG_Init, MT_genrand, fixup) and
src/main/random.c (R_unif_index with rejection sampling, do_sample).
"""

from __future__ import annotations

import math

_N, _M = 624, 397
_MATRIX_A = 0x9908B0DF
_UPPER, _LOWER = 0x80000000, 0x7FFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)


class RMersenneTwister:
    """R's Mersenne-Twister, seeded as ``set.seed()`` seeds it."""

    def __init__(self, seed: int):
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):  # initial scrambling
            s = (69069 * s + 1) & 0xFFFFFFFF
        dummy = []
        for _ in range(_N + 1):
            s = (69069 * s + 1) & 0xFFFFFFFF
            dummy.append(s)
        self.mt = dummy[1:]
        self.mti = _N  # FixupSeeds(initial = TRUE): dummy[0] = 624

    def _genrand(self) -> float:
        mt = self.mt
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return (y & 0xFFFFFFFF) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        x = self._genrand()
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int) -> list[int]:
        """``sample.int(n)``: a permutation of 1..n."""
        if n < 2:
            return [self.unif_index(n) + 1 for _ in range(n)]
        x = list(range(n))
        out = []
        m = n
        for _ in range(n):
            j = self.unif_index(m)
            out.append(x[j] + 1)
            m -= 1
            x[j] = x[m]
        return out


def r_sample_int(n: int, seed: int) -> list[int]:
    """``set.seed(seed); sample.int(n)`` in R (Mersenne-Twister, Rejection)."""
    return RMersenneTwister(seed).sample_int(n)
