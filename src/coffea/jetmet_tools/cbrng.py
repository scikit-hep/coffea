"""Counter-based random number generator (CBRNG) using the Squares algorithm.

Provides partition-independent, reproducible random numbers for JER smearing
by constructing per-jet counters from physics coordinates (event number, phi, eta).

Based on the Squares algorithm: https://arxiv.org/abs/2004.06278
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import numpy.typing as npt

_2_PI = 2 * np.pi
_UINT64_11 = np.uint64(11)
_UINT64_32 = np.uint64(32)
_BIT52_COUNT = np.float64(1 << 53)

SeedLike = int | str | Iterable[int | str]


def _str_to_entropy(__str: str) -> list[np.uint64]:
    return np.frombuffer(hashlib.md5(__str.encode()).digest(), dtype=np.uint64).tolist()


def _seed(*entropy: SeedLike) -> tuple[int, ...]:
    seeds = []
    for e in entropy:
        if isinstance(e, str):
            seeds.extend(_str_to_entropy(e))
        elif isinstance(e, Iterable):
            seeds.extend(_seed(*e))
        else:
            seeds.append(e)
    return (*seeds,)


class Squares:
    """Counter-based RNG using the Squares algorithm.

    Unlike standard PRNGs, Squares maps a counter (derived from per-jet physics
    quantities) to a random number.  This makes outputs deterministic regardless
    of how the data is partitioned.

    Parameters
    ----------
    seed : SeedLike
        One or more ints/strings used to derive the key.

    Examples
    --------
    >>> rng = Squares("JER")
    >>> counters = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    >>> rng.normal(counters)  # returns one float per row
    """

    def __init__(self, *seed: SeedLike):
        self._seed = _seed(seed)
        self._keys: dict[int | None, np.uint64] = {}
        self._offset: int | None = None

    def _make_key(self, gen: np.random.Generator) -> np.uint64:
        bits = np.arange(1, 16, dtype=np.uint64)
        offsets = np.arange(0, 29, 4, dtype=np.uint64)
        lower8 = gen.choice(bits, 8, replace=False)
        for i in range(16):
            if lower8[i] % 2 == 1:
                lower8 = np.roll(lower8, -i)
                break
        higher8 = np.zeros(8, dtype=np.uint64)
        higher8[0:1] = gen.choice(np.delete(bits, int(lower8[-1]) - 1), 1)
        higher8[1:] = gen.choice(
            np.delete(bits, int(higher8[0]) - 1), 7, replace=False
        )
        return np.sum(lower8 << offsets) + (np.sum(higher8 << offsets) << _UINT64_32)

    @property
    def _key(self) -> np.uint64:
        k, o = self._keys, self._offset
        if o not in k:
            s = self._seed
            if o is not None:
                s += (o,)
            k[o] = self._make_key(np.random.Generator(np.random.PCG64(s)))
        return k[o]

    def _shift(self, offset: int | None = None) -> Squares:
        new = Squares.__new__(Squares)
        new._seed = self._seed
        new._keys = self._keys
        new._offset = offset
        return new

    # -- Core bit generators --------------------------------------------------

    def _round(self, lr: npt.NDArray, shift: npt.NDArray, last: bool = False):
        lr *= lr
        lr += shift
        if last:
            yield lr.copy()
        l = lr >> _UINT64_32
        lr <<= _UINT64_32
        lr |= l
        yield lr

    def bit32(self, ctrs: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint32]:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        x *= x
        x += z
        x >>= _UINT64_32
        return x.astype(np.uint32)

    def bit64(self, ctrs: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        (t, _) = self._round(x, z, last=True)
        x *= x
        x += y
        x >>= _UINT64_32
        x ^= t
        return x

    # -- Reduction & distributions --------------------------------------------

    def uint64(self, counters: npt.ArrayLike) -> npt.NDArray[np.uint64]:
        """Reduce the last dimension of counters to a single uint64 per row."""
        counters = np.asarray(counters, dtype=np.uint64)
        if counters.ndim == 1:
            return self.bit64(counters)
        while True:
            shape = counters.shape[-1]
            if shape == 1:
                return self.bit64(counters).reshape(counters.shape[:-1])
            elif shape % 2 == 0:
                counters = self.bit32(counters).view(np.uint64)
            else:
                counters = np.concatenate(
                    [
                        counters[..., -1:],
                        self.bit32(counters[..., :-1]).view(np.uint64),
                    ],
                    axis=-1,
                )

    def float64(self, counters: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Uniform [0, 1) from counters."""
        x = self.uint64(counters)
        x >>= _UINT64_11
        return x / _BIT52_COUNT

    def normal(
        self,
        counters: npt.ArrayLike,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Normal distribution via Box-Muller transform."""
        o = self._offset
        o = 0 if o is None else o + 1
        x = self.float64(counters)
        y = self._shift(o).float64(counters)
        x = np.sqrt(-2.0 * np.log(x)) * np.cos(_2_PI * y)
        x *= scale
        x += loc
        return x
