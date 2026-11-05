"""Exact superposition test: N sparse Boolean flags stored in d < N channels, read back exactly.

Code: Kautz-Singleton superimposed code. A flag is a polynomial p of degree < m over GF(q); its code vector has
one 1 per x in GF(q), at position (x, p(x)), so d = q*q channels and every code has weight w = q. Two different
polynomials agree on at most m-1 points, so any two codes overlap in at most m-1 channels.

Storage: the state is the sum of the codes of the flags that are on (integer channels).
Read flag j: dot = c_j . s is an integer. If j is on, dot >= w. If j is off, dot <= k*(m-1) when at most k flags
are on. With k*(m-1) <= w-1, the compiler's threshold gate relu(dot-(w-1)) - relu(dot-w) is exactly 1 or 0.
Overflow: the total of s is exactly w*|on|, so one threshold on the total detects more than k flags on.

OR storage: keep each channel 0/1 by storing the OR of the codes instead of the sum. Flag j is on exactly when all
w channels of its code are on, a plain AND of channel literals, because a flag that is off has at most k*(m-1) < w
of its channels covered by the k flags that are on (the code is "k-disjunct"). This version fits a compiler whose
channels are all 0/1; it needs a separate counter to detect more than k flags on.

pipec implements this as the set type Sparse[N, K, C], with OR storage (see pipec/README.md).

    python3 exact_superposition.py
"""
import itertools
import math
import sys

import numpy as np


def code_matrix(q, m, n):
    """The first n codes (columns) of the Kautz-Singleton code over GF(q), polynomials of degree < m."""
    cols = []
    for coeffs in itertools.product(range(q), repeat=m):
        v = np.zeros(q * q, dtype=np.int64)
        for x in range(q):
            px = sum(c * x ** i for i, c in enumerate(coeffs)) % q
            v[x * q + px] = 1
        cols.append(v)
        if len(cols) == n:
            break
    return np.stack(cols, axis=1)                      # d x n


def relu(z):
    return np.maximum(z, 0.0)


def gelu(z):
    from math import erf, sqrt
    return z * 0.5 * (1.0 + np.vectorize(lambda t: erf(t / sqrt(2.0)))(z))


def read(C, S, w, act=relu, K=1.0):
    """Threshold gates on the superposed state S (rows are states). act(K z)/K is the GELU back end's scaling."""
    dots = S @ C.astype(np.float64)
    return act(K * (dots - (w - 1))) / K - act(K * (dots - w)) / K


def overflow(S, w, k, act=relu, K=1.0):
    tot = S.sum(axis=1)
    t = (k + 1) * w                                    # fires when more than k flags are on
    return act(K * (tot - (t - 1))) / K - act(K * (tot - t)) / K


def subsets(n, size):
    for combo in itertools.combinations(range(n), size):
        yield combo


def check(q, m, n, k, exhaustive_up_to):
    C = code_matrix(q, m, n)
    d, w, lam = C.shape[0], q, m - 1
    O = C.T @ C
    assert (np.diag(O) == w).all() and O[~np.eye(n, dtype=bool)].max() <= lam
    assert k * lam <= w - 1, "this code does not support k flags on"
    print(f"\nq={q}, m={m}: {n} flags in {d} channels ({n / d:.1f}x fewer), code weight {w}, overlap <= {lam}, "
          f"exact for up to k={k} flags on")
    errors = checked = 0
    for size in range(0, min(k, exhaustive_up_to) + 1):
        combos = list(subsets(n, size)) if size else [()]
        for start in range(0, len(combos), 200_000):
            chunk = combos[start:start + 200_000]
            X = np.zeros((len(chunk), n))
            for r, c in enumerate(chunk):
                X[r, list(c)] = 1
            S = X @ C.T.astype(np.float64)
            out = read(C, S, w)
            errors += int((out != X).sum())                     # exact equality: 0 or 1, nothing in between
            errors += int((overflow(S, w, k) != 0).sum())       # never fires within the limit
            checked += len(chunk)
        print(f"  every set of {size} flags on: {math.comb(n, size):,} states, all {n} flags read exactly"
              if errors == 0 else f"  ERRORS at size {size}: {errors}")
    rng = np.random.default_rng(0)
    if k > exhaustive_up_to:
        X = np.zeros((100_000, n))
        for r in range(len(X)):
            X[r, rng.choice(n, size=int(rng.integers(exhaustive_up_to + 1, k + 1)), replace=False)] = 1
        S = X @ C.T.astype(np.float64)
        e = int((read(C, S, w) != X).sum()) + int((overflow(S, w, k) != 0).sum())
        errors += e
        print(f"  100,000 random states with {exhaustive_up_to + 1}..{k} flags on: {e} read errors")
    # OR storage: channels stay 0/1, and a flag is on when all w channels of its code are on
    for size in range(0, min(k, exhaustive_up_to) + 1):
        combos = list(subsets(n, size)) if size else [()]
        for start in range(0, len(combos), 200_000):
            chunk = combos[start:start + 200_000]
            X = np.zeros((len(chunk), n))
            for r, c in enumerate(chunk):
                X[r, list(c)] = 1
            S_or = (X @ C.T.astype(np.float64) > 0).astype(np.float64)
            e = int((((S_or @ C.astype(np.float64)) == w).astype(np.float64) != X).sum())
            errors += e
    print(f"  OR storage (0/1 channels, AND reads), every state up to {min(k, exhaustive_up_to)} flags on: "
          f"{'all reads exact' if errors == 0 else f'{errors} errors'}")
    # more than k flags on: reads may fail, but the overflow gate must fire every time
    X = np.zeros((100_000, n))
    for r in range(len(X)):
        X[r, rng.choice(n, size=int(rng.integers(k + 1, k + 6)), replace=False)] = 1
    S = X @ C.T.astype(np.float64)
    bad_reads = int((read(C, S, w) != X).any(axis=1).sum())
    missed = int((overflow(S, w, k) != 1).sum())
    print(f"  100,000 states with {k + 1}..{k + 5} flags on: {bad_reads:,} have a wrong read, overflow gate missed "
          f"{missed}")
    # GELU back end: the same gates scaled by K = 256, on a sample within the limit
    X = np.zeros((2_000, n))
    for r in range(len(X)):
        X[r, rng.choice(n, size=int(rng.integers(0, k + 1)), replace=False)] = 1
    S = X @ C.T.astype(np.float64)
    dev = float(np.abs(read(C, S, w, gelu, 256.0) - X).max())
    print(f"  GELU back end (K=256), 2,000 states: largest deviation from exact 0/1 = {dev:.1e}")
    return errors + missed


if __name__ == "__main__":
    total = 0
    total += check(q=7, m=3, n=256, k=3, exhaustive_up_to=3)
    total += check(q=11, m=3, n=1024, k=5, exhaustive_up_to=2)
    total += check(q=17, m=3, n=4096, k=8, exhaustive_up_to=1)
    print("\nALL EXACT" if total == 0 else f"\nFAILURES: {total}")
    sys.exit(1 if total else 0)
