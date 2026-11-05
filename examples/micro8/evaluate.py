"""Test battery for any MICRO-8 weights file. It treats the network as a black box: it only runs it and compares
its answers with the exact oracle in task.py.

    python3 evaluate.py WEIGHTS.npz          shape check, then the full battery (about 618k programs, a few minutes)

Exact success S = 1 means 0 errors on every in-spec test. The last test (float32) is reported but not counted.
This is the battery that scored every network in the experiment, unchanged.
"""
import argparse
import time

import numpy as np

from network import SHAPES, forward
from task import ADD, CALL, DJNZ, HALT, JC, JZ, LD, N_ADDR, N_DATA, RET, TIMEOUT, encode, oracle, sample

HALTY = np.r_[0.12, np.full(15, 0.88 / 15)]                      # more programs that stop
FLOWY = np.ones(16)
FLOWY[[JZ, JC, DJNZ, CALL, RET]] = 3.0
FLOWY[HALT] = 1.5
FLOWY /= FLOWY.sum()                                            # control-flow heavy


def load(path):
    th = dict(np.load(path))
    problems = [f"missing tensor {k}" for k in SHAPES if k not in th]
    problems += [f"unexpected tensor {k}" for k in th if k not in SHAPES]
    problems += [f"{k} has shape {th[k].shape}, expected {s}"
                 for k, s in SHAPES.items() if k in th and th[k].shape != s]
    problems += [f"{k} has non-finite values" for k in th if not np.all(np.isfinite(th[k]))]
    return th, problems


def predict(theta, batch, dtype=np.float64, chunk=10_000):
    op, arg, data, order, A0, B0 = batch
    th = {k: v.astype(dtype) for k, v in theta.items()}
    out = []
    for s in range(0, len(A0), chunk):
        sl = slice(s, s + chunk)
        Xq, Xr = encode(op[sl], arg[sl], data[sl], order[sl], A0[sl], B0[sl])
        with np.errstate(all="ignore"):
            out.append(forward(th, Xq.astype(dtype), Xr.astype(dtype)))
    return np.concatenate(out)


def worked_example():
    op = np.full((1, N_ADDR), HALT)
    arg = np.zeros((1, N_ADDR), int)
    for p, (o, k) in {0: (LD, 2), 1: (CALL, 5), 2: (DJNZ, 1), 3: (HALT, 0), 5: (ADD, 250), 6: (RET, 0)}.items():
        op[0, p], arg[0, p] = o, k
    data = np.zeros((1, N_DATA), int)
    data[0, 2] = 7
    return op, arg, data, np.arange(24)[None], np.array([0]), np.array([3])


def op_pairs(rng, per=400):
    """Every ordered pair of opcodes at addresses 0 and 1, random everything else."""
    op, arg, data, order, A0, B0 = sample(256 * per, rng, HALTY)
    pairs = np.repeat(np.arange(256), per)
    op[:, 0], op[:, 1] = pairs // 16, pairs % 16
    return op, arg, data, order, A0, B0


def op_stress(rng, per=6000):
    """For each opcode, programs where half the instructions are that opcode."""
    parts = []
    for o in range(16):
        p = np.full(16, 0.5 / 15)
        p[o] = 0.5
        parts.append(sample(per, rng, p))
    return tuple(np.concatenate(x) for x in zip(*parts))


def special_values(rng, n=60_000):
    """Registers, args and data drawn from edge values (0, 1, 127, 128, 255, ...)."""
    op, arg, data, order, A0, B0 = sample(n, rng, HALTY)
    edge = np.array([0, 1, 2, 127, 128, 129, 254, 255])
    pick = lambda shape: edge[rng.integers(0, len(edge), shape)]
    return op, pick(arg.shape), pick(data.shape), order, pick(A0.shape), pick(B0.shape)


def battery(theta, seed=0):
    """Run the hidden test battery. Returns True iff exact success (S = 1)."""
    rng = np.random.default_rng(seed)
    tests = [
        ("worked example from the spec", worked_example(), np.float64, True),
        ("every opcode pair at addresses 0-1 (400 programs each)", op_pairs(rng), np.float64, True),
        ("per-opcode stress programs", op_stress(rng), np.float64, True),
        ("edge values (0, 1, 127, 128, 255, ...)", special_values(rng), np.float64, True),
        ("random programs", sample(120_000, rng), np.float64, True),
        ("random programs, more HALTs", sample(120_000, rng, HALTY), np.float64, True),
        ("random programs, control-flow heavy", sample(120_000, rng, FLOWY), np.float64, True),
        ("extra: random programs in float32", sample(30_000, rng, HALTY), np.float32, False),
    ]
    in_spec_errors = 0
    for name, batch, dtype, in_spec in tests:
        t0 = time.time()
        z = predict(theta, batch, dtype)
        want = oracle(batch[0], batch[1], batch[2], batch[4], batch[5])
        bad = z.argmax(1) != want
        err = int(bad.sum())
        zs = np.sort(z, axis=1)
        margin = np.nanmin(zs[:, -1] - zs[:, -2]) if np.all(np.isfinite(z)) else float("nan")
        in_spec_errors += err if in_spec else 0
        print(f"  {'PASS' if err == 0 else 'FAIL'}  {name}: {len(want):,} episodes, {err:,} errors "
              f"({1 - err / len(want):.4%} correct), min margin {margin:.3g}  ({time.time() - t0:.1f}s)")
        if err:
            h = want != TIMEOUT
            print(f"        wrong on HALTED {np.mean(bad[h]):.1%}, on TIMEOUT {np.mean(bad[~h]):.1%}")
    return in_spec_errors == 0


def stats(theta):
    total = nz = 0
    for name, t in theta.items():
        vals = sorted(set(np.round(t[t != 0], 4).tolist()))
        shown = vals if len(vals) <= 8 else f"{len(vals)} distinct values"
        print(f"  {name:6s} {str(t.shape):11s} nonzero {np.count_nonzero(t):6d}/{t.size:<7d} values {shown}")
        total += t.size
        nz += np.count_nonzero(t)
    print(f"  total {total} parameters, {nz} nonzero ({nz / total:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the MICRO-8 test battery on a weights file")
    ap.add_argument("weights")
    args = ap.parse_args()
    theta, problems = load(args.weights)
    if problems:
        print("INVALID weights file:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    ok = battery(theta)
    stats(theta)
    print(f"\nEXACT SUCCESS S = {int(ok)}")
