"""Independent end-to-end verification that a MICRO-8 network executes the CPU exactly.

    python3 verify_cpu.py WEIGHTS.npz [label]      prints a JSON summary line

Every test is a real program run through the whole network: attention fetches the instruction,
the looped block executes it, the head reads out A. Records are always given in random order.

1. Exhaustive ALU: every 8-bit operand pair (65,536) for ADD, SUB, ADDB, AND, XOR, checking the
   result, and the carry and zero flags through conditional-jump probe programs. All 256 values
   for SHL (result, carry, zero) and DJNZ (result, branch).
2. Programs: multiplication by repeated addition (all x in 0..255, y in 1..7) and nested
   subroutine calls (all x).
3. 200,000 fresh random programs (new seed).
"""
import json
import sys
import time

import numpy as np

from evaluate import load, predict
from task import (ADD, ADDB, AND, CALL, DJNZ, HALT, JC, JZ, LDI, N_ADDR, N_DATA, N_REC, RET, SHL, SUB, XOR,
                  oracle, sample)

rng = np.random.default_rng(2026)


def programs(n, rows, A0, B0):
    """rows: dict addr -> (op array or int, arg array or int). Everything else HALT."""
    op = np.full((n, N_ADDR), HALT)
    arg = np.zeros((n, N_ADDR), int)
    for p, (o, k) in rows.items():
        op[:, p], arg[:, p] = o, k
    data = rng.integers(0, 256, (n, N_DATA))
    order = np.argsort(rng.random((n, N_REC)), axis=1)
    return op, arg, data, order, np.broadcast_to(A0, (n,)).copy(), np.broadcast_to(B0, (n,)).copy()


def probe(op_code, what, x, k, y=0):
    """what: 'value' -> output A; 'C' or 'Z' -> output 1 if the flag is set else 0."""
    n = len(x)
    rows = {0: (op_code, k)}
    if what in ("C", "Z"):
        rows.update({1: (JC if what == "C" else JZ, 4), 2: (LDI, 0), 3: (HALT, 0), 4: (LDI, 1), 5: (HALT, 0)})
    return programs(n, rows, x, y)


def run(theta, batch):
    want = oracle(batch[0], batch[1], batch[2], batch[4], batch[5])
    got = predict(theta, batch).argmax(1)
    return len(want), int(np.sum(got != want))


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    theta, problems = load(path)
    assert not problems, problems
    t0 = time.time()
    xs, ks = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    x, k = xs.ravel(), ks.ravel()
    zero = np.zeros_like(x)
    results = {}
    for name, code, flags in (("ADD", ADD, "CZ"), ("SUB", SUB, "CZ"), ("AND", AND, "Z"), ("XOR", XOR, "Z")):
        for what in ("value",) + tuple(flags):
            results[f"{name} {what}"] = run(theta, probe(code, what, x, k))
    for what in ("value", "C", "Z"):                     # ADDB: A + B, all pairs
        results[f"ADDB {what}"] = run(theta, probe(ADDB, what, x, zero, y=k))
    v = np.arange(256)
    for what in ("value", "C", "Z"):
        results[f"SHL {what}"] = run(theta, probe(SHL, what, v, 0))
    # DJNZ: B - 1 via ADDB readout, and the branch
    results["DJNZ value"] = run(theta, programs(256, {0: (DJNZ, 1), 1: (ADDB, 0)}, 0, v))
    results["DJNZ branch"] = run(theta, programs(256, {0: (DJNZ, 4), 1: (LDI, 0), 2: (HALT, 0),
                                                       4: (LDI, 1), 5: (HALT, 0)}, 0, v))
    # programs: multiplication by repeated addition, nested subroutine calls
    mx, my = np.meshgrid(np.arange(256), np.arange(1, 8), indexing="ij")
    results["multiply x*y (y<=7)"] = run(theta, programs(mx.size, {0: (LDI, 0), 1: (ADD, mx.ravel()),
                                                                   2: (DJNZ, 1), 3: (HALT, 0)}, 0, my.ravel()))
    results["subroutine ((2x+1)*2+1)"] = run(theta, programs(256, {0: (LDI, v), 1: (CALL, 5), 2: (CALL, 5),
                                                                   3: (HALT, 0), 5: (SHL, 0), 6: (ADD, 1),
                                                                   7: (RET, 0)}, 0, 0))
    results["random programs (new seed)"] = run(theta, sample(200_000, rng))
    total = sum(n for n, _ in results.values())
    errors = sum(e for _, e in results.values())
    print(json.dumps({"label": label, "total": total, "errors": errors, "seconds": round(time.time() - t0),
                      "tests": results}))


if __name__ == "__main__":
    main()
