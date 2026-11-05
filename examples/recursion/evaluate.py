"""Test battery for any EXPR-128 weights file. It treats the network as a black box: it runs it and compares every
answer with the exact oracle in task.py.

    python3 evaluate.py WEIGHTS.npz [--quick]

Exact success S = 1 means 0 errors on every test. --quick runs a smaller sample of each test.
"""
import argparse
import time

import numpy as np

from network import SHAPES, forward
from programs import ARG, SUITES, Add, C, Call, If, Le, Mul, assemble, expected, game, library, random_expr
from task import BINARY, DEPTH, FAULT, NOP, N_NODES, OPS, TIMEOUT, UNKNOWN, encode, oracle


def load(path):
    th = dict(np.load(path))
    problems = [f"missing tensor {k}" for k in SHAPES if k not in th]
    problems += [f"{k} has shape {th[k].shape}, expected {s}" for k, s in SHAPES.items()
                 if k in th and th[k].shape != s]
    return th, problems


def families():
    out = []
    for name, (fns, entry) in library().items():
        prog, root = assemble(fns, entry)
        for n in SUITES[name]:
            want = oracle(prog, root, n)
            assert want == expected(name, n), (name, n)        # the oracle agrees with the mathematical answer
            out.append((prog, root, n))
    return out


def expressions(rng, count):
    out = []
    while len(out) < count:
        try:
            prog, root = assemble({"main": random_expr(rng, int(rng.integers(2, 8)))}, "main")
        except ValueError:
            continue
        out.append((prog, root, int(rng.integers(0, 256))))
    return out


def games(rng, count):
    """Minimax over random binary game trees, as in the lab: the answer packs the root's score and move."""
    out = []
    for i in range(count):
        depth = 2 + i % 4
        left, _ = game(rng, depth - 1, False)
        right, _ = game(rng, depth - 1, False)
        for role in ("MAX", "MIN"):
            test = Le(left, right)
            best = If(test, right, left) if role == "MAX" else If(test, left, right)
            move = If(test, C(1), C(0)) if role == "MAX" else If(test, C(0), C(1))
            try:
                prog, root = assemble({"main": Add(Mul(C(2), best), move)}, "main")
            except ValueError:
                continue
            out.append((prog, root, 0))
    return out


def operations(rng, per_op):
    """Every two-operand op on random operands (constants and the argument), plus IF and CALL."""
    out = []
    for op in BINARY:
        for _ in range(per_op):
            a = C(int(rng.integers(0, 256))) if rng.random() < .5 else ARG
            b = C(int(rng.integers(0, 256)))
            prog, root = assemble({"main": (OPS[op], a, b)}, "main")
            out.append((prog, root, int(rng.integers(0, 256))))
    for _ in range(per_op):
        c, t, f = (C(int(rng.integers(0, 3))) for _ in range(3))
        prog, root = assemble({"main": If(c, t, Add(f, ARG)), "g": Add(ARG, C(1))}, "main")
        out.append((prog, root, int(rng.integers(0, 256))))
        prog, root = assemble({"main": Call("g", Mul(ARG, C(3))), "g": Add(ARG, C(int(rng.integers(0, 256))))}, "main")
        out.append((prog, root, int(rng.integers(0, 256))))
    return out


def faults(rng, count):
    """Evaluating an empty node (NOP), and recursion deeper than the stack."""
    out = []
    for _ in range(count):
        prog, root = assemble({"main": Add(ARG, If(Le(ARG, C(int(rng.integers(0, 256)))), C(1), C(2)))}, "main")
        node = int(rng.integers(0, root + 1))
        prog[node] = [NOP, 0, 0, 0, 0]                    # empty memory somewhere in the program
        out.append((prog, root, int(rng.integers(0, 256))))
    fns, entry = library()["sum"]
    prog, root = assemble(fns, entry)
    for n in range(DEPTH - 2, DEPTH + 30):                # sum(n) needs about n frames
        out.append((prog, root, n))
    return out


def never(rng, count):
    """Programs that never finish: tail recursion that loops forever (no stack growth)."""
    out = []
    for _ in range(count):
        k = int(rng.integers(0, 256))
        body = Call("loop", Add(ARG, C(k))) if rng.random() < .5 else If(C(1), Call("loop", ARG), C(0))
        prog, root = assemble({"loop": body}, "loop")
        out.append((prog, root, int(rng.integers(0, 256))))
    return out


def random_memories(rng, count):
    """Random program memories, like the compiler's self-check: every node random."""
    out = []
    while len(out) < count:
        prog = np.stack([rng.integers(0, len(OPS), N_NODES)] + [rng.integers(0, 256, N_NODES) for _ in range(4)], 1)
        root, arg = int(rng.integers(0, N_NODES)), int(rng.integers(0, 256))
        if oracle(prog, root, arg) != UNKNOWN:
            out.append((prog, root, arg))
    return out


def run(theta, cases, batch=512, seed=0):
    rng = np.random.default_rng(seed)
    got, steps = [], []
    for s in range(0, len(cases), batch):
        chunk = cases[s:s + batch]
        Xq, Xr = encode([c[0] for c in chunk], [c[1] for c in chunk], [c[2] for c in chunk], rng)
        logits, st = forward(theta, Xq, Xr, return_steps=True)
        got.append(logits.argmax(1))
        steps.append(st)
    return np.concatenate(got), np.concatenate(steps)


def battery(theta, quick=False):
    rng = np.random.default_rng(2026)
    k = 0.1 if quick else 1.0
    tests = [
        ("program families of the lab (sum, factorial, fib, pow2, xor_range, even/odd, gcd, popcount, dead branch)",
         families()),
        ("random expressions (the lab's generator)", expressions(rng, int(1000 * k))),
        ("minimax over random game trees, maximizing and minimizing root", games(rng, int(112 * k) or 12)),
        ("every operation on random operands", operations(rng, int(500 * k))),
        ("faults: empty nodes, recursion deeper than the stack", faults(rng, int(200 * k))),
        ("programs that never finish", never(rng, int(30 * k) or 5)),
        ("random program memories", random_memories(rng, int(2000 * k))),
    ]
    ok = True
    for name, cases in tests:
        t0 = time.time()
        want = np.array([oracle(*c) for c in cases])
        got, steps = run(theta, cases)
        err = int(np.sum(got != want))
        ok &= err == 0
        kinds = f"{np.sum(want < 256)} values, {np.sum(want == FAULT)} FAULT, {np.sum(want == TIMEOUT)} TIMEOUT"
        print(f"  {'PASS' if err == 0 else 'FAIL'}  {name}: {len(cases):,} programs ({kinds}), {err} errors, "
              f"up to {steps.max():,} loop iterations  ({time.time() - t0:.0f}s)", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the EXPR-128 test battery on a weights file")
    ap.add_argument("weights")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    theta, problems = load(args.weights)
    if problems:
        print("INVALID weights file:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    ok = battery(theta, args.quick)
    print(f"\nEXACT SUCCESS S = {int(ok)}")
