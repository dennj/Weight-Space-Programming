"""Expression programs for EXPR-128: an assembler from expression trees to program memory, and the program families
of the recursive circuit lab (sum, factorial, fib, pow2, xor_range, even/odd, gcd, popcount, a dead branch), random
expressions and minimax game trees.

    e = If(Le(ARG, C(1)), C(1), Mul(ARG, Call("fact", Sub(ARG, C(1)))))
    prog, root = assemble({"fact": e}, "fact")          # prog: int array [128, 5] of (op, x, y, z, imm)
"""
import math

import numpy as np

from task import BINARY, N_NODES, OPS

ARG = ("ARG",)


def C(v):
    return ("CONST", int(v) & 255)


def Add(a, b): return ("ADD", a, b)
def Sub(a, b): return ("SUB", a, b)
def Mul(a, b): return ("MUL", a, b)
def Le(a, b): return ("LEQ", a, b)
def And(a, b): return ("BAND", a, b)
def Xor(a, b): return ("BXOR", a, b)
def Shr(a, b): return ("SHR", a, b)
def If(c, t, f): return ("IF", c, t, f)
def Call(name, arg): return ("CALL", name, arg)


def assemble(functions, entry):
    """Expression trees -> program memory. Identical subtrees share a node; CALL points at the callee's root."""
    rows, memo, fixups, roots = [], {}, [], {}

    def emit(e):
        if e in memo:
            return memo[e]
        tag = e[0]
        row = [OPS.index(tag), 0, 0, 0, 0]
        if tag == "CONST":
            row[4] = e[1]
        elif tag in ("IF",):
            row[1], row[2], row[3] = emit(e[1]), emit(e[2]), emit(e[3])
        elif tag == "CALL":
            row[1] = emit(e[2])
        elif tag != "ARG":
            row[1], row[2] = emit(e[1]), emit(e[2])
        idx = len(rows)
        if idx >= N_NODES:
            raise ValueError("program memory exhausted")
        rows.append(row)
        memo[e] = idx
        if tag == "CALL":
            fixups.append((idx, e[1]))
        return idx

    for name, e in functions.items():
        roots[name] = emit(e)
    for idx, name in fixups:
        rows[idx][2] = roots[name]
    prog = np.zeros((N_NODES, 5), dtype=np.int64)
    prog[:len(rows)] = rows
    return prog, roots[entry]


def library():
    n = ARG
    a, b = Shr(n, C(4)), And(n, C(15))
    return {
        "sum": ({"sum": If(Le(n, C(0)), C(0), Add(n, Call("sum", Sub(n, C(1)))))}, "sum"),
        "factorial": ({"factorial": If(Le(n, C(1)), C(1), Mul(n, Call("factorial", Sub(n, C(1)))))}, "factorial"),
        "fib": ({"fib": If(Le(n, C(1)), n, Add(Call("fib", Sub(n, C(1))), Call("fib", Sub(n, C(2)))))}, "fib"),
        "pow2": ({"pow2": If(Le(n, C(0)), C(1), Mul(C(2), Call("pow2", Sub(n, C(1)))))}, "pow2"),
        "xor_range": ({"xor_range": If(Le(n, C(0)), C(0), Xor(n, Call("xor_range", Sub(n, C(1)))))}, "xor_range"),
        "even": ({"even": If(Le(n, C(0)), C(1), Call("odd", Sub(n, C(1)))),
                  "odd": If(Le(n, C(0)), C(0), Call("even", Sub(n, C(1))))}, "even"),
        "gcd": ({"gcd": If(Le(a, C(0)), b, If(Le(b, C(0)), a,
                                               If(Le(a, b), Call("gcd", Add(Mul(a, C(16)), Sub(b, a))),
                                                  Call("gcd", Add(Mul(Sub(a, b), C(16)), b)))))}, "gcd"),
        "popcount": ({"popcount": If(Le(n, C(0)), C(0), Add(And(n, C(1)), Call("popcount", Shr(n, C(1)))))},
                     "popcount"),
        "dead_branch": ({"dead": If(C(1), C(42), Call("dead", n))}, "dead"),
    }


def expected(name, n):
    """The mathematical answer, independent of any evaluator (all arithmetic mod 256)."""
    if name == "sum":
        return (n * (n + 1) // 2) & 255
    if name == "factorial":
        return math.factorial(n) & 255
    if name == "fib":
        x, y = 0, 1
        for _ in range(n):
            x, y = y, (x + y) & 255
        return x
    if name == "pow2":
        return (1 << n) & 255
    if name == "xor_range":
        r = 0
        for i in range(n + 1):
            r ^= i
        return r
    if name == "even":
        return int(n % 2 == 0)
    if name == "gcd":
        return math.gcd(n >> 4, n & 15)
    if name == "popcount":
        return bin(n).count("1")
    if name == "dead_branch":
        return 42
    raise KeyError(name)


SUITES = {"sum": list(range(41)), "factorial": list(range(21)), "fib": list(range(13)),
          "pow2": list(range(17)) + [32, 50, 60], "xor_range": list(range(17)) + [31, 50, 60],
          "even": list(range(25)) + [32, 64, 100], "gcd": list(range(256)), "popcount": list(range(256)),
          "dead_branch": [0, 1, 5, 32, 64, 127, 255]}


def random_expr(rng, depth):
    """The lab's generator of random non-recursive expressions."""
    if depth == 0 or rng.random() < .28:
        return ARG if rng.random() < .5 else C(int(rng.integers(0, 256)))
    if rng.random() < .2:
        return If(random_expr(rng, depth - 1), random_expr(rng, depth - 1), random_expr(rng, depth - 1))
    op = OPS[int(rng.choice(BINARY))]
    right = random_expr(rng, depth - 1)
    if op == "SHR" and rng.random() < .7:
        right = C(int(rng.integers(0, 10)))
    return (op, random_expr(rng, depth - 1), right)


def game(rng, depth, maximizing):
    """A random binary game tree as a minimax expression, and its value."""
    if depth == 0:
        v = int(rng.integers(0, 128))
        return C(v), v
    a, va = game(rng, depth - 1, not maximizing)
    b, vb = game(rng, depth - 1, not maximizing)
    return (If(Le(a, b), b, a), max(va, vb)) if maximizing else (If(Le(a, b), a, b), min(va, vb))
