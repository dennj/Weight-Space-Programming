# Test programs for functions, recursion and while loops, on the EXPR-128 schema (query.root: Int128, query.arg: Byte,
# a program table, output Byte or FAULT / TIMEOUT). Each is compiled and compared with the lowered program's semantics.
PROGRAMS["factorial by recursion (MUL by a shift-add loop)"] = """
recursion(depth=16, overflow=FAULT)
def main() -> Byte:
    return fact(query.arg % 8)
def fact(n: Byte) -> Byte:
    if n <= 1:
        return 1
    return mul(n, fact(n - 1))
def mul(a: Byte, b: Byte) -> Byte:
    acc: Byte = 0
    i: Int9 = 0
    while i != 8:
        if (b & 1) == 1:
            acc = acc + a
        a = a << 1
        b = b >> 1
        i = i + 1
    return acc
"""
PROGRAMS["fibonacci, two recursive calls per level"] = """
recursion(depth=24, overflow=FAULT)
def main() -> Byte:
    return fib(query.arg % 8)
def fib(n: Byte) -> Byte:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
PROGRAMS["mutual recursion in tail position (constant stack)"] = """
def main() -> Byte:
    return even(query.arg)
def even(n: Byte) -> Byte:
    if n == 0:
        return 1
    return odd(n - 1)
def odd(n: Byte) -> Byte:
    if n == 0:
        return 0
    return even(n - 1)
"""
PROGRAMS["recursion that overflows the stack, and a fault"] = """
recursion(depth=8, overflow=FAULT)
def main() -> Byte:
    if query.arg == 255:
        stop(FAULT)
    return down(query.arg % 16)
def down(n: Byte) -> Byte:
    if n == 0:
        return 7
    return down(n - 1) + 1
"""
PROGRAMS["gcd by repeated subtraction in a while loop"] = """
def main() -> Byte:
    a: Byte = (query.arg >> 4) | 1
    b: Byte = (query.arg & 15) | 1
    while a != b:
        if a < b:
            b = b - a
        else:
            a = a - b
    return a
"""
PROGRAMS["recursion over the program table (depth-limited tree walk)"] = """
recursion(depth=12, overflow=FAULT)
def main() -> Byte:
    return walk(query.root, 4)
def walk(node: Int128, d: Byte) -> Byte:
    n = attend(program.addr, node)
    if d == 0:
        return n.imm
    a: Byte = walk(n.x, d - 1)
    return a ^ walk(n.y, d - 1)
"""
BAD["missing return"] = ("""
def main() -> Byte:
    x: Byte = 1
""", "can reach its end without returning")
BAD["untyped local"] = ("""
def main() -> Byte:
    y = 3
    return y
""", "declare y with a type")
BAD["non-tail call without a stack"] = ("""
def main() -> Byte:
    return f(query.arg) + 1
def f(n: Byte) -> Byte:
    return n
""", "needs a stack")

# The intended answers, computed directly in Python.
import math as _math


def _node(tables, k):
    return next(r for r in tables["program"] if r["addr"] == k)


def _walk(t, node, d):
    n = _node(t, node % 128)
    if d == 0:
        return n["imm"]
    return _walk(t, n["x"], d - 1) ^ _walk(t, n["y"], d - 1)


def _down(arg):
    if arg == 255:
        return 256                                   # FAULT
    n = arg % 16
    return 256 if n >= 9 else n + 7                  # n pending calls need n frames; the stack has 8


EXPECT["factorial by recursion (MUL by a shift-add loop)"] = lambda q, t: _math.factorial(q["arg"] % 8) & 255
EXPECT["fibonacci, two recursive calls per level"] = lambda q, t: [0, 1, 1, 2, 3, 5, 8, 13][q["arg"] % 8]
EXPECT["mutual recursion in tail position (constant stack)"] = lambda q, t: int(q["arg"] % 2 == 0)
EXPECT["recursion that overflows the stack, and a fault"] = lambda q, t: _down(q["arg"])
EXPECT["gcd by repeated subtraction in a while loop"] = lambda q, t: _math.gcd((q["arg"] >> 4) | 1, (q["arg"] & 15) | 1)
EXPECT["recursion over the program table (depth-limited tree walk)"] = lambda q, t: _walk(t, q["root"], 4)
