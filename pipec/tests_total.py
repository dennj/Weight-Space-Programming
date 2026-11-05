# Programs that do not fit one layer per step as written: the compiler must schedule them over more steps
# (levels 1 and 2). Each is compared with the lowered program's semantics, and with EXPECT, the intended answer
# computed directly in Python. Schema: EXPR-128 (query.root: Int128, query.arg: Byte, a program table).

def _node(tables, k):
    return next(r for r in tables["program"] if r["addr"] == k)


PROGRAMS["chained operations in one statement"] = """
def main() -> Byte:
    a: Byte = query.arg
    y: Byte = ((a + 3) & 15) ^ (a >> 1)
    return y
"""
EXPECT["chained operations in one statement"] = lambda q, t: (((q["arg"] + 3) & 15) ^ (q["arg"] >> 1)) & 255

PROGRAMS["a comparison of two computed sums"] = """
def main() -> Byte:
    a: Byte = query.arg
    b: Byte = a >> 2
    if a + b > 200 - b:
        return a - b
    return b + 7
"""
EXPECT["a comparison of two computed sums"] = lambda q, t: (
    (q["arg"] - (q["arg"] >> 2)) & 255 if q["arg"] + (q["arg"] >> 2) > 200 - (q["arg"] >> 2) else (q["arg"] >> 2) + 7)

PROGRAMS["two comparisons combined in one condition"] = """
def main() -> Byte:
    a: Byte = query.arg
    if a > 20 and a < 100:
        return a + 1
    elif a == 7 or a >= 200:
        return 0
    return a ^ 85
"""
EXPECT["two comparisons combined in one condition"] = lambda q, t: (
    q["arg"] + 1 if 20 < q["arg"] < 100 else 0 if q["arg"] == 7 or q["arg"] >= 200 else q["arg"] ^ 85)

PROGRAMS["a lookup keyed by a field of another lookup"] = """
def main() -> Byte:
    n = attend(program.addr, query.root)
    m = attend(program.addr, n.x % 128)
    return n.imm ^ m.imm
"""
EXPECT["a lookup keyed by a field of another lookup"] = lambda q, t: (
    _node(t, q["root"])["imm"] ^ _node(t, _node(t, q["root"])["x"] % 128)["imm"])

PROGRAMS["a loop whose condition combines two comparisons"] = """
def main() -> Byte:
    x: Byte = query.arg | 1
    steps: Byte = 0
    while x != 1 and steps < 40:
        if (x & 1) == 1:
            x = x + x + x + 1
        else:
            x = x >> 1
        steps = steps + 1
    return steps
"""


def _collatz(a):
    x, steps = a | 1, 0
    while x != 1 and steps < 40:
        x = (3 * x + 1) & 255 if x & 1 else x >> 1
        steps += 1
    return steps


EXPECT["a loop whose condition combines two comparisons"] = lambda q, t: _collatz(q["arg"])

PROGRAMS["a conditional expression inside a comparison (exact values)"] = """
def main() -> Byte:
    a: Byte = query.arg
    if (a + 100 if a < 50 else a - 50) > 120:
        return 1
    return 2
"""
EXPECT["a conditional expression inside a comparison (exact values)"] = lambda q, t: (
    1 if (q["arg"] + 100 if q["arg"] < 50 else q["arg"] - 50) > 120 else 2)

PROGRAMS["recursion with arithmetic on both sides of a comparison"] = """
recursion(depth=12, overflow=FAULT)
def main() -> Byte:
    return f(query.arg & 7, query.arg >> 3)
def f(n: Byte, acc: Byte) -> Byte:
    if n + acc > 40 or n == 0:
        return acc
    return f(n - 1, acc + n + n) + 1
"""


def _f(n, acc):
    if n + acc > 40 or n == 0:
        return acc
    return (_f(n - 1, (acc + 2 * n) & 255) + 1) & 255


EXPECT["recursion with arithmetic on both sides of a comparison"] = lambda q, t: _f(q["arg"] & 7, q["arg"] >> 3)

BAD["a type error is still an error"] = ("""
def main() -> Byte:
    a: Byte = query.arg
    b: Bool = a
    return a
""", "Bool")
