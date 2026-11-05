# Test programs for sets, Sparse[N, K, CLASS], on the EXPR-128 schema (query.root: Int128, query.arg: Byte, a program
# table, output Byte or FAULT / TIMEOUT). Each is compiled, compared with the lowered program's semantics, and checked
# against the intended answer computed in Python (EXPECT). FAULT is output class 256.
FAULT = 256


def _node(tables, k):
    return next(r for r in tables["program"] if r["addr"] == k)


PROGRAMS["superposed set: 64 elements in 49 channels, membership by name, constant and expression"] = """
seen: Sparse[64, 3, FAULT] = set()
def main() -> Byte:
    a: Int64 = query.arg
    seen.add(a)
    seen.add(5)
    b: Int64 = query.arg >> 2
    if b in seen:
        return 1
    if 7 in seen:
        return 2
    if (query.arg & 63) not in seen:
        return 3
    return 4
"""


def _members(q):
    s = {q["arg"] % 64, 5}
    if (q["arg"] >> 2) in s:
        return 1
    if 7 in s:
        return 2
    if (q["arg"] & 63) not in s:
        return 3
    return 4


EXPECT["superposed set: 64 elements in 49 channels, membership by name, constant and expression"] = \
    lambda q, t: _members(q)

PROGRAMS["visited set: walk the program table until a node repeats (overflow is FAULT)"] = """
visited: Sparse[128, 5, FAULT] = set()
def main() -> Byte:
    node: Int128 = query.root
    count: Byte = 0
    seen: Bool = node in visited
    while not seen:
        visited.add(node)
        n = attend(program.addr, node)
        node = n.x
        count = count + 1
        seen = node in visited
    return count
"""


def _visit(t, root):
    visited, node, count = set(), root, 0
    while node not in visited:
        if len(visited) == 5:
            return FAULT
        visited.add(node)
        node = _node(t, node)["x"] % 128
        count += 1
    return count


EXPECT["visited set: walk the program table until a node repeats (overflow is FAULT)"] = \
    lambda q, t: _visit(t, q["root"])

PROGRAMS["one channel per element when no code is narrower; duplicates and clear()"] = """
s: Sparse[16, 2, FAULT] = set()
def main() -> Byte:
    s.add(query.arg % 16)
    s.add(3)
    s.add(query.arg % 16)
    r: Byte = 0
    if 3 in s:
        r = r + 1
    s.clear()
    if 3 in s:
        r = r + 10
    s.add(query.arg >> 4)
    if (query.arg >> 4) in s:
        r = r + 100
    return r
"""
EXPECT["one channel per element when no code is narrower; duplicates and clear()"] = lambda q, t: 101

PROGRAMS["a third distinct element overflows Sparse[16, 2, FAULT]"] = """
s: Sparse[16, 2, FAULT] = set()
def main() -> Byte:
    s.add(query.arg % 16)
    s.add(query.arg >> 4)
    s.add(7)
    return 9
"""
EXPECT["a third distinct element overflows Sparse[16, 2, FAULT]"] = \
    lambda q, t: FAULT if len({q["arg"] % 16, q["arg"] >> 4, 7}) > 2 else 9

PROGRAMS["one channel per element in groups (N > 32)"] = """
s: Sparse[40, 3, FAULT] = set()
def main() -> Byte:
    a: Int32 = query.arg
    s.add(a)
    s.add(39)
    if a in s:
        if 0 in s:
            return 2
        return 1
    return 0
"""
EXPECT["one channel per element in groups (N > 32)"] = lambda q, t: 2 if q["arg"] % 32 == 0 else 1
BAD["a computed element when N is not a power of two"] = ("""
s: Sparse[40, 3, FAULT] = set()
def main() -> Byte:
    s.add(query.arg % 40)
    return 1
""", "must be a constant or an IntM variable")

BAD["a set in a while condition"] = ("""
s: Sparse[16, 2, FAULT] = set()
def main() -> Byte:
    while 3 in s:
        s.clear()
    return 1
""", "a while condition cannot test a set")
BAD["a set used as a value"] = ("""
s: Sparse[16, 2, FAULT] = set()
def main() -> Byte:
    x: Byte = s
    return x
""", "is a set")
BAD["a set without an overflow class"] = ("""
s: Sparse[16, 2] = set()
def main() -> Byte:
    return 1
""", "declare a set as name: Sparse[N, K, CLASS]")
