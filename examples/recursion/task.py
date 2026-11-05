"""EXPR-128: evaluate a recursive expression program stored in memory. Specification, encoding and exact oracle.

Program memory: 128 node records (addr 0..127; op; x, y, z; imm). x, y and z point to other nodes; only their low
7 bits are used (pointer mod 128). Values are bytes, and all arithmetic is mod 256.

Evaluating node `a` with argument `v`:
    CONST   imm
    ARG     v
    ADD     ev(x) + ev(y)        SUB  ev(x) - ev(y)        MUL  ev(x) * ev(y)          (x is evaluated first)
    LEQ     1 if ev(x) <= ev(y) else 0
    BAND    ev(x) & ev(y)        BXOR ev(x) ^ ev(y)
    SHR     ev(x) >> ev(y), which is 0 when the shift is 8 or more
    IF      ev(y) if ev(x) != 0 else ev(z)      only the chosen branch is evaluated
    CALL    ev(y) with argument ev(x)           y is the root of the called function
    NOP     invalid: evaluating it is a fault (empty memory holds NOP)

Query: root (0..127) and arg (byte). The answer is ev(root) with argument arg.

Stack. An evaluator waits on some evaluations: the condition of an IF, the argument of a CALL, the left and then the
right operand of a two-operand node. Each such pending evaluation holds one frame. The chosen branch of an IF, the
body of a CALL and the root are tail positions and hold none. A machine has DEPTH = 64 frames; needing one more is
a fault.

Output, 258 classes: the value (0..255), FAULT (256), or TIMEOUT (257) when the evaluation does not finish within
the machine's step budget. The tests use programs that finish within MAX_EVALS node evaluations, and programs that
never finish.
"""
import numpy as np

N_NODES = 128
DEPTH = 64
MAX_EVALS = 5000       # at most 13 steps per evaluation: 65,001 steps fit in the 65,536-step budget
OPS = ("NOP", "CONST", "ARG", "ADD", "SUB", "MUL", "LEQ", "BAND", "BXOR", "SHR", "IF", "CALL")
NOP, CONST, ARG, ADD, SUB, MUL, LEQ, BAND, BXOR, SHR, IF, CALL = range(12)
BINARY = (ADD, SUB, MUL, LEQ, BAND, BXOR, SHR)
FAULT, TIMEOUT = 256, 257
N_OUT = 258
UNKNOWN = -1               # finishes, but needs more than MAX_EVALS evaluations: not tested

# Input token features. Numbers are given in binary (bit i = (value >> i) & 1), pointers and addresses one-hot.
IN_QUERY = 0       # 1    1 on the query token
IN_NODE = 1        # 1    1 on node records
IN_ADDR = 2        # 128  one-hot node address
IN_OP = 130        # 12   one-hot op
IN_X = 142         # 8    bits of x
IN_Y = 150         # 8    bits of y
IN_Z = 158         # 8    bits of z
IN_IMM = 166       # 8    bits of imm
IN_ROOT = 174      # 128  one-hot root (query)
IN_ARG = 302       # 8    bits of arg (query)
D_IN = 310


def apply(op, a, b):
    if op == ADD:
        return (a + b) & 255
    if op == SUB:
        return (a - b) & 255
    if op == MUL:
        return (a * b) & 255
    if op == LEQ:
        return int(a <= b)
    if op == BAND:
        return a & b
    if op == BXOR:
        return a ^ b
    if op == SHR:
        return a >> b if b < 8 else 0
    raise ValueError(op)


def oracle(prog, root, arg, max_evals=MAX_EVALS, depth=DEPTH, with_stats=False):
    """prog: int array [128, 5] of (op, x, y, z, imm). Returns the answer class, or UNKNOWN for a program that is
    still running after max_evals node evaluations and might finish later. A program that provably never finishes
    (it revisits an identical machine state) is TIMEOUT."""
    prog = np.asarray(prog)
    stack, node, env, mode, val = [], root % N_NODES, arg, "eval", 0
    evals, deepest, seen = 0, 0, set()
    while True:
        if mode == "eval":
            key = (node, env, tuple(stack))                # the whole machine state
            if key in seen:
                return (TIMEOUT, evals, deepest) if with_stats else TIMEOUT
            seen.add(key)
            evals += 1
            if evals > max_evals:
                return (UNKNOWN, evals, deepest) if with_stats else UNKNOWN
            op, x, y, z, imm = (int(v) for v in prog[node])
            if op == CONST:
                mode, val = "ret", imm
            elif op == ARG:
                mode, val = "ret", env
            elif op == NOP:
                return (FAULT, evals, deepest) if with_stats else FAULT
            else:
                if len(stack) == depth:
                    return (FAULT, evals, deepest) if with_stats else FAULT
                if op == IF:
                    stack.append(("IF", y, z, env))
                elif op == CALL:
                    stack.append(("ARG", y))
                else:
                    stack.append(("L", op, y, env))
                deepest = max(deepest, len(stack))
                node = x % N_NODES
        else:
            if not stack:
                return (val, evals, deepest) if with_stats else val
            fr = stack.pop()
            if fr[0] == "IF":
                node, env, mode = (fr[1] if val != 0 else fr[2]) % N_NODES, fr[3], "eval"
            elif fr[0] == "ARG":
                node, env, mode = fr[1] % N_NODES, val, "eval"
            elif fr[0] == "L":
                stack.append(("R", fr[1], val))
                node, env, mode = fr[2] % N_NODES, fr[3], "eval"
            else:
                val = apply(fr[1], fr[2], val)


def encode(progs, roots, args, rng=None):
    """progs: [n, 128, 5]. Records are presented in a random order."""
    rng = rng or np.random.default_rng(0)
    progs = np.asarray(progs)
    n = len(progs)
    Xq = np.zeros((n, D_IN))
    Xq[:, IN_QUERY] = 1
    Xq[np.arange(n), IN_ROOT + np.asarray(roots) % N_NODES] = 1
    Xq[:, IN_ARG:IN_ARG + 8] = (np.asarray(args)[:, None] >> np.arange(8)) & 1
    Xr = np.zeros((n, N_NODES, D_IN))
    for b in range(n):
        order = rng.permutation(N_NODES)
        for j, a in enumerate(order):
            op, x, y, z, imm = (int(v) for v in progs[b, a])
            r = Xr[b, j]
            r[IN_NODE] = 1
            r[IN_ADDR + a] = 1
            r[IN_OP + op] = 1
            for off, v in ((IN_X, x), (IN_Y, y), (IN_Z, z), (IN_IMM, imm)):
                r[off:off + 8] = (v >> np.arange(8)) & 1
    return Xq, Xr
