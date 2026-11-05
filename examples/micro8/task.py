"""MICRO-8: a tiny 8-bit CPU. Task specification, input encoding and exact oracle.

Machine state
    A, B   8-bit registers (0..255). Initial values come from the query.
    Z, C   zero flag and carry flag, both 0 initially.
    R      return-address register (0..15), 0 initially.
    pc     program counter (0..15), 0 initially.

Memory (given as record tokens, presented in an arbitrary order)
    program  16 instruction records (addr, op, arg): one per addr 0..15, arg is 0..255.
    data      8 data records (daddr, value): one per daddr 0..7, value is 0..255.
    The memory is read-only.

Each executed instruction is the program record whose addr equals pc. All arithmetic on A and B
is mod 256 and all pc arithmetic is mod 16. "t" below means arg mod 16 (a jump target) and
"next" means pc = pc + 1.

    op  name  effect                                                       flags changed
     0  HALT  stop; the output is A                                        -
     1  LDI   A = arg; next                                                -
     2  LD    A = data[arg mod 8]; next                                    -
     3  LDX   A = data[B mod 8]; next                                      -
     4  ADD   A = A + arg; next                                            C = carry out (A + arg >= 256), Z = (new A == 0)
     5  SUB   A = A - arg; next                                            C = borrow (A < arg),           Z = (new A == 0)
     6  ADDB  A = A + B; next                                              C = carry out (A + B >= 256),   Z = (new A == 0)
     7  AND   A = A & arg (bitwise); next                                  Z = (new A == 0)
     8  XOR   A = A ^ arg (bitwise); next                                  Z = (new A == 0)
     9  SHL   A = (A * 2) mod 256; next                                    C = old bit 7 of A,             Z = (new A == 0)
    10  MOVB  B = A; next                                                  -
    11  JZ    if Z == 1: pc = t, else next                                 -
    12  JC    if C == 1: pc = t, else next                                 -
    13  DJNZ  B = B - 1; then if the NEW B != 0: pc = t, else next         -
    14  CALL  R = (pc + 1) mod 16; pc = t                                  -
    15  RET   pc = R                                                       -

Flags not listed for an instruction keep their value.

The machine executes at most 16 instructions.
Output, 257 classes: the value of A (0..255) if the machine stopped with HALT, otherwise
256 (TIMEOUT). A HALT executed as the 16th instruction still counts as stopping.

Worked example. Initial A = 0, B = 3; data[2] = 7; program:
    0: LD 2      1: CALL 5    2: DJNZ 1    3: HALT    5: ADD 250    6: RET    (others HALT)
    LD 2 -> A = 7. CALL 5 (R = 2) -> ADD 250 -> A = 1, C = 1 -> RET to 2 -> DJNZ: B = 2, jump 1.
    Two more rounds: A = 251 then 245, B = 1 then 0. DJNZ falls through to HALT at 3.
    That is 15 instructions. Output: 245.
"""
import numpy as np

N_ADDR = 16           # program addresses
N_DATA = 8            # data addresses
N_REC = N_ADDR + N_DATA
MAX_INSTR = 16        # maximum number of executed instructions
STEPS = 32            # network loop iterations
OPS = ("HALT", "LDI", "LD", "LDX", "ADD", "SUB", "ADDB", "AND", "XOR", "SHL", "MOVB", "JZ", "JC",
       "DJNZ", "CALL", "RET")
HALT, LDI, LD, LDX, ADD, SUB, ADDB, AND, XOR, SHL, MOVB, JZ, JC, DJNZ, CALL, RET = range(16)
TIMEOUT = 256
N_OUT = 257

# Input token features (75 dims). Every token uses the same layout.
# Numbers 0..255 are given in binary, bit i = (value >> i) & 1, bit 0 first.
IN_QUERY = 0      # 1  1 on the query token
IN_INSTR = 1      # 1  1 on program records
IN_DATA = 2       # 1  1 on data records
IN_ADDR = 3       # 16 one-hot program address  (program records)
IN_OP = 19        # 16 one-hot opcode           (program records)
IN_ARG = 35       # 8  bits of arg              (program records)
IN_DADDR = 43     # 8  one-hot data address     (data records)
IN_DVAL = 51      # 8  bits of the data value   (data records)
IN_A = 59         # 8  bits of initial A        (query)
IN_B = 67         # 8  bits of initial B        (query)
D_IN = 75


def bits(v):
    return (np.asarray(v)[..., None] >> np.arange(8)) & 1


def oracle(op, arg, data, A0, B0):
    """Exact answer. op[b, addr], arg[b, addr], data[b, daddr]."""
    n = len(A0)
    b = np.arange(n)
    pc = np.zeros(n, int)
    a, r = A0.astype(int).copy(), B0.astype(int).copy()
    Z, C, R = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    halted = np.zeros(n, bool)
    for _ in range(MAX_INSTR):
        o, k = op[b, pc], arg[b, pc]
        na, nb, nZ, nC, nR, npc = a.copy(), r.copy(), Z.copy(), C.copy(), R.copy(), (pc + 1) % N_ADDR
        m = o == LDI
        na[m] = k[m]
        m = o == LD
        na[m] = data[b, k % N_DATA][m]
        m = o == LDX
        na[m] = data[b, r % N_DATA][m]
        for code, val, carry in ((ADD, a + k, a + k >= 256), (SUB, a - k, a < k), (ADDB, a + r, a + r >= 256),
                                 (SHL, a * 2, a >= 128)):
            m = o == code
            na[m] = val[m] % 256
            nC[m] = carry[m]
        m = o == AND
        na[m] = (a & k)[m]
        m = o == XOR
        na[m] = (a ^ k)[m]
        m = np.isin(o, (ADD, SUB, ADDB, AND, XOR, SHL))
        nZ[m] = na[m] == 0
        m = o == MOVB
        nb[m] = a[m]
        m = (o == JZ) & (Z == 1)
        npc[m] = k[m] % N_ADDR
        m = (o == JC) & (C == 1)
        npc[m] = k[m] % N_ADDR
        m = o == DJNZ
        dec = (r - 1) % 256
        nb[m] = dec[m]
        m = m & (dec != 0)
        npc[m] = k[m] % N_ADDR
        m = o == CALL
        nR[m] = (pc[m] + 1) % N_ADDR
        npc[m] = k[m] % N_ADDR
        m = o == RET
        npc[m] = R[m]
        stop = ~halted & (o == HALT)
        go = ~halted & ~stop
        a, r, Z, C, R, pc = (np.where(go, x, y) for x, y in ((na, a), (nb, r), (nZ, Z), (nC, C), (nR, R), (npc, pc)))
        halted |= stop
    return np.where(halted, a, TIMEOUT)


def encode(op, arg, data, order, A0, B0):
    """order[b, j] = which record sits at position j: 0..15 = program addr, 16..23 = data addr 0..7.
    Returns query token Xq [B, 75] and record tokens Xr [B, 24, 75]."""
    n = len(A0)
    b = np.arange(n)
    Xq = np.zeros((n, D_IN))
    Xq[b, IN_QUERY] = 1
    Xq[:, IN_A:IN_A + 8] = bits(A0)
    Xq[:, IN_B:IN_B + 8] = bits(B0)
    rec = np.zeros((n, N_REC, D_IN))
    k = np.arange(N_ADDR)
    rec[:, k, IN_INSTR] = 1
    rec[:, k, IN_ADDR + k] = 1
    rec[b[:, None], k[None], IN_OP + op] = 1
    rec[:, k, IN_ARG:IN_ARG + 8] = bits(arg)
    d = np.arange(N_DATA)
    rec[:, N_ADDR + d, IN_DATA] = 1
    rec[:, N_ADDR + d, IN_DADDR + d] = 1
    rec[:, N_ADDR + d, IN_DVAL:IN_DVAL + 8] = bits(data)
    Xr = rec[b[:, None], order]
    return Xq, Xr


def sample(n, rng, op_probs=None):
    """Random programs, data and registers (generated on the fly, no dataset)."""
    op = rng.choice(16, size=(n, N_ADDR), p=op_probs)
    arg = rng.integers(0, 256, (n, N_ADDR))
    data = rng.integers(0, 256, (n, N_DATA))
    order = np.argsort(rng.random((n, N_REC)), axis=1)
    return op, arg, data, order, rng.integers(0, 256, n), rng.integers(0, 256, n)
