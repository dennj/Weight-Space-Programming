# Test programs for the MICRO-8 schema (Byte registers, program + data tables, 2 iterations per step).
PROGRAMS["checksum of args (byte add, wrap)"] = """
pc: Int16 = 0
s: Byte = query.a
def step():
    ins = attend(program.addr, pc)
    s = s + ins.arg
    pc = pc + 1
def output():
    return s
"""
PROGRAMS["data indirection, xor, subtract with borrow flag"] = """
pc: Int16 = 0
acc: Byte = query.b
mix: Byte = query.a
neg: Bool = False
def step():
    ins = attend(program.addr, pc)
    d = attend(data.daddr, ins.arg % 8).value
    if acc < ins.arg:
        neg = True
    acc = acc - ins.arg
    mix = mix ^ d
    pc = pc + 1
def output():
    if neg:
        return acc
    return mix
"""
PROGRAMS["zero test of AND or XOR (finished next step), shifts, jumps, counter"] = """
pc: Int16 = 0
x: Byte = query.a
y: Byte = query.b
zf: Bool = False
n: Int16 = 0
def step():
    ins = attend(program.addr, pc)
    nxt = pc + 1
    if zf:
        nxt = ins.arg % 16
        n = n + 1
    if ins.op == AND or ins.op == ADD:
        x = x & ins.arg
    else:
        x = x ^ ins.arg
    zf = x == 0
    if y > 200:
        y = y >> 2
    else:
        y = y << 1
    pc = nxt
def output():
    return n if y > 7 else x
"""
PROGRAMS["lookup key from a register, halting, decrement loop"] = """
pc: Int16 = 0
a: Byte = query.a
b: Byte = query.b
def step():
    ins = attend(program.addr, pc)
    if ins.op == HALT and a > 100:
        halt
    if ins.op == LD or ins.op == LDX:
        a = attend(data.daddr, b % 8).value
    b = b - 1
    if b != 0:
        pc = ins.arg % 16
    else:
        pc = pc + 3
def output():
    if halted:
        return a
    return b
"""
PROGRAMS["TASK MICRO-8 (capacity check)"] = """
pc: Int16 = 0
a: Byte = query.a
b: Byte = query.b
z: Bool = False
c: Bool = False
r: Int16 = 0

def step():
    ins = attend(program.addr, pc)
    op = ins.op
    k = ins.arg
    nxt = pc + 1
    if op == HALT:
        halt
    elif op == LDI:
        a = k
    elif op == LD:
        a = attend(data.daddr, k % 8).value
    elif op == LDX:
        a = attend(data.daddr, b % 8).value
    elif op == ADD:
        c = a + k > 255
        a = a + k
        z = a == 0
    elif op == SUB:
        c = a < k
        a = a - k
        z = a == 0
    elif op == ADDB:
        c = a + b > 255
        a = a + b
        z = a == 0
    elif op == AND:
        a = a & k
        z = a == 0
    elif op == XOR:
        a = a ^ k
        z = a == 0
    elif op == SHL:
        c = a > 127
        a = a << 1
        z = a == 0
    elif op == MOVB:
        b = a
    elif op == JZ:
        if z:
            nxt = k % 16
    elif op == JC:
        if c:
            nxt = k % 16
    elif op == DJNZ:
        b = b - 1
        if b != 0:
            nxt = k % 16
    elif op == CALL:
        r = pc + 1
        nxt = k % 16
    elif op == RET:
        nxt = r
    pc = nxt

def output():
    if halted:
        return a
    return TIMEOUT
"""
PROGRAMS["TASK MICRO-8, flags set after the branches"] = """
pc: Int16 = 0
a: Byte = query.a
b: Byte = query.b
z: Bool = False
c: Bool = False
r: Int16 = 0

def step():
    ins = attend(program.addr, pc)
    op = ins.op
    k = ins.arg
    nxt = pc + 1
    if op == HALT:
        halt
    elif op == LDI:
        a = k
    elif op == LD:
        a = attend(data.daddr, k % 8).value
    elif op == LDX:
        a = attend(data.daddr, b % 8).value
    elif op == ADD:
        c = a + k > 255
        a = a + k
    elif op == SUB:
        c = a < k
        a = a - k
    elif op == ADDB:
        c = a + b > 255
        a = a + b
    elif op == AND:
        a = a & k
    elif op == XOR:
        a = a ^ k
    elif op == SHL:
        c = a > 127
        a = a << 1
    elif op == MOVB:
        b = a
    elif op == JZ:
        if z:
            nxt = k % 16
    elif op == JC:
        if c:
            nxt = k % 16
    elif op == DJNZ:
        b = b - 1
        if b != 0:
            nxt = k % 16
    elif op == CALL:
        r = pc + 1
        nxt = k % 16
    elif op == RET:
        nxt = r
    if op == ADD or op == SUB or op == ADDB or op == AND or op == XOR or op == SHL:
        z = a == 0
    pc = nxt

def output():
    if halted:
        return a
    return TIMEOUT
"""
BAD["lookup key out of range"] = ("pc: Int16 = 0\nv: Byte = 0\ndef step():\n    ins = attend(program.addr, pc)\n"
                                  "    v = attend(data.daddr, ins.arg).value\n    pc = pc + 1\ndef output():\n    return v\n",
                                  "outside the table")
BAD["bitwise on computed value"] = ("pc: Int16 = 0\nv: Byte = 0\ndef step():\n    ins = attend(program.addr, pc)\n"
                                    "    v = (v + ins.arg) & 7\n    pc = pc + 1\ndef output():\n    return v\n",
                                    "bitwise")
