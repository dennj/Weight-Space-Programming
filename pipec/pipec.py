"""pipec 2: compile a decision-pipe program into the weights of the fixed network.

    python3 pipec.py program.pipe [-o weights.npz] [-z] [--check N] [--lowered plain.pipe]

The task's input/output format comes from schema.py and the network from network.py, both
next to the program. The language is described in PIPES.md. The compiler knows how to lower
generic constructs (typed state, lookups, if/else, assignment, halt, integer and bitwise
arithmetic, and functions with recursion and while loops); it knows nothing about any particular task.

Functions and recursion (programs that define main()): a front end first lowers them into a plain
step()/output() program. Each function is cut into blocks, one per step; a call that is not a tail
call pushes a frame (the block to return to and the caller's live values) onto a stack held in state
variables; a return jumps to the saved block, which restores the values and pops. See the Lowering class.

How it compiles:
  * step() is executed symbolically. Every state variable ends up as a set of disjoint cases
    "under condition c, the new value is e".
  * Every condition and every bit of every new value is turned into "terms": a conjunction of
    one-hot/bit literals, optionally with one integer threshold test on a weighted sum of bits.
    Each term becomes 1-4 ReLU units in the recurrent block.
  * Each lookup gets its own loop iteration ("phase"). A lookup whose key depends on an
    earlier lookup runs one phase later; its key is computed into a bus by the previous phase.
    Lookups into the same table in different branches share one phase.
  * A new value that needs two layers (for example a zero test of a bitwise AND) is finished
    at the start of the next step from the stored state, if nothing reads it before then.
  * Before saving, the weights are checked against the program's own semantics on random
    inputs (this checks the compiler, not the program).
"""
import argparse
import ast
import copy
import itertools
import os
import re
import sys

import numpy as np

BETA = 1000.0        # attention score gap; exp(-1000) == 0.0 in float64, so lookups are exact
LIMIT = 64           # a combined condition with more cases than this counts as too deep for one layer
VALUE_LIMIT = 100_000  # safety limit on the number of (case, value) pairs

# Compile target. "relu": the original lowering (default). "relu_margin": every gate is a clean
# step relu(2p+0.5) - relu(2p-0.5) that snaps its output back to exactly 0 or 1 even if its input
# is off by up to 0.25. "tanh": every gate is one saturating unit 0.5 + 0.5*tanh(K*p), and values
# that must pass through linearly use tanh's linear regime.
TARGET = "relu"
TANH_K = 64.0
TANH_EPS = 2.0 ** -7       # 1/EPS = 128 stays far below the float16 maximum
HALTED = "_halted"


class PipeError(Exception):
    def __init__(self, msg, node=None):
        line = getattr(node, "lineno", None)
        super().__init__(f"line {line}: {msg}" if line else msg)


class Deep(Exception):
    """Something one layer of the network can't compute."""


class Halt(Exception):
    pass


class Return(Exception):
    def __init__(self, cls):
        self.cls = cls


# ============================================================================ schema & types

class Schema:
    def __init__(self, mod, D_IN):
        self.enums = {k: tuple(v) for k, v in mod.ENUMS.items()}
        self.consts = {}
        for name, vals in self.enums.items():
            for i, v in enumerate(vals):
                self.consts[v] = (("enum", name), i)
        self.records = {}
        for kind, spec in mod.RECORDS.items():
            fields = {f: (self.type_of(t), off) for f, (t, off) in spec["fields"].items()}
            key = spec["key"]
            if not self.is_cat(fields[key][0]):
                raise ValueError("record keys must be small integers or enums")
            self.records[kind] = dict(key=key, indicator=spec.get("indicator"), fields=fields,
                                      count=self.size(fields[key][0]))
        self.query = {f: (self.type_of(t), off) for f, (t, off) in mod.QUERY.items()}
        self.qflag = mod.QUERY_FLAG
        self.runs = mod.RUNS
        self.out_type = self.type_of(mod.OUTPUT["values"])
        self.classes = dict(mod.OUTPUT["classes"])
        self.n_out = self.size(self.out_type) + len(self.classes)
        self.D_IN = D_IN

    def type_of(self, name):
        if name == "Bool":
            return ("bool",)
        if name == "Byte":
            return ("byte",)
        if name in self.enums:
            return ("enum", name)
        m = re.fullmatch(r"Int(\d+)", name)
        if m and 2 <= int(m.group(1)) <= 256:
            return ("mod", int(m.group(1)))
        return None

    def size(self, t):
        return {"bool": 2, "byte": 256}.get(t[0]) or (t[1] if t[0] == "mod" else len(self.enums[t[1]]))

    @staticmethod
    def is_cat(t):
        return t[0] in ("bool", "mod", "enum")

    @staticmethod
    def tname(t):
        return {"bool": "Bool", "byte": "Byte", "int": "integer"}.get(t[0]) or (
            f"Int{t[1]}" if t[0] == "mod" else t[1])


def intlike(t):
    return t[0] in ("mod", "byte", "int")


# ============================================================================ program + interpreter

class Ctx:
    def __init__(self, env, records, mode):
        self.env, self.records, self.mode, self.locals = env, records, mode, {}


class Program:
    """Parsed program plus the reference interpreter (the language semantics)."""

    def __init__(self, src, schema):
        self.S = schema
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise PipeError(f"syntax error: {e.msg}", e) from None
        self.state, self.pipes = {}, {}
        self.reserved = {"query", "attend", "halt", "halted", "step", "output", "True", "False", HALTED} | set(
            schema.records) | set(schema.consts) | set(schema.classes)
        for node in tree.body:
            if isinstance(node, ast.AnnAssign):
                self._declare(node)
            elif isinstance(node, ast.FunctionDef):
                a = node.args
                if a.args or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or node.decorator_list:
                    raise PipeError("pipes take no arguments", node)
                if node.name in self.pipes or node.name in self.state or node.name in self.reserved - {"step", "output"}:
                    raise PipeError(f"{node.name} is already defined", node)
                self.pipes[node.name] = node
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise PipeError("only state declarations and pipe definitions are allowed at top level", node)
        for name in ("step", "output"):
            if name not in self.pipes:
                raise PipeError(f"missing pipe {name}()")
        self.uses_halt = False
        self._scan(self.pipes["step"].body, "step", ("step",))
        self._scan(self.pipes["output"].body, "output", ("output",))
        if self.uses_halt:
            self.state[HALTED] = (("bool",), ("const", 0))

    def _declare(self, node):
        if not isinstance(node.target, ast.Name):
            raise PipeError("state declaration must be `name: Type = initial_value`", node)
        name = node.target.id
        if name.startswith("_") or name in self.reserved or name in self.state:
            raise PipeError(f"cannot use {name!r} as a state name", node)
        t = self.S.type_of(node.annotation.id) if isinstance(node.annotation, ast.Name) else None
        if t is None:
            raise PipeError(f"unknown type for {name}", node)
        v = node.value
        if v is None:
            raise PipeError(f"state {name} needs an initial value", node)
        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "query":
            if v.attr not in self.S.query:
                raise PipeError(f"query has fields {', '.join(self.S.query)}", node)
            qt = self.S.query[v.attr][0]
            if qt != t:
                raise PipeError(f"query.{v.attr} is {Schema.tname(qt)}, not {Schema.tname(t)}", node)
            init = ("query", v.attr)
        else:
            val = self._const_value(v, t)
            if val is None:
                raise PipeError(f"initial value of {name} must be a query field or a {Schema.tname(t)} constant", node)
            init = ("const", val)
        self.state[name] = (t, init)

    def _const_value(self, v, t):
        if isinstance(v, ast.Constant):
            if t[0] == "bool" and isinstance(v.value, bool):
                return int(v.value)
            if intlike(t) and type(v.value) is int and 0 <= v.value < self.S.size(t):
                return v.value
        if isinstance(v, ast.Name) and v.id in self.S.consts and self.S.consts[v.id][0] == t:
            return self.S.consts[v.id][1]
        return None

    def _scan(self, body, ctx, stack):
        for node in (n for stmt in body for n in ast.walk(stmt)):
            if isinstance(node, ast.Name) and node.id == "halt":
                if ctx != "step":
                    raise PipeError("halt is only allowed in step()", node)
                self.uses_halt = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                f = node.func.id
                if f == "attend" and ctx != "step":
                    raise PipeError("attend() is only allowed in step()", node)
                if f in ("step", "output"):
                    raise PipeError(f"{f}() cannot be called", node)
                if f in self.pipes:
                    if f in stack:
                        raise PipeError(f"pipe {f}() calls itself", node)
                    self._scan(self.pipes[f].body, ctx, stack + (f,))

    # ---------------------------------------------------------------- interpreter

    def init_state(self, query):
        return {n: query[i[1]] if i[0] == "query" else i[1] for n, (t, i) in self.state.items()}

    def step(self, env, records):
        """records: kind -> list of dicts field -> int."""
        if env.get(HALTED):
            return dict(env)
        ctx = Ctx(dict(env), records, "step")
        try:
            self._block(self.pipes["step"].body, ctx)
        except Halt:
            ctx.env[HALTED] = 1
        return ctx.env

    def output(self, env):
        ctx = Ctx(dict(env), None, "output")
        try:
            self._block(self.pipes["output"].body, ctx)
        except Return as r:
            return r.cls
        raise PipeError("output() can finish without returning", self.pipes["output"])

    def _block(self, stmts, ctx):
        for s in stmts:
            self._stmt(s, ctx)

    def _stmt(self, s, ctx):
        if isinstance(s, ast.Pass):
            return
        if isinstance(s, ast.Expr):
            v = s.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return
            if (isinstance(v, ast.Name) and v.id == "halt") or (
                    isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "halt" and not v.args):
                raise Halt()
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in self.pipes and not v.args:
                return self._block(self.pipes[v.func.id].body, ctx)
            raise PipeError("this statement has no effect", s)
        if isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name):
                raise PipeError("assign to a single name", s)
            s = ast.Assign(targets=[s.target], value=ast.BinOp(left=ast.Name(id=s.target.id, lineno=s.lineno),
                                                                op=s.op, right=s.value, lineno=s.lineno), lineno=s.lineno)
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                raise PipeError("assign to a single name", s)
            name = s.targets[0].id
            if name in self.reserved or name in self.pipes or name.startswith("_"):
                raise PipeError(f"cannot assign to {name!r}", s)
            val = self._eval(s.value, ctx)
            if name in self.state:
                if ctx.mode != "step":
                    raise PipeError("state can only change in step()", s)
                ctx.env[name] = self._store(self.state[name][0], val, name, s)
            else:
                ctx.locals[name] = val
            return
        if isinstance(s, ast.If):
            return self._block(s.body if self._bool(s.test, ctx) else s.orelse, ctx)
        if isinstance(s, ast.Return):
            if ctx.mode != "output":
                raise PipeError("return is only allowed in output()", s)
            if isinstance(s.value, ast.Name) and s.value.id in self.S.classes:
                raise Return(self.S.classes[s.value.id])
            if s.value is None:
                raise PipeError("output() must return a value", s)
            t, v = self._eval(s.value, ctx)
            ot = self.S.out_type
            if not ((intlike(t) and intlike(ot)) or t == ot):
                raise PipeError(f"output() must return a {Schema.tname(ot)} or one of {', '.join(self.S.classes)}", s)
            if not 0 <= v < self.S.size(ot):
                raise PipeError(f"output() returned {v}, outside 0..{self.S.size(ot) - 1}", s)
            raise Return(v)
        raise PipeError(f"unsupported statement: {type(s).__name__}", s)

    def _store(self, t, val, name, node):
        vt, v = val
        if t[0] in ("mod", "byte"):
            if not intlike(vt):
                raise PipeError(f"{name} is {Schema.tname(t)} but the value is {Schema.tname(vt)}", node)
            return v % self.S.size(t)
        if vt != t:
            raise PipeError(f"{name} is {Schema.tname(t)} but the value is {Schema.tname(vt)}", node)
        return v

    def _bool(self, node, ctx):
        t, v = self._eval(node, ctx)
        if t[0] != "bool":
            raise PipeError(f"condition must be Bool, not {Schema.tname(t)}", node)
        return v

    def _attend(self, n, ctx):
        if ctx.mode != "step":
            raise PipeError("attend() is only allowed in step()", n)
        if len(n.args) != 2 or n.keywords:
            raise PipeError("use attend(table.key_field, value)", n)
        a, q = n.args
        if not (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id in self.S.records):
            raise PipeError(f"first argument of attend() must be one of "
                            f"{', '.join(k + '.' + r['key'] for k, r in self.S.records.items())}", n)
        kind, field = a.value.id, a.attr
        if field != self.S.records[kind]["key"]:
            raise PipeError(f"records of {kind} are looked up by {kind}.{self.S.records[kind]['key']}", n)
        kt = self.S.records[kind]["fields"][field][0]
        qt, qv = self._eval(q, ctx)
        if not ((intlike(qt) and kt[0] == "mod") or qt == kt):
            raise PipeError(f"{kind}.{field} is {Schema.tname(kt)}; the lookup value is {Schema.tname(qt)}", n)
        for rec in ctx.records[kind]:
            if rec[field] == qv:
                return ("rec", kind), rec
        raise PipeError(f"no {kind} record has {field} == {qv} (keys are 0..{self.S.size(kt) - 1})", n)

    def _eval(self, n, ctx):
        S = self.S
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                return ("bool",), int(n.value)
            if type(n.value) is int and n.value >= 0:
                return ("int",), n.value
            raise PipeError(f"unsupported constant {n.value!r}", n)
        if isinstance(n, ast.Name):
            if n.id in ctx.locals:
                if ctx.locals[n.id][0][0] == "rec":
                    raise PipeError(f"{n.id} is a record; read a field such as {n.id}.value", n)
                return ctx.locals[n.id]
            if n.id in self.state and not n.id.startswith("_"):
                return self.state[n.id][0], ctx.env[n.id]
            if n.id in S.consts:
                return S.consts[n.id]
            if n.id == "halted":
                if ctx.mode != "output":
                    raise PipeError("halted can only be read in output()", n)
                return ("bool",), ctx.env.get(HALTED, 0)
            raise PipeError(f"unknown name {n.id!r}", n)
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "attend":
                rt, rec = self._attend(n.value, ctx)
            elif isinstance(n.value, ast.Name) and n.value.id in ctx.locals and ctx.locals[n.value.id][0][0] == "rec":
                rt, rec = ctx.locals[n.value.id]
            elif isinstance(n.value, ast.Name) and n.value.id == "query":
                raise PipeError("query fields can only be used as initial values of state", n)
            else:
                raise PipeError("unsupported attribute access", n)
            fields = S.records[rt[1]]["fields"]
            if n.attr not in fields:
                raise PipeError(f"{rt[1]} records have fields {', '.join(fields)}", n)
            return fields[n.attr][0], rec[n.attr]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "attend":
            return self._attend(n, ctx)
        if isinstance(n, ast.BoolOp):
            want = isinstance(n.op, ast.Or)
            for x in n.values:
                if self._bool(x, ctx) == want:
                    return ("bool",), int(want)
            return ("bool",), int(not want)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return ("bool",), 1 - self._bool(n.operand, ctx)
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1:
                raise PipeError("chained comparisons are not supported", n)
            (ta, a), (tb, b) = self._eval(n.left, ctx), self._eval(n.comparators[0], ctx)
            op = n.ops[0]
            if not ((intlike(ta) and intlike(tb)) or ta == tb):
                raise PipeError(f"cannot compare {Schema.tname(ta)} with {Schema.tname(tb)}", n)
            if isinstance(op, (ast.Eq, ast.NotEq)):
                return ("bool",), int((a == b) == isinstance(op, ast.Eq))
            if not intlike(ta):
                raise PipeError(f"{Schema.tname(ta)} values support only == and !=", n)
            r = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b}.get(type(op))
            if r is None:
                raise PipeError("unsupported comparison", n)
            return ("bool",), int(r)
        if isinstance(n, ast.BinOp):
            (ta, a), (tb, b) = self._eval(n.left, ctx), self._eval(n.right, ctx)
            if not (intlike(ta) and intlike(tb)):
                raise PipeError("arithmetic works on integers only", n)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                return ("int",), a + b if isinstance(n.op, ast.Add) else a - b
            if isinstance(n.op, (ast.BitAnd, ast.BitXor, ast.BitOr)):
                for t, v in ((ta, a), (tb, b)):
                    if not (t[0] == "byte" or (t[0] == "int" and 0 <= v < 256 and self._is_const(n))):
                        raise PipeError("bitwise operators work on Byte values (store a computed value in a "
                                        "Byte variable first)", n)
                r = a & b if isinstance(n.op, ast.BitAnd) else a ^ b if isinstance(n.op, ast.BitXor) else a | b
                return ("byte",), r
            if isinstance(n.op, (ast.LShift, ast.RShift, ast.Mod)):
                if not isinstance(n.right, ast.Constant) or type(n.right.value) is not int or n.right.value < 0:
                    raise PipeError("<<, >> and % need a constant on the right", n)
                if isinstance(n.op, ast.Mod):
                    if b == 0:
                        raise PipeError("% 0", n)
                    return ("int",), a % b
                if ta[0] != "byte":
                    raise PipeError("<< and >> work on Byte values", n)
                return (("int",), a << b) if isinstance(n.op, ast.LShift) else (("byte",), a >> b)
            raise PipeError("unsupported operator", n)
        if isinstance(n, ast.IfExp):
            return self._eval(n.body if self._bool(n.test, ctx) else n.orelse, ctx)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in (REC + "bit", REC + "bits"):
            return self._builtin(n, ctx)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.pipes:
            raise PipeError("pipes do not return values; call them as statements", n)
        raise PipeError(f"unsupported expression: {ast.unparse(n)}", n)

    def _builtin(self, n, ctx):
        """Internal functions that lowered programs use to pack values into stack frames of Bool channels:
        rec_bit(x, i) is bit i of x (mod 256), and rec_bits(b0, ..., b7) is the Byte with those bits."""
        if n.func.id == REC + "bit":
            t, v = self._eval(n.args[0], ctx)
            if not intlike(t):
                raise PipeError(f"{REC}bit() takes an integer", n)
            return ("bool",), (v % 256 >> n.args[1].value) & 1
        return ("byte",), sum(self._bool(a, ctx) << i for i, a in enumerate(n.args))

    @staticmethod
    def _is_const(n):
        return True      # integer operands of bitwise ops are range-checked above


# ============================================================================ symbolic values

class Node:
    __slots__ = ("t",)


class Inp(Node):           # state variable at the start of the step (or final state, in output())
    __slots__ = ("v",)

    def __init__(self, v, t):
        self.v, self.t = v, t


class Fld(Node):           # field of a lookup
    __slots__ = ("site", "f")

    def __init__(self, site, f, t):
        self.site, self.f, self.t = site, f, t


class Const(Node):
    __slots__ = ("val",)

    def __init__(self, val, t):
        self.val, self.t = val, t


class Op(Node):             # arithmetic / bitwise / shift / mod / wrap / compare / bool ops
    __slots__ = ("op", "args")

    def __init__(self, op, args, t):
        self.op, self.args, self.t = op, args, t


class ITE(Node):
    __slots__ = ("cond", "a", "b")

    def __init__(self, cond, a, b):
        self.cond, self.a, self.b, self.t = cond, a, b, a.t


class Site:
    def __init__(self, idx, kind, query, path, node):
        self.idx, self.kind, self.query, self.path, self.node = idx, kind, query, path, node
        self.group = None


# ============================================================================ term algebra
# A term is (lits, test): lits maps a slot to its allowed set of values; test is None or
# (linkey, lo, hi) meaning lo <= sum(coef * slot) + const <= hi (None = unbounded).
# A condition (Cond) is a list of disjoint terms.

class Algebra:
    def __init__(self):
        self.size = {}             # slot -> number of values (2 for bits / bools)
        self.hoister = None        # turns a test into a precomputed 0/1 slot, when possible

    def lit(self, slot, allowed):
        allowed = frozenset(allowed)
        if not allowed:
            return []
        if len(allowed) == self.size[slot]:
            return [({}, None)]
        return [({slot: allowed}, None)]

    def lin_range(self, linkey):
        items, const = linkey
        return const + sum(min(0, c) for _, c in items), const + sum(max(0, c) for _, c in items)

    def norm_test(self, test):
        """Clip a test to the possible range. Returns 'all', 'none' or a normalized test."""
        linkey, lo, hi = test
        lmin, lmax = self.lin_range(linkey)
        lo = None if lo is None or lo <= lmin else lo
        hi = None if hi is None or hi >= lmax else hi
        if (lo is not None and lo > lmax) or (hi is not None and hi < lmin) or (
                lo is not None and hi is not None and lo > hi):
            return "none"
        if lo is None and hi is None:
            return "all"
        return (linkey, lo, hi)

    def test_term(self, test):
        t = self.norm_test(test)
        return [] if t == "none" else [({}, None)] if t == "all" else [({}, t)]

    def tconj(self, a, b):
        lits = dict(a[0])
        for s, v in b[0].items():
            w = lits[s] & v if s in lits else v
            if not w:
                return None
            lits[s] = w
        ta, tb = a[1], b[1]
        if ta is None or tb is None:
            test = ta or tb
        elif ta[0] == tb[0]:
            lo = ta[1] if tb[1] is None else tb[1] if ta[1] is None else max(ta[1], tb[1])
            hi = ta[2] if tb[2] is None else tb[2] if ta[2] is None else min(ta[2], tb[2])
            test = self.norm_test((ta[0], lo, hi))
            if test == "none":
                return None
            test = None if test == "all" else test
        else:
            if self.hoister is not None:
                for keep, other in ((ta, tb), (tb, ta)):
                    slot = self.hoister(other)
                    if slot is not None:
                        return self.tconj((lits, keep), ({slot: frozenset({1})}, None))
            raise Deep("two different arithmetic tests in one condition")
        return (lits, test)

    def conj(self, A, B):
        out = []
        for a in A:
            for b in B:
                t = self.tconj(a, b)
                if t is not None:
                    out.append(t)
        if len(out) > LIMIT:
            raise Deep("too many cases")
        return out

    def neg_term(self, t):
        lits, test = t
        out, prefix = [], {}
        for s in sorted(lits, key=repr):
            comp = frozenset(range(self.size[s])) - lits[s]
            if comp:
                out.append(({**prefix, s: comp}, None))
            prefix[s] = lits[s]
        if test is not None:
            linkey, lo, hi = test
            for part in ((linkey, None, lo - 1) if lo is not None else None,
                         (linkey, hi + 1, None) if hi is not None else None):
                if part is not None:
                    p = self.norm_test(part)
                    if p != "none":
                        out.append((dict(prefix), None if p == "all" else p))
        return out

    def neg(self, A):
        out = [({}, None)]
        for t in A:
            out = self.conj(out, self.neg_term(t))
        return out

    def disj(self, A, B):
        return A + self.conj(self.neg(A), B)

    def merge(self, terms):
        """Merge terms that differ only in one slot's allowed set (same test)."""
        terms = [(dict(l), t) for l, t in terms]
        changed = True
        while changed:
            changed = False
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    (la, ta), (lb, tb) = terms[i], terms[j]
                    if ta != tb or set(la) != set(lb):
                        continue
                    diff = [s for s in la if la[s] != lb[s]]
                    if len(diff) == 1 and not (la[diff[0]] & lb[diff[0]]):
                        s = diff[0]
                        u = la[s] | lb[s]
                        new = dict(la)
                        if len(u) == self.size[s]:
                            del new[s]
                        else:
                            new[s] = u
                        terms[i] = (new, ta)
                        del terms[j]
                        changed = True
                        break
                if changed:
                    break
        return terms


def linkey(coefs, const):
    return (tuple(sorted(((s, c) for s, c in coefs.items() if c), key=repr)), const)


class Lin:
    """sum(coef * bit) + const, optionally reduced mod `mod`."""
    __slots__ = ("coefs", "const", "mod")

    def __init__(self, coefs, const, mod=None):
        self.coefs, self.const, self.mod = {s: c for s, c in coefs.items() if c}, const, mod

    def add(self, o, sign):
        if self.mod or o.mod:
            raise Deep("arithmetic on a value that was already reduced")
        c = dict(self.coefs)
        for s, v in o.coefs.items():
            c[s] = c.get(s, 0) + sign * v
        return Lin(c, self.const + sign * o.const)

    def word(self):
        """If this is exactly one byte word (bits with coefficients 2^i), return its slots by bit."""
        if self.const or self.mod:
            return None
        if self.coefs and all(s[0] == "st" for s in self.coefs):     # rec_bits(...) of Bool state variables
            by = {}
            for s, c in self.coefs.items():
                if c <= 0 or c & (c - 1) or c >= 256 or c.bit_length() - 1 in by:
                    return None
                by[c.bit_length() - 1] = s
            return by
        by = {}
        for s, c in self.coefs.items():
            if s[0] not in ("stb", "fdb") or c != 2 ** s[-1]:
                return None
            by[s[-1]] = s
        return by if len({s[:-1] for s in self.coefs}) <= 1 else None


class BitsV:
    """A byte given bit by bit (each bit a condition)."""
    __slots__ = ("bits",)

    def __init__(self, bits):
        self.bits = bits


# ============================================================================ compiler

class Compiler:
    def __init__(self, prog, net, task):
        self.P, self.S, self.net, self.task = prog, prog.S, net, task
        self.A = Algebra()
        self.sites = []
        if net.STEPS % self.S.runs:
            raise PipeError("internal: loop count is not a multiple of the step count")
        self.phases = net.STEPS // self.S.runs
        for v, (t, _) in prog.state.items():
            self._register_state(v, t)
        self.temps, self.tmp_index = [], {}
        self.A.hoister = self.hoist

    def hoist(self, test):
        """Precompute a test in the previous loop iteration of the step, if it only reads start-of-step
        state or results of first-iteration lookups."""
        if self.phases < 2:
            return None
        if test in self.tmp_index:
            return self.tmp_index[test]
        for s, _ in test[0][0]:
            if not (s[0] == "stb" or (s[0] == "fdb" and self.sites[s[1]].early)):
                return None
        slot = ("tmp", len(self.temps))
        self.temps.append(test)
        self.tmp_index[test] = slot
        self.A.size[slot] = 2
        return slot

    # ---------------------------------------------------------------- slots
    def _register_state(self, v, t):
        if t[0] == "byte":
            for i in range(8):
                self.A.size[("stb", v, i)] = 2
        else:
            self.A.size[("st", v)] = self.S.size(t)

    def _register_field(self, site, f, t):
        if t[0] == "byte":
            for i in range(8):
                self.A.size[("fdb", site.idx, f, i)] = 2
        else:
            self.A.size[("fd", site.idx, f)] = self.S.size(t)

    def slot_of(self, node):
        if isinstance(node, Inp):
            return ("st", node.v) if node.t[0] != "byte" else None
        return ("fd", node.site.idx, node.f) if node.t[0] != "byte" else None

    def bit_slots(self, node):
        if isinstance(node, Inp):
            return [("stb", node.v, i) for i in range(8)]
        return [("fdb", node.site.idx, node.f, i) for i in range(8)]

    # ---------------------------------------------------------------- symbolic execution
    def run_step(self):
        env = {v: Inp(v, t) for v, (t, _) in self.P.state.items()}
        self.terminated = []
        live, path, env = self._sblock(self.P.pipes["step"].body, env, {}, [({}, None)])
        self.live_env = env if live else {}
        final = {}
        for v in self.P.state:
            node = env[v] if live else None
            for cond, tenv in reversed(self.terminated):
                node = tenv[v] if node is None else ITE(cond, tenv[v], node)
            final[v] = node
        return final

    def _sblock(self, stmts, env, loc, path):
        for s in stmts:
            alive, path, env, loc = self._sstmt(s, env, loc, path)
            if not alive:
                return False, path, env
        return True, path, env

    def _sstmt(self, s, env, loc, path):
        A = self.A
        if isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
            return True, path, env, loc
        if isinstance(s, ast.Expr):
            v = s.value
            if (isinstance(v, ast.Name) and v.id == "halt") or (
                    isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "halt"):
                tenv = dict(env)
                tenv[HALTED] = Const(1, ("bool",))
                self.terminated.append((path, tenv))
                return False, path, env, loc
            if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in self.P.pipes):
                raise PipeError("this statement has no effect", s)
            return self._sblock_loc(self.P.pipes[v.func.id].body, env, loc, path)
        if isinstance(s, ast.AugAssign):
            s = ast.Assign(targets=[s.target], value=ast.BinOp(left=ast.Name(id=s.target.id, lineno=s.lineno),
                                                                op=s.op, right=s.value, lineno=s.lineno), lineno=s.lineno)
        if isinstance(s, ast.Assign):
            name = s.targets[0].id
            val = self._seval(s.value, env, loc, path)
            if name in self.P.state:
                t = self.P.state[name][0]
                if t[0] in ("mod", "byte") and val.t[0] != t[0] or (t[0] == "mod" and val.t != t):
                    val = Op("wrap", (val, self.S.size(t)), t)
                env = dict(env)
                env[name] = val
            else:
                loc = dict(loc)
                loc[name] = val
            return True, path, env, loc
        if isinstance(s, ast.If):
            c = self.cond_of(self._seval(s.test, env, loc, path), s.test)
            pt, pf = self._pconj(path, c, s), self._pconj(path, self._neg(c, s), s)
            if pt:
                at, _, et, lt = self._sblock_loc(s.body, env, loc, pt)
            else:
                at, et, lt = False, env, loc
            if pf:
                af, _, ef, lf = self._sblock_loc(s.orelse, env, loc, pf)
            else:
                af, ef, lf = False, env, loc
            if at and af:
                env = {v: et[v] if et[v] is ef[v] else ITE(c, et[v], ef[v]) for v in et}
                merged = {}
                for k in lt:
                    if k in lf:
                        if lt[k] is lf[k]:
                            merged[k] = lt[k]
                        elif isinstance(lt[k], Node) and isinstance(lf[k], Node):
                            merged[k] = ITE(c, lt[k], lf[k])
                return True, path, env, merged
            if at:
                return True, pt, et, lt
            if af:
                return True, pf, ef, lf
            return False, path, env, loc
        raise PipeError(f"unsupported statement: {type(s).__name__}", s)

    def _sblock_loc(self, stmts, env, loc, path):
        for s in stmts:
            alive, path, env, loc = self._sstmt(s, env, loc, path)
            if not alive:
                return False, path, env, loc
        return True, path, env, loc

    def _pconj(self, path, c, node):
        try:
            return self.A.conj(path, c)
        except Deep as e:
            raise PipeError(f"this condition is too complex to combine with the enclosing conditions ({e})", node)

    def _neg(self, c, node):
        try:
            return self.A.neg(c)
        except Deep as e:
            raise PipeError(f"this condition is too complex ({e})", node)

    def _attend_site(self, n, env, loc, path):
        a, q = n.args
        kind = a.value.id
        site = Site(len(self.sites), kind, self._seval(q, env, loc, path), path, n)
        key_t = self.S.records[kind]["fields"][self.S.records[kind]["key"]][0]
        site.early = isinstance(site.query, Inp) and site.query.t == key_t and not self._cond_sites(path)
        self.sites.append(site)
        for f, (t, _) in self.S.records[kind]["fields"].items():
            self._register_field(site, f, t)
        return site

    def _seval(self, n, env, loc, path):
        S = self.S
        if isinstance(n, ast.Constant):
            return Const(int(n.value), ("bool",) if isinstance(n.value, bool) else ("int",))
        if isinstance(n, ast.Name):
            if n.id in loc:
                return loc[n.id]
            if n.id in self.P.state:
                return env[n.id]
            if n.id in S.consts:
                t, v = S.consts[n.id]
                return Const(v, t)
            if n.id == "halted":
                return env[HALTED] if HALTED in env else Const(0, ("bool",))
            raise PipeError(f"unknown name {n.id!r}", n)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "attend":
            return self._attend_site(n, env, loc, path)
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Call):
                site = self._attend_site(n.value, env, loc, path)
            elif isinstance(n.value, ast.Name) and isinstance(loc.get(n.value.id), Site):
                site = loc[n.value.id]
            else:
                raise PipeError("unsupported attribute access", n)
            return Fld(site, n.attr, S.records[site.kind]["fields"][n.attr][0])
        if isinstance(n, ast.BoolOp):
            return Op("and" if isinstance(n.op, ast.And) else "or",
                      tuple(self._seval(x, env, loc, path) for x in n.values), ("bool",))
        if isinstance(n, ast.UnaryOp):
            return Op("not", (self._seval(n.operand, env, loc, path),), ("bool",))
        if isinstance(n, ast.Compare):
            op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(n.ops[0])]
            return Op(op, (self._seval(n.left, env, loc, path), self._seval(n.comparators[0], env, loc, path)),
                      ("bool",))
        if isinstance(n, ast.BinOp):
            a, b = self._seval(n.left, env, loc, path), self._seval(n.right, env, loc, path)
            op = {ast.Add: "+", ast.Sub: "-", ast.BitAnd: "&", ast.BitXor: "^", ast.BitOr: "|",
                  ast.LShift: "<<", ast.RShift: ">>", ast.Mod: "%"}[type(n.op)]
            t = ("byte",) if op in ("&", "^", "|", ">>") else ("int",)
            return Op(op, (a, b), t)
        if isinstance(n, ast.IfExp):
            c = self.cond_of(self._seval(n.test, env, loc, path), n.test)
            return ITE(c, self._seval(n.body, env, loc, path), self._seval(n.orelse, env, loc, path))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == REC + "bit":
            return Op("bit", (self._seval(n.args[0], env, loc, path), Const(n.args[1].value, ("int",))), ("bool",))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == REC + "bits":
            return Op("bits", tuple(self._seval(a, env, loc, path) for a in n.args), ("byte",))
        raise PipeError(f"unsupported expression: {ast.unparse(n)}", n)

    # ---------------------------------------------------------------- lowering of values
    def cond_of(self, node, src=None):
        try:
            return self._cond(node)
        except Deep as e:
            if src is None:
                raise
            raise PipeError(f"this condition needs more than one layer of the network ({e})", src)

    def _cond(self, n):
        A = self.A
        if isinstance(n, Const):
            return [({}, None)] if n.val else []
        if isinstance(n, (Inp, Fld)):
            return A.lit(self.slot_of(n), {1})
        if isinstance(n, ITE):
            return A.disj(A.conj(n.cond, self._cond(n.a)), A.conj(A.neg(n.cond), self._cond(n.b)))
        if n.op == "not":
            return A.neg(self._cond(n.args[0]))
        if n.op in ("and", "or"):
            out = self._cond(n.args[0])
            for x in n.args[1:]:
                out = A.conj(out, self._cond(x)) if n.op == "and" else A.disj(out, self._cond(x))
            return out
        if n.op in ("==", "!=", "<", "<=", ">", ">="):
            return self._compare(n.op, *n.args)
        if n.op == "bit":                                   # bit i of a byte value
            out = []
            for c, v in self.values(n.args[0]):
                out += A.conj(c, self.byte_bits(v)[n.args[1].val])
            return out
        raise Deep(f"unsupported Boolean operation {n.op}")

    def _compare(self, op, a, b):
        A = self.A
        for x, y in ((a, b), (b, a)):               # (u ^ v) == 0  <=>  u == v
            if isinstance(x, Op) and x.op == "^" and isinstance(y, Const) and y.val == 0 and op in ("==", "!="):
                return self._compare(op, *x.args)
        out = []
        for ca, va in self.values(a):
            for cb, vb in self.values(b):
                c = A.conj(ca, cb)
                if not c:
                    continue
                if isinstance(va, int) and isinstance(vb, int):
                    if {"==": va == vb, "!=": va != vb, "<": va < vb, "<=": va <= vb, ">": va > vb,
                            ">=": va >= vb}[op]:
                        out += c
                    continue
                if isinstance(va, BitsV) or isinstance(vb, BitsV):
                    bv, other = (va, vb) if isinstance(va, BitsV) else (vb, va)
                    if not isinstance(other, int) or op not in ("==", "!="):
                        raise Deep("comparison of a bitwise result with something other than a constant")
                    eq = [({}, None)]
                    for i in range(8):
                        eq = A.conj(eq, bv.bits[i] if (other >> i) & 1 else A.neg(bv.bits[i]))
                    out += A.conj(c, eq if op == "==" else A.neg(eq))
                    continue
                la, lb = self._as_lin(va), self._as_lin(vb)
                if la.mod or lb.mod:
                    lm, k = (la, lb) if la.mod else (lb, la)
                    if lb.mod and la.mod or k.coefs:
                        raise Deep("comparison of two computed values that wrap around")
                    flip = lm is lb
                    out += A.conj(c, self._mod_compare(lm, self._flip(op) if flip else op, k.const))
                    continue
                L = la.add(lb, -1)
                key = linkey(L.coefs, L.const)
                rng = {"==": [(0, 0)], "!=": [(None, -1), (1, None)], "<": [(None, -1)], "<=": [(None, 0)],
                       ">": [(1, None)], ">=": [(0, None)]}[op]
                for lo, hi in rng:
                    out += A.conj(c, A.test_term((key, lo, hi)))
        if len(out) > VALUE_LIMIT:
            raise Deep("too many cases")
        return out

    @staticmethod
    def _flip(op):
        return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)

    def _mod_compare(self, L, op, k):
        """(L mod m) op k, as a union of ranges of L."""
        m = L.mod
        key = linkey(L.coefs, L.const)
        lmin, lmax = self.A.lin_range(key)
        out = []
        for blk in range(lmin // m, lmax // m + 1):
            base = blk * m
            ok = [r for r in range(m) if {"==": r == k, "!=": r != k, "<": r < k, "<=": r <= k, ">": r > k,
                                         ">=": r >= k}[op]]
            for lo, hi in _runs(ok):
                out += self.A.test_term((key, base + lo, base + hi))
        return out

    def _as_lin(self, v):
        if isinstance(v, int):
            return Lin({}, v)
        return v

    def values(self, n):
        """Disjoint list of (cond, value); value is an int, a Lin or a BitsV."""
        A = self.A
        if isinstance(n, Const):
            return [([({}, None)], n.val)]
        if isinstance(n, (Inp, Fld)):
            if n.t[0] == "byte":
                return [([({}, None)], Lin({s: 2 ** i for i, s in enumerate(self.bit_slots(n))}, 0))]
            s = self.slot_of(n)
            return [(A.lit(s, {k}), k) for k in range(A.size[s])]
        if isinstance(n, ITE):
            out = [(A.conj(n.cond, c), v) for c, v in self.values(n.a)]
            nc = A.neg(n.cond)
            out += [(A.conj(nc, c), v) for c, v in self.values(n.b)]
            return [(c, v) for c, v in out if c]
        if n.t[0] == "bool":
            c = self._cond(n)
            return [(c, 1), (A.neg(c), 0)]
        op = n.op
        if op == "bits":                                    # a Byte from Bool bits
            if all(isinstance(a, Inp) and a.t[0] == "bool" for a in n.args):
                return [([({}, None)], Lin({("st", a.v): 2 ** i for i, a in enumerate(n.args)}, 0))]
            return [([({}, None)], BitsV([self._cond(a) for a in n.args] + [[]] * (8 - len(n.args))))]
        if op in ("+", "-"):
            out = []
            sign = 1 if op == "+" else -1
            for ca, va in self.values(n.args[0]):
                for cb, vb in self.values(n.args[1]):
                    c = A.conj(ca, cb)
                    if not c:
                        continue
                    if isinstance(va, BitsV) or isinstance(vb, BitsV):
                        raise Deep("arithmetic on the result of a bitwise operation")
                    if isinstance(va, int) and isinstance(vb, int):
                        out.append((c, va + sign * vb))
                    else:
                        out.append((c, self._as_lin(va).add(self._as_lin(vb), sign)))
            return self._check(out)
        if op == "wrap" or op == "%":
            x, m = n.args[0], (n.args[1] if op == "wrap" else n.args[1].val)
            out = []
            for c, v in self.values(x):
                if isinstance(v, int):
                    out.append((c, v % m))
                elif isinstance(v, BitsV):
                    if m == 256:
                        out.append((c, v))
                    elif 256 % m == 0:
                        bits = m.bit_length() - 1
                        for r in range(m):
                            cc = c
                            for i in range(bits):
                                cc = A.conj(cc, v.bits[i] if (r >> i) & 1 else A.neg(v.bits[i]))
                            if cc:
                                out.append((cc, r))
                    else:
                        raise Deep("% of a bitwise result by a non-power of two")
                else:
                    word = v.word()
                    lmin, lmax = A.lin_range(linkey(v.coefs, v.const))
                    if word is not None and m & (m - 1) == 0 and m <= 256:
                        bits = m.bit_length() - 1
                        for r in range(m):
                            cc = c
                            for i in range(bits):
                                cc = A.conj(cc, A.lit(word[i], {(r >> i) & 1}) if i in word else
                                            ([({}, None)] if not (r >> i) & 1 else []))
                            if cc:
                                out.append((cc, r))
                    elif 0 <= lmin and lmax < m and not v.mod:
                        out.append((c, v))
                    elif m == 256 and not v.mod:
                        out.append((c, Lin(v.coefs, v.const, 256)))
                    else:
                        raise Deep(f"% {m} of a computed value")
            return self._check(out)
        if op in ("&", "^", "|"):
            out = []
            for ca, va in self.values(n.args[0]):
                for cb, vb in self.values(n.args[1]):
                    c = A.conj(ca, cb)
                    if c:
                        ba, bb = self._bits(va), self._bits(vb)
                        if op == "&":
                            bits = [A.conj(x, y) for x, y in zip(ba, bb)]
                        elif op == "|":
                            bits = [A.disj(x, y) for x, y in zip(ba, bb)]
                        else:
                            bits = [A.conj(x, A.neg(y)) + A.conj(A.neg(x), y) for x, y in zip(ba, bb)]
                        out.append((c, BitsV(bits)))
            return self._check(out)
        if op == "<<":
            k = n.args[1].val
            out = []
            for c, v in self.values(n.args[0]):
                if isinstance(v, int):
                    out.append((c, v << k))
                elif isinstance(v, Lin) and not v.mod:
                    out.append((c, Lin({s: x << k for s, x in v.coefs.items()}, v.const << k)))
                else:
                    raise Deep("<< of a computed value")
            return self._check(out)
        if op == ">>":
            k = n.args[1].val
            out = []
            for c, v in self.values(n.args[0]):
                if isinstance(v, int):
                    out.append((c, v >> k))
                else:
                    bits = self._bits(v)
                    out.append((c, BitsV([bits[i + k] if i + k < 8 else [] for i in range(8)])))
            return self._check(out)
        raise Deep(f"unsupported operation {op}")

    def _check(self, out):
        out = [(c, v) for c, v in out if c]
        if sum(len(c) for c, _ in out) > VALUE_LIMIT:
            raise Deep("too many cases")
        return out

    def _bits(self, v):
        """Bit conditions of a byte value that is directly available (no arithmetic)."""
        A = self.A
        if isinstance(v, int):
            if not 0 <= v < 256:
                raise Deep("bitwise operation on a value outside 0..255")
            return [[({}, None)] if (v >> i) & 1 else [] for i in range(8)]
        if isinstance(v, BitsV):
            return v.bits
        word = v.word()
        if word is None:
            raise Deep("bitwise operation on a computed value")
        return [A.lit(word[i], {1}) if i in word else [] for i in range(8)]

    def byte_bits(self, v):
        """Bit conditions of (v mod 256) for an int, Lin or BitsV value."""
        A = self.A
        if isinstance(v, (int, BitsV)):
            return self._bits(v % 256 if isinstance(v, int) else v)
        word = v.word()
        if word is not None:
            return self._bits(v)
        out = []
        for i in range(8):
            M = 2 ** (i + 1)
            coefs, lin_const, K = {}, 0, v.const
            for s, c in v.coefs.items():
                if c >= 0:
                    coefs[s] = c % M
                else:                                   # c*x == a*(1-x) - a  (mod M), a = |c| mod M
                    a = (-c) % M
                    coefs[s] = -a
                    lin_const += a
                    K -= a
            key = linkey(coefs, lin_const + K % M)
            lmin, lmax = A.lin_range(key)
            bit = []
            for blk in range(lmin // M, lmax // M + 1):
                bit += A.test_term((key, blk * M + M // 2, blk * M + M - 1))
            out.append(bit)
        return out

    # ---------------------------------------------------------------- flatten a final value
    def flatten(self, node, cond=None):
        cond = [({}, None)] if cond is None else cond
        if isinstance(node, ITE):
            return (self.flatten(node.a, self.A.conj(cond, node.cond)) +
                    self.flatten(node.b, self.A.conj(cond, self.A.neg(node.cond))))
        return [(cond, node)] if cond else []

    # ---------------------------------------------------------------- per-variable updates
    def updates(self, v, leaf, cond):
        """Terms that write the new value `leaf` of state variable v under `cond`.
        Returns (writes, erases): lists of (channel key, term)."""
        A = self.A
        t = self.P.state[v][0]
        w, e = [], []
        if t[0] == "byte":
            got = self.values(leaf)
            bits_by_case = [(c, self.byte_bits(val)) for c, val in got]
            for i in range(8):
                slot = ("stb", v, i)
                for c, bits in bits_by_case:
                    for term in A.conj(A.conj(cond, c), bits[i]):
                        w.append((slot, 1, term))
                for term in A.conj(cond, A.lit(slot, {1})):
                    e.append((slot, 1, term))
        elif t[0] == "bool":
            slot = ("st", v)
            nc = A.conj(cond, self._cond(leaf))
            w += [(slot, 1, term) for term in nc]
            e += [(slot, 1, term) for term in A.conj(cond, A.lit(slot, {1}))]
        else:
            slot = ("st", v)
            n = A.size[slot]
            for c, val in self.values(leaf):
                if not isinstance(val, int):
                    raise Deep("a computed value that is not a small integer")
                val %= n
                w += [(slot, val, term) for term in A.conj(cond, c)]
            for k in range(n):
                e += [(slot, k, term) for term in A.conj(cond, A.lit(slot, {k}))]
        return w, e

    # ---------------------------------------------------------------- deferral
    def deferral(self, v, cond, leaf, finals):
        """Rewrite `leaf` in terms of the state at the end of the step, if possible."""
        A = self.A
        owners = {}
        for u, cases in finals.items():
            for c, lf in cases:
                owners.setdefault(id(lf), []).append((u, c))
        whole = {id(node): u for u, node in self.final.items()}
        live = {id(node): u for u, node in self.live_env.items()}

        def sub(n):
            if id(n) in whole and whole[id(n)] != v:          # the variable's complete new value
                u = whole[id(n)]
                return Inp(u, self.P.state[u][0])
            if id(n) in live and live[id(n)] != v:            # its value on the paths that don't halt
                u = live[id(n)]
                if all(tenv[u] is n for tc, tenv in self.terminated if A.conj(tc, cond)):
                    return Inp(u, self.P.state[u][0])
            cands = owners.get(id(n), [])
            for u, _ in cands:
                if u == v:
                    continue
                ok = all(lf is n for c, lf in finals[u] if A.conj(c, cond))
                if ok:
                    return Inp(u, self.P.state[u][0])
            if isinstance(n, Inp):
                raise Deep("needs a start-of-step value that changes in this step")
            if isinstance(n, Fld):
                raise Deep("reads a lookup result")
            if isinstance(n, Const):
                return n
            if isinstance(n, ITE):
                raise Deep("conditional value")
            return Op(n.op, tuple(sub(x) if isinstance(x, Node) else x for x in n.args), n.t)

        return sub(leaf)

    # ---------------------------------------------------------------- main
    def compile(self):
        P, S, A, net = self.P, self.S, self.A, self.net
        final = self.run_step()
        self.final = final
        # ---- lookups: groups and phases
        groups = []
        for site in self.sites:
            q = site.query
            direct = isinstance(q, Inp) and q.t == S.records[site.kind]["fields"][S.records[site.kind]["key"]][0]
            deps = self._site_deps(q) | self._cond_sites(site.path)
            ph = 1 + max([self.sites[d].group["phase"] for d in deps] + [-1])
            if not direct:
                ph = max(ph, 1)
            g = next((g for g in groups if g["phase"] == ph and g["kind"] == site.kind and
                      (ph > 0 or g["sites"][0].query.v == q.v)), None)
            if g is None:
                if any(g["phase"] == ph for g in groups):
                    raise PipeError("two different lookups would need the same loop iteration; the network "
                                    "has one attention head per iteration", site.node)
                g = {"phase": ph, "kind": site.kind, "sites": [], "idx": len(groups)}
                groups.append(g)
            g["sites"].append(site)
            site.group = g
        if any(g["phase"] >= self.phases for g in groups):
            raise PipeError(f"lookups need {1 + max(g['phase'] for g in groups)} loop iterations per step, but "
                            f"the network has {self.phases} per step")
        for g in groups:
            for i, a in enumerate(g["sites"]):
                for b in g["sites"][i + 1:]:
                    if A.conj(a.path, b.path):
                        raise PipeError("two lookups into the same table can happen in the same step",
                                        b.node)
        self.groups = groups
        last = self.phases - 1

        # ---- state updates (last phase), with deferral when one layer isn't enough
        finals = {v: self.flatten(final[v]) for v in P.state}
        writes, erases, deferred = [], [], []
        for v in P.state:
            for cond, leaf in finals[v]:
                if isinstance(leaf, Inp) and leaf.v == v:
                    continue
                try:
                    w, e = self.updates(v, leaf, cond)
                except Deep as err:
                    if self.phases < 2:
                        raise PipeError(f"the new value of {v} needs more than one layer ({err})")
                    try:
                        leaf2 = self.deferral(v, cond, leaf, finals)
                        w, e = self.updates(v, leaf2, [({}, None)])
                    except Deep as err2:
                        raise PipeError(f"the new value of {v} needs more than one layer ({err}), and it can't be "
                                        f"finished at the start of the next step either ({err2})")
                    deferred.append((v, cond, w, e))
                    continue
                writes += w
                erases += e
        self.deferred = deferred
        self.bus_terms = {}
        for g in groups:
            if g["phase"] > 0:
                kt = S.records[g["kind"]]["fields"][S.records[g["kind"]]["key"]][0]
                n = S.size(kt)
                terms = []
                for site in g["sites"]:
                    for c, val in self.values(site.query):
                        if not isinstance(val, int) or not 0 <= val < n:
                            raise PipeError("a lookup value can fall outside the table's keys; reduce it with %",
                                            site.node)
                        for term in A.conj(site.path, c):
                            if any(s[0] == "tmp" for s in term[0]):
                                raise PipeError("this lookup is inside a condition that combines two arithmetic "
                                                "tests; that needs more loop iterations", site.node)
                            terms.append((term, val))
                self.bus_terms[g["idx"]] = terms
        return self._emit(final, finals, writes, erases)

    def _site_deps(self, n):
        out = set()

        def walk(x):
            if isinstance(x, Fld):
                out.add(x.site.idx)
            elif isinstance(x, Op):
                for a in x.args:
                    if isinstance(a, Node):
                        walk(a)
            elif isinstance(x, ITE):
                out.update(self._cond_sites(x.cond))
                walk(x.a)
                walk(x.b)
        walk(n)
        return out

    def _cond_sites(self, cond):
        out = set()
        for lits, test in cond:
            for s in lits:
                if s[0] in ("fd", "fdb"):
                    out.add(s[1])
            if test:
                for s, _ in test[0][0]:
                    if s[0] in ("fd", "fdb"):
                        out.add(s[1])
        return out

    # ---------------------------------------------------------------- weights
    def _emit(self, final, finals, writes, erases):
        P, S, A, net, task = self.P, self.S, self.A, self.net, self.task
        from network import zeros
        th = zeros()
        D = th["W_E"].shape[0]
        ch = {}
        nxt = [0]

        def take(k):
            c = list(range(nxt[0], nxt[0] + k))
            nxt[0] += k
            if nxt[0] > D:
                raise PipeError(f"the program needs more than the network's {D} residual channels")
            return c

        last = self.phases - 1
        # record fields used, by kind (shared by all lookups into that kind)
        used, read_last, read_bus = {}, set(), set()
        for site in self.sites:
            used.setdefault(site.kind, set()).add(S.records[site.kind]["key"])
        for s in self._all_field_slots(writes, erases):
            site = self.sites[s[1]]
            used[site.kind].add(s[2])
            read_last.add((site.group["idx"], s[2]))
        for g in self.groups:
            for site in g["sites"]:
                for s in self._query_field_slots(site):
                    src = self.sites[s[1]]
                    used[src.kind].add(s[2])
                    if src.group["phase"] < g["phase"] - 1:
                        read_bus.add((src.group["idx"], s[2]))
        for test in self.temps:
            for s, _ in test[0][0]:
                if s[0] == "fdb":
                    src = self.sites[s[1]]
                    used[src.kind].add(s[2])
                    if src.group["phase"] < last - 1:
                        read_bus.add((src.group["idx"], s[2]))
        read = {(self.groups[gi]["kind"], f) for gi, f in read_last | read_bus}
        for test in self.temps:
            read |= {(self.sites[s[1]].kind, s[2]) for s, _ in test[0][0] if s[0] == "fdb"}
        for g in self.groups:
            for site in g["sites"]:
                for s in self._query_field_slots(site):
                    read.add((self.sites[s[1]].kind, s[2]))
        fch = {}
        for kind in S.records:
            for f in sorted(used.get(kind, ())):
                t = S.records[kind]["fields"][f][0]
                fch[(kind, f)] = take(8 if t[0] == "byte" else S.size(t))
        kind_ind = {}
        if self.phases > 1 or len(used) > 1:
            for kind in used:
                kind_ind[kind] = take(1)[0]
        # A table's key channels are only used on record tokens, so on the state token they can hold
        # the variable or bus that is matched against them.
        share_var, share_bus, taken = {}, {}, set()
        for g in self.groups:
            kk = (g["kind"], S.records[g["kind"]]["key"])
            if kk in read or kk in taken:
                continue
            if g["phase"] == 0:
                v = g["sites"][0].query.v
                t = P.state[v][0]
                if v not in share_var and t[0] != "byte" and S.size(t) == len(fch[kk]):
                    share_var[v] = kk
                    taken.add(kk)
            else:
                share_bus[g["idx"]] = kk
                taken.add(kk)
        for v, (t, _) in P.state.items():
            ch[v] = fch[share_var[v]] if v in share_var else take(
                8 if t[0] == "byte" else (1 if t[0] == "bool" else S.size(t)))
        phase_ch = take(self.phases) if self.phases > 1 else []
        bus = {}
        for g in self.groups:
            if g["phase"] > 0:
                kt = S.records[g["kind"]]["fields"][S.records[g["kind"]]["key"]][0]
                bus[g["idx"]] = fch[share_bus[g["idx"]]] if g["idx"] in share_bus else take(S.size(kt))
        latch = {}
        for gi, f in sorted(read_last | read_bus):
            if self.groups[gi]["phase"] < last:
                latch[(gi, f)] = take(len(fch[(self.groups[gi]["kind"], f)]))
        pend = [take(1)[0] for _ in self.deferred]
        tmp = [take(1)[0] for _ in self.temps]
        self.layout = dict(fields=fch, state=ch, phase=phase_ch, bus=bus, latch=latch, pending=pend, temps=tmp)

        # ---- encoder
        W_E = th["W_E"]
        for (kind, f), c in fch.items():
            t, off = S.records[kind]["fields"][f]
            for i in range(len(c)):
                W_E[c[i], off + i] = 1
        for kind, c in kind_ind.items():
            W_E[c, S.records[kind]["indicator"]] = 1
        for v, (t, init) in P.state.items():
            c = ch[v]
            if init[0] == "query":
                off = S.query[init[1]][1]
                for i in range(len(c)):
                    W_E[c[i], off + i] = 1
            else:
                val = init[1]
                if t[0] == "byte":
                    for i in range(8):
                        if (val >> i) & 1:
                            W_E[c[i], S.qflag] = 1
                elif t[0] == "bool":
                    if val:
                        W_E[c[0], S.qflag] = 1
                else:
                    W_E[c[val], S.qflag] = 1
        if phase_ch:
            W_E[phase_ch[0], S.qflag] = 1

        # ---- attention
        dq, dv = 0, 0
        for g in self.groups:
            kind = g["kind"]
            key = S.records[kind]["key"]
            kc = fch[(kind, key)]
            qc = ch[g["sites"][0].query.v] if g["phase"] == 0 else bus[g["idx"]]
            for i in range(len(kc)):
                th["W_Q"][dq, qc[i]] = BETA
                th["W_K"][dq, kc[i]] = 1
                dq += 1
            if kind in kind_ind and phase_ch:
                th["W_Q"][dq, phase_ch[g["phase"]]] = BETA
                th["W_K"][dq, kind_ind[kind]] = 1
                dq += 1
        copied = []
        for (kind, f), c in fch.items():
            if (kind, f) in read:
                for x in c:
                    th["W_V"][dv, x] = 1
                    th["W_O"][x, dv] = 1
                    copied.append(x)
                    dv += 1
        if dq > th["W_Q"].shape[0] or dv > th["W_V"].shape[0]:
            raise PipeError("the lookups need a wider attention head than the network has")

        # ---- controller units
        self.th, self.units = th, 0
        halted = HALTED in P.state

        def chans(slot, phase):
            if slot[0] == "st":
                return ch[slot[1]]
            if slot[0] == "stb":
                return [ch[slot[1]][slot[2]]]
            if slot[0] in ("ph",):
                return [phase_ch[slot[1]]]
            if slot[0] == "pend":
                return [pend[slot[1]]]
            if slot[0] == "tmp":
                return [tmp[slot[1]]]
            if slot[0] == "bus":
                return bus[slot[1]]
            if slot[0] == "latch":
                return latch[(slot[1], slot[2])] if len(slot) == 3 else [latch[(slot[1], slot[2])][slot[3]]]
            site = self.sites[slot[1]]
            g = site.group
            base = fch[(site.kind, slot[2])] if g["phase"] == phase else latch.get((g["idx"], slot[2]))
            if g["phase"] > phase or base is None:
                raise PipeError("internal: field read before its lookup")
            return base if slot[0] == "fd" else [base[slot[3]]]

        def gate(term, phase):
            lits = dict(term[0])
            if phase_ch:
                A.size[("ph", phase)] = 2
                lits[("ph", phase)] = frozenset({1})
            if halted:
                lits[("st", HALTED)] = lits.get(("st", HALTED), frozenset({0, 1})) & frozenset({0})
                if not lits[("st", HALTED)]:
                    return None
            return (lits, term[1])

        def emit(term, outs, phase, gated=True):
            if gated:
                term = gate(term, phase)
                if term is None:
                    return
            lower_term(A, term, outs, lambda s: chans(s, phase), self._unit)

        def target(slot, value):
            if slot[0] == "st":
                c = ch[slot[1]]
                return c[0] if len(c) == 1 else c[value]
            return ch[slot[1]][slot[2]]

        def emit_updates(ws, es, phase):
            by = {}
            for slot, val, term in ws:
                by.setdefault((target(slot, val), 1.0), []).append(term)
            for slot, val, term in es:
                by.setdefault((target(slot, val), -1.0), []).append(term)
            for (c, sign), terms in by.items():
                for term in A.merge(terms):
                    emit(term, {c: sign}, phase)

        # last phase: state updates
        emit_updates(writes, erases, last)
        # deferred updates: set the pending bit now, finish at phase 0 of the next step
        for k, (v, cond, w, e) in enumerate(self.deferred):
            A.size[("pend", k)] = 2
            for term in A.merge(cond):
                emit(term, {pend[k]: 1.0}, last)
            pl = A.lit(("pend", k), {1})
            emit_updates([(s, val, t2) for s, val, t in w for t2 in A.conj(pl, [t])],
                         [(s, val, t2) for s, val, t in e for t2 in A.conj(pl, [t])], 0)
            emit(pl[0], {pend[k]: -1.0}, 0)
        # earlier phases: latch lookup results, compute buses
        for (gi, f), dst in latch.items():
            g = self.groups[gi]
            src = fch[(g["kind"], f)]
            for i in range(len(src)):
                self._copy_unit(src[i], dst[i], g["phase"], gate)
        for g in self.groups:
            if g["phase"] > 0:
                by_val = {}
                for term, val in self.bus_terms[g["idx"]]:
                    by_val.setdefault(val, []).append(term)
                for val, terms in by_val.items():
                    for term in A.merge(terms):
                        emit(term, {bus[g["idx"]][val]: 1.0}, g["phase"] - 1)
                for i, c in enumerate(bus[g["idx"]]):
                    if TARGET == "relu":
                        self._unit({c: 1.0}, 0.0, {c: -1.0})
                    else:                                   # the bus is exactly 0/1
                        self._unit({c: 1.0}, -0.5, {c: -1.0}, kind="step")
        for k, test in enumerate(self.temps):                 # hoisted tests, one iteration early
            emit(({}, test), {tmp[k]: 1.0}, last - 1)
            emit(({("tmp", k): frozenset({1})}, None), {tmp[k]: -1.0}, last - 1)
        # deferred values must be ready before anything reads them
        early = {s for g in self.groups if g["phase"] > 0 for site in g["sites"]
                 for s in self._node_state_slots(site.query) | self._cond_state(site.path)}
        early |= {g["sites"][0].query.v for g in self.groups if g["phase"] == 0}
        dvars = {v for v, *_ in self.deferred}
        for test in self.temps:
            if {s[1] for s, _ in test[0][0] if s[0] == "stb"} & dvars:
                raise PipeError("a precomputed test reads a value that is only finished at the start of the step")
        for v, cond, w, e in self.deferred:
            reads = set()
            for _, _, t in w + e:
                reads |= self._cond_state([t])
            if reads & (dvars - {v}):
                raise PipeError(f"{v} is finished at the start of the next step from another value that is "
                                f"also finished then")
        for v, *_ in self.deferred:
            if v in early:
                raise PipeError(f"{v} is finished at the start of the next step, but a lookup needs it earlier")
        # phase counter
        if phase_ch:
            for p in range(self.phases):
                emit(({}, None), {phase_ch[p]: -1.0, phase_ch[(p + 1) % self.phases]: 1.0}, p)
        # clear lookup scratch every iteration
        for c in sorted(copied):
            self._unit({c: 1.0}, 0.0, {c: -1.0}, kind="linear")
        if self.units > th["W_R1"].shape[0]:
            raise PipeError(f"step() needs {self.units} controller units but the network has "
                            f"{th['W_R1'].shape[0]}")
        n_ctrl = self.units

        # ---- output head
        n_head = self._head(ch)
        return th, n_ctrl, n_head, nxt[0]

    def _copy_unit(self, src, dst, phase, gate):
        # dst := src (both 0/1), at the given phase
        th = self.th
        lits = gate(({}, None), phase)
        if lits is None:
            return
        base_w, base_c = {}, 0.0
        for slot, allowed in lits[0].items():
            c = (self.layout["phase"][slot[1]] if slot[0] == "ph" else self.layout["state"][slot[1]][0])
            if allowed == {1}:
                base_w[c] = base_w.get(c, 0.0) - 1.0
                base_c += 1.0
            else:
                base_w[c] = base_w.get(c, 0.0) + 1.0
        for x, out in ((src, 1.0), (dst, -1.0)):
            w = {k: -v for k, v in base_w.items()}
            w[x] = w.get(x, 0.0) + 1.0
            if TARGET == "relu":
                self._unit(w, -base_c, {dst: out})
            else:
                self._unit(w, -base_c - 0.5, {dst: out}, kind="step")

    def _unit(self, w, b, outs, kind="relu"):
        def put(wi, bi, oi):
            u = self.units
            if u < self.th["W_R1"].shape[0]:
                for k, x in wi.items():
                    self.th["W_R1"][u, k] += x
                self.th["b_R1"][u] = bi
                for k, x in oi.items():
                    self.th["W_R2"][k, u] += x
            self.units += 1
        realize(kind, w, b, outs, put, self.th["b_R2"])

    def _all_field_slots(self, writes, erases):
        out = set()
        for lst in (writes, erases):
            for _, _, (lits, test) in lst:
                for s in lits:
                    if s[0] in ("fd", "fdb"):
                        out.add(s)
                if test:
                    for s, _ in test[0][0]:
                        if s[0] in ("fd", "fdb"):
                            out.add(s)
        for v, cond, w, e in getattr(self, "deferred", []):
            for lits, test in cond:
                for s in lits:
                    if s[0] in ("fd", "fdb"):
                        out.add(s)
                if test:
                    for s, _ in test[0][0]:
                        if s[0] in ("fd", "fdb"):
                            out.add(s)
        return out

    def _query_field_slots(self, site):
        out = set()
        for c, _ in self.values(site.query):
            for lits, test in c:
                out |= {s for s in lits if s[0] in ("fd", "fdb")}
        for lits, test in site.path:
            out |= {s for s in lits if s[0] in ("fd", "fdb")}
            if test:
                out |= {s for s, _ in test[0][0] if s[0] in ("fd", "fdb")}
        return out

    def _node_state_slots(self, n):
        out = set()
        for c, _ in self.values(n):
            out |= self._cond_state(c)
        return out

    def _cond_state(self, cond):
        out = set()
        for lits, test in cond:
            out |= {s[1] for s in lits if s[0] in ("st", "stb")}
            if test:
                out |= {s[1] for s, _ in test[0][0] if s[0] in ("st", "stb")}
        return out

    def _head(self, ch):
        P, S, A, th = self.P, self.S, self.A, self.th
        env = {v: Inp(v, t) for v, (t, _) in P.state.items()}
        rets, path = [], [({}, None)]
        A.hoister = None
        self._hblock(P.pipes["output"].body, env, {}, path, rets)
        covered = []
        for c, _ in rets:
            covered += c
        if A.neg(covered):
            raise PipeError("output() can finish without returning", P.pipes["output"])
        deferred = {v for v, *_ in self.deferred}
        H = th["W_H"].shape[0]
        u = [0]
        nvals = S.size(S.out_type)
        gain = 9.0 if S.out_type[0] == "byte" else 1.0

        def hunit(w, b, outs, kind="relu"):
            def put(wi, bi, oi):
                if u[0] < H:
                    for k, x in wi.items():
                        th["W_H"][u[0], k] += x
                    th["b_H"][u[0]] = bi
                    for k, x in oi.items():
                        th["W_out"][k, u[0]] += x
                u[0] += 1
            realize(kind, w, b, outs, put, th["b_out"])

        def hchans(slot):
            if slot[1] in deferred:
                raise PipeError(f"output() reads {slot[1]}, which may not be finished when the program ends")
            if slot[0] == "st":
                return ch[slot[1]]
            if slot[0] == "stb":
                return [ch[slot[1]][slot[2]]]
            raise PipeError("output() can only read state variables")

        def unit(term, outs):
            lower_term(A, term, outs, hchans, hunit)

        cases = []
        for cond, ret in rets:
            if isinstance(ret, int):
                cases.append((cond, ret))
            else:
                cases += self.flatten(ret, cond)
        for cond, ret in cases:
            if isinstance(ret, int):
                for term in A.merge(cond):
                    unit(term, {ret: gain})
                continue
            if ret.t[0] == "byte" or (isinstance(ret, Op) and ret.op == "wrap" and ret.t[0] == "byte"):
                vals = self.values(ret)
                for c, val in vals:
                    bits = self.byte_bits(val)
                    cc = A.conj(cond, c)
                    # logit_k = sum_i (2k_i - 1)(2 v_i - 1) = sum_i 2(2k_i - 1)[v_i] - sum_i (2k_i - 1)
                    for term in A.merge(cc):
                        unit(term, {k: -sum(2.0 * ((k >> i) & 1) - 1.0 for i in range(8)) for k in range(nvals)})
                    for i in range(8):
                        for term in A.merge(A.conj(cc, bits[i])):
                            unit(term, {k: 2.0 * (2.0 * ((k >> i) & 1) - 1.0) for k in range(nvals)})
                continue
            for c, val in self.values(ret):
                if not isinstance(val, int):
                    raise PipeError("output() can only return stored values")
                for term in A.merge(A.conj(cond, c)):
                    unit(term, {val % nvals: gain})
        if u[0] > H:
            raise PipeError(f"output() needs {u[0]} head units but the network has {H}")
        return u[0]

    def _hblock(self, stmts, env, loc, path, rets):
        for s in stmts:
            if not path:
                return path
            path = self._hstmt(s, env, loc, path, rets)
        return path

    def _hstmt(self, s, env, loc, path, rets):
        A = self.A
        if isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
            return path
        if isinstance(s, ast.Expr):
            return self._hblock(self.P.pipes[s.value.func.id].body, env, loc, path, rets)
        if isinstance(s, ast.Assign):
            loc[s.targets[0].id] = self._seval(s.value, env, loc, path)
            return path
        if isinstance(s, ast.If):
            c = self.cond_of(self._seval(s.test, env, loc, path), s.test)
            pt, pf = A.conj(path, c), A.conj(path, A.neg(c))
            rt = self._hblock(s.body, env, dict(loc), pt, rets) if pt else []
            rf = self._hblock(s.orelse, env, dict(loc), pf, rets) if pf else []
            return rt + rf
        if isinstance(s, ast.Return):
            if isinstance(s.value, ast.Name) and s.value.id in self.S.classes:
                rets.append((path, self.S.classes[s.value.id]))
            else:
                rets.append((path, self._seval(s.value, env, loc, path)))
            return []
        raise PipeError(f"unsupported statement in output(): {type(s).__name__}", s)


def realize(kind, w, b, outs, put, out_bias):
    """Write one logical unit for the current TARGET. put(weights, bias, outputs) adds a hidden unit.
    kind "relu": relu(w.x + b). kind "step": [w.x + b > 0] for inputs at least 0.5 away from 0.
    kind "linear": w.x + b for inputs >= 0 (used to clear scratch values)."""
    scale = lambda d, f: {k: f * v for k, v in d.items()}
    if kind == "relu" or (kind == "linear" and TARGET != "tanh"):
        put(w, b, outs)
    elif kind == "linear":                                   # tanh(eps*x)/eps ~ x
        put(scale(w, TANH_EPS), TANH_EPS * b, scale(outs, 1.0 / TANH_EPS))
    elif TARGET == "relu_margin":                            # relu(2p+0.5) - relu(2p-0.5)
        put(scale(w, 2.0), 2.0 * b + 0.5, outs)
        put(scale(w, 2.0), 2.0 * b - 0.5, scale(outs, -1.0))
    elif TARGET == "tanh":                                   # 0.5 + 0.5*tanh(K p)
        put(scale(w, TANH_K), TANH_K * b, scale(outs, 0.5))
        for k, v in outs.items():
            out_bias[k] += 0.5 * v
    else:
        raise ValueError(f"unknown target {TARGET}")


def lower_term(A, term, outs, chans, unit):
    """Write the units for one term: exactly 1 inside the term, 0 outside (for 0/1 inputs).
    unit(weights, bias, outputs[, kind]) appends one logical unit."""
    lits, test = term
    w, const = {}, 0.0                       # number of unsatisfied literals = const + w.x
    for slot, allowed in lits.items():
        size = A.size[slot]
        if len(allowed) == size:
            continue
        c = chans(slot)
        if size == 2 and len(c) == 1:
            if allowed == {1}:
                w[c[0]] = w.get(c[0], 0.0) - 1.0
                const += 1.0
            else:
                w[c[0]] = w.get(c[0], 0.0) + 1.0
        else:
            const += 1.0
            for s in allowed:
                w[c[s]] = w.get(c[s], 0.0) - 1.0
    neg = {k: -x for k, x in w.items()}
    step_mode = TARGET != "relu"
    if test is None:
        if step_mode:
            unit(neg, 0.5 - const, outs, "step")             # [#unsatisfied == 0]
        else:
            unit(neg, 1.0 - const, outs)
        return
    (items, lconst), lo, hi = test
    lw = {}
    for s, cf in items:
        c = chans(s)[0]
        lw[c] = lw.get(c, 0.0) + cf
    lmin, lmax = A.lin_range(test[0])
    M = float(lmax - lmin + 8)               # big enough to switch the unit off when a literal fails
    ww = dict(lw)
    for k, x in w.items():
        ww[k] = ww.get(k, 0.0) - M * x

    def put(t, shift, sg):                   # relu(L - t + shift - M * unsatisfied)
        unit(ww, lconst - t + shift - M * const, {k: sg * x for k, x in outs.items()})

    if step_mode:                            # [L >= t and all literals hold] as one step each
        def ge(t, sg):
            unit(ww, lconst - t + 0.5 - M * const, {k: sg * x for k, x in outs.items()}, "step")
        if lo is not None:
            ge(lo, 1.0)
        else:
            unit(neg, 0.5 - const, outs, "step")
        if hi is not None:
            ge(hi + 1, -1.0)
        return

    if lo is not None and hi is not None and lo == hi:
        put(lo, 1.0, 1.0)
        put(lo, 0.0, -2.0)
        put(lo, -1.0, 1.0)
        return
    if lo is not None:
        put(lo, 1.0, 1.0)
        put(lo, 0.0, -1.0)
    else:
        unit(neg, 1.0 - const, outs)
    if hi is not None:
        put(hi + 1, 1.0, -1.0)
        put(hi + 1, 0.0, 1.0)


def _runs(vals):
    out = []
    for v in vals:
        if out and out[-1][1] == v - 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return [tuple(r) for r in out]


# ============================================================================ recursion front end

REC = "rec_"           # prefix of every name the front end generates


def _primes(limit):
    return [p for p in range(2, limit) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def _set_code(N, K):
    """The narrowest exact superposition code for sets of at most K out of N elements (Kautz-Singleton).

    Element j is the polynomial over GF(q) whose coefficients are j's base-q digits (degree < m, q^m >= N), and its
    channels are (t, p_j(t)) for t in GF(q): q*q channels in all, q per element. Two elements share at most m-1
    channels, so when K*(m-1) < q a set of K elements never covers all the channels of an element it does not hold,
    and "j is in the set" is exactly "all of j's channels are on". Returns (q, m), or None when one channel per
    element is narrower."""
    best = None
    for q in _primes(62):
        m = 1
        while q ** m < N:
            m += 1
        if K * (m - 1) < q and q * q < N and (best is None or q < best[0]):
            best = (q, m)
    return best


def _set_channels(j, q, m):
    digits = [(j // q ** i) % q for i in range(m)]
    return [t * q + sum(c * t ** i for i, c in enumerate(digits)) % q for t in range(q)]


class _NeedSplit(Exception):
    """A statement uses a value computed earlier in the same step: start a new block."""


class _Blk:
    """What the block being written knows: which records were looked up in it, which values it computed."""

    def __init__(self, f):
        self.f, self.fresh, self.dirty = f, set(), set()

    def copy(self):
        b = _Blk(self.f)
        b.fresh, b.dirty = set(self.fresh), set(self.dirty)
        return b


class _Lift(ast.NodeTransformer):
    """Move calls of user functions out of expressions into temporaries, innermost first."""

    def __init__(self, low, f):
        self.low, self.f, self.pre = low, f, []

    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id in self.low.funcs:
            t = self.low._temp(self.f)
            ret = self.low.sig[n.func.id][1]
            self.pre.append(ast.AnnAssign(target=ast.Name(id=t, ctx=ast.Store()), annotation=ast.Name(id=ret),
                                          value=n, simple=1, lineno=getattr(n, "lineno", 0)))
            return ast.Name(id=t, ctx=ast.Load())
        return n


class Lowering:
    """Lower a program with functions, recursion and while loops into a plain step()/output() program.

    Each function body is cut into blocks, and one block runs per step(), chosen by the state variable rec_blk.
    A block ends at a call, a return, a loop, the join after an if that contains calls, or where a statement
    would use a value computed earlier in the same step (the network computes one layer per step). A call pushes
    a frame (the block to come back to, and the caller's live locals) onto a stack held in state variables, sets
    the callee's parameters and jumps to its first block. A return stores the value and jumps to the block saved
    on top of the stack, which restores the locals and pops the frame. `return f(x)` is a tail call: it jumps
    without pushing, so tail recursion runs in constant stack space.
    """

    def __init__(self, src, schema, level=0):
        self.S, self.level = schema, level
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise PipeError(f"syntax error: {e.msg}", e) from None
        for n in ast.walk(tree):
            name = n.id if isinstance(n, ast.Name) else n.arg if isinstance(n, ast.arg) else None
            if name is not None and name.startswith(REC):
                raise PipeError(f"names starting with {REC} are reserved", n)
        self.decls, self.globals, self.funcs, self.sig = [], {}, {}, {}
        self.depth, self.overflow, self.timeout = None, None, "TIMEOUT"
        self.sets, self.nmember = {}, 0
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Subscript) and \
                    isinstance(node.annotation.value, ast.Name) and node.annotation.value.id == "Sparse":
                self._declare_set(node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.startswith(REC):
                    raise PipeError(f"names starting with {REC} are reserved", node)
                self.decls.append(ast.unparse(node))
                self.globals[node.target.id] = self._type(node.annotation, node)
            elif isinstance(node, ast.FunctionDef):
                if node.name in self.funcs:
                    raise PipeError(f"{node.name} is already defined", node)
                self.funcs[node.name] = node
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and \
                    isinstance(node.value.func, ast.Name) and node.value.func.id == "recursion":
                self._directive(node.value)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise PipeError("only state declarations, functions and recursion(...) are allowed at top level", node)
        for name in ("step", "output"):
            if name in self.funcs:
                raise PipeError(f"with functions, main() replaces step() and output(); remove {name}()",
                                self.funcs[name])
        if "main" not in self.funcs:
            raise PipeError("missing main()")
        if self.timeout not in schema.classes:
            raise PipeError(f"the task has no output class {self.timeout}; set recursion(timeout=...)")
        for name, f in self.funcs.items():
            a = f.args
            if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults or f.decorator_list:
                raise PipeError("functions take plain typed parameters", f)
            if f.returns is None:
                raise PipeError(f"{name}() needs a return type, for example -> Byte", f)
            params = []
            for p in a.args:
                if p.annotation is None:
                    raise PipeError(f"parameter {p.arg} of {name}() needs a type", f)
                params.append((p.arg, self._type(p.annotation, p)))
            self.sig[name] = (params, self._type(f.returns, f))
        if self.sig["main"][0]:
            raise PipeError("main() takes no parameters", self.funcs["main"])
        self.vars, self.recs, self.ntemp = {}, {}, {}
        self.body = {}
        for name, f in self.funcs.items():
            self.ntemp[name] = 0
            self.body[name] = self._hoist(self._set_pass(f.body, name), name)
            self._collect(name)
            self._typecheck(name)
            if level >= 2:                                   # one operation per step
                self.body[name] = self._flatten(self.body[name], name)
                self._collect(name)
        self.regs, self.qregs, self.rvs, self.stops = {}, {}, set(), set()
        self.blocks, self.owner, self.entry, self.pushes = [], [], {}, []
        for name in self.funcs:
            self.entry[name] = self._new(name)
        self.fin = self._new("the end")                 # the only block that halts
        self.blocks[self.fin] = ["halt"]
        for name in self.funcs:
            st = _Blk(name)
            self.blocks[self.entry[name]] = self._seq(self.body[name], st, ("end",), set())

    # ---------------------------------------------------------------- declarations and analysis

    def _directive(self, call):
        if call.args:
            raise PipeError("use recursion(depth=..., overflow=CLASS, timeout=CLASS)", call)
        for kw in call.keywords:
            v = kw.value
            if kw.arg == "depth" and isinstance(v, ast.Constant) and type(v.value) is int and v.value >= 1:
                self.depth = v.value
            elif kw.arg in ("overflow", "timeout") and isinstance(v, ast.Name) and v.id in self.S.classes:
                setattr(self, kw.arg, v.id)
            else:
                raise PipeError(f"recursion(): bad {kw.arg}= (depth is a positive integer; overflow and timeout are "
                                f"output classes: {', '.join(self.S.classes)})", call)

    def _type(self, ann, node):
        if not isinstance(ann, ast.Name) or self.S.type_of(ann.id) is None:
            raise PipeError(f"unknown type {ast.unparse(ann) if ann is not None else ''}", node)
        return ann.id

    # ---------------------------------------------------------------- sets: Sparse[N, K, CLASS]

    def _global(self, name, t, init):
        self.decls.append(f"{name}: {t} = {init}")
        self.globals[name] = t

    def _declare_set(self, node):
        """s: Sparse[N, K, CLASS] = set(): at most K elements out of 0..N-1, held in Bool channels. When a code
        narrower than N channels exists, the elements share channels (exact superposition, see _set_code);
        otherwise there is one channel per element. Adding a K+1-th element ends the program with CLASS."""
        name = node.target.id if isinstance(node.target, ast.Name) else None
        sl = node.annotation.slice
        parts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        if name is None or len(parts) != 3 or not all(isinstance(x, ast.Constant) and type(x.value) is int
                                                       for x in parts[:2]) \
                or not (isinstance(parts[2], ast.Name) and parts[2].id in self.S.classes):
            raise PipeError("declare a set as name: Sparse[N, K, CLASS] = set(): elements 0..N-1, at most K of them "
                            f"at once, and CLASS (one of {', '.join(self.S.classes)}) when a K+1-th is added", node)
        N, K, cls = parts[0].value, parts[1].value, parts[2].id
        if not (2 <= N <= 256 and 1 <= K < N):
            raise PipeError("Sparse[N, K, CLASS] needs 2 <= N <= 256 and 1 <= K < N", node)
        v = node.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "set" and not v.args):
            raise PipeError(f"a set starts empty: {name}: Sparse[...] = set()", node)
        code = _set_code(N, K)
        pre = f"{REC}set_{name}"
        sp = dict(N=N, K=K, cls=cls, code=code, count=f"{pre}_n", idx=f"{pre}_x")
        if code:
            q, m = code
            sp["chans"] = [f"{pre}_c{i}" for i in range(q * q)]
            sp["members"] = [_set_channels(j, q, m) for j in range(N)]          # element -> its channels
            sp["ys"] = [f"{pre}_y{t}" for t in range(q)]
        else:
            sp["chans"] = [f"{pre}_c{j}" for j in range(N)]
            sp["members"] = [[j] for j in range(N)]
            sp["ys"] = [f"{pre}_y{g}" for g in range((N + 31) // 32)] if N > 32 else []
        for c in sp["chans"] + sp["ys"]:
            self._global(c, "Bool", "False")
        self._global(sp["count"], f"Int{K + 1}", 0)
        self._global(sp["idx"], f"Int{N}", 0)
        self.sets[name] = sp

    def _gen(self, code, like):
        """Statements parsed from code, carrying the line number of the statement they replace."""
        out = ast.parse(code).body
        for st in out:
            for n in ast.walk(st):
                if "lineno" in n._attributes:
                    n.lineno, n.end_lineno = like.lineno, like.lineno
                    n.col_offset, n.end_col_offset = 0, 0
        return out

    def _int_names(self, f, N):
        """Names in f that are integers of at most N values, usable as an element directly."""
        ok = lambda t: t.startswith("Int") and t[3:].isdigit() and int(t[3:]) <= N
        names = {x for x, t in self.globals.items() if ok(t)} | {x for x, t in self.sig[f][0] if ok(t)}
        for n in ast.walk(ast.Module(body=self.funcs[f].body, type_ignores=[])):
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and isinstance(n.annotation, ast.Name):
                if ok(n.annotation.id):
                    names.add(n.target.id)
                else:
                    names.discard(n.target.id)
        return names

    def _set_index(self, sp, x, f, pre, like):
        """The element as source text: a constant, a small-integer name, or x copied into the set's index (mod N)."""
        if isinstance(x, ast.Constant) and type(x.value) is int and x.value >= 0:
            return str(x.value % sp["N"])
        if isinstance(x, ast.Name) and x.id in self._int_names(f, sp["N"]):
            return x.id
        if sp["N"] & (sp["N"] - 1):
            raise PipeError(f"an element of Sparse[{sp['N']}, ...] must be a constant or an IntM variable with "
                            f"M <= {sp['N']} ({sp['N']} is not a power of two, so a computed value can't be reduced "
                            f"mod {sp['N']})", like)
        pre += self._gen(f"{sp['idx']} = {ast.unparse(x)}", like)
        return sp["idx"]

    def _set_member(self, sp, ix, pre, like):
        """Statements computing "ix in s" into a new Bool, and its name. With a code, membership takes two steps:
        each y_t picks the channel at position t that belongs to ix (q cases), and the member is the AND of the
        y_t. Each pick is a chain "c_a if ix is one of A else c_b if ... else False": its cases test ix alone and
        exclude each other, so the compiler handles it in size linear in the number of cases."""
        self.nmember += 1
        m = f"{REC}m{self.nmember}"
        self._global(m, "Bool", "False")

        def pick(cases):                       # cases: (elements, channel)
            cases = [(js, c) for js, c in cases if js]
            return "".join(f"{c} if ({' or '.join(f'{ix} == {j}' for j in js)}) else " for js, c in cases) + "False"

        if sp["code"]:
            q, _ = sp["code"]
            for t, y in enumerate(sp["ys"]):
                cases = [([j for j in range(sp["N"]) if sp["members"][j][t] == t * q + v], sp["chans"][t * q + v])
                         for v in range(q)]
                pre += self._gen(f"{y} = {pick(cases)}", like)
            pre += self._gen(f"{m} = {' and '.join(sp['ys'])}", like)
        elif sp["ys"]:
            for g, y in enumerate(sp["ys"]):
                js = range(32 * g, min(32 * g + 32, sp["N"]))
                pre += self._gen(f"{y} = {pick([([j], sp['chans'][j]) for j in js])}", like)
            pre += self._gen(f"{m} = {' or '.join(sp['ys'])}", like)
        else:
            pre += self._gen(f"{m} = {pick([([j], c) for j, c in enumerate(sp['chans'])])}", like)
        return m

    def _set_hoist(self, e, f, pre, like):
        """Replace each "x in s" / "x not in s" in expression e by a Bool computed by statements added to pre."""
        low = self

        class T(ast.NodeTransformer):
            def visit_Compare(self, n):
                if len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and \
                        isinstance(n.comparators[0], ast.Name) and n.comparators[0].id in low.sets:
                    n.left = self.visit(n.left)
                    sp = low.sets[n.comparators[0].id]
                    m = low._set_member(sp, low._set_index(sp, n.left, f, pre, like), pre, like)
                    out = ast.Name(id=m, ctx=ast.Load())
                    return ast.UnaryOp(op=ast.Not(), operand=out) if isinstance(n.ops[0], ast.NotIn) else out
                if any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
                    raise PipeError("`in` tests membership of a set declared as Sparse[N, K, CLASS]", like)
                return self.generic_visit(n)

            def visit_Name(self, n):
                if n.id in low.sets:
                    raise PipeError(f"{n.id} is a set: use x in {n.id}, {n.id}.add(x) or {n.id}.clear()", like)
                return n

        return T().visit(e)

    def _set_method(self, s, f):
        call = s.value
        name, meth = call.func.value.id, call.func.attr
        sp = self.sets[name]
        if meth == "clear" and not call.args and not call.keywords:
            return self._gen("\n".join([f"{c} = False" for c in sp["chans"]] + [f"{sp['count']} = 0"]), s)
        if meth != "add" or len(call.args) != 1 or call.keywords:
            raise PipeError(f"sets support {name}.add(x), {name}.clear() and x in {name}", s)
        pre = []
        ix = self._set_index(sp, call.args[0], f, pre, s)
        m = self._set_member(sp, ix, pre, s)
        sets = []
        for c, chan in enumerate(sp["chans"]):
            js = [j for j in range(sp["N"]) if c in sp["members"][j]]
            if js:
                sets.append(f"    {chan} = {chan} or " + " or ".join(f"{ix} == {j}" for j in js))
        code = (f"if not {m}:\n    if {sp['count']} == {sp['K']}:\n        stop({sp['cls']})\n"
                f"    {sp['count']} = {sp['count']} + 1\n" + "\n".join(sets))
        return pre + self._gen(code, s)

    def _set_pass(self, stmts, f):
        """Rewrite set operations into plain statements on the sets' channels."""
        out = []
        for s in stmts:
            pre = []
            if isinstance(s, ast.While):
                if any(isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops)
                       for n in ast.walk(s.test)):
                    raise PipeError("a while condition cannot test a set; compute it into a Bool before the loop "
                                    "and again at the end of the body", s)
                s.test = self._set_hoist(s.test, f, pre, s)
                s.body = self._set_pass(s.body, f)
                out.append(s)
            elif isinstance(s, ast.If):
                node = s
                while True:                       # the whole elif chain: its tests are computed before it
                    node.test = self._set_hoist(node.test, f, pre, node)
                    node.body = self._set_pass(node.body, f)
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        node = node.orelse[0]
                    else:
                        node.orelse = self._set_pass(node.orelse, f)
                        break
                out += pre + [s]
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and \
                    isinstance(s.value.func, ast.Attribute) and isinstance(s.value.func.value, ast.Name) and \
                    s.value.func.value.id in self.sets:
                out += self._set_method(s, f)
            else:
                if getattr(s, "value", None) is not None:
                    s.value = self._set_hoist(s.value, f, pre, s)
                out += pre + [s]
        return out

    def _temp(self, f):
        self.ntemp[f] += 1
        return f"{REC}t{self.ntemp[f]}"

    def _is_user_call(self, n):
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.funcs

    def _hoist(self, stmts, f):
        out = []
        for s in stmts:
            if isinstance(s, ast.AugAssign):
                if not isinstance(s.target, ast.Name):
                    raise PipeError("assign to a single name", s)
                s = ast.Assign(targets=[ast.Name(id=s.target.id, ctx=ast.Store())],
                               value=ast.BinOp(left=ast.Name(id=s.target.id, ctx=ast.Load()), op=s.op, right=s.value),
                               lineno=s.lineno)
            lift = _Lift(self, f)
            if isinstance(s, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)) and s.value is not None:
                if self._is_user_call(s.value):
                    s.value.args = [lift.visit(a) for a in s.value.args]
                else:
                    s.value = lift.visit(s.value)
                out += lift.pre + [s]
            elif isinstance(s, ast.If):
                s.test = lift.visit(s.test)
                s.body, s.orelse = self._hoist(s.body, f), self._hoist(s.orelse, f)
                out += lift.pre + [s]
            elif isinstance(s, ast.While):
                if any(self._is_user_call(n) for n in ast.walk(s.test)):
                    raise PipeError("a while condition cannot call a function; compute it into a variable", s)
                if s.orelse:
                    raise PipeError("while ... else is not supported", s)
                s.body = self._hoist(s.body, f)
                out.append(s)
            else:
                out.append(s)
        return out

    def _collect(self, f):
        vs = dict(self.sig[f][0])
        recs = {}
        stmts = [n for n in ast.walk(ast.Module(body=self.body[f], type_ignores=[])) if isinstance(n, ast.stmt)]
        for s in stmts:
            if isinstance(s, ast.AnnAssign):
                if not isinstance(s.target, ast.Name):
                    raise PipeError("declare a single name", s)
                if s.value is None:
                    raise PipeError(f"give {s.target.id} a value where it is declared", s)
                t = self._type(s.annotation, s)
                if vs.get(s.target.id, t) != t:
                    raise PipeError(f"{s.target.id} is declared with two types", s)
                vs[s.target.id] = t
            elif isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                x, v = s.targets[0].id, s.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "attend":
                    a = v.args[0] if v.args else None
                    if not (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
                            and a.value.id in self.S.records):
                        raise PipeError("use attend(table.key_field, value)", s)
                    recs[x] = a.value.id
                elif self._is_user_call(v):
                    t = self.sig[v.func.id][1]
                    if vs.get(x, t) != t:
                        raise PipeError(f"{x} is {vs[x]} but {v.func.id}() returns {t}", s)
                    vs[x] = t
        for s in stmts:
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    raise PipeError("assign to a single name", s)
                x = s.targets[0].id
                if x not in vs and x not in recs and x not in self.globals:
                    raise PipeError(f"declare {x} with a type the first time, for example {x}: Byte = ...", s)
        self.vars[f], self.recs[f] = vs, recs

    # ---------------------------------------------------------------- level 2: one operation per step

    def _atomic(self, e, f):
        if isinstance(e, (ast.Constant, ast.Name)):
            return True
        return isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and \
            (e.value.id in self.recs[f] or e.value.id == "query")

    def _linear(self, e, f):
        """An atom, or a sum or difference of atoms: one threshold-friendly operation, kept exact."""
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
            return self._linear(e.left, f) and self._linear(e.right, f)
        return self._atomic(e, f)

    def _lift(self, e):
        """Move conditional expressions up: op(a if c else b, x) -> op(a, x) if c else op(b, x). Exact values
        inside an expression then never have to be stored before they are used."""
        if isinstance(e, ast.IfExp):
            return ast.IfExp(test=self._lift(e.test), body=self._lift(e.body), orelse=self._lift(e.orelse))
        if isinstance(e, ast.BinOp):
            l, r = self._lift(e.left), self._lift(e.right)
            if isinstance(l, ast.IfExp):
                return self._lift(ast.IfExp(test=l.test, body=ast.BinOp(left=l.body, op=e.op, right=r),
                                            orelse=ast.BinOp(left=l.orelse, op=e.op, right=r)))
            if isinstance(r, ast.IfExp):
                return self._lift(ast.IfExp(test=r.test, body=ast.BinOp(left=l, op=e.op, right=r.body),
                                            orelse=ast.BinOp(left=l, op=e.op, right=r.orelse)))
            return ast.BinOp(left=l, op=e.op, right=r)
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            l, r = self._lift(e.left), self._lift(e.comparators[0])
            if isinstance(l, ast.IfExp):
                return self._lift(ast.IfExp(test=l.test, body=ast.Compare(left=l.body, ops=e.ops, comparators=[r]),
                                            orelse=ast.Compare(left=l.orelse, ops=e.ops, comparators=[r])))
            if isinstance(r, ast.IfExp):
                return self._lift(ast.IfExp(test=r.test, body=ast.Compare(left=l, ops=e.ops, comparators=[r.body]),
                                            orelse=ast.Compare(left=l, ops=e.ops, comparators=[r.orelse])))
            return ast.Compare(left=l, ops=e.ops, comparators=[r])
        return e

    def _infer(self, e, f):
        """The type a temporary holding e gets."""
        if isinstance(e, ast.Constant):
            return "Bool" if isinstance(e.value, bool) else "Byte"
        if isinstance(e, ast.Name):
            t = self.vars[f].get(e.id) or self.globals.get(e.id)
            if t is None and e.id in self.S.consts:
                t = self.S.consts[e.id][0][1]
            return t or "Byte"
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            if e.value.id in self.recs[f]:
                return Schema.tname(self.S.records[self.recs[f][e.value.id]]["fields"][e.attr][0])
            if e.value.id == "query":
                return Schema.tname(self.S.query[e.attr][0])
        if isinstance(e, (ast.Compare, ast.BoolOp)) or (isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not)):
            return "Bool"
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub, ast.Mod)):
            for side in (e.left, e.right):
                if not isinstance(side, ast.Constant):
                    t = self._infer(side, f)
                    if t.startswith("Int"):
                        return t                              # one-hot arithmetic stays one-hot
            return "Byte"
        if isinstance(e, ast.IfExp):
            return self._infer(e.orelse if isinstance(e.body, ast.Constant) else e.body, f)
        return "Byte"

    def _store_temp(self, e, t, f, pre):
        name = self._temp(f)
        pre.append(ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()), annotation=ast.Name(id=t), value=e,
                                 simple=1, lineno=0))
        return ast.Name(id=name, ctx=ast.Load())

    def _atom(self, e, f, pre):
        if self._atomic(e, f):
            return e
        return self._store_temp(self._one(e, f, pre), self._infer(e, f), f, pre)

    def _piece(self, e, f, pre):
        """An operand that may stay a linear expression (inside a sum or a comparison)."""
        return e if self._linear(e, f) else self._atom(e, f, pre)

    def _one(self, e, f, pre):
        """e as a single operation on atoms (a whole sum or difference counts as one), with temporaries in pre."""
        if self._linear(e, f):
            return e
        if isinstance(e, ast.IfExp):
            return ast.IfExp(test=self._cond(e.test, f, pre), body=self._piece(e.body, f, pre),
                             orelse=self._piece(e.orelse, f, pre))
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            return ast.Compare(left=self._piece(e.left, f, pre), ops=e.ops,
                               comparators=[self._piece(e.comparators[0], f, pre)])
        if isinstance(e, ast.BinOp):
            if isinstance(e.op, (ast.Add, ast.Sub)):
                return ast.BinOp(left=self._piece(e.left, f, pre), op=e.op, right=self._piece(e.right, f, pre))
            right = e.right if isinstance(e.right, ast.Constant) else self._atom(e.right, f, pre)
            return ast.BinOp(left=self._atom(e.left, f, pre), op=e.op, right=right)
        if isinstance(e, ast.BoolOp):
            return ast.BoolOp(op=e.op, values=[self._atom(v, f, pre) for v in e.values])
        if isinstance(e, ast.UnaryOp):
            return ast.UnaryOp(op=e.op, operand=self._atom(e.operand, f, pre))
        return e

    def _cond(self, test, f, pre):
        """A branch condition: kept when it needs no arithmetic test, else computed into a Bool first."""
        test = self._lift(test)
        if isinstance(test, ast.Name) or not self._arith(test, f):
            return test
        return self._store_temp(self._one(test, f, pre), "Bool", f, pre)

    def _flatten(self, stmts, f):
        out = []
        for s in stmts:
            pre = []
            if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:
                v = s.value
                if self._is_user_call(v):
                    v.args = [self._one(self._lift(a), f, pre) for a in v.args]
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "attend":
                    v.args = [v.args[0], self._one(self._lift(v.args[1]), f, pre)]
                else:
                    s.value = self._one(self._lift(v), f, pre)
                out += pre + [s]
            elif isinstance(s, ast.Return) and s.value is not None:
                if self._is_user_call(s.value):
                    s.value.args = [self._one(self._lift(a), f, pre) for a in s.value.args]
                else:
                    s.value = self._one(self._lift(s.value), f, pre)
                out += pre + [s]
            elif isinstance(s, ast.Expr) and self._is_user_call(s.value):
                s.value.args = [self._one(self._lift(a), f, pre) for a in s.value.args]
                out += pre + [s]
            elif isinstance(s, ast.If):
                s.test = self._cond(s.test, f, pre)
                s.body, s.orelse = self._flatten(s.body, f), self._flatten(s.orelse, f)
                out += pre + [s]
            elif isinstance(s, ast.While):
                test = self._cond(s.test, f, pre)
                again = copy.deepcopy(pre)                   # recompute the condition at the end of each pass
                out += pre + [ast.While(test=test, body=self._flatten(s.body, f) + again, orelse=[],
                                        lineno=getattr(s, "lineno", 0))]
            else:
                out.append(s)
        return out

    # ---------------------------------------------------------------- names

    def _reg(self, f, key):
        if isinstance(key, tuple):
            rec, fld = key
            name = f"{REC}{f}_{rec}_{fld}"
            t = self.S.records[self.recs[f][rec]]["fields"][fld][0]
            self.regs[name] = Schema.tname(t)
        else:
            name = f"{REC}{f}_{key}"
            self.regs[name] = self.vars[f][key]
        return name

    def _live(self, stmts, out, f):
        """Keys (locals and record fields of f) live before stmts, given those live after them."""
        live = set(out)
        for s in reversed(stmts):
            live = self._live1(s, live, f)
        return live

    def _live1(self, s, live, f):
        keys = lambda *nodes: self._keys(nodes, f)
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            x = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id
            if x in self.recs[f]:
                live = {k for k in live if not (isinstance(k, tuple) and k[0] == x)}
            else:
                live = live - {x}
            v = s.value
            return live | (keys(*v.args) if self._is_user_call(v) else keys(v))
        if isinstance(s, ast.Expr):
            v = s.value
            if self._is_user_call(v):
                return live | keys(*v.args)
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "stop":
                return set()
            return live
        if isinstance(s, ast.Return):
            if s.value is None:
                return set()
            return keys(*s.value.args) if self._is_user_call(s.value) else keys(s.value)
        if isinstance(s, ast.If):
            return keys(s.test) | self._live(s.body, live, f) | self._live(s.orelse, live, f)
        if isinstance(s, ast.While):
            L = live | keys(s.test)
            while True:
                L2 = live | keys(s.test) | self._live(s.body, L, f)
                if L2 == L:
                    return L
                L = L2
        return live

    def _keys(self, nodes, f):
        """Locals and record fields of f read in the given statements or expressions."""
        out = set()
        for root in nodes:
            for n in ast.walk(root):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in self.recs[f]:
                    out.add((n.value.id, n.attr))
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in self.vars[f]:
                    out.add(n.id)
        return out

    def _expr(self, e, st):
        """Rename an expression for the lowered program."""
        f, low = st.f, self

        class R(ast.NodeTransformer):
            def visit_Attribute(self, n):
                if isinstance(n.value, ast.Name) and n.value.id in low.recs[f]:
                    if n.value.id in st.fresh:
                        return n
                    return ast.Name(id=low._reg(f, (n.value.id, n.attr)), ctx=ast.Load())
                if isinstance(n.value, ast.Name) and n.value.id == "query":
                    if n.attr not in low.S.query:
                        raise PipeError(f"query has fields {', '.join(low.S.query)}", n)
                    name = f"{REC}q_{n.attr}"
                    low.qregs[name] = (Schema.tname(low.S.query[n.attr][0]), n.attr)
                    return ast.Name(id=name, ctx=ast.Load())
                self.generic_visit(n)
                return n

            def visit_Name(self, n):
                if n.id in low.recs[f]:
                    raise PipeError(f"{n.id} is a record; read a field such as {n.id}.value", n)
                if n.id in low.vars[f]:
                    return ast.Name(id=low._reg(f, n.id), ctx=n.ctx)
                return n

            def visit_Call(self, n):
                if low._is_user_call(n):
                    raise PipeError("internal: call left in an expression", n)
                if isinstance(n.func, ast.Name) and n.func.id == "attend":
                    n.args = [n.args[0]] + [self.visit(a) for a in n.args[1:]]
                    return n
                self.generic_visit(n)
                return n
        return ast.unparse(R().visit(copy.deepcopy(e)))

    def _target(self, st, x):
        if x in self.vars[st.f]:
            return self._reg(st.f, x)
        if x in self.globals:
            return x
        raise PipeError(f"unknown name {x}")

    def _byteish(self, e, f):
        """Is e a Byte value (binary), as opposed to a one-hot, Bool or enum value or a constant?"""
        if isinstance(e, ast.Name):
            return (self.vars[f].get(e.id) or self.globals.get(e.id)) == "Byte"
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            if e.value.id in self.recs[f]:
                fld = self.S.records[self.recs[f][e.value.id]]["fields"].get(e.attr)
                return fld is not None and fld[0][0] == "byte"
            if e.value.id == "query":
                q = self.S.query.get(e.attr)
                return q is not None and q[0][0] == "byte"
            return False
        if isinstance(e, ast.BinOp):
            return self._byteish(e.left, f) or self._byteish(e.right, f)
        if isinstance(e, ast.IfExp):
            return self._byteish(e.body, f) or self._byteish(e.orelse, f)
        return False

    @staticmethod
    def _bit_test(cmp):
        """(x & 2^k) == 0 or == 2^k (or !=): one bit, which needs no arithmetic test."""
        if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], (ast.Eq, ast.NotEq)):
            return False
        for a, b in ((cmp.left, cmp.comparators[0]), (cmp.comparators[0], cmp.left)):
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.BitAnd) and isinstance(b, ast.Constant):
                m = a.right if isinstance(a.right, ast.Constant) else a.left if isinstance(a.left, ast.Constant) else None
                if m is not None and type(m.value) is int and m.value > 0 and not m.value & (m.value - 1) \
                        and b.value in (0, m.value):
                    return True
        return False

    def _arith(self, node, f):
        """Does node compute with arithmetic tests (Byte sums, differences or comparisons)?"""
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and self._byteish(n, f):
                return True
            if isinstance(n, ast.Compare) and not self._bit_test(n) and \
                    any(self._byteish(x, f) for x in [n.left] + n.comparators):
                return True
        return False

    def _test_first(self, test, branches, f):
        """A Byte comparison guarding Byte arithmetic needs two tests in one layer: compute the test a step earlier."""
        return self._arith(test, f) and (self.level >= 1 or any(self._arith(s, f) for s in branches))

    def _bool_local(self, f):
        self.ntemp[f] += 1
        name = f"{REC}c{self.ntemp[f]}"
        self.vars[f][name] = "Bool"
        return name

    def _compatible(self, want, e, f, node, what):
        got = self._infer(e, f)
        if isinstance(e, ast.Constant) and type(e.value) is int and (want == "Byte" or want.startswith("Int")):
            return
        ints = lambda t: t == "Byte" or t.startswith("Int")
        if want == got or (ints(want) and ints(got)):
            return
        raise PipeError(f"{what} is {want}, but the value is {got}", node)

    def _typecheck(self, f):
        """Static types: assignments, returns and call arguments must match their declared types."""
        for s in (n for n in ast.walk(ast.Module(body=self.body[f], type_ignores=[])) if isinstance(n, ast.stmt)):
            if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:
                x = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id
                v = s.value
                if x in self.recs[f] or (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                                         and v.func.id == "attend"):
                    continue
                want = self.vars[f].get(x) or self.globals.get(x)
                if self._is_user_call(v):
                    got = self.sig[v.func.id][1]
                    if want and want != got and not ({want, got} <= {"Byte"} | {t for t in (want, got) if t.startswith("Int")}):
                        raise PipeError(f"{x} is {want}, but {v.func.id}() returns {got}", s)
                elif want:
                    self._compatible(want, v, f, s, x)
            elif isinstance(s, ast.Return) and s.value is not None and not self._is_user_call(s.value):
                self._compatible(self.sig[f][1], s.value, f, s, f"the result of {f}()")
            for call in (n for n in ast.walk(s) if self._is_user_call(n)) if isinstance(s, (ast.Assign, ast.AnnAssign,
                                                                                       ast.Return, ast.Expr)) else ():
                for (p, t), a in zip(self.sig[call.func.id][0], call.args):
                    self._compatible(t, a, f, s, f"parameter {p} of {call.func.id}()")

    def _check_reads(self, nodes, st):
        names = {n.id for root in nodes for n in ast.walk(root) if isinstance(n, ast.Name)}
        mapped = {self._reg(st.f, x) if x in self.vars[st.f] else x for x in names if x not in self.recs[st.f]}
        if mapped & st.dirty:
            raise _NeedSplit()

    # ---------------------------------------------------------------- blocks

    def _new(self, f):
        self.blocks.append(None)
        self.owner.append(f)
        return len(self.blocks) - 1

    def _goto(self, J, st, live):
        lines = []
        for k in sorted(live, key=str):
            if isinstance(k, tuple) and k[0] in st.fresh:
                lines.append(f"{self._reg(st.f, k)} = {k[0]}.{k[1]}")
        return lines + [f"{REC}blk = {J}"]

    def _ret_dispatch(self):
        return ["@@RET@@"]

    def _args(self, st, callee, args, node):
        params, _ = self.sig[callee]
        if len(args) != len(params):
            raise PipeError(f"{callee}() takes {len(params)} arguments", node)
        lines = [f"{REC}a{i} = {self._expr(a, st)}" for i, a in enumerate(args)]
        lines += [f"{self._reg(callee, p)} = {REC}a{i}" for i, (p, _) in enumerate(params)]
        return lines + [f"{REC}blk = {self.entry[callee]}"]

    @staticmethod
    def _transfers(stmts):
        return any(isinstance(n, (ast.Return, ast.While)) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                                              and n.func.id == "stop")
                   for s in stmts for n in ast.walk(s))

    def _always(self, stmts):
        """Does every path through stmts end in a return or stop()?"""
        if not stmts:
            return False
        s = stmts[-1]
        if isinstance(s, ast.Return):
            return True
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) \
                and s.value.func.id == "stop":
            return True
        if isinstance(s, ast.If):
            return self._always(s.body) and self._always(s.orelse)
        return False

    def _has_call(self, stmts):
        return any(self._is_user_call(n) for s in stmts for n in ast.walk(s))

    def _seq(self, stmts, st, cont, live_after):
        lines = []
        f = st.f
        for i, s in enumerate(stmts):
            rest = stmts[i + 1:]
            live_rest = self._live(rest, live_after, f)
            try:
                if isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
                    continue
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None and self._is_user_call(s.value):
                    self._check_reads(s.value.args, st)
                    x = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id
                    return lines + self._call(st, s.value, x, rest, cont, live_after)
                if isinstance(s, ast.Expr) and self._is_user_call(s.value):
                    self._check_reads(s.value.args, st)
                    return lines + self._call(st, s.value, None, rest, cont, live_after)
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and self.level >= 1 and isinstance(s.value, ast.Call) \
                        and isinstance(s.value.func, ast.Name) and s.value.func.id == "attend":
                    kind = s.value.args[0].value.id
                    r = self.S.records[kind]
                    kreg = f"{REC}key_{kind}"                  # every lookup into a table uses one key register
                    key = s.value.args[1]
                    if not (isinstance(key, ast.Name) and key.id == kreg):
                        self._check_reads([key], st)
                        self.regs[kreg] = Schema.tname(r["fields"][r["key"]][0])
                        lines.append(f"{kreg} = {self._expr(key, st)}")
                        s2 = copy.copy(s)
                        s2.value = ast.Call(func=s.value.func, args=[s.value.args[0], ast.Name(id=kreg, ctx=ast.Load())],
                                            keywords=[])
                        J = self._new(f)
                        self.blocks[J] = self._seq([s2] + rest, _Blk(f), cont, live_after)
                        return lines + self._goto(J, st, self._live([s2] + rest, live_after, f))
                if isinstance(s, (ast.Assign, ast.AnnAssign)):
                    lines += self._assign(s, st)
                    continue
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) \
                        and s.value.func.id == "stop":
                    c = s.value.args[0] if len(s.value.args) == 1 else None
                    if not (isinstance(c, ast.Name) and c.id in self.S.classes):
                        raise PipeError(f"use stop(CLASS) with one of {', '.join(self.S.classes)}", s)
                    self.stops.add(c.id)
                    return lines + [f"{REC}stop_{c.id} = True", f"{REC}blk = {self.fin}"]
                if isinstance(s, ast.Return):
                    if s.value is None:
                        raise PipeError("return needs a value", s)
                    if self._is_user_call(s.value):                       # tail call
                        self._check_reads(s.value.args, st)
                        return lines + self._args(st, s.value.func.id, s.value.args, s)
                    self._check_reads([s.value], st)
                    t = self.sig[f][1]
                    self.rvs.add(t)
                    return lines + [f"{REC}rv_{t} = {self._expr(s.value, st)}"] + self._ret_dispatch()
                merged = rest if isinstance(s, ast.If) and (self._always(s.body) or self._always(s.orelse)) else []
                if isinstance(s, ast.If) and self._test_first(s.test, s.body + s.orelse + merged, f):
                    self._check_reads([s.test], st)
                    c = self._bool_local(f)
                    rest_if = [ast.If(test=ast.Name(id=c, ctx=ast.Load()), body=s.body, orelse=s.orelse,
                                      lineno=getattr(s, "lineno", 0))] + rest
                    J = self._new(f)
                    lines.append(f"{self._reg(f, c)} = {self._expr(s.test, st)}")
                    self.blocks[J] = self._seq(rest_if, _Blk(f), cont, live_after)
                    return lines + self._goto(J, st, self._live(rest_if, live_after, f))
                if isinstance(s, ast.If):
                    if not self._transfers([s]) and not self._has_call([s]):
                        try:
                            trial = st.copy()
                            body = self._inline(s.body, trial)
                            other = self._inline(s.orelse, trial)
                            self._check_reads([s.test], st)
                            lines += [f"if {self._expr(s.test, st)}:"] + _ind(body or ["pass"])
                            if other:
                                lines += ["else:"] + _ind(other)
                            st.dirty, st.fresh = trial.dirty, trial.fresh
                            continue
                        except _NeedSplit:
                            pass
                    self._check_reads([s.test], st)
                    test = self._expr(s.test, st)
                    tb, eb = self._always(s.body), self._always(s.orelse)
                    if rest and not (tb and eb):
                        if tb:
                            then_s, else_s, bc, bl = s.body, s.orelse + rest, cont, live_after
                        elif eb:
                            then_s, else_s, bc, bl = s.body + rest, s.orelse, cont, live_after
                        else:
                            J = self._new(f)
                            self.blocks[J] = self._seq(rest, _Blk(f), cont, live_after)
                            then_s, else_s, bc, bl = s.body, s.orelse, ("goto", J), live_rest
                    else:
                        then_s, else_s, bc, bl = s.body, s.orelse, cont, live_after
                    return lines + [f"if {test}:"] + _ind(self._seq(then_s, st.copy(), bc, bl)) + \
                        ["else:"] + _ind(self._seq(else_s, st.copy(), bc, bl))
                if isinstance(s, ast.While):
                    H = self._new(f)
                    loop_live = self._live([s], live_rest, f)
                    test = self._expr(s.test, _Blk(f))
                    top = H
                    if self._test_first(s.test, s.body, f):           # compute the test one step earlier
                        c = self._bool_local(f)
                        top = self._new(f)
                        self.blocks[H] = [f"{self._reg(f, c)} = {test}", f"{REC}blk = {top}"]
                        test = self._reg(f, c)
                    self.blocks[top] = [f"if {test}:"] + \
                        _ind(self._seq(s.body, _Blk(f), ("goto", H), loop_live)) + \
                        ["else:"] + _ind(self._seq(rest, _Blk(f), cont, live_after))
                    return lines + self._goto(H, st, loop_live)
                raise PipeError(f"unsupported statement in a function: {type(s).__name__}", s)
            except _NeedSplit:
                J = self._new(f)
                self.blocks[J] = self._seq(stmts[i:], _Blk(f), cont, live_after)
                return lines + self._goto(J, st, self._live(stmts[i:], live_after, f))
        if cont[0] == "goto":
            return lines + self._goto(cont[1], st, live_after)
        raise PipeError(f"{f}() can reach its end without returning", self.funcs[f])

    def _assign(self, s, st):
        x = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id
        v = s.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "attend":
            self._check_reads(v.args[1:], st)
            if self.level >= 1 and st.fresh:                 # one lookup per block
                raise _NeedSplit()
            st.fresh.add(x)
            return [f"{x} = {self._expr(v, st)}"]
        self._check_reads([v], st) if not isinstance(v, ast.Name) else None
        tgt = self._target(st, x)
        src = self._expr(v, st)
        if isinstance(v, ast.Name):
            mapped = self._reg(st.f, v.id) if v.id in self.vars[st.f] else v.id
            if mapped in st.dirty:
                st.dirty.add(tgt)
            else:
                st.dirty.discard(tgt)
        elif isinstance(v, (ast.Constant, ast.Attribute)):
            st.dirty.discard(tgt)
        else:
            st.dirty.add(tgt)
        return [f"{tgt} = {src}"]

    def _inline(self, stmts, st):
        lines = []
        for s in stmts:
            if isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
                continue
            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                lines += self._assign(s, st)
            elif isinstance(s, ast.If):
                self._check_reads([s.test], st)
                body, other = self._inline(s.body, st), self._inline(s.orelse, st)
                lines += [f"if {self._expr(s.test, st)}:"] + _ind(body or ["pass"])
                if other:
                    lines += ["else:"] + _ind(other)
            else:
                raise _NeedSplit()
        return lines

    def _call(self, st, call, x, rest, cont, live_after):
        f, callee = st.f, call.func.id
        live = self._live(rest, live_after, f) - ({x} if x else set())
        saved = sorted(live, key=str)
        if self.depth is None:
            raise PipeError("a call that is not a tail call needs a stack: add recursion(depth=..., overflow=CLASS)",
                            call)
        K = self._new(f)
        # A frame is a pool of Bool channels c0, c1, ... shared by all call sites. Each site packs its saved values
        # into the pool from c0: a Bool in one channel, a Byte in 8 (one per bit), an enum or IntN in one channel
        # per value (one-hot). A frame is then as wide as the call site that saves the most channels. The return
        # block in the frame says which site pushed it, so a restore always unpacks its own layout.
        vals, restore, k = [], [], 0
        for key in saved:
            reg = self._reg(f, key)
            val = f"{key[0]}.{key[1]}" if isinstance(key, tuple) and key[0] in st.fresh else reg
            t = self.regs[reg]
            ch = [f"c{k + i}" for i in range(self._width(t))]
            k += len(ch)
            src = [f"{REC}s0_{c}" for c in ch]
            if t == "Bool":
                vals.append((ch[0], val))
                restore.append(f"{reg} = {src[0]}")
            elif t == "Byte":
                vals += [(c, f"{REC}bit({val}, {i})") for i, c in enumerate(ch)]
                restore.append(f"{reg} = {REC}bits({', '.join(src)})")
            else:
                names = list(self.S.enums[t]) if t in self.S.enums else [str(i) for i in range(len(ch))]
                vals += [(c, f"{val} == {nm}") for c, nm in zip(ch, names)]
                restore.append(f"{reg} = " + "".join(f"{nm} if {c} else " for c, nm in zip(src[:-1], names[:-1]))
                               + names[-1])
        self.pushes.append(vals)
        lines = [f"@@PUSH {len(self.pushes) - 1} {K}@@"] + _ind(self._args(st, callee, call.args, call))
        if x is not None:
            t = self.sig[callee][1]
            self.rvs.add(t)
            restore.append(f"{self._target(st, x)} = {REC}rv_{t}")
        self.blocks[K] = restore + ["@@POP@@"] + self._seq(rest, _Blk(f), cont, live_after)
        return lines

    def _frame(self):
        """The frame's slots and their types, and its width in channels."""
        n = max((len(v) for v in self.pushes), default=0)
        types = {f"c{i}": "Bool" for i in range(n)} | {"ret": f"Int{max(len(self.blocks), 2)}", "v": "Bool"}
        return types, sum(self._width(t) for t in types.values())

    def _width(self, t):
        if t == "Bool":
            return 1
        if t == "Byte":
            return 8
        if t.startswith("Int") and t[3:].isdigit():
            return int(t[3:])
        return len(self.S.enums[t])

    # ---------------------------------------------------------------- output

    def _default(self, t):
        if t == "Bool":
            return "False"
        if t in self.S.enums:
            return self.S.enums[t][0]
        return "0"

    def source(self):
        nb = len(self.blocks)
        if nb > 256:
            raise PipeError(f"the program needs {nb} blocks; the limit is 256")
        bt = f"Int{max(nb, 2)}"
        ftype, _ = self._frame()
        K = self.depth or 0
        allf = list(ftype)
        out = list(self.decls)
        out.append(f"{REC}blk: {bt} = {self.entry['main']}")
        for t in sorted(self.rvs):
            out.append(f"{REC}rv_{t}: {t} = {self._default(t)}")
        for c in sorted(self.stops | ({self.overflow} if self.pushes and self.overflow else set())):
            out.append(f"{REC}stop_{c}: Bool = False")
        for r, t in sorted(self.regs.items()):
            out.append(f"{r}: {t} = {self._default(t)}")
        for r, (t, q) in sorted(self.qregs.items()):
            out.append(f"{r}: {t} = query.{q}")
        if self.pushes:
            if self.overflow is None:
                raise PipeError("recursion(...) needs overflow=CLASS: the output when the stack is full")
            for i in range(K):
                for fld in allf:
                    out.append(f"{REC}s{i}_{fld}: {ftype[fld]} = {self._default(ftype[fld])}")

        def expand(line):
            ind = line[:len(line) - len(line.lstrip())]
            body = line.strip()
            if body == "@@RET@@":
                res = [f"if {REC}s0_v:", f"    {REC}blk = {REC}s0_ret", "else:", f"    {REC}blk = {self.fin}"] \
                    if self.pushes else [f"{REC}blk = {self.fin}"]
                return [ind + r for r in res]
            if body == "@@POP@@":
                res = [f"{REC}s{i}_{fld} = {REC}s{i + 1}_{fld}" for i in range(K - 1) for fld in allf]
                return [ind + r for r in res + [f"{REC}s{K - 1}_v = False"]]
            if body.startswith("@@PUSH"):
                _, idx, blk = body.strip("@").split()
                res = [f"{REC}s{i}_{fld} = {REC}s{i - 1}_{fld}" for i in range(K - 1, 0, -1) for fld in allf]
                res += [f"{REC}s0_ret = {blk}", f"{REC}s0_v = True"]
                res += [f"{REC}s0_{fld} = {val}" for fld, val in self.pushes[int(idx)]]
                head = [f"if {REC}s{K - 1}_v:", f"    {REC}stop_{self.overflow} = True", f"    {REC}blk = {self.fin}",
                        "else:"]
                return [ind + r for r in head] + [ind + "    " + r for r in res]   # the call's own lines follow
            return [line]

        out += ["", "def step():"]
        for b, code in enumerate(self.blocks):
            out.append(f"    {'if' if b == 0 else 'elif'} {REC}blk == {b}:    # {self.owner[b]}()")
            out += ["        " + x for line in code for x in expand(line)]
        out += ["", "def output():", "    if halted:"]
        for c in sorted(self.stops | ({self.overflow} if self.pushes else set())):
            out += [f"        if {REC}stop_{c}:", f"            return {c}"]
        out += [f"        return {REC}rv_{self.sig['main'][1]}", f"    return {self.timeout}", ""]
        return "\n".join(out)

    def report(self):
        return (f"recursion: {len(self.funcs)} functions in {len(self.blocks)} blocks (one per step), "
                + (f"stack of {self.depth} frames of {self._frame()[1]} channels, {len(self.pushes)} call sites that push"
                   if self.pushes else "no stack")
                + "".join(f"\nset {n}: {sp['N']} elements, at most {sp['K']} at once, in {len(sp['chans'])} channels"
                          + (f" (superposed, Kautz-Singleton code over GF({sp['code'][0]}))" if sp["code"] else "")
                          for n, sp in self.sets.items()))


def _ind(lines):
    return ["    " + x for x in lines]


def is_structured(src):
    """Does the program define main() (functions, recursion, loops), rather than step() and output()?"""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PipeError(f"syntax error: {e.msg}", e) from None
    return any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)


def lower(src, schema, level=0):
    """Plain pipe source. A program that defines main() is lowered (level 0: greedy blocks; 1: every Byte comparison
    a step ahead and one lookup per block; 2: one operation per step); any other program is returned unchanged."""
    if not is_structured(src):
        return src, None
    low = Lowering(src, schema, level)
    return low.source(), low


# ============================================================================ self-check & entry

def random_episodes(S, n, rng):
    """Random valid inputs in the task's format, plus the same data as Python values."""
    D_IN = S.D_IN
    Xq = np.zeros((n, D_IN))
    recs = []
    for kind, r in S.records.items():
        recs += [(kind, j) for j in range(r["count"])]
    Xr = np.zeros((n, len(recs), D_IN))
    episodes = []
    for b in range(n):
        q = {}
        for f, (t, off) in S.query.items():
            v = int(rng.integers(0, S.size(t)))
            q[f] = v
            _put(Xq[b], t, off, v)
        Xq[b, S.qflag] = 1
        tables, rows = {}, []
        for kind, r in S.records.items():
            keys = rng.permutation(r["count"])
            tables[kind] = []
            for k in keys:
                rec = {}
                for f, (t, off) in r["fields"].items():
                    rec[f] = int(k) if f == r["key"] else int(rng.integers(0, S.size(t)))
                tables[kind].append(rec)
                rows.append((kind, rec))
        order = rng.permutation(len(rows))
        for j, idx in enumerate(order):
            kind, rec = rows[idx]
            r = S.records[kind]
            if r["indicator"] is not None:
                Xr[b, j, r["indicator"]] = 1
            for f, (t, off) in r["fields"].items():
                _put(Xr[b, j], t, off, rec[f])
        episodes.append((q, tables))
    return Xq, Xr, episodes


def _put(x, t, off, v):
    if t[0] == "byte":
        for i in range(8):
            x[off + i] = (v >> i) & 1
    elif t[0] == "bool":
        x[off] = v
    else:
        x[off + v] = 1


def run_program(prog, q, tables, runs):
    env = prog.init_state(q)
    for _ in range(runs):
        env = prog.step(env, tables)
        if env.get(HALTED):                  # a halted program's state never changes again
            break
    return prog.output(env)


def target_forward(network, th, Xq, Xr):
    """The network's forward pass with the activation the current TARGET is meant for."""
    if TARGET != "tanh":
        return network.forward(th, Xq, Xr)
    steps = getattr(network, "STEPS", None) or network.T_MAX
    act = np.tanh
    x, R = Xq @ th["W_E"].T, Xr @ th["W_E"].T
    for _ in range(steps):
        x = x + network.attention(th, x, R)
        x = x + act(x @ th["W_R1"].T + th["b_R1"]) @ th["W_R2"].T + th["b_R2"]
    return act(x @ th["W_H"].T + th["b_H"]) @ th["W_out"].T + th["b_out"]


FIT_ERRORS = ("more than one layer", "arithmetic tests", "too complex", "too many cases", "loop iteration",
              "lookups into the same table", "two different lookups", "store a computed value", "work on Byte values")


def _fits_error(e):
    """Is this a 'does not fit in one step' error, which a finer schedule can fix?"""
    return isinstance(e, Deep) or any(k in str(e) for k in FIT_ERRORS)


def compile_program(src, n_check=600, seed=0, fit=False, return_plain=False):
    """Compile to weights. A program with main() is lowered at level 0, then 1, then 2 until it fits the network's
    one layer per step; level 2 (one operation per step) always fits, up to the network's resources."""
    import network
    import schema as schema_mod
    import task
    S = Schema(schema_mod, task.D_IN)
    levels = (0, 1, 2) if is_structured(src) else (0,)
    for level in levels:
        plain, low = lower(src, S, level)
        try:
            th, report = _build(plain, low, S, network, task, n_check, seed, fit)
        except (PipeError, Deep) as e:
            if level == levels[-1] or not _fits_error(e):
                if len(levels) > 1 and _fits_error(e):
                    raise PipeError(f"does not fit even with one operation per step: {e}") from None
                if isinstance(e, Deep):
                    raise PipeError(str(e)) from None
                raise
            continue
        if level:
            report.insert(1, f"  scheduled at level {level} ("
                             + ("Byte comparisons one step ahead, one lookup per step" if level == 1
                                else "one operation per step") + ")")
        return (th, report, plain) if return_plain else (th, report)


def _zeros_sized(dims, d_in, n_out):
    D, Dh, H, Ho = dims["D"], dims["Dh"], dims["H"], dims["Ho"]
    shapes = {"W_E": (D, d_in), "W_Q": (Dh, D), "W_K": (Dh, D), "W_V": (Dh, D), "W_O": (D, Dh),
              "W_R1": (H, D), "b_R1": (H,), "W_R2": (D, H), "b_R2": (D,),
              "W_H": (Ho, D), "b_H": (Ho,), "W_out": (n_out, Ho), "b_out": (n_out,)}
    return {k: np.zeros(v) for k, v in shapes.items()}


def _compile_fitted(prog, network, task):
    """Compile on a network grown until the program fits (widths only; the loop count is the network's)."""
    base = network.zeros()
    dims = {"D": base["W_E"].shape[0], "Dh": base["W_Q"].shape[0], "H": base["W_R1"].shape[0],
            "Ho": base["W_H"].shape[0]}
    d_in, n_out = base["W_E"].shape[1], base["W_out"].shape[0]
    original = network.zeros
    for _ in range(16):
        network.zeros = lambda: _zeros_sized(dims, d_in, n_out)
        try:
            comp = Compiler(prog, network, task)
            th, n_ctrl, n_head, n_ch = comp.compile()
            return comp, th, n_ctrl, n_head, n_ch
        except PipeError as e:
            msg = str(e)
            m = re.search(r"needs (\d+) controller units", msg) or re.search(r"needs (\d+) head units", msg)
            if m and "controller" in msg:
                dims["H"] = int(m.group(1))
            elif m:
                dims["Ho"] = int(m.group(1))
            elif "residual channels" in msg:
                dims["D"] *= 2
            elif "wider attention head" in msg:
                dims["Dh"] *= 2
            else:
                raise
        finally:
            network.zeros = original
    raise PipeError("could not size the network")


def _trim(th):
    """Drop channels and units that nothing uses: the same function on a smaller network."""
    def last(*arrs_axes):
        top = 0
        for a, axis in arrs_axes:
            nz = np.nonzero(np.any(a != 0, axis=1 - axis) if a.ndim == 2 else a != 0)[0]
            if len(nz):
                top = max(top, int(nz[-1]) + 1)
        return max(top, 1)
    D = last((th["W_E"], 0), (th["W_O"], 0), (th["W_R2"], 0), (th["b_R2"], 0), (th["W_Q"], 1), (th["W_K"], 1),
             (th["W_V"], 1), (th["W_R1"], 1), (th["W_H"], 1))
    H = last((th["W_R1"], 0), (th["b_R1"], 0), (th["W_R2"], 1))
    Dh = last((th["W_Q"], 0), (th["W_K"], 0), (th["W_V"], 0), (th["W_O"], 1))
    Ho = last((th["W_H"], 0), (th["b_H"], 0), (th["W_out"], 1))
    out = {"W_E": th["W_E"][:D], "W_Q": th["W_Q"][:Dh, :D], "W_K": th["W_K"][:Dh, :D], "W_V": th["W_V"][:Dh, :D],
           "W_O": th["W_O"][:D, :Dh], "W_R1": th["W_R1"][:H, :D], "b_R1": th["b_R1"][:H], "W_R2": th["W_R2"][:D, :H],
           "b_R2": th["b_R2"][:D], "W_H": th["W_H"][:Ho, :D], "b_H": th["b_H"][:Ho], "W_out": th["W_out"][:, :Ho],
           "b_out": th["b_out"]}
    return {k: np.ascontiguousarray(v) for k, v in out.items()}, dict(D=D, Dh=Dh, H=H, Ho=Ho)


def _build(src, low, S, network, task, n_check, seed, fit):
    prog = Program(src, S)
    if fit:
        comp, th, n_ctrl, n_head, n_ch = _compile_fitted(prog, network, task)
    else:
        comp = Compiler(prog, network, task)
        th, n_ctrl, n_head, n_ch = comp.compile()
    # compiler self-check on random inputs: weights vs the program's own semantics
    rng = np.random.default_rng(seed)
    Xq, Xr, eps = random_episodes(S, n_check, rng)
    want = np.array([run_program(prog, q, tables, S.runs) for q, tables in eps])
    got = target_forward(network, th, Xq, Xr).argmax(1)
    ok = want >= 0
    if np.any(got[ok] != want[ok]):
        raise RuntimeError(f"internal compiler error: the weights disagree with the program on "
                           f"{int(np.sum(got[ok] != want[ok]))} of {int(ok.sum())} random inputs")
    report = [f"channels {n_ch}/{th['W_E'].shape[0]}, controller units {n_ctrl}/{th['W_R1'].shape[0]}, "
              f"head units {n_head}/{th['W_H'].shape[0]}, loop iterations per step {comp.phases}"]
    if fit:
        small, dims = _trim(th)
        if np.any(target_forward(network, small, Xq, Xr).argmax(1) != got):
            raise RuntimeError("internal compiler error: the trimmed network disagrees with the full one")
        th = small
        report = [f"fitted network: width {dims['D']}, attention width {dims['Dh']}, {dims['H']} controller units, "
                  f"{dims['Ho']} head units; loop iterations per step {comp.phases}"]
    if low is not None:
        report += ["  " + line for line in low.report().split("\n")]
    for g in comp.groups:
        report.append(f"  lookup into {g['kind']} in iteration {g['phase'] + 1} of each step "
                      f"({len(g['sites'])} call site{'s' if len(g['sites']) > 1 else ''})")
    for v, *_ in comp.deferred:
        report.append(f"  {v}: part of its update is finished at the start of the next step")
    return th, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compile a decision-pipe program into weights.npz")
    ap.add_argument("program")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--check", type=int, default=600, help="random inputs for the self-check (default 600)")
    ap.add_argument("--lowered", default=None, help="also write the plain pipe program (after lowering functions)")
    ap.add_argument("-z", "--compress", action="store_true", help="write a compressed .npz")
    ap.add_argument("--fit", action="store_true", help="size the network to the program (widths and units)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(args.program))
    sys.path.insert(0, here)
    out = args.out or os.path.join(here, "weights.npz")
    src = open(args.program).read()
    try:
        if args.lowered:
            import schema as _schema
            import task as _task
            open(args.lowered, "w").write(lower(src, Schema(_schema, _task.D_IN))[0])
        theta, report = compile_program(src, n_check=args.check, fit=args.fit)
    except PipeError as e:
        print(f"error: {e}")
        sys.exit(1)
    (np.savez_compressed if args.compress else np.savez)(out, **theta)
    print(f"compiled {args.program} -> {out}")
    print("\n".join(report))
