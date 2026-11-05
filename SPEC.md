# The Pipe language

Pipe is a small, statically typed language whose programs compile into the weights of a neural network. This
document defines the language as `pipec` implements it. For how compilation works, see [pipec/README.md](pipec/README.md).

A Pipe program comes in one of two forms:

- **Structured** (it defines `main()`): functions, recursion and loops. Time is abstract; the compiler decides how
  many network steps each part takes. **Every structured program that passes the checks in this document
  compiles** (see [What always compiles](#7-what-always-compiles)).
- **Fixed-schedule** (it defines `step()` and `output()`): one `step()` is one network step, by definition. The
  compiler does not reschedule it, so a step that needs more than one layer is an error.

The structured form compiles to the fixed-schedule form; `pipec --lowered` shows the result.

The syntax is a subset of Python's, but the meaning is Pipe's. A Pipe file is never run as Python.

## 1. The target

A program runs against a **task**: three files next to the program.

| File | Defines |
|---|---|
| `schema.py` | enums, record tables (with a key field), query fields, output values and output classes, and `RUNS` |
| `network.py` | the network: an encoder, one attention head over the records, a ReLU block applied in a loop, an output head |
| `task.py` | `D_IN`, the width of an input token |

The **input** is a query (named fields) and the records of each table, one record per key value, in any order.
The **output** is one value of the output type, or one output class (for example `FAULT` or `TIMEOUT`).

## 2. Types

| Type | Values | Stored as |
|---|---|---|
| `Bool` | `True`, `False` | 1 channel |
| `Byte` | 0..255 | 8 bit channels |
| `IntN`, 2 ≤ N ≤ 256 | 0..N−1 | N one-hot channels |
| an enum from the schema | its value names | one-hot channels |

There are no other types: no signed numbers, no multiplication or division, no floating point, no strings, no
arrays. Record tables are read-only and come from the input.

## 3. Program structure (structured form)

```python
total: Byte = 0                            # global state: a query field or a constant
recursion(depth=32, overflow=FAULT)        # the call stack (needed if any call is not a tail call)

def main() -> Byte:                        # required; no parameters; returns the output type
    return f(query.arg)

def f(n: Byte) -> Byte:                    # typed parameters and result
    ...
```

- **Global state** is declared at top level as `name: Type = initial`, where the initial value is a query field
  (`query.a`) or a constant. Functions can read and assign it. It is shared: calls do not save it.
- **`recursion(depth=K, overflow=C, timeout=T)`**:
  - `depth` is the number of stack frames;
  - `overflow` is the output class when a call would need frame K + 1;
  - `timeout` (default `TIMEOUT`) is the output class when the network's step budget runs out before `main()`
    returns.
- **Functions** have typed parameters and a result type. Every path must end in `return` or `stop(...)`.
- **Names** starting with `rec_` are reserved for the compiler.

## 4. Statements

| Statement | Meaning |
|---|---|
| `x: T = e` | declares local `x` of type `T` and assigns it |
| `x = e`, `x += e`, `x -= e` | assigns a declared local, a parameter or global state |
| `n = attend(table.key, e)` | looks up the record whose key equals `e`; `n` is a record local |
| `x = f(args)`, `f(args)` | calls a function |
| `if` / `elif` / `else` | branches |
| `while c:` | loops while `c` holds |
| `return e` | returns `e`; `return f(args)` is a tail call |
| `stop(C)` | ends the program with output class `C` |
| `pass` | does nothing |

- **A local's type** comes from its first assignment's annotation, or from the result type of the function it is
  assigned from.
- **Storing wraps.** Storing an integer into `Byte` or `IntN` keeps it modulo 256 or N. Storing into `Bool` or an
  enum requires a value of that type.
- **Statements run in order,** and later statements see earlier assignments.

## 5. Expressions

| Expression | Meaning |
|---|---|
| integer constants, `True`, `False`, enum value names | constants |
| names, `query.f`, `n.field` | locals, parameters, global state, query fields, record fields |
| `a + b`, `a - b` | exact integers (no wrap until stored) |
| `==`, `!=` | any two values of the same type, or two integers |
| `<`, `<=`, `>`, `>=` | integers |
| `a & b`, `a ^ b`, `a | b` | bitwise, on Bytes |
| `a << k`, `a >> k` | shift by a constant `k` |
| `a % 2^k` | the low `k` bits |
| `and`, `or`, `not`, `x if c else y` | logic and choice |
| `f(args)` | the function's result |

- **Exact within an expression.** Values are exact integers within an expression, so `a + b > 255` tests the
  carry, and `a - b < 0` tests the borrow. A value is reduced only when it is stored.
- **Operands of bitwise operators and shifts are Bytes.** A computed integer used there is taken modulo 256. In the
  fixed-schedule form it must already be a stored Byte.
- **Evaluation order.** Calls inside an expression run first, innermost first and left to right.
- **Lookups.** The lookup value must stay within the key's range; reduce it with `%` if needed. The compiler checks
  this.

## 6. Meaning

- **Calls** pass values. A call that is not a tail call holds one stack frame until it returns. A tail call
  (`return f(x)`) holds none, so tail recursion runs in constant stack space.
- **The answer** is the value `main()` returns, or the class of `stop(C)`, of a stack overflow, or of the timeout.
- **Time is abstract.** The answer of a program that finishes does not depend on how the compiler schedules it. The
  step budget is the network's loop count. Exactly where the timeout falls depends on the schedule, so a program
  meant to finish should stay well inside the budget.
- **The reference.** The meaning of a program is fixed by the reference interpreter in `pipec.py`, run on the
  program the compiler actually compiles. Before saving, the compiler checks the weights against it on random
  inputs.

## 7. What always compiles

A structured program that passes the checks above (syntax, names, declarations and types) compiles. The compiler
first tries greedy blocks. If any step does not fit one layer, it reschedules the whole program, ending at one
operation per step, which always fits. The report says which level was used:

| Level | Schedule |
|---|---|
| 0 | greedy: each step does as much as one layer can |
| 1 | every Byte comparison is computed one step before it is used; one lookup per step |
| 2 | one operation per step (a whole sum or difference, kept exact, counts as one) |

What can still stop compilation:

- **Resources of a fixed network:** channels, controller units, head units or attention width. `pipec --fit` sizes
  the network to the program instead, then checks that the trimmed network computes the same.
- **Primitive limits:** a lookup value the compiler cannot bound within the key range (reduce it with `%`); `%` by a
  number that is not a power of two; and one condition or value whose case analysis exceeds 64 cases. An example
  is arithmetic between two one-hot `IntN` variables with large N.

The fixed-schedule form has no such guarantee: one `step()` is one network step, and the compiler reports the line
of any statement that does not fit.

## 8. The fixed-schedule form

```python
pc: Int16 = 0                    # state
def step():                      # runs exactly RUNS times
    ins = attend(program.addr, pc)
    if ins.op == HALT:
        halt                     # stops: nothing changes after this
    pc = pc + 1
def output():                    # the answer from the final state; can read `halted`
    if halted:
        return pc
    return TIMEOUT
```

- **Statements:** everything in section 4 except functions with parameters, `while`, `return` (allowed only in
  `output()`) and `stop`. `halt` ends the run.
- **Helpers:** helper pipes without parameters can be called as statements.
- **Lookups:** each lookup uses one loop iteration of the network; the compiler schedules lookups within a step.
- **One layer per step:** every new value must be computable in one layer from the values at the start of the step.
  A value that needs a second layer is finished at the start of the next step when possible.

The full manual for this form, written for MICRO-8, is [examples/micro8/PIPES.md](examples/micro8/PIPES.md).
