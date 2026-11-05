# The decision-pipe language

You describe the computation as a small program: typed state plus *pipes*, which are named
blocks of decisions and updates. The compiler `pipec.py` turns the program into the weights of
the fixed network in `network.py`. You never write weights.

```
python3 pipec.py program.pipe      # writes weights.npz next to the program and prints a resource report
```

## Program shape

```python
name: Type = initial_value      # state declarations (top level)

def SomePipe():                 # any number of helper pipes, called as SomePipe()
    ...

def step():                     # required: one step of the computation. It runs 16 times.
    ...

def output():                   # required: computes the answer from the final state
    ...
```

Execution: the state is initialised, `step()` runs 16 times, and then `output()` returns
the answer. The syntax is Python's, but the file is never executed as Python. Only the constructs
below are allowed.

## Types

| Type | Values | Notes |
|---|---|---|
| `Bool` | `True`, `False` | |
| `Byte` | 0..255 | stored as 8 bits |
| `IntN`, e.g. `Int8`, `Int16` | 0..N-1 | stored one-hot (N channels) |
| `Op` | `HALT`, `LDI`, `LD`, `LDX`, `ADD`, `SUB`, `ADDB`, `AND`, `XOR`, `SHL`, `MOVB`, `JZ`, `JC`, `DJNZ`, `CALL`, `RET` | an enum (one-hot) |

An initial value is either a query field (below) or a constant of the right type.

## This task's data (from `schema.py`)

- **Query fields** (initial values only): `query.a: Byte`, `query.b: Byte`
- **Tables**, each looked up by its key field: `program` with fields `addr: Int16` (the key), `op: Op`, `arg: Byte`; `data` with fields `daddr: Int8` (the key), `value: Byte`
- **`output()` returns** a `Byte` value, or the class `TIMEOUT`

## Lookups

```python
r = attend(data.daddr, x)    # the record whose daddr equals x; then read r.<field>
v = attend(data.daddr, x).value  # or read a field directly
```

- The value you look up must stay within the key's range. Reduce it with `%` if needed; the
  compiler checks this.
- **Loop iterations.** The network loops 32 times for 16 steps, so each step has
  2 loop iteration(s), and each iteration does at most one lookup.
  - A lookup whose key depends on another lookup's result runs in a later iteration.
  - Lookups into the same table in different branches of an `if` share one iteration.
  - The compiler arranges all of this and reports it.

## Statements

- `x = expr`: assigns a state variable (only in `step()`) or a local name.
  - Storing an integer into a `Byte` or `IntN` variable keeps it modulo the type's size (it
    wraps around).
  - Locals keep exact integers.
  - `x += k` and `x -= k` also work.
- `if` / `elif` / `else`.
- Statements run in order, and later statements see earlier assignments.
- `halt` (only in `step()`): the program stops here. Nothing else happens in this or any later
  step, and assignments made before it are kept.
- `SomePipe()`: runs that pipe's statements here.
- `return expr` (only in `output()`): the first return reached is the answer. Every path must
  return. In `output()`, the name `halted` tells whether the program stopped with `halt`.
- `pass`.

## Expressions

- **Integers:** `+` and `-` give exact integers, so `a + b` can be 300 before it is stored, and
  `a + b > 255` works as expected. `==`, `!=`, `<`, `<=`, `>`, `>=` compare exact integers.
- **Bytes:**
  - `&`, `^`, `|` work on stored Bytes (variables, lookup fields or constants).
  - `x << k` and `x >> k` shift by a constant.
  - `x % 16` gives the low 4 bits; use any power of two.
- **Enums:** `==`, `!=`.
- **Logic:** `and`, `or`, `not`, and `a if cond else b`.
- **Constants:** integers, `True`, `False`, enum value names, and output class names (in
  `return`).

## What fits in one layer

Each loop iteration of the network is one layer of ReLU units. The compiler computes every new
value directly from the values at the start of the step and the lookup results.

- **Fits:** almost anything on stored values:
  - any logic on small values;
  - sums and differences of Bytes, with carry, borrow and comparison tests;
  - bitwise operations;
  - shifts;
  - several conditions combined.
- **Doesn't fit:** another arithmetic or bitwise operation on a value computed earlier in the
  same step, such as `y = (a + b) & 15`, or `x = x + 1` followed by `y = x << 1`. Keep such
  values in state and use them in the next step.
- **Finished one step later:** some new values are finished automatically at the start of the
  next step, for example `z = (a & k) == 0`. This works as long as nothing reads them before
  then, and the compiler reports it.
- **Otherwise:** the compiler stops with an error that names the line.

## Resource limits

These are the network's limits. `pipec` reports what the program uses.

- residual channels: 160;
- controller units: 2048;
- output-head units: 64;
- attention width: 64.

`Byte` costs 8 channels, `IntN` costs N, and `Bool` costs 1. Lookups, tables and bookkeeping
take the rest.

## How it compiles

- Every state variable gets its own channels in the residual stream. The encoder writes the
  initial values.
- `step()` is executed symbolically. Every new value becomes a set of disjoint cases, and each
  case becomes a few ReLU units that switch exactly on and off.
- Byte arithmetic uses exact threshold units on weighted sums of bits. Lookups become exact
  attention.
- `output()` is compiled the same way into the output head.
- Before saving, the compiler checks that the weights reproduce your program's own behaviour on
  random inputs. This checks the compiler, not whether your program solves the task.

## Example (a different task)

Walk the program addresses 0, 1, 2, ... in order. For each instruction, add the data value selected by its arg (arg mod 8) to a running total. Stop at the first HALT and output the total (TIMEOUT if there is no HALT). The data lookup depends on the program lookup, so the compiler gives it the second loop iteration of each step.

```python
pc: Int16 = 0
total: Byte = 0

def step():
    ins = attend(program.addr, pc)
    if ins.op == HALT:
        halt
    v = attend(data.daddr, ins.arg % 8).value
    total = total + v
    pc = pc + 1

def output():
    if halted:
        return total
    return TIMEOUT
```
