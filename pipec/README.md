# pipec: the decision-pipe compiler

`pipec.py` compiles a decision-pipe program into the weights of a fixed neural network: an encoder, one attention
head over a set of record tokens, a ReLU block applied in a loop, and a ReLU output head. One file, about 2,650
lines, NumPy only. Programs can use functions, recursion and while loops.

```bash
python3 pipec.py path/to/program.pipe [-o weights.npz] [-z] [--fit] [--check N] [--lowered plain.pipe]
```

- `-z` writes a compressed `.npz`, which matters for large, mostly-zero networks.
- `--fit` sizes the network to the program: widths, attention width and units
  ([below](#always-compiles-scheduling-and-sizing)).
- `--check N` sets how many random inputs the self-check uses (default 600).
- `--lowered plain.pipe` also writes the plain program that a program with functions is lowered to.

The language is defined in [SPEC.md](../SPEC.md); [examples/micro8/PIPES.md](../examples/micro8/PIPES.md) is a
manual for the fixed-schedule form.

It writes the weights (default: `weights.npz` next to the program) and prints what the program uses, for example:

```
channels 120/160, controller units 817/2048, head units 10/64, loop iterations per step 2
  lookup into program in iteration 1 of each step (1 call site)
  lookup into data in iteration 2 of each step (2 call sites)
  Z: part of its update is finished at the start of the next step
```

Errors name the line: syntax and type errors, a value that needs two layers in one step, a lookup out of the key's
range, or a program that needs more channels or units than the network has.

## How functions and recursion are lowered

A program that defines `main()` is first rewritten into a plain `step()`/`output()` program, which the compiler
then compiles as usual. `--lowered` shows the result.

- **Blocks.** Each function is cut into blocks, and one block runs per step, chosen by a state variable
  (`rec_blk`). A block ends at:
  - a call or a return;
  - a loop;
  - the join after an `if` that contains calls;
  - a statement that would use a value computed earlier in the same step.
- **Calls.** A call that is not a tail call pushes a frame onto a stack held in state variables: the block to come
  back to, and the caller's live values (only those still needed after the call, down to single record fields).
  Pushing and popping shift every frame by one, in one step. A return stores the value and jumps to the block
  saved in the top frame, which restores the values and pops.
  A frame is a pool of Bool channels shared by all call sites. Each site packs its values into it (a Bool in one
  channel, a Byte in 8, an enum or `IntN` one-hot), so a frame is as wide as the call site that saves the most
  channels. The internal functions `rec_bit(x, i)` and `rec_bits(b0, ..., b7)` move a Byte in and out of channels.
- **Two arithmetic tests.** A Byte comparison that guards Byte arithmetic in the same step would need two
  arithmetic tests in one layer. The comparison is then computed into a Bool one step earlier.
- **Halting.** Returning from `main()`, `stop()` and a full stack all go through one final block, the only one that
  halts.

The cost is size: a push or pop copies every frame, so the network grows with stack depth times frame width
([examples/recursion/](../examples/recursion/)).

## Sets: superposition as a type

`s: Sparse[N, K, C] = set()` declares a set of at most K elements out of 0..N−1 ([SPEC.md](../SPEC.md)). It is held
in Bool channels, superposed when a code is narrower than one channel per element:

- **The code** (Kautz–Singleton). Element j is the polynomial p over GF(q) whose coefficients are j's base-q
  digits, and its channels are (t, p(t)) for every t: q of q² channels. Two elements share at most m−1 channels
  (m is the number of digits), so when K·(m−1) < q a set of K elements never covers an element it doesn't hold,
  and "j in s" is exactly "all of j's channels are on". The compiler picks the smallest such q; for example 64
  elements fit in 49 channels with K = 3, and 128 in 121 with K = 5. Without such a code, a set has one channel
  per element.
- **`x in s`** takes two steps: q Bools each pick x's channel at one position, then their AND. Each pick is a chain
  whose cases test x alone, so it stays small.
- **`s.add(x)`** tests membership first. A new element either increments a counter or, when the set already has K,
  stops with class C. Then it turns on x's channels.

Every channel stays 0 or 1, and every step is one layer, so a set compiles like any other program and the
self-check verifies it. [experiments/exact_superposition.py](../experiments/exact_superposition.py) checks the code
exhaustively on its own.

## Always compiles: scheduling and sizing

A structured program (one that defines `main()`) that passes the name and type checks always compiles:

1. **Level 0, greedy.** Each step does as much as one layer of the network can.
2. **Level 1.** If a step does not fit, every Byte comparison is computed into a Bool one step before it is used,
   and each step makes at most one lookup. All lookups into a table go through one key register.
3. **Level 2, one operation per step.** Expressions become three-address code. A whole sum or difference counts as
   one operation and stays exact, and conditional expressions are moved up first, so no exact value is stored
   before it is used. This always fits one layer.

The report names the level when it is not 0. A finer level makes the program take more steps, never a different
answer: every test program is checked against the answer computed directly in Python
([tests_total.py](tests_total.py)).

`--fit` removes the other limit, a network that is too small. The compiler grows the network until the program fits,
then trims every channel and unit nothing uses, and checks on the self-check inputs that the trimmed network gives
the same answers. The loop count stays the network's.

The few remaining limits are in [SPEC.md](../SPEC.md#7-what-always-compiles). Fixed-schedule programs
(`step()`/`output()`) are not rescheduled: there, one `step()` is one network step by definition.

## How it compiles

- **Symbolic execution.** `step()` is executed symbolically. Every state variable ends up as a set of disjoint
  cases: "under condition c, the new value is e".
- **Cases become units.** Each condition, and each bit of each new value, becomes terms: an AND of one-hot or bit
  literals, with at most one integer threshold on a weighted sum of bits. Each term is 1–4 ReLU units:
  - an AND of conditions is `relu(1 − misses)`;
  - a threshold such as `A + arg > 255` is `relu(s − t + 1) − relu(s − t)` on whole numbers;
  - outside its case, a unit is switched off by a large negative bias.
- **Byte arithmetic** (`+`, `-`, `&`, `^`, `|`, `<<`, `>>`, comparisons) is built from these exact threshold units.
- **Lookups** become exact attention: score gaps of 1,000 make `exp(-1000)` exactly 0 in float64, so the head reads
  exactly one record. A lookup whose key depends on an earlier lookup runs one loop iteration later; lookups into the
  same table in different branches share an iteration.
- **Delays.** A value that needs a second layer is finished at the start of the next step, if nothing reads it first.
- **Self-check.** Before saving, the compiler runs the program (with its own interpreter) and the compiled network on
  600 random inputs and stops with an "internal compiler error" if they disagree. This checks the translation, not
  whether the program solves your task.

Every unit's input is a whole number (or, for a few scratch values, a multiple of 1/9), so it never lands near a
border. That gap is what makes the networks exact, and what lets them move to other activations.

## Targets

The `TARGET` constant at the top of `pipec.py` chooses how each gate is written:

| `TARGET` | Gate | Use |
|---|---|---|
| `"relu"` (default) | `relu(1 − misses)`, `relu(s − t + 1) − relu(s − t)` | the reference |
| `"relu_margin"` | `relu(2p + 0.5) − relu(2p − 0.5)`: snaps back to exactly 0 or 1 | absorbs noise that builds up over steps (low precision) |
| `"tanh"` | `0.5 + 0.5·tanh(64p)`; pass-throughs use `128·tanh(x/128)` | networks with a saturating activation |

GELU, SiLU and SwiGLU networks can be made from the ReLU output by scaling every unit by K (input weights and bias
×K, output weights ÷K; K = 256 for GELU and 512 for SiLU): `gelu(K·z)/K` equals `relu(z)` to float64 precision when
z keeps its gap from 0. That rewrite is not included here.

## Targeting another network

The compiler reads three files from the program's folder. To use it with your own network of the same family, provide:

| File | Must define |
|---|---|
| `network.py` | `zeros()`: every tensor at zero, named `W_E`, `W_Q`, `W_K`, `W_V`, `W_O`, `W_R1`, `b_R1`, `W_R2`, `b_R2`, `W_H`, `b_H`, `W_out`, `b_out` (their shapes set the widths); `STEPS`, the number of loop iterations; `forward(theta, Xq, Xr)`, which returns the logits; `attention(theta, x, R)` (for `TARGET = "tanh"`) |
| `task.py` | `D_IN`, the width of an input token |
| `schema.py` | `ENUMS`; `RECORDS` (each table's key, indicator channel, and fields with type and input offset); `QUERY` and `QUERY_FLAG` (the query token's fields); `RUNS`, how many times `step()` runs (`STEPS` must be a multiple of it); `OUTPUT` (the value type and named output classes) |

See [examples/micro8/](../examples/micro8/) for a complete set.

## Tests

```bash
python3 test_pipec.py ../examples/micro8             # about 15 seconds
python3 test_pipec.py ../examples/micro8 --battery   # also scores the TASK programs on the full battery
python3 test_pipec.py ../examples/recursion --tests=tests_recursion.py   # functions and recursion: about 4 minutes
python3 test_pipec.py ../examples/recursion --tests=tests_total.py       # programs that need rescheduling: about 2 minutes
python3 test_pipec.py ../examples/recursion --tests=tests_sparse.py      # sets: about 2 minutes
```

- [tests_micro8.py](tests_micro8.py) holds programs for other problems on the MICRO-8 schema, two complete MICRO-8
  interpreters (marked TASK), and invalid programs that must be rejected.
- [tests_recursion.py](tests_recursion.py) holds recursive programs on the EXPR-128 schema: factorial, Fibonacci,
  mutual recursion, a stack overflow, a while loop, a recursive walk over the program table. It also holds invalid
  programs.
- [tests_total.py](tests_total.py) holds programs that do not fit one layer per step as written. Examples: chained
  operations, two comparisons in one condition, a lookup keyed by another lookup's field, a conditional expression
  inside a comparison. Each is also checked against its intended answer computed in Python (`EXPECT`).
- [tests_sparse.py](tests_sparse.py) holds programs with sets: superposed and one-channel-per-element sets,
  duplicates, `clear()`, a set overflow, and a walk over the program table that stops when a node repeats. Each is
  checked against its intended answer (`EXPECT`), and invalid uses must be rejected.

Each program is compiled and compared with the compiler's interpreter on 3,000 fresh random inputs.

## Provenance

This is the compiler used in the experiment ([examples/micro8/README.md](../examples/micro8/README.md#the-experiment)),
plus things added afterwards: the `relu_margin` and `tanh` targets, the functions-and-recursion front end with its
scheduling levels and static type checks, `--fit`, `IntN` up to 256, shared stack frames, and sets. For programs without functions, the default target's output is byte-identical to the frozen copy
the agents used.
