# Example: recursion, a recursive evaluator compiled into a network

A 77-line decision-pipe program with recursive functions, compiled into the weights of a looped attention + ReLU network.
The network evaluates recursive programs stored in its memory: factorial, Fibonacci, mutual recursion, gcd, minimax
search over game trees and random expressions. It gets 0 errors on a battery of 8,648 programs.

| File | What it is |
|---|---|
| [task.py](task.py) | EXPR-128: the specification, the input encoding and the exact oracle |
| [eval.pipe](eval.pipe) | the program: a recursive evaluator |
| [network.py](network.py) | the fixed network (a runtime; it knows nothing about the task) |
| [schema.py](schema.py) | the input/output format, as the compiler sees it |
| [weights.npz](weights.npz) | the compiled output (compressed; 146M parameters, almost all zero) |
| [programs.py](programs.py) | an assembler from expression trees to program memory, and the program families |
| [evaluate.py](evaluate.py) | the test battery |

## The task: EXPR-128

The network's memory holds a program of up to 128 nodes. Each node has an op, three pointers (`x`, `y`, `z`) and a
byte `imm`. The query gives a root node and a byte argument. The answer is the value of the root:

| op | value |
|---|---|
| `CONST` | `imm` |
| `ARG` | the current argument |
| `ADD`, `SUB`, `MUL`, `LEQ`, `BAND`, `BXOR`, `SHR` | `ev(x) op ev(y)`, mod 256 |
| `IF` | `ev(y)` if `ev(x) != 0`, else `ev(z)`; only the chosen branch is evaluated |
| `CALL` | `ev(y)` with the argument `ev(x)`; `y` is the called function's root |
| `NOP` | invalid: a fault (empty memory holds `NOP`) |

So `CALL` makes programs recursive. Factorial is three nodes deep:
`If(Le(ARG, 1), 1, Mul(ARG, Call(fact, Sub(ARG, 1))))`.

The output is the value (256 classes), FAULT (an invalid node, or more than 64 pending evaluations), or TIMEOUT
(still running after the step budget). The full specification is the docstring of [task.py](task.py).

## The program

[eval.pipe](eval.pipe) is the evaluator, written as the recursive function anyone would write:

```python
recursion(depth=64, overflow=FAULT, timeout=TIMEOUT)

def main() -> Byte:
    return ev(query.root, query.arg)

def ev(node: Int128, env: Byte) -> Byte:
    n = attend(program.addr, node)
    if n.op == CONST:
        return n.imm
    elif n.op == ARG:
        return env
    elif n.op == IF:
        c: Byte = ev(n.x, env)
        if c != 0:
            return ev(n.y, env)
        return ev(n.z, env)
    elif n.op == CALL:
        a: Byte = ev(n.x, env)          # the argument, in the caller's environment
        return ev(n.y, a)               # the body, with the new argument
    ...
        l: Byte = ev(n.x, env)
        r: Byte = ev(n.y, env)
        if n.op == ADD:
            return l + r
        ...
        elif n.op == MUL:
            return mul(l, r)            # a shift-and-add loop, one bit per step
```

The compiler turns the recursion into what a looped network can run:

- **Blocks.** Each function is cut into blocks, and one block runs per step. The evaluator has 10 blocks: `ev`'s
  entry, one continuation after each of its four kinds of call, `mul`'s entry and loop, `shr`, `main`, and the end.
- **A stack in the network's state.** A call that is not a tail call pushes a frame: the block to come back to and
  the caller's live values. Here that is `env`, the left operand `l`, and the fields `n.op`, `n.y`, `n.z` that are
  still needed. All call sites share one pool of channels per frame, each packing its own values into it (a Byte
  as 8 bits, `Op` one-hot), so a frame is as wide as the widest call site: the stack is 64 frames of 39 channels in
  the residual stream. Pushing or popping shifts every frame by one, in one step.
- **Tail calls** (`return ev(n.y, env)`) jump without pushing, so mutual recursion such as even/odd runs in constant
  stack space.

Compiling it:

```
$ python3 ../../pipec/pipec.py eval.pipe -z
compiled eval.pipe -> weights.npz
channels 2942/4096, controller units 12861/17408, head units 11/16, loop iterations per step 1
  recursion: 4 functions in 10 blocks (one per step), stack of 64 frames of 39 channels, 4 call sites that push
  lookup into program in iteration 1 of each step (1 call site)
```

That takes about a minute. The plain program it compiles, after lowering, is about 17,000 lines, most of them
copying stack channels on pushes and pops; `--lowered out.pipe` writes it out.

## The network

[network.py](network.py) is the same family as MICRO-8's: an encoder; one attention head from a state token over the
128 node records; a block of 17,408 ReLU units applied up to 65,536 times with shared weights; and an output head.
It has 146M parameters, of which about 149,000 are nonzero. The largest weight is 2,105.

The runtime is sparse, and an input whose state stops changing leaves the loop. A halted program's state never
changes again, so this gives the same answer as running all 65,536 iterations. `fib(12)` finishes after 7,434.

## The test battery

[evaluate.py](evaluate.py) runs the network as a black box and compares every answer with the oracle in
[task.py](task.py), an independent iterative evaluator that counts pending evaluations the same way the spec does.

| Test | Programs | Errors |
|---|---|---|
| sum, factorial, fib, pow2, xor_range, even/odd, gcd, popcount, a dead recursive branch | 662 | 0 |
| random expressions | 1,000 | 0 |
| minimax over random game trees, maximizing and minimizing root | 224 | 0 |
| every operation on random operands | 4,500 | 0 |
| faults: empty nodes, and recursion deeper than the stack (`sum(n)` around 64) | 232 | 0 |
| programs that never finish (expected TIMEOUT) | 30 | 0 |
| random program memories (363 values, 1,632 FAULT, 5 TIMEOUT) | 2,000 | 0 |
| **total** | **8,648** | **0** |

The stack-overflow boundary is exact: `sum(62)` and `sum(63)` return their values, and `sum(64)` and beyond are
FAULT, exactly as the oracle counts. The tests include only programs that finish within 5,000 node evaluations, or
never finish. At most 13 steps per evaluation means those always fit in the budget.

## Compared with building it by hand

This machine was first built gate by gate, as a 60-layer circuit with a separate writable stack memory. The program
families and generators in [programs.py](programs.py) come from that version. On the same programs:

| Program | Hand-built: steps / stack frames | Decision-pipe program: loop iterations / stack frames |
|---|---|---|
| `fib(12)` | 7,897 / 24 | 7,434 / 13 |
| `even(100)` | 1,509 / 102 | 1,410 / 2 |
| `sum(40)` | 769 / 82 | 730 / 42 |
| `pow2(60)` | 1,149 / 122 | 1,690 / 62 |
| `factorial(20)` | 370 / 40 | 542 / 21 |

- **Stack:** about half as deep, because tail calls hold no frame.
- **Steps:** similar. Programs heavy in `MUL` take more, since a multiply loops over 8 bits instead of using a deep
  circuit.
- **Depth per step:** each compiled step is one ReLU layer; each hand-built step was a 60-layer circuit.
- **What you write:** the hand-built version is a circuit library and a decoder written by hand; this is the
  recursive evaluator above, and the compiler derives the frames.
- **Cost:** the compiled network is far larger (12,861 units and a 2,496-channel stack in the residual stream,
  against about 1,250 units and a separate memory).

## Run it

Python 3.10 or later, NumPy and SciPy (for the sparse runtime).

```bash
cd examples/recursion
python3 ../../pipec/pipec.py eval.pipe -z -o my_weights.npz          # compile: about a minute
python3 evaluate.py my_weights.npz                                   # the battery: about 5 minutes
python3 evaluate.py my_weights.npz --quick                           # a 10% sample: about a minute
cd ../../pipec && python3 test_pipec.py ../examples/recursion --tests=tests_recursion.py   # compiler tests
```

## Limitations

- **Bounded:** 128 program nodes, 64 stack frames, 65,536 steps, bytes mod 256.
- **One layer per step** makes some programs slower: a byte comparison that guards byte arithmetic costs an extra
  step, and `MUL` is a loop.
- **Exact means 0 errors on the stated tests, in float64.** A test battery is not a proof.
