# Example: MICRO-8, an 8-bit CPU inside a neural network

A decision-pipe program of 64 lines of code, compiled into the weights of a fixed attention + ReLU network, that
runs programs on a small 8-bit CPU. It gets 0 errors on a battery of 618,401 test programs.

| File | What it is |
|---|---|
| [task.py](task.py) | the CPU's specification, the input encoding, and the exact oracle |
| [network.py](network.py) | the fixed network (a runtime; it knows nothing about the task) |
| [schema.py](schema.py) | the input/output format, as the compiler sees it |
| [PIPES.md](PIPES.md) | the language manual, as given to the agents who wrote programs for this task |
| [program.pipe](program.pipe) | the program: an interpreter, one instruction per step |
| [program_notes.md](program_notes.md) | the program's author explaining its design |
| [weights.npz](weights.npz) | the compiled output: every tensor of the network |
| [evaluate.py](evaluate.py) | the test battery |
| [verify_cpu.py](verify_cpu.py) | an extra check: exhaustive ALU tests, longer programs, fresh random programs |

## The CPU

- **Registers:** A and B, 8 bits each (0..255), stored in binary. Flags Z (zero) and C (carry). A return register R
  and a program counter pc (0..15).
- **16 instructions:** `HALT`, `LDI`, `LD`, `LDX` (load from data, indexed by B), `ADD`, `SUB`, `ADDB` (with carry),
  `AND`, `XOR`, `SHL`, `MOVB`, `JZ`, `JC`, `DJNZ`, `CALL`, `RET`.
- **Memory:** 16 program records and 8 data records, given to the network as 24 tokens in a random order.
- **Run:** up to 16 executed instructions. The output is A (256 classes) if the program reached `HALT`, otherwise
  `TIMEOUT`.

The full specification, with a worked example, is the docstring of [task.py](task.py).

## The network

[network.py](network.py): an encoder; one attention head (64 wide) from a state token over the 24 record tokens; a
block of 2,048 ReLU units applied 32 times with shared weights; an output head with 64 ReLU units and 257 logits.
737,537 parameters, all starting at zero.

## The program

[program.pipe](program.pipe) executes one instruction per step: 16 steps of 2 loop iterations each. Iteration 1
fetches the instruction at pc, and iteration 2 fetches data for `LD` and `LDX` and executes. Each instruction is
one `elif`:

```python
    elif op == ADD:
        C = A + arg > 255                         # carry out (old A)
        Z = (A + arg) % 256 == 0                  # new A == 0
        A = A + arg
```

It was written by an AI agent (Claude Opus 5.5) in the experiment below, without running anything, and compiled once.

Compiling it:

```
$ python3 ../../pipec/pipec.py program.pipe
compiled program.pipe -> weights.npz
channels 120/160, controller units 817/2048, head units 10/64, loop iterations per step 2
  lookup into program in iteration 1 of each step (1 call site)
  lookup into data in iteration 2 of each step (2 call sites)
  Z: part of its update is finished at the start of the next step
```

Compiling takes about half a second, and the output is byte-identical to the stored [weights.npz](weights.npz).

## The test battery

[evaluate.py](evaluate.py) runs the network as a black box and compares every answer with the oracle in
[task.py](task.py). Exact success means 0 errors on every in-spec test:

| Test | Programs |
|---|---|
| the worked example from the spec | 1 |
| every ordered pair of opcodes at addresses 0–1 (400 programs each) | 102,400 |
| per-opcode stress (half the instructions are one opcode) | 96,000 |
| edge values (0, 1, 2, 127, 128, 129, 254, 255) | 60,000 |
| random programs: uniform, more HALTs, control-flow heavy | 3 × 120,000 |
| **in-spec total** | **618,401** |
| extra, not counted: random programs in float32 | 30,000 |

The tests are sensitive: deleting one randomly chosen unit from a correct network caused errors in 15 of 15 trials.

[verify_cpu.py](verify_cpu.py) adds every 8-bit operand pair for `ADD`, `SUB`, `ADDB`, `AND` and `XOR` (result,
carry and zero flag), all values for `SHL` and `DJNZ`, multiplication by repeated addition, nested subroutine calls,
and 200,000 fresh random programs: 1,055,296 programs in all, 0 errors for the stored weights (about 6 minutes).

## Run it

Python 3.10 or later and NumPy.

```bash
cd examples/micro8
python3 ../../pipec/pipec.py program.pipe -o my_weights.npz   # compile (about 0.5 s)
python3 evaluate.py my_weights.npz                            # the test battery (a few minutes)
python3 verify_cpu.py my_weights.npz                          # the extra CPU checks (about 6 minutes)
cd ../../pipec && python3 test_pipec.py ../examples/micro8    # the compiler's own tests
```

Without `-o`, the compiler writes `weights.npz` next to the program, replacing the stored copy (with identical
bytes).

## The experiment

Language-model agents can write a network's weights directly. We compared that with writing a decision-pipe
program and letting a compiler produce the weights.

- **Two arms, one model.** Every agent was Claude Opus 5.5 with a fresh context, 3 agents per arm per task.
  - **Weights:** write Python that sets every weight of the network.
  - **Program:** write a decision-pipe program; a frozen copy of this compiler turns it into the weights.
- **The same task text and the same network for both arms.** Program agents also got the language manual
  ([PIPES.md](PIPES.md)), the compiler and the I/O schema.
- **No feedback.** No training, fitting or search, and no running or testing the network. Program agents could run
  the compiler, which reports syntax, type and resource errors. Its self-check runs the program and the compiled
  network side by side on random inputs. It tells the agent whether the translation is faithful, never whether the
  program solves the task.
- **Scored on the hidden battery above.** Every agent's tool calls were audited afterwards, and all followed the
  rules.

MICRO-8 was the hardest of three tasks, and was designed to break direct weight writing: binary arithmetic (a one-hot
`ADD` table would need 65,536 cases), one attention head that must fetch both instructions and data, and a zero flag
too deep for one layer. It didn't break it:

| | Pointer machine | Register machine | MICRO-8 |
|---|---|---|---|
| Exactly correct: weights / program | 3/3 · 3/3 | 3/3 · 3/3 | 3/3 · 3/3 |
| Code written (lines): weights / program | 68–99 / 21–25 | 87–112 / 38–45 | 171–219 / 64–78 |
| Agent time (mean minutes): weights / program | 3.3 / 2.3 | 4.1 / 3.8 | 8.2 / 5.4 |
| Agent tokens (mean): weights / program | 56k / 65k | 64k / 95k | 94k / 105k |
| Network parameters | 8,393 | 147,786 | 737,537 |

- **Programs were 2.4–3.7× shorter,** and program agents finished faster on all three tasks.
- **Program agents used 12–50% more tokens,** mostly reading the compiler's source, which they chose to do.
- **Compiled networks are bigger:** 817–822 ReLU units on MICRO-8, against 362–518 for hand-written weights. The
  compiler doesn't share units across instructions the way the weight writers did.

Only MICRO-8 is included in this repository. Code lines exclude comments and docstrings.

## Limitations

- **Small samples.** Three successes out of three are consistent with a true success rate as low as 29% (exact 95%
  interval).
- **Large weights:** circuits are switched off with weights in the thousands, which low-precision formats cannot
  hold precisely.
- **The compiler was designed with these tasks in view.** It has no task-specific constructs and was frozen before any
  agent ran, but its feature set (byte arithmetic, two tables, delayed flags) was chosen knowing what the tasks need.
- **Exact means 0 errors on the stated tests, in float64.** The exact attention lookups rely on `exp(-1000)` being
  exactly 0 in float64. A test battery is not a proof.
