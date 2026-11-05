# MICRO-8 as a decision-pipe program

`program.pipe` is a direct interpreter: each of the 16 `step()` runs executes exactly one
instruction, so a HALT reached by the 16th instruction sets `halted`, and otherwise the result is
TIMEOUT.

State: `A`, `B` (Byte, from the query), `Z`, `C` (Bool, False), `R`, `pc` (Int16, 0).

Each step uses two loop iterations:

1. **Fetch.** `attend(program.addr, pc)` gets `op` and `arg`, and the compiler latches them.
2. **Data and execute.** LD looks up `data[arg % 8]` and LDX looks up `data[B % 8]`. The two
   lookups are in exclusive branches, so they share this iteration's lookup. All state
   updates happen here.

Every value is computed from the values at the start of the step:

- `pc = pc + 1` by default. JZ, JC and DJNZ may override it with `arg % 16`. CALL sets
  `R = pc + 1` and `pc = arg % 16`. RET sets `pc = R`.
- Flags are computed from the *old* A, before A is reassigned:
  - ADD and ADDB: `C = A + x > 255` and `Z = (A + x) % 256 == 0`.
  - SUB: `C = A < arg` and `Z = A == arg`.
  - XOR: `Z = A == arg`.
  - SHL: `C = A > 127` and `Z = (A & 127) == 0`.
- DJNZ tests the old B (`B != 1` means the new B is nonzero), then sets `B = B - 1`.
- AND sets `Z = (A & arg) == 0`. This needs two layers, so it is written as `A = A & arg`
  followed by `Z = A == 0`. The compiler finishes it at the start of the next step, before
  anything can read Z: JZ only reads Z in the second iteration. `output()` never reads Z.
- HALT executes `halt`, which freezes the state.

`output()` returns A if the program halted, otherwise TIMEOUT.

Compile report: 120/160 channels, 817/2048 controller units, 10/64 head units, 2 loop
iterations per step.
