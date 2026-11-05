"""Compiler tests. Usage: python3 test_pipec.py TASK_DIR [--battery]

For every test program in tests_<task>.py: compile it, then compare the compiled network with the
program's own semantics (the reference interpreter in pipec.py, after lowering functions) on 3,000 fresh random
inputs. The test file is tests_<task>.py, or the file named with --tests.
Also checks that invalid programs are rejected with the right error.

Programs whose name starts with TASK solve the task itself (capacity checks). With --battery, each
of them is also scored on the task's full test battery (evaluate.py), which takes a few minutes each.

    python3 test_pipec.py ../examples/micro8
"""
import os
import sys
import time

import numpy as np

TASK_DIR = os.path.abspath(sys.argv[1])
RUN_BATTERY = "--battery" in sys.argv[2:]
TESTS = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--tests=")),
             "tests_" + os.path.basename(TASK_DIR) + ".py")
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network  # noqa: E402
import pipec  # noqa: E402
import schema as schema_mod  # noqa: E402
import task  # noqa: E402

PROGRAMS = {}
BAD = {}
EXPECT = {}          # optional: name -> function(query, tables) giving the intended answer
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), TESTS)).read())


def differential(src, n=3000, seed=99):
    th, report, plain = pipec.compile_program(src, return_plain=True)
    S = pipec.Schema(schema_mod, task.D_IN)
    prog = pipec.Program(plain, S)
    Xq, Xr, eps = pipec.random_episodes(S, n, np.random.default_rng(seed))
    want = np.array([pipec.run_program(prog, q, t, S.runs) for q, t in eps])
    got = network.forward(th, Xq, Xr).argmax(1)
    return th, report, int(np.sum(got != want)), want


ok = True
for name, src in PROGRAMS.items():
    t0 = time.time()
    try:
        th, report, err, want = differential(src)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        continue
    sem = ""
    if name in EXPECT:                                  # the lowered program means what the source says
        S = pipec.Schema(schema_mod, task.D_IN)
        _, _, eps = pipec.random_episodes(S, 3000, np.random.default_rng(99))
        bad = int(np.sum(want != np.array([EXPECT[name](q, t) for q, t in eps])))
        err += bad
        sem = f", {bad} differ from the expected answer"
    ok &= err == 0
    spread = len(np.unique(want))
    print(f"{'PASS' if err == 0 else 'FAIL'}  {name}: {err} mismatches / 3000 ({spread} distinct outputs{sem}, "
          f"{time.time() - t0:.1f}s)\n      {report[0]}")
    for line in report[1:]:
        print("      " + line)
    if name.startswith("TASK") and RUN_BATTERY:
        import evaluate
        print("      task battery:")
        ok &= evaluate.battery(th)
for name, (src, expect) in BAD.items():
    try:
        pipec.compile_program(src)
        msg = "compiled without error"
    except pipec.PipeError as e:
        msg = str(e)
    good = expect in msg
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  rejects {name}: {msg}")
print("\nALL COMPILER TESTS PASSED" if ok else "\nCOMPILER TESTS FAILED")
sys.exit(0 if ok else 1)
