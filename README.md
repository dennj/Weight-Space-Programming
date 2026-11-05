# Weight-space programming

Weight-space programming means setting a neural network's weights without training. Here you do it with decision
pipes: write a short program, and a compiler turns it into the exact weights of the network.

```python
def step():
    ins = attend(program.addr, pc)        # look up the instruction at pc
    if ins.op == ADD:
        C = A + ins.arg > 255             # carry
        A = A + ins.arg                   # wraps mod 256 when stored
    ...
```

Every `if` case in a decision-pipe program names a region of the network's input, and the statements under it are
the rule for that region. `pipec` turns each case into ReLU units that switch exactly on inside the region and off
outside it, and turns lookups into exact attention. Programs can also use functions, recursion and while loops: the
compiler turns them into blocks run one per step and a call stack held in the network's state.

Two examples are included:

- **MICRO-8:** a 64-line interpreter for an 8-bit CPU, compiled into a 737,537-parameter attention + ReLU network
  that gets 0 errors on 618,401 test programs.
- **Recursion:** a 77-line recursive evaluator. It runs recursive programs stored in the network's memory
  (factorial, Fibonacci, mutual recursion, gcd, minimax search) with 0 errors on 8,648 test programs.

**Start here:** [the article](https://dennjosele.com/blog/weight-space-programming) explains decision pipes; [SPEC.md](SPEC.md) defines the language.

## Quick start

Python 3.10 or later and NumPy (and SciPy for the recursion example's sparse runtime).

```bash
git clone https://github.com/dennj/Weight-Space-Programming.git
cd Weight-Space-Programming/examples/micro8
python3 ../../pipec/pipec.py program.pipe -o my_weights.npz   # compile: about half a second
python3 evaluate.py my_weights.npz                            # the 618,401-program test battery: a few minutes

cd ../recursion
python3 ../../pipec/pipec.py eval.pipe -z -o my_weights.npz   # compile the recursive evaluator: about 30 seconds
python3 evaluate.py my_weights.npz                            # the 8,648-program battery: about 5 minutes
```

## What's here

| Path | Contents |
|---|---|
| [SPEC.md](SPEC.md) | the Pipe language: types, statements, meaning, and what always compiles |
| [docs/](docs/) | the script that draws the article's pipe figure from a real network and checks what the article says about it |
| [pipec/](pipec/) | the compiler (one file), functions and recursion, its tests, and how to target another network |
| [examples/micro8/](examples/micro8/) | the 8-bit CPU: specification, network, program, compiled weights, test battery, and the experiment |
| [examples/recursion/](examples/recursion/) | a recursive evaluator: specification, network, program, compiled weights, program families and test battery |

## The experiment

Frontier language models can write neural-network weights directly
([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)). We gave the same tasks to agents
that wrote weights and to agents that wrote decision-pipe programs, with no feedback and hidden tests. All 18 agents
were exactly correct; programs were 2.4–3.7× shorter. Protocol and numbers:
[examples/micro8/README.md](examples/micro8/README.md#the-experiment); what it means in practice:
[direct weights or decision pipes](https://dennjosele.com/blog/weight-space-programming#compare).

## Related work

What is and is not new: [related work](https://dennjosele.com/blog/weight-space-programming#related).

- [A Neural Compiler](https://gpbib.pmacs.upenn.edu/gp-html/DBLP_journals_tcs_GruauRW95.html) (Gruau et al., 1995)
  compiles Pascal, including bounded recursion, into neural networks.
- [Tracr](https://arxiv.org/abs/2301.05062) compiles RASP programs into transformer weights.
- [ALTA](https://arxiv.org/abs/2410.18077) compiles programs with loops into looped transformers.
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196) builds a computer from a looped
  transformer.

## License

MIT. See [LICENSE](LICENSE).
