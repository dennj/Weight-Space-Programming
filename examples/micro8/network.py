"""The fixed network that runs MICRO-8 (a runtime: it knows nothing about the task, it only executes tensors).

    Encoder -> Attention -> Recurrent Controller (looped, shared weights) -> Feedforward Output Head

    x  = W_E q                      state token (residual stream, width D = 160)
    R  = W_E r_j                    record tokens (static memory: 16 program + 8 data records)
    repeat STEPS = 32 times, same parameters every iteration:
        a  = softmax_j( (W_Q x) . (W_K R_j) )       one head, scale folded into W_Q
        x += W_O sum_j a_j W_V R_j
        x += W_R2 relu(W_R1 x + b_R1) + b_R2        2,048 ReLU units
    logits = W_out relu(W_H x + b_H) + b_out        257 logits: A = 0..255, TIMEOUT

737,537 parameters. Every tensor starts at zero; a program compiled by pipec, or hand-written weights, fill them in.
"""
import numpy as np

from task import D_IN, N_OUT, STEPS

D = 160       # residual / state width
D_HEAD = 64   # attention head width
H_R = 2048    # recurrent controller hidden width
H_OUT = 64    # output head hidden width

SHAPES = {
    "W_E": (D, D_IN),
    "W_Q": (D_HEAD, D), "W_K": (D_HEAD, D), "W_V": (D_HEAD, D), "W_O": (D, D_HEAD),
    "W_R1": (H_R, D), "b_R1": (H_R,), "W_R2": (D, H_R), "b_R2": (D,),
    "W_H": (H_OUT, D), "b_H": (H_OUT,), "W_out": (N_OUT, H_OUT), "b_out": (N_OUT,),
}


def zeros():
    """Every tensor starts at zero."""
    return {name: np.zeros(shape) for name, shape in SHAPES.items()}


def relu(z):
    return np.maximum(z, 0)


def attention(theta, x, R):
    q = x @ theta["W_Q"].T                          # [B, Dh]
    Kr = R @ theta["W_K"].T                         # [B, 24, Dh]
    Vr = R @ theta["W_V"].T                         # [B, 24, Dh]
    s = (Kr @ q[..., None])[..., 0]                 # [B, 24]
    a = np.exp(s - s.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return (a[:, None, :] @ Vr)[:, 0] @ theta["W_O"].T


def controller(theta, x):
    return x + relu(x @ theta["W_R1"].T + theta["b_R1"]) @ theta["W_R2"].T + theta["b_R2"]


def head(theta, x):
    return relu(x @ theta["W_H"].T + theta["b_H"]) @ theta["W_out"].T + theta["b_out"]


def forward(theta, Xq, Xr, trace=False):
    x = Xq @ theta["W_E"].T
    R = Xr @ theta["W_E"].T
    states = [x]
    for _ in range(STEPS):
        x = x + attention(theta, x, R)
        x = controller(theta, x)
        states.append(x)
    logits = head(theta, x)
    return (logits, states) if trace else logits
