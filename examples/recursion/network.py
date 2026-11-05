"""The fixed network that runs EXPR-128 (a runtime: it knows nothing about the task, it only executes tensors).

Same family as examples/micro8/network.py, larger:

    x  = W_E q                      state token (residual stream, width D)
    R  = W_E r_j                    record tokens: the 128 program nodes
    repeat up to STEPS times, same parameters every iteration:
        a  = softmax_j( (W_Q x) . (W_K R_j) )       one head, scale folded into W_Q
        x += W_O sum_j a_j W_V R_j
        x += W_R2 relu(W_R1 x + b_R1) + b_R2
    logits = W_out relu(W_H x + b_H) + b_out        258 logits: value 0..255, FAULT, TIMEOUT

Two things make it fast without changing any result:
  * the weights are sparse, so the products use sparse matrices;
  * an input whose state stops changing (x after an iteration equals x before it, exactly) is finished: every
    later iteration would compute the same state again, so it leaves the loop early.
"""
import numpy as np
import scipy.sparse as sp

from task import D_IN, N_OUT

D = 4096      # residual / state width
D_HEAD = 128  # attention head width
H_R = 17408   # recurrent controller hidden width
H_OUT = 16    # output head hidden width
STEPS = 65536 # loop iterations

SHAPES = {
    "W_E": (D, D_IN),
    "W_Q": (D_HEAD, D), "W_K": (D_HEAD, D), "W_V": (D_HEAD, D), "W_O": (D, D_HEAD),
    "W_R1": (H_R, D), "b_R1": (H_R,), "W_R2": (D, H_R), "b_R2": (D,),
    "W_H": (H_OUT, D), "b_H": (H_OUT,), "W_out": (N_OUT, H_OUT), "b_out": (N_OUT,),
}
_CACHE = {}


def zeros():
    """Every tensor starts at zero."""
    return {name: np.zeros(shape) for name, shape in SHAPES.items()}


def relu(z):
    return np.maximum(z, 0)


def _sparse(theta):
    key = tuple(id(theta[k]) for k in ("W_E", "W_Q", "W_K", "W_V", "W_O", "W_R1", "W_R2"))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = {k: sp.csr_matrix(theta[k]) for k in ("W_E", "W_Q", "W_K", "W_V", "W_O", "W_R1", "W_R2")}
    return _CACHE[key]


def attention_parts(theta, R):
    """Keys and values of the record tokens (they never change)."""
    s = _sparse(theta)
    B, N, _ = R.shape
    flat = R.reshape(B * N, -1)
    K = np.asarray((s["W_K"] @ flat.T).T).reshape(B, N, -1)
    V = np.asarray((s["W_V"] @ flat.T).T).reshape(B, N, -1)
    return K, V


def attention(theta, x, R, parts=None):
    s = _sparse(theta)
    K, V = parts if parts is not None else attention_parts(theta, R)
    q = np.asarray((s["W_Q"] @ x.T).T)
    sc = np.einsum("bnd,bd->bn", K, q)
    a = np.exp(sc - sc.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return np.asarray((s["W_O"] @ np.einsum("bn,bnd->bd", a, V).T).T)


def controller(theta, x):
    s = _sparse(theta)
    h = relu(np.asarray((s["W_R1"] @ x.T).T) + theta["b_R1"])
    return x + np.asarray((s["W_R2"] @ h.T).T) + theta["b_R2"]


def head(theta, x):
    return relu(x @ theta["W_H"].T + theta["b_H"]) @ theta["W_out"].T + theta["b_out"]


def forward(theta, Xq, Xr, trace=False, return_steps=False):
    s = _sparse(theta)
    x = np.asarray((s["W_E"] @ Xq.T).T)
    R = np.asarray((s["W_E"] @ Xr.reshape(-1, Xr.shape[-1]).T).T).reshape(Xr.shape[0], Xr.shape[1], -1)
    K, V = attention_parts(theta, R)
    active = np.arange(len(x))
    steps = np.zeros(len(x), dtype=np.int64)
    for t in range(STEPS):
        xa = x[active]
        xa = xa + attention(theta, xa, None, (K[active], V[active]))
        xa = controller(theta, xa)
        moved = np.any(xa != x[active], axis=1)
        x[active] = xa
        steps[active[moved]] = t + 1
        active = active[moved]
        if len(active) == 0:
            break
    logits = head(theta, x)
    return (logits, steps) if return_steps else logits
