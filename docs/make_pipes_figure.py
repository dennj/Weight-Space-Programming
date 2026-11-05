"""Draw the "what is a decision pipe" figure of the article from a real network (docs/pipes_dark.svg).

The article: https://dennjosele.com/blog/weight-space-programming

    python3 docs/make_pipes_figure.py

The network is two residual ReLU blocks in 2D, the same kind of block the networks in this
project use:  x1 = x0 + V1 relu(U1 x0 + c1),  x2 = x1 + V2 relu(U2 x1 + c2).
Every tile and pipe is computed exactly by clipping convex polygons (no sampling), and the script
checks the two facts the article states: inside a pipe the network applies one affine rule, and
two inputs in the same pipe still get different outputs.
"""
import itertools
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# two residual ReLU blocks, 3 units each
U1 = np.array([[1.0, 0.35], [-0.45, 1.0], [0.8, -0.9]])
c1 = np.array([0.15, 0.25, -0.35])
V1 = 0.6 * np.array([[-0.55, 0.25, 0.45], [0.3, -0.5, 0.35]])
U2 = np.array([[0.25, 1.0], [1.0, -0.2], [-0.7, -0.75]])
c2 = np.array([-0.05, 0.3, 0.05])
V2 = 0.6 * np.array([[0.4, -0.45, 0.3], [-0.35, 0.3, 0.45]])
BOX = [(-1.6, -1.6), (1.6, -1.6), (1.6, 1.6), (-1.6, 1.6)]
P = np.array([-0.95, 0.85])
Q = np.array([0.95, -0.55])


def clip(poly, a, b, keep_positive):
    """Clip a convex polygon to the half-plane a.x + b >= 0 (or <= 0)."""
    s = 1.0 if keep_positive else -1.0
    out = []
    for i in range(len(poly)):
        p, q = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        fp, fq = s * (a @ p + b), s * (a @ q + b)
        if fp >= 0:
            out.append(tuple(p))
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append(tuple(p + t * (q - p)))
    return out


def area(poly):
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def tiles(poly_in, U, c, pre=(np.eye(2), np.zeros(2))):
    """Split a convex polygon (in input coordinates) by a layer whose input is A x + t."""
    A, t = pre
    out = {}
    for pattern in itertools.product((0, 1), repeat=len(c)):
        poly = list(poly_in)
        for k, on in enumerate(pattern):
            poly = clip(poly, U[k] @ A, U[k] @ t + c[k], bool(on))
            if len(poly) < 3:
                break
        if len(poly) >= 3 and area(poly) > 1e-9:
            out[pattern] = poly
    return out


def block_affine(U, c, V, pattern):
    """The affine map x -> x + V relu(U x + c) on the tile with this activation pattern."""
    D = np.diag(pattern).astype(float)
    return np.eye(2) + V @ D @ U, V @ D @ c


def pattern_of(x, U, c):
    return tuple(int(v > 0) for v in U @ x + c)


def build():
    layer1 = tiles(BOX, U1, c1)                                  # tiles of layer 1 in the input plane
    pipes = {}
    for p1, poly in layer1.items():
        A1, t1 = block_affine(U1, c1, V1, p1)
        for p2, sub in tiles(poly, U2, c2, (A1, t1)).items():    # layer-2 borders pulled back
            pipes[(p1, p2)] = sub
    return layer1, pipes


def check(pipes):
    """The facts the page states, checked on this network."""
    rng = np.random.default_rng(0)

    def net(x):
        x1 = x + V1 @ np.maximum(U1 @ x + c1, 0)
        return x1 + V2 @ np.maximum(U2 @ x1 + c2, 0)

    def key(x):
        x1 = x + V1 @ np.maximum(U1 @ x + c1, 0)
        return (pattern_of(x, U1, c1), pattern_of(x1, U2, c2))

    kP = key(P)
    poly = np.array(pipes[kP])
    inside = []
    while len(inside) < 3:                                       # random points in P's pipe
        w = rng.dirichlet(np.ones(len(poly)))
        x = w @ poly
        if key(x) == kP:
            inside.append(x)
    p1, p2 = kP
    A1, t1 = block_affine(U1, c1, V1, p1)
    A2, t2 = block_affine(U2, c2, V2, p2)
    A, t = A2 @ A1, A2 @ t1 + t2                                 # the pipe's single affine rule
    assert all(np.allclose(net(x), A @ x + t) for x in inside), "pipe is not one affine rule"
    outs = [net(x) for x in inside]
    assert not np.allclose(outs[0], outs[1]), "same pipe gave the same output"
    assert key(Q) != kP
    return len(pipes)


THEMES = {
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", line="#6f6e69",
                 faint="#2c2c2a", s1="#3987e5", s2="#d95926"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def figure(theme, layer1, pipes):
    c = THEMES[theme]
    W, H, S, top = 920, 430, 258, 92
    xs = [24, 24 + S + 44, 24 + 2 * (S + 44)]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family=\'{FONT}\' role="img" aria-labelledby="t d">',
           '<title id="t">How ReLU layers cut space into pipes</title>',
           '<desc id="d">Left: three ReLU units of layer 1 draw three lines that cut the input plane into tiles. '
           'Middle: each tile is moved by its own affine rule; layer 2 draws three more lines in the new space. '
           'Right: pulled back to the input plane, the lines of both layers cut it into pipes. Point P and point Q '
           'travel through different pipes.</desc>',
           f'<rect width="{W}" height="{H}" rx="10" fill="{c["surface"]}"/>']
    # legend
    lx = 24
    for col, name in ((c["s1"], "the pipe of input P"), (c["s2"], "the pipe of input Q")):
        svg.append(f'<rect x="{lx}" y="20" width="12" height="12" rx="3" fill="{col}"/>')
        svg.append(f'<text x="{lx + 18}" y="30" font-size="13" fill="{c["ink"]}">{name}</text>')
        lx += 18 + 7.2 * len(name) + 30
    svg.append(f'<line x1="{lx}" x2="{lx + 18}" y1="26" y2="26" stroke="{c["line"]}" stroke-width="1.5"/>')
    svg.append(f'<text x="{lx + 26}" y="30" font-size="13" fill="{c["ink2"]}">a border drawn by one ReLU unit</text>')

    def frame(i, lo, hi):
        x0 = xs[i]
        to = lambda p: (x0 + (p[0] - lo[0]) / (hi[0] - lo[0]) * S, top + (hi[1] - p[1]) / (hi[1] - lo[1]) * S)
        return to

    def poly_el(poly, to, fill, stroke, sw, opacity=1.0):
        pts = " ".join(f"{to(p)[0]:.1f},{to(p)[1]:.1f}" for p in poly)
        return (f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
                f'stroke-width="{sw}" stroke-linejoin="round"/>')

    def dot(to, p, col, label, dx=8, dy=-8):
        x, y = to(p)
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="{c["surface"]}" stroke-width="2"/>'
                f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="13" font-weight="600" fill="{c["ink"]}">{label}</text>')

    titles = [("1. Layer 1 cuts the input plane", "3 ReLU units, 3 borders, 7 tiles"),
              ("2. Each tile moves by its own rule", "then layer 2 draws 3 new borders"),
              ("3. Back in the input plane", f"{len(pipes)} pipes: one rule each")]
    for i, (t1, t2) in enumerate(titles):
        svg.append(f'<text x="{xs[i]}" y="{top - 30}" font-size="14" font-weight="600" fill="{c["ink"]}">{t1}</text>')
        svg.append(f'<text x="{xs[i]}" y="{top - 13}" font-size="12" fill="{c["ink2"]}">{t2}</text>')

    kP = (pattern_of(P, U1, c1), None)
    x1 = lambda x: x + V1 @ np.maximum(U1 @ x + c1, 0)
    kP = (pattern_of(P, U1, c1), pattern_of(x1(P), U2, c2))
    kQ = (pattern_of(Q, U1, c1), pattern_of(x1(Q), U2, c2))

    # panel 1: layer-1 tiles in the input plane
    to = frame(0, (-1.6, -1.6), (1.6, 1.6))
    for pat, poly in layer1.items():
        fill, op = (c["s1"], 0.18) if pat == kP[0] else (c["s2"], 0.18) if pat == kQ[0] else (c["surface"], 1)
        svg.append(poly_el(poly, to, fill, c["line"], 1.5, op))
    svg.append(dot(to, P, c["s1"], "P"))
    svg.append(dot(to, Q, c["s2"], "Q"))

    # panel 2: the tiles after layer 1 moved them, and layer 2's borders
    moved = {}
    for pat, poly in layer1.items():
        A, t = block_affine(U1, c1, V1, pat)
        moved[pat] = [tuple(A @ np.array(p) + t) for p in poly]
    allp = np.array([p for poly in moved.values() for p in poly])
    lo, hi = allp.min(0) - 0.15, allp.max(0) + 0.15
    span = max(hi - lo)
    mid = (hi + lo) / 2
    lo, hi = mid - span / 2, mid + span / 2
    to = frame(1, lo, hi)
    view = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    svg.append(poly_el(view, to, c["surface"], c["line"], 1.5, 1))
    for pat, poly in moved.items():                              # the layer-1 tiles after their move
        fill, op = (c["s1"], 0.18) if pat == kP[0] else (c["s2"], 0.18) if pat == kQ[0] else ("none", 1)
        svg.append(poly_el(poly, to, fill, c["muted"], 1, op))
    for k in range(len(c2)):                                     # layer 2's borders: U2[k].x + c2[k] = 0
        seg = clip(clip(view, U2[k], c2[k] + 1e-9, True), -U2[k], -c2[k] + 1e-9, True)
        if len(seg) >= 2:
            ends = sorted(seg, key=lambda q: (q[0], q[1]))
            (ax, ay), (bx, by) = to(ends[0]), to(ends[-1])
            svg.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{c["line"]}" '
                       f'stroke-width="1.5"/>')
    svg.append(dot(to, x1(P), c["s1"], "P"))
    svg.append(dot(to, x1(Q), c["s2"], "Q"))

    # panel 3: all pipes pulled back into the input plane
    to = frame(2, (-1.6, -1.6), (1.6, 1.6))
    for key, poly in pipes.items():
        fill, op = (c["s1"], 0.30) if key == kP else (c["s2"], 0.30) if key == kQ else (c["surface"], 1)
        svg.append(poly_el(poly, to, fill, c["line"], 1.0, op))
    svg.append(dot(to, P, c["s1"], "P"))
    svg.append(dot(to, Q, c["s2"], "Q"))
    svg.append(f'<text x="{xs[0]}" y="{H - 30}" font-size="12" fill="{c["muted"]}">Computed exactly from a real '
               f'two-layer ReLU network (docs/make_pipes_figure.py). Inside one pipe the network applies one</text>')
    svg.append(f'<text x="{xs[0]}" y="{H - 14}" font-size="12" fill="{c["muted"]}">linear-plus-offset rule, so '
               f'different points in the same pipe still get different outputs.</text>')
    svg.append("</svg>")
    return "\n".join(svg)


if __name__ == "__main__":
    layer1, pipes = build()
    n = check(pipes)
    for theme in THEMES:
        path = os.path.join(ROOT, "docs", f"pipes_{theme}.svg")
        open(path, "w").write(figure(theme, layer1, pipes))
        print("wrote", path)
    print(f"layer-1 tiles: {len(layer1)}, pipes: {n}; checks passed")
