"""Draw the superposition figure of the article from the real code (docs/superposition_dark.svg).

The article: https://dennjosele.com/blog/exact-superposition

    python3 docs/make_superposition_figure.py

The code is the one pipec uses for Sparse[N, K, C] sets: element j is the polynomial over GF(7) whose coefficients
are j's base-7 digits (degree < 3), and its channels are (t, p_j(t)) for t = 0..6, so 7 of 49 channels per element.
Two elements share at most 2 channels, so 3 stored elements cover at most 6 channels of any other element, and an
absent element always has a channel that is off. The script picks three stored elements and an absent one with 6
of its 7 channels covered (the worst case), checks those facts, and draws them.
"""
import itertools
import os

Q, M = 7, 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def channels(j):
    digits = [(j // Q ** i) % Q for i in range(M)]
    return [(t, sum(c * t ** i for i, c in enumerate(digits)) % Q) for t in range(Q)]


def pick():
    """Three stored elements that overlap, and an absent element with 6 of its 7 channels covered."""
    N = Q ** M
    for a, b, c in itertools.combinations(range(1, N), 3):
        A, B, C = (set(channels(x)) for x in (a, b, c))
        if not (A & B and B & C):
            continue
        on = A | B | C
        for d in range(N):
            if d in (a, b, c):
                continue
            covered = sum(ch in on for ch in channels(d))
            if covered == 6:
                return (a, b, c), d
    raise RuntimeError("no example found")


def check(stored, d):
    on = set().union(*(channels(x) for x in stored))
    for x in range(Q ** M):                       # the guarantee, on every element of the code
        shared = max(len(set(channels(x)) & set(channels(y))) for y in stored if y != x) if x not in stored else 0
        assert shared <= M - 1
        member = all(ch in on for ch in channels(x))
        assert member == (x in stored), x
    assert sum(ch in on for ch in channels(d)) == 6


def svg(stored, d):
    W, H, cell, top = 920, 470, 40, 118
    left = [70, 70 + 460]
    ink, ink2, muted, line, faint, surface = "#ecebe8", "#a3a3a8", "#77777d", "#3a3a40", "#34343e", "#111114"
    colors = ["#3987e5", "#d95926", "#199e70"]
    names = ["a", "b", "c"]
    on = set().union(*(channels(x) for x in stored))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family=\'system-ui, -apple-system, "Segoe UI", sans-serif\' role="img" aria-labelledby="t d">',
           '<title id="t">Three elements stored in 49 shared channels, and an absent element</title>',
           '<desc id="d">Left: a 7 by 7 grid of channels. Each of three stored elements, a, b and c, turns on one channel '
           'in every column, along its own polynomial; some channels are shared. Right: element d, which is not stored, '
           'has 6 of its 7 channels on because of the overlaps, but its channel in one column is off, so d reads as '
           'absent.</desc>',
           f'<rect width="{W}" height="{H}" rx="10" fill="{surface}"/>']

    def grid(x0, filled):
        for t in range(Q):
            for v in range(Q):
                x, y = x0 + t * cell, top + (Q - 1 - v) * cell
                fill = faint if (t, v) in filled else surface
                out.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="4" '
                           f'fill="{fill}" stroke="{line}" stroke-width="1"/>')
        for t in range(Q):
            out.append(f'<text x="{x0 + t * cell + cell / 2}" y="{top + Q * cell + 18}" font-size="12" '
                       f'text-anchor="middle" fill="{muted}">{t}</text>')
        for v in range(Q):
            out.append(f'<text x="{x0 - 10}" y="{top + (Q - 1 - v) * cell + cell / 2 + 4}" font-size="12" '
                       f'text-anchor="end" fill="{muted}">{v}</text>')
        out.append(f'<text x="{x0 + Q * cell / 2}" y="{top + Q * cell + 38}" font-size="12" text-anchor="middle" '
                   f'fill="{ink2}">position t</text>')
        out.append(f'<text x="{x0 - 38}" y="{top + Q * cell / 2}" font-size="12" text-anchor="middle" fill="{ink2}" '
                   f'transform="rotate(-90 {x0 - 38} {top + Q * cell / 2})">value p(t)</text>')

    # panel 1: the stored elements
    x0 = left[0]
    out.append(f'<text x="{x0 - 46}" y="34" font-size="15" font-weight="600" fill="{ink}">1. Store a, b and c</text>')
    out.append(f'<text x="{x0 - 46}" y="54" font-size="12.5" fill="{ink2}">each element turns on one channel per '
               f'column: 7 of the 49</text>')
    grid(x0, on)
    offsets = [-9, 0, 9]
    for k, x in enumerate(stored):
        pts = [(x0 + t * cell + cell / 2 + offsets[k], top + (Q - 1 - v) * cell + cell / 2) for t, v in channels(x)]
        out.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in pts)}" fill="none" '
                   f'stroke="{colors[k]}" stroke-width="2" stroke-opacity="0.55"/>')
        for px, py in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colors[k]}" stroke="{surface}" '
                       f'stroke-width="2"/>')
    for k in range(3):                                  # legend, with direct labels
        lx = x0 + k * 72
        out.append(f'<circle cx="{lx + 6}" cy="{top - 26}" r="5" fill="{colors[k]}"/>')
        out.append(f'<text x="{lx + 16}" y="{top - 22}" font-size="13" fill="{ink}">{names[k]}</text>')
    shared = sum(1 for ch in on if sum(ch in channels(x) for x in stored) > 1)
    out.append(f'<text x="{x0 + 220}" y="{top - 22}" font-size="12" fill="{ink2}">{shared} shared channels</text>')

    # panel 2: the absent element
    x0 = left[1]
    out.append(f'<text x="{x0 - 46}" y="34" font-size="15" font-weight="600" fill="{ink}">2. Ask whether d is stored'
               f'</text>')
    out.append(f'<text x="{x0 - 46}" y="54" font-size="12.5" fill="{ink2}">d is stored only if all 7 of its channels '
               f'are on</text>')
    grid(x0, on)
    off = None
    for t, v in channels(d):
        x, y = x0 + t * cell, top + (Q - 1 - v) * cell
        if (t, v) in on:
            out.append(f'<rect x="{x + 7}" y="{y + 7}" width="{cell - 14}" height="{cell - 14}" rx="3" fill="none" '
                       f'stroke="{ink}" stroke-width="2"/>')
        else:
            off = (x, y)
            out.append(f'<rect x="{x + 4}" y="{y + 4}" width="{cell - 8}" height="{cell - 8}" rx="4" fill="none" '
                       f'stroke="{ink}" stroke-width="2.5" stroke-dasharray="4 3"/>')
    ox, oy = off
    out.append(f'<text x="{ox + cell / 2}" y="{oy + cell / 2 + 4}" font-size="11" text-anchor="middle" '
               f'font-weight="600" fill="{ink}">off</text>')
    out.append(f'<rect x="{x0}" y="{top - 34}" width="14" height="14" rx="3" fill="{faint}" stroke="{line}"/>')
    out.append(f'<text x="{x0 + 22}" y="{top - 22}" font-size="13" fill="{ink}">on (set by a, b, c)</text>')
    out.append(f'<rect x="{x0 + 160}" y="{top - 34}" width="14" height="14" rx="3" fill="none" stroke="{ink}" '
               f'stroke-width="2"/>')
    out.append(f'<text x="{x0 + 182}" y="{top - 22}" font-size="13" fill="{ink}">d\'s channels: 6 on, 1 off</text>')
    out.append(f'<text x="24" y="{H - 16}" font-size="11.5" fill="{muted}">Computed from the code pipec uses: '
               f'Kautz-Singleton over GF(7) (docs/make_superposition_figure.py).</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    stored, d = pick()
    check(stored, d)
    path = os.path.join(ROOT, "docs", "superposition_dark.svg")
    open(path, "w").write(svg(stored, d))
    print("wrote", path, "| stored", stored, "| absent", d, "| checks passed")
