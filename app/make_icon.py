# =============================================================================
# app/make_icon.py — Icône de l'application : quelques trajectoires simulées
# en bleu pâle et la médiane en bleu foncé, comme sur les fiches.
#     python -m app.make_icon chemin.png        (image 1024 x 1024)
# =============================================================================

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.patches import FancyBboxPatch            # noqa: E402

PALE, DARK, PAPER = "#8fb8e8", "#0b2a66", "#ffffff"


def make_icon(path):
    rng = np.random.default_rng(7)
    steps = 24
    paths = np.cumsum(rng.normal(0.035, 0.11, (60, steps)), axis=1)
    paths = np.column_stack([np.zeros(60), paths])
    median = paths[np.argmin(np.abs(paths[:, -1] - np.median(paths[:, -1])))]

    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Carré arrondi blanc (format des icônes macOS), avec un fin liseré
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=PAPER, edgecolor="#d5e3f5", linewidth=6))
    inner = fig.add_axes([0.2, 0.22, 0.6, 0.56])
    inner.axis("off")
    x = np.arange(steps + 1)
    for p in paths:
        inner.plot(x, p, color=PALE, linewidth=2.2, alpha=0.55, solid_capstyle="round")
    inner.plot(x, median, color=DARK, linewidth=9, solid_capstyle="round", solid_joinstyle="round")
    inner.plot([0], [0], "o", color=DARK, markersize=26)
    fig.savefig(path, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    make_icon(sys.argv[1])
