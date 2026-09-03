"""The eight symmetries of the square, as relabellings of a task.

A task rotated or reflected is the same task: every demo and the test
input transformed alike, the answer transformed back. Pooling the eight
copies of the demos into one fit is the cheapest training distribution
ARC offers, TRM's thousand augmentations in kind, and it is sound
exactly when the rule is symmetric -- a gravity that falls down is not,
and the pooled cell then contradicts itself, which leave-one-demo-out
sees. The eight predictions of a pooled cell, each undone, vote.
"""

import numpy as np

GROUP = range(8)


def apply(grid, g):
    """Transform `g` of the grid: a flip when `g >= 4`, then `g % 4`
    quarter turns."""
    grid = np.asarray(grid)
    return np.rot90(grid[:, ::-1] if g >= 4 else grid, g % 4)


def undo(grid, g):
    grid = np.rot90(np.asarray(grid), -(g % 4))
    return grid[:, ::-1] if g >= 4 else grid


def vote(grids):
    """The colour most of the grids agree on, pixel by pixel."""
    counts = (np.stack(grids)[..., None] == np.arange(10)).sum(0)
    return np.argmax(counts, -1)
