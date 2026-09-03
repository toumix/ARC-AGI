"""Survey ARC-AGI-1 for the circuit families of discopy#703, by rule tables.

    python -m goi.survey            # both splits, every table below

Two questions a wiring family has to answer before any cell is trained,
answered here by counting over the 400 tasks of each split.

**Shapes.** A family declares its output-size rule, so the first count is
how the output size relates to the input size across every pair of a
task, and how often the test grid's shape is one no demonstration had --
the out-of-distribution axis CLRS trains for on purpose and ARC has by
construction.

**Tables.** For the two data-oblivious families -- recolour (one cell
per pixel reading its own colour) and neighbourhood (one cell per pixel
reading the Moore neighbourhood of radius `r`) -- the cell can be
*enumerated* from the demonstrations rather than trained: the table of
every visited row, which is what clrs#1's `rule_table` did on the
recorded CLRS traffic and what the #703 hand-off asks to try first at
three demos. The interface of the cell is what is at stake, so the row
is read in three encodings, from the raw colours to a quotient the way
clrs#1 quotiented real keys through `key1 > key2`:

* `raw`: the colours of the neighbourhood, the output an absolute colour.
* `rel`: the centre's colour, and for every neighbour whether it equals
  the centre, the background or the border; the output either "the
  centre's colour" or an absolute one.
* `count`: the centre's colour, how many neighbours share it and how
  many carry another non-background colour; the output as in `rel`.

Three numbers per encoding and radius:

* `fits`: the table is consistent across the demonstrations, i.e. no
  row asks for two colours. This grows with the radius for free, since
  a wide enough neighbourhood is unique per pixel, so it is not coverage.
* `loo`: leave-one-demo-out -- the table read from the other demos
  predicts every pixel of the held-out one, for every choice; the
  hand-off's selection criterion.
* `solved`: the table read from all demos predicts every test output
  exactly, the benchmark's own criterion; and, as a solver would pick
  it, at the smallest radius that passes leave-one-out.
"""

import collections
import json
import os
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'
RADII = (0, 1, 2, 3)
ENCODINGS = ('raw', 'rel', 'count')
BACKGROUND, BORDER = 0, -1

#: The split that is frozen: read only to score a claim, never while a
#: design decision is open. `EVALUATION.md` is the ledger of every score.
FROZEN = 'evaluation'

#: The environment variable that unseals it, naming the claim being scored.
UNSEAL = 'GOI_UNSEAL'

SEALED = f"""The {FROZEN} split is frozen. ARC's own README says not to \
leak it into an algorithm "by repeatedly modifying an algorithm while \
using its evaluation score as feedback", which is what the first three \
rounds did. Design against `training` and re-arc; when a claim is ready \
to be scored, set {UNSEAL} to the claim it is for and add the score to \
goi/EVALUATION.md, so that the reads stay counted."""


def unsealed():
    """The claim the frozen split is being scored for, or nothing."""
    return os.environ.get(UNSEAL)


def tasks(split):
    """The tasks of a split, refusing the frozen one unless unsealed."""
    if split == FROZEN and not unsealed():
        raise RuntimeError(SEALED)
    for path in sorted((DATA / split).glob('*.json')):
        with open(path) as stream:
            yield path.stem, json.load(stream)


def shape(grid):
    return len(grid), len(grid[0])


def size_rule(task):
    """How the output size relates to the input size, over every pair."""
    pairs = task['train'] + task['test']
    inputs = [shape(p['input']) for p in pairs]
    outputs = [shape(p['output']) for p in pairs]
    if inputs == outputs:
        return 'same as the input'
    if len(set(outputs)) == 1:
        return 'constant'
    if all(o[0] % i[0] == 0 and o[1] % i[1] == 0
           for i, o in zip(inputs, outputs)):
        return 'a multiple of the input'
    if all(o[0] <= i[0] and o[1] <= i[1] for i, o in zip(inputs, outputs)):
        return 'smaller, varying'
    return 'other'


def shapes(split):
    counts = collections.Counter()
    for _, task in tasks(split):
        counts['tasks'] += 1
        counts[f'output size {size_rule(task)}'] += 1
        counts[f'{min(len(task["train"]), 5)}{"+" if len(task["train"]) >= 5 else ""} demos'] += 1
        demo_shapes = {shape(p['input']) for p in task['train']}
        if any(shape(p['input']) not in demo_shapes for p in task['test']):
            counts['test input of a shape no demo had'] += 1
    return counts


def neighbourhood(grid, i, j, radius):
    height, width = shape(grid)
    return [grid[a][b] if 0 <= a < height and 0 <= b < width else BORDER
            for a in range(i - radius, i + radius + 1)
            for b in range(j - radius, j + radius + 1)]


def row(grid, i, j, radius, encoding):
    """The cell's input row at one pixel, in one encoding."""
    centre, around = grid[i][j], neighbourhood(grid, i, j, radius)
    if encoding == 'raw':
        return (centre, *around)
    if encoding == 'rel':
        return (centre, *((c == centre, c == BACKGROUND, c == BORDER)
                          for c in around))
    return (centre, sum(c == centre for c in around),
            sum(c not in (centre, BACKGROUND, BORDER) for c in around))


def visits(pair, radius, encoding):
    """The cell's boundary traffic on one pair: (row, centre, output)."""
    grid, output = pair['input'], pair['output']
    height, width = shape(grid)
    for i in range(height):
        for j in range(width):
            yield (row(grid, i, j, radius, encoding), grid[i][j],
                   output[i][j])


def encode(colour, centre, encoding):
    """The output as the cell writes it: relative to the centre or not."""
    if encoding != 'raw' and colour == centre:
        return 'centre'
    return colour


def decode(code, centre):
    return centre if code == 'centre' else code


def table(pairs, radius, encoding):
    """The enumerated cell, or None when two demos disagree on a row."""
    cell = {}
    for pair in pairs:
        for key, centre, colour in visits(pair, radius, encoding):
            code = encode(colour, centre, encoding)
            if cell.setdefault(key, code) != code:
                return None
    return cell


def predicts(cell, pairs, radius, encoding):
    """Whether the cell reproduces every output pixel of the pairs."""
    return cell is not None and all(
        key in cell and decode(cell[key], centre) == colour
        for pair in pairs for key, centre, colour in visits(
            pair, radius, encoding))


def leave_one_out(demos, radius, encoding):
    return len(demos) > 1 and all(
        predicts(table(demos[:i] + demos[i + 1:], radius, encoding),
                 [demo], radius, encoding)
        for i, demo in enumerate(demos))


def same_size(task):
    return size_rule(task) == 'same as the input'


def tables(split, encoding):
    counts, solved = collections.Counter(), []
    for name, task in tasks(split):
        if not same_size(task):
            continue
        selected = None
        for radius in RADII:
            cell = table(task['train'], radius, encoding)
            if cell is None:
                continue
            counts[radius, 'fits'] += 1
            loo = leave_one_out(task['train'], radius, encoding)
            counts[radius, 'loo'] += loo
            if predicts(cell, task['test'], radius, encoding):
                counts[radius, 'solved'] += 1
                if loo and selected is None:
                    selected, _ = radius, solved.append(name)
    return counts, solved


def report(split):
    counts = shapes(split)
    print(f'## {split}: {counts.pop("tasks")} tasks')
    for key, count in sorted(counts.items()):
        print(f'{count:>4}  {key}')
    for encoding in ENCODINGS:
        counts, solved = tables(split, encoding)
        print(f'\n{encoding}: radius  fits  loo  solved')
        for radius in RADII:
            print(f'{"":>{len(encoding) + 1}} {radius:>6}  '
                  f'{counts[radius, "fits"]:>4}  {counts[radius, "loo"]:>3}  '
                  f'{counts[radius, "solved"]:>6}')
        print(f'{"":>{len(encoding) + 1}} solved at the smallest radius '
              f'passing leave-one-out: {len(solved)} {solved}')
    print()


if __name__ == '__main__':
    for split in sys.argv[1:] or ('training', 'evaluation'):
        report(split)
