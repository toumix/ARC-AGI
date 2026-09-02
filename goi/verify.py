"""Score committed predictions with the benchmark's own criterion.

    python -m goi.verify goi/predictions/<split>.json

A prediction file maps a task id to a list of attempts per test input,
each attempt an output grid, at most `ATTEMPTS` of them; a task is solved
when every one of its test inputs has an attempt equal to its output,
cell for cell and shape included -- the benchmark's rule, pass@2 as ARC
Prize scores it. A task missing from the file, or with more attempts
than allowed, counts as unsolved rather than raising, so a partial run
scores honestly. The split is read from the file's own `split` field and
the tasks from `data/`, which is the only thing trusted.

Nothing here imports beyond the standard library, so the score is
re-derivable in any environment from the committed file and the data.
"""

import json
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'
ATTEMPTS = 2


def load(split):
    for path in sorted((DATA / split).glob('*.json')):
        with open(path) as stream:
            yield path.stem, json.load(stream)


def solved(task, attempts):
    """Whether every test input has an exact attempt among at most two."""
    if len(attempts) != len(task['test']):
        return False
    return all(
        len(tries) <= ATTEMPTS and any(grid == pair['output'] for grid in tries)
        for pair, tries in zip(task['test'], attempts))


def score(split, predictions):
    """The solved tasks of a split, given attempts keyed by task id."""
    return sorted(name for name, task in load(split)
                  if solved(task, predictions.get(name, [])))


def report(path):
    with open(path) as stream:
        predictions = json.load(stream)
    split, attempts = predictions['split'], predictions['attempts']
    hits, total = score(split, attempts), sum(1 for _ in load(split))
    print(f'{split}: {len(hits)} / {total} solved, pass@{ATTEMPTS} '
          f'({100 * len(hits) / total:.2f} %), {len(attempts)} attempted')
    return hits


if __name__ == '__main__':
    for path in sys.argv[1:]:
        report(path)
