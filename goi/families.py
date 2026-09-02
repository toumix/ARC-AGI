"""What re-arc's verifiers say a task needs, crossed with what a run solved.

    python -m goi.families v1                # a run under goi/results/

Michael Hodel's re-arc carries a verifier per training task, a program
in his ARC-DSL that reproduces the task's outputs. The DSL functions a
verifier calls are a label of the task's family a hand reading of the
demos could not give: `objects` means the task is about connected
components, `shift`/`move`/`gravitate` that something moves,
`connect`/`shoot` that a line is drawn. `rearc_primitives.json` is that
list per task, read off `verifiers.py` by `ast` (the calls by bare name
inside each `verify_*`), vendored so this runs without the clone.

The table counts, for each family of primitives, the tasks of the
training split that call one, how many of those the same-size families
cover, and how many the run solved -- what the run's family lacks, by
name.
"""

import glob
import json
import pathlib
import sys

from goi import survey

HERE = pathlib.Path(__file__).resolve().parent
FAMILIES = {
    'objects: objects / partition / fgpartition':
        ('objects', 'partition', 'fgpartition'),
    'motion: shift / move / gravitate': ('shift', 'move', 'gravitate'),
    'lines: connect / shoot / frontiers':
        ('connect', 'shoot', 'hfrontier', 'vfrontier'),
    'counting: size / colorcount / argmax / argmin / mostcolor':
        ('size', 'colorcount', 'argmax', 'argmin', 'mostcolor',
         'leastcolor', 'sizefilter'),
    'symmetry: mirrors and rotations':
        ('hmirror', 'vmirror', 'dmirror', 'cmirror', 'rot90', 'rot180',
         'rot270'),
    'fill / recolor / replace / switch, and no objects': None,
}


def primitives():
    with open(HERE / 'rearc_primitives.json') as stream:
        return json.load(stream)


def uses(calls, family):
    names = FAMILIES[family]
    if names is None:
        return any(p in calls for p in ('fill', 'recolor', 'replace', 'switch')) \
            and not uses(calls, next(iter(FAMILIES)))
    return any(p in calls for p in names)


def solved(run):
    out = set()
    for path in glob.glob(str(HERE / 'results' / run / 'training' / '*.json')):
        with open(path) as stream:
            if json.load(stream)['solved']:
                out.add(pathlib.Path(path).stem)
    return out


def report(run):
    calls = primitives()
    covered = {name for name, task in survey.tasks('training')
               if survey.same_size(task)}
    hits = solved(run)
    print(f'training: {len(calls)} verifiers, {len(covered)} same-size '
          f'tasks, {len(hits)} solved by {run}')
    print(f'{"verifier calls":58} {"all":>4} {"covered":>8} {"solved":>7}')
    for family in FAMILIES:
        counts = [sum(uses(calls[n], family) for n in group)
                  for group in (calls, covered, hits)]
        print(f'{family:58} {counts[0]:>4} {counts[1]:>8} {counts[2]:>7}')
    median = lambda group: sorted(len(calls[n]) for n in group)[len(group) // 2]
    print(f'primitives per verifier, median: all {median(calls)}, '
          f'solved {median(hits) if hits else "-"}')


if __name__ == '__main__':
    for run in sys.argv[1:] or ('v1',):
        report(run)
