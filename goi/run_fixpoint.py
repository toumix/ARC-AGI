"""Fit family 3 on every same-size task of a split, and write predictions.

    python -m goi.run_fixpoint training          # resumable, 4 workers
    python -m goi.verify goi/predictions/training.json

Per task, one cell per number of rounds in `ROUNDS` and seed in `SEEDS`,
selected by leave-one-demo-out: a candidate's score is the held-out
demos it reproduces exactly when trained on the others, then its fit
on all demos, then fewer rounds. The two best distinct predictions are
the task's attempts -- the benchmark's pass@2. A task whose output is
not the size of its input is uncovered by this family and gets no
attempt. Results are written per task as they finish, so a run resumes
where it stopped.
"""

import argparse
import json
import multiprocessing
import os
import pathlib
import time

#: One XLA thread per worker: four workers on four cores, no contention.
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false '
                      'intra_op_parallelism_threads=1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from goi import fixpoint, survey  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / 'results'
PREDICTIONS = pathlib.Path(__file__).resolve().parent / 'predictions'
ROUNDS = (1, 4, 12)
SEEDS = (0, 1)
RUN = 'v2'


def curriculum(pairs, seed):
    """The cell fitted at each number of rounds, each from the last."""
    params, out = None, {}
    for rounds in ROUNDS:
        params = fixpoint.fit(pairs, rounds, seed, start=params)
        out[rounds] = params
    return out


def candidates(task):
    """Every (rounds, seed) fitted and scored by leave-one-demo-out."""
    demos = task['train']
    out = []
    for seed in SEEDS:
        fitted = curriculum(demos, seed)
        held = [curriculum(demos[:i] + demos[i + 1:], seed)
                for i in range(len(demos))] if len(demos) > 1 else []
        for rounds in ROUNDS:
            out.append({
                'rounds': rounds, 'seed': seed,
                'held_out': sum(
                    fixpoint.exact(params[rounds], [demo], rounds)
                    for params, demo in zip(held, demos)),
                'fit': fixpoint.exact(fitted[rounds], demos, rounds),
                'grids': fixpoint.predict(
                    fitted[rounds], task['test'], rounds)})
    return sorted(out, key=lambda c: (-c['held_out'], -c['fit'], c['rounds']))


def attempts(ranked, task):
    """The two best distinct predictions per test input."""
    out = []
    for k in range(len(task['test'])):
        tries = []
        for candidate in ranked:
            grid = candidate['grids'][k]
            if grid not in tries:
                tries.append(grid)
            if len(tries) == 2:
                break
        out.append(tries)
    return out


def solve(item):
    split, name = item
    path = RESULTS / RUN / split / f'{name}.json'
    if path.exists():
        return name
    with open(survey.DATA / split / f'{name}.json') as stream:
        task = json.load(stream)
    started = time.time()
    ranked = candidates(task)
    tries = attempts(ranked, task)
    result = {
        'attempts': tries,
        'solved': all(pair['output'] in t
                      for pair, t in zip(task['test'], tries)),
        'candidates': [{k: v for k, v in c.items() if k != 'grids'}
                       for c in ranked],
        'seconds': time.time() - started}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as stream:
        json.dump(result, stream)
    print(f'{name}: {"solved" if result["solved"] else "missed"} '
          f'{[(c["rounds"], c["held_out"], c["fit"]) for c in ranked]} '
          f'{result["seconds"]:.0f}s', flush=True)
    return name


def collect(split):
    """Every finished task of the split as one predictions file."""
    out = {}
    for path in sorted((RESULTS / RUN / split).glob('*.json')):
        with open(path) as stream:
            out[path.stem] = json.load(stream)['attempts']
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS / f'{RUN}-{split}.json', 'w') as stream:
        json.dump({'split': split, 'family': 'fixpoint', 'run': RUN,
                   'attempts': out}, stream)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    arguments = parser.parse_args()
    covered = [(arguments.split, name)
               for name, task in survey.tasks(arguments.split)
               if survey.same_size(task)][:arguments.limit]
    print(f'{len(covered)} same-size tasks', flush=True)
    if arguments.workers == 1:
        for item in covered:
            solve(item)
    else:
        with multiprocessing.Pool(arguments.workers) as pool:
            for _ in pool.imap_unordered(solve, covered):
                pass
    out = collect(arguments.split)
    print(f'{len(out)} tasks written to predictions/{RUN}-{arguments.split}.json')
