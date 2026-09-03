"""Family 3 as one batch: every task, fold, seed and relabelling of a
split fitted together, `vmap` over the cells, on one device.

    python -m goi.batched training --limit 2 --steps 20      # smoke test
    modal run --detach goi/modal_batched.py --split training  # the real thing
    python -m goi.batched --collect training                  # predictions

A *job* is one cell to fit: a task, a kind, a seed and a fold. The
`plain` kind fits the demos as they are; the `pooled` kind fits their
eight dihedral copies together (`dihedral.py`). A fold holds one demo
out, every copy of it, and fold `-1` holds none out. Every job pads its
pairs onto the same canvas, so the fits differ only in their data and
run as one program: the step of `fixpoint.update` mapped over a leading
axis of cells. What used to be one container per task for ten minutes
is a chunk of a few hundred cells for the same time, and the split is a
handful of chunks.

The selection is `run_fixpoint`'s, with two more readings: the pixel
accuracy of the held-out demos breaks ties between candidates that
reproduce the same number of them whole, and a pooled cell's eight
undone predictions vote, the vote a candidate of its own, scored by the
vote's own leave-one-out.
"""

import argparse
import functools
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from goi import dihedral, fixpoint, run_fixpoint, survey

ROUNDS = (1, 4, 12)
KINDS = ('plain', 'pooled')
SEEDS = {'plain': (0, 1), 'pooled': (0,)}


def slots(task, kind, fold):
    """The pairs of one job: (grid, output or None, weight, role), the
    role naming the demo or test input and the transform it wears."""
    group = dihedral.GROUP if kind == 'pooled' else (0,)
    out = []
    for i, pair in enumerate(task['train']):
        for g in group:
            out.append((dihedral.apply(pair['input'], g),
                        dihedral.apply(pair['output'], g),
                        float(i != fold), ('demo', i, g)))
    for t, pair in enumerate(task['test']):
        for g in group:
            out.append((dihedral.apply(pair['input'], g), None, 0.,
                        ('test', t, g)))
    return out


def jobs(task, kinds=KINDS):
    """Every job of a task, with its slots."""
    return [{'kind': kind, 'seed': seed, 'fold': fold,
             'slots': slots(task, kind, fold)}
            for kind in kinds for seed in SEEDS[kind]
            for fold in range(-1, len(task['train']))]


def chunk(jobs):
    """One chunk's arrays: the jobs' grids side by side on the canvas."""
    width = max(len(job['slots']) for job in jobs)
    size = fixpoint.SIZE
    inputs = np.zeros((len(jobs), width, size, size), np.int8)
    targets = np.zeros_like(inputs)
    shapes = np.zeros((len(jobs), width, 2), np.int32)
    weight = np.zeros((len(jobs), width), np.float32)
    for j, job in enumerate(jobs):
        for k, (grid, output, w, _) in enumerate(job['slots']):
            height, width_ = shapes[j, k] = grid.shape
            inputs[j, k, :height, :width_] = grid
            if output is not None:
                targets[j, k, :height, :width_] = output
            weight[j, k] = w
    return {'inputs': inputs, 'targets': targets, 'shapes': shapes,
            'weight': weight,
            'seeds': np.array([job['seed'] for job in jobs], np.int32)}


def chunks(tasks, budget, kinds=KINDS):
    """Whole tasks' jobs, one kind at a time, in chunks of at most
    `budget` pairs on the canvas."""
    for kind in kinds:
        batch = []
        for name, task in tasks:
            new = jobs(task, (kind,))
            if batch and (len(batch) + len(new)) * max(
                    len(j['slots']) for j in batch + new) > budget:
                yield kind, batch
                batch = []
            batch += [dict(job, name=name) for job in new]
        if batch:
            yield kind, batch


@functools.partial(jax.jit, static_argnames=('rounds', 'ports'))
def update(params, opt, data, count, rounds, ports):
    grads = jax.vmap(lambda p, d: jax.grad(fixpoint.loss)(p, d, rounds, ports))(
        params, data)
    return fixpoint.adam(params, opt, grads, count)


@functools.partial(jax.jit, static_argnames=('rounds', 'ports'))
def predict(params, data, rounds, ports):
    states = jax.vmap(lambda p, d: fixpoint.run(p, d, rounds, True, ports)[-1])(
        params, data)
    return jnp.argmax(states, -1).astype(jnp.int8)


def solve(arrays, ports='all', steps=600, hidden=64, tail=100,
          rounds=ROUNDS, log=print):
    """Fit every cell of the chunk through the curriculum over the
    rounds, and return the hard predictions on every slot at each."""
    data = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[
        fixpoint.arrays(*job) for job in zip(
            arrays['inputs'], arrays['targets'], arrays['shapes'],
            arrays['weight'])])
    params = jax.vmap(lambda s: fixpoint.init(
        jax.random.PRNGKey(s), hidden, ports))(jnp.array(arrays['seeds']))
    out = {}
    for r in rounds:
        opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
               for key, value in params.items()}
        mean = None
        for count in range(steps):
            params, opt = update(params, opt, data, count, r, ports)
            if count >= steps - tail:
                mean = params if mean is None \
                    else {key: mean[key] + params[key] for key in params}
        params = {key: value / tail for key, value in mean.items()}
        out[r] = np.asarray(predict(params, data, r, ports))
        log(f'{len(arrays["seeds"])} cells x {arrays["inputs"].shape[1]} '
            f'slots at {r} rounds: {steps} steps')
    return out


def crop(canvas, grid):
    return canvas[:grid.shape[0], :grid.shape[1]]


def candidates(task, cells):
    """Every (kind, seed, rounds) of one task read off its cells'
    predictions, `(job, {rounds: grids})` each, as `run_fixpoint.candidates`
    records them."""
    out = []
    by = {(job['kind'], job['seed'], job['fold']): (job, preds)
          for job, preds in cells}
    demos = [np.asarray(pair['output']) for pair in task['train']]
    for kind in KINDS:
        for seed in SEEDS[kind]:
            if (kind, seed, -1) not in by:
                continue
            for r in ROUNDS:
                for voting in (False, True) if kind == 'pooled' else (False,):
                    def answer(fold, role):
                        job, preds = by[kind, seed, fold]
                        grids = [dihedral.undo(crop(preds[r][k], grid), g)
                                 for k, (grid, _, _, (what, i, g))
                                 in enumerate(job['slots'])
                                 if (what, i) == role and (voting or g == 0)]
                        return dihedral.vote(grids) if voting else grids[0]
                    held = [(answer(i, ('demo', i)), demos[i])
                            for i in range(len(demos)) if (kind, seed, i) in by]
                    fit = [(answer(-1, ('demo', i)), demos[i])
                           for i in range(len(demos))]
                    out.append({
                        'kind': kind + ('-vote' if voting else ''),
                        'rounds': r, 'seed': seed,
                        'held_out': sum(np.array_equal(a, d) for a, d in held),
                        'held_pixels': float(np.mean(
                            [np.mean(a == d) for a, d in held])) if held else 0.,
                        'fit': sum(np.array_equal(a, d) for a, d in fit),
                        'grids': [answer(-1, ('test', t)).tolist()
                                  for t in range(len(task['test']))]})
    return sorted(out, key=lambda c: (-c['held_out'], -c['held_pixels'],
                                      -c['fit'], c['rounds'], c['kind']))


def record(task, ranked):
    tries = run_fixpoint.attempts(ranked, task)
    return {'attempts': tries,
            'solved': all(pair['output'] in t
                          for pair, t in zip(task['test'], tries)),
            'candidates': [{k: v for k, v in c.items() if k != 'grids'}
                           for c in ranked]}


def gather(results, batch, preds):
    """File each cell's predictions under its task."""
    for k, job in enumerate(batch):
        results.setdefault(job['name'], []).append(
            (job, {r: preds[r][k] for r in ROUNDS}))


def assemble(tasks, results, done, write):
    """Write every task whose cells of every kind came back."""
    for name, task in tasks:
        cells = results.get(name, [])
        if name in done or {job['kind'] for job, _ in cells} != set(KINDS):
            continue
        write(name, record(task, candidates(task, cells)))
        done.add(name)


def collect(run, split):
    """Every finished task of the split as one predictions file."""
    out = {}
    folder = run_fixpoint.RESULTS / run / split
    for path in sorted(folder.glob('*.json')):
        with open(path) as stream:
            out[path.stem] = json.load(stream)['attempts']
    run_fixpoint.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with open(run_fixpoint.PREDICTIONS / f'{run}-{split}.json', 'w') as stream:
        json.dump({'split': split, 'family': 'fixpoint', 'run': run,
                   'attempts': out}, stream)
    return out


def covered(split, limit=0, done=()):
    names = [(name, task) for name, task in survey.tasks(split)
             if survey.same_size(task) and name not in done]
    return names[:limit] if limit else names


def writer(run, split):
    folder = run_fixpoint.RESULTS / run / split
    folder.mkdir(parents=True, exist_ok=True)

    def write(name, result):
        with open(folder / f'{name}.json', 'w') as stream:
            json.dump(result, stream)
        print(f'{name}: {"solved" if result["solved"] else "missed"} '
              f'{[(c["kind"], c["rounds"], c["held_out"], c["fit"]) for c in result["candidates"][:3]]}',
              flush=True)
    return write, {p.stem for p in folder.glob('*.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('--collect', action='store_true',
                        help='only write the predictions file from results')
    parser.add_argument('--run', default='v3')
    parser.add_argument('--ports', default='all')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.collect:
        print(f'{len(collect(args.run, args.split))} tasks in '
              f'predictions/{args.run}-{args.split}.json')
        sys.exit()
    write, done = writer(args.run, args.split)
    tasks = covered(args.split, args.limit, done)
    results = {}
    for kind, batch in chunks(tasks, args.budget):
        preds = solve(chunk(batch), args.ports, args.steps)
        gather(results, batch, preds)
        assemble(tasks, results, done, write)
    print(f'{len(collect(args.run, args.split))} tasks in '
          f'predictions/{args.run}-{args.split}.json')
