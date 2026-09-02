"""Family 3 with a training distribution: re-arc's generated examples.

    python -m goi.rearc 00d62c1b            # one task, locally
    modal run goi/modal_rearc.py            # every same-size training task

Two to five demonstrations are not a training distribution, and the
first run said so: the cell fitted them and reproduced none of its
held-out demos on 215 tasks of 262. Michael Hodel's re-arc
(https://github.com/michaelhodel/re-arc) ships a verified generator for
every one of the 400 training tasks, and 1,000 examples of each. Fitting
the same cell on those separates the two questions the few-shot number
mixes: whether the family can express the task at all, and whether it
can be read off three grids. The score stays the benchmark's -- exact
match on the task's own test pair, two attempts -- and the task's own
demonstrations are never trained on, so they are a second held-out set.

Per task: `TRAIN` generated examples for the fit, minibatched, with the
curriculum over the rounds of `run_fixpoint`; `VALID` others select
among the round counts. The evaluation split has no generators, so this
is a training-split measurement only.
"""

import json
import pathlib
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np

from goi import fixpoint, run_fixpoint, survey

ARCHIVE = 'https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip'
TASKS = pathlib.Path(__file__).resolve().parent / 'rearc' / 'tasks'
TRAIN, VALID, BATCH, STEPS = 256, 32, 8, 1500
RUN = 'v2-rearc'


def examples(name, seed=0):
    """A task's generated examples that fit the canvas, shuffled -- and of
    the family's shape: a transpose is same-size on the square grids the
    benchmark drew and not on the rectangles the generator also draws."""
    with open(TASKS / f'{name}.json') as stream:
        pairs = [pair for pair in json.load(stream)
                 if max(len(pair['input']), len(pair['input'][0]))
                 <= fixpoint.SIZE
                 and len(pair['input']) == len(pair['output'])
                 and len(pair['input'][0]) == len(pair['output'][0])]
    random.Random(seed).shuffle(pairs)
    return pairs[:TRAIN], pairs[TRAIN:TRAIN + VALID]


def fit(pairs, rounds, seed, start=None, steps=STEPS, batch=BATCH,
        tail=100):
    """The cell on a stream of minibatches, Polyak-averaged at the end."""
    rng = random.Random(seed)
    params = start or fixpoint.init(jax.random.PRNGKey(seed), 64)
    opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
           for key, value in params.items()}
    mean = None
    for count in range(steps):
        data = fixpoint.pad(rng.sample(pairs, batch))
        params, opt = fixpoint.update(params, opt, data, count, rounds)
        if count >= steps - tail:
            mean = params if mean is None \
                else {key: mean[key] + params[key] for key in params}
    return {key: value / tail for key, value in mean.items()}


def exact(params, pairs, rounds, batch=BATCH):
    """Exact grids over the pairs, in batches of a fixed shape."""
    return sum(fixpoint.exact(params, pairs[i:i + batch], rounds)
               for i in range(0, len(pairs), batch))


def candidates(task, name, seed=0):
    train, valid = examples(name, seed)
    params, out = None, []
    for rounds in run_fixpoint.ROUNDS:
        params = fit(train, rounds, seed, start=params)
        out.append({
            'rounds': rounds, 'seed': seed,
            'valid': exact(params, valid, rounds),
            'demos': fixpoint.exact(params, task['train'], rounds),
            'grids': fixpoint.predict(params, task['test'], rounds)})
    return sorted(out, key=lambda c: (-c['valid'], -c['demos'], c['rounds']))


def solve(name, split='training'):
    with open(survey.DATA / split / f'{name}.json') as stream:
        task = json.load(stream)
    ranked = candidates(task, name)
    tries = run_fixpoint.attempts(ranked, task)
    return name, {
        'attempts': tries,
        'solved': all(pair['output'] in t
                      for pair, t in zip(task['test'], tries)),
        'candidates': [{k: v for k, v in c.items() if k != 'grids'}
                       for c in ranked]}


if __name__ == '__main__':
    for name in sys.argv[1:]:
        _, result = solve(name)
        print(name, 'solved' if result['solved'] else 'missed',
              [(c['rounds'], c['valid'], c['demos'])
               for c in result['candidates']])
