"""Run the family-3 sweep on Modal, one container per task.

    modal run goi/modal_run.py --split training            # CPU fan-out
    modal run goi/modal_run.py --split training --gpu      # one T4 per task

Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment (or
`modal token set`). A task is a few hundred Adam steps on a 30 x 30
canvas, so what Modal buys is not one fast device but 262 tasks at
once: the CPU fan-out finishes the split in the time of its slowest
task. The GPU flag is there for the deeper rounds and the larger cells
of stage 2, where a step stops being latency-bound. Results land in
`goi/results/<split>/` exactly as `run_fixpoint` writes them, so
`collect` and `verify` read them the same way.
"""

import json
import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = modal.App('goi-arc')


def image(gpu):
    return (modal.Image.debian_slim(python_version='3.11')
            .pip_install('numpy', 'jax[cuda12]' if gpu else 'jax')
            .add_local_dir(ROOT / 'goi', remote_path='/root/goi',
                           ignore=['results', 'predictions', '__pycache__'])
            .add_local_dir(ROOT / 'data', remote_path='/root/data'))


def solve(split, name):
    sys.path.insert(0, '/root')
    from goi import run_fixpoint, survey
    with open(survey.DATA / split / f'{name}.json') as stream:
        task = json.load(stream)
    ranked = run_fixpoint.candidates(task)
    tries = run_fixpoint.attempts(ranked, task)
    return name, {
        'attempts': tries,
        'solved': all(pair['output'] in t
                      for pair, t in zip(task['test'], tries)),
        'candidates': [{k: v for k, v in c.items() if k != 'grids'}
                       for c in ranked]}


@app.function(image=image(False), cpu=2, timeout=3600)
def solve_cpu(split, name):
    return solve(split, name)


@app.function(image=image(True), gpu='T4', timeout=3600)
def solve_gpu(split, name):
    return solve(split, name)


@app.local_entrypoint()
def main(split: str = 'training', gpu: bool = False, limit: int = 0):
    from goi import run_fixpoint, survey
    done = {p.stem for p in (run_fixpoint.RESULTS / split).glob('*.json')}
    names = [name for name, task in survey.tasks(split)
             if survey.same_size(task) and name not in done]
    names = names[:limit] if limit else names
    print(f'{len(names)} tasks to run, {len(done)} already done')
    function = solve_gpu if gpu else solve_cpu
    (run_fixpoint.RESULTS / split).mkdir(parents=True, exist_ok=True)
    for name, result in function.starmap([(split, n) for n in names]):
        with open(run_fixpoint.RESULTS / split / f'{name}.json', 'w') as f:
            json.dump(result, f)
        print(f'{name}: {"solved" if result["solved"] else "missed"}')
    print(f'{len(run_fixpoint.collect(split))} tasks in predictions/{split}.json')
