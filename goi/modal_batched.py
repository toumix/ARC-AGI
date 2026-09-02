"""Run `goi.batched` on Modal: one GPU per chunk of cells.

    modal run goi/modal_batched.py --split training [--ports all]
        [--run v3] [--budget 1024] [--limit N] [--kinds plain,pooled]

The host builds the jobs and their chunks from `data/`, the containers
fit them (`batched.solve`) and return the hard predictions of every
slot, and the host assembles each task's candidates and writes
`goi/results/<run>/<split>/<task>.json` as soon as its chunks are all
back, then `goi/predictions/<run>-<split>.json`, which `goi.verify`
scores like any other run. A task with a result file is skipped, so a
run resumes where it stopped. Needs `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` in the environment.
"""

import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = modal.App('goi-arc-batched')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('numpy', 'jax[cuda12]')
         .add_local_dir(ROOT / 'goi', remote_path='/root/goi',
                        ignore=['results', 'predictions', '__pycache__',
                                'rearc']))


@app.function(image=image, gpu='A10G', timeout=3600)
def solve(arrays, ports, steps):
    sys.path.insert(0, '/root')
    import time
    from goi import batched
    started = time.time()
    preds = batched.solve(arrays, ports, steps)
    return preds, time.time() - started


@app.local_entrypoint()
def main(split: str = 'training', ports: str = 'all', run: str = 'v3',
         budget: int = 1024, limit: int = 0, steps: int = 600,
         kinds: str = 'plain,pooled'):
    from goi import batched
    kinds = tuple(kinds.split(','))
    batched.KINDS = kinds
    write, done = batched.writer(run, split)
    tasks = batched.covered(split, limit, done)
    work = list(batched.chunks(tasks, budget, kinds))
    print(f'{len(tasks)} tasks, {len(done)} already done, '
          f'{len(work)} chunks of {sum(len(b) for _, b in work)} cells')
    results = {}
    for (kind, batch), item in zip(work, solve.map(
            [batched.chunk(batch) for _, batch in work],
            kwargs={'ports': ports, 'steps': steps},
            return_exceptions=True)):
        if isinstance(item, Exception):
            print(f'{kind} chunk of {len(batch)} cells: '
                  f'{type(item).__name__}: {item}')
            continue
        preds, seconds = item
        print(f'{kind} chunk of {len(batch)} cells x '
              f'{max(len(j["slots"]) for j in batch)} slots: {seconds:.0f}s',
              flush=True)
        batched.gather(results, batch, preds)
        batched.assemble(tasks, results, done, write)
    print(f'{len(batched.collect(run, split))} tasks in '
          f'predictions/{run}-{split}.json')
