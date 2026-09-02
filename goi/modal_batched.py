"""Run `goi.batched` on Modal: one GPU per chunk of cells, detached.

    modal run --detach goi/modal_batched.py --split training
        [--ports all] [--run v3] [--budget 1024] [--steps 600]
        [--kinds plain,pooled] [--limit N]
    modal volume get goi-arc-results v3/training goi/results/v3/  # fetch
    python -m goi.batched --collect training                      # predictions

An orchestrator container builds the jobs and their chunks from `data/`,
maps the chunks over GPU containers (`batched.solve`), assembles each
task's candidates as soon as its chunks are all back and writes
`<run>/<split>/<task>.json` to the volume, which `goi/results/` mirrors
once fetched. A task with a file on the volume is skipped, so a run
resumes where it stopped, and the sandbox that launched it may die
without stopping it. Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
"""

import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = modal.App('goi-arc-batched')
volume = modal.Volume.from_name('goi-arc-results', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('numpy', 'jax[cuda12]')
         .add_local_dir(ROOT / 'goi', remote_path='/root/goi',
                        ignore=['results', 'predictions', '__pycache__',
                                'rearc'])
         .add_local_dir(ROOT / 'data', remote_path='/root/data'))


@app.function(image=image, gpu='A10G', timeout=7200)
def solve(arrays, ports, steps):
    sys.path.insert(0, '/root')
    import time
    from goi import batched
    started = time.time()
    preds = batched.solve(arrays, ports, steps)
    return preds, time.time() - started


@app.function(image=image, timeout=24 * 3600, volumes={'/results': volume})
def orchestrate(split, ports, run, budget, steps, kinds, limit):
    sys.path.insert(0, '/root')
    import json
    from goi import batched
    batched.KINDS = kinds
    folder = pathlib.Path('/results') / run / split
    folder.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in folder.glob('*.json')}

    def write(name, result):
        with open(folder / f'{name}.json', 'w') as stream:
            json.dump(result, stream)
        volume.commit()
        print(f'{name}: {"solved" if result["solved"] else "missed"}',
              flush=True)

    tasks = batched.covered(split, limit, done)
    work = list(batched.chunks(tasks, budget, kinds))
    print(f'{len(tasks)} tasks, {len(done)} already done, '
          f'{len(work)} chunks of {sum(len(b) for _, b in work)} cells',
          flush=True)
    results = {}
    for (kind, batch), item in zip(work, solve.map(
            [batched.chunk(batch) for _, batch in work],
            kwargs={'ports': ports, 'steps': steps},
            return_exceptions=True)):
        if isinstance(item, Exception):
            print(f'{kind} chunk of {len(batch)} cells: '
                  f'{type(item).__name__}: {item}', flush=True)
            continue
        preds, seconds = item
        print(f'{kind} chunk of {len(batch)} cells x '
              f'{max(len(j["slots"]) for j in batch)} slots: {seconds:.0f}s',
              flush=True)
        batched.gather(results, batch, preds)
        batched.assemble(tasks, results, done, write)
    print(f'{len(done)} tasks done', flush=True)


@app.local_entrypoint()
def main(split: str = 'training', ports: str = 'all', run: str = 'v3',
         budget: int = 1024, steps: int = 600, kinds: str = 'plain,pooled',
         limit: int = 0):
    call = orchestrate.spawn(split, ports, run, budget, steps,
                             tuple(kinds.split(',')), limit)
    print(f'{run} on {split} spawned as {call.object_id}: '
          f'`modal app logs goi-arc-batched` follows it')
