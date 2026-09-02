"""Run `goi.rearc` on Modal, one container per same-size training task.

    modal run goi/modal_rearc.py [--limit N]

The image downloads re-arc's archive once; results land in
`goi/results/v2-rearc/training/` and `goi/predictions/v2-rearc-training.json`,
which `goi.verify` scores like any other run.
"""

import json
import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = modal.App('goi-arc-rearc')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('curl', 'unzip')
         .pip_install('numpy', 'jax')
         .run_commands(
             'curl -sL https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip'
             ' -o /tmp/re_arc.zip',
             'mkdir -p /root/goi/rearc && cd /tmp && unzip -q re_arc.zip'
             ' && mv re_arc/tasks /root/goi/rearc/tasks && rm -rf /tmp/re_arc*')
         .add_local_dir(ROOT / 'goi', remote_path='/root/goi',
                        ignore=['results', 'predictions', '__pycache__',
                                'rearc'])
         .add_local_dir(ROOT / 'data', remote_path='/root/data'))


@app.function(image=image, cpu=2, memory=4096, timeout=3600)
def solve(name):
    sys.path.insert(0, '/root')
    from goi import rearc
    try:
        return rearc.solve(name)
    except Exception as error:
        return name, {'error': f'{type(error).__name__}: {error}'}


@app.local_entrypoint()
def main(limit: int = 0):
    from goi import rearc, run_fixpoint, survey
    results = run_fixpoint.RESULTS / rearc.RUN / 'training'
    results.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in results.glob('*.json')}
    names = [name for name, task in survey.tasks('training')
             if survey.same_size(task) and name not in done]
    names = names[:limit] if limit else names
    print(f'{len(names)} tasks to run, {len(done)} already done')
    for name, result in solve.map(names):
        if 'error' in result:
            print(f'{name}: {result["error"]}')
            continue
        with open(results / f'{name}.json', 'w') as f:
            json.dump(result, f)
        print(f'{name}: {"solved" if result["solved"] else "missed"}')
    out = {}
    for path in sorted(results.glob('*.json')):
        with open(path) as f:
            out[path.stem] = json.load(f)['attempts']
    with open(run_fixpoint.PREDICTIONS / f'{rearc.RUN}-training.json', 'w') as f:
        json.dump({'split': 'training', 'family': 'fixpoint',
                   'run': rearc.RUN, 'attempts': out}, f)
    print(f'{len(out)} tasks in predictions/{rearc.RUN}-training.json')
