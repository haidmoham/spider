"""Evaluate a trusted local notebook-04 checkpoint; never train or change its learner.

Run: python -m lab.heldout_reinforce RUN_DIRECTORY --updates 20
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from lab.reinforce_inspection import collect, save_episode, summary
from spider.recording import launch_replay_grid


def evaluate(run, updates=20):
    run = Path(run).resolve()
    checkpoints = list(run.glob(f'block-*/checkpoint-{updates:05d}.pt'))
    if len(checkpoints) != 1:
        raise ValueError('Expected one checkpoint for this update count')
    checkpoint = checkpoints[0]
    block = checkpoint.parent
    nb = json.loads((block / 'notebook.ipynb').read_text(encoding='utf-8'))
    sources = {c.get('id'): ''.join(c['source']) for c in nb['cells']}
    functions = json.loads((block / 'learning-functions.json').read_text())
    if any('Source unavailable' in s for s in functions.values()):
        raise ValueError('Live learning-function provenance is required')
    from IPython.display import display
    ns = {"display": display}
    # Only established setup and the saved actor construction; no training cell.
    for name in ('mdp-setup', 'carried-geometry-helper', 'carried-geometry-settings',
                 'carried-gait-policies'):
        exec(compile(sources[name], f'<baseline:{name}>', 'exec'), ns)
    init = sources['rf4-initialize'].split('optimizer = torch.optim.Adam', 1)[0]
    exec(compile(init, '<baseline:actor>', 'exec'), ns)
    for name, source in functions.items():
        exec(compile(source, f'<baseline:{name}>', 'exec'), ns)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    settings = saved['settings']
    before, after = ns['actor'], copy.deepcopy(ns['actor'])
    before.load_state_dict(torch.load(run / 'initial.pt', map_location='cpu', weights_only=True)['actor'])
    after.load_state_dict(saved['actor'])
    before.eval(); after.eval()
    seeds = list(range(201, 213))
    training = pd.read_csv(run / 'training.csv')
    previous = set(settings['eval_seeds']) | set(training.seed)
    assert not previous.intersection(seeds)
    output = run / f'evaluation-{updates:05d}-{datetime.now(timezone.utc).strftime("%H%M%S%f")}'
    output.mkdir()
    frozen = output / 'baseline'
    frozen.mkdir()
    inputs = [checkpoint, run / 'initial.pt', block / 'notebook.ipynb',
              block / 'learning-functions.json', block / 'settings.json', Path(__file__),
              Path('lab/reinforce_inspection.py')]
    hashes = {}
    for path in inputs:
        shutil.copy2(path, frozen / path.name)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'receipt.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), updates=updates, seeds=seeds, input_sha256=hashes,
        training_updates_executed=0, reset='unchanged neutral reset',
        seed_scope='action sampling only; one trained checkpoint',
        replay_seed=seeds[0], replay_selection='first predeclared seed, not best result',
        settings=settings), indent=2))
    episodes = []
    def add(label, network, seed, deterministic=False, control=None):
        e = collect(network, ns['observation_from'], ns['sample_action'], ns['reward_terms'],
                    dict(settings, deterministic=deterministic), seed=seed, label=label, control=control)
        save_episode(e, output / f'{len(episodes):02d}')
        episodes.append(e)
        print(label, seed, flush=True)
        return e
    for seed in seeds:
        add('before stochastic', before, seed)
        add('after stochastic', after, seed)
    add('before mean', before, seeds[0], True)
    add('after mean', after, seeds[0], True)
    add('neutral control', None, seeds[0], control=lambda state: np.zeros(18))
    add('fixed shuffle', None, seeds[0], control=ns['geometry_gait'])
    records = []
    for e in episodes:
        t = e['log']
        records.append(dict(**summary(e),
            **{k: float(t[k].sum()) for k in t if k.startswith('reward/')},
            fraction_backward=float((t.dx_m < 0).mean()),
            mean_foot_contacts=float(t.contacts.mean())))
    table = pd.DataFrame(records)
    table.to_csv(output / 'comparison.csv', index=False)
    columns = ['forward_m', 'sideways_m', 'total_reward', 'reward/progress', 'reward/height']
    table.groupby('treatment', sort=False)[columns].agg(['mean', 'min', 'max']).to_csv(output / 'ranges.csv')
    # Show the predeclared first pair and controls before inviting interpretation.
    launch_replay_grid([episodes[i]['recording'].replay for i in (0, 1, -2, -1)], output / 'views', speed=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for e in [episodes[i] for i in (0, 1, -2, -1)]:
        t = e['log']; label = e['label']
        axes[0, 0].plot(t.next_t_s, t.dx_m.cumsum(), label=label)
        axes[0, 1].plot(t.next_t_s, t['reward/progress'].cumsum(), label=label)
        axes[1, 0].plot(t.next_t_s, t['reward/height'].cumsum(), label=label)
        axes[1, 1].step(t.next_t_s, t.contacts, where='post', label=label)
    for ax, ylabel in zip(axes.flat, ['World +X displacement (m)', 'Accumulated progress reward',
                                    'Accumulated height reward', 'Measured foot contacts']):
        ax.set(xlabel='Time (s)', ylabel=ylabel); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / 'diagnostic.png', dpi=150)
    plt.close(fig)
    compact = table.groupby('treatment', sort=False)[columns].mean().round(5)
    (output / 'review.md').write_text('# Held-out REINFORCE review\n\n'
        '20-update baseline; no further training. Seeds 201–212 change action samples only. '
        'Controls and mean policies are deterministic and evaluated once.\n\n'
        '![Paired seed 201](diagnostic.png)\n\n```text\n' + compact.to_string() + '\n```\n\n'
        'Full per-seed results: comparison.csv. Ranges: ranges.csv.\n\n'
        '## Your diagnosis\n\n'
        '1. Describe the repeated motion in the trained pane, without using its score.\n'
        '2. Which reward term pays for that behavior? Compare with neutral.\n'
        '3. Propose one change or measurement that could disprove your explanation.\n\n'
        'Edit observation_from or reward_terms in notebook 04 only after choosing your hypothesis. '
        'Contact count alone does not establish stepping or distinguish foot sliding.\n', encoding='utf-8')
    print(compact.to_string())
    print('Saved review:', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--updates', type=int, default=20)
    args = parser.parse_args()
    evaluate(args.run, args.updates)
