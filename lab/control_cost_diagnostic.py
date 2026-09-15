"""Unweighted control diagnostics from frozen recordings; no stepping or training.

python -m lab.control_cost_diagnostic EVALUATION_DIRECTORY
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import mujoco
import numpy as np
import pandas as pd

from spider.recording import STATE


def command_quantities(offsets, targets):
    """Sum over joints; first action change is relative to neutral (zero offset)."""
    previous = np.vstack([np.zeros_like(offsets[:1]), offsets[:-1]])
    return dict(offset_sq_rad2=np.square(offsets).sum(axis=1),
                absolute_target_sq_rad2=np.square(targets).sum(axis=1),
                action_change_sq_rad2=np.square(offsets - previous).sum(axis=1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(source):
    source = Path(source).resolve()
    directories = sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())
    if not directories:
        raise ValueError('No saved episode directories')
    inputs = [p for d in directories for p in d.iterdir() if p.is_file()]
    hashes = {str(p): digest(p) for p in inputs}
    output = Path('telemetry/control-cost') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    traces, summaries = [], []
    for directory in directories:
        metadata = json.loads((directory / 'summary.json').read_text())
        model = mujoco.MjModel.from_binary_path(str(directory / 'model.mjb'))
        # The units below apply to this direct, unit-gear hinge-actuated robot only.
        assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        joints = model.actuator_trnid[:, 0]
        assert np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE)
        np.testing.assert_allclose(model.actuator_gear[:, 0], 1)
        np.testing.assert_allclose(model.actuator_gear[:, 1:], 0)
        data = mujoco.MjData(model)
        with np.load(directory / 'transitions.npz') as archive:
            offsets, targets = archive['offsets_rad'], archive['applied_targets_rad']
        with np.load(directory / 'states.npz') as archive:
            states, times = archive['states'], archive['times']
        assert len(states) == len(offsets) + 1
        trace = pd.read_csv(directory / 'transitions.csv')
        np.testing.assert_allclose(times[1:], trace.next_t_s, atol=1e-12)
        quantities = command_quantities(offsets, targets)
        torque_sq, power, abs_power = [], [], []
        for state, target, time in zip(states[1:], targets, times[1:]):
            mujoco.mj_setState(model, data, state, STATE)
            np.testing.assert_allclose(data.ctrl, target, atol=1e-12)
            mujoco.mj_forward(model, data)
            assert data.time == time  # Reconstruct only; never advance the physics.
            tau = data.qfrc_actuator[model.jnt_dofadr[joints]]
            np.testing.assert_allclose(tau, data.actuator_force, atol=1e-10)
            joint_power = tau * data.qvel[model.jnt_dofadr[joints]]
            torque_sq.append(np.square(tau).sum())
            power.append(joint_power.sum())
            abs_power.append(np.abs(joint_power).sum())
        quantities.update(endpoint_torque_sq_nm2=torque_sq,
                          endpoint_signed_power_w=power,
                          endpoint_absolute_power_w=abs_power)
        frame = pd.DataFrame(quantities)
        assert np.isfinite(frame.to_numpy()).all()
        frame.insert(0, 'time_s', times[1:])
        frame['treatment'], frame['seed'] = metadata['treatment'], metadata['seed']
        frame['episode'] = directory.name
        frame['forward_m'] = trace.dx_m.cumsum()
        traces.append(frame)
        for quantity in quantities:
            values = frame[quantity]
            summaries.append(dict(episode=directory.name, treatment=metadata['treatment'],
                                  seed=metadata['seed'], quantity=quantity,
                                  mean=values.mean(), p50=values.quantile(.5),
                                  p95=values.quantile(.95), maximum=values.max(),
                                  sum_samples=values.sum(), forward_m=metadata['forward_m']))
    all_samples = pd.concat(traces, ignore_index=True)
    stats = pd.DataFrame(summaries)
    all_samples.to_csv(output / 'samples.csv', index=False)
    stats.to_csv(output / 'episodes.csv', index=False)
    pooled = stats.groupby(['treatment', 'quantity'], sort=False).agg(
        episode_mean=('mean', 'mean'), minimum_episode_mean=('mean', 'min'),
        maximum_episode_mean=('mean', 'max'), mean_episode_p95=('p95', 'mean'))
    pooled.to_csv(output / 'summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    labels = list(all_samples.treatment.unique())
    for ax, quantity in zip(axes.flat, quantities):
        for label in labels:
            values = np.sort(all_samples.loc[all_samples.treatment == label, quantity])
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
        ax.set(xlabel=quantity, ylabel='Fraction of saved action endpoints')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / 'distributions.png', dpi=150)
    plt.close(fig)
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    (output / 'receipt.json').write_text(json.dumps(dict(
        source=str(source), input_sha256=hashes, source_unchanged=True,
        script_sha256=digest(Path(__file__)), mujoco=mujoco.__version__,
        physics_steps_executed=0, training_updates=0, coefficients_assigned=False,
        force_power_sampling='Recomputed at post-action saved states, every 20 ms; not substep integrals',
        first_action_change='From zero neutral-relative offset',
        distribution_scope='Within-rollout endpoint distribution, not independent statistical samples',
        work='Not estimated: missing 2 ms substep power samples'), indent=2), encoding='utf-8')
    print(pooled.to_string())
    print('Saved:', output.resolve())
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    diagnose(parser.parse_args().evaluation)
