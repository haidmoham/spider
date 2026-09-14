"""Robot collection and evidence plumbing for notebook 04, not a trainer.

The notebook owns observation, sampling, rewards, returns, loss, and updates.
Collection retains both sides of every transition and never reuses a gradient graph.
"""

from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np
import pandas as pd
import torch

from spider.learning import LearningSimulation, PolicyRecording
from spider.recording import TreatmentReplay
from spider.simulation import neutral_targets


def collect(network, observe, sample_action, reward_terms, settings, *, seed,
            label, track_grad=False, control=None, env_factory=LearningSimulation):
    """Collect one fresh finite-horizon episode through the canonical adapter.

    sample_action(network, observation, std, deterministic) returns latent/log_prob.
    A fixed tanh maps the latent to 18 neutral-relative radian offsets. The learned
    random variable is the latent, not the clipped actuator command. Deterministic
    evaluation uses the mean. Controls bypass sampling but use identical physics.
    """
    env = env_factory()
    state = env.reset()
    neutral = np.asarray(neutral_targets())
    limit = np.minimum(settings['bound_rad'], np.minimum(
        env.model.actuator_ctrlrange[:, 1] - neutral,
        neutral - env.model.actuator_ctrlrange[:, 0]))
    if np.any(limit < 0):
        raise ValueError('Neutral target outside actuator range')
    dt = settings['physics_steps'] * env.model.opt.timestep
    replay = TreatmentReplay(env.model, f'{label} | seed={seed} | dt={dt:g}s')
    replay.capture(env.data)
    states, rows, trace = [state], [], []
    with torch.random.fork_rng(devices=[]), torch.set_grad_enabled(track_grad):
        torch.manual_seed(seed)
        for i in range(settings['horizon']):
            observation = observe(state)
            if control is None:
                latent, log_prob = sample_action(network, observation, settings['std'],
                                                settings.get('deterministic', False))
                offsets = limit * torch.tanh(latent).detach().cpu().numpy()
            else:
                latent, log_prob = torch.zeros(18), torch.zeros(())
                offsets = np.asarray(control(state), dtype=float)
            after = env.step(offsets, physics_steps=settings['physics_steps'])
            replay.capture(env.data)
            if not np.isfinite(replay.states[-1]).all() or not np.isclose(
                    after.time - state.time, dt, atol=1e-9, rtol=0):
                raise RuntimeError('Invalid simulator state or clock reset; discard episode')
            next_observation = observe(after)
            terms = {key: float(value) for key, value in reward_terms(state, offsets, after).items()}
            reward = sum(terms.values())
            if not np.isfinite(reward) or not torch.isfinite(observation).all() or not torch.isfinite(next_observation).all():
                raise RuntimeError('Non-finite observation or reward')
            if not torch.isfinite(log_prob).all():
                raise RuntimeError('Non-finite log probability')
            terminated = after.torso_position[2] < settings['fall_height_m']
            truncated = i + 1 == settings['horizon']
            applied = env.data.ctrl.copy()
            row = dict(observation=observation.detach().clone(), next_observation=next_observation.detach().clone(),
                       latent=latent.detach().clone(), offsets=offsets.copy(), applied=applied,
                       log_prob=log_prob, reward=reward, terms=terms,
                       terminated=terminated, truncated=truncated, time=state.time)
            rows.append(row)
            trace.append(dict(action_index=i, t_s=state.time, next_t_s=after.time,
                              vx_before_m_s=state.torso_velocity[0], vy_before_m_s=state.torso_velocity[1],
                              vx_after_m_s=after.torso_velocity[0], vy_after_m_s=after.torso_velocity[1],
                              dx_m=after.torso_position[0]-state.torso_position[0],
                              dy_m=after.torso_position[1]-state.torso_position[1], height_m=after.torso_position[2],
                              contacts=len(after.foot_contacts), reward=reward,
                              **{f'reward/{k}': v for k, v in terms.items()},
                              log_prob=float(log_prob.detach()),
                              latent_norm=float(latent.detach().norm()),
                              max_offset_rad=float(np.max(np.abs(offsets))),
                              clipped_targets=int(np.count_nonzero(np.abs(neutral+offsets-applied)>1e-7)),
                              terminated=terminated, truncated=truncated))
            states.append(after)
            state = after
            if terminated or truncated:
                break
    recording = PolicyRecording(replay, states, np.stack([r['offsets'] for r in rows]),
                                np.stack([r['applied'] for r in rows]),
                                np.array([r['time'] for r in rows]), settings['physics_steps'])
    return dict(rows=rows, recording=recording, log=pd.DataFrame(trace), seed=seed, label=label)


def summary(episode):
    states = episode['recording'].measurements
    first, last = states[0], states[-1]
    trace = episode['log']
    return dict(treatment=episode['label'], seed=episode['seed'], seconds=last.time-first.time,
                forward_m=last.torso_position[0]-first.torso_position[0],
                sideways_m=last.torso_position[1]-first.torso_position[1],
                minimum_height_m=min(s.torso_position[2] for s in states),
                total_reward=float(trace.reward.sum()), terminated=bool(trace.terminated.any()),
                truncated=bool(trace.truncated.any()), clipped_targets=int(trace.clipped_targets.sum()))


def save_episode(episode, directory):
    """Save detached evidence; never serialize a live autograd graph."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    recording, rows = episode['recording'], episode['rows']
    episode['log'].to_csv(directory / 'transitions.csv', index=False)
    mujoco.mj_saveModel(recording.replay.model, str(directory / 'model.mjb'))
    np.savez_compressed(directory / 'states.npz', states=np.asarray(recording.replay.states),
                        times=np.asarray(recording.replay.times), label=recording.replay.label, actuator_name='')
    np.savez_compressed(directory / 'transitions.npz',
                        observations=torch.stack([r['observation'] for r in rows]).cpu().numpy(),
                        next_observations=torch.stack([r['next_observation'] for r in rows]).cpu().numpy(),
                        latent_actions=torch.stack([r['latent'] for r in rows]).cpu().numpy(),
                        offsets_rad=recording.offsets, applied_targets_rad=recording.targets,
                        log_probs=np.array([float(r['log_prob'].detach()) for r in rows]),
                        rewards=np.array([r['reward'] for r in rows]),
                        terminated=np.array([r['terminated'] for r in rows]),
                        truncated=np.array([r['truncated'] for r in rows]))
    metadata = summary(episode)
    metadata['model_sha256'] = hashlib.sha256((directory / 'model.mjb').read_bytes()).hexdigest()
    (directory / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def gradient_table(network):
    return pd.DataFrame([dict(parameter=name, weight_norm=float(p.detach().norm()),
                             grad_norm=None if p.grad is None else float(p.grad.norm()))
                         for name, p in network.named_parameters()])


def plot_evaluations(episodes):
    """Motion and score side by side. No inference that more reward means walking."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for episode in episodes:
        trace = episode['log']
        label = episode['label']
        axes[0].plot(trace.next_t_s, trace.dx_m.cumsum(), label=label)
        axes[1].plot(trace.dx_m.cumsum(), trace.dy_m.cumsum(), label=label)
        axes[2].plot(trace.next_t_s, trace.reward.cumsum(), label=label)
    axes[0].set(xlabel='Time (s)', ylabel='World +X displacement (m)')
    axes[1].set(xlabel='World +X displacement (m)', ylabel='World +Y displacement (m)')
    axes[1].axis('equal')
    axes[2].set(xlabel='Time (s)', ylabel='Accumulated draft reward')
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    return fig, axes
