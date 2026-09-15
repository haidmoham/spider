"""PPO record assembly and diagnostics. Learning equations stay in notebook 04."""
import numpy as np
import torch


def frozen_batch(episodes, critic):
    rows = [row for episode in episodes for row in episode['rows']]
    observations = torch.stack([r['observation'] for r in rows]).detach()
    next_observations = torch.stack([r['next_observation'] for r in rows]).detach()
    end = torch.tensor([r['terminated'] or r['truncated'] for r in rows])
    with torch.no_grad():
        values = critic(observations).squeeze(-1)
        next_values = critic(next_observations).squeeze(-1)
    batch = dict(obs=observations, actions=torch.stack([r['latent'] for r in rows]).detach(),
                 old_logp=torch.stack([r['log_prob'] for r in rows]).detach(),
                 rewards=torch.tensor([r['reward'] for r in rows]), values=values,
                 next_values=next_values, episode_end=end,
                 # Preserve baseline's finite 5-second task: horizon ends rewards.
                 bootstrap_allowed=~end)
    n = len(rows)
    assert batch['obs'].shape == (n, 47) and batch['actions'].shape == (n, 18)
    for key in ('old_logp', 'rewards', 'values', 'next_values', 'episode_end'):
        assert batch[key].shape == (n,), (key, batch[key].shape)
    assert all(not t.requires_grad for t in batch.values())
    return batch


def policy_diagnostics(actor, critic, batch, std, clip_ratio):
    with torch.no_grad():
        distribution = torch.distributions.Normal(actor(batch['obs']), std)
        logp = distribution.log_prob(batch['actions']).sum(-1)
        log_ratio = logp - batch['old_logp']
        ratio = log_ratio.exp()
        predictions = critic(batch['obs']).squeeze(-1)
        target = batch['targets']
        variance = target.var(unbiased=False)
        return dict(approx_kl=float((ratio - 1 - log_ratio).mean()),
                    clip_fraction=float(((ratio - 1).abs() > clip_ratio).float().mean()),
                    value_mse=float((predictions - target).square().mean()),
                    explained_variance=float(1 - (target-predictions).var(unbiased=False)/variance)
                    if variance > 1e-12 else None,
                    latent_entropy=float(distribution.entropy().sum(-1).mean()),
                    ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))


def save_batch(batch, path):
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in batch.items()})
