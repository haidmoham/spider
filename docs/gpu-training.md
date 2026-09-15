# GPU PPO training

Use the RTX 3080 for minibatch actor and critic optimization. MuJoCo physics
and small per-step rollout inference stay on CPU. The trainer copies the
current networks to CPU for each episode and transfers the collected batch
to GPU for optimization. This does not make the simulator GPU-based.

The separate environment preserves the CPU environment used by the controls:

```powershell
uv venv .venv-cuda --python .venv/Scripts/python.exe
uv pip install --python .venv-cuda/Scripts/python.exe -r requirements-test.txt
uv pip install --python .venv-cuda/Scripts/python.exe torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

Resume from a named checkpoint into a new output directory:

```powershell
.venv-cuda/Scripts/python -m spider.cadence_action_training --from-checkpoint artifacts/walk_fast_300/walk_fast_300.pt --device cuda --updates 200 --output telemetry/tuning/20260915-policy-cadence-continuation-round-03
```

Do not rerun that command if the named output already exists. Inspect the
existing process and its completed checkpoints first. `--updates` means
additional updates. This round stops at 500 total updates.

CUDA is the CLI default and fails explicitly when unavailable. Use `--device
cpu` only for an intentional CPU run. Actor/critic weights, Adam history and
the CPU minibatch RNG are restored. Device changes can change floating-point
results; CPU and GPU training are not claimed to be bitwise identical.

The command prints the selected device and GPU name. Configuration and every
checkpoint record the optimization/rollout device split and CUDA build.
Tests verify GPU weights and Adam moments, a real GPU optimizer update, and
loading that checkpoint back on CPU. No speedup is claimed without timing.
