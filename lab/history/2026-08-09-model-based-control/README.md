# Model-based control: exploit dynamics, then break the model

Status: resolved on 2026-08-09.

Question: What does knowing the robot dynamics buy, and what breaks when that knowledge is wrong?

## Learning progression

The experiment started with a narrower gravity-compensation question and advanced only after that comparison was resolved:

1. `pd` — joint-space feedback only.
2. `gravity-comp` — the same feedback controller plus model-predicted gravity torque.
3. `computed-torque` — the same feedback controller plus inverse dynamics evaluated at desired position, velocity, and acceleration.
4. `computed-torque-wrong-mass` — the same computed-torque controller, but its private model assumes a 0.35 kg link-2 mass while the simulated plant remains 0.7 kg.

The plant, initial pose, target, PD gains, and timestep stay fixed across conditions. The wrong-mass condition changes only the controller model.

## Predictions recorded before execution

The initial prediction was that gravity compensation would reduce the burden on feedback. After that was supported by telemetry, the experiment advanced to the prediction that accurate inverse dynamics would reduce the remaining tracking residual, while an underestimated controller mass would under-supply feedforward torque and restore a feedback correction.

See `agent-log.md` for the full Q/R/E/A/O record and source provenance.

## Run visually

```powershell
pythonw model_based_control.py --controller pd
pythonw model_based_control.py --controller gravity-comp
pythonw model_based_control.py --controller computed-torque
pythonw model_based_control.py --controller computed-torque-wrong-mass
```

The controller colors are blue for PD, orange for gravity compensation, green for accurate computed torque, and red for computed torque with the wrong controller mass.

For a visualization-only comparison of PD and gravity compensation, run:

```powershell
pythonw model_based_control.py --controller overlay
```

The overlay advances controllers in separate `MjData` objects. It is an inspection aid, not experiment evidence. Its rolling torque, rate, and acceleration graphs use the repository's canonical responsive viewer telemetry stack.

## Run telemetry

```powershell
python model_based_control.py --controller pd --headless --duration 16
python model_based_control.py --controller gravity-comp --headless --duration 16
python model_based_control.py --controller computed-torque --headless --duration 16
python model_based_control.py --controller computed-torque-wrong-mass --headless --duration 16
```

The script reports target state, actual state, tracking error, and decomposed torque terms:

- `tau_fb` — PD feedback torque.
- `tau_g` — model-predicted gravity torque.
- `tau_ff` — model-based feedforward torque.
- `tau_total` — actuator torque actually applied.

The headless summary also reports final-cycle tracking error, feedback effort, acceleration error, turnaround-window error, and target-frequency phase offset.

## Result

Over the final full target cycle:

- Gravity compensation reduced RMS position error from `[0.679127, 0.082077]` rad to `[0.019344, 0.004964]` rad. The gravity-shaped burden moved from feedback into an explicit model term; total applied torque did not disappear.
- Accurate computed torque reduced RMS position error to `[0.000414, 0.000519]` rad and feedback-torque RMS to `[0.000536, 0.000149]` N-m.
- With only the controller mass wrong, RMS position error rose to `[0.268017, 0.053296]` rad and feedback-torque RMS rose to `[4.826850, 0.962259]` N-m.
- At a target reversal, the wrong model supplied `3.644697` N-m too little joint-1 feedforward torque and `1.166463` N-m too little joint-2 feedforward torque.

### Measurement interpretation note

The position-error vectors above are per-joint RMS values `[joint 1, joint 2]`, not confidence intervals. This experiment also has no sensor-noise or state-estimation model: the controller reads MuJoCo state directly.

The current headless loop computes position error after `mj_step()` against the desired position evaluated before that step. With a 2 ms timestep, that timing convention is on the same scale as the reported matched-model residual. Therefore `[0.000414, 0.000519]` rad should not be interpreted as a physical sensing-precision limit or as a confidence bound.

This narrows only the interpretation of the tiny matched-model residual. The main comparison is unchanged: accurate model feedforward leaves almost no feedback burden, while the controller-only mass error restores a large structured residual.

## Model update

Higher feedback gain reacts after error. Gravity compensation supplies pose-dependent support before that error must grow. Computed torque also includes desired acceleration and motion-dependent terms, so it can supply torque implied by the planned motion rather than waiting for tracking error.

A wrong model creates a structured feedforward shortfall and restores a feedback residual. That is different from merely poor gain tuning.

## Stop boundary

The experiment answered its question. A later corrective-acceleration or overshoot event was predicted but not isolated as a time-resolved trace. Do not extend this experiment only to chase that event.

## Next node

Move to issue #5, trajectory tracking, when starting the next experiment. Carry forward the distinction among desired position, velocity, and acceleration, and treat the next test as a rollout through time.

Related: #4
