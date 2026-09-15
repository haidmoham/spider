# walk_fast_200

User-selected n=200 speed checkpoint, preserved before the n=200 to n=300
continuation. The user requested: "checkpoint this ... create a demo with a
speed comparison against the stable baseline, and then train".

See [manifest.json](manifest.json) for exact weights and measurements, and
[the original experiment](experiments/policy-cadence-action-20260915/README.md)
for archived evidence and its pre-selection review. The experiment's historical
unapproved status precedes this named selection.

Mean-policy speed is 0.597616 m/s versus 0.317506 m/s for `walk_stable_100`.
All 24 five-second evaluations finished without a fall. Contact fragmentation
and the slip-proxy limitation remain. Selection does not establish the full
deliberate-stride goal or terrain/push robustness. Preserve the stable control.

Load with `spider.cadence_action_training.CadenceResidualPolicy`.
The saved mean replay is in the experiment ZIP at
`comparison-n00200/n00200/mean-201/`. Its local measured recording is
`telemetry/tuning/20260915-policy-cadence-action-round-01/comparison-n00200/n00200/mean-201`.
