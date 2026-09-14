# Static support boundary

## Question

For a fixed three-foot support triangle, does forward payload shift make the
signed support margin cross zero before the rear support load reaches zero?

## Initial prediction

The human predicts that support margin reaches zero earlier. They also expect
the morphology to become less stable while moving forward because a gait can
reduce the contact set. This experiment isolates only the first claim. It does
not include a gait, leg reach, or actuator saturation.

## Run

```bash
python static_support_boundary.py --shift 0.00
python static_support_boundary.py --shift 0.30
python static_support_boundary.py --shift 0.60
```

Record the signed support margin, per-foot normal loads, and torso attitude.

## Observed boundary

The centered condition had a `+0.1304 m` support margin. At a `+0.96 m`
payload shift, the margin was `+0.0010 m` and each rear foot carried `0.046 N`.
At `+0.98 m`, both rear contacts had unloaded and the body had tipped, leaving
a `-0.0965 m` final margin. The transition is therefore bracketed between
`+0.96 m` and `+0.98 m`.

This fixed-foot fixture does not show margin exhaustion earlier than limiting
rear-foot unloading. It cannot test C-1N's dynamic gait instability because
the feet and contact set do not change during a rollout.

See [`static_support_boundary.ipynb`](static_support_boundary.ipynb) for the
executed sweep and plot.

## C-1N hook

If this fixed-foot relation is clear, repeat the same measurements for the
C-1N stance-only pose. Only then compare them against a gait phase with a
reduced contact set.
