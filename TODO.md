# Spider experiment queue

This file is the authoritative ordered queue for Spider experiments.

- The first unchecked experiment is the next experiment.
- Do not skip or reorder experiments automatically.
- Keep completed experiments in this file with a short result note or commit reference.
- Add new ideas after the current next experiment unless the user explicitly changes priority.

## 1. Task-space telemetry: see what the feet are doing

**Status:** NEXT

### Question

Why can the same joint-space hip command produce different forward motion at different legs?

### Learning target

Build working intuition for:

- local, torso, and world coordinate frames;
- forward kinematics as the map from joint configuration to foot position;
- task-space position and velocity;
- the Jacobian as the local relationship between joint motion and foot motion.

Do not implement inverse kinematics yet. The goal is to observe and explain the mapping first.

### Prediction before code

Before running the probe, write down the expected sign and rough relative size of torso-frame foot motion for the front, middle, and rear legs when each receives the same small positive hip-angle change.

The prediction must exist before the measurement.

### Experiment

1. Start each probe from the same standing configuration.
2. Use each foot contact site as the measured foot point.
3. Measure the foot position in world coordinates.
4. Transform that position into the torso coordinate frame.
5. Apply the same small hip-angle perturbation to one front, one middle, and one rear leg separately.
6. Recompute kinematics without advancing a walking episode.
7. Record the torso-frame foot displacement for each probe.
8. Compare the measured displacement with the prediction.
9. Extend the viewer telemetry so that the existing joint-space stack can be compared with torso-frame foot X position, velocity, and acceleration during the current gait.
10. Mark foot contact state on or beside the task-space telemetry so stance and swing are distinguishable.

### Constraints

- Do not tune the gait to make Spider walk better during this experiment.
- Do not add reinforcement learning.
- Do not add inverse kinematics or a task-space controller.
- Do not change the robot geometry to make the result easier.
- Keep the current controller intact except for instrumentation needed by the experiment.

### Done when

The experiment is complete when all of these are true:

- the same joint perturbation has been compared across front, middle, and rear legs;
- foot motion is expressed in the torso frame rather than only in each leg's local frame or the world frame;
- the viewer exposes joint-space motion beside task-space foot motion;
- contact state can be related to the task-space trace;
- the result explains, in plain language, why synchronized joint commands do not imply synchronized forward foot motion;
- the result states what the Jacobian would add next, without implementing it as the solution.

### Expected epistemic change

Before: "The legs share a phase, so similar joint motion should create a coordinated gait."

After: "A joint command only has meaning through the leg geometry and coordinate frame. Coordinated locomotion requires reasoning about the foot motion that those joint changes produce in a shared task frame."
