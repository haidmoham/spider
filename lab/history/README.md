# Preserved test-bench records

These files preserve the findings, failed cases, and stable interaction records
from robotics-test-bench at commit
`9cdd22b377f17c65fc41a85ddbef19b2b9cbacc7`.
Their wording describes the repository and route at the time. Relative links
and issue numbers inside these records refer to that original repository.
They do not create a second current queue or override the root instructions.

Start at the [notebook index](../notebooks/README.md) for the current paths.
[LEARNING.md](../../LEARNING.md) selects ongoing work.
[migration.json](migration.json) maps all 47 tracked source files to their
destination or retirement reason. Source hashes identify the pre-migration
bytes; edited notebooks and fixtures are not claimed to have those same hashes.
Original source remains accessible through the repository and commit recorded
in the manifest. No experiment was rerun to create new findings during migration.

The pendulum, coupling, and model-based control scripts feed one dynamics
notebook. The one-link and two-joint velocity maps feed one Jacobian notebook.
The tripod torque comparison joins the fixed-foot support notebook. The leg
workspace and unfinished Ant notebook keep their distinct questions. Ant uses
a separate environment; it is not the current C-1N algorithm implementation.

The workspace notebook now uses an explicitly reconstructed historical model.
See [model provenance](../models/README.md) for its exact commit and the limit
on reproducing the original unversioned model reference.
