# Third-party policy

Voxera's core source code is Apache-2.0. Third-party code and model weights must be reviewed independently before they are vendored or redistributed.

## Current runtime dependencies

- NumPy — BSD-3-Clause.

No pretrained model weights are committed in the Phase 0 scaffold.

## Integration rule

Every future model or library integration must record:

1. upstream project and exact version or commit;
2. source-code license;
3. model/data license when weights are involved;
4. whether redistribution is permitted;
5. whether attribution or NOTICE text is required.

GPL components must not be copied into the Apache-2.0 core. If a GPL tool is useful for research or data preparation, keep it outside the distributed core and document the boundary explicitly.
