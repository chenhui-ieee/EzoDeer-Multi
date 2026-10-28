# Packaging validation

This training-only package was assembled from the already validated EzoDeer-Multi detection and segmentation baselines. Original datasets and completed experiments were not modified, and full training was not rerun during packaging.

Checks performed for this package:

- All Python source files parsed successfully.
- All 11 public command-line entry points displayed help successfully.
- Seven wrapper tests passed, covering detection and segmentation routing, preparation/report skipping, and preservation of existing training runs.
- A distribution wheel was built locally without network access or installing it into the existing environment.
- Final source-file inventory and credential/path pattern scanning are available through `scripts/audit_release.py`; the detailed audit report is kept outside this directory.

The underlying portable baseline code was previously checked against completed experiment checkpoints and saved evaluation results. This packaging check is not a claim of a new training result or bit-for-bit reproduction on different hardware. No trained checkpoints, data, private videos, font files or third-party source trees are bundled.
