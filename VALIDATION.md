# Validation record

Implementation prepared on 2026-11-01 from the available manuscript and inspected
related server code. It is a newly reconstructed implementation.

## Completed locally

- Python commands ran in Conda environment `coder`.
- 35 standard-library unittest cases passed.
- Python compilation check passed for package, Docker runner, helper scripts and tests.
- Canonical example data passed schema/ID validation (two instances).
- Tests cover exact reward equations and gates, shared candidate-group denominator,
  rejection of partial/mixed-source groups, code/trace format handling, lexical
  deduplication, top-k/ablation selection, output provenance mismatch, CA/SFA
  denominators, evaluator failures, measured/unknown token counts, human blinding
  and consensus, and mocked Sonar asynchronous-task/pagination/coverage contracts.

The local `coder` environment has CPU PyTorch 2.12.1 and Transformers 4.39.1;
it is used for core checks, not claimed to be the pinned GPU-training environment.

## Deployment checks

Completed on the deployment L20 server:

- All 37 files listed in `SHA256SUMS` matched their SHA256 checksums.
- All eight shell scripts passed individual `bash -n` checks.
- Fourteen package Python source files were present.
- Existing neighboring projects were not edited.

The server deployment is source-only. File inventory is also recorded in
`SOURCE_MANIFEST.json`; rerun shell checks with the per-file loop in
`scripts/check.sh` after activating the required environment.

## Not executed during this task

- Building the execution Docker image or executing benchmark programs in it.
- Live SonarQube scans or PMD/Pylint/LLM evaluation.
- Downloading model weights, installing the pinned GPU environment or starting inference servers.
- SFT, GRPO, multi-GPU training, or reproduction of any reported numerical result.

The mocked Sonar contract tests validate client logic, not compatibility with an
unavailable running server. The TRL adapter was checked against the official
0.22.1 source interface; its distributed/GPU path still needs an actual smoke run
after the required environment and data are configured.
