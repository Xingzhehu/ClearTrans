# Method correspondence and reconstruction choices

Source of specification: the revised manuscript and supplementary material, especially
`sections/method_4dots_sftrl.tex`, `experiment.tex`, `appendix_implementation.tex`,
and `appendix_prompt.tex`. Existing server code was inspected for its structure,
including `sft_train/eval_cot.py` and the related code-translation reward modules.
The implementation here is newly
written; it does not copy embedded API credentials or reuse the MI reward.

## Explicit manuscript settings

| Setting | Value | Implementation |
|---|---|---|
| Languages | C, C++, Go, Java, Python | schema, Docker executor, Sonar adapter |
| Anchors | original source, clean source, clean target | AnchorSynthesizer |
| Anchor acceptance | all tests pass and no detected smells | synthesis.py |
| Teacher | Qwen2.5-32B, non-reasoning | compatible chat service |
| Candidate pipelines/source | 15 | synthesis.n_candidates |
| Rating backbone | student backbone | scoring.model |
| Rating | mean log p(cot given source) + mean log p(target given source and cot) | scoring.py |
| Selected responses/source | 3 | select; SFT fixed-k validation |
| SFT scale/direction | 10,000 sources | data.sft_per_direction |
| RL scale/direction | 2,000 sources | data.rl_per_direction |
| SFT epochs/lr | 3 / 1e-5 | sft config |
| RL epochs/lr | 10 / 1e-6 | grpo config |
| GRPO candidates/group | 12 | grpo.group_size |
| Reward alpha, epsilon | 0.2, 1e-6 | rewards.py |
| Inference temperature | 0.7 | inference config |
| Primary analyzer | SonarQube Server 2025.1 LTA | fixed manifest and CODE_SMELL query |
| Additional analyzers | PMD 7.17.0, Pylint 3.3.8 | independent evaluation adapters |
| Human smell assessment | five assessors, three independent labels/instance, consensus | export/import workflow |
| Reported functional metrics | CA and joint SFA | evaluation.py |

## Explicit choices where the available manuscript is incomplete

These are implementation defaults, not assertions about the historical experiments:

1. TRL 0.22.1 + Transformers 4.56.2 + DeepSpeed ZeRO-3, full-parameter tuning.
   Prior server code used verl; this reconstruction uses TRL to expose the full
   rollout group before evaluating its shared normalization denominator. Core
   reward equations do not depend on that framework choice.
2. Instruct-form compatible teacher endpoint; exact checkpoint and revision must
   be supplied by the authors. Backbone defaults to Qwen/Qwen2.5-3B; 7B requires
   changing both scoring and SFT model settings.
3. Five maximum feedback refinement rounds; 15 attempts are not guaranteed to be
   15 accepted traces. A rejected attempt is logged; fewer than k accepted traces
   is an error at selection, not an implicit dataset-size reduction.
4. 8,192-token SFT/scoring limit and 4,096-token prompt/completion RL limits;
   generation temperature 0.7 and KL beta 0.04 during RL. The original values are
   not recoverable from the available text. No source-code truncation is allowed.
5. Paper-format source markers are preserved. The task prompt explicitly requests
   the think/clean-source/final-code layout. Scoring includes markup and code
   fences in their respective sequence segments; SFT includes an additional EOS.
   Segments are encoded independently and concatenated identically in scoring
   and SFT, avoiding inconsistent subword boundary slicing.
6. Output comparison defaults to whitespace-token equality; byte-exact mode is
   available. No specialized floating-point or interactive checker is implied.
7. Compiler standards are C11/C++17; toolchains come from an Ubuntu 24.04 image
   built on the reproducer's machine. The image digest is recorded at build time.
   Exact historical compiler versions must be supplied separately. Java packages
   and multi-file build systems require a dedicated benchmark adapter.
8. Problem-first randomized SFT/RL split, exact lexical normalization plus a
   5-token-shingle Jaccard threshold of 0.9 for approximate overlap. Lexical
   near-duplication is not equivalent to semantic clone detection. Input problem
   IDs must be aligned across datasets by the authors, not guessed from integers.
9. Format-invalid candidates receive zero and are excluded from analyzable-target
   normalization. Compile-invalid programs receive no correctness bonus and no
   invented smell count. Compilable wrong-answer programs are analyzed and enter
   the shared maximum. Infrastructure failures abort all ranks synchronously.
10. Sonar way, the explicit Pylint rule list, and the PMD rule list are initial
    configurations, not recovered study settings. Restore and export the original
    configurations before claiming numerical reproduction. No invented mapping
    from rule IDs into the paper's five smell categories is supplied.
11. Human sampling selects the same instance IDs across methods, then randomizes
    program order and assessor assignments. It does not invent a historical
    language-stratified sample or independent annotations. The source-origin task
    is separate and is not implemented as if it were the blinded smell task.
12. Default repair baseline permits three extra calls after initial translation.
    Match the historical stopping budget before numerical comparison. API token
    counters are used verbatim; unknown usage remains null.

## External prerequisites, not included study artifacts

- Actual training/benchmark records, their provenance, licenses and exact splits.
- Original Sonar/PMD/Pylint rules and original rule-to-category mapping.
- Original model weights/revisions, selected candidates and trained checkpoints.
- Original independent human annotations and original source-origin annotations.
- Original experimental outputs, logs and timings.

The repository deliberately contains no generated replacement results for these.

## API references used when writing the adapters

- [TRL 0.22.1 GRPO trainer documentation](https://huggingface.co/docs/trl/v0.22.1/grpo_trainer)
- [TRL 0.22.1 trainer source](https://github.com/huggingface/trl/blob/v0.22.1/trl/trainer/grpo_trainer.py)
- [SonarQube 2025.1 CFamily prerequisites](https://docs.sonarsource.com/sonarqube-server/2025.1/analyzing-source-code/languages/c-family/prerequisites)
- [PMD 7.17.0 rule definitions](https://github.com/pmd/pmd/tree/pmd_releases/7.17.0/pmd-java/src/main/resources/category/java)

Sonar Web API schemas are served by the configured instance at `/web_api`; verify
the frozen server's endpoints when deploying this newly written adapter.
