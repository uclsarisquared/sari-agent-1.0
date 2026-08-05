# Validation

Maintained checks and evidence-producing scripts live here. This directory is deliberately named
`validation`, not `tests`: only `tests/` contains deterministic pytest checks.

| Directory | Purpose | Simulator? |
|---|---|:---:|
| `tests/agent/` | Offline unit and regression tests for the live agent | No |
| `tests/mapping/` | Offline unit tests for the mapping pipeline | No |
| `probes/` | Focused diagnostics that expose or replay one behavior | Sometimes |
| `calibration/` | Measurement scripts and fitters used to choose runtime constants | Usually |
| `evals/` | Repeatable comparisons that produce scores or reports | Varies |
| `acceptance/` | Supervised smoke and pass-threshold checks for feature acceptance | Yes |
| `fixtures/` | Checked-in batteries and reference results | No |
| `evidence/` | Selected, checked-in evidence from completed validation runs | No |
| `artifacts/` | Generated output from new runs; gitignored | No |

Run the maintained offline suite from the repository root:

```bash
uv run poe test
```

`sari_bench/tests/` remains package-local by design and outside this validation command; the
benchmark package itself was not moved into this taxonomy.

Run probes, calibrations, evals, and acceptance checks as individual scripts. Their module docstrings
state whether Unity, an OCR service, or model credentials are required. New deterministic assertions
belong in `tests/`; observational debugging belongs in `probes/`; parameter measurement belongs in
`calibration/`; scored comparisons belong in `evals/`; and explicit pass/fail feature criteria belong
in `acceptance/`.
