# Project Checkpoint

Status: locked milestone
Date: 2026-03-29

This project is being paused in a stable state.

What is finished:

- late-layer convergence signal identified and evaluated
- reviewed dev and holdout benchmark workflow completed
- locked external test implemented
- best minimal system established as `late_window_slope + logit_confidence`
- detection scripts reorganized into a shared `detection/` package
- README rewritten for clearer presentation

Current locked result:

- single feature: `late_window_slope`
- best minimal system: `late_window_slope + logit_confidence`
- holdout accuracy: `0.640`
- holdout ROC AUC: `0.703`

If work resumes later, the next software step should be:

- run one robustness check on either a slightly different prompt format or a fresh benchmark slice

This checkpoint exists so the repository can be closed without losing the exact stopping point.
