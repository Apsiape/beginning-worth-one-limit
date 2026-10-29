Finite consistency checks for A beginning is worth one limit
Version 1.0.0, October 2026. A preprint; it has not been peer reviewed.

With Python, NumPy, psutil and threadpoolctl, first verify this pristine
ancillary distribution:
    python verify_hashes.py
The programs write beside themselves. Copy the complete anc/ directory to a
new working directory and run the copy:
    python WORKING_DIRECTORY/run_checks.py
Reruns produce new timing and memory values and are not expected to match the
distribution SHA256SUMS. Preserve the pristine manifest for archive verification.
The runner executes the five programs sequentially with one numerical-library
thread. It guards sampled child-process-tree plus runner RSS below 900 MB.
This is sampled RSS, not an operating-system historical peak. Optional
per-script peak-working-set fields are null when unavailable.

renewal/check_renewal.py:
Exact rational renewal identities and separation constants related to Lemma
4.1 and Proposition 8.2; seeded finite-torus preparation and escape-time
illustrations. The latter do not prove seed universality or infinite phases.

renewal/check_renewal_independent.py:
An independent countdown-chain implementation checks 1,530 exact binary-word
probabilities, including zero-only words and the period-two counterexample.
It checks one-sided blocking, finite cones and separation, unconditional
cloned-seed covariance, and finite disjoint-cone and torus simulations.

fluctuations/check_fluctuations.py:
Four numerical criterion groups check finite conditional-cone transfer,
the tested seed-density grid, finite decoder performance and a preparation
comparison. Five further groups only record coarsening and guarded-rule
observations. They have observation_recorded fields, not success flags for
an infinite theorem. Sitewise row agreement is not whole-row equality.

fluctuations/check_fluctuations_independent.py:
A different finite PCA checks all 168 conditioning patterns and correlated
seed reweighting. Exact arithmetic checks certificate constants through
scale 128 and 1,277 continuous-quantizer plateau cases. Counterexamples cover
fair majority, plurality versus strict majority, seed covariance and an
unrestricted guard assertion. A separate finite guarded-rule diagnostic
does not establish a universal obstruction.

fluctuations/check_modal.py:
Replays the coarsening stream and checks the final modal rows explicitly.
The ordered rule-30 rows are fixed points; the rule-110 comparison is not.
Its cycle search is bounded. No cycle found means only no cycle within that
search, not infinite period.

Every intended numerical criterion raises on failure. A recorded diagnostic
can show a failed proposed computing mechanism without making the program
fail: it is an observation, not a pass criterion. No preregistration claim
is inferred from the final sources.

Limitations:
The finite noise values are not certified as members of the all-time
reliability region. Floating-point stationary calculations are consistency
checks, not exact arithmetic or interval certificates. A finite positive-rate
torus has a unique stationary law; its branches are metastable observations.
None of these runs proves universality, noncomputability, an infinite-volume
phase theorem or the analytic all-scale bounds. The proofs and named primary
imports have those roles.

Recorded JSON and stdout are retained. SHA256SUMS covers every ancillary file
except itself, with LF line endings. Reruns are assessed by their numerical criteria; the distribution manifest
verifies the pristine copy, rather than the new run outputs.
