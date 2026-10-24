# Architecture and numerical verification

The pipeline fits thermal dynamics from historical transitions, optimizes HVAC
schedules with the fitted model, and evaluates those schedules against an
external system response. Decision-focused learning updates the model using
control outcomes. Historical prediction error and realized operating cost are
evaluated separately.

## Components

| Stage | Modules | Inputs and outputs |
|---|---|---|
| Identification | `public_data`, `training`, `neural`, `linear_baselines` | Historical temperature, HVAC power and weather transitions → fitted dynamics |
| Convex control | `control`, `differentiable` | RC dynamics and scenario → schedule and differentiable optimization layer |
| Neural control | `neural_control`, `bounds` | ReLU dynamics, scenario and bounds → mixed-integer schedule, gap and residual |
| System response | `benchmark_system`, `energyplus_runtime` | Requested temperature trajectory → realized temperatures and consumption |
| Decision-focused updates | `dfl_training`, `neural_dfl`, `smoothing` | Observed control losses → validation-selected model parameters |
| Evaluation | `evaluation`, `scripts/build_report.py`, `scripts/build_energyplus_report.py` | Recorded outcomes → metrics, figures and offline comparison demo |

Normalized neural parameters are stored as JSON. Training and scheduling use the
same normalization and forward equations. The public-data models have eleven
inputs: five zone temperatures, five signed electrical powers and outdoor
temperature. Positive signed power denotes heating; negative power denotes cooling.

## Optimization and learning

The RC controller is a convex program. CvxpyLayers supplies derivatives of the
optimization solution with respect to RC parameters. The observed system response
is detached from backpropagation.

The neural controller encodes the ReLU equations with mixed-integer constraints.
SCIP represents the quadratic comfort objective through an equivalent convex
epigraph. This encoding is exact for the learned dynamics; a time-limited solve
can still return a suboptimal incumbent. Every accepted schedule includes its
solver status, relative gap and forward-network residual.

The QP and fixed-activation baselines differentiate supervised power loss through
a continuous surrogate. Stochastic smoothing instead estimates a gradient from
scalar Expost+ evaluations. The score and antithetic estimators use the same total
evaluation budget. Validation selects checkpoints, and the initial ITO model
remains eligible.

## Evaluation controls

- Standardization is fitted on training rows. A transition embargo prevents
  adjacent training and validation examples from sharing a temperature state.
- The cross-weather evaluation reports both the full test file and the diagnostic
  excluding exact input collisions between weather files.
- EnergyPlus runs in fresh processes with identical warmup and preconditioning
  for paired controllers. Native SQL outputs check API temperatures, actuation,
  energy units and hourly/subhourly reconciliation.
- Electricity and gas consumption are recorded separately. Gas cost and
  quarter-hour comfort are diagnostics; they are not terms in the electrical
  Expost+ training objective.
- Solver timeouts, gaps and fallback outcomes remain in the results. A feasible
  incumbent is not treated as a certified optimum.

## Numerical verification

| Check | Independent reference |
|---|---|
| RC optimizer and derivatives | Separately constructed convex program and central finite differences |
| Neural mixed-integer optimizer | Enumeration of activation patterns with independent convex subproblems |
| Reachable bounds | Mixed-sign interval corners, sampled trajectories and jointly certified optima |
| EnergyPlus adapter | Native SQL/API reconciliation, repeated fresh runs and changed-setpoint response |
| Published metrics | Recalculation from committed hourly and quarter-hour traces |

The relevant tests are in `tests/test_gradient_consistency.py`,
`tests/test_neural_methods.py`, `tests/test_energyplus_runtime.py` and
`tests/test_recorded_energyplus_results.py`.

## Evidence and limits

The [public-data and bounds study](../reports/README.md) reports a 39.1% reduction
in temperature-prediction RMSE against persistence and a 14.2% mean per-instance
reduction in preactivation interval width against weather-only bounds. The
bounds experiment does not demonstrate a runtime improvement.

The [EnergyPlus pilot](../reports/energyplus/README.md) compares eight controllers
on four seasonal days. NN5-SS improves mean electrical Expost+ relative to NN5-ITO
in that run; passive RC has the lowest mean HVAC bill. The paired synthetic RC
diagnostic shows initialization damage and incomplete recovery under its short
DFL training budget.

These experiments verify an independent implementation and its declared protocols.
They do not reproduce the original papers' complete numerical tables. The
[reproduction plan](REPRODUCTION_PLAN.md) records the remaining inputs and
experimental differences. The [extension analysis](ORIGINAL_EXTENSION.md)
distinguishes this project's implementation and evaluation from prior work on
interval reachability, bound tightening and gradient estimators.
