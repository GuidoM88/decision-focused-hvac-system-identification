# Paper 2: independent methodological implementation

Reference: Favaro, Toubeau, Vallée & Dvorkin,
[Decision-Focused Learning for Neural Network-Constrained HVAC Scheduling](https://arxiv.org/abs/2506.19717v2).
The reference version is **v2, 1 December 2025**. Its reported values differ from v1.
The authors' implementation is pinned at
[`6565894`](https://github.com/PSMRB/dfl_hvac_management/tree/656589444a66819f2338632b6a9931f867f954b0).

| Component | Implementation | Verification |
|---|---|---|
| Normalized thermal network | `neural.ReLUDynamics`, `fit_neural_ito` | Portable parameter round trip; held-out public-data experiment |
| HVAC scheduling, Eqs. 1–12, 20 | `neural_control.solve_neural_hvac` | Independent enumeration of every activation pattern on a small instance |
| Exact ReLU constraints, Eqs. 24–30 | SCIP binary formulation | Forward-network residual checked after every solve |
| Interval bounds with fixed weather | `bounds.horizon_bounds(mode="weather")` | Mixed-sign corner checks |
| Gaussian score estimator, Eqs. 17–19 | `smoothing.smoothing_gradient(estimator="score")` | Explicit evaluation accounting; analytic finite-sample identity |
| Sample-wise SS training | `smoothing.fit_smoothing` | Adam, exponential decay, validation-only checkpoints |
| Expost+ loss | `neural_control.expost_metrics` | Energy, comfort and prediction-cost mismatch reported separately |
| Continuous relaxation baseline | `neural_dfl`, method `qp` | CvxpyLayers, finite gradients, bounds differentiated with weights |
| Fixed-activation baseline | `neural_dfl`, method `fb` | Exact MIP first; local convex derivative with its pattern fixed |
| Supervised power loss, Eq. 44 | `neural_dfl.neural_surrogate_loss` | Building absolute error plus mean zonal absolute error |

## Resolved implementation choices

- The text gives conflicting input counts (6 and 11), while Eq. 2 includes HVAC
  actions. The authors' `src/IndoorTemperatureModelling.py` supplies **11 inputs**:
  five temperatures, five **signed electrical powers**, and outdoor temperature.
  This implementation follows that code. Positive power means heating; negative
  power means cooling. This is material: a model without control inputs cannot
  describe the response to HVAC actions.
- The authors convert hourly Wh to average kW and assign the sign using the larger
  of the sensible heating/cooling rates. The public-data adapter preserves this
  convention and the authors' row-t to row-(t+1) alignment.
- Import price must exceed export price. This prevents simultaneous import/export
  arbitrage; a reversed inequality in the paper text is not copied into the code.
- SCIP replaces Gurobi. An epigraph of the convex quadratic comfort cost gives an
  equivalent mixed-integer convex formulation, not a linear approximation.
- Finite temperature domains are declared configuration choices, distinct from
  comfort targets. Recursive tightening preserves this **declared domain**.
  Soft comfort does not imply all network inputs are unbounded.
- At tied peaks, the demand charge is allocated once to the first maximum. The
  sum of hourly cost components therefore equals the electricity bill.
- Expost+ coincides with the HVAC-only formulation when background load and
  generation are zero. With either present, the implementation uses **net grid
  cost mismatch**, a documented generalization, rather than claiming literal Eq. 21.
- The physical response is detached in QP/FB. Their objective is the supervised
  power loss, not the derivative of true realized operating cost. SS instead
  observes the scalar Expost+ signal from the plant.
- The independent public-data study uses training-only standardization, held-out
  validation weeks with a shared-state embargo, and a diagnostic excluding exact
  inputs shared by the two weather files. These are project evaluation choices,
  not claims that the author's original split has been replicated.

## What has actually been reproduced

The reusable computational methods have been independently implemented and
validated. The repository contains a cross-weather-year identification experiment
on the authors' **EnergyPlus-generated** records, a synthetic control benchmark,
and an executed five-zone, 24-hour EnergyPlus pilot. The pilot uses an explicit
independent protocol; the original tables are **not claimed as reproduced**.

The portable control benchmark deliberately has a small, declared horizon and
training budget. Check `reports/benchmark/manifest.json`, not the paper defaults,
for the settings that produced its results. Every fitting seed is retained;
results are not selected by the best test outcome.

For original EnergyPlus numerical replication, the authors' pinned repository
provides the IDF, both weather years and `ReproducePaperResults.sh`. Their runtime
requires EnergyPlus 22.1 and their solver/dependency stack. The independent
`energyplus_runtime` adapter now validates actual setpoint actuation, paired
initial conditions and native energy accounting on that building. Its hourly
constant setpoints and two-day preconditioning are declared project choices.
Matching the authors' actuation/interpolation, initial-state handling, full
training budgets and original table values remains a separate replication task.
See [the integration guide](ENERGYPLUS_INTEGRATION.md) and
[measured pilot](../reports/energyplus/README.md).
