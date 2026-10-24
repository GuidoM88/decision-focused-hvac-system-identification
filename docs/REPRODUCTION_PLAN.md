# Reproduction status and remaining gates

The repository distinguishes **method implementation**, **independent numerical
experiments**, and **matching the original paper tables**. The first two have
executable evidence; the third is not claimed.

| Workstream | Status | Evidence |
|---|---|---|
| Paper 1 RC/convex DFL methods | Implemented and numerically checked | Method mapping, independent solver/gradient tests, synthetic benchmark |
| Paper 2 neural/MIQP/SS, QP and FB methods | Implemented and numerically checked | Activation enumeration, gradient/budget tests, synthetic benchmark |
| Public five-zone identification | Executed, three seeds and three widths | `reports/public_data/`, pinned sources and overlap audit |
| Reachable-state extension | Executed, 54 paired instances / 162 solves | `reports/bounds/`, proof and raw outcomes including timeouts |
| Five-zone EnergyPlus control pilot | Executed, eight controllers × four seasonal days | `reports/energyplus/`, native SQL/API audits, recorded demo |
| Stronger identification baselines | Executed on identical split masks | `reports/prediction_baselines/`, ARX/Ridge and passive RC |
| RC initialization diagnosis | Executed, paired three-seed diagnostic | `reports/rc_initialization/`, warm starts remain eligible |
| Original Paper 1 Tables 1–2 | Not numerically reproduced | Missing material experimental details below |
| Original Paper 2 control tables | Not numerically reproduced | Exact author actuation/initial-state protocol and full budget remain to match |

## Paper 1: material specification gaps

The paper describes a modified fifteen-zone DOE Medium Office model. Before
claiming its table values, establish:

- The exact modified IDF, EnergyPlus version and both Denver weather inputs.
- The free-parameter structure: a fully free 15×15 matrix and four 15-vectors have
  285 entries, whereas the paper reports 265.
- Zonal/floor capacities, grid capacity, demand charge, discomfort schedules and
  representative-day ordering choices that are not fully specified.
- Historical trajectory extraction, day-ahead temperature actuation and initial
  state handling under the actual physical simulator.

The first paper's printed dynamics are implemented as written. Reconstruction
choices are in `EXPERIMENT_SPEC.md`; the five-zone Small Office data from the
second paper do not resolve the fifteen-zone first-paper specification.

## Paper 2: a concrete numerical replication path

The [pinned author repository](https://github.com/PSMRB/dfl_hvac_management/tree/656589444a66819f2338632b6a9931f867f954b0)
contains the building, both weather files, and `ReproducePaperResults.sh`.
It requires EnergyPlus 22.1 and the authors' Gurobi/dependency stack. That script
is a reference entry point, not evidence that this independent implementation
has rerun it successfully.

The independent physical integration is now exercised by
`dfl_hvac.energyplus_runtime`: setpoints, zone temperatures and energy are
cross-checked against native outputs, with matched preconditioning and weather.
Its declared pilot is useful evidence but is not the authors' experiment.

To close the original numerical-replication gate:

1. Compare the validated pilot adapter with the authors' trajectory interpolation
   and day-reset semantics; document or eliminate each difference.
2. Match actuator timing, weather time indexing, warmup, physical initial states,
   units and zone/AHU mappings against the author's execution.
3. Run 24-hour scenarios and the full paper training budgets for all reported
   network sizes and methods; retain seeds, checkpoints, solver gaps and runtime.
4. Compute the paper-version-specific metrics and report absolute/relative
   differences to its tables. Declare any reconstructed settings explicitly.

The historical `EnergyPlusCLI` and the physical `energyplus_runtime` adapter
serve different purposes. [Integration details](ENERGYPLUS_INTEGRATION.md).

## Completed project experiments

- Public data: two EnergyPlus-generated typical meteorological files, immutable
  download checksums, validation-week embargo, train-only normalization, three
  seeds and an exact-input-overlap diagnostic.
- Control: a declared nonlinear synthetic plant, eight-hour horizon, three seeds,
  three training and two validation days, four test days and a +4 °C stress set.
  The small training budget is a portability choice, not a converged paper run.
- Extension: same networks, feasible domains and objectives across static,
  weather-only and recursive bounds. Certified objective comparison is limited
  to jointly certified pairs; timeout incumbents are never treated as optima.
- Physical control: five zones, 24 hours, two training and two validation days,
  four test days, one fitting seed and four learning epochs. Eight controllers,
  electricity/gas accounting, subhourly comfort and native audits are retained.
- Diagnosis: ARX/Ridge and passive RC strengthen the prediction comparison;
  the paired RC study isolates initialization damage from DFL updates.

See [the measured report](../reports/README.md) for outcomes and commands. Do not
convert these studies into claims that the original EnergyPlus tables match.
