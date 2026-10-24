# EnergyPlus integration

The repository executes an independent five-zone, 24-hour control pilot on the
authors' pinned Small Office case. It validates physical actuation and energy
accounting. The original papers' full control tables remain a separate task.

## Install and run

From the repository root, after installing the research/MIQP/plot dependencies:

```bash
python scripts/setup_energyplus.py --install-linux
python scripts/run_public_data.py --seeds 0 1 2 --epochs 250
python scripts/run_prediction_baselines.py
python scripts/run_energyplus_pilot.py --resume
python scripts/run_rc_initialization_ablation.py
python scripts/build_energyplus_report.py
```

The installer supports Linux x86-64. It verifies the official EnergyPlus
**22.1.0-ed759b17ee** archive by SHA-256 and the authors' IDF/EPWs by pinned Git
blob checksums. Other platforms need a matching manual installation; set
`ENERGYPLUS_PATH` to its directory containing `pyenergyplus/` and the executable.
Source downloads and the engine stay outside version control. Their provenance
is recorded in `data/external/energyplus_case/sources.json` and run audits.

The repository includes completed model checkpoints. `--resume` reuses them
only when the protocol, input-model hashes, code and numerical-library versions
match. It then evaluates all test days, reusing matching simulation/plan caches
when present. For an independent fresh run, use
`python scripts/run_energyplus_pilot.py --output outputs/pilot_reproduction`.
Inspect that directory's CSVs; the report builder reads `reports/energyplus`.
Time-limited MIP incumbents can vary with machine load, so bitwise equality of
new fitted models is not promised.

## Runtime and initial conditions

`dfl_hvac.energyplus_runtime.run_energyplus_day` runs each simulation in a fresh
subprocess and releases the EnergyPlus state. It keeps the source HVAC equipment,
sizing settings and original output requests. A derived IDF changes the run
period and adds native SQL/variable outputs; its checksum is saved.

Each day starts with EnergyPlus warmup and **two preceding days at 21 °C**,
followed by the target day. Controllers share the entire preconditioning trace,
actual initial zone temperatures and outdoor weather. These are checked before
metrics are accepted. There is no artificial reset of air temperature while
leaving wall/storage states unchanged. Two days are a declared conditioning
choice, not a demonstrated bound on sensitivity to thermal history.

The five controlled zones are `Core_ZN`, `Perimeter_ZN_1` through
`Perimeter_ZN_4`. The attic remains uncontrolled. Heating and cooling thermostat
actuators are set to the requested hourly trajectory minus/plus 0.05 °C at the
system-timestep callback before prediction. The request is constant during each
hour. API environment type 3 identifies weather simulation; sizing and warmup
are excluded from measurement. Requested setpoints need not equal achieved
temperatures when equipment cannot track them.

## Accounting and independent checks

- Native SQL `Air System Electricity Energy` and `Air System NaturalGas Energy`
  for `PSZ-AC:1` through `PSZ-AC:5` provide gross HVAC consumption. Joules are
  divided by 3,600,000 to obtain kWh. Gas backup is retained explicitly.
- There must be exactly 24 hourly energy observations per system and 96 zone
  temperature/setpoint observations per zone. Missing keys, duplicates, invalid
  values or incorrect units fail validation.
- Recorded API temperatures and commanded setpoints are compared with native SQL
  values. Quarter-hour energy sums must match the native hourly totals within
  numerical tolerance. SQL timestamps mark interval **ends**: 00:15, 00:30,
  00:45 and 01:00 belong to the first hour.
- Comfort violation integrates every quarter hour across five zones. The report
  also retains hourly endpoint discomfort used by the learning loss. These are
  different measures. Hourly kWh is numerically equal to mean kW over that hour.
- Severe/fatal native errors stop the run. Warnings are retained in `audits.json`;
  the unmodified source case itself produces numerous legacy output and sizing
  warnings. Completion does not mean the case is warning-free.

An integration test repeats a constant-setpoint day in separate engine processes,
checks identical outcomes and changes the setpoint to verify a physical response.
Unit fixtures check SQL time boundaries, units and incomplete/duplicate records.
GitHub Actions runs the real-engine test separately from the Python research suite.

## Pilot protocol and limits

`reports/energyplus/protocol.json` records the dates, tariffs, caps, versions and
hashes. Training uses two days, validation two different days, and evaluation
four seasonal days in calendar 2017, with the 2006–2020 typical weather file.
Hourly weather is a perfect forecast for this pilot. The eight controllers share
weather, equipment and initial conditions. The electrical caps are 1.25 times
training-data maxima; they bound the surrogate, not the physical equipment.

Four SS/AS epochs use the same loss-evaluation budget and validation-only model
selection, with the ITO warm start eligible. The neural controller has a ten-second
SCIP budget; feasible incumbents and gaps are retained. Failed plans fall back to
the declared occupied/unoccupied thermostat and receive a training penalty.

Learning minimizes electrical Expost+ with hourly endpoint discomfort and cost
mismatch. Gas cost and subhourly comfort are reported separately and are **not**
optimized by that objective. The tariffs are illustrative; PV, other building
loads, fixed fees and demand charges are excluded. Four days and one fitting seed
do not establish annual savings or a universal DFL advantage.

The authors' full table replication still requires matching their actuation
interpolation, initial-state handling, scenario selection, solver settings and
full training budget. Paper 1 additionally needs its exact modified fifteen-zone
building and unresolved model parameters. See [the remaining gates](REPRODUCTION_PLAN.md).

## Historical-data workflow

The separate `dfl_hvac.energyplus.EnergyPlusCLI` wrapper launches ordinary
historical simulations with `energyplus -w WEATHER.epw -d OUTPUT -r MODEL.idf`.
The public identification study reads the authors' recorded hourly CSVs. Neither
that wrapper nor those records alone constitute physical control validation;
the runtime pilot supplies that additional evidence.
