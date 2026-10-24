"""Build the EnergyPlus report and offline demo from complete measured outputs."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from build_report import table

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
METHODS = ["Fixed-21", "Scheduled", "ARX-Ridge", "Passive-RC", "NN2-ITO", "NN5-ITO", "NN5-SS", "NN5-AS"]


def main():
    output = REPORTS / "energyplus"
    completion = json.loads((output / "completion.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    if completion["protocol_sha256"] != hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest():
        raise ValueError("Protocol checksum changed after completion")
    metrics = pd.read_csv(output / "metrics.csv")
    hourly = pd.read_csv(output / "hourly.csv")
    fine = pd.read_csv(output / "quarter_hour.csv")
    expected = {(day, method) for day in protocol["test"] for method in METHODS}
    if set(zip(metrics.day, metrics.method)) != expected or len(metrics) != len(expected):
        raise ValueError("Missing or duplicate declared outcomes")
    if len(hourly) != len(expected)*120 or len(fine) != len(expected)*480:
        raise ValueError("Incomplete hourly/subhourly trajectories")
    summary = metrics.groupby("method").agg(
        electricity_kwh=("electricity_kwh", "mean"), gas_kwh=("gas_kwh", "mean"), bill_eur=("hvac_bill_eur", "mean"),
        comfort=("comfort_zone_degree_hours", "mean"), occupied_comfort=("occupied_comfort_zone_degree_hours", "mean"),
        max_gap=("solver_gap", "max"), solve_seconds=("solve_seconds", "mean"),
        fallback_count=("fallback", "sum"), electrical_expost=("electrical_expost_plus", "mean")).loc[METHODS]
    summary.to_csv(output / "summary.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "hvac-energyplus-pilot"})
    colors = ["#526076", "#b0652d", "#7864aa", "#709364", "#729bba", "#2b69a1", "#087f78", "#bc4f63"]
    figures = REPORTS / "figures"

    def save(fig, name):
        for extension in ("svg", "png"):
            fig.savefig(figures / f"{name}.{extension}", dpi=160, metadata={"Date": None} if extension == "svg" else None)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, column, label in ((axes[0], "bill_eur", "Gross HVAC bill (illustrative € / day)"),
                              (axes[1], "comfort", "Comfort-band violation (zone-degree-hours / day)")):
        ax.barh(METHODS, summary[column], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.18)
        for i, value in enumerate(summary[column]):
            ax.text(value, i, f"  {value:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, summary[column].max()*1.2)
    fig.suptitle("EnergyPlus control pilot · means over four declared seasonal days", fontweight="bold")
    save(fig, "energyplus_overview")
    curves = json.loads((REPORTS / "rc_initialization/learning_curves.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for ax, initialization in zip(axes, ("unperturbed", "perturbed")):
        for row in curves:
            if row["initialization"] == initialization:
                ax.plot(range(-1, len(row["validation_loss"])), [row["initial_validation_loss"], *row["validation_loss"]], marker=".", label=f"seed {row['seed']}")
        ax.set(title=initialization.capitalize(), xlabel="Epoch (−1 = warm start)", ylabel="Validation hierarchical power loss")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("RC initialization diagnostic · same training budget; independent panel scales", fontweight="bold")
    save(fig, "rc_initialization")
    scores = pd.read_csv(REPORTS / "prediction_baselines/metrics.csv")
    neural_scores = pd.read_csv(REPORTS / "public_data/identification.csv")
    baseline_table = table(["Model", "Disjoint-input RMSE (°C)"], [(r.method, f"{r.disjoint_rmse_c:.3f}") for r in scores.itertuples()] + [("NN10 · three-seed mean", f"{neural_scores.loc[neural_scores.hidden_units == 10, 'disjoint_input_rmse_c'].mean():.3f}"), ("Persistence", f"{neural_scores.disjoint_persistence_rmse_c.iloc[0]:.3f}")])
    cost_table = table(["Controller", "Electricity kWh", "Gas kWh", "HVAC €", "Comfort zone-°C·h", "Max solver gap"],
        [(m, f"{r.electricity_kwh:.2f}", f"{r.gas_kwh:.2f}", f"{r.bill_eur:.2f}", f"{r.comfort:.2f}", "—" if pd.isna(r.max_gap) else f"{100*r.max_gap:.1f}%") for m, r in summary.iterrows()])
    neural_rows = metrics[metrics.method.str.startswith("NN")]
    certified = int(neural_rows.solver_status.isin(["optimal", "gaplimit"]).sum())
    audits = json.loads((output / "audits.json").read_text())
    max_audit = max(max(a["native_output_max_errors"].values()) for a in audits)
    fit_logs = {m: json.loads((output / f"{m}_training.json").read_text()) for m in ("NN5-SS", "NN5-AS")}
    fit_table = table(["Method", "Selected epoch", "Initial validation Expost+", "Best validation Expost+", "Train / validation evaluations"],
        [(m, r["best_epoch"], f"{r['initial_validation_loss']:.3f}", f"{min([r['initial_validation_loss'], *r['validation_loss']]):.3f}", f"{r['training_evaluations']} / {r['validation_evaluations']}") for m, r in fit_logs.items()])
    rc = pd.read_csv(REPORTS / "rc_initialization/metrics.csv")
    rc_means = rc[rc.split == "test"].groupby(["initialization", "method"]).expost_plus.mean()
    rollback = sum(r["selected_epoch"] == -1 for r in curves if r["initialization"] == "unperturbed")
    ss_change = 100 * (1 - summary.loc["NN5-SS", "electrical_expost"] / summary.loc["NN5-ITO", "electrical_expost"])
    cheapest = summary.bill_eur.idxmin()
    report = f"""# From learned dynamics to an EnergyPlus building

**Executed pilot:** five controlled zones, 24-hour schedules, eight controllers,
four seasonal test days, EnergyPlus 22.1.0. The authors' pinned building and weather
are used; this does **not** reproduce either paper's complete numerical tables.

[Open the offline demo](demo.html) locally after cloning/downloading. Choose a day,
zone and two controllers to inspect recorded temperatures, electricity, gas,
weather and tariff. It displays saved simulations, not live optimization.

![Four-day pilot comparison](../figures/energyplus_overview.svg)

## Measured outcomes

Equal-weight **means over four declared days**, not annual savings or confidence
intervals. Controllers share the weather forecast, equipment, tariff and initial
state. Fixed and scheduled target trajectories are included as references.

{cost_table}

In this run, NN5-SS has **{ss_change:.1f}% lower mean electrical Expost+** than
NN5-ITO ({summary.loc['NN5-SS', 'electrical_expost']:.3f} versus
{summary.loc['NN5-ITO', 'electrical_expost']:.3f}). However, **{cheapest} has the
lowest mean HVAC bill**, at €{summary.loc[cheapest, 'bill_eur']:.2f} per day.
The learned objective and operating cost rank the controllers differently;
the small pilot does not show DFL dominating simpler controllers.

`Fixed-21` requests 21 °C all day; `Scheduled` requests 21 °C during 08:00–18:00
and 18 °C otherwise. Both use the same ±0.05 °C heating/cooling deadband. The
latter can actively cool to 18 °C overnight: it is a fixed trajectory comparator,
**not** a conventional wide-deadband energy-saving setback strategy.

Gross HVAC consumption excludes PV and other building loads. Illustrative tariffs:
€0.25/0.45/0.65 per kWh electricity and €0.10 per kWh gas. Comfort-band violation
is integrated every 15 minutes across all zones: 20–24 °C during 08:00–18:00,
18–26 °C otherwise. Lower cost at worse comfort is a tradeoff, not a universal win.
[All daily metrics](metrics.csv), [hourly trajectories](hourly.csv) and
[quarter-hour temperatures](quarter_hour.csv) remain available.

**Optimization:** {certified}/{len(neural_rows)} neural test schedules reach the
requested 0.01% relative-gap tolerance within {protocol['solver_time_limit']:g} seconds.
All other incumbents and gaps are retained. Fallbacks: {int(metrics.fallback.sum())}.
Shared-machine timing does not support a speed comparison. The maximum native
SQL/API and hourly/subhourly reconciliation residual is {max_audit:.3g}.
Native engine warning summaries remain in [audits.json](audits.json); successful completion
does not imply a warning-free building or deployment readiness.

## Learning and evaluation protocol

Pretraining uses the 1991–2020 public file with held-out weeks. Physical experiments
use the 2006–2020 EPW. Training: January 10 / July 11; validation: April 11 / October
10; test: January 17 / April 18 / July 18 / October 17, calendar 2017. Each run has
the same two days of 21 °C preconditioning. Future test days never select weights.

The seed-0 five-neuron model gets {protocol['epochs']} SS/AS epochs and two loss
evaluations per update on each of two training days. Both estimators receive the
same evaluation budget. The warm start remains eligible; epoch −1 means it wins.

{fit_table}

The learning objective is **electrical** Expost+ with hourly endpoint discomfort
and cost mismatch. Gas and subhourly comfort are independent diagnostics, so
improving the objective need not improve them. Selected NN weights, loss calls
and [protocol.json](protocol.json) are committed. This small one-seed pilot does
not establish convergence or a universal DFL advantage.

## Stronger prediction baselines

Identical public data, split masks, author time convention, train-only scaling
and 8,692 exact-input-disjoint test transitions:

{baseline_table}

Regularization is selected on validation only. Passive RC constrains thermal
exchange signs and Euler state weights; it is an extra comparator, not a change
to Paper 1's literal model. [Fits and selections](../prediction_baselines/).

## RC initialization diagnosis

![RC learning curves](../figures/rc_initialization.svg)

The paired three-seed synthetic diagnostic repeats eight DFL epochs with and
without 1/25 relative Gaussian parameter perturbation, allowing validation
rollback. Without perturbation, **{rollback}/3 seeds retain epoch −1**. Perturbed
DFL achieves mean test Expost+ {rc_means.loc[('perturbed', 'DFL-selected')]:.3f},
versus {rc_means.loc[('unperturbed', 'warm')]:.3f} for intact ITO. Partial recovery
from initialization damage is not a gain over ITO. [Curves, selected parameters
and all outcomes](../rc_initialization/). This diagnostic is synthetic, not an
explanation of every EnergyPlus outcome.

## Reproduce

```bash
python scripts/setup_energyplus.py --install-linux
python scripts/run_prediction_baselines.py
python scripts/run_energyplus_pilot.py --resume
python scripts/run_rc_initialization_ablation.py
python scripts/build_energyplus_report.py
```

For the committed pilot directory use `--resume`, or choose a fresh `--output`
directory to rerun it. Resume rejects a changed protocol. Time-limited incumbents
and checkpoint selection can vary with hardware. [Integration details and
remaining original-paper gates](../../docs/ENERGYPLUS_INTEGRATION.md).
"""
    (output / "README.md").write_text(report)
    payload = dict(days=protocol["test"], methods=METHODS, zones=sorted(hourly.zone.unique()),
        metrics=json.loads(metrics.to_json(orient="records")), hourly=json.loads(hourly.round(6).to_json(orient="records")))
    template = (ROOT / "scripts/templates/energyplus_demo.html").read_text()
    (output / "demo.html").write_text(template.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")))
    included = [*output.glob("*"), *(REPORTS / "prediction_baselines").glob("*"), *(REPORTS / "rc_initialization").glob("*"), *figures.glob("energyplus_*"), *figures.glob("rc_initialization.*")]
    hashes = {str(p.relative_to(REPORTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(included) if p.is_file()}
    (REPORTS / "portfolio_artifacts.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(summary.to_string())
    print(f"Generated demo, report, figures and {len(hashes)} artifact checksums")


if __name__ == "__main__":
    main()
