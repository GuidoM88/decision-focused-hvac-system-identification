"""Regenerate the research report and static figures from committed raw results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "reports"
METHODS = ["RC-ITO", "RC-warm", "RC-DFL", "NN-ITO", "NN-SS", "NN-AS", "NN-QP", "NN-FB"]


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main():
    public = pd.read_csv(ROOT / "public_data/identification.csv")
    bounds = pd.read_csv(ROOT / "bounds/runs.csv")
    paired = pd.read_csv(ROOT / "bounds/paired.csv")
    control = pd.read_csv(ROOT / "benchmark/metrics.csv")
    if len(public) != 9 or len(bounds) != 162 or len(control) != 192:
        raise ValueError("expected complete declared studies: 9 fits, 162 solves, 192 control rows")
    if control.duplicated(["seed", "method", "split", "day"]).any():
        raise ValueError("duplicate control outcomes")
    if not bounds.groupby(["seed", "width", "horizon", "mean_c"]).size().eq(3).all():
        raise ValueError("incomplete paired ablation")
    seed_means = control.groupby(["seed", "method", "split"]).mean(numeric_only=True)
    public_summary = public.groupby("hidden_units").agg(
        rmse_mean=("test_rmse_c", "mean"), rmse_sd=("test_rmse_c", "std"),
        disjoint_mean=("disjoint_input_rmse_c", "mean"),
        disjoint_sd=("disjoint_input_rmse_c", "std"),
    )
    public_summary.to_csv(ROOT / "public_data/summary.csv")
    bound_summary = bounds.groupby("mode").agg(
        instances=("certified", "size"), certified=("certified", "sum"),
        mean_binaries=("binary_variables", "mean"),
        mean_wall_seconds=("wall_seconds", "mean"),
    )
    bound_summary.to_csv(ROOT / "bounds/summary.csv")
    control_summary = seed_means.groupby(["method", "split"]).agg(
        expost_mean=("expost_plus", "mean"), expost_sd=("expost_plus", "std"),
        bill_mean=("realized_bill", "mean"), bill_sd=("realized_bill", "std"),
        cost_mse_mean=("cost_mse", "mean"),
        prediction_mse_mean=("prediction_mse_c2", "mean"),
    )
    control_summary.to_csv(ROOT / "benchmark/summary.csv")
    baseline = float(public.disjoint_persistence_rmse_c.iloc[0])
    reduction = 100 * float(paired.width_reduction_fraction.mean())
    certified = paired.loc[paired.both_certified]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "svg.hashsalt": "dfl-hvac-v0.2",
    })
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    a, b = axes
    a.bar([0, 1, 2], public_summary.disjoint_mean,
          yerr=public_summary.disjoint_sd, capsize=5, color=["#88b5d6", "#4185af", "#12645e"])
    a.axhline(baseline, color="#ad5537", ls="--", label=f"Persistence: {baseline:.3f} °C")
    a.set(xticks=[0, 1, 2], xticklabels=["2 neurons", "5 neurons", "10 neurons"],
          ylabel="Temperature RMSE (°C) · lower is better", ylim=(0, 0.7),
          title="Public-data identification")
    a.legend(frameon=False, loc="upper right")
    for i, value in enumerate(public_summary.disjoint_mean):
        a.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)
    a.text(0.02, 0.02, "8,692 exact-input-disjoint transitions\nBars: mean ± sample SD across 3 seeds",
           transform=a.transAxes, va="bottom", fontsize=9, color="#555555",
           bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 4})
    b.hist(100 * paired.width_reduction_fraction, bins=np.arange(0, 40, 4),
           color="#12645e", edgecolor="white")
    b.axvline(reduction, color="#ad5537", ls="--", label=f"Mean: {reduction:.1f}%")
    b.set(xlabel="Interval-width reduction vs weather-only (%)", ylabel="Scheduling instances",
          title="Reachable-state bound ablation", xlim=(0, 36))
    b.legend(frameon=False)
    b.text(0.98, 0.65, "54 paired instances\nNo extra binary removals\n35/54 certified in each mode\nNo demonstrated speedup",
           transform=b.transAxes, va="top", ha="right", fontsize=9, color="#555555")
    fig.savefig(figures / "overview.svg", metadata={"Date": None})
    fig.savefig(figures / "overview.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.6), layout="constrained")
    positions = np.arange(len(METHODS))
    for offset, split, color, label in [
        (-0.18, "test", "#4185af", "Test"), (0.18, "hot", "#b86443", "+4 °C stress")
    ]:
        group = control_summary.xs(split, level="split").loc[METHODS]
        ax.barh(positions + offset, group.expost_mean, xerr=group.expost_sd,
                height=0.34, capsize=3, color=color, label=label)
    ax.set(yticks=positions, yticklabels=METHODS, xlabel="Expost+ score · lower is better",
           title="Synthetic control benchmark · mean ± sample SD across 3 seeds")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(figures / "control.svg", metadata={"Date": None})
    fig.savefig(figures / "control.png", dpi=150)
    plt.close(fig)

    public_table = table(["Hidden neurons", "Full-file RMSE, °C", "Disjoint-input RMSE, °C"], [
        [width, f"{r.rmse_mean:.3f} ± {r.rmse_sd:.3f}",
         f"{r.disjoint_mean:.3f} ± {r.disjoint_sd:.3f}"]
        for width, r in public_summary.iterrows()
    ])
    bound_table = table(["Bounds", "Certified / total", "Mean binaries", "Mean wall time, s"], [
        [mode, f"{int(r.certified)}/{int(r.instances)}", f"{r.mean_binaries:.2f}",
         f"{r.mean_wall_seconds:.3f}"] for mode, r in bound_summary.iterrows()
    ])
    control_table = table(["Method", "Test Expost+", "+4 °C Expost+", "Prediction RMSE, °C"], [
        [m,
         f"{control_summary.loc[(m, 'test'), 'expost_mean']:.3f} ± {control_summary.loc[(m, 'test'), 'expost_sd']:.3f}",
         f"{control_summary.loc[(m, 'hot'), 'expost_mean']:.3f} ± {control_summary.loc[(m, 'hot'), 'expost_sd']:.3f}",
         f"{np.sqrt(seed_means.xs(m, level='method').xs('test', level='split').prediction_mse_c2).mean():.3f}"]
        for m in METHODS
    ])
    improvement = 100 * (1 - public_summary.loc[10, "disjoint_mean"] / baseline)
    report = f"""# Measured results

Start with the [five-zone EnergyPlus control pilot](energyplus/README.md) and
[offline comparison demo](energyplus/demo.html) for actual building control,
stronger prediction baselines and the RC initialization diagnosis. The studies
below provide the earlier identification, synthetic-control and bounds evidence.

Generated by `python scripts/build_report.py` from the committed raw CSVs.
These are independent project experiments, not numerical matches to the original paper tables.

## Public EnergyPlus-data identification

Two author-published **typical meteorological files** are used, with source periods
1991–2020 and 2006–2020. They are not independent calendar-year field observations.
The first file provides 7,059 training and 1,680 validation transitions; the second
provides 8,759 test transitions. Every fifth week is validation, with adjacent
training transitions embargoed. Normalization is fitted on training rows only.

An audit finds 67 exact input matches between the two files. The additional
diagnostic removes all of them without refitting, leaving 8,692 test transitions.
This does not prove statistical independence: both files concern the same building
and location, and their source weather periods overlap.

{public_table}

Values are mean ± **sample standard deviation across three initialization seeds**,
not confidence intervals. Persistence RMSE is 0.602 °C on the complete file and
{baseline:.3f} °C on the disjoint-input diagnostic. The 10-neuron model improves
that diagnostic RMSE by **{improvement:.1f}%**. This measures one-step prediction,
not physical control performance or energy savings.

[Raw fits](public_data/identification.csv) · [Source/split manifest](public_data/manifest.json)

## Original extension: reachable-state intervals

54 paired instances × three formulations = 162 solves. Networks have 2 or 5
hidden neurons, five zones, three seeds, horizons of 6/12/24 hours and ambient
means of 5/20/32 °C. Every formulation has the same network, objective, initial
state, 10–35 °C declared domain and 3 kW zonal heating/cooling capacities.
The solver has five seconds per solve and target relative gap 1e-6.

{bound_table}

Recursive bounds reduce total preactivation width by **{reduction:.1f}% on average
per instance**, relative to weather-only bounds. The additional binary reduction
is zero on every instance. Both modes certify 35/54 instances, with
**{len(certified)} jointly certified pairs**. The largest objective difference in
those pairs is **{certified.objective_difference.max():.2g}**.

The remaining pairs are not evidence of optimal-objective equality: their
incumbents can differ substantially at timeout. All outcomes remain in the raw
table. The induction argument and independent small-instance enumeration support
the formulation's equivalence separately from these time-limited comparisons.

**No runtime improvement is demonstrated.** There is one timing per case; the
initial ablation ran alongside other experiments on a shared machine. Timing is
included for transparency and must not be used as a controlled speed benchmark.
Tighter interval bounds alone did not remove additional unstable neurons here.

[Raw solves](bounds/runs.csv) · [Paired comparisons](bounds/paired.csv) ·
[Manifest](bounds/manifest.json) · [Derivation and prior work](../docs/ORIGINAL_EXTENSION.md)

## Synthetic decision-focused control

![Control scores](figures/control.svg)

One nonlinear synthetic zone, eight-hour horizon, three training days, two
validation days, four test days and four +4 °C stress days per seed. There are
three seeds and eight methods, for 192 evaluation rows. NN pretraining uses up
to 250 epochs; downstream learning uses eight epochs. RC historical pretraining
uses 150 epochs. This is a small methodological benchmark, not a claim of
convergence or a reproduction of the full paper budgets.

{control_table}

Each score first averages the four days **within a seed**; the table then reports
mean ± sample SD across the three seeds. Prediction RMSE uses 1,000 separate
synthetic transitions and is averaged across seeds. It is measured on the
historical transition distribution, not the controller's visited trajectories.

Expost+ = realized electricity bill + weighted squared-temperature discomfort +
mean squared hourly cost-prediction error. This is a composite learning score,
**not a euro-denominated bill**. The component metrics are available in the raw
CSV. The mean and SD are descriptive; three initializations are not a significance
test, and there is no claim that one method universally dominates another.

`RC-warm` is the perturbed RC-ITO initialization used for RC-DFL. Reporting it
separately exposes degradation caused by initialization noise. NN methods retain
the original checkpoint if it beats all trained candidates on validation. SS and
AS both receive two scalar objective evaluations per update; AS uses a paired
positive/negative perturbation. Validation costs are accounted for separately.
QP/FB select checkpoints using supervised power loss, while SS/AS use Expost+,
following the distinction between their learning signals.

[Raw per-day outcomes](benchmark/metrics.csv) · [Training histories](benchmark/training.json) ·
[Experiment manifest](benchmark/manifest.json) · [Aggregated components](benchmark/summary.csv)

## Reproduction and limitations

Run the experiment commands in the [README](../README.md#reproduce-the-experiments).
Public-data networks, synthetic NN initializations, raw measurements and manifests
are versioned; selected DFL parameters are regenerated by rerunning training. The demo and
mathematical correctness tests run in CI; the long training experiments are not
rerun on every push. `artifacts.json` records hashes of the measured inputs.

The independent EnergyPlus pilot now validates its physical actuation. Original
control-table replication still requires matching the authors' protocol, full
training runs and, for Paper 1, missing experimental details. See the
[remaining gates](../docs/REPRODUCTION_PLAN.md). The learned-model bounds provide
no physical safety guarantee.
"""
    (ROOT / "README.md").write_text(report)
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("public_data", "bounds", "benchmark")
        for p in sorted((ROOT / folder).glob("*")) if p.suffix in {".json", ".csv"}
    }
    (ROOT / "artifacts.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Generated report, figures and {len(hashes)} artifact checksums")


if __name__ == "__main__":
    main()
