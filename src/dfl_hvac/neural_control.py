"""Exact ReLU-constrained scheduling, solved with open-source SCIP.

The quadratic objective uses an equivalent convex epigraph, so SCIP handles the
MIQP as a mixed-integer convex nonlinear problem. ReLU equations remain exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .bounds import horizon_bounds


@dataclass(frozen=True)
class NeuralScenario:
    ambient_temperature_c: np.ndarray
    initial_temperature_c: np.ndarray
    energy_price: np.ndarray
    heating_capacity_kw: object = 10.0
    cooling_capacity_kw: object = 10.0
    target_c: object = 21.0
    discomfort_weight: object = 1.0
    demand_charge: float = 0.4
    export_price: object = 0.0
    nondispatchable_kw: object = 0.0
    generation_kw: object = 0.0
    grid_capacity_kw: float = 10000.0
    temperature_bounds_c: tuple[float, float] = (-50.0, 80.0)
    ambient_bounds_c: tuple[float, float] = (-40.0, 55.0)
    dt_hours: float = 1.0
    name: str = "scenario"

    def arrays(self, zones):
        price = np.asarray(self.energy_price, dtype=float)
        if price.ndim != 1 or not len(price) or zones < 1:
            raise ValueError("nonempty price vector and positive zone count required")
        t = len(price)

        def time(value):
            return np.broadcast_to(np.asarray(value, dtype=float), (t,)).copy()

        def tz(value):
            a = np.asarray(value, dtype=float)
            if a.shape == (t,):
                a = a[:, None]
            return np.broadcast_to(a, (t, zones)).copy()

        d = dict(
            price=price,
            sell=time(self.export_price),
            ambient=time(self.ambient_temperature_c),
            initial=np.asarray(self.initial_temperature_c, dtype=float),
            heat_cap=tz(self.heating_capacity_kw),
            cool_cap=tz(self.cooling_capacity_kw),
            target=tz(self.target_c),
            weight=tz(self.discomfort_weight),
            load=time(self.nondispatchable_kw),
            generation=time(self.generation_kw),
            state_lower=np.full(zones, float(self.temperature_bounds_c[0])),
            state_upper=np.full(zones, float(self.temperature_bounds_c[1])),
        )
        if d["initial"].shape != (zones,) or not all(np.isfinite(a).all() for a in d.values()):
            raise ValueError("scenario arrays must be finite and dimensionally consistent")
        if any(
            np.any(d[k] < 0)
            for k in ("heat_cap", "cool_cap", "weight", "load", "generation", "sell")
        ):
            raise ValueError(
                "capacities, weights, loads, generation and export prices must be >= 0"
            )
        if np.any(price <= d["sell"]):
            raise ValueError("import tariff must exceed export tariff (no arbitrage)")
        if (
            not np.isfinite([self.demand_charge, self.grid_capacity_kw, self.dt_hours]).all()
            or self.demand_charge < 0
            or self.grid_capacity_kw <= 0
            or self.dt_hours <= 0
        ):
            raise ValueError("invalid tariff, grid capacity or interval duration")
        if (
            np.any(d["state_lower"] >= d["state_upper"])
            or np.any(d["initial"] < d["state_lower"])
            or np.any(d["initial"] > d["state_upper"])
        ):
            raise ValueError("invalid temperature domain or initial state")
        amin, amax = self.ambient_bounds_c
        if (
            not np.isfinite([amin, amax]).all()
            or amin > amax
            or np.any(d["ambient"] < amin)
            or np.any(d["ambient"] > amax)
        ):
            raise ValueError("ambient bounds must enclose the scenario")
        return d


@dataclass(frozen=True)
class NeuralSchedule:
    indoor_temperature_c: np.ndarray
    heating_power_kw: np.ndarray
    cooling_power_kw: np.ndarray
    imported_power_kw: np.ndarray
    exported_power_kw: np.ndarray
    objective: float
    energy_cost: float
    demand_cost: float
    discomfort_cost: float
    status: str
    relative_gap: float
    solve_seconds: float
    build_seconds: float
    nodes: int
    binary_variables: int
    total_relus: int
    preactivation_width: float
    max_dynamics_residual: float

    @property
    def hvac_power_kw(self):
        return self.heating_power_kw + self.cooling_power_kw


def solve_neural_hvac(
    network,
    scenario,
    *,
    bound_mode="reachable",
    time_limit=60.0,
    relative_gap=1e-7,
    require_optimal=True,
):
    """Reject unproven solutions by default; always return gap and dynamics residual."""
    from pyscipopt import Model, quicksum

    if time_limit <= 0 or relative_gap < 0:
        raise ValueError("invalid solver budget")
    begin = perf_counter()
    d = scenario.arrays(network.n_zones)
    horizon, zones = d["heat_cap"].shape
    bounds = horizon_bounds(network, scenario, mode=bound_mode)
    mip = Model("neural_hvac")
    mip.hideOutput()
    mip.setRealParam("limits/time", float(time_limit))
    mip.setRealParam("limits/gap", float(relative_gap))
    mip.setRealParam("numerics/feastol", 1e-8)
    mip.setIntParam("randomization/randomseedshift", 0)
    tau = [
        [
            mip.addVar(
                lb=float(bounds.state_lower[t, z]),
                ub=float(bounds.state_upper[t, z]),
                name=f"tau_{t}_{z}",
            )
            for z in range(zones)
        ]
        for t in range(horizon + 1)
    ]
    heat = [
        [mip.addVar(lb=0, ub=float(d["heat_cap"][t, z])) for z in range(zones)]
        for t in range(horizon)
    ]
    cool = [
        [mip.addVar(lb=0, ub=float(d["cool_cap"][t, z])) for z in range(zones)]
        for t in range(horizon)
    ]
    imported = [mip.addVar(lb=0, ub=scenario.grid_capacity_kw) for _ in range(horizon)]
    exported = [mip.addVar(lb=0, ub=scenario.grid_capacity_kw) for _ in range(horizon)]
    peak = mip.addVar(lb=0, ub=2 * scenario.grid_capacity_kw)
    binaries = 0
    for t in range(horizon):
        raw = [*tau[t], *[heat[t][z] - cool[t][z] for z in range(zones)], d["ambient"][t]]
        inputs = [
            (v - float(m)) / float(s)
            for v, m, s in zip(raw, network.input_mean, network.input_scale)
        ]
        for layer, (weight, bias) in enumerate(zip(network.weights, network.biases)):
            outputs = []
            for j in range(len(bias)):
                affine = quicksum(float(w) * v for w, v in zip(weight[j], inputs)) + float(bias[j])
                if layer == len(network.weights) - 1:
                    outputs.append(affine)
                    continue
                lo, hi = (float(a[j]) for a in bounds.preactivations[t][layer])
                if hi <= 0:
                    outputs.append(0.0)
                elif lo >= 0:
                    outputs.append(affine)
                else:
                    y, active = mip.addVar(lb=0, ub=hi), mip.addVar(vtype="B")
                    mip.addCons(y >= affine)
                    mip.addCons(y <= hi * active)
                    mip.addCons(y <= affine - lo * (1 - active))
                    outputs.append(y)
                    binaries += 1
            inputs = outputs
        for z in range(zones):
            mip.addCons(
                tau[t + 1][z]
                == inputs[z] * float(network.output_scale[z]) + float(network.output_mean[z])
            )
        consumption = quicksum(heat[t][z] + cool[t][z] for z in range(zones))
        mip.addCons(
            imported[t] - exported[t] == consumption + float(d["load"][t] - d["generation"][t])
        )
        mip.addCons(peak >= imported[t] + exported[t])
    comfort = quicksum(
        float(d["weight"][t, z]) * (tau[t + 1][z] - float(d["target"][t, z])) ** 2
        for t in range(horizon)
        for z in range(zones)
    )
    epigraph = mip.addVar(lb=0)
    mip.addCons(epigraph >= scenario.dt_hours * comfort)
    energy = quicksum(
        float(d["price"][t]) * imported[t] - float(d["sell"][t]) * exported[t]
        for t in range(horizon)
    )
    mip.setObjective(scenario.dt_hours * energy + scenario.demand_charge * peak + epigraph)
    build_time = perf_counter() - begin
    mip.optimize()
    status = str(mip.getStatus())
    if mip.getNSols() == 0:
        raise RuntimeError(f"SCIP returned no feasible schedule: {status}")
    gap = float(mip.getGap())
    if (
        require_optimal
        and status != "optimal"
        and not (status == "gaplimit" and gap <= relative_gap)
    ):
        raise RuntimeError(f"SCIP has not certified the requested gap: {status}, gap={gap}")

    def values(rows):
        return np.asarray([[mip.getVal(v) for v in row] for row in rows])

    temperature, heating, cooling = values(tau), values(heat), values(cool)
    pin, pout = [np.asarray([mip.getVal(v) for v in row]) for row in (imported, exported)]
    pred = network.predict(np.column_stack((temperature[:-1], heating - cooling, d["ambient"])))
    residual = float(np.max(np.abs(temperature[1:] - pred)))
    if residual > 1e-4:
        raise RuntimeError(f"ReLU dynamics residual too large: {residual}")
    ec = scenario.dt_hours * float(d["price"] @ pin - d["sell"] @ pout)
    dc = scenario.demand_charge * float(np.max(pin + pout))
    cc = scenario.dt_hours * float(np.sum(d["weight"] * (temperature[1:] - d["target"]) ** 2))
    return NeuralSchedule(
        temperature,
        heating,
        cooling,
        pin,
        pout,
        ec + dc + cc,
        ec,
        dc,
        cc,
        status,
        gap,
        float(mip.getSolvingTime()),
        build_time,
        int(mip.getNNodes()),
        binaries,
        bounds.total_relus,
        bounds.preactivation_width,
        residual,
    )


def expost_metrics(schedule, response, scenario):
    """Expost+ at no generation; explicit net-grid-cost generalization otherwise.

    Assign the demand charge once, at the first maximum, including tied peaks.
    """
    d = scenario.arrays(response.hvac_power_kw.shape[1])

    def cost_profile(power):
        net = np.asarray(power).sum(1) + d["load"] - d["generation"]
        cost = scenario.dt_hours * (
            d["price"] * np.maximum(net, 0) - d["sell"] * np.maximum(-net, 0)
        )
        idx = int(np.argmax(np.abs(net)))
        cost[idx] += scenario.demand_charge * abs(net[idx])
        return cost, net

    expected, _ = cost_profile(schedule.hvac_power_kw)
    realized, net = cost_profile(response.hvac_power_kw)
    discomfort = scenario.dt_hours * float(
        np.sum(d["weight"] * (response.indoor_temperature_c[1:] - d["target"]) ** 2)
    )
    mse = float(np.mean((realized - expected) ** 2))
    bill = float(realized.sum())
    return dict(
        expected_bill=float(expected.sum()),
        realized_bill=bill,
        comfort_penalty=discomfort,
        cost_mse=mse,
        expost_plus=bill + discomfort + mse,
        peak_kw=float(np.max(np.abs(net))),
        tracking_mae_c=float(
            np.mean(np.abs(response.indoor_temperature_c[1:] - schedule.indoor_temperature_c[1:]))
        ),
        grid_exceedance_kw=float(np.maximum(np.abs(net) - scenario.grid_capacity_kw, 0).max()),
    )
