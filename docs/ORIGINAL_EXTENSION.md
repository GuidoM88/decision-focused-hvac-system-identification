# Reachable-state bounds for neural HVAC scheduling

The contribution is an independently implemented and evaluated **extension of
these two HVAC reproductions**, not a claim that interval reachability or ReLU
bound tightening was invented here. The engineering question is precise:

> After fixing weather as in Paper 2, does propagating reachable thermal states
> through the planning horizon reduce the mixed-integer problem without changing
> its feasible schedules or optimum?

The implementation is `src/dfl_hvac/bounds.py`. The paired experiment is
`scripts/benchmark_bounds.py`; raw outcomes, including timeouts, are committed in
`reports/bounds/`. It uses networks independently trained on the authors' public
five-zone EnergyPlus data, rather than networks designed to favor the extension.

## Signed interval arithmetic

For an affine map $a=Wx+b$ with $l\le x\le u$, write
$W^+=\max(W,0)$ and $W^-=\min(W,0)$. Then

$$\underline a=W^+l+W^-u+b,\qquad
  \overline a=W^+u+W^-l+b.$$

These are exact extrema over an axis-aligned input box. ReLU maps the interval to
$[\max(0,\underline a),\max(0,\overline a)]$. Repeated application gives an outer
bound for a multilayer network; it generally loses correlations and is therefore
not an exact network reachable set. Positive standardization scales preserve the
ordering of bounds.

For a neuron with variable inputs $x$ and known inputs $p$, the interval-width
ratio after fixing $p$ is exactly

$$r=\frac{\sum_i |w_{x,i}|(u_{x,i}-l_{x,i})}
 {\sum_i |w_{x,i}|(u_{x,i}-l_{x,i})+
  \sum_j |w_{p,j}|(u_{p,j}-l_{p,j})}.$$

The ratio requires a nonzero denominator. Unlike the approximate equal-width
expression in Paper 2, this identity handles unequal widths and signed weights.
It is elementary interval arithmetic, not a new theorem.

## Temporal recursion and equivalence argument

Let $X$ be the declared physical temperature domain, $U_t$ the admissible signed
power box, $w_t$ known weather, and $F_\theta$ the learned network. Initialize
$X_0=\{x_0\}$ and construct

$$X_{t+1}=X\cap\operatorname{IBP}(F_\theta;X_t\times U_t\times\{w_t\}).$$

**Claim.** Every trajectory feasible for the original bounded neural scheduling
problem remains inside these sets. Therefore adding these state bounds and
recomputing the ReLU big-M constants leaves its feasible schedules and optimal
objective unchanged in exact arithmetic.

**Proof.** The claim holds at $t=0$ by the initial-state equality. If a feasible
state lies in $X_t$, its feasible control lies in $U_t$, and interval propagation
contains the network output for every such input. Its next state also belongs
to $X$ by the original domain constraint, hence to their intersection. Induction
completes the argument. Substituting valid tighter bounds in an exact big-M
encoding preserves the same ReLU graph. A neuron whose entire interval is
nonpositive is zero; a neuron whose interval is nonnegative is affine. Removing
its binary variable therefore also preserves that graph.

The software adds conservative numerical margins, checks forward-network
residuals and records solver gaps. These are numerical checks, not a formal
floating-point proof. An empty propagated intersection is reported as infeasible.
The bounds certify the **learned model's bounded optimization problem**, never
the comfort or safety of the actual building.

## Experimental comparison

| Mode | Weather bounds | Thermal state bounds |
|---|---|---|
| `static` | Global declared range | Global domain, known initial state |
| `weather` | Actual weather as singleton | Global domain, known initial state |
| `reachable` | Actual weather as singleton | Recursive reachable intervals |

Each triplet has identical weights, prices, capacities, initial conditions,
temperature domain and objective. Mode order is deterministically randomized.
The ablation records binary counts, interval widths, solve/build times, gaps and
the number of certified solutions. Objective equality is assessed only where
**both** modes meet the optimality tolerance. Timeout incumbents remain in the raw
data and are not treated as proven optima. Tighter bounds can affect presolve and
branching, so their effect on runtime is measured separately from interval width.

## Related work and novelty boundary

The [executed ablation](../reports/README.md#original-extension-reachable-state-intervals)
finds 14.2% mean per-instance width reduction, but no additional binary removals
relative to weather-only bounds and no demonstrated runtime improvement. Each
mode certifies 35 of 54 cases within the five-second limit. Objective agreement
is observed for the 34 jointly certified pairs; timeout incumbents are excluded
from that equality claim, not from the raw results. This is evidence that tighter
intervals need not cross enough activation thresholds to simplify these models.


- Favaro et al., [Paper 2](https://arxiv.org/abs/2506.19717v2): exact NN scheduling,
  stochastic smoothing and exogenous-input singleton bounds are theirs.
- Sosnin & Tsay, [Scaling Mixed-Integer Programming for Certification of Neural
  Network Controllers Using Bounds Tightening](https://arxiv.org/abs/2403.17874):
  prior work on tightening and propagation across multiple control steps.
- Xiang et al., [Reachable Set Estimation and Safety Verification for Piecewise
  Linear Systems with Neural Network Controllers](https://arxiv.org/abs/1802.06981):
  prior reachability analysis for ReLU-controlled systems.

This study contributes an implementation in the HVAC DFL pipeline, a reproducible
three-way ablation and tests of solution equivalence. It does not introduce a new
interval-reachability or bound-tightening algorithm.
The optional antithetic Gaussian estimator is a standard comparator, with the
same total objective-evaluation budget as the vanilla score estimator.
