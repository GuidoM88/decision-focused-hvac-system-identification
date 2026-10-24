# Paper 1 method mapping

Reference: Favaro et al. (2025), *Decision-Focused Learning for Complex System Identification: HVAC Management System Application*.

| Paper element | Equation / section | Repository object | Status |
|---|---|---|---|
| Unknown system dynamics | Eq. (1) | `SystemRunner` interface | synthetic response verified; separate five-zone EnergyPlus pilot; original fifteen-zone case unresolved |
| Convex control policy | Eq. (2) | `solve_day_ahead_hvac` | implemented |
| DFL chain rule | Eq. (3) | `DifferentiableHVACLayer` + `train_dfl` | implemented; synthetic benchmark verified |
| Cone-program differentiation | Eqs. (4)-(5) | `DifferentiableHVACLayer` | implemented with CvxpyLayers |
| Regret reference | Eq. (6) | evaluation documentation | mapped |
| Supervised DFL principle | Eq. (7) | `dfl_training.py` | implemented |
| State-target relaxation | Eq. (8) | quadratic target term | implemented |
| HVAC objective | Eq. (9) | `solve_day_ahead_hvac` | implemented |
| Multi-zone RC dynamics | Eq. (10) | `rc_model.py` | implemented as printed |
| Initial state | Eq. (11) | controller | implemented |
| Zone capacities | Eq. (12) | controller | implemented |
| Floor/AHU capacities | Eq. (13) | controller | implemented |
| HVAC power identity | Eq. (14) | controller | implemented |
| Grid import balance | Eq. (15) | controller | implemented |
| Peak definition | Eq. (16) | epigraph constraint | implemented |
| Import capacity | Eq. (17) | controller | implemented |
| Power nonnegativity | Eq. (18) | variable domains | implemented |
| Hierarchical power loss | Eq. (19) | NumPy and PyTorch implementations | implemented and cross-tested |
| Historical MSE pre-training | Sec. 3.5 | `fit_ito` | implemented |
| Gaussian perturbation | Sec. 3.5 | `add_relative_gaussian_noise` | implemented |
| Representative days | Sec. 3.5 | `scenarios.py` | implemented; ordering rule reconstructed |
| DFL training | Sec. 3.5 | `train_dfl` | implemented; synthetic benchmark verified |
| Hot-year stress test | Sec. 3.6 | `build_hot_year` | implemented |
| Paper metrics | Tables 1-2 | `evaluation.py`, `reference_results.py` | implemented |

## Reconstruction choices

- Control arrays use `T` decision intervals and `T+1` thermal states; the target penalty is applied to `tau[1:]`.
- Eq. (10) is implemented literally as printed.
- The paper reports 265 RC parameters but a fully free 15x15 `alpha` plus four 15-vectors contains 285. The implied restriction remains unresolved.
- The Gaussian perturbation is interpreted as standard deviation `abs(theta_i)/25`, consistent with the stated SNR of 625.
- The smooth ordering of medoids is reconstructed as a deterministic nearest-neighbour path in daily mean/standard-deviation space.
- Historical ITO optimizer defaults are informed by the authors' later public implementation and are not attributed to Paper 1.
- Paper 1 reports ECOS together with CvxpyLayers. Modern CvxpyLayers solver support is treated as an implementation-compatibility issue rather than silently changing the mathematical program.
