# Paper 1 experiment specification

This document separates statements supported directly by Favaro et al. (2025)
from reconstruction choices required for implementation.

## Directly specified by the paper

The case study uses the U.S. Department of Energy Medium Office reference
building in Denver: approximately 4,928 square metres, three floors, six zones
per floor, five controlled zones per floor, and one AHU per floor. The
controller is hourly and day-ahead. The RC proxy uses parameters
`theta=(alpha, eta_c, eta_h, R, C)`. The indoor-temperature target is centred
at 21 C.

The time-of-use tariff is EUR 0.3/kWh from 19:00 to 06:00 and EUR 0.6/kWh
from 06:00 to 19:00.

Historical pre-training uses one year generated from a first typical
meteorological data set and an MSE loss. After pre-training, independent
element-wise Gaussian perturbations use a scale described as `theta_i/25`,
corresponding to SNR 625. The implementation interprets this as standard
deviation `abs(theta_i)/25`.

A second typical meteorological year supplies DFL scenarios. Ten medoids are
selected from daily ambient-temperature profiles. Three fixed medoids are the
coldest day, hottest day, and day with the largest temperature variance.
Seven additional medoids complete the partition. One additional day from every
cluster is used for validation and one for testing.

DFL performs one parameter update after each solved and simulated day. ECOS
solves the conic program, CvxpyLayers provides optimization derivatives, and
PyTorch performs backpropagation. Adam starts at 0.001, polynomial decay is
reported with parameter 0.9, maximum training length is 50 epochs, and early
stopping patience is 10 epochs.

The task-aware loss compares expected and ex-post zonal HVAC power. It is
aggregated at building, floor, and zone levels with weights 15, 5, and 1.
The energy price weights every time step and the demand-charge coefficient is
added at the maximum ex-post building-power interval.

The hot-year stress set selects the hottest day from each cluster and adds
2 C to the hottest-cluster profile.

## Unresolved from Paper 1

The paper does not sufficiently specify the following for exact numerical
replication:

- identities of the two meteorological files;
- demand-charge value;
- detailed hourly discomfort weights;
- zone and AHU capacity values;
- grid-import capacity;
- exact random seeds;
- historical MSE optimizer settings;
- exact smooth-cycle ordering algorithm;
- exact free-parameter structure of `alpha`.

The parameter-count statement is internally inconsistent if `alpha` is fully
free: a 15 x 15 `alpha` plus four 15-vectors gives 285 parameters, while the
paper reports 265. No sparsity pattern or fixed-entry set is specified.

Exact numerical replication will not be claimed until these gaps are resolved
or their effect is characterized.


## Auxiliary reconstruction defaults

For executable methodological-reproduction runs, `configs/paper1.yaml`
contains optional defaults recovered from the authors' later public HVAC code:
0.4 EUR/kW demand charge, 10,000 kW import capacity, and a daily comfort-weight
profile of 0.1 for eight hours, 1.0 for eleven hours, and 0.3 for five hours.

These values are deliberately isolated under `reconstruction_defaults`. They
do not resolve the Paper 1 specification gap and must not be presented as
numbers reported by Favaro et al. (2025).
