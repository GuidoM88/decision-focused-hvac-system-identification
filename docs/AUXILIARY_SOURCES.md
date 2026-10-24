# Auxiliary implementation evidence

Paper 1 does not provide an implementation. The same research group later
released the public repository `PSMRB/dfl_hvac_management` for a follow-up
HVAC DFL study associated with arXiv:2506.19717.

That repository is used only as auxiliary implementation evidence, not as proof
that Paper 1 used identical settings.

Relevant observations from the later code are:

- PyTorch RC pre-training with Adam;
- heating/cooling efficiency initialization at 3 and 2;
- resistance and capacitance initialization approximately in [5, 21];
- diagonal spatial RC coefficients initialized at 1 with small off-diagonal
  coefficients;
- randomized 80/20 train/validation splitting;
- historical RC defaults of batch size 64, learning rate 1e-2, 500 maximum
  epochs, and 20-epoch patience.

These defaults are marked as reconstruction choices in `configs/paper1.yaml`
and `src/dfl_hvac/training.py`. The later implementation uses a different
building configuration and a more advanced methodology, so its assets and
results are not substitutes for the Paper 1 experiment.

## Market and comfort defaults

The follow-up implementation also uses the following case-study defaults:

- grid/import capacity: 10,000 kW;
- demand charge: 0.4 EUR/kW;
- import tariff: 0.3 EUR/kWh for the first six hours, 0.6 EUR/kWh for the next
  thirteen hours, and 0.3 EUR/kWh for the final five hours;
- indoor-temperature target: 21 C;
- daily occupancy/comfort weights: 0.1 for eight hours, 1.0 for eleven hours,
  and 0.3 for five hours.

The tariff and target are independently consistent with Paper 1. The numerical
demand charge, line capacity and comfort weights are not reported in Paper 1 and
remain reconstruction defaults only.

## Clustering differences

The later repository must not be used as a drop-in specification for the
Paper 1 representative-day construction. Its clustering code combines ambient
temperature and solar radiation with an 0.8/0.2 weighting after standardization,
and additionally filters days according to thermostat feasibility.

Paper 1 instead states that clustering uses ambient temperature because that is
the only weather input to its RC model. The implementation in this repository
therefore follows the Paper 1 description, while the follow-up code is retained
only as evidence for unresolved implementation conventions.
