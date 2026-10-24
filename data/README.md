# Data and simulator assets

Large EnergyPlus assets and generated trajectories are excluded from git.

Paper 1 cites the U.S. Department of Energy ASHRAE 90.1 Medium Office 2019
reference-building archive:

`https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_OfficeMedium_STD2019.zip`

Run:

```bash
python scripts/download_building_template.py
```

Paper 1 states that two typical meteorological data sets are used for Denver.
One generates historical operation for MSE pre-training; the other supplies
representative-day clustering and DFL evaluation. The paper does not identify
the exact two weather files.

The reproduction layer ultimately requires:

- zone indoor temperature;
- next-step zone indoor temperature;
- outdoor dry-bulb temperature;
- zonal heating electric power;
- zonal cooling electric power;
- ex-post zonal HVAC electric power under an imposed thermostat schedule.

Every external input added later should record its source, retrieval date,
license, and checksum.
