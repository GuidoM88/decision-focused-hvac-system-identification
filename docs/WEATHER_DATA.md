# Weather-data handling

Paper 1 uses two hourly typical meteorological data sets for Denver but does not
identify the exact files. Weather-file identity remains an unresolved
reproduction input.

The repository can parse standard EnergyPlus Weather (EPW) files directly:

```python
from dfl_hvac.epw import epw_daily_temperature_profiles

profiles, labels = epw_daily_temperature_profiles("weather.epw")
```

The EPW parser uses the official EnergyPlus field order: calendar fields are
followed by data-source flags, then dry-bulb temperature in degrees Celsius.
Only complete 24-hour days are passed to the representative-day clustering
pipeline.

Weather files used in a numerical experiment must be recorded in the run
manifest with their SHA-256 checksums. No specific Denver EPW file is treated
as the Paper 1 input without source evidence.
