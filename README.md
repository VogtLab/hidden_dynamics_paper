# Windowed KL analysis of spectral time series

This repository analyzes fluctuations in raw single-beam spectra. At each wavenumber it converts detector intensity to photon counts and compares the measured count distribution with a Poisson reference of the same mean. It reports Kullback–Leibler (KL) divergence for the full selected record and for successive time windows. A separate script estimates detector gain from a stable reference measurement.

## Installation

Use Python 3.12. From the repository directory, create an environment and install the package versions used to verify the examples.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. Both scripts take their settings from the top of the corresponding Python file. Relative paths resolve from the directory where you run the command.

## Try a small dataset

Save the following as `example.csv` in the repository directory. The upper-left field is the acquisition timestamp in the literal format `DD.MM.YYYY HH:MM:SS UTC`. The rest of the header gives sample times in seconds, in chronological order. Each remaining row starts with a wavenumber in cm⁻¹ and contains one raw intensity per time. Intensities in the retained region must be finite and strictly positive.

```csv
01.01.2026 12:00:00 UTC,0,1,2,3,4,5,6,7,8,9,10,11
2100,12,15,13,16,15,13,14,12,15,16,14,13
2078,22,23,21,20,20,21,23,22,22,23,21,20
2050,30,31,30,32,31,32,30,31,32,30,31,30
```

In `windowed_kl.py`, set these values in the settings block. This example uses synthetic counts with a known gain of one intensity unit per count.

```python
OPERANDO_CSV_PATH = 'example.csv'
OUTPUT_DIR = 'results/example'
GAIN_ESTIMATE = 1.0
HISTOGRAM_ENERGIES = [2078.0]
SLIDING_WINDOW_LENGTH = 8
SLIDING_STEP = 2
N_JOBS = 1
```

Run the analysis from the repository directory.

```sh
python windowed_kl.py
```

Open `results/example/csv/sliding_KL_matrix.csv`. It should contain three wavenumber rows and three KL columns centered at **3.5, 5.5 and 7.5 s**. The histogram and time-trace figures show the 2078 cm⁻¹ channel.

## Choose the detector gain

The conversion is `counts = round(intensity / gain)`, with NumPy rounding exact half counts to the nearest even integer. Choose the gain method appropriate to your measurement in `windowed_kl.py`.

| `GAIN_ESTIMATE` | Source of gain |
| --- | --- |
| Positive number | A known scalar gain shared by all channels. |
| `'reference_energy_gain'` (script default) | A CSV at `REFERENCE_CSV_PATH` with columns `energy,gain`. Look up each channel by its wavenumber. |
| `'reference_intensity_gain'` | A CSV at `REFERENCE_CSV_PATH` with columns `intensity,gain`. Look up each channel by its mean over the selected record. |
| `'single_channel'` | Population variance divided by mean intensity at the retained channel nearest `NOISE_ENERGY`. Choose a channel representative of detector noise. |
| `'regression'` | The slope of noise variance versus mean intensity. Noise variance is half the mean squared difference between consecutive samples. The fit uses medians from 32 intensity bins and needs at least two bins with three or more channels each, and at least three timepoints. |

Both reference methods interpolate linearly between finite table entries and use the endpoint gains outside the calibrated range. The mapped gains must be positive. The legacy value `'reference'` selects the energy-based lookup.

### Calibrate from a stable reference

Use a stable-target measurement in the same XYYYY format. In `gain_calibration.py`, set the two paths and run the script.

```python
reference_csv_path = 'data/stable_reference.csv'
output_dir = 'results/calibration'
```

```sh
python gain_calibration.py
```

The script measures gain as population variance divided by mean intensity at each wavenumber. Set `time_range` to select the reference interval and `exclude_energies` to omit specified wavenumbers from fitting. The `approximate_gain` setting controls both calibration tables.

| `approximate_gain` | Values written to the table |
| --- | --- |
| `'fit_cubic_full'` (script default) | Evaluate the positive cubic at every channel. |
| `'fig_cubic_interpolate'` | Keep valid, included measurements and fill the other channels from the fitted cubic. Use the spelling shown. |
| `'raw'` | Keep valid, included measurements only. |

The energy cubic requires at least four usable channels. The optional intensity cubic requires at least five to estimate the residual variance for its 95% confidence and prediction bands. Those bands are approximate when the positivity constraint affects the fit. Set `calibrate_gain_per_intensity = False` to request only the energy calibration. The energy calibration figure plots **reciprocal gain**. Both CSVs store **gain** in intensity units per count.

To apply the calibrated energy gain, set these values in the analysis script, alongside your operando input and output paths.

```python
GAIN_ESTIMATE = 'reference_energy_gain'
REFERENCE_CSV_PATH = 'results/calibration/csv/gain_calibration.csv'
```

## How the KL value is calculated

For one channel and one interval, the code counts the occurrences of each distinct integer photon count. Their relative frequencies form the empirical distribution. It evaluates a Poisson probability mass function at those counts, using the sample mean as its rate. If `MAX_HISTOGRAM_SUPPORT` is set, larger counts are discarded and the mean is recalculated.

**The reference distribution is normalized over the observed counts.** Specifically, the code retains observed counts with positive numerical Poisson mass, normalizes both distributions on that support, and computes `sum(p * ln(p / q))` in nats. A constant positive count series therefore gives zero. Fewer than `MIN_POINTS_PER_KL` valid samples gives a missing value, as does an initially zero mean.

Histogram bars are a separate visualization. `bin_size` sets their width in photon counts. Its default, `'auto'`, applies the Freedman–Diaconis rule per channel, rounded to an integer with a minimum width of one. The plotted Poisson overlay is the PMF multiplied by the sample count and bin width. Changing the bar width leaves the KL calculation unchanged.

## Select the analysis interval

- `TIME_RANGE` applies inclusive bounds in seconds on the input time axis. `ENERGY_RANGES` keeps the union of inclusive wavenumber ranges. Either endpoint may be `None`.
- `choose_static_distribution = True` compares the mean and population variance between halves of the raw intensity trace nearest `static_parameters['wavenumber']`. It discards one leading sample per attempt and keeps the first interval satisfying both relative-change limits and the minimum number of observations per half. An unsuccessful search raises an error. With the default `False`, the comparison is logged once.
- `SLIDING_WINDOW_LENGTH` and `SLIDING_STEP` are in spectra. Only complete windows are evaluated. Exported time starts at the first retained spectrum, and each window is labeled by the midpoint of its two endpoint times.
- `HISTOGRAM_ENERGIES` selects the nearest retained channels for histograms and both time-trace figures. An empty list skips those figures. `N_JOBS = 1` evaluates windows serially. The default `None` uses the available cores, up to the number of windows.

## Output files

All paths below are relative to the output directory for the script that produced them.

| Output | Contents |
| --- | --- |
| `csv/global_KL_spectrum_full_QSS.csv` | One row per retained channel, with columns `Energy (cm^-1)` and `Global KL divergence`. “Full” refers to the entire selected interval. |
| `csv/sliding_KL_matrix.csv` | Wavenumbers in the first column. Subsequent column headers are window-center times in seconds. Empty cells represent undefined KL values. |
| `csv/gain_calibration.csv` | Wavenumber, final gain, measured gain and inclusion flag (`energy,gain,gain_measured,used`). |
| `csv/gain_calibration_intensity.csv` | Optional table ordered by channel mean intensity (`intensity,gain,gain_measured,used`). |
| `png/`, `svg/`, `html/` | Calibration plots or analysis figures. The latter include the global spectrum, sliding heatmap (interactive surface in HTML), and the optional channel histograms and traces. |
| `*_parameters.py`, `*.log` | Configured settings and optional runtime log. The settings file records the requested interval. When static selection is enabled, the log reports the interval actually used. |

PNG figures default to 600 dpi and SVG labels remain editable text. HTML figures load Plotly.js from a CDN, so viewing them requires network access. The interactive surface displays up to 250 samples per axis. The CSV preserves every computed cell.

## Source files

`windowed_kl.py` and `gain_calibration.py` are the entry points. Shared loading, calibration, KL calculations and exports live in `windowed_kl_utils.py`. The `_matplotlib.py` and `_plotly.py` modules construct the figures, and `styling.py` defines their palette, typography and export settings.
