"""Configure and run photon-count KL analysis of a spectral time series."""

# Operando spectrum file in XYYYY CSV format. The header contains a UTC timestamp
# and times in seconds. Each data row starts with a wavenumber (cm^-1), followed by
# one raw intensity per time.
OPERANDO_CSV_PATH:str = ''

# Output directory for CSV tables, HTML plots, PNG/SVG figures, the settings
# snapshot and the optional log.
OUTPUT_DIR:str = ''

# Dataset name to show in spectrum and trace titles. An empty string omits the name.
CONDITION_LABEL:str = ''

# Requested wavenumbers (cm^-1) for count histograms, raw intensity traces and
# sliding KL traces. Each maps to the nearest retained channel. Use [] to skip these
# three plots.
HISTOGRAM_ENERGIES:list[float] = []

# Width of each histogram bar in photon counts. Set a positive integer or 'auto' for
# the integer-rounded Freedman–Diaconis width of each channel. This setting affects
# only the histogram display.
bin_size:int | str = 'auto'

# Retain the union of inclusive wavenumber ranges (cm^-1). Use None for the entire
# axis or for either open endpoint, e.g. [(1800.0, None)].
ENERGY_RANGES:list[tuple[float | None, float | None]] | None = None

# Inclusive start and stop in seconds on the input time axis. Use None for the
# entire record or either open endpoint, e.g. (100.0, None).
TIME_RANGE:tuple[float | None, float | None] | None = None

# Detector gain in intensity units per photon count. Supply a positive number or
# select a method:
# 'single_channel' uses variance/mean at NOISE_ENERGY.
# 'regression' fits a photon-transfer slope across channels.
# 'reference_energy_gain' looks up gain by wavenumber.
# 'reference_intensity_gain' looks up gain by channel mean intensity.
# The legacy spelling 'reference' selects 'reference_energy_gain'.
GAIN_ESTIMATE:float | str = 'reference_energy_gain'

# Noise-channel wavenumber (cm^-1) for 'single_channel'. Use a number for this
# method. Other methods ignore this setting and accept None.
NOISE_ENERGY:float | None = 2627

# Gain table for either reference method. Supply a CSV with energy,gain or
# intensity,gain columns, respectively. Interpolation is linear between tabulated
# values and clamps at the endpoints. Other gain methods ignore this path.
REFERENCE_CSV_PATH:str = ''

# Optional file-log level ('DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'). None
# disables the file. Console output starts at INFO.
LOG_LEVEL:str | None = None

# Number of consecutive spectra per window, from 1 to the retained record length.
# Use at least MIN_POINTS_PER_KL for defined KL values.
SLIDING_WINDOW_LENGTH:int = 100

# Number of spectra between successive window starts (positive integer).
SLIDING_STEP:int = 1

# Wavenumber axis title. The plotting modules support the cm$^{-1}$ superscript in
# both static and interactive figures.
ENERGY_LABEL:str = 'Energy / wavenumber (cm$^{-1}$)'

# Display energy axes from high to low when True.
INVERT_ENERGY_AXIS:bool = True

# Nonnegative padding in photon counts. The upper support reaches the larger of the
# observed maximum and mean plus padding, subject to MAX_HISTOGRAM_SUPPORT.
# Histogram bins start near the observed minimum minus padding, bounded below by
# zero.
X_RANGE_BUFFER:int = 200

# Largest photon count to retain (nonnegative integer), or None for no cap. The
# Poisson mean is recalculated after this cutoff.
MAX_HISTOGRAM_SUPPORT:int | None = None

# Minimum number of finite observations per channel and window (positive integer).
# Shorter series receive NaN.
MIN_POINTS_PER_KL:int = 8

# When True, discard initial spectra one at a time until the selected channel meets
# both half-to-half thresholds. False logs the changes once and retains the full
# time selection.
choose_static_distribution:bool = False

# Settings for comparing the two halves of the raw intensity trace.
# wavenumber: target in cm^-1, matched to the nearest retained channel.
# delta_mean, delta_var: maximum absolute relative changes in the mean and
# population variance (0.05 = 5%). For a zero first-half value the change is zero if
# both values are zero, otherwise infinite.
# minimal_points_number: minimum observations in each half during selection. An
# odd-length record puts the extra observation in the second half.
static_parameters:dict = {
    'wavenumber': 2078.0,
    'delta_mean': 0.05,
    'delta_var': 0.05,
    'minimal_points_number': 100,
}

# Worker processes for sliding KL. Use 1 for serial execution, a positive count for
# a process pool, or None or a nonpositive value for all available cores.
N_JOBS:int | None = None

# Apply a logarithmic colour scale to the static KL heatmap and log10 heights to the
# interactive surface when True. Exported KL values stay on their original scale.
PLOT_HEATMAP_LOG_SCALE:bool = False

# Positive floor for logarithmic KL display.
EPS:float = 1e-12

# Required version string from styling.py. Both scripts check for an exact match.
expected_styling_version:str = '20260921_2104'


if __name__ == '__main__':
    from windowed_kl_utils import *

    import styling

    config = Config(
        operando_csv_path=OPERANDO_CSV_PATH,
        output_dir=OUTPUT_DIR,
        condition_label=CONDITION_LABEL,
        dpi=styling.PNG_DPI,
        log_level=LOG_LEVEL,
        histogram_energies=HISTOGRAM_ENERGIES,
        sliding_window_length=SLIDING_WINDOW_LENGTH,
        sliding_step=SLIDING_STEP,
        gain_estimate=GAIN_ESTIMATE,
        noise_energy=NOISE_ENERGY,
        reference_csv_path=REFERENCE_CSV_PATH,
        time_range=TIME_RANGE,
        energy_ranges=ENERGY_RANGES,
        energy_label=ENERGY_LABEL,
        invert_energy_axis=INVERT_ENERGY_AXIS,
        x_range_buffer=X_RANGE_BUFFER,
        max_histogram_support=MAX_HISTOGRAM_SUPPORT,
        min_points_per_kl=MIN_POINTS_PER_KL,
        choose_static_distribution=choose_static_distribution,
        static_parameters=static_parameters,
        n_jobs=N_JOBS,
        plot_heatmap_log_scale=PLOT_HEATMAP_LOG_SCALE,
        eps=EPS,
        bin_size=bin_size,
    )
    export_user_parameters(
        OUTPUT_DIR, {name: globals()[name] for name in __annotations__},
        'windowed_kl_parameters.py',
    )
    verify_styling_version(expected_styling_version)
    configure_logging(config.log_level, config.output_dir, 'windowed_kl.log')
    run_pipeline(config)
