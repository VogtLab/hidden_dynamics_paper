"""Configure and run detector-gain calibration from a stable reference."""

# Stable single-beam reference file in XYYYY CSV format. The timestamp and times
# form the header. Each subsequent row contains a wavenumber (cm^-1) and its raw
# intensities.
reference_csv_path:str = ''

# Output directory for calibration tables in csv/, plots in html/, png/ and svg/,
# the settings snapshot and the optional log.
output_dir:str = ''

# Filename within csv/ for the energy-gain table. Columns are energy, gain,
# gain_measured and used.
gain_csv_name:str = 'gain_calibration.csv'

# When True, also calibrate gain against channel mean intensity using the selected
# gain mode.
calibrate_gain_per_intensity:bool = True

# Filename within csv/ for the optional intensity-gain table. Columns are intensity,
# gain, gain_measured and used, ordered by increasing intensity.
gain_intensity_csv_name:str = 'gain_calibration_intensity.csv'

# Horizontal axis title for the gain-versus-mean-intensity plot.
intensity_label:str = 'Mean intensity (arb. units)'

# Optional file-log level ('DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'). None
# disables the file. Console output starts at INFO.
log_level:str | None = 'INFO'

# Inclusive start and stop in seconds on the reference time axis. Use None for the
# entire record or either open endpoint.
time_range:tuple[float | None, float | None] | None = None

# Inclusive wavenumber intervals (cm^-1) to exclude from the fit. Use None for an
# open endpoint and [] for no exclusions. Cubic modes still evaluate excluded
# channels. Raw mode omits them.
exclude_energies:list[tuple[float | None, float | None]] = []

# Mode for both gain tables. The accepted strings are:
# 'fit_cubic_full' uses the fitted positive cubic at every channel.
# 'fig_cubic_interpolate' keeps valid included measurements and uses the cubic for
# the other channels.
# 'raw' exports only valid included measurements and skips fitting.
approximate_gain:str = 'fit_cubic_full'

# Horizontal axis title for the energy-gain plot. Use cm$^{-1}$ for inverse
# centimetres.
energy_label:str = 'Energy / wavenumber (cm$^{-1}$)'

# Display the energy axis from high to low when True.
invert_energy_axis:bool = True

# Required version string from styling.py. Both scripts check for an exact match.
expected_styling_version:str = '20260921_2104'


if __name__ == '__main__':
    from windowed_kl_utils import *
    import styling

    config = GainCalibrationConfig(
        reference_csv_path=reference_csv_path,
        output_dir=output_dir,
        gain_csv_name=gain_csv_name,
        calibrate_gain_per_intensity=calibrate_gain_per_intensity,
        gain_intensity_csv_name=gain_intensity_csv_name,
        intensity_label=intensity_label,
        dpi=styling.PNG_DPI,
        log_level=log_level,
        time_range=time_range,
        exclude_energies=exclude_energies,
        approximate_gain=approximate_gain,
        energy_label=energy_label,
        invert_energy_axis=invert_energy_axis,
    )
    export_user_parameters(
        output_dir, {name: globals()[name] for name in __annotations__},
        'gain_calibration_parameters.py',
    )
    verify_styling_version(expected_styling_version)
    configure_logging(config.log_level, config.output_dir, 'gain_calibration.log')
    run_gain_calibration(config)
