"""Read spectral time series, calibrate gain and compute photon-count KL divergence."""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.optimize import minimize
from scipy.special import gammaln, rel_entr, xlogy
from scipy.stats import poisson
from scipy.stats import t as student_t

import styling
import windowed_kl_matplotlib as mpl_plots
import windowed_kl_plotly as plotly_plots

logger = logging.getLogger(__name__)

# Use medians from populated bins for the photon-transfer regression.
_PTC_N_BINS = 32
_PTC_MIN_PER_BIN = 3

# Both gain-intensity bands use the same two-sided confidence level.
_GAIN_CI_LEVEL = 0.95

# Forked workers inherit these inputs. Other start methods initialize them per process.
_WORKER_STATE: dict = {}


@dataclass
class Config:
    """Settings and output paths for the spectral time-series analysis.

    Parameters
    ----------
    operando_csv_path : str
        Path to the operando XYYYY csv dataset.
    output_dir : str
        Root directory for all outputs; html/png/svg subfolders are derived.
    condition_label : str, default='KBr reference'
        Dataset name included in spectrum and trace titles.
    dpi : int, default=300
        Raster resolution for PNG exports.
    log_level : Optional[str], default=None
        Logging verbosity name for the log file (e.g. 'DEBUG', 'INFO',
        'WARNING'), or None to disable file logging. Stdout logging is
        always active at INFO and is not affected by this setting.
    histogram_energies : Sequence[float], default=()
        Energies in cm^-1 at which photon-count histograms are drawn.
    sliding_window_length : int, default=200
        Number of timepoints per sliding window.
    sliding_step : int, default=1
        Step in timepoints between successive window starts.
    gain_estimate : object, default='single_channel'
        A positive float (fixed gain) or one of 'single_channel',
        'regression', 'reference_energy_gain', 'reference_intensity_gain'.
        ('reference' is accepted as a deprecated alias of
        'reference_energy_gain'.)
    noise_energy : Optional[float], default=2627.0
        Noise-channel energy in cm^-1 for the 'single_channel' strategy.
    reference_csv_path : Optional[str], default=None
        Gain-calibration csv for the two 'reference_*' strategies, as exported
        by gain_calibration.py. For 'reference_energy_gain' it must hold
        'energy' and 'gain' columns; for 'reference_intensity_gain' it must hold
        'intensity' and 'gain' columns.
    time_range : Optional[Tuple[Optional[float], Optional[float]]], default=None
        (start, stop) seconds to crop the record, or None to keep all. Either
        side of the tuple may itself be None for an open-ended bound (e.g.
        (100.0, None) keeps everything from 100 s onward).
    energy_label : str, default='Energy / wavenumber (cm$^{-1}$)'
        Energy-axis label (matplotlib mathtext; converted to HTML for plotly).
    invert_energy_axis : bool, default=True
        If True energy axes are drawn high-to-low.
    energy_ranges : Optional[Sequence[Tuple[Optional[float], Optional[float]]]], default=None
        (min, max) cm^-1 ranges to KEEP (their union), or None to keep the
        entire energy axis unfiltered. Either side of a given tuple may be
        None for an open-ended bound (e.g. (None, 2000.0) keeps everything up
        to 2000 cm^-1; (450.0, None) keeps everything from 450 cm^-1 up).
    x_range_buffer : float, default=200.0
        Count margin above the mean for the upper bound and below the observed minimum for histogram display.
    max_histogram_support : Optional[int], default=None
        Hard cap on the count support, or None for no cap.
    min_points_per_kl : int, default=8
        Minimum valid samples required to compute a KL value.
    choose_static_distribution : bool, default=False
        Trim initial samples until the selected raw-intensity trend passes
        the two-half mean and variance checks. If False, only log changes.
    static_parameters : dict
        wavenumber (nearest retained channel, cm^-1), delta_mean and delta_var
        (nonnegative absolute relative limits), minimal_points_number
        (positive integer minimum per half when selection is enabled).
    n_jobs : Optional[int], default=None
        Worker processes: None/<=0 all cores, 1 serial, N>1 that many.
    plot_heatmap_log_scale : bool, default=True
        If True the heatmap colour is mapped logarithmically.
    eps : float, default=1e-12
        Small positive floor for logarithmic colour mapping.
    bin_size : int or str, default='auto'
        Positive histogram bin width in photon counts, or 'auto' to use
        the Freedman-Diaconis rule for each channel.
    """
    operando_csv_path:str
    output_dir:str
    condition_label:str = 'KBr reference'
    dpi:int = 300
    log_level:Optional[str] = None
    histogram_energies:Sequence[float] = field(default_factory=tuple)
    sliding_window_length:int = 200
    sliding_step:int = 1
    gain_estimate:object = 'single_channel'
    noise_energy:Optional[float] = 2627.0
    reference_csv_path:Optional[str] = None
    time_range:Optional[Tuple[Optional[float], Optional[float]]] = None
    energy_label:str = 'Energy / wavenumber (cm$^{-1}$)'
    invert_energy_axis:bool = True
    energy_ranges:Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None
    x_range_buffer:float = 200.0
    max_histogram_support:Optional[int] = None
    min_points_per_kl:int = 8
    choose_static_distribution:bool = False
    static_parameters:dict = field(default_factory=lambda: {
        'wavenumber': 2078.0,
        'delta_mean': 0.05,
        'delta_var': 0.05,
        'minimal_points_number': 100,
    })
    n_jobs:Optional[int] = None
    plot_heatmap_log_scale:bool = True
    eps:float = 1e-12
    bin_size:int | str = 'auto'

    html_dir:str = field(init=False, default='')
    png_dir:str = field(init=False, default='')
    svg_dir:str = field(init=False, default='')
    csv_dir:str = field(init=False, default='')

    def __post_init__(self) -> None:
        """Resolve the figure directories under the configured output path."""
        self.html_dir = os.path.join(self.output_dir, 'html')
        self.png_dir = os.path.join(self.output_dir, 'png')
        self.svg_dir = os.path.join(self.output_dir, 'svg')
        self.csv_dir = os.path.join(self.output_dir, 'csv')


@dataclass
class GainCalibrationConfig:
    """Settings and output paths for reference-based detector calibration.

    Parameters
    ----------
    reference_csv_path : str
        Path to the Single Beam XYYYY csv recorded on a temporally stable target.
    output_dir : str
        Root directory for outputs; html/png/svg subfolders are derived.
    gain_csv_name : str, default='gain_calibration.csv'
        Filename of the exported per-channel energy-gain table g(E).
    calibrate_gain_per_intensity : bool, default=True
        If True a second calibration is produced in addition to g(E): the gain
        is fitted as a function of each channel's mean intensity, g(mean), and
        exported and plotted alongside the energy calibration.
    gain_intensity_csv_name : str, default='gain_calibration_intensity.csv'
        Filename of the exported gain-vs-intensity table. It carries the columns
        intensity, gain, gain_measured and used; only intensity and gain are
        read back by the 'reference_intensity_gain' gain strategy.
    intensity_label : str, default='Mean intensity (arb. units)'
        Intensity-axis label for the g(mean) plot (matplotlib mathtext allowed;
        converted to HTML for the plotly figure).
    dpi : int, default=300
        Raster resolution for PNG exports.
    log_level : str, default='INFO'
        Logging verbosity name (e.g. 'DEBUG', 'INFO', 'WARNING').
    time_range : Optional[Tuple[Optional[float], Optional[float]]], default=None
        (start, stop) seconds to crop the reference record on import, or None to
        keep the entire record. Either side of the tuple may itself be None for
        an open-ended bound (e.g. (100.0, None) keeps everything from 100 s
        onward).
    exclude_energies : Sequence[Tuple[Optional[float], Optional[float]]], default=()
        Inclusive energy ranges in cm^-1 omitted only from fitting. None
        endpoints are unbounded. Empty fits every valid channel; excluded
        channels remain with fitted gains in cubic modes, and are omitted in raw mode.
    approximate_gain : str, default='fig_cubic_interpolate'
        'fit_cubic_full' fits all output gains; 'fig_cubic_interpolate' keeps
        analyzed measurements and fills remaining channels from the cubic.
        'raw' plots/exports only analyzed measurements without any cubic fit.
    energy_label : str, default='Energy / wavenumber (cm$^{-1}$)'
        Energy-axis label (matplotlib mathtext; converted to HTML for plotly).
    invert_energy_axis : bool, default=True
        If True energy axes are drawn high-to-low.
    """
    reference_csv_path:str
    output_dir:str
    gain_csv_name:str = 'gain_calibration.csv'
    calibrate_gain_per_intensity:bool = True
    gain_intensity_csv_name:str = 'gain_calibration_intensity.csv'
    intensity_label:str = 'Mean intensity (arb. units)'
    dpi:int = 300
    log_level:str | None = 'INFO'
    time_range:Optional[Tuple[Optional[float], Optional[float]]] = None
    exclude_energies:Sequence[Tuple[Optional[float], Optional[float]]] = field(default_factory=tuple)
    approximate_gain:str = 'fig_cubic_interpolate'
    energy_label:str = 'Energy / wavenumber (cm$^{-1}$)'
    invert_energy_axis:bool = True

    html_dir:str = field(init=False, default='')
    png_dir:str = field(init=False, default='')
    svg_dir:str = field(init=False, default='')

    def __post_init__(self) -> None:
        """Resolve the figure directories under the configured output path."""
        self.html_dir = os.path.join(self.output_dir, 'html')
        self.png_dir = os.path.join(self.output_dir, 'png')
        self.svg_dir = os.path.join(self.output_dir, 'svg')


def verify_styling_version(expected_styling_version:str) -> None:
    """Check that the figure settings match the version expected by the script.

    Parameters
    ----------
    expected_styling_version : str
        Required exact value of styling.version.
    """
    actual_version = getattr(styling, 'version', None)
    if actual_version != expected_styling_version:
        raise RuntimeError(
            f'Styling version mismatch: expected {expected_styling_version!r}, '
            f'got {actual_version!r}.')


def export_user_parameters(out_dir:str, parameters:dict[str, object], filename:str) -> str:
    """Write the supplied settings as an importable Python file.

    Parameters
    ----------
    out_dir : str
        Output directory to create if missing.
    parameters : dict[str, object]
        User settings keyed by their original names.
    filename : str
        Parameter snapshot filename ending in .py.

    Returns
    -------
    path : str
        Written parameter snapshot.
    """
    if not out_dir:
        raise ValueError('Set the output directory in the entry script before running it.')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as stream:
        for name, value in parameters.items():
            stream.write(f'{name} = {value!r}\n')
    return path


def configure_logging(
    log_level:Optional[str],
    output_dir:str,
    log_filename:str,
) -> None:
    """Send INFO messages and above to stdout and optionally log to a file.

    Parameters
    ----------
    log_level : Optional[str]
        Logging level name for the file handler (e.g. 'DEBUG', 'INFO',
        'WARNING'). If None, no file handler is added and only stdout
        logging (fixed at INFO) is active.
    output_dir : str
        Directory the log file is written to. Created if missing. Ignored
        if log_level is None.
    log_filename : str
        Filename of the log file (e.g. 'windowed_kl.log'). Ignored
        if log_level is None.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(stdout_handler)

    if log_level is not None:
        file_level = getattr(logging, str(log_level).upper(), logging.INFO)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
                              datefmt='%Y.%m.%d %H:%M:%S'))
        root_logger.addHandler(file_handler)


def ensure_output_dirs(config:'Config | GainCalibrationConfig') -> None:
    """Create the output root and the three figure directories.

    Parameters
    ----------
    config : Config or GainCalibrationConfig
        Output paths for the current analysis.
    """
    for path in (config.output_dir, config.html_dir, config.png_dir, config.svg_dir):
        os.makedirs(path, exist_ok=True)


def in_any_range(
    values:np.ndarray,
    ranges:Sequence[Tuple[Optional[float], Optional[float]]],
) -> np.ndarray:
    """Return the union of inclusive intervals, allowing open or reversed bounds.

    Parameters
    ----------
    values : ndarray, shape (n,)
        Values to test.
    ranges : Sequence[Tuple[Optional[float], Optional[float]]]
        Iterable of (min, max) pairs (order-insensitive). Either side of a
        pair may be None for an open-ended (unbounded) bound.

    Returns
    -------
    mask : ndarray, shape (n,)
        Boolean mask, True where a value lies in some range.
    """
    values = np.asarray(values, dtype=float)
    mask = np.zeros(values.shape, dtype=bool)
    for lo, hi in ranges:
        lo_f = -np.inf if lo is None else float(lo)
        hi_f = np.inf if hi is None else float(hi)
        a, b = min(lo_f, hi_f), max(lo_f, hi_f)
        mask |= (values >= a) & (values <= b)
    return mask


def log_negative_intensity_regions(
    filename:str,
    time_series:np.ndarray,
    energy:np.ndarray,
    time:np.ndarray,
) -> None:
    """Log the time and wavenumber bounds of each connected negative region.

    Cells that touch along an edge or a diagonal belong to the same region.

    Parameters
    ----------
    filename : str
        Path of the dataset being validated, for the log message.
    time_series : ndarray, shape (n_time, n_energy)
        Intensity matrix laid out as (time, energy).
    energy : ndarray, shape (n_energy,)
        Energy axis in cm^-1, matching the second dimension of time_series.
    time : ndarray, shape (n_time,)
        Time axis in seconds, matching the first dimension of time_series.
        Must be on the full-input (original, un-re-zeroed) time scale so
        that reported spot time ranges remain interpretable against the
        original file, even if the caller applied a time-range crop.
    """
    negative_mask = time_series < 0
    n_negative = int(np.count_nonzero(negative_mask))
    if n_negative == 0:
        return

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, n_spots = ndi_label(negative_mask, structure=structure)

    lines = [
        f"Negative intensity values found in '{filename}': "
        f"{n_negative} value(s) in {n_spots} spot(s); "
        f"minimum value = {float(time_series.min()):.6g}."
    ]
    for spot_id in range(1, n_spots + 1):
        time_idx, energy_idx = np.where(labeled == spot_id)
        e_lo, e_hi = float(energy[energy_idx].min()), float(energy[energy_idx].max())
        t_lo, t_hi = float(time[time_idx].min()), float(time[time_idx].max())
        lines.append(
            f"  - spot {spot_id}: {energy_idx.size} value(s), "
            f"energy range [{e_lo:g}, {e_hi:g}] cm^-1, "
            f"time range [{t_lo:g}, {t_hi:g}] s"
        )
    logger.warning('\n'.join(lines))


def import_data(
    filename:str,
    time_range:Optional[Tuple[Optional[float], Optional[float]]],
    energy_ranges:Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    rezero_time:bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a timestamped XYYYY intensity matrix and apply the requested crops.

    Time limits refer to the original header. Values are checked after both axes
    have been selected.

    Parameters
    ----------
    filename : str
        Path to the XYYYY csv (header timestamp + times, then energy rows).
    time_range : Optional[Tuple[Optional[float], Optional[float]]]
        (start, stop) seconds to crop the record, or None to keep all. Either
        side may itself be None for an open-ended bound (e.g. (100.0, None)
        keeps everything from 100 s onward).
    energy_ranges : Optional[Sequence[Tuple[Optional[float], Optional[float]]]], default=None
        (min, max) cm^-1 ranges to KEEP (their union), or None to keep the
        entire energy axis unfiltered. Either side of a given tuple may be
        None for an open-ended bound.

    rezero_time : bool, default=True
        Subtract the first retained time. False preserves the input time
        coordinates for static-range selection and logging.

    Returns
    -------
    energy : ndarray, shape (n_energy,)
        Energy axis in cm^-1, after energy-range selection.
    time : ndarray, shape (n_time,)
        Time axis in seconds, re-zeroed when rezero_time is True.
    time_series : ndarray, shape (n_time, n_energy)
        Intensity matrix laid out as (time, energy), after time and energy
        selection.
    """
    if not filename:
        raise ValueError('Set the input CSV path in the entry script before running it.')
    logger.info('Importing data...')

    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
    timestamp_str = header[0]
    timestamp_format = '%d.%m.%Y %H:%M:%S UTC'
    timestamp = datetime.strptime(timestamp_str, timestamp_format)
    time = np.array([float(t) for t in header[1:]])
    data = np.loadtxt(filename, delimiter=',', skiprows=1, ndmin=2)
    if not time.size or data.shape[1] != time.size + 1:
        raise ValueError('The CSV must contain one intensity per channel for every time in the header.')
    energy = data[:, 0]
    time_series = data[:, 1:].T

    if time_range is not None:
        t_start = -np.inf if time_range[0] is None else float(time_range[0])
        t_stop = np.inf if time_range[1] is None else float(time_range[1])
        mask = (time >= t_start) & (time <= t_stop)
        if mask.any():
            start_idx = int(mask.argmax())
            finish_idx = len(mask) - 1 - int(mask[::-1].argmax())
            logger.info(f"Filtered dataset by time: start index {start_idx}, finish index {finish_idx}")
        else:
            raise RuntimeError('No data found within the specified time range.')
        time = time[mask]
        time_series = time_series[mask, :]
        logger.info(f"Applied time filter: {time_range}")


    if energy_ranges is not None:
        energy_mask = in_any_range(energy, energy_ranges)
        n_kept = int(np.count_nonzero(energy_mask))
        if n_kept == 0:
            raise RuntimeError('No energy channels found within the specified energy ranges.')
        logger.info(
            f"Filtered dataset by energy: kept {n_kept}/{energy.size} channel(s) "
            f"within {energy_ranges}"
        )
        energy = energy[energy_mask]
        time_series = time_series[:, energy_mask]

    if np.isnan(time_series).any():
        nan_positions = np.argwhere(np.isnan(time_series))
        error_msg = f"Found {len(nan_positions)} NaN values in raw time_series data.\n"
        error_msg += 'Positions (time_index, energy_index):\n'
        max_print = 20
        for i, pos in enumerate(nan_positions):
            if i >= max_print:
                error_msg += f"  ... and {len(nan_positions) - max_print} more."
                break
            error_msg += f"  - Time Index: {pos[0]}, Energy Index: {pos[1]}\n"
        logger.error(error_msg)
        raise ValueError(f"Dataset '{filename}' contains NaN values. Execution halted.")

    if np.isinf(time_series).any():
        raise ValueError(f'Dataset {filename!r} contains infinite intensities in the selected range.')
    time_original_scale = time
    if rezero_time:
        time = time - time[0]

    logger.info('First spectrum was taken on %s', timestamp.strftime(timestamp_format))

    logger.debug(
        f"File '{filename}': energy range [{float(energy.min()):g}, {float(energy.max()):g}] cm^-1, "
        f"time range [{float(time.min()):g}, {float(time.max()):g}] s"
    )

    if np.any(time_series <= 0):
        log_negative_intensity_regions(filename, time_series, energy, time_original_scale)
        raise RuntimeError('Retained intensities must be strictly positive.')

    logger.debug(f"{time_series.shape = }")
    logger.debug(f"{energy.shape = }")
    logger.debug(f"{time.shape = }")

    return energy, time, time_series


def estimate_gain_single_channel(
    spectra:np.ndarray,
    energies:np.ndarray,
    noise_energy:float,
) -> float:
    """Estimate gain from population variance divided by mean intensity at the nearest
    noise channel.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Intensity matrix laid out as (energy, time).
    energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1.
    noise_energy : float
        Target noise-channel energy; the nearest channel is used.

    Returns
    -------
    gain : float
        Estimated scalar gain g = var / mean.
    """
    energies = np.asarray(energies, dtype=float)
    idx = int(np.argmin(np.abs(energies - float(noise_energy))))
    channel = np.asarray(spectra[idx, :], dtype=float)
    mean_val = float(np.mean(channel))
    var_val = float(np.var(channel))
    logger.info(
        f"Gain (single_channel): requested {noise_energy} -> nearest "
        f"{energies[idx]:.2f} cm^-1 (index {idx})"
    )
    if mean_val <= 0.0:
        raise RuntimeError(f"Mean intensity at the noise channel is {mean_val:.4g}; cannot compute g = var/mean.")
    if var_val <= 0.0:
        raise RuntimeError(f"Variance at the noise channel is {var_val:.4g}; cannot compute g = var/mean.")
    gain = var_val / mean_val
    logger.info(f"Estimated gain g = var/mean = {gain:.6g} (mean={mean_val:.6g}, var={var_val:.6g})")
    return gain


def estimate_gain_regression(spectra:np.ndarray) -> float:
    """Fit noise variance against mean intensity across channels.

    Half the mean squared difference between consecutive spectra estimates the noise
    variance. Fit a line to the median values in occupied intensity bins with at
    least three channels each.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Intensity matrix laid out as (energy, time).

    Returns
    -------
    gain : float
        Fitted gain (slope of noise variance vs mean).
    """
    arr = np.asarray(spectra, dtype=float)
    n_time = arr.shape[1]
    if n_time < 3:
        raise RuntimeError(f"Need at least 3 time samples for the regression, got {n_time}.")

    # Per-channel signal level and difference-based (signal-cancelling) noise variance.
    means = arr.mean(axis=1)
    diffs = np.diff(arr, axis=1)
    noise_var = 0.5 * np.mean(diffs * diffs, axis=1)

    # Bin channels by mean intensity and take robust median summaries per bin.
    lo, hi = float(means.min()), float(means.max())
    if hi <= lo:
        raise RuntimeError('All channels share the same mean; cannot fit a photon-transfer curve.')
    edges = np.linspace(lo, hi, _PTC_N_BINS + 1)
    bin_idx = np.clip(np.digitize(means, edges) - 1, 0, _PTC_N_BINS - 1)
    bin_mean, bin_var = [], []
    for b in range(_PTC_N_BINS):
        sel = bin_idx == b
        if int(np.count_nonzero(sel)) >= _PTC_MIN_PER_BIN:
            bin_mean.append(float(np.median(means[sel])))
            bin_var.append(float(np.median(noise_var[sel])))
    if len(bin_mean) < 2:
        raise RuntimeError(
            f"Only {len(bin_mean)} usable intensity bins; cannot fit a line "
            "(try 'single_channel' or a constant gain)."
        )

    # Least-squares line var = gain * mean + read-noise variance.
    slope, intercept = np.polyfit(np.asarray(bin_mean), np.asarray(bin_var), 1)
    logger.info(
        f"Gain (regression): photon-transfer fit over {len(bin_mean)} bins -> "
        f"gain (slope) = {slope:.6g}, read-noise variance (intercept) = {intercept:.6g}"
    )
    if slope <= 0.0:
        raise RuntimeError(
            f"Fitted gain (slope) is {slope:.6g} <= 0; the photon-transfer model does not hold "
            "(try 'single_channel' or a constant gain)."
        )
    return float(slope)


def _read_gain_table(reference_path:str, axis_column:str) -> tuple[np.ndarray, np.ndarray]:
    """Read finite calibration points sorted by their lookup coordinate.

    Parameters
    ----------
    reference_path : str
        Calibration CSV containing the lookup coordinate and gain.
    axis_column : str
        Coordinate column, either 'energy' or 'intensity'.

    Returns
    -------
    coordinate, gain : ndarray, shape (n_valid,)
        Paired coordinates and measured or fitted gains in ascending coordinate order.
    """
    table = pd.read_csv(reference_path)
    missing = {axis_column, 'gain'} - set(table.columns)
    if missing:
        raise RuntimeError(f'Gain table {reference_path} is missing columns {sorted(missing)}.')
    coordinate = np.asarray(table[axis_column], dtype=float)
    gain = np.asarray(table['gain'], dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(gain)
    if not finite.all():
        logger.warning('Dropping %d non-finite rows from the gain table.', np.count_nonzero(~finite))
    coordinate, gain = coordinate[finite], gain[finite]
    if not coordinate.size:
        raise RuntimeError(f'Gain table {reference_path} contains no finite rows.')
    order = np.argsort(coordinate)
    return coordinate[order], gain[order]


def estimate_gain_reference(reference_path:str, energies:np.ndarray) -> np.ndarray:
    """Interpolate tabulated gains onto the spectral axis.

    Use the nearest endpoint gain for channels outside the calibrated range.

    Parameters
    ----------
    reference_path : str
        Path to the gain-calibration csv with ``energy`` and ``gain`` columns.
    energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1 of the dataset being analysed.

    Returns
    -------
    gain_map : ndarray, shape (n_energy,)
        Strictly positive per-channel gain vector aligned with energies.
    """
    logger.info(f"Loading per-channel gain curve from {reference_path}...")
    ref_energy, ref_gain = _read_gain_table(reference_path, 'energy')

    energies = np.asarray(energies, dtype=float)
    if ref_energy.shape != energies.shape or not np.allclose(ref_energy, np.sort(energies)):
        logger.warning(
            f"Gain csv has {ref_energy.size} channels spanning "
            f"[{ref_energy.min():.2f}, {ref_energy.max():.2f}] cm^-1 while the dataset has "
            f"{energies.size} spanning [{energies.min():.2f}, {energies.max():.2f}]; "
            f"interpolating the gain onto the dataset axis."
        )
        outside = (energies < ref_energy.min()) | (energies > ref_energy.max())
        if outside.any():
            logger.warning(
                f"{int(np.count_nonzero(outside))} dataset channel(s) lie outside the "
                f"tabulated energy range; their gain is clamped to the end values."
            )
    gain_map = np.interp(energies, ref_energy, ref_gain)

    if not np.all(np.isfinite(gain_map)) or np.any(gain_map <= 0.0):
        raise RuntimeError('The loaded gain curve must be finite and strictly positive.')

    logger.info(
        f"Per-channel gain over {gain_map.shape[0]} channels: "
        f"min={gain_map.min():.6g}, median={np.median(gain_map):.6g}, max={gain_map.max():.6g}"
    )
    return gain_map


def estimate_gain_reference_intensity(reference_path:str, spectra:np.ndarray) -> np.ndarray:
    """Interpolate gain at each channel’s mean intensity over the selected record.

    Channel means outside the calibration range use the nearest endpoint gain.

    Parameters
    ----------
    reference_path : str
        Path to the gain csv with ``intensity`` and ``gain`` columns.
    spectra : ndarray, shape (n_energy, n_time)
        Intensity matrix of the dataset laid out as (energy, time).

    Returns
    -------
    gain_map : ndarray, shape (n_energy,)
        Strictly positive per-channel gain aligned with the spectra rows.
    """
    logger.info(f"Loading gain-vs-intensity curve from {reference_path}...")
    ref_intensity, ref_gain = _read_gain_table(reference_path, 'intensity')

    channel_means = np.asarray(spectra, dtype=float).mean(axis=1)

    outside = (channel_means < ref_intensity.min()) | (channel_means > ref_intensity.max())
    if outside.any():
        logger.warning(
            f"{int(np.count_nonzero(outside))} channel(s) have a mean intensity outside the "
            f"tabulated range [{ref_intensity.min():.6g}, {ref_intensity.max():.6g}]; their gain "
            f"is clamped to the end values."
        )
    gain_map = np.interp(channel_means, ref_intensity, ref_gain)

    if not np.all(np.isfinite(gain_map)) or np.any(gain_map <= 0.0):
        raise RuntimeError('The loaded gain-vs-intensity curve must map to finite, strictly positive gains.')

    logger.info(
        f"Per-channel gain from intensity over {gain_map.shape[0]} channels: "
        f"min={gain_map.min():.6g}, median={np.median(gain_map):.6g}, max={gain_map.max():.6g}"
    )
    return gain_map


def resolve_gain(
    spectra:np.ndarray,
    energies:np.ndarray,
    config:'Config',
) -> np.ndarray | float:
    """Select the configured scalar or per-channel detector gain.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Intensity matrix laid out as (energy, time).
    energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1.
    config : Config
        Resolved pipeline configuration; supplies gain_estimate, noise_energy,
        the KL-scan settings and reference_csv_path.

    Returns
    -------
    gain : np.ndarray | float
        Scalar gain, or a per-channel vector of shape (n_energy,).
    """
    _valid = ("'single_channel', 'regression', 'reference_energy_gain' or "
              "'reference_intensity_gain'")
    gain_estimate = config.gain_estimate
    # bool is a subclass of int, so reject it explicitly before the numeric path.
    if isinstance(gain_estimate, bool):
        raise RuntimeError(f"gain_estimate must be a positive number, {_valid}.")
    if isinstance(gain_estimate, (int, float)):
        gain = float(gain_estimate)
        if gain <= 0.0:
            raise RuntimeError(f"Constant gain must be > 0, got {gain:.6g}.")
        logger.info(f"Using user-defined constant gain g = {gain:.6g}")
        return gain
    if gain_estimate == 'single_channel':
        if config.noise_energy is None:
            raise RuntimeError("gain_estimate == 'single_channel' requires noise_energy to be set.")
        return estimate_gain_single_channel(spectra, energies, config.noise_energy)
    if gain_estimate == 'regression':
        return estimate_gain_regression(spectra)
    if gain_estimate == 'reference':
        logger.warning(
            "gain_estimate == 'reference' is deprecated; treating it as "
            "'reference_energy_gain'. Update the setting to silence this warning."
        )
        gain_estimate = 'reference_energy_gain'
    if gain_estimate == 'reference_energy_gain':
        if not config.reference_csv_path:
            raise RuntimeError("gain_estimate == 'reference_energy_gain' requires reference_csv_path to be set.")
        return estimate_gain_reference(config.reference_csv_path, energies)
    if gain_estimate == 'reference_intensity_gain':
        if not config.reference_csv_path:
            raise RuntimeError("gain_estimate == 'reference_intensity_gain' requires reference_csv_path to be set.")
        return estimate_gain_reference_intensity(config.reference_csv_path, spectra)
    raise RuntimeError(
        f"Unknown gain_estimate {gain_estimate!r}; expected a positive number, {_valid}."
    )


def apply_count_scaling(spectra:np.ndarray, gain:np.ndarray | float) -> np.ndarray:
    """Divide each channel by its gain and round to the nearest integer count.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Intensity matrix laid out as (energy, time).
    gain : np.ndarray | float (shape: () or (n_energy,))
        Scalar gain, or a per-channel gain vector.

    Returns
    -------
    counts : ndarray, shape (n_energy, n_time)
        Integer-valued photon-count matrix.
    """
    gain_arr = np.asarray(gain, dtype=float)
    if not np.all(np.isfinite(gain_arr) & (gain_arr > 0.0)):
        raise RuntimeError('All gain values must be finite and strictly positive.')
    if gain_arr.ndim == 0:
        divisor = gain_arr
    elif gain_arr.ndim == 1:
        if gain_arr.shape[0] != spectra.shape[0]:
            raise RuntimeError(
                f"Per-channel gain has length {gain_arr.shape[0]} but data has "
                f"{spectra.shape[0]} channels."
            )
        divisor = gain_arr[:, None]
    else:
        raise RuntimeError(f"Gain must be a scalar or 1-D array, got ndim={gain_arr.ndim}.")
    counts = np.round(np.asarray(spectra, dtype=float) / divisor)
    logger.info(
        f"Applied count scaling: counts = round(intensity / gain); "
        f"range [{counts.min():.0f}, {counts.max():.0f}]"
    )
    return counts


def compute_poisson_kl_from_series(
    series:np.ndarray,
    min_points:int,
    x_range_buffer:float,
    max_support:Optional[int],
) -> float:
    """Calculate empirical-to-Poisson KL divergence on the observed counts.

    The Poisson rate is the mean of the retained integer counts. Normalize both
    distributions over the observed counts with positive numerical Poisson mass.
    Natural logarithms give the result in nats.

    Parameters
    ----------
    series : ndarray, shape (n_samples,)
        Count series for one channel/window.
    min_points : int
        Minimum valid samples required to return a value.
    x_range_buffer : float
        Count margin above the mean when choosing the upper support bound.
    max_support : Optional[int]
        Hard cap on the support, or None.

    Returns
    -------
    kl : float
        KL divergence of the empirical distribution from its matched Poisson,
        or NaN when the series is too small or degenerate.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]

    if series.size < min_points:
        return np.nan

    scaled = np.round(series).astype(int)

    lam = float(np.mean(scaled))
    if not np.isfinite(lam) or lam <= 0:
        return np.nan

    max_observed = int(np.max(scaled))
    max_range = int(max(max_observed, lam + float(x_range_buffer)))

    if max_support is not None:
        max_range = min(max_range, int(max_support))
        scaled = scaled[scaled <= max_range]
        if scaled.size < min_points:
            return np.nan
        lam = float(np.mean(scaled))

    # Only observed counts enter the divergence.
    in_support = (scaled >= 0) & (scaled <= max_range)
    vals, counts = np.unique(scaled[in_support], return_counts=True)
    total = counts.sum()
    if total == 0:
        return np.nan

    probs = counts.astype(float) / total

    # Evaluate the Poisson mass in log space before normalizing over the observed support.
    poisson_dist = np.exp(xlogy(vals, lam) - gammaln(vals + 1.0) - lam)

    mask = poisson_dist > 0
    if not np.any(mask):
        return np.nan

    f_obs = probs[mask] / probs[mask].sum()
    f_exp = poisson_dist[mask] / poisson_dist[mask].sum()
    return float(np.sum(rel_entr(f_obs, f_exp)))


def compute_global_kl_spectrum(
    spectra:np.ndarray,
    min_points:int,
    x_range_buffer:float,
    max_support:Optional[int],
) -> np.ndarray:
    """Evaluate the count-distribution divergence for every channel over the supplied
    time interval.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Count matrix laid out as (energy, time).
    min_points : int
        Minimum valid samples required per channel.
    x_range_buffer : float
        Count margin above the mean when choosing the upper support bound.
    max_support : Optional[int]
        Hard cap on the support, or None.

    Returns
    -------
    out : ndarray, shape (n_energy,)
        Per-channel KL divergence (NaN where undefined).
    """
    n_energy = spectra.shape[0]
    out = np.full(n_energy, np.nan, dtype=float)
    for i in range(n_energy):
        out[i] = compute_poisson_kl_from_series(spectra[i, :], min_points, x_range_buffer, max_support)
    return out


def make_sliding_windows(
    n_time:int,
    time_axis:np.ndarray,
    window_length:int,
    step:int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate complete windows and return the midpoint of their endpoint times.

    Parameters
    ----------
    n_time : int
        Number of timepoints in the record.
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    window_length : int
        Number of timepoints per window.
    step : int
        Step in timepoints between window starts.

    Returns
    -------
    starts : ndarray, shape (n_windows,)
        Window start indices.
    ends : ndarray, shape (n_windows,)
        Window end indices (inclusive).
    centers : ndarray, shape (n_windows,)
        Window centre times in seconds.
    """
    window_length, step = int(window_length), int(step)
    if not 1 <= window_length <= n_time or step < 1:
        raise ValueError(f'Sliding windows require 1 <= length <= {n_time} and a positive step.')
    starts = np.arange(0, n_time - window_length + 1, step, dtype=int)
    ends = starts + window_length - 1
    centers = np.array([(time_axis[a] + time_axis[b]) / 2.0 for a, b in zip(starts, ends)])
    return starts, ends, centers


def _resolve_n_jobs(n_jobs:Optional[int]) -> int:
    """Convert a worker setting into the requested number of processes.

    Parameters
    ----------
    n_jobs : Optional[int]
        None/<=0 for all cores, otherwise the requested count.

    Returns
    -------
    n : int
        Effective number of worker processes.
    """
    cores = os.cpu_count() or 1
    if n_jobs is None:
        return cores
    n = int(n_jobs)
    return cores if n <= 0 else n


def _init_kl_worker(spectra:np.ndarray, min_points:int, x_range_buffer:float, max_support:Optional[int]) -> None:
    """Store the count matrix and KL settings for use by a worker process.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Count matrix shared with the workers.
    min_points : int
        Minimum valid samples required per window.
    x_range_buffer : float
        Count margin above the mean when choosing the upper support bound.
    max_support : Optional[int]
        Hard cap on the support, or None.
    """
    _WORKER_STATE['spectra'] = spectra
    _WORKER_STATE['min_points'] = min_points
    _WORKER_STATE['x_range_buffer'] = x_range_buffer
    _WORKER_STATE['max_support'] = max_support


def _compute_kl_window_row(window_bounds:Tuple[int, int]) -> np.ndarray:
    """Evaluate all channels between two inclusive window indices.

    Parameters
    ----------
    window_bounds : Tuple[int, int]
        (start, end) inclusive time indices of the window.

    Returns
    -------
    row : ndarray, shape (n_energy,)
        KL divergence per energy for this window.
    """
    a, b = window_bounds
    spectra = _WORKER_STATE['spectra']
    min_points = _WORKER_STATE['min_points']
    x_range_buffer = _WORKER_STATE['x_range_buffer']
    max_support = _WORKER_STATE['max_support']
    return compute_global_kl_spectrum(spectra[:, a:b + 1], min_points, x_range_buffer, max_support)


def _compute_sliding_kl_serial(
    spectra:np.ndarray,
    starts:np.ndarray,
    ends:np.ndarray,
    centers:np.ndarray,
    min_points:int,
    x_range_buffer:float,
    max_support:Optional[int],
) -> np.ndarray:
    """Evaluate successive windows in the current process.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Count matrix laid out as (energy, time).
    starts : ndarray, shape (n_windows,)
        Window start indices.
    ends : ndarray, shape (n_windows,)
        Window end indices (inclusive).
    centers : ndarray, shape (n_windows,)
        Window centre times.
    min_points : int
        Minimum valid samples required per window.
    x_range_buffer : float
        Count margin above the mean when choosing the upper support bound.
    max_support : Optional[int]
        Hard cap on the support, or None.

    Returns
    -------
    K : ndarray, shape (n_windows, n_energy)
        Sliding-window KL matrix.
    """
    n_windows = len(starts)
    n_energy = spectra.shape[0]
    K = np.full((n_windows, n_energy), np.nan, dtype=float)
    for wi, (a, b) in enumerate(zip(starts, ends)):
        if (wi + 1) % 10 == 0 or wi == 0 or wi == n_windows - 1:
            logger.info(f"  Sliding KL window {wi + 1}/{n_windows}: spectra {a + 1}-{b + 1} (center time: {centers[wi]:.2f})")
        K[wi] = compute_global_kl_spectrum(spectra[:, a:b + 1], min_points, x_range_buffer, max_support)
    return K


def compute_sliding_kl_matrix(
    spectra:np.ndarray,
    time_axis:np.ndarray,
    config:Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate KL divergence for each complete window and spectral channel.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Count matrix laid out as (energy, time).
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    config : Config
        Spectral analysis settings.

    Returns
    -------
    K : ndarray, shape (n_windows, n_energy)
        Sliding-window KL matrix.
    starts : ndarray, shape (n_windows,)
        Window start indices.
    ends : ndarray, shape (n_windows,)
        Window end indices (inclusive).
    centers : ndarray, shape (n_windows,)
        Window centre times.
    """
    starts, ends, centers = make_sliding_windows(
        spectra.shape[1], time_axis, config.sliding_window_length, config.sliding_step)
    n_windows = len(starts)
    n_energy = spectra.shape[0]
    min_points, x_range_buffer, max_support = (
        config.min_points_per_kl, config.x_range_buffer, config.max_histogram_support)

    n_jobs = _resolve_n_jobs(config.n_jobs)
    if n_jobs <= 1 or n_windows < 2:
        logger.info(f"Computing sliding-window KL matrix (serial): {n_windows} windows.")
        K = _compute_sliding_kl_serial(spectra, starts, ends, centers, min_points, x_range_buffer, max_support)
        return K, starts, ends, centers

    n_jobs = min(n_jobs, n_windows)
    K = np.full((n_windows, n_energy), np.nan, dtype=float)
    bounds = list(zip(starts.tolist(), ends.tolist()))
    chunksize = max(1, n_windows // (n_jobs * 8))

    # Prefer 'fork' so the (potentially large) spectra array is shared by
    # inheritance rather than pickled; fall back to 'spawn' with an initializer.
    methods = mp.get_all_start_methods()
    if 'fork' in methods:
        ctx = mp.get_context('fork')
        _init_kl_worker(spectra, min_points, x_range_buffer, max_support)
        executor_kwargs = dict(max_workers=n_jobs, mp_context=ctx)
    else:
        ctx = mp.get_context()
        executor_kwargs = dict(
            max_workers=n_jobs, mp_context=ctx,
            initializer=_init_kl_worker,
            initargs=(spectra, min_points, x_range_buffer, max_support),
        )

    logger.info(f"Computing sliding-window KL matrix ({n_jobs} processes): {n_windows} windows.")
    try:
        with cf.ProcessPoolExecutor(**executor_kwargs) as ex:
            for wi, row in enumerate(ex.map(_compute_kl_window_row, bounds, chunksize=chunksize)):
                K[wi, :] = row
                if (wi + 1) % 50 == 0 or wi == n_windows - 1:
                    logger.info(f"  Completed sliding KL window {wi + 1}/{n_windows}")
    finally:
        _WORKER_STATE.clear()

    return K, starts, ends, centers


def compute_photon_count_histogram(
    series:np.ndarray,
    x_range_buffer:float,
    max_support:Optional[int],
    bin_size:int | str = 'auto',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Bin a count series and evaluate its Poisson overlay.

    The bars count observations in integer-width bins. The reference curve scales
    the Poisson mass by sample count and bin width. Its mean comes from the
    individual retained counts.

    Parameters
    ----------
    series : ndarray, shape (n_samples,)
        Gain-scaled count series for a single channel.
    x_range_buffer : float
        Count margin above the mean when choosing the upper support bound.
    max_support : Optional[int]
        Hard cap on the support, or None.
    bin_size : int or str, default='auto'
        Positive bin width in photon counts, or 'auto' to calculate the width
        from the data.

    Returns
    -------
    count_values : ndarray, shape (n_bins,)
        Bin centres in photon counts.
    hist_counts : ndarray, shape (n_bins,)
        Observed timepoints per bin.
    poisson_x : ndarray, shape (n_fine,)
        Fine integer grid across the support for the smooth Poisson curve.
    poisson_y : ndarray, shape (n_fine,)
        Expected timepoint count per bin_width-wide window at each poisson_x.
    lam : float
        Poisson rate (mean count), or NaN when undefined.
    n_obs : int
        Number of contributing timepoints.
    bin_width : int
        Width of each bin in photon counts.
    """
    if bin_size != 'auto' and (
        not isinstance(bin_size, (int, np.integer)) or isinstance(bin_size, bool) or bin_size < 1
    ):
        raise ValueError("bin_size must be 'auto' or a positive integer in photon counts.")

    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]

    scaled = np.round(series).astype(int)
    n_obs = int(scaled.size)
    if n_obs == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), float('nan'), 0, 0

    lam = float(np.mean(scaled))
    if not np.isfinite(lam) or lam <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), float('nan'), n_obs, 0

    max_observed = int(np.max(scaled))
    max_range = int(max(max_observed, lam + float(x_range_buffer)))
    if max_support is not None:
        max_range = min(max_range, int(max_support))
        scaled = scaled[scaled <= max_range]
        n_obs = int(scaled.size)
        if n_obs == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), float('nan'), 0, 0
        lam = float(np.mean(scaled))

    # Start near the data to avoid allocating empty bins far below the smallest count.
    min_range = max(0, int(np.min(scaled)) - int(np.ceil(float(x_range_buffer))))

    if bin_size == 'auto':
        q25, q75 = np.percentile(scaled, [25.0, 75.0])
        iqr = float(q75 - q25)
        if iqr > 0.0:
            bin_width = max(1, int(round(2.0 * iqr / (n_obs ** (1.0 / 3.0)))))
        else:
            bin_width = 1
    else:
        bin_width = int(bin_size)

    n_bins = int(np.ceil((max_range - min_range + 1) / bin_width))
    edges = min_range + bin_width * np.arange(n_bins + 1)
    hist_counts, _ = np.histogram(scaled, bins=edges)

    count_values = 0.5 * (edges[:-1] + edges[1:])

    # Scale the Poisson mass into histogram units and limit the overlay to 2,000 samples.
    n_fine = min(max_range - min_range + 1, 2000)
    poisson_x = np.unique(np.linspace(min_range, max_range, n_fine).round().astype(int))
    poisson_y = float(n_obs) * bin_width * poisson.pmf(poisson_x, mu=lam)

    logger.info('Photon-count histogram bin_size=%r (width=%d photon counts)', bin_size, bin_width)
    return count_values, hist_counts, poisson_x, poisson_y, lam, n_obs, bin_width


def test_static_distribution(
    time_values:np.ndarray,
    energy_values:np.ndarray,
    intensity:np.ndarray,
    static_parameters:dict,
    choose_static_distribution:bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Compare the two halves of one channel and optionally trim the record.

    In selection mode the end time is fixed. The first suffix satisfying both
    thresholds and the minimum length supplies the interval for every channel.

    Parameters
    ----------
    time_values : ndarray, shape (n_time,)
        Time axis already restricted to TIME_RANGE, in chronological order.
    energy_values : ndarray, shape (n_energy,)
        Available wavenumbers, in the same units as the requested wavenumber.
    intensity : ndarray, shape (n_energy, n_time)
        Raw intensities corresponding to both axes.
    static_parameters : dict
        Keys: wavenumber (float), delta_mean and delta_var (finite nonnegative
        relative limits), minimal_points_number (positive int per half).
        Variances use ddof=0. Odd counts put the extra point in the second
        half. A zero first-half statistic passes only if the second is zero.

    choose_static_distribution : bool, default=False
        Search successive suffixes if True; otherwise log the initial range
        once and return it unchanged, regardless of the minimum/thresholds.

    Returns
    -------
    selected_time : ndarray, shape (n_selected,)
        First qualifying suffix of time_values.
    selected_intensity : ndarray, shape (n_energy, n_selected)
        All energy channels restricted to the same suffix.
    time_range : tuple of float
        Inclusive first and last selected times for downstream use.

    Raises
    ------
    ValueError
        Invalid parameters, unordered times, nonfinite selected signal, or
        no passing suffix with enough points in each half.
    """
    wavenumber = static_parameters['wavenumber']
    delta_mean = static_parameters['delta_mean']
    delta_var = static_parameters['delta_var']
    minimum = static_parameters['minimal_points_number']
    if not np.isfinite(wavenumber):
        raise ValueError('wavenumber must be finite')
    if any(not np.isfinite(value) or value < 0 for value in (delta_mean, delta_var)):
        raise ValueError('delta_mean and delta_var must be finite and nonnegative')
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError('minimal_points_number must be a positive integer per half')
    if not np.all(np.isfinite(time_values)) or np.any(np.diff(time_values) < 0):
        raise ValueError('Static range selection requires finite, chronologically ordered times')

    index = int(np.argmin(np.abs(energy_values - wavenumber)))
    signal = intensity[index]
    if not np.all(np.isfinite(signal)):
        raise ValueError('Static range selection requires finite intensities at the selected wavenumber')
    logger.info(
        'Static distribution: requested wavenumber %.6g matched to %.6g',
        wavenumber, energy_values[index],
    )
    if time_values.size < 2:
        logger.info('Static distribution: absolute relative mean change nan, '
                    'variance change nan (fewer than two timepoints)')
        if choose_static_distribution:
            raise ValueError('No static distribution found: fewer than two timepoints')
        return time_values, intensity, (float(time_values[0]), float(time_values[-1]))
    attempts = max(1, time_values.size - 2 * minimum + 1) if choose_static_distribution else 1
    for start in range(attempts):
        midpoint = start + (time_values.size - start) // 2
        first = signal[start:midpoint]
        second = signal[midpoint:]
        mean1, mean2 = float(np.mean(first)), float(np.mean(second))
        var1, var2 = float(np.var(first)), float(np.var(second))
        mean_change = (abs((mean2 - mean1) / mean1) if mean1 != 0
                       else (0.0 if mean2 == 0 else float('inf')))
        var_change = (abs((var2 - var1) / var1) if var1 != 0
                      else (0.0 if var2 == 0 else float('inf')))
        logger.info(
            'Static-distribution check: time range (%g, %g), %d / %d points per half; '
            'absolute relative mean change %.6g, variance change %.6g '
            '(limits %.6g / %.6g)',
            time_values[start], time_values[-1], first.size, second.size,
            mean_change, var_change, delta_mean, delta_var,
        )
        if not choose_static_distribution:
            return time_values, intensity, (float(time_values[0]), float(time_values[-1]))
        if (first.size >= minimum and second.size >= minimum
                and mean_change <= delta_mean and var_change <= delta_var):
            time_range = (float(time_values[start]), float(time_values[-1]))
            logger.info(
                'Static TIME_RANGE=%s; removed %d initial points; kept %d points '
                '(%d / %d per half); absolute relative mean change %.6g, '
                'variance change %.6g',
                time_range, start, time_values.size - start, first.size,
                second.size, mean_change, var_change,
            )
            return time_values[start:], intensity[:, start:], time_range

    logger.info('Static-distribution search exhausted: no qualifying range with %d points per half', minimum)
    raise ValueError(
        'No static distribution found within TIME_RANGE with '
        f'at least {minimum} points in each half '
        f'(delta_mean={delta_mean}, delta_var={delta_var})'
    )


def render_global_kl(config:Config, energies:np.ndarray, global_kl:np.ndarray) -> None:
    """Export the full-record KL spectrum as PNG, SVG and HTML.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    energies : ndarray, shape (n_energy,)
        Energy axis used in the analysis.
    global_kl : ndarray, shape (n_energy,)
        Per-channel KL divergence over the whole record.
    """
    logger.info('Rendering global KL spectrum...')
    base = '01_global_KL_spectrum_full_QSS'
    fig = mpl_plots.create_global_kl_figure(
        energies, global_kl, config.energy_label, config.condition_label, config.invert_energy_axis)
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_global_kl_figure(
        energies, global_kl, config.energy_label, config.condition_label, config.invert_energy_axis)
    plotly_plots.export_figure(fig, config.html_dir, base)


def render_sliding_kl_heatmap(config:Config, energies:np.ndarray, centers:np.ndarray, K:np.ndarray) -> None:
    """Save the static heatmap and interactive surface of the window-by-channel KL
    matrix.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    energies : ndarray, shape (n_energy,)
        Energy axis (matrix columns).
    centers : ndarray, shape (n_windows,)
        Window centre times (matrix rows).
    K : ndarray, shape (n_windows, n_energy)
        Sliding-window KL matrix.
    """
    logger.info('Rendering sliding-window KL matrix (matplotlib heatmap, plotly surface)...')
    base = '02_sliding_KL_heatmap'
    fig = mpl_plots.create_sliding_kl_heatmap_figure(
        energies, centers, K, config.energy_label, config.invert_energy_axis,
        config.plot_heatmap_log_scale, config.eps)
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_sliding_kl_surface_figure(
        energies, centers, K, config.energy_label, config.invert_energy_axis,
        config.plot_heatmap_log_scale, config.eps)
    plotly_plots.export_figure(fig, config.html_dir, base)


_HISTOGRAM_GRID_ROWS = 4
_HISTOGRAM_GRID_COLS = 4
_HISTOGRAM_GRID_SIZE = _HISTOGRAM_GRID_ROWS * _HISTOGRAM_GRID_COLS


def resolve_histogram_channels(all_energies:np.ndarray, targets:Sequence[float]) -> list:
    """Find the closest retained wavenumber for each requested target, in request
    order.

    Parameters
    ----------
    all_energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1 (after any energy_ranges filtering).
    targets : Sequence[float]
        Requested histogram energies in cm^-1.

    Returns
    -------
    channels : list
        List of (index, matched_energy) pairs, one per requested target.
    """
    all_energies = np.asarray(all_energies, dtype=float)
    channels = []
    for target in targets:
        idx = int(np.argmin(np.abs(all_energies - float(target))))
        matched = float(all_energies[idx])
        logger.info(
            f"Histogram channel: requested {target} cm^-1 -> "
            f"nearest available {matched:.2f} cm^-1 (index {idx})"
        )
        channels.append((idx, matched))
    return channels


def render_photon_count_histograms(
    config:Config,
    all_energies:np.ndarray,
    counts_all:np.ndarray,
    *,
    global_kl:np.ndarray | None = None,
) -> None:
    """Export count histograms in pages of 16 channels with matched Poisson overlays.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    all_energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1 (after any energy_ranges filtering).
    counts_all : ndarray, shape (n_energy, n_time)
        Gain-scaled count matrix laid out as (energy, time).
    global_kl : ndarray, shape (n_energy,), or None, default=None
        Full-record values for the panel labels. If omitted, evaluate each requested channel.
    """
    if not config.histogram_energies:
        logger.info('No histogram energies specified; skipping photon-count histograms.')
        return

    panels = []
    for idx, matched in resolve_histogram_channels(all_energies, config.histogram_energies):
        count_values, hist_counts, poisson_x, poisson_y, lam, n_obs, bin_width = compute_photon_count_histogram(
            counts_all[idx, :], config.x_range_buffer, config.max_histogram_support, bin_size=config.bin_size)
        if n_obs == 0 or not np.isfinite(lam):
            logger.warning(f"  Skipping {matched:.2f} cm^-1: no valid photon counts.")
            continue
        if global_kl is None:
            kl = compute_poisson_kl_from_series(
                counts_all[idx], config.min_points_per_kl, config.x_range_buffer,
                config.max_histogram_support)
        else:
            kl = global_kl[idx]
        panels.append(dict(
            energy=matched, count_values=count_values, hist_counts=hist_counts,
            poisson_x=poisson_x, poisson_y=poisson_y, lam=lam, n_obs=n_obs,
            bin_width=bin_width, kl=kl))

    if not panels:
        logger.warning('No valid photon-count histograms to render.')
        return

    n_pages = int(np.ceil(len(panels) / _HISTOGRAM_GRID_SIZE))
    for page in range(n_pages):
        page_panels = panels[page * _HISTOGRAM_GRID_SIZE:(page + 1) * _HISTOGRAM_GRID_SIZE]
        base = '03_photon_count_histogram' if n_pages == 1 else f"03_photon_count_histogram_p{page + 1}"
        logger.info(f"Rendering photon-count histogram grid {base} ({len(page_panels)} panel(s))...")

        fig = mpl_plots.create_photon_count_histogram_grid_figure(
            page_panels, _HISTOGRAM_GRID_ROWS, _HISTOGRAM_GRID_COLS)
        mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
        fig = plotly_plots.create_photon_count_histogram_grid_figure(
            page_panels, _HISTOGRAM_GRID_ROWS, _HISTOGRAM_GRID_COLS)
        plotly_plots.export_figure(fig, config.html_dir, base)


def render_intensity_time_traces(
    config:Config,
    all_energies:np.ndarray,
    spectra_all:np.ndarray,
    time_axis:np.ndarray,
) -> None:
    """Export the raw intensity history for each requested histogram channel.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    all_energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1 (after any energy_ranges filtering).
    spectra_all : ndarray, shape (n_energy, n_time)
        Raw intensity matrix laid out as (energy, time).
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    """
    if not config.histogram_energies:
        logger.info('No histogram energies specified; skipping intensity time traces.')
        return

    channels = resolve_histogram_channels(all_energies, config.histogram_energies)
    energies = [matched for _, matched in channels]
    traces = [np.asarray(spectra_all[idx, :], dtype=float) for idx, _ in channels]

    logger.info('Rendering raw-intensity time traces...')
    base = '04_time_traces'
    args = (time_axis, traces, energies, config.condition_label)
    fig = mpl_plots.create_intensity_time_traces_figure(*args)
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_intensity_time_traces_figure(*args)
    plotly_plots.export_figure(fig, config.html_dir, base)


def compute_windowed_kl_series(
    series:np.ndarray,
    time_axis:np.ndarray,
    config:Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one channel at the analysis window centers.

    Parameters
    ----------
    series : ndarray, shape (n_time,)
        Gain-scaled count series for one channel.
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    config : Config
        Spectral analysis settings.

    Returns
    -------
    centers : ndarray, shape (n_windows,)
        Window centre times in seconds.
    kl : ndarray, shape (n_windows,)
        Per-window KL divergence (NaN where undefined).
    """
    series = np.asarray(series, dtype=float)
    starts, ends, centers = make_sliding_windows(
        series.size, time_axis, config.sliding_window_length, config.sliding_step)
    kl = np.full(len(starts), np.nan, dtype=float)
    for wi, (a, b) in enumerate(zip(starts, ends)):
        kl[wi] = compute_poisson_kl_from_series(
            series[a:b + 1], config.min_points_per_kl, config.x_range_buffer,
            config.max_histogram_support)
    return centers, kl


def render_windowed_kl_time_traces(
    config:Config,
    all_energies:np.ndarray,
    counts_all:np.ndarray,
    time_axis:np.ndarray,
    *,
    windowed_kl:tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Export KL histories at the requested histogram channels.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    all_energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1 (after any energy_ranges filtering).
    counts_all : ndarray, shape (n_energy, n_time)
        Gain-scaled count matrix laid out as (energy, time).
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    windowed_kl : tuple of ndarray or None, default=None
        Window centers and the KL matrix (windows, channels). If omitted,
        compute the requested traces from the count series.
    """
    if not config.histogram_energies:
        logger.info('No histogram energies specified; skipping windowed-KL time traces.')
        return

    channels = resolve_histogram_channels(all_energies, config.histogram_energies)
    energies = [matched for _, matched in channels]

    logger.info('Rendering windowed-KL time traces...')
    if windowed_kl is None:
        traces = []
        for idx, _ in channels:
            centers, trace = compute_windowed_kl_series(counts_all[idx], time_axis, config)
            traces.append(trace)
    else:
        centers, matrix = windowed_kl
        traces = [matrix[:, idx] for idx, _ in channels]

    base = '05_time_traces_KL'
    args = (centers, traces, energies, config.condition_label)
    fig = mpl_plots.create_windowed_kl_time_traces_figure(*args)
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_windowed_kl_time_traces_figure(*args)
    plotly_plots.export_figure(fig, config.html_dir, base)


def export_matrix_like_input(
    matrix:np.ndarray,
    energies:np.ndarray,
    times:np.ndarray,
    out_dir:str,
    filename:str,
) -> None:
    """Write one CSV row per channel with window centers in the header.

    Parameters
    ----------
    matrix : ndarray, shape (n_rows, n_energy)
        Matrix whose transpose is written (rows: energies, cols: times).
    energies : ndarray, shape (n_energy,)
        Energy axis written as the first column.
    times : ndarray, shape (n_rows,)
        Time axis written as the column headers.
    out_dir : str
        Output directory.
    filename : str
        Output filename.
    """
    df = pd.DataFrame(matrix.T, columns=times)
    df.insert(0, '', energies)
    df.to_csv(os.path.join(out_dir, filename), index=False)


def export_global_kl_spectrum(
    global_kl:np.ndarray,
    energies:np.ndarray,
    out_dir:str,
    filename:str,
) -> None:
    """Write the spectral axis and its full-record KL values to CSV.

    Parameters
    ----------
    global_kl : ndarray, shape (n_energy,)
        Per-channel KL divergence over the whole record.
    energies : ndarray, shape (n_energy,)
        Energy axis aligned with global_kl.
    out_dir : str
        Output directory.
    filename : str
        Output filename.
    """
    pd.DataFrame({
        'Energy (cm^-1)': energies,
        'Global KL divergence': global_kl,
    }).to_csv(os.path.join(out_dir, filename), index=False)


def run_pipeline(config:Config) -> None:
    """Select the record, convert intensities to counts, and export KL tables and
    figures.

    Parameters
    ----------
    config : Config
        Spectral analysis settings.
    """
    ensure_output_dirs(config)
    os.makedirs(config.csv_dir, exist_ok=True)

    logger.info('Loading XYYYY data...')
    energies, time_axis, time_series = import_data(
        config.operando_csv_path, config.time_range, config.energy_ranges,
        rezero_time=False)
    time_axis, spectra_all, effective_time_range = test_static_distribution(
        time_axis, energies, time_series.T, config.static_parameters,
        config.choose_static_distribution,
    )
    if config.choose_static_distribution:
        config.time_range = effective_time_range
    # Export window times relative to the first retained spectrum.
    time_axis = time_axis - time_axis[0]

    gain = resolve_gain(spectra_all, energies, config)
    counts = apply_count_scaling(spectra_all, gain)

    logger.info('Computing global full-record KL spectrum...')
    global_kl = compute_global_kl_spectrum(
        counts, config.min_points_per_kl, config.x_range_buffer, config.max_histogram_support)

    logger.info('Computing sliding-window KL matrix...')
    K, _, _, centers = compute_sliding_kl_matrix(counts, time_axis, config)

    logger.info('Saving figures...')
    render_global_kl(config, energies, global_kl)
    render_sliding_kl_heatmap(config, energies, centers, K)
    render_photon_count_histograms(config, energies, counts, global_kl=global_kl)
    render_intensity_time_traces(config, energies, spectra_all, time_axis)
    render_windowed_kl_time_traces(config, energies, counts, time_axis, windowed_kl=(centers, K))

    logger.debug(f"Applied gain: {gain}")

    logger.info('Saving KL tables...')
    export_matrix_like_input(K, energies, centers, config.csv_dir, 'sliding_KL_matrix.csv')
    export_global_kl_spectrum(global_kl, energies, config.csv_dir, 'global_KL_spectrum_full_QSS.csv')

    logger.info('Done.')


def compute_per_channel_gain(spectra:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate each channel’s gain from its population variance and mean intensity.

    Parameters
    ----------
    spectra : ndarray, shape (n_energy, n_time)
        Reference intensity matrix laid out as (energy, time).

    Returns
    -------
    gain : ndarray, shape (n_energy,)
        Measured gain per channel; NaN where mean or variance is non-positive.
    valid : ndarray, shape (n_energy,)
        Boolean mask, True where the measured gain is finite and positive.
    """
    arr = np.asarray(spectra, dtype=float)
    means = arr.mean(axis=1)
    variances = arr.var(axis=1)

    valid = (means > 0.0) & (variances > 0.0)
    gain = np.full(means.shape, np.nan, dtype=float)
    gain[valid] = variances[valid] / means[valid]

    n_bad = int(np.count_nonzero(~valid))
    if n_bad:
        logger.warning(
            f"{n_bad}/{valid.size} reference channels have non-positive mean or "
            f"variance; their measured gain is undefined."
        )
    return gain, valid


def _positive_cubic_coefficients(
    design:np.ndarray,
    gain:np.ndarray,
    lower:float,
    upper:float,
) -> tuple[np.ndarray, bool]:
    """Fit cubic coefficients with a positive minimum across the evaluation interval.

    Check the endpoints and interior extrema of the ordinary least-squares cubic. If
    the minimum is too low, constrain the fit and recheck those extrema.

    Parameters
    ----------
    design : np.ndarray, shape (n, 4)
        Standardised-axis Vandermonde matrix, with ascending powers.
    gain : np.ndarray, shape (n,)
        Measured gains entering the least-squares objective.
    lower : float
        Lower standardised-axis bound, including excluded output channels.
    upper : float
        Upper standardised-axis bound, including excluded output channels.

    Returns
    -------
    beta : np.ndarray, shape (4,)
        Ascending cubic coefficients in the standardised coordinate.
    constrained : bool
        Whether positivity required constrained optimisation.
    """
    scale = float(np.max(np.abs(gain)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('Positive cubic fitting requires finite, nonzero gains.')
    target = gain / scale
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    # A relative numerical floor enforces strict positivity rather than zero.
    floor = 1e-8

    def extrema(coefficients:np.ndarray) -> np.ndarray:
        """Locate interval endpoints and real stationary points inside the bounds.

        Parameters
        ----------
        coefficients : np.ndarray, shape (4,)
            Ascending coefficients in the standardised coordinate.

        Returns
        -------
        points : np.ndarray
            All candidates for the continuous-interval minimum.
        """
        roots = np.roots(
            np.array([3.0 * coefficients[3], 2.0 * coefficients[2], coefficients[1]]))
        interior = roots.real[(np.abs(roots.imag) < 1e-10)
                              & (roots.real > lower) & (roots.real < upper)]
        return np.r_[lower, upper, interior]

    candidates = extrema(beta)
    if np.min(np.polynomial.polynomial.polyval(candidates, beta)) >= floor:
        return beta * scale, False

    points = np.unique(np.r_[np.linspace(lower, upper, 65), candidates])
    beta = np.array([max(float(np.mean(target)), 1.0), 0.0, 0.0, 0.0])
    gram = design.T @ design / gain.size
    rhs = design.T @ target / gain.size
    for _ in range(50):
        constraints = np.vander(points, 4, increasing=True)
        result = minimize(
            lambda coefficients: float(0.5 * coefficients @ gram @ coefficients
                                       - rhs @ coefficients),
            beta,
            jac=lambda coefficients: gram @ coefficients - rhs,
            constraints={
                'type': 'ineq',
                'fun': lambda coefficients: constraints @ coefficients - 2.0 * floor,
                'jac': lambda coefficients: constraints,
            },
            method='SLSQP',
            options={'ftol': 1e-12, 'maxiter': 1000},
        )
        if not result.success:
            raise RuntimeError(f'Positive cubic fitting failed: {result.message}')
        beta = result.x
        candidates = extrema(beta)
        values = np.polynomial.polynomial.polyval(candidates, beta)
        if np.min(values) >= floor:
            logger.info('Cubic positivity enforced across the full output range.')
            return beta * scale, True
        points = np.unique(np.r_[points, candidates[values < floor]])
    raise RuntimeError('Positive cubic fitting did not satisfy the continuous-range constraint.')


def fit_cubic_gain(
    energies:np.ndarray,
    gain:np.ndarray,
    energies_eval:np.ndarray | None = None,
) -> np.poly1d:
    """Fit a cubic gain curve in wavenumber and enforce positivity across the output
    domain.

    Parameters
    ----------
    energies : ndarray, shape (n_used,)
        Energies of the channels entering the fit, in cm^-1.
    gain : ndarray, shape (n_used,)
        Measured gain at those channels.
    energies_eval : np.ndarray | None, default=None
        Full output energy axis defining the positivity interval. None uses
        the fitted energy range. Excluded channels do not enter the objective.

    Returns
    -------
    poly : np.poly1d
        Degree-3 polynomial in energy.
    """
    energies = np.asarray(energies, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if energies.size < 4:
        raise RuntimeError(
            f"A cubic fit needs at least 4 usable channels, got {energies.size}; "
            f"exclude fewer energy channels."
        )
    domain = energies if energies_eval is None else np.asarray(energies_eval, dtype=float)
    lower = float(min(domain.min(), energies.min()))
    upper = float(max(domain.max(), energies.max()))
    mu = 0.5 * (lower + upper)
    sigma = 0.5 * (upper - lower)
    if sigma == 0.0:
        raise RuntimeError('All fitted energies are identical; cannot fit a cubic.')
    design = np.vander((energies - mu) / sigma, 4, increasing=True)
    beta, _ = _positive_cubic_coefficients(design, gain, -1.0, 1.0)
    poly = np.poly1d(beta[::-1])(np.poly1d([1.0 / sigma, -mu / sigma]))
    if np.any(poly(domain) <= 0.0):
        raise RuntimeError('Positive cubic lost numerical precision on the output energy axis.')

    residual = gain - poly(energies)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((gain - gain.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    logger.info(f"Cubic gain fit over {energies.size} channels: R^2 = {r2:.6f}")
    return poly


def validate_gain_mode(approximate_gain:str) -> None:
    """Check the calibration mode before constructing either gain table.

    Parameters
    ----------
    approximate_gain : str
        One of 'fit_cubic_full', 'fig_cubic_interpolate', or 'raw'.
    """
    if approximate_gain not in ('fit_cubic_full', 'fig_cubic_interpolate', 'raw'):
        raise ValueError(
            "approximate_gain must be 'fit_cubic_full', "
            "'fig_cubic_interpolate', or 'raw'.")


def build_gain_curve(
    energies:np.ndarray,
    gain_measured:np.ndarray,
    used:np.ndarray,
    approximate_gain:str,
) -> Tuple[np.ndarray, np.poly1d | None]:
    """Construct the energy-gain table from measured or fitted gains.

    The full cubic replaces every measurement. Interpolation retains included
    measurements and fills the other channels. Raw mode leaves excluded or invalid
    measurements empty.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Full energy axis in cm^-1.
    gain_measured : ndarray, shape (n_energy,)
        Measured gain per channel; NaN where undefined.
    used : ndarray, shape (n_energy,)
        Boolean mask of channels measured and outside the excluded energy ranges.
    approximate_gain : str
        'fit_cubic_full', 'fig_cubic_interpolate', or 'raw' (no fit).

    Returns
    -------
    gain : ndarray, shape (n_energy,)
        Final per-channel gain; cubic predictions are positive over the output range.
    poly : np.poly1d or None
        The fitted cubic polynomial, or None in raw mode.
    """
    energies = np.asarray(energies, dtype=float)
    gain_measured = np.asarray(gain_measured, dtype=float)

    validate_gain_mode(approximate_gain)
    if approximate_gain == 'raw':
        return np.where(used, gain_measured, np.nan), None

    poly = fit_cubic_gain(energies[used], gain_measured[used], energies)
    fitted = poly(energies)

    if approximate_gain == 'fit_cubic_full':
        gain = fitted
        logger.info(f"Gain curve: cubic approximation applied to all {energies.size} channels.")
    else:
        gain = np.where(used, gain_measured, fitted)
        n_filled = int(np.count_nonzero(~used))
        logger.info(
            f"Gain curve: measured var/mean kept on {int(np.count_nonzero(used))} channels, "
            f"{n_filled} channel(s) filled from the cubic."
        )

    nonpositive = ~(gain > 0.0) | ~np.isfinite(gain)
    if nonpositive.any():
        logger.warning(
            f'{int(np.count_nonzero(nonpositive))} fitted channel gains are '
            'non-positive or non-finite; their CSV values will be NaN. '
            'Negative fitted gains are shown in #ff0000.')

    logger.info(
        f"Final gain over {gain.size} channels: min={gain.min():.6g}, "
        f"median={np.median(gain):.6g}, max={gain.max():.6g}"
    )
    return gain, poly


def export_gain_csv(
    out_dir:str,
    csv_name:str,
    energies:np.ndarray,
    gain:np.ndarray,
    gain_measured:np.ndarray,
    used:np.ndarray,
) -> str:
    """Write the measured gain, final gain and inclusion mask for each wavenumber.

    Parameters
    ----------
    out_dir : str
        Output directory.
    csv_name : str
        Filename of the gain table.
    energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1.
    gain : ndarray, shape (n_energy,)
        Final per-channel gain.
    gain_measured : ndarray, shape (n_energy,)
        Measured var/mean per channel; NaN where undefined.
    used : ndarray, shape (n_energy,)
        Boolean mask of channels that were measured and selected.

    Returns
    -------
    csv_path : str
        Path of the written csv.
    """
    csv_path = os.path.join(out_dir, csv_name)
    pd.DataFrame({
        'energy': np.asarray(energies, dtype=float),
        'gain': np.where(np.isfinite(gain) & (gain > 0.0), gain, np.nan),
        'gain_measured': np.asarray(gain_measured, dtype=float),
        'used': np.asarray(used, dtype=bool),
    }).to_csv(csv_path, index=False)
    logger.info(f"Wrote gain table to {csv_path}")
    return csv_path


def render_gain_calibration(
    config:GainCalibrationConfig,
    energies:np.ndarray,
    gain:np.ndarray,
    gain_measured:np.ndarray,
    used:np.ndarray,
    poly:np.poly1d | None,
) -> None:
    """Plot reciprocal gain against wavenumber and export the calibration figure.

    Parameters
    ----------
    config : GainCalibrationConfig
        Resolved calibration configuration.
    energies : ndarray, shape (n_energy,)
        Energy axis in cm^-1.
    gain : ndarray, shape (n_energy,)
        Final per-channel gain.
    gain_measured : ndarray, shape (n_energy,)
        Measured var/mean per channel; NaN where undefined.
    used : ndarray, shape (n_energy,)
        Boolean mask of channels that were measured and selected.
    poly : np.poly1d or None
        The fitted cubic polynomial, or None in raw mode.
    """
    logger.info('Rendering gain calibration curve...')
    base = '00_gain_calibration'

    # The energy calibration plot uses reciprocal gain. The CSV stores gain itself.
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse_gain = np.where(gain != 0.0, 1.0 / gain, np.nan)
        fitted_inverse = None
        if config.approximate_gain == 'fit_cubic_full':
            fitted = poly(energies)
            fitted_inverse = np.where(fitted != 0.0, 1.0 / fitted, np.nan)
    args = (energies, inverse_gain, energies[used], 1.0 / gain_measured[used],
            fitted_inverse, config.energy_label, config.invert_energy_axis)

    fig = mpl_plots.create_gain_calibration_figure(*args, raw=config.approximate_gain == 'raw')
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_gain_calibration_figure(*args, raw=config.approximate_gain == 'raw')
    plotly_plots.export_figure(fig, config.html_dir, base)


def fit_cubic_gain_vs_intensity(
    intensity:np.ndarray,
    gain:np.ndarray,
    intensity_eval:np.ndarray,
    level:float,
) -> Tuple[np.poly1d, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a positive cubic in mean intensity and calculate two-sided regression bands.

    Standardize the intensity axis for fitting and return the polynomial in input
    units. Confidence and prediction intervals use ordinary least-squares residuals
    and are approximate when the positivity constraint is active.

    Parameters
    ----------
    intensity : ndarray, shape (n_used,)
        Mean intensity of the channels entering the fit.
    gain : ndarray, shape (n_used,)
        Measured gain var/mean at those channels.
    intensity_eval : ndarray, shape (n_eval,)
        Intensities at which the fit and its bands are evaluated.
    level : float
        Two-sided confidence level (e.g. 0.95).

    Returns
    -------
    poly : np.poly1d
        Degree-3 polynomial in the raw (unstandardised) intensity.
    fit : ndarray, shape (n_eval,)
        Fitted gain at intensity_eval.
    ci_low : ndarray, shape (n_eval,)
        Lower mean-response confidence bound at intensity_eval.
    ci_high : ndarray, shape (n_eval,)
        Upper mean-response confidence bound at intensity_eval.
    pi_low : ndarray, shape (n_eval,)
        Lower prediction bound at intensity_eval.
    pi_high : ndarray, shape (n_eval,)
        Upper prediction bound at intensity_eval.
    r2 : float
        Coefficient of determination on the fitted points.
    """
    intensity = np.asarray(intensity, dtype=float)
    gain = np.asarray(gain, dtype=float)
    intensity_eval = np.asarray(intensity_eval, dtype=float)

    degree = 3
    n_params = degree + 1
    n = intensity.size
    # A cubic needs 4 points; one extra gives >= 1 residual degree of freedom for
    # the variance estimate the intervals depend on.
    if n < n_params + 1:
        raise RuntimeError(
            f"A cubic gain(intensity) fit with uncertainty bands needs at least "
            f"{n_params + 1} usable channels, got {n}; exclude fewer energy channels."
        )

    # Standardise the abscissa so the Vandermonde system stays well conditioned.
    mu = float(intensity.mean())
    sigma = float(intensity.std())
    if sigma == 0.0:
        raise RuntimeError('All fitted channels share the same mean intensity; cannot fit gain(intensity).')

    def _design(values:np.ndarray) -> np.ndarray:
        """Express intensities on the scaled cubic basis.

        Parameters
        ----------
        values : ndarray, shape (n,)
            Intensities in the original detector units.

        Returns
        -------
        design : ndarray, shape (n, 4)
            Columns of ascending powers of standardized intensity.
        """
        return np.vander((np.asarray(values, dtype=float) - mu) / sigma, N=n_params, increasing=True)

    x_design = _design(intensity)
    domain = np.r_[intensity, intensity_eval]
    beta, constrained = _positive_cubic_coefficients(
        x_design, gain, float((domain.min() - mu) / sigma),
        float((domain.max() - mu) / sigma))
    if constrained:
        logger.warning(
            'Positivity constrained the intensity fit; ordinary-regression '
            'confidence/prediction bands are approximate and may cross zero.')

    residual = gain - x_design @ beta
    dof = n - n_params
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((gain - gain.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    sigma2 = ss_res / dof
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)

    x_eval_design = _design(intensity_eval)
    fit = x_eval_design @ beta

    # Leverage of each evaluation point; mean-fit variance is sigma2 * leverage,
    # a new observation adds sigma2 on top of that.
    leverage = np.einsum('ij,jk,ik->i', x_eval_design, xtx_inv, x_eval_design)
    se_mean = np.sqrt(np.maximum(sigma2 * leverage, 0.0))
    se_pred = np.sqrt(np.maximum(sigma2 * (1.0 + leverage), 0.0))
    t_crit = float(student_t.ppf(0.5 + 0.5 * level, dof))

    ci_low, ci_high = fit - t_crit * se_mean, fit + t_crit * se_mean
    pi_low, pi_high = fit - t_crit * se_pred, fit + t_crit * se_pred

    # Fold the standardisation back into a plain polynomial in raw intensity.
    poly = np.poly1d(beta[::-1])(np.poly1d([1.0 / sigma, -mu / sigma]))
    if np.any(poly(domain) <= 0.0):
        raise RuntimeError('Positive cubic lost numerical precision on the output intensity axis.')
    logger.info(
        f"Cubic gain(intensity) fit over {n} channels: R^2 = {r2:.6f}, "
        f"{level * 100:.0f}% bands from {dof} residual dof."
    )
    return poly, fit, ci_low, ci_high, pi_low, pi_high, r2


def build_gain_intensity_curve(
    means:np.ndarray,
    gain_measured:np.ndarray,
    used:np.ndarray,
    approximate_gain:str,
    level:float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.poly1d | None]:
    """Evaluate the selected gain model at the channel means and on a plotting grid.

    The grid spans all input means, including excluded channels. Raw mode returns
    measured gains and empty arrays for the fitted curve and its intervals.

    Parameters
    ----------
    means : ndarray, shape (n_energy,)
        Mean intensity over time per channel.
    gain_measured : ndarray, shape (n_energy,)
        Measured gain per channel; NaN where undefined.
    used : ndarray, shape (n_energy,)
        Boolean mask of channels measured and outside the excluded energy ranges.
    approximate_gain : str
        'fit_cubic_full', 'fig_cubic_interpolate', or 'raw' (no fit).
    level : float
        Two-sided confidence level for the bands (e.g. 0.95).

    Returns
    -------
    gain : ndarray, shape (n_energy,)
        Final per-channel gain; cubic predictions are positive over the output range.
    fit_x : ndarray, shape (n_grid,)
        Dense intensity grid spanning all channel means.
    fit_y : ndarray, shape (n_grid,)
        Fitted gain on fit_x.
    ci_low : ndarray, shape (n_grid,)
        Lower mean-fit confidence band on fit_x.
    ci_high : ndarray, shape (n_grid,)
        Upper mean-fit confidence band on fit_x.
    pi_low : ndarray, shape (n_grid,)
        Lower prediction band on fit_x.
    pi_high : ndarray, shape (n_grid,)
        Upper prediction band on fit_x.
    poly : np.poly1d or None
        The fitted cubic in raw intensity, or None in raw mode.
    """
    means = np.asarray(means, dtype=float)
    gain_measured = np.asarray(gain_measured, dtype=float)

    validate_gain_mode(approximate_gain)
    if approximate_gain == 'raw':
        gain = np.where(used, gain_measured, np.nan)
        return (gain, *(np.empty(0) for _ in range(6)), None)

    used_intensity = means[used]
    used_gain = gain_measured[used]

    # Include excluded channels in the plotted domain, without fitting them.
    grid = np.linspace(float(means.min()), float(means.max()), 300)
    poly, fit_y, ci_low, ci_high, pi_low, pi_high, _ = fit_cubic_gain_vs_intensity(
        used_intensity, used_gain, grid, level)

    fitted_channels = poly(means)
    if approximate_gain == 'fit_cubic_full':
        gain = fitted_channels
        logger.info(f"Gain(intensity) curve: cubic approximation applied to all {means.size} channels.")
    else:
        gain = np.where(used, gain_measured, fitted_channels)
        n_filled = int(np.count_nonzero(~used))
        logger.info(
            f"Gain(intensity) curve: measured var/mean kept on {int(np.count_nonzero(used))} "
            f"channels, {n_filled} channel(s) filled from the cubic."
        )

    nonpositive = ~(gain > 0.0) | ~np.isfinite(gain)
    if nonpositive.any():
        logger.warning(
            f'{int(np.count_nonzero(nonpositive))} fitted channel gains are '
            'non-positive or non-finite; their CSV values will be NaN. '
            'Negative fitted gains are shown in #ff0000.')

    logger.info(
        f"Final gain(intensity) over {gain.size} channels: min={gain.min():.6g}, "
        f"median={np.median(gain):.6g}, max={gain.max():.6g}"
    )
    return gain, grid, fit_y, ci_low, ci_high, pi_low, pi_high, poly


def export_gain_intensity_csv(
    out_dir:str,
    csv_name:str,
    means:np.ndarray,
    gain:np.ndarray,
    gain_measured:np.ndarray,
    used:np.ndarray,
) -> str:
    """Write gains and inclusion flags sorted by channel mean intensity.

    Parameters
    ----------
    out_dir : str
        Output directory.
    csv_name : str
        Filename of the gain-vs-intensity table.
    means : ndarray, shape (n_energy,)
        Mean intensity over time per channel.
    gain : ndarray, shape (n_energy,)
        Final per-channel gain.
    gain_measured : ndarray, shape (n_energy,)
        Measured var/mean per channel; NaN where undefined.
    used : ndarray, shape (n_energy,)
        Boolean mask of channels measured and selected.

    Returns
    -------
    csv_path : str
        Path of the written csv.
    """
    csv_path = os.path.join(out_dir, csv_name)
    order = np.argsort(np.asarray(means, dtype=float))
    pd.DataFrame({
        'intensity': np.asarray(means, dtype=float)[order],
        'gain': np.where(np.isfinite(gain) & (gain > 0.0), gain, np.nan)[order],
        'gain_measured': np.asarray(gain_measured, dtype=float)[order],
        'used': np.asarray(used, dtype=bool)[order],
    }).to_csv(csv_path, index=False)
    logger.info(f"Wrote gain-vs-intensity table to {csv_path}")
    return csv_path


def render_gain_intensity_calibration(
    config:GainCalibrationConfig,
    means_used:np.ndarray,
    gain_used:np.ndarray,
    fit_x:np.ndarray,
    fit_y:np.ndarray,
    ci_low:np.ndarray,
    ci_high:np.ndarray,
    pi_low:np.ndarray,
    pi_high:np.ndarray,
) -> None:
    """Export measured gains and any fitted intensity curve with its regression bands.

    Parameters
    ----------
    config : GainCalibrationConfig
        Resolved calibration configuration.
    means_used : ndarray, shape (n_used,)
        Mean intensity of the measured, selected channels (scatter x).
    gain_used : ndarray, shape (n_used,)
        Measured gain of those channels (scatter y).
    fit_x : ndarray, shape (n_grid,)
        Dense intensity grid for the curve and bands.
    fit_y : ndarray, shape (n_grid,)
        Fitted gain on fit_x.
    ci_low : ndarray, shape (n_grid,)
        Lower mean-fit confidence band on fit_x.
    ci_high : ndarray, shape (n_grid,)
        Upper mean-fit confidence band on fit_x.
    pi_low : ndarray, shape (n_grid,)
        Lower prediction band on fit_x.
    pi_high : ndarray, shape (n_grid,)
        Upper prediction band on fit_x.
    """
    logger.info('Rendering gain-vs-intensity calibration curve...')
    base = '01_gain_intensity_calibration'
    args = (means_used, gain_used, fit_x, fit_y, ci_low, ci_high, pi_low, pi_high,
            config.intensity_label, _GAIN_CI_LEVEL)

    fig = mpl_plots.create_gain_intensity_figure(*args, raw=config.approximate_gain == 'raw')
    mpl_plots.export_figure(fig, config.png_dir, config.svg_dir, base, config.dpi)
    fig = plotly_plots.create_gain_intensity_figure(*args, raw=config.approximate_gain == 'raw')
    plotly_plots.export_figure(fig, config.html_dir, base)


def run_gain_calibration(config:GainCalibrationConfig) -> None:
    """Calibrate energy-dependent gain and optionally intensity-dependent gain from a
    stable time series.

    Parameters
    ----------
    config : GainCalibrationConfig
        Resolved calibration configuration.
    """
    validate_gain_mode(config.approximate_gain)
    ensure_output_dirs(config)
    csv_dir = os.path.join(config.output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    logger.info('Loading reference XYYYY data...')
    energies, _, time_series = import_data(config.reference_csv_path, config.time_range)

    # Calibration estimates one gain per row of the transposed matrix.
    spectra = time_series.T
    channel_means = spectra.mean(axis=1)
    gain_measured, valid = compute_per_channel_gain(spectra)

    excluded = in_any_range(energies, config.exclude_energies)
    used = ~excluded & valid
    logger.info(
        f'Energy exclusions omit {int(np.count_nonzero(excluded))}/{energies.size} '
        f'channels from fitting; {int(np.count_nonzero(used))} valid channels remain.')
    if not used.any():
        raise RuntimeError('No usable channels remain after applying exclude_energies.')

    gain, poly = build_gain_curve(energies, gain_measured, used, config.approximate_gain)

    if config.approximate_gain == 'raw':
        energies = energies[used]
        channel_means = channel_means[used]
        gain_measured = gain_measured[used]
        gain = gain[used]
        used = np.ones(energies.shape, dtype=bool)

    logger.info('Saving energy-gain figures...')
    render_gain_calibration(config, energies, gain, gain_measured, used, poly)

    logger.info('Saving energy-gain CSV output...')
    export_gain_csv(csv_dir, config.gain_csv_name, energies, gain, gain_measured, used)

    if config.calibrate_gain_per_intensity:
        logger.info('Calibrating gain as a function of intensity...')
        (gain_i, fit_x, fit_y, ci_low, ci_high, pi_low, pi_high, _) = build_gain_intensity_curve(
            channel_means, gain_measured, used, config.approximate_gain, _GAIN_CI_LEVEL)

        logger.info('Saving gain-vs-intensity figure...')
        render_gain_intensity_calibration(
            config, channel_means[used], gain_measured[used],
            fit_x, fit_y, ci_low, ci_high, pi_low, pi_high)

        logger.info('Saving gain-vs-intensity CSV output...')
        export_gain_intensity_csv(
            csv_dir, config.gain_intensity_csv_name,
            channel_means, gain_i, gain_measured, used)


    logger.info('Done.')
