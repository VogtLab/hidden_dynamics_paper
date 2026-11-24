"""Static figures and PNG/SVG export for gain calibration and KL analysis."""

import os
from typing import Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so figures can be exported on a headless machine.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

import styling

plt.rcParams.update(styling.MPL_RCPARAMS)
plt.rcParams['axes.prop_cycle'] = cycler(color=styling.COLOR_CYCLE)


def build_gruvbox_cmap() -> LinearSegmentedColormap:
    """Interpolate the shared gradient into a continuous Matplotlib colormap.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Colormap interpolated from ``styling.COLORSCALE_GRADIENT``.
    """
    return LinearSegmentedColormap.from_list('gruvbox_terrain', styling.COLORSCALE_GRADIENT, N=256)


def create_global_kl_figure(
    energies:np.ndarray,
    global_kl:np.ndarray,
    energy_label:str,
    condition_label:str,
    invert_energy_axis:bool,
) -> Figure:
    """Plot one full-record KL value per spectral channel.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Energy axis used in the analysis.
    global_kl : ndarray, shape (n_energy,)
        Per-channel KL divergence over the whole record.
    energy_label : str
        Axis label for the energy axis (matplotlib mathtext allowed).
    condition_label : str
        Human-readable dataset label used in the title.
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))
    ax.plot(energies, global_kl, color=styling.COLOR_PRIMARY, lw=styling.MPL_RCPARAMS['lines.linewidth'])
    ax.axhline(0, color=styling.COLOR_TEXT, lw=0.8, alpha=0.25)
    ax.set_xlabel(energy_label, fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('Global KL divergence', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title(f"Full-QSS KL spectrum: {condition_label}", fontsize=styling.FONT_SIZE_TITLE)
    if invert_energy_axis:
        ax.invert_xaxis()
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def create_sliding_kl_heatmap_figure(
    energies:np.ndarray,
    centers:np.ndarray,
    kl_matrix:np.ndarray,
    energy_label:str,
    invert_energy_axis:bool,
    log_scale:bool,
    eps:float,
) -> Figure:
    """Display windowed KL as a heatmap over wavenumber and window-center time.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Energy axis (columns of the matrix).
    centers : ndarray, shape (n_windows,)
        Sliding-window centre times (rows of the matrix).
    kl_matrix : ndarray, shape (n_windows, n_energy)
        KL divergence per window and energy.
    energy_label : str
        Axis label for the energy axis (matplotlib mathtext allowed).
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.
    log_scale : bool
        If True colour is mapped on a logarithmic norm.
    eps : float
        Small positive floor for the logarithmic norm.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))
    plot_matrix = np.where(np.isfinite(kl_matrix), kl_matrix, np.nan)

    # Percentile-clipped log norm keeps the colour range robust to outliers.
    norm = None
    if log_scale:
        positive = plot_matrix[np.isfinite(plot_matrix) & (plot_matrix > 0)]
        if positive.size > 0:
            vmin = max(np.nanpercentile(positive, 1), eps)
            vmax = np.nanpercentile(positive, 99)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))

    # Sort both axes with the matrix before placing it in the image extent.
    energies = np.asarray(energies, dtype=float)
    centers = np.asarray(centers, dtype=float)
    e_order = np.argsort(energies)
    c_order = np.argsort(centers)
    energies = energies[e_order]
    centers = centers[c_order]
    plot_matrix = plot_matrix[np.ix_(c_order, e_order)]

    extent = [np.nanmin(energies), np.nanmax(energies),
              np.nanmin(centers), np.nanmax(centers)]
    im = ax.imshow(plot_matrix, aspect='auto', origin='lower', extent=extent,
                   cmap=build_gruvbox_cmap(), interpolation='nearest', norm=norm)
    if invert_energy_axis:
        ax.invert_xaxis()
    ax.set_xlabel(energy_label, fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('Sliding-window center time', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title('Sliding-window KL matrix, K(window, energy)', fontsize=styling.FONT_SIZE_TITLE)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('KL divergence' + (' (log scale)' if log_scale else ''), fontsize=styling.FONT_SIZE_BASE)
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def create_photon_count_histogram_grid_figure(
    panels:Sequence[dict],
    nrows:int,
    ncols:int,
) -> Figure:
    """Plot count histograms and their Poisson overlays in row-major order.

    Each occupied panel identifies the channel, sample count, mean count and full-
    record KL divergence.

    Parameters
    ----------
    panels : Sequence[dict]
        Per-histogram descriptions, each holding energy, count_values,
        hist_counts, poisson_x, poisson_y, lam, n_obs, bin_width and kl.
    nrows : int
        Number of grid rows.
    ncols : int
        Number of grid columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(styling.narrow * ncols, styling.low * nrows))
    axes = np.atleast_2d(axes)

    for cell in range(nrows * ncols):
        row, col = cell // ncols, cell % ncols
        ax = axes[row, col]

        if cell >= len(panels):
            ax.axis('off')
            continue

        p = panels[cell]
        count_values = np.asarray(p['count_values'], dtype=float)
        hist_counts = np.asarray(p['hist_counts'], dtype=float)
        poisson_x = np.asarray(p['poisson_x'], dtype=float)
        poisson_y = np.asarray(p['poisson_y'], dtype=float)

        ax.bar(count_values, hist_counts, width=p['bin_width'], align='center',
               color=styling.COLOR_SECONDARY, edgecolor=styling.COLOR_TEXT, linewidth=0.4,
               alpha=0.85)
        ax.plot(poisson_x, poisson_y, color=styling.COLOR_HIGHLIGHT, lw=styling.MPL_RCPARAMS['lines.linewidth'])

        # Focus the view on the populated region (observed data or Poisson mass).
        populated = np.concatenate([count_values[hist_counts > 0], poisson_x[poisson_y > 1e-3]])
        if populated.size > 0:
            pad = max(5, int(0.05 * (populated.max() - populated.min() + 1)))
            ax.set_xlim(max(0, populated.min() - pad), populated.max() + pad)

        ax.set_title(f"{p['energy']:.2f} cm$^{{-1}}$", fontsize=styling.FONT_SIZE_BASE - 1)
        ax.text(
            0.97, 0.95,
            f"N={p['n_obs']}\n$\\lambda$={p['lam']:.1f}\nKL={p['kl']:.3g}",
            transform=ax.transAxes, ha='right', va='top', fontsize=styling.FONT_SIZE_BASE - 2,
            bbox=dict(boxstyle='round', facecolor=styling.COLOR_BACKGROUND,
                      edgecolor=styling.COLOR_NEUTRAL, alpha=0.85))
        ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE - 2)

    fig.supxlabel('Photon count', fontsize=styling.FONT_SIZE_LABEL)
    fig.supylabel('Number of timepoints', fontsize=styling.FONT_SIZE_LABEL)
    fig.suptitle('Photon-count histograms with matched Poisson references', fontsize=styling.FONT_SIZE_LABEL)
    fig.tight_layout()
    return fig


def create_intensity_time_traces_figure(
    time_axis:np.ndarray,
    traces:Sequence[np.ndarray],
    energies:Sequence[float],
    condition_label:str,
) -> Figure:
    """Plot raw intensities at selected channels against elapsed time.

    Parameters
    ----------
    time_axis : ndarray, shape (n_time,)
        Time axis in seconds.
    traces : Sequence[np.ndarray]
        One raw-intensity series per channel, each of shape (n_time,).
    energies : Sequence[float]
        Matched energies in cm^-1, one per trace (legend labels).
    condition_label : str
        Human-readable dataset label used in the title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))
    for trace, energy in zip(traces, energies):
        ax.plot(time_axis, trace, lw=styling.MPL_RCPARAMS['lines.linewidth'], label=f"{energy:.1f} cm$^{{-1}}$")

    ax.set_xlabel('Time (s)', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('Raw intensity (arb. units)', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title(f"Raw intensity vs time: {condition_label}", fontsize=styling.FONT_SIZE_TITLE)
    ax.legend(frameon=False, fontsize=styling.FONT_SIZE_BASE - 1, title='Wavenumber')
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def create_windowed_kl_time_traces_figure(
    centers:np.ndarray,
    traces:Sequence[np.ndarray],
    energies:Sequence[float],
    condition_label:str,
) -> Figure:
    """Plot the KL history of each selected channel at the window centers.

    Parameters
    ----------
    centers : ndarray, shape (n_windows,)
        Sliding-window centre times in seconds.
    traces : Sequence[np.ndarray]
        One windowed-KL series per channel, each of shape (n_windows,).
    energies : Sequence[float]
        Matched energies in cm^-1, one per trace (legend labels).
    condition_label : str
        Human-readable dataset label used in the title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))
    for trace, energy in zip(traces, energies):
        ax.plot(centers, trace, lw=styling.MPL_RCPARAMS['lines.linewidth'], label=f"{energy:.1f} cm$^{{-1}}$")

    ax.set_xlabel('Sliding-window center time (s)', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('Windowed KL divergence', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title(f"Windowed KL vs time: {condition_label}", fontsize=styling.FONT_SIZE_TITLE)
    ax.legend(frameon=False, fontsize=styling.FONT_SIZE_BASE - 1, title='Wavenumber')
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def create_gain_calibration_figure(
    energies:np.ndarray,
    inverse_gain:np.ndarray,
    used_energies:np.ndarray,
    used_inverse_gain:np.ndarray,
    fitted_inverse_gain:Optional[np.ndarray],
    energy_label:str,
    invert_energy_axis:bool,
    raw:bool = False,
) -> Figure:
    """Plot reciprocal gain against wavenumber for the selected calibration mode.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Full energy axis in cm^-1.
    inverse_gain : ndarray, shape (n_energy,)
        Reciprocal of the final per-channel gain.
    used_energies : ndarray, shape (n_used,)
        Energies of the channels whose gain was measured.
    used_inverse_gain : ndarray, shape (n_used,)
        Reciprocal of the measured gain at those channels.
    fitted_inverse_gain : Optional[np.ndarray] (shape: (n_energy,))
        Reciprocal of the cubic approximation, or None to omit it.
    energy_label : str
        Axis label for the energy axis (matplotlib mathtext allowed).
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.

    raw : bool, default=False
        Show only measured points, without fitted curves or regression bands.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))
    ax.plot(used_energies, used_inverse_gain, marker='.', linestyle='None',
            color=styling.COLOR_NEUTRAL, alpha=0.55, label='Measured 1/g')
    if not raw:
        ax.plot(energies, np.where(inverse_gain < 0.0, np.nan, inverse_gain), color=styling.COLOR_PRIMARY, lw=styling.MPL_RCPARAMS['lines.linewidth'], label='Final 1/g')
    if fitted_inverse_gain is not None:
        ax.plot(energies, np.where(fitted_inverse_gain < 0.0, np.nan, fitted_inverse_gain), color=styling.COLOR_HIGHLIGHT, lw=styling.MPL_RCPARAMS['lines.linewidth'], ls='--',
                label='Cubic approximation')

    if np.any(inverse_gain < 0.0):
        ax.plot(energies, np.where(inverse_gain < 0.0, inverse_gain, np.nan),
                color='#ff0000', marker='.', label='Negative fitted gain')

    ax.set_xlabel(energy_label, fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('1 / gain', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title('Gain calibration: 1/g(E)', fontsize=styling.FONT_SIZE_TITLE)
    if invert_energy_axis:
        ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=styling.FONT_SIZE_BASE - 1)
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def create_gain_intensity_figure(
    means_used:np.ndarray,
    gain_used:np.ndarray,
    fit_x:np.ndarray,
    fit_y:np.ndarray,
    ci_low:np.ndarray,
    ci_high:np.ndarray,
    pi_low:np.ndarray,
    pi_high:np.ndarray,
    intensity_label:str,
    level:float,
    raw:bool = False,
) -> Figure:
    """Plot measured gain against channel mean intensity, with fitted intervals in
    cubic modes.

    Parameters
    ----------
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
    intensity_label : str
        Axis label for the intensity axis (matplotlib mathtext allowed).
    level : float
        Confidence level of the bands, used only for the legend labels.

    raw : bool, default=False
        Show only measured points, without fitted curves or regression bands.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    pct = f"{level * 100:.0f}%"
    fig, ax = plt.subplots(figsize=(styling.wide, styling.high_med))

    if not raw:
        # Prediction band underneath (wider, lighter), then the mean-fit band on top.
        ax.fill_between(fit_x, pi_low, pi_high, color=styling.COLOR_NEUTRAL,
                        alpha=0.18, lw=0, label=f"{pct} prediction interval")
        ax.fill_between(fit_x, ci_low, ci_high, color=styling.COLOR_SECONDARY,
                        alpha=0.30, lw=0, label=f"{pct} confidence interval")

    ax.plot(means_used, gain_used, marker='.', linestyle='None',
            color=styling.COLOR_NEUTRAL, alpha=0.55, label='Measured var/mean')
    if not raw:
        ax.plot(fit_x, np.where(fit_y < 0.0, np.nan, fit_y), color=styling.COLOR_PRIMARY, lw=styling.MPL_RCPARAMS['lines.linewidth'], label='Cubic fit')

    if np.any(fit_y < 0.0):
        ax.plot(fit_x, np.where(fit_y < 0.0, fit_y, np.nan),
                color='#ff0000', marker='.', label='Negative fitted gain')

    ax.set_xlabel(intensity_label, fontsize=styling.FONT_SIZE_LABEL)
    ax.set_ylabel('Gain (var / mean)', fontsize=styling.FONT_SIZE_LABEL)
    ax.set_title('Gain calibration: g(mean intensity)', fontsize=styling.FONT_SIZE_TITLE)
    ax.legend(frameon=False, fontsize=styling.FONT_SIZE_BASE - 1)
    ax.tick_params(axis='both', labelsize=styling.FONT_SIZE_BASE)
    fig.tight_layout()
    return fig


def export_figure(
    fig:Figure,
    png_dir:str,
    svg_dir:str,
    base_name:str,
    dpi:int,
) -> Tuple[str, str]:
    """Write the figure as PNG and editable-text SVG, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write out.
    png_dir : str
        Directory for the PNG output.
    svg_dir : str
        Directory for the SVG output.
    base_name : str
        Filename stem (no extension).
    dpi : int
        Raster resolution for the PNG.

    Returns
    -------
    paths : Tuple[str, str]
        (png_path, svg_path) of the written files.
    """
    png_path = os.path.join(png_dir, f"{base_name}.png")
    svg_path = os.path.join(svg_dir, f"{base_name}.svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor=styling.COLOR_BACKGROUND)
    fig.savefig(svg_path, bbox_inches='tight', facecolor=styling.COLOR_BACKGROUND)
    plt.close(fig)
    return png_path, svg_path
