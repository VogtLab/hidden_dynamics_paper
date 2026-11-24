"""Interactive figures and HTML export for gain calibration and KL analysis."""

import os
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import styling

_WHITE_LAYOUT = dict(
    autosize=True,
    plot_bgcolor=styling.COLOR_BACKGROUND,
    paper_bgcolor=styling.COLOR_BACKGROUND,
    colorway=list(styling.COLOR_CYCLE),
    font=dict(family=styling.FONT_FAMILY_PLOTLY, size=styling.FONT_SIZE_BASE, color=styling.COLOR_TEXT),
    margin=dict(l=70, r=40, t=70, b=60),
)
_AXIS_STYLE = dict(showgrid=False, zeroline=False, linecolor=styling.COLOR_TEXT,
                   linewidth=styling.MPL_RCPARAMS['axes.linewidth'], ticks='outside', mirror=False)


def _html_label(label:str) -> str:
    """Format inverse-centimetre superscripts for Plotly axis titles.

    Parameters
    ----------
    label : str
        Axis label that may contain ``$...$`` mathtext.

    Returns
    -------
    text : str
        Label with ``cm^-1`` style superscripts rendered as HTML ``<sup>``.
    """
    return (label.replace('cm$^{-1}$', 'cm<sup>-1</sup>')
                 .replace('$^{-1}$', '<sup>-1</sup>')
                 .replace('$', ''))


def build_gruvbox_colorscale() -> list:
    """Convert the shared colour gradient to Plotly positions and colours.

    Returns
    -------
    colorscale : list
        List of ``[position, color]`` pairs derived from
        ``styling.COLORSCALE_GRADIENT``.
    """
    return [[float(pos), color] for pos, color in styling.COLORSCALE_GRADIENT]


def create_global_kl_figure(
    energies:np.ndarray,
    global_kl:np.ndarray,
    energy_label:str,
    condition_label:str,
    invert_energy_axis:bool,
) -> go.Figure:
    """Plot one full-record KL value per spectral channel.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Energy axis used in the analysis.
    global_kl : ndarray, shape (n_energy,)
        Per-channel KL divergence over the whole record.
    energy_label : str
        Axis label for the energy axis (mathtext is converted to HTML).
    condition_label : str
        Human-readable dataset label used in the title.
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line_color=styling.COLOR_TEXT, line_width=0.8, opacity=0.25)
    fig.add_trace(go.Scatter(x=energies, y=global_kl, mode='lines',
                             line=dict(color=styling.COLOR_PRIMARY, width=styling.MPL_RCPARAMS['lines.linewidth']), name='Global KL'))
    fig.update_layout(title=f"Full-QSS KL spectrum: {condition_label}", **_WHITE_LAYOUT)
    fig.update_xaxes(title_text=_html_label(energy_label),
                     autorange='reversed' if invert_energy_axis else True, **_AXIS_STYLE)
    fig.update_yaxes(title_text='Global KL divergence', **_AXIS_STYLE)
    return fig


def create_sliding_kl_surface_figure(
    energies:np.ndarray,
    centers:np.ndarray,
    kl_matrix:np.ndarray,
    energy_label:str,
    invert_energy_axis:bool,
    log_scale:bool,
    eps:float,
) -> go.Figure:
    """Display windowed KL as a surface over wavenumber and window-center time.

    The interactive display samples at most 250 coordinates per axis.

    Parameters
    ----------
    energies : ndarray, shape (n_energy,)
        Energy axis (x).
    centers : ndarray, shape (n_windows,)
        Sliding-window centre times (y).
    kl_matrix : ndarray, shape (n_windows, n_energy)
        KL divergence per window and energy (z).
    energy_label : str
        Axis label for the energy axis (mathtext is converted to HTML).
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.
    log_scale : bool
        If True the z surface is mapped on log10 of the KL values.
    eps : float
        Small positive floor applied before taking log10.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """

    energies = np.asarray(energies, dtype=float)
    centers = np.asarray(centers, dtype=float)
    kl_matrix = np.asarray(kl_matrix, dtype=float)
    if kl_matrix.shape[0] != centers.shape[0] or kl_matrix.shape[1] != energies.shape[0]:
        raise ValueError('kl_matrix.shape must be (centers.size, energies.size).')

    z_full = np.where(np.isfinite(kl_matrix), kl_matrix, np.nan)
    if log_scale:
        z_full = np.where(z_full > 0, np.log10(np.maximum(z_full, eps)), np.nan)
        z_title = 'KL divergence (log10)'
    else:
        z_title = 'KL divergence'

    # Down-sample each axis to at most 250 points to keep the html light.
    e_ind = (np.linspace(0, energies.shape[0] - 1, 250, dtype=int)
             if energies.shape[0] > 250 else np.arange(energies.shape[0]))
    t_ind = (np.linspace(0, centers.shape[0] - 1, 250, dtype=int)
             if centers.shape[0] > 250 else np.arange(centers.shape[0]))
    energy_filt = energies[e_ind]
    center_filt = centers[t_ind]
    z_filt = z_full[np.ix_(t_ind, e_ind)]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=energy_filt, y=center_filt, z=z_filt,
        colorscale=build_gruvbox_colorscale(), colorbar=dict(title=z_title)))

    elevation, azimuth = np.radians([35.26, 255.0])
    eye = dict(x=2.165 * np.cos(elevation) * np.cos(azimuth),
               y=2.165 * np.cos(elevation) * np.sin(azimuth),
               z=2.165 * np.sin(elevation))
    _scene_axis = dict(backgroundcolor=styling.COLOR_BACKGROUND, gridcolor=styling.COLOR_GRID,
                       showbackground=False, zeroline=False, showline=True,
                       linecolor=styling.COLOR_TEXT, ticks='outside', tickcolor=styling.COLOR_TEXT)
    fig.update_layout(
        title='Sliding-window KL matrix, K(window, energy)',
        scene=dict(
            xaxis=dict(**_scene_axis, title=_html_label(energy_label),
                       autorange='reversed' if invert_energy_axis else True),
            yaxis=dict(**_scene_axis, title='Sliding-window center time'),
            zaxis=dict(**_scene_axis, title=z_title),
            camera=dict(eye=eye),
        ),
        **_WHITE_LAYOUT)
    return fig


def create_photon_count_histogram_grid_figure(
    panels:Sequence[dict],
    nrows:int,
    ncols:int,
) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    titles = [''] * (nrows * ncols)
    for p_idx, p in enumerate(panels):
        titles[p_idx] = (
            f"{p['energy']:.2f} cm<sup>-1</sup><br>"
            f"N={p['n_obs']} \u03bb={p['lam']:.1f} KL={p['kl']:.3g}")

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.09)

    for cell in range(nrows * ncols):
        row, col = cell // ncols, cell % ncols

        if cell >= len(panels):
            fig.update_xaxes(visible=False, row=row + 1, col=col + 1)
            fig.update_yaxes(visible=False, row=row + 1, col=col + 1)
            continue

        p = panels[cell]
        count_values = np.asarray(p['count_values'], dtype=float)
        hist_counts = np.asarray(p['hist_counts'], dtype=float)
        poisson_x = np.asarray(p['poisson_x'], dtype=float)
        poisson_y = np.asarray(p['poisson_y'], dtype=float)

        fig.add_trace(go.Bar(
            x=count_values, y=hist_counts, width=p['bin_width'],
            marker=dict(color=styling.COLOR_SECONDARY, line=dict(color=styling.COLOR_TEXT, width=0.4)),
            opacity=0.85, showlegend=False), row=row + 1, col=col + 1)
        fig.add_trace(go.Scatter(
            x=poisson_x, y=poisson_y, mode='lines',
            line=dict(color=styling.COLOR_HIGHLIGHT, width=styling.MPL_RCPARAMS['lines.linewidth']), showlegend=False),
            row=row + 1, col=col + 1)

        # Focus the view on the populated region (observed data or Poisson mass).
        populated = np.concatenate([count_values[hist_counts > 0], poisson_x[poisson_y > 1e-3]])
        if populated.size > 0:
            pad = max(5, int(0.05 * (populated.max() - populated.min() + 1)))
            fig.update_xaxes(range=[max(0, populated.min() - pad), populated.max() + pad],
                             row=row + 1, col=col + 1)

    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    fig.update_layout(
        title='Photon-count histograms with matched Poisson references',
        bargap=0, **_WHITE_LAYOUT)

    for ann in fig.layout.annotations:
        ann.font.size = 11

    fig.add_annotation(text='Photon count', x=0.5, y=-0.06, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=styling.FONT_SIZE_LABEL))
    fig.add_annotation(text='Number of timepoints', x=-0.06, y=0.5, xref='paper', yref='paper',
                       showarrow=False, textangle=-90, font=dict(size=styling.FONT_SIZE_LABEL))
    return fig


def create_intensity_time_traces_figure(
    time_axis:np.ndarray,
    traces:Sequence[np.ndarray],
    energies:Sequence[float],
    condition_label:str,
) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    fig = go.Figure()
    for trace, energy in zip(traces, energies):
        fig.add_trace(go.Scatter(
            x=time_axis, y=trace, mode='lines',
            line=dict(width=styling.MPL_RCPARAMS['lines.linewidth']), name=f"{energy:.1f} cm<sup>-1</sup>"))

    fig.update_layout(title=f"Raw intensity vs time: {condition_label}",
                      legend_title_text='Wavenumber', **_WHITE_LAYOUT)
    fig.update_xaxes(title_text='Time (s)', **_AXIS_STYLE)
    fig.update_yaxes(title_text='Raw intensity (arb. units)', **_AXIS_STYLE)
    return fig


def create_windowed_kl_time_traces_figure(
    centers:np.ndarray,
    traces:Sequence[np.ndarray],
    energies:Sequence[float],
    condition_label:str,
) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    fig = go.Figure()
    for trace, energy in zip(traces, energies):
        fig.add_trace(go.Scatter(
            x=centers, y=trace, mode='lines',
            line=dict(width=styling.MPL_RCPARAMS['lines.linewidth']), name=f"{energy:.1f} cm<sup>-1</sup>"))

    fig.update_layout(title=f"Windowed KL vs time: {condition_label}",
                      legend_title_text='Wavenumber', **_WHITE_LAYOUT)
    fig.update_xaxes(title_text='Sliding-window center time (s)', **_AXIS_STYLE)
    fig.update_yaxes(title_text='Windowed KL divergence', **_AXIS_STYLE)
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
) -> go.Figure:
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
        Axis label for the energy axis (mathtext is converted to HTML).
    invert_energy_axis : bool
        If True the energy axis is drawn high-to-low.

    raw : bool, default=False
        Show only measured points, without fitted curves or regression bands.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=used_energies, y=used_inverse_gain, mode='markers',
        marker=dict(color=styling.COLOR_NEUTRAL, size=4, opacity=0.55),
        name='Measured 1/g'))
    if not raw:
        fig.add_trace(go.Scatter(
            x=energies, y=np.where(inverse_gain < 0.0, np.nan, inverse_gain), mode='lines',
            line=dict(color=styling.COLOR_PRIMARY, width=styling.MPL_RCPARAMS['lines.linewidth']), name='Final 1/g'))
    if fitted_inverse_gain is not None:
        fig.add_trace(go.Scatter(
            x=energies, y=np.where(fitted_inverse_gain < 0.0, np.nan, fitted_inverse_gain), mode='lines',
            line=dict(color=styling.COLOR_HIGHLIGHT, width=styling.MPL_RCPARAMS['lines.linewidth'], dash='dash'),
            name='Cubic approximation'))

    if np.any(inverse_gain < 0.0):
        fig.add_trace(go.Scatter(
            x=energies, y=np.where(inverse_gain < 0.0, inverse_gain, np.nan),
            mode='lines+markers', line=dict(color='#ff0000'),
            marker=dict(color='#ff0000', size=4), name='Negative fitted gain'))

    fig.update_layout(title='Gain calibration: 1/g(E)', **_WHITE_LAYOUT)
    fig.update_xaxes(title_text=_html_label(energy_label),
                     autorange='reversed' if invert_energy_axis else True, **_AXIS_STYLE)
    fig.update_yaxes(title_text='1 / gain', **_AXIS_STYLE)
    return fig


def _hex_to_rgba(hex_color:str, alpha:float) -> str:
    """Format an RGB hex colour and opacity as a Plotly RGBA string.

    Parameters
    ----------
    hex_color : str
        Colour in ``#rrggbb`` form.
    alpha : float
        Opacity in [0, 1].

    Returns
    -------
    rgba : str
        ``rgba(r, g, b, alpha)`` string.
    """
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


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
) -> go.Figure:
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
        Axis label for the intensity axis (mathtext is converted to HTML).
    level : float
        Confidence level of the bands, used only for the legend labels.

    raw : bool, default=False
        Show only measured points, without fitted curves or regression bands.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The rendered figure.
    """
    pct = f"{level * 100:.0f}%"
    x = np.asarray(fit_x, dtype=float)
    x_ring = np.concatenate([x, x[::-1]])

    fig = go.Figure()

    if not raw:
        # Prediction band underneath (wider, lighter), then the mean-fit band on top.
        fig.add_trace(go.Scatter(
            x=x_ring, y=np.concatenate([np.asarray(pi_high), np.asarray(pi_low)[::-1]]),
            fill='toself', fillcolor=_hex_to_rgba(styling.COLOR_NEUTRAL, 0.18),
            line=dict(width=0), hoverinfo='skip', name=f"{pct} prediction interval"))
        fig.add_trace(go.Scatter(
            x=x_ring, y=np.concatenate([np.asarray(ci_high), np.asarray(ci_low)[::-1]]),
            fill='toself', fillcolor=_hex_to_rgba(styling.COLOR_SECONDARY, 0.30),
            line=dict(width=0), hoverinfo='skip', name=f"{pct} confidence interval"))

    fig.add_trace(go.Scatter(
        x=means_used, y=gain_used, mode='markers',
        marker=dict(color=styling.COLOR_NEUTRAL, size=4, opacity=0.55),
        name='Measured var/mean'))
    if not raw:
        fig.add_trace(go.Scatter(
            x=x, y=np.where(fit_y < 0.0, np.nan, fit_y), mode='lines',
            line=dict(color=styling.COLOR_PRIMARY, width=styling.MPL_RCPARAMS['lines.linewidth']), name='Cubic fit'))

    if np.any(fit_y < 0.0):
        fig.add_trace(go.Scatter(
            x=x, y=np.where(fit_y < 0.0, fit_y, np.nan), mode='lines+markers',
            line=dict(color='#ff0000'), marker=dict(color='#ff0000', size=4),
            name='Negative fitted gain'))

    fig.update_layout(title='Gain calibration: g(mean intensity)', **_WHITE_LAYOUT)
    fig.update_xaxes(title_text=_html_label(intensity_label), **_AXIS_STYLE)
    fig.update_yaxes(title_text='Gain (var / mean)', **_AXIS_STYLE)
    return fig


def export_figure(fig:go.Figure, html_dir:str, base_name:str) -> str:
    """Save a responsive HTML figure that loads Plotly.js from its CDN.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to write out.
    html_dir : str
        Directory for the HTML output.
    base_name : str
        Filename stem (no extension).

    Returns
    -------
    html_path : str
        Path of the written HTML file.
    """
    html_path = os.path.join(html_dir, f"{base_name}.html")
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True,
                   config={'responsive': True},
                   default_width='100%', default_height='100vh')
    return html_path
