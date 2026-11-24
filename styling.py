"""Shared colour palette, font settings and figure export sizes."""

light = '#DFCFBC'
mid_light = '#928877'
mid_dark = '#3D5D72'
dark = '#103547'

version:str = '20260921_2104'

COLORSCALE_GRADIENT = [(0.00, light), (0.33, mid_light), (0.67, mid_dark), (1.00, dark)]
COLOR_CYCLE = [light, mid_light, mid_dark, dark]
COLOR_NEUTRAL = light
COLOR_PRIMARY = mid_dark
COLOR_SECONDARY = mid_light
COLOR_HIGHLIGHT = dark
COLOR_BACKGROUND = '#ffffff'
COLOR_TEXT = '#282828'
COLOR_GRID = '#dddddd'

FONT_FAMILY = ['Liberation Sans', 'DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
FONT_FAMILY_PLOTLY = 'Liberation Sans, DejaVu Sans, Helvetica, Arial, sans-serif'
FONT_SIZE_BASE = 11.0
FONT_SIZE_LABEL = 12.0
FONT_SIZE_TITLE = 13.0
PNG_DPI = 600

# Matplotlib figure dimensions in inches.
wide = 7.0
narrow = 3.0
high_med = 4.0
low = 3.0

# Keep SVG labels editable and embed TrueType glyphs in PDF and PostScript.
MPL_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_FAMILY,
    'font.size': FONT_SIZE_BASE,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_LABEL,
    'xtick.labelsize': FONT_SIZE_BASE - 1.0,
    'ytick.labelsize': FONT_SIZE_BASE - 1.0,
    'legend.fontsize': FONT_SIZE_BASE - 1.0,
    'figure.titlesize': FONT_SIZE_TITLE + 1.0,
    'axes.unicode_minus': True,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.color': COLOR_TEXT,
    'axes.labelcolor': COLOR_TEXT,
    'axes.edgecolor': COLOR_TEXT,
    'xtick.color': COLOR_TEXT,
    'ytick.color': COLOR_TEXT,
    'axes.facecolor': COLOR_BACKGROUND,
    'figure.facecolor': COLOR_BACKGROUND,
    'savefig.facecolor': COLOR_BACKGROUND,
    'savefig.edgecolor': COLOR_BACKGROUND,
    'savefig.dpi': PNG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.dpi': 100,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6.0,
}
