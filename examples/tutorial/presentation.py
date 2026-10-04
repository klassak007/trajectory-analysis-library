"""Presentation defaults for the worked tutorials; no TAL analysis is hidden here."""

from types import ModuleType

import matplotlib.pyplot as plt
from cycler import cycler
from IPython.display import display
from matplotlib.figure import Figure

BLUE = "#2864A0"
ORANGE = "#C76528"
TEAL = "#23857B"
GRAY = "#687582"
PALETTE = (BLUE, ORANGE, TEAL, "#8B68A6", "#A18A30", GRAY)


def configure() -> None:
    """Use readable, consistent defaults for static and interactive figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.axisbelow": True,
            "grid.color": "#DDE3E8",
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.figsize": (8, 3.6),
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def configure_interactive(hv: ModuleType) -> None:
    """Apply the same palette and typography to HoloViews' Bokeh backend."""
    fonts = {"title": "13pt", "labels": "11pt", "xticks": "10pt", "yticks": "10pt"}
    hv.opts.defaults(
        hv.opts.Curve(
            width=760,
            height=320,
            line_width=2,
            color=hv.Cycle(list(PALETTE)),
            fontsize=fonts,
        ),
        hv.opts.Scatter(
            width=760, height=320, size=6, color=hv.Cycle(list(PALETTE)), fontsize=fonts
        ),
    )


def show(fig: Figure) -> None:
    """Display once and close the figure to keep notebook execution tidy."""
    display(fig)
    plt.close(fig)
