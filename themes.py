"""Color themes for the plots. Use: from themes import apply_theme; apply_theme("light")

The six line colors of every theme except "ordered" are checked in their listed order:
neighboring colors stay distinguishable for colorblind readers (protan/deutan) and
with normal vision, and every color is readable against the theme's background.
"""

import matplotlib.pyplot as plt
from cycler import cycler


def chrome(surface, ink, muted, grid, axis):  # background, text, tick labels, grid lines, axis lines
    return {"surface": surface, "ink": ink, "muted": muted, "grid": grid, "axis": axis}


LIGHT_CHROME = chrome("#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7")

THEMES = {
    # ---------- light ----------
    "light": {"colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], **LIGHT_CHROME},
    # one blue, light to dark along the chain: site 1 lightest, site 6 darkest (not colorblind-checked)
    "ordered": {"colors": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"], **LIGHT_CHROME},

    # ---------- dark ----------
    # neutral charcoal
    "dark": {"colors": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
             **chrome("#1a1a19", "#ffffff", "#898781", "#2c2c2a", "#383835")},
    # deep navy with clear, saturated colors
    "midnight": {"colors": ["#05ac7d", "#9163d5", "#e76444", "#0593e6", "#bf8909", "#cd5394"],
                 **chrome("#0f172a", "#e2e8f0", "#94a3b8", "#1e293b", "#334155")},
    # cool slate gray with soft, slightly muted colors
    "nord": {"colors": ["#c86265", "#369dd1", "#70a550", "#a86ab6", "#b58d2d", "#35a7a1"],
             **chrome("#2e3440", "#eceff4", "#a7b1c2", "#3b4252", "#4c566a")},
    # purple-gray with vivid pink, orange and cyan
    "dracula": {"colors": ["#db509c", "#d57a16", "#05a5b3", "#a19705", "#8a66d9", "#35ad44"],
                **chrome("#282a36", "#f8f8f2", "#a4acc8", "#343746", "#44475a")},
    # dark teal with warm, earthy colors
    "solarized": {"colors": ["#d6483e", "#398ad6", "#b88c19", "#c34e97", "#7c9205", "#04a19b"],
                  **chrome("#002b36", "#eee8d5", "#93a1a1", "#073642", "#586e75")},
    # pure black with the most saturated colors, good for screens and projectors
    "oled": {"colors": ["#04a78b", "#ed6300", "#c33dbd", "#ae9203", "#047df1", "#d73246"],
             **chrome("#000000", "#ffffff", "#8f8f8f", "#1c1c1c", "#3a3a3a")},

    "matplotlib": None,  # matplotlib's default look
}


def apply_theme(name):  # sets the theme as matplotlib defaults, call before creating figures
    theme = THEMES[name]
    if theme is None:
        return

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=theme["colors"]),
        "lines.linewidth": 1.5,
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "legend.facecolor": theme["surface"],
        "legend.edgecolor": theme["grid"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink"],
        "axes.titlecolor": theme["ink"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "axes.edgecolor": theme["axis"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": theme["grid"],
        "grid.linewidth": 0.6,
    })
