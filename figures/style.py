"""Shared figure style: Okabe-Ito palette, fixed colours per temperature, fixed markers per R, column widths, save helper, sphere-density helper."""
import os, json, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
W1, W2 = 3.4, 7.0                                           # single / double column width (in)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#7F7F7F")
# same quantity -> same colour in every figure.  Nematic temperatures (T/J) and kagome temperatures (T/J) use disjoint hues.
COL_LL = {0.80: OI["blue"], 0.95: OI["sky"], 1.05: OI["orange"], 1.10: OI["verm"], 1.12: OI["purple"], 1.14: OI["green"], 1.18: OI["grey"], 1.30: OI["black"]}
COL_KAG = {0.0005: OI["blue"], 0.001: OI["sky"], 0.002: OI["orange"], 0.004: OI["verm"]}
MARK_R = {1: "o", 2: "s", 3: "^", 4: "D", 6: "v", 12: "P"}
# one colour per information channel (used in Figs 8, 13, 14)
COL_CH = dict(director=OI["blue"], amp=OI["orange"], biax=OI["green"], local=OI["sky"], defect=OI["verm"], ring=OI["purple"], colour=OI["grey"], vae=OI["black"])
LAYOUT = {}
def setup():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
        "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
setup()
def panel(ax, s, dx=-0.02, dy=1.02):
    dy = max(dy, 1.06) if dy > 1.0 else dy
    ax.text(dx - 0.04, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
def savefig(fig, name):
    os.makedirs(HERE, exist_ok=True)
    import overlap_check; LAYOUT[name] = overlap_check.check(fig, name)
    for ext in ("pdf", "png"): fig.savefig(f"{HERE}/{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
def load(path):                                                # repo-relative JSON
    return json.load(open(f"{ROOT}/{path}"))
def git_hash():
    try: return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception: return "unknown"
# ------------------------------------------------------------------ sphere densities
def plot_sphere_density(ax, logp_fn, frame=None, n=(180, 90), vmax=None, cmap="viridis", proj="mollweide", label=True):
    """Colour = probability density p(n) on the sphere, Mollweide projection (lon=phi, lat=pi/2-theta of the *frame* axes).
    logp_fn: callable taking unit vectors [...,3] (in the plotting frame) -> log density (unnormalised ok; normalised here by quadrature).
    frame: optional 3x3 matrix F; vectors are evaluated as n_data = n_frame @ F.T (columns of F = frame axes in data coordinates).  Returns the image and the max density (for a shared colour range)."""
    lon = np.linspace(-np.pi, np.pi, n[0]); lat = np.linspace(-np.pi / 2, np.pi / 2, n[1]); LON, LAT = np.meshgrid(lon, lat)
    X = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    V = X if frame is None else X @ np.asarray(frame).T
    lp = logp_fn(V); p = np.exp(lp - lp.max()); w = np.cos(LAT); dA = (lon[1] - lon[0]) * (lat[1] - lat[0]); p = p / (p * w).sum() / dA        # normalise: integral p dOmega = 1
    vm = p.max() if vmax is None else vmax
    im = ax.pcolormesh(LON, LAT, p, shading="auto", cmap=cmap, vmin=0, vmax=vm, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.spines["geo"].set_linewidth(0.5) if "geo" in ax.spines else None
    return im, p.max()
def Ylm_table(L):
    """real spherical harmonics (integral norm, scipy convention, orthonormal) for l=1..L as a function of unit vectors -> [..., n_lm]."""
    from scipy.special import sph_harm_y
    def f(V):
        th = np.arccos(np.clip(V[..., 2], -1, 1)); ph = np.arctan2(V[..., 1], V[..., 0]); cols = []
        for l in range(1, L + 1):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, abs(m), th, ph); cols.append(np.sqrt(2) * (-1) ** m * Y.imag if m < 0 else (Y.real if m == 0 else np.sqrt(2) * (-1) ** m * Y.real))
        return np.stack(cols, -1)
    return f
