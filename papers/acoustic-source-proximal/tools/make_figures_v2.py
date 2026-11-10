#!/usr/bin/env python3
"""Figures for version 2 of the source-proximal sound-field paper, drawn at their printed size.

Run from the paper folder:

    python3 -B tools/make_figures_v2.py [STUDY_ROOT]

STUDY_ROOT defaults to the frozen study bundle of this repository (archive/2026-11-05-taslp/acoustic). Every interval
and every win/tie/loss class is read from the registered summary files; nothing is recomputed. Only the geometry
panels (positions read from the simulation module and the measured preprocessing archive) and the order histogram
(sums of the registered per-condition counts) aggregate raw values.

Writes figures/v2-geometry.pdf, v2-simulation.pdf, v2-measured.pdf (main: best generic, arrival-time models),
v2-measured-variants.pdf (supplement: five-ball variants), v2-competitors.pdf, v2-miracle.pdf, v2-absolute.pdf
(absolute error on both measured grids), v2-order.pdf,
v2-identifiability.pdf (theory curves of the post hoc identifiability bound; kept as a record, no longer in the manuscript,
whose supplement states the curves in words) and figures/v2-figure-check.txt (page size, smallest font size and embedded fonts of each PDF, and a layout audit).

Text and mathtext are Arial (TrueType, from ~/.local/share/fonts/scmetaintel-article or any installed Arial), with
mathtext set to the custom font set (Arial upright, italic and bold) so that no second family appears; fonts are embedded
as Type 42. The IEEE template's text font is a serif, so the house rule for figures (Arial unless the template embeds a
sans family) applies.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import warnings
import zlib
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker, transforms as mtransforms  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve()
PAPER = HERE.parents[1]
OUT = PAPER / 'figures'
DEFAULT_STUDY = HERE.parents[3] / 'archive' / '2026-11-05-taslp' / 'acoustic'

DW, CW = 7.16, 3.5                      # IEEE page and column widths (in)
FAMILY = 'Arial'                        # figure font family (house rule)
MIN_PT = 8.5                            # smallest admissible text size at printed size
FS, FS_LABEL, FS_PANEL = 8.5, 9.0, 9.0  # ticks/legends/annotations, axis labels and titles, panel letters

# Okabe-Ito colours
INK, MUTED, LIGHT, SHADE = '#222222', '#707070', '#BEBEBE', '#EFEFEF'
BLUE, ORANGE, VERMILLION = '#0072B2', '#E69F00', '#D55E00'
GREEN = '#009E73'
FARC = '#7F7F7F'
BG = 'background'                       # gid of artists that labels may touch (zero lines, shading, frames)
INSIDE: set = set()                     # ids of texts and legends placed inside a plotting area on purpose

# semantic names of the method keys (figure text never shows the keys)
NAME = {'j_cv_ext': 'Proposed', 'sinc_ev': 'Sinc kernel', 'gsmf': 'Shared five-ball variant',
        'mo26_cap5': 'Per-bin five-ball variant', 'two_stage_point': 'Arrival-time point model',
        'two_stage_region': 'Arrival-time region model', 'v1': 'Single-frequency estimator',
        'mo26_prox': 'Sparse adaptation', 'mo26_prox_s2': 'Sparse adaptation',
        'gsmf_prox': 'Group-sparse adaptation', 'ribeiro_official': 'Code of Ribeiro et al.',
        'ind_cv_ext': 'Proposed, each bin alone', 'adk_standin': 'Adaptive kernel'}
SHORT = {'gsmf': 'Shared five balls', 'mo26_cap5': 'Per-bin five balls'}
MINUS = '−'
LEQ = ' ≤ '  # the Arial glyph has no relation spacing of its own


# ----------------------------------------------------------------------------------------------------------- fonts
def _intact(path) -> bool:
    """True when every glyph of a TrueType file decompiles (some installed italic Arial files are truncated)."""
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(str(path))
        glyf = f['glyf']
        for g in f.getGlyphOrder():
            glyf[g]
        return True
    except Exception:
        return False


MATH_ITALIC = {'ok': False}


def install_arial(tmp: Path) -> dict:
    """Register the Arial TrueType files with matplotlib and check that the upright and bold styles resolve to intact
    Arial files. Math variables use Arial Italic when an intact Arial Italic resolves; otherwise they are set upright
    in Arial, so that no other family appears. Returns {style: file name}. `tmp` is unused (kept for the call site)."""
    from pathlib import Path as _P
    used = {}
    home = _P.home() / '.local' / 'share' / 'fonts' / 'scmetaintel-article'
    for fname in ('Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf'):
        if (home / fname).exists() and _intact(home / fname):
            font_manager.fontManager.addfont(str(home / fname))
    for weight, style in (('normal', 'normal'), ('bold', 'normal'), ('normal', 'italic')):
        try:
            got = font_manager.findfont(font_manager.FontProperties(family=FAMILY, weight=weight, style=style),
                                        fallback_to_default=False)
        except ValueError:
            got = ''
        good = bool(got) and 'arial' in _P(got).name.lower() and _intact(got)
        if style == 'italic':
            MATH_ITALIC['ok'] = good and 'italic' in _P(got).name.lower()
            used[f'{weight}/{style}'] = _P(got).name + ('' if MATH_ITALIC['ok'] else ' (not intact: math set upright)')
            continue
        if not good:
            raise SystemExit(f'no intact Arial for {weight}/{style}: {got}')
        used[f'{weight}/{style}'] = _P(got).name
    return used


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [FAMILY], 'font.size': FS, 'mathtext.fontset': 'custom',
        'mathtext.rm': FAMILY, 'mathtext.it': f'{FAMILY}:italic' if MATH_ITALIC['ok'] else FAMILY,
        'mathtext.bf': f'{FAMILY}:bold',
        'mathtext.sf': FAMILY, 'mathtext.tt': FAMILY, 'mathtext.cal': FAMILY, 'mathtext.fallback': None,
        'axes.unicode_minus': True, 'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_LABEL,
        'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
        'xtick.minor.visible': False, 'ytick.minor.visible': False, 'axes.labelpad': 2.0,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'legend.frameon': False, 'legend.borderpad': 0.2, 'legend.borderaxespad': 0.3, 'legend.handlelength': 1.0,
        'legend.handletextpad': 0.4, 'legend.columnspacing': 1.0, 'legend.labelspacing': 0.3,
        'lines.linewidth': 0.8, 'lines.markersize': 3.4, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.dpi': 100, 'savefig.transparent': False, 'patch.linewidth': 0.6,
    })


# ----------------------------------------------------------------------------------------------------------- data
def jload(path: Path):
    with open(path) as fh:
        return json.load(fh)


def ivl(d: dict) -> dict:
    """A registered interval: mean, low, high and (when registered) the class. NaN means undefined."""
    out = {k: float(d[k]) for k in ('mean', 'low', 'high')}
    out['cls'] = d.get('cls')
    return out


def defined(v) -> bool:
    return v is not None and all(math.isfinite(v[k]) for k in ('mean', 'low', 'high'))


# ----------------------------------------------------------------------------------------------------------- layout
def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the lower-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def renderer(fig):
    return fig.canvas.get_renderer()


def text_size_in(fig, s, **kw):
    t = fig.text(0, 0, s, **kw)
    bb = t.get_window_extent(renderer(fig))
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def header(fig, ax, letter, title=None, x_in=None, dy_in=0.055):
    """Panel letter in the header margin above the plotting area, left-aligned at x_in (inches; default: the left
    edge of the panel including its y label), with the short title right after it on the same baseline."""
    r = renderer(fig)
    W, H = fig.get_size_inches()
    if x_in is None:
        x_in = ax.get_tightbbox(r).x0 / fig.dpi
    y = (ax.get_position().y1 * H + dy_in) / H
    lab = fig.text(x_in / W, y, f'({letter})', fontsize=FS_PANEL, fontweight='bold', ha='left', va='baseline')
    if title:
        x1 = lab.get_window_extent(r).x1 / fig.dpi + 0.06
        fig.text(x1 / W, y, title, fontsize=FS_LABEL, ha='left', va='baseline')
    return lab


def header_legend(fig, ax, handles, labels, ncol, dy_in=0.012):
    """Single-row legend in the header margin, right-aligned with the plotting area of ax."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    return fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(pos.x1, pos.y1 + dy_in / H),
                      bbox_transform=fig.transFigure, ncol=ncol, borderaxespad=0.0, borderpad=0.0,
                      handletextpad=0.35, columnspacing=1.1, fontsize=FS)


def inside(artist):
    """Register a text or legend as placed inside a plotting area (it must then clear the data)."""
    INSIDE.add(id(artist))
    if isinstance(artist, Legend):
        for t in artist.get_texts():
            INSIDE.add(id(t))
    return artist


MARGIN_DB, MARGIN_C = 0.1, '#D2D2D2'    # practical margin of the decision rules, drawn as a band about zero


def zero_line(ax, vertical=False):
    """Zero line of a paired-difference axis, with the +/-0.1 dB practical-margin band behind it."""
    band = dict(color=MARGIN_C, lw=0, zorder=0.5, gid=BG)
    (ax.axvspan if vertical else ax.axhspan)(-MARGIN_DB, MARGIN_DB, **band)
    kw = dict(color=INK, lw=0.6, zorder=1, gid=BG)
    return ax.axvline(0.0, **kw) if vertical else ax.axhline(0.0, **kw)


def fixed_ticks(ax, axis, lo, hi, nbins=4, steps=(1, 2, 2.5, 5, 10)):
    vals = ticker.MaxNLocator(nbins=nbins, steps=list(steps)).tick_values(lo, hi)
    vals = [v for v in vals if lo - 1e-9 <= v <= hi + 1e-9]
    (ax.set_yticks if axis == 'y' else ax.set_xticks)(vals)
    return vals


def draw_interval(ax, x, v, color, marker, ms=3.4, lw=0.8, loss_color=None):
    """Interval line and mean marker of one registered comparison: filled marker for a win, open marker for a tie
    and a vermillion cross for a loss (class read from the summary). Returns False for an undefined value."""
    if not defined(v):
        return False
    ax.plot([x, x], [v['low'], v['high']], color=color, lw=lw, solid_capstyle='butt', zorder=2)
    cls = v['cls']
    if cls == 'loss':
        ax.plot([x], [v['mean']], ls='none', marker='x', ms=ms + 0.9, mew=1.1, color=loss_color or VERMILLION, zorder=4)
    elif cls in ('win', 'tie'):
        ax.plot([x], [v['mean']], ls='none', marker=marker, ms=ms, mew=0.8, mec=color,
                mfc=color if cls == 'win' else 'white', zorder=3)
    else:
        raise ValueError(f'unregistered class {cls!r}')
    return True


def class_handles(color, marker='o', far=False):
    h = [Line2D([], [], ls='none', marker=marker, ms=3.6, mew=0.8, mec=color, mfc=color),
         Line2D([], [], ls='none', marker=marker, ms=3.6, mew=0.8, mec=color, mfc='white'),
         Line2D([], [], ls='none', marker='x', ms=4.3, mew=1.1, color=VERMILLION)]
    labels = ['Win', 'Tie', 'Loss']
    if far:
        h.append(Line2D([], [], ls='none', marker='D', ms=3.2, mew=0.8, mec=FARC, mfc=FARC))
        labels.append('Far control')
    return h, labels


def mark_undefined(ax, x):
    """A registered undefined value: a dotted gap line at its position and a small label."""
    ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (1.2, 1.6)), zorder=1, gid=BG)
    tr = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, x=2.5, y=-2.0, units='points')
    return inside(ax.text(x, 1.0, 'undefined', transform=tr, ha='left', va='top', fontsize=FS, color=MUTED))


# ----------------------------------------------------------------------------------------------------------- audit
def _bbox_pt(fig, artist, r):
    bb = artist.get_window_extent(r)
    s = 72.0 / fig.dpi
    return np.array([bb.x0 * s, bb.y0 * s, bb.x1 * s, bb.y1 * s])


def drawn_texts(fig):
    seen, out = set(), []
    for t in fig.findobj(Text):
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        out.append(t)
    return out


def data_marks(fig, ax):
    """Data marks of ax in points: (x, y, radius) samples of markers and lines, and bar rectangles."""
    s = 72.0 / fig.dpi
    pts, rects = [], []
    for ln in ax.get_lines():
        if ln.get_gid() == BG or not ln.get_visible():
            continue
        xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
        xy = ln.get_transform().transform(xy) * s
        mk = ln.get_marker()
        if mk not in (None, 'None', 'none', '', ' '):
            rad = ln.get_markersize() / 2 + ln.get_markeredgewidth() / 2
            pts += [(x, y, rad) for x, y in xy if np.isfinite(x) and np.isfinite(y)]
        if ln.get_linestyle() not in ('None', 'none', '', ' '):
            half = ln.get_linewidth() / 2
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                for u in np.linspace(0.0, 1.0, 41):
                    pts.append((x0 + u * (x1 - x0), y0 + u * (y1 - y0), half))
    for coll in ax.collections:
        if coll.get_gid() == BG or not isinstance(coll, PathCollection):
            continue
        offs = coll.get_offset_transform().transform(coll.get_offsets()) * s
        sizes = coll.get_sizes()
        for i, (x, y) in enumerate(offs):
            area = sizes[i % len(sizes)] if len(sizes) else 36.0
            pts.append((x, y, math.sqrt(area) / 2 + 0.3))
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_gid() != BG and p.get_visible():
            bb = p.get_window_extent()
            rects.append(np.array([bb.x0 * s, bb.y0 * s, bb.x1 * s, bb.y1 * s]))
    return pts, rects


def hits_data(fig, ax, box, tol=0.3):
    b = box + np.array([tol, tol, -tol, -tol])
    pts, rects = data_marks(fig, ax)
    for x, y, rad in pts:
        dx = max(b[0] - x, 0.0, x - b[2])
        dy = max(b[1] - y, 0.0, y - b[3])
        if math.hypot(dx, dy) < rad:
            return True
    for q in rects:
        if min(b[2], q[2]) - max(b[0], q[0]) > 0 and min(b[3], q[3]) - max(b[1], q[1]) > 0:
            return True
    return False


def audit(fig):
    """Layout audit at printed size: text-text overlaps, text off the page, text over a plotting area it does not
    belong inside, and in-panel labels or legends over data marks. Returns (issues, smallest size, sizes used)."""
    fig.canvas.draw()
    r = renderer(fig)
    W, H = fig.get_size_inches() * 72
    texts = drawn_texts(fig)
    boxes = [_bbox_pt(fig, t, r) for t in texts]
    issues = []
    for i in range(len(texts)):
        a = boxes[i]
        if a[0] < -0.2 or a[1] < -0.2 or a[2] > W + 0.2 or a[3] > H + 0.2:
            issues.append(f'off the page: {texts[i].get_text()!r}')
        for j in range(i + 1, len(texts)):
            b = boxes[j]
            if min(a[2], b[2]) - max(a[0], b[0]) > 0.3 and min(a[3], b[3]) - max(a[1], b[1]) > 0.3:
                issues.append(f'text overlap: {texts[i].get_text()!r} / {texts[j].get_text()!r}')
    areas = [(ax, _bbox_pt(fig, ax.patch, r)) for ax in fig.axes if ax.get_visible()]
    for t, a in zip(texts, boxes):
        if id(t) in INSIDE:
            continue
        for ax, q in areas:
            if min(a[2], q[2]) - max(a[0], q[0]) > 0.3 and min(a[3], q[3]) - max(a[1], q[1]) > 0.3:
                issues.append(f'text over a plotting area: {t.get_text()!r}')
    for ax in fig.axes:
        cands = [t for t in ax.texts if id(t) in INSIDE and t.get_visible() and t.get_text().strip()]
        legs = [c for c in ax.get_children() if isinstance(c, Legend) and id(c) in INSIDE]
        for art in cands + legs:
            if hits_data(fig, ax, _bbox_pt(fig, art, r)):
                name = art.get_text() if isinstance(art, Text) else 'legend'
                issues.append(f'in-panel label over data: {name!r}')
    for ax in fig.axes:
        q = _bbox_pt(fig, ax.patch, r)
        for t in ax.texts:
            if id(t) in INSIDE and t.get_visible() and t.get_text().strip():
                b = _bbox_pt(fig, t, r)
                if b[0] < q[0] - 0.3 or b[1] < q[1] - 0.3 or b[2] > q[2] + 0.3 or b[3] > q[3] + 0.3:
                    issues.append(f'in-panel text crosses the axes frame: {t.get_text()!r}')
    sizes = sorted({round(float(t.get_fontsize()), 2) for t in texts})
    return issues, min(sizes), sizes


def clear_top(fig, ax, arts, step=0.05, tries=40):
    """Raise the upper y limit until the given in-panel artists clear the data."""
    for _ in range(tries):
        fig.canvas.draw()
        r = renderer(fig)
        if not any(hits_data(fig, ax, _bbox_pt(fig, a, r)) for a in arts):
            return
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + step * (hi - lo))
    raise RuntimeError('in-panel legend could not be cleared from the data')


# ----------------------------------------------------------------------------------------------------------- figures
def fig_geometry(S: Path, path: Path):
    """(a) Side view of the simulated base condition; (b) plan view of the measured grid and loudspeakers."""
    sys.path.insert(0, str(S / 'common'))
    import fields as F  # noqa: E402
    import sim  # noqa: E402
    base = sim.cells()['base']
    assert base['place'] == 'below'
    shape = tuple(int(n) for n in sim.SHAPE)
    sp = float(F.SPACING)
    ext = np.asarray(sim.EXTENT, float)
    h, d = float(base['height']), float(base['dist'])
    z0 = h + d                                           # lower face of the region
    nodes = F.region_grid(np.array([0.0, 0.0, z0]), shape)
    assert np.isclose(nodes[:, 0].max(), ext[0]) and np.isclose(nodes[:, 2].max(), z0 + ext[2])
    xs_src = ext[0] / 2                                  # centre of the registered horizontal placement range
    n_mic = int(base['M'])
    mic = nodes[np.random.default_rng(0).choice(len(nodes), n_mic, replace=False)]   # one draw of the base M, projected

    prep = np.load(S / 'B3-measured' / 'meshrir-prep' / 'meshrir-prep.npz', allow_pickle=False)
    meta = jload(S / 'B3-measured' / 'meshrir-prep' / 'meshrir-prep.json')
    pos_mic, pos_src = np.asarray(prep['pos_mic'], float), np.asarray(prep['pos_src'], float)
    assert pos_mic.shape == (441, 3) and pos_src.shape == (32, 3) and meta['microphones_present'] == 441
    plane = float(np.mean(pos_mic[:, 2]))
    assert np.allclose(pos_mic[:, 2], plane)
    sel = np.random.default_rng(0).choice(len(pos_mic), 24, replace=False)
    edge = float(np.max(np.abs(pos_mic[:, :2])))         # half side of the grid
    ring = float(np.min(np.max(np.abs(pos_src[:, :2]), axis=1)))  # half side of the loudspeaker square

    fig = plt.figure(figsize=(DW, 2.6))
    # (a) scales: equal aspect, 1.95 in of height
    za, zb = -h - 0.22, z0 + ext[2] + 0.30
    xa, xb = -0.42, ext[0] + 1.55
    ha_in = 1.95
    sa = ha_in / (zb - za)
    a = axes_in(fig, 0.52, 0.42, (xb - xa) * sa, ha_in)
    a.set_xlim(xa, xb)
    a.set_ylim(za, zb)
    a.set_aspect('equal', adjustable='datalim')
    a.axhspan(za, 0.0, color=SHADE, lw=0, zorder=0, gid=BG)
    a.axhline(0.0, color=INK, lw=0.9, zorder=1, gid=BG)
    a.add_patch(Rectangle((0.0, z0), ext[0], ext[2], fill=False, ec=MUTED, lw=0.5, zorder=1, gid=BG))
    side = np.unique(np.round(nodes[:, [0, 2]], 6), axis=0)
    a.scatter(side[:, 0], side[:, 1], s=2.4, color=LIGHT, lw=0, zorder=2)
    a.scatter(mic[:, 0], mic[:, 2], s=10, color=BLUE, lw=0, zorder=3)
    a.plot([xs_src], [h], ls='none', marker='s', ms=5.0, mfc=VERMILLION, mec=VERMILLION, mew=0.8, zorder=4)
    a.plot([xs_src], [-h], ls='none', marker='s', ms=5.0, mfc='white', mec=VERMILLION, mew=0.8, zorder=4)
    xd = ext[0] + 0.14                                   # dimension line just right of the region
    for y0_, y1_ in ((0.0, h), (h, z0)):
        a.annotate('', xy=(xd, y1_), xytext=(xd, y0_), arrowprops=dict(arrowstyle='<->', lw=0.6, color=MUTED,
                                                                       shrinkA=0, shrinkB=0, mutation_scale=6))
    a.plot([xs_src + 0.07, xd + 0.04], [h, h], color=MUTED, lw=0.4, ls=(0, (2, 1.5)), zorder=1, gid=BG)
    a.plot([ext[0], xd + 0.04], [z0, z0], color=MUTED, lw=0.4, ls=(0, (2, 1.5)), zorder=1, gid=BG)
    for txt, y in ((f'{d:g} m', (h + z0) / 2), (f'{h:g} m', h / 2)):
        inside(a.text(xd + 0.06, y, txt, ha='left', va='center', fontsize=FS))
    inside(a.text(xs_src, h - 0.09, 'Loudspeaker', ha='center', va='top', fontsize=FS))
    inside(a.text(xs_src - 0.12, -h, 'Floor image', ha='right', va='center', fontsize=FS))
    inside(a.text(xb - 0.06, 0.04, 'Floor', ha='right', va='bottom', fontsize=FS))
    inside(a.text(0.0, z0 + ext[2] + 0.08, f'{shape[0]} × {shape[1]} × {shape[2]} nodes, {sp:g} m apart',
                  ha='left', va='bottom', fontsize=FS))
    hm = [Line2D([], [], ls='none', marker='o', ms=3.6, mfc=BLUE, mec=BLUE, mew=0),
          Line2D([], [], ls='none', marker='o', ms=1.9, mfc=LIGHT, mec=LIGHT, mew=0)]
    inside(a.legend(hm, [f'Microphones ({n_mic})', 'Targets'], loc='upper left',
                    bbox_to_anchor=(ext[0] + 0.12, z0 + ext[2] - 0.02), bbox_transform=a.transData,
                    borderaxespad=0.0, handletextpad=0.3))
    a.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    a.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    a.set_xlabel('Horizontal position (m)')
    a.set_ylabel('Height above floor (m)')

    # (b) plan view of the measured grid
    lim = ring + 0.17
    hb_in = 1.95
    b = axes_in(fig, 3.62, 0.42, hb_in, hb_in)
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    b.set_aspect('equal', adjustable='datalim')
    others = np.setdiff1d(np.arange(len(pos_mic)), sel)
    b.scatter(pos_mic[others, 0], pos_mic[others, 1], s=1.4, color=LIGHT, lw=0, zorder=2)
    b.scatter(pos_mic[sel, 0], pos_mic[sel, 1], s=10, color=BLUE, lw=0, zorder=3)
    up = pos_src[:, 2] > plane
    b.plot(pos_src[up, 0], pos_src[up, 1], ls='none', marker='^', ms=4.6, mfc=VERMILLION, mec=VERMILLION,
           mew=0.8, zorder=4)
    b.plot(pos_src[~up, 0], pos_src[~up, 1], ls='none', marker='^', ms=4.6, mfc='white', mec=VERMILLION,
           mew=0.8, zorder=4)
    dz = np.unique(np.round(np.abs(pos_src[:, 2] - plane), 6))
    assert len(dz) == 1
    bottom_row = np.sort(pos_src[np.isclose(pos_src[:, 1], -ring), 0])
    gaps = np.diff(bottom_row)
    k = int(np.argmin(np.abs((bottom_row[:-1] + bottom_row[1:]) / 2)))
    xg = float((bottom_row[k] + bottom_row[k + 1]) / 2)
    b.annotate('', xy=(xg, -ring), xytext=(xg, -edge), arrowprops=dict(arrowstyle='<->', lw=0.6, color=MUTED,
                                                                         shrinkA=0, shrinkB=0, mutation_scale=6))
    assert gaps.min() > 0.2
    inside(b.text(xg + 0.05, -(edge + ring) / 2, f'{ring - edge:g} m', ha='left', va='center', fontsize=FS))
    b.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    b.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    b.set_xlabel('$x$ (m)')
    b.set_ylabel('$y$ (m)')
    hl = [Line2D([], [], ls='none', marker='o', ms=3.6, mfc=BLUE, mec=BLUE, mew=0),
          Line2D([], [], ls='none', marker='o', ms=1.6, mfc=LIGHT, mec=LIGHT, mew=0),
          Line2D([], [], ls='none', marker='^', ms=4.6, mfc=VERMILLION, mec=VERMILLION, mew=0.8),
          Line2D([], [], ls='none', marker='^', ms=4.6, mfc='white', mec=VERMILLION, mew=0.8)]
    b.legend(hl, ['Microphones (24)', 'Targets', f'Loudspeaker {dz[0]:g} m\nabove the grid plane',
                  f'Loudspeaker {dz[0]:g} m\nbelow the grid plane'],
             loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0.0, handletextpad=0.4, labelspacing=0.6)
    fig.canvas.draw()
    header(fig, a, 'a', 'Simulation, side view')
    header(fig, b, 'b', 'Measured grid, plan view')
    return fig


SIM_FAMILIES = (('base', 'Base'), ('A1-', 'Placement'), ('A2-', 'Freq. × mics'), ('A3-', 'Reflection'),
                ('A4-', 'Directivity'), ('A5-', 'Coherent pair'), ('A6-', 'Second source'), ('A7-', 'Mic. error'),
                ('A8-', 'Speed error'), ('A9-', 'SNR'), ('FF-', 'Factorial'))
FAR_CELL = 'A1-beside1.2'
WRAP = {'Coherent pair': 'Coherent\npair', 'Second source': 'Second\nsource', 'Mic. error': 'Mic.\nerror',
        'Speed error': 'Speed\nerror'}


def sim_groups(S: Path, cells: dict):
    """Registered simulation cells in registration order, grouped by the factor varied; the far control last."""
    sys.path.insert(0, str(S / 'common'))
    import sim  # noqa: E402
    order = list(sim.cells())
    assert set(order) == set(cells), 'summary cells differ from the registered cells'
    groups = []
    for prefix, name in SIM_FAMILIES:
        members = [c for c in order if (c == prefix if prefix == 'base' else c.startswith(prefix)) and c != FAR_CELL]
        groups.append((name, members))
    groups.append(('Far', [FAR_CELL]))
    assert sum(len(m) for _, m in groups) == len(order) == 64
    return groups


def slot_layout(fig, groups, labels, width_in, pad_in=0.045, margin=0.4):
    """Label-aware x layout: each group gets the wider of its cells and its label, so labels sit in one row under
    their group. Returns cell positions (cell units), slots and the axis limits."""
    widths = [text_size_in(fig, lab, fontsize=FS)[0] for lab in labels]
    n = [len(m) for _, m in groups]

    def total(u):
        return sum(max(k * u, w + pad_in) for k, w in zip(n, widths)) + 2 * margin * u
    lo, hi = 0.01, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) <= width_in else (lo, mid)
    u = lo
    x, pos, slots = 0.0, {}, []
    for (name, members), k, w in zip(groups, n, widths):
        cells_w = max(k, (w + pad_in) / u)
        first = x + (cells_w - k) / 2 + 0.5
        for i, c in enumerate(members):
            pos[c] = first + i
        slots.append((x, x + cells_w, first, first + k - 1))
        x += cells_w
    return pos, slots, (-margin, x + margin), u


def fig_simulation(S: Path, path: Path):
    """Per-condition paired differences (Proposed minus comparator) with Bonferroni intervals, TB-S2: the best generic
    estimator and the shared five-ball variant (the per-bin variant, which lost every condition, is in the supplement's
    condition table)."""
    summ = jload(S / 'studies' / 'TB-S2' / 'primary' / 'summary.json')
    cells = summ['cells']
    groups = sim_groups(S, cells)
    labels = [WRAP.get(g, g) for g, _ in groups]
    # conditions with a driver inside the region (post hoc geometry, posthoc/review3/saved.json)
    interior = set(jload(S / 'posthoc' / 'review3' / 'saved.json')['A1_driver_geometry']['interior_conditions'])
    fig = plt.figure(figsize=(DW, 2.90))
    left, width, hgt = 0.47, DW - 0.47 - 0.04, 0.87
    bottoms = (1.76, 0.61)
    pos, slots, xlim, u = slot_layout(fig, groups, labels, width)
    if u < 0.072:
        raise RuntimeError(f'cell spacing {u:.3f} in is too tight')
    panels = [('Best generic', lambda c: f"j_cv_ext-vs-{cells[c]['best_generic']}"),
              (NAME['gsmf'], lambda c: 'j_cv_ext-vs-gsmf')]
    axes = []
    for p, ((title, key), bot) in enumerate(zip(panels, bottoms)):
        ax = axes_in(fig, left, bot, width, hgt)
        axes.append(ax)
        for gi, (s0, s1, _, _) in enumerate(slots):
            if gi % 2 == 1:
                ax.axvspan(s0, s1, color=SHADE, lw=0, zorder=0, gid=BG)
        zero_line(ax)
        lows, highs = [0.0], [0.0]
        for name, members in groups:
            for c in members:
                v = ivl(cells[c]['comparisons'][key(c)])
                far = c == FAR_CELL
                mk, ms = ('D', 3.1) if far else (('^', 3.7) if c in interior else ('o', 3.3))
                if draw_interval(ax, pos[c], v, FARC if far else BLUE, mk, ms=ms):
                    lows.append(v['low'])
                    highs.append(v['high'])
                else:
                    mark_undefined(ax, pos[c])
        span = max(highs) - min(lows)
        ylo, yhi = min(lows) - 0.05 * span, max(highs) + 0.07 * span
        ax.set_xlim(*xlim)
        ax.set_ylim(ylo, yhi)
        fixed_ticks(ax, 'y', ylo, yhi, nbins=4)
        ax.set_xticks([])
        ax.set_ylabel('Difference (dB)')
    bottom = axes[-1]
    blend = bottom.get_xaxis_transform()
    for (name, _), lab, (s0, s1, c0, c1) in zip(groups, labels, slots):
        tr = mtransforms.offset_copy(blend, fig=fig, y=-3.0, units='points')
        bottom.plot([c0 - 0.42, c1 + 0.42], [0, 0], transform=tr, color=MUTED, lw=0.7, clip_on=False, gid=BG,
                    solid_capstyle='butt')
        bottom.annotate(lab, xy=((c0 + c1) / 2, 0), xycoords=blend, xytext=(0, -5.5), textcoords='offset points',
                        ha='center', va='top', fontsize=FS, linespacing=1.0, annotation_clip=False)
    bottom.annotate('Simulated condition', xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -27.5),
                    textcoords='offset points', ha='center', va='top', fontsize=FS_LABEL, annotation_clip=False)
    fig.canvas.draw()
    r = renderer(fig)
    x_in = min(ax.yaxis.get_tightbbox(r).x0 for ax in axes) / fig.dpi
    for ax, letter, (title, _) in zip(axes, 'ab', panels):
        header(fig, ax, letter, f'Proposed {MINUS} {title}', x_in=x_in)
    hh, ll = class_handles(BLUE, far=True)
    hh.insert(3, Line2D([], [], ls='none', marker='^', ms=4.0, mew=0.8, mec=BLUE, mfc=BLUE)); ll.insert(3, 'Driver inside')
    header_legend(fig, axes[0], hh, ll, ncol=5)
    return fig


def measured_x():
    bands, Ms = (125, 250, 500, 1000), (24, 48, 96, 192)
    pos = {f'{f}Hz-M{M}': b * 5 + m for b, f in enumerate(bands) for m, M in enumerate(Ms)}
    return bands, Ms, pos


def fig_measured(S: Path, path: Path, keys=('a', 'd')):
    """Measured per-condition paired differences with Bonferroni intervals: TB-M2 and its registered follow-ups, one
    row of two panels. keys picks the comparisons: 'a' best generic (corrected sparse Bayesian beside it), 'd'
    arrival-time models (the main-text figure), 'b'/'c' the shared and per-bin five-ball variants with the cubature
    and the exact series (the supplementary figure). Panels are lettered (a), (b) in the order given."""
    m2 = jload(S / 'studies' / 'TB-M2' / 'primary' / 'summary.json')['cells']
    sb = jload(S / 'studies' / 'TB-M2-SBLFIX' / 'primary' / 'summary.json')['M1_corrected']['per_cell']
    ex = jload(S / 'studies' / 'TB-M2-EXACT' / 'primary' / 'summary.json')['cells_table']
    hi = jload(S / 'studies' / 'TB-M2-EXACT-HI' / 'primary' / 'summary.json')['part_b']['cells_table']
    bands, Ms, pos = measured_x()
    assert set(m2) == set(pos) == set(sb) == set(hi), 'measured cells differ between summaries'
    low_M = {c for c in pos if int(c.split('-M')[1]) <= 48}
    assert set(ex) == low_M, 'exact cap-5 rerun cells are not the M <= 48 cells'
    c = list(pos)
    series = {
        'a': [(lambda k: ivl(m2[k]['comparisons'][f"j_cv_ext-vs-{m2[k]['best_generic']}"]), 'As registered'),
              (lambda k: ivl(sb[k]), 'Corrected sparse Bayesian')],
        'b': [(lambda k: ivl(m2[k]['comparisons']['j_cv_ext-vs-gsmf']), 'Cubature'),
              (lambda k: ivl(hi[k]['comparisons']['j_cv_ext-vs-gsmf_exact']), 'Exact series')],
        'c': [(lambda k: ivl(m2[k]['comparisons']['j_cv_ext-vs-mo26_cap5']), 'Cubature'),
              (lambda k: ivl(ex[k]['comparisons']['j_cv_ext-vs-mo26_cap5_exact']) if k in ex else None,
               f'Exact series, $M${LEQ}48')],
        'd': [(lambda k: ivl(m2[k]['comparisons']['j_cv_ext-vs-two_stage_point']), 'Point model'),
              (lambda k: ivl(m2[k]['comparisons']['j_cv_ext-vs-two_stage_region']), 'Region model')],
    }
    titles = {'a': 'Best generic', 'b': SHORT['gsmf'], 'c': SHORT['mo26_cap5'], 'd': 'Arrival-time models'}
    assert len(keys) == 2 and set(keys) <= set(series)
    fig = plt.figure(figsize=(DW, 2.22))
    w, hgt = 3.03, 1.02
    lefts = (0.48, 0.48 + w + 0.55)
    styles = ((BLUE, 'o', -0.2), (ORANGE, 's', 0.2))
    axes = {}
    for i, letter in enumerate(keys):
        ax = axes_in(fig, lefts[i], 0.70, w, hgt)
        axes[letter] = ax
        for b in (1, 3):
            ax.axvspan(b * 5 - 0.5, b * 5 + 3.5, color=SHADE, lw=0, zorder=0, gid=BG)
        zero_line(ax)
        lows, highs, inner = [0.0], [0.0], []
        for (get, _), (col, mk, off) in zip(series[letter], styles):
            for k in c:
                v = get(k)
                if v is None:
                    continue                               # not part of that rerun
                if draw_interval(ax, pos[k] + off, v, col, mk, ms=3.3):
                    lows.append(v['low'])
                    highs.append(v['high'])
                else:
                    inner.append(mark_undefined(ax, pos[k] + off))
        span = max(highs) - min(lows)
        ax.set_xlim(-0.65, 18.65)
        ax.set_ylim(min(lows) - 0.05 * span, max(highs) + 0.08 * span)
        hs = [Line2D([], [], ls='none', marker=mk, ms=3.6, mew=0.8, mec=col, mfc=col) for col, mk, _ in styles]
        leg = inside(ax.legend(hs, [lab for _, lab in series[letter]], loc='lower left', ncol=1,
                               borderaxespad=0.25, handletextpad=0.3, labelspacing=0.25))
        inner.append(leg)
        fig.canvas.draw()
        if any(hits_data(fig, ax, _bbox_pt(fig, a, renderer(fig))) for a in inner):
            leg.remove()                                   # the corner is occupied: fall back to the top, cleared
            leg = inside(ax.legend(hs, [lab for _, lab in series[letter]], loc='upper right', ncol=2,
                                   borderaxespad=0.15, handletextpad=0.3, columnspacing=1.0))
            inner[-1] = leg
            clear_top(fig, ax, inner)
        ylo, yhi = ax.get_ylim()
        fixed_ticks(ax, 'y', ylo, yhi, nbins=4)
        ax.set_ylabel('Difference (dB)')
        ax.set_xticks([pos[k] for k in c])
        blend = ax.get_xaxis_transform()
        ax.set_xticklabels([k.split('-M')[1] for k in c], rotation=90, fontsize=FS)
        ax.annotate('Microphones $M$ within each band', xy=(0.5, 0), xycoords='axes fraction',
                    xytext=(0, -33.5), textcoords='offset points', ha='center', va='top', fontsize=FS_LABEL,
                    annotation_clip=False)
        for bi, f in enumerate(bands):
            ax.annotate(f'{f} Hz' if f < 1000 else f'{f // 1000} kHz', xy=(bi * 5 + 1.5, 0), xycoords=blend,
                        xytext=(0, -20.0), textcoords='offset points', ha='center', va='top', fontsize=FS,
                        annotation_clip=False)
    fig.canvas.draw()
    r = renderer(fig)
    for k, letter in zip(keys, 'ab'):
        header(fig, axes[k], letter, f'Proposed {MINUS} {titles[k]}', x_in=axes[k].yaxis.get_tightbbox(r).x0 / fig.dpi)
    hh, ll = class_handles(INK)
    fig.legend(hh, ll, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.0, borderpad=0.0,
               handletextpad=0.35, columnspacing=1.2, fontsize=FS)
    return fig


def fig_absolute(S: Path, path: Path):
    """Absolute error by condition on both measured grids, read from the registered summaries: the mean over each
    condition's problems of the error in dB (10 log10 of the mean over six bins of the target-normalized squared
    pressure error) for the proposed estimator, the best generic estimator of the condition (as registered), the sinc
    kernel and the arrival-time point model."""
    m2 = jload(S / 'studies' / 'TB-M2' / 'primary' / 'summary.json')['cells']
    m3 = jload(S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'summary.json')
    series = (('j_cv_ext', 'Proposed', INK, 'o'), ('bg', 'Best generic', BLUE, 's'),
              ('sinc_ev', NAME['sinc_ev'], ORANGE, '^'), ('two_stage_point', NAME['two_stage_point'], GREEN, 'D'))
    studies = (('a', 'Room grid', m2, (125, 250, 500, 1000), 2.26),
               ('b', 'Anechoic, 0.74 m', m3['A1']['cells_table'], (250, 500, 1000), 1.72),
               ('c', 'Anechoic, 1.47 m, reflector', m3['R2']['cells_table'], (250, 500, 1000), 1.72))
    Ms = (24, 48, 96, 192)
    H = 2.12
    fig = plt.figure(figsize=(DW, H))
    bot, hgt, gap, x = 0.62, 0.97, 0.47, 0.47
    axes = {}
    for letter, title, T, bands, w in studies:
        assert x + w <= DW - 0.03, 'absolute-error figure wider than the page'
        ax = axes_in(fig, x, bot, w, hgt)
        x += w + gap
        axes[letter] = (ax, title)
        pos = {f'{f}Hz-M{M}': b * 5 + m for b, f in enumerate(bands) for m, M in enumerate(Ms)}
        assert set(T) == set(pos), 'conditions differ from the registered design'
        for b in range(1, len(bands), 2):
            ax.axvspan(b * 5 - 0.5, b * 5 + 3.5, color=SHADE, lw=0, zorder=0, gid=BG)
        vals = []
        for key, _, col, mk in series:
            for f in bands:
                ks = [f'{f}Hz-M{M}' for M in Ms]
                ys = [float(T[k]['mean_db'][T[k]['best_generic'] if key == 'bg' else key]) for k in ks]
                assert all(math.isfinite(y) for y in ys), f'non-finite mean error: {key} {f} Hz'
                vals += ys
                ax.plot([pos[k] for k in ks], ys, color=col, lw=0.8, marker=mk, ms=3.0, mfc=col, mec=col, zorder=3)
        lo, hi = min(vals), max(vals)
        span = hi - lo
        ax.set_xlim(-0.65, len(bands) * 5 - 1.35)
        ax.set_ylim(lo - 0.06 * span, hi + 0.06 * span)
        fixed_ticks(ax, 'y', *ax.get_ylim(), nbins=4)
        ax.set_ylabel('Error (dB)')
        ax.set_xticks([pos[k] for k in pos])
        ax.set_xticklabels([k.split('-M')[1] for k in pos], rotation=90, fontsize=FS)
        blend = ax.get_xaxis_transform()
        for bi, f in enumerate(bands):
            ax.annotate(f'{f} Hz' if f < 1000 else f'{f // 1000} kHz', xy=(bi * 5 + 1.5, 0), xycoords=blend,
                        xytext=(0, -20.0), textcoords='offset points', ha='center', va='top', fontsize=FS,
                        annotation_clip=False)
    fig.text(0.5, 0.02, 'Microphones $M$ within each band', ha='center', va='bottom', fontsize=FS_LABEL)
    fig.canvas.draw()
    r = renderer(fig)
    for letter, (ax, title) in axes.items():
        header(fig, ax, letter, title, x_in=ax.yaxis.get_tightbbox(r).x0 / fig.dpi)
    hs = [Line2D([], [], color=col, lw=0.8, marker=mk, ms=3.0, mfc=col, mec=col) for _, _, col, mk in series]
    fig.legend(hs, [lab for _, lab, _, _ in series], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4,
               borderaxespad=0.0, borderpad=0.0, handletextpad=0.35, columnspacing=1.4, fontsize=FS)
    return fig


def fig_miracle(S: Path, path: Path):
    """Anechoic grid, three configurations (near, far, far with a reflector): per-condition paired differences with
    Bonferroni intervals against the best generic estimator, the gridded multi-frequency sparse model (better variant of
    the condition) and the arrival-time point model."""
    sm = jload(S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'summary.json')
    m4 = jload(S / 'studies' / 'TB-M4-MIRACLE-A2' / 'primary' / 'summary.json')
    mf = jload(S / 'posthoc' / 'upgrade-mfsbl' / 'summary-seen.json')
    bands, Ms = (250, 500, 1000), (24, 48, 96, 192)
    pos = {f'{f}Hz-M{M}': b * 5 + m for b, f in enumerate(bands) for m, M in enumerate(Ms)}
    c = list(pos)
    styles = ((BLUE, 'o', -0.27), (ORANGE, 's', 0.0), (INK, '^', 0.27))
    labels = ('Best generic', 'Gridded multi-frequency sparse', NAME['two_stage_point'])
    table = dict(A1=sm['A1']['cells_table'], A2=m4['A2']['cells_table'], R2=sm['R2']['cells_table'])
    grid = dict(A1=mf['sets']['A1']['better_variant']['by_cell'], A2=m4['A2']['mfsbl']['better_variant']['by_cell'],
                R2=mf['sets']['R2']['better_variant']['by_cell'])
    fig = plt.figure(figsize=(DW, 2.22))
    w, hgt = 1.84, 1.02
    lefts = tuple(0.48 + i * (w + 0.52) for i in range(3))
    axes = {}
    for i, (letter, sc) in enumerate((('a', 'A1'), ('b', 'A2'), ('c', 'R2'))):
        T, G = table[sc], grid[sc]
        assert set(T) == set(pos) == set(G), 'anechoic conditions differ from the design'
        series = (lambda k: ivl(T[k]['comparisons'][f"j_cv_ext-vs-{T[k]['best_generic']}"]),
                  lambda k: ivl(G[k]),
                  lambda k: ivl(T[k]['comparisons']['j_cv_ext-vs-two_stage_point']))
        ax = axes_in(fig, lefts[i], 0.70, w, hgt)
        axes[letter] = ax
        ax.axvspan(4.5, 8.5, color=SHADE, lw=0, zorder=0, gid=BG)
        zero_line(ax)
        lows, highs = [0.0], [0.0]
        for get, (col, mk, off) in zip(series, styles):
            for k in c:
                v = get(k)
                assert draw_interval(ax, pos[k] + off, v, col, mk, ms=2.6, lw=0.7, loss_color=col), 'undefined interval'
                lows.append(v['low']); highs.append(v['high'])
        span = max(highs) - min(lows)
        ax.set_xlim(-0.75, 13.75)
        ax.set_ylim(min(lows) - 0.05 * span, max(highs) + 0.08 * span)
        ylo, yhi = ax.get_ylim()
        fixed_ticks(ax, 'y', ylo, yhi, nbins=4)
        if i == 0:
            ax.set_ylabel('Difference (dB)')
        ax.set_xticks([pos[k] for k in c])
        blend = ax.get_xaxis_transform()
        ax.set_xticklabels([k.split('-M')[1] for k in c], rotation=90, fontsize=FS)
        ax.annotate('Microphones $M$ in each band', xy=(0.5, 0), xycoords='axes fraction',
                    xytext=(0, -33.5), textcoords='offset points', ha='center', va='top', fontsize=FS_LABEL,
                    annotation_clip=False)
        for bi, f in enumerate(bands):
            ax.annotate(f'{f} Hz' if f < 1000 else f'{f // 1000} kHz', xy=(bi * 5 + 1.5, 0), xycoords=blend,
                        xytext=(0, -20.0), textcoords='offset points', ha='center', va='top', fontsize=FS,
                        annotation_clip=False)
    fig.canvas.draw()
    r = renderer(fig)
    for k, t in (('a', 'Near, 0.74 m'), ('b', 'Far, 1.47 m'), ('c', 'Far, with a reflector')):
        header(fig, axes[k], k, t, x_in=axes[k].yaxis.get_tightbbox(r).x0 / fig.dpi)
    hs = [Line2D([], [], ls='none', marker=mk, ms=3.6, mew=0.8, mec=col, mfc=col) for col, mk, _ in styles]
    hh, ll = class_handles(INK)
    hh[2] = Line2D([], [], ls='none', marker='x', ms=4.3, mew=1.1, color=INK)   # here a loss keeps its series colour
    spacer = Line2D([], [], ls='none')
    fig.legend(hs + [spacer] + hh, list(labels) + [''] + ll, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=7,
               borderaxespad=0.0, borderpad=0.0, handletextpad=0.35, columnspacing=1.0, fontsize=FS)
    return fig


def fig_competitors(S: Path, path: Path):
    """Pooled paired differences with 95% intervals over the independent units, as a forest plot: (a) measured,
    (b) simulated, with the post hoc comparisons of the cross-validated order with fixed orders."""
    m2 = jload(S / 'studies' / 'TB-M2' / 'primary' / 'summary.json')['rules']
    sbl = jload(S / 'studies' / 'TB-M2-SBLFIX' / 'primary' / 'summary.json')
    exa = jload(S / 'studies' / 'TB-M2-EXACT' / 'primary' / 'summary.json')['pooled']
    exh = jload(S / 'studies' / 'TB-M2-EXACT-HI' / 'primary' / 'summary.json')['part_b']['headline']
    rib = jload(S / 'studies' / 'TB-M2-RIB' / 'primary' / 'summary.json')['R1']
    per = jload(S / 'studies' / 'TB-M2-RIB' / 'descriptive-perbin.json')['ind_cv_ext-vs-ribeiro_official']
    s2 = jload(S / 'studies' / 'TB-S2' / 'primary' / 'summary.json')['rules']
    fo = jload(S / 'posthoc' / 'fixed-order' / 'fixed-order.json')['studies']
    cap5_reg = ivl(m2['M2']['per_competitor']['mo26_cap5']['pooled'])
    assert not defined(cap5_reg), 'the registered pooled per-bin five-ball comparison is expected to be undefined'
    rib_label = 'Published adaptive-kernel code, one bin'
    fx = lambda st, K: ivl(fo[st]['cv_vs_fixed'][f'K{K}']['pooled'])
    A = [('h', 'Proposed minus'),
         ('r', 'Per-problem oracle', ivl(m2['descriptive']['pooled_vs_best_generic_task']), BLUE),
         ('r', 'Same, corrected sparse Bayesian', ivl(sbl['pooled_vs_task_best_generic']), BLUE),
         ('r', NAME['v1'], ivl(m2['descriptive']['pooled_vs_v1']), BLUE),
         ('r', f"{SHORT['gsmf']}, cubature", ivl(m2['M2']['per_competitor']['gsmf']['pooled']), BLUE),
         ('r', f"{SHORT['gsmf']}, exact kernel", ivl(exh['exact']), BLUE),
         ('r', f"{SHORT['mo26_cap5']}, cubature", cap5_reg, BLUE),
         ('r', f"{SHORT['mo26_cap5']}, exact, $M${LEQ}48", ivl(exa['j_cv_ext-vs-mo26_cap5_exact']), BLUE),
         ('r', rib_label, ivl(rib['pooled']), BLUE),
         ('h', 'Post hoc: proposed with each\nbin fitted alone, minus'),
         ('r', rib_label, ivl(per['pooled']), BLUE, True)]
    B = [('h', 'Proposed minus'),
         ('r', SHORT['gsmf'], ivl(s2['S2']['per_competitor']['gsmf']['pooled']), ORANGE),
         ('r', SHORT['mo26_cap5'], ivl(s2['S2']['per_competitor']['mo26_cap5']['pooled']), ORANGE),
         ('r', NAME['v1'], ivl(s2['S7']['pooled']), ORANGE),
         ('h', 'Post hoc: cross-validated order\nminus a fixed one, simulation'),
         ('r', 'One point', fx('TB-S2', 1), ORANGE, True),
         ('r', 'Five points', fx('TB-S2', 5), ORANGE, True),
         ('h', 'The same, measured'),
         ('r', 'One point', fx('TB-M2', 1), BLUE, True),
         ('r', 'Two points', fx('TB-M2', 2), BLUE, True),
         ('r', 'Five points', fx('TB-M2', 5), BLUE, True)]
    H = 2.28
    fig = plt.figure(figsize=(DW, H))
    indent = 0.09

    def col_width(entries):
        ws = [max(text_size_in(fig, ln, fontsize=FS, fontweight='bold' if e[0] == 'h' else 'normal')[0]
                  for ln in e[1].split('\n')) + (0 if e[0] == 'h' else indent) for e in entries]
        return max(ws) + 0.1
    wa, wb = col_width(A), col_width(B)
    x0, gap, right = 0.03, 0.16, 0.10
    data_w = (DW - x0 - right - wa - gap - wb) / 2
    if data_w < 1.0:
        raise RuntimeError(f'forest data width {data_w:.2f} in is too narrow')
    bot, hgt = 0.42, H - 0.42 - 0.31
    ax_a = axes_in(fig, x0 + wa, bot, data_w, hgt)
    ax_b = axes_in(fig, x0 + wa + data_w + gap + wb, bot, data_w, hgt)
    xl = (-2.8, 0.6)
    for ax, entries, lab_x in ((ax_a, A, x0), (ax_b, B, x0 + wa + data_w + gap)):
        lines = [e[1].count('\n') + 1 for e in entries]
        tot = max(sum(e[1].count('\n') + 1 for e in A), sum(e[1].count('\n') + 1 for e in B))
        ys, acc = [], 0.0
        for n in lines:
            ys.append(acc + n / 2)
            acc += n
        ax.set_ylim(tot + 0.25, -0.25)
        ax.set_xlim(*xl)
        zero_line(ax, vertical=True)
        tr = mtransforms.blended_transform_factory(fig.dpi_scale_trans, ax.transData)
        for e, y in zip(entries, ys):
            if e[0] == 'h':
                ax.text(lab_x, y, e[1], transform=tr, ha='left', va='center', fontsize=FS, fontweight='bold',
                        linespacing=1.0)
                continue
            _, label, v, col = e[:4]
            post = len(e) > 4 and e[4]
            ax.text(lab_x + indent, y, label, transform=tr, ha='left', va='center', fontsize=FS, linespacing=1.0)
            if defined(v):
                ax.plot([v['low'], v['high']], [y, y], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
                ax.plot([v['mean']], [y], ls='none', marker='s', ms=3.6, mfc='white' if post else col, mec=col, mew=0.8,
                        zorder=3)
                assert xl[0] < v['low'] and v['high'] < xl[1], f'interval outside the axis: {label}'
            else:
                tu = mtransforms.offset_copy(ax.transData, fig=fig, x=-3.0, units='points')
                inside(ax.text(0.0, y, 'undefined', transform=tu, ha='right', va='center', fontsize=FS,
                               color=MUTED))
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        fixed_ticks(ax, 'x', *xl, nbins=5)
        ax.set_xlabel('Difference (dB)')
    fig.canvas.draw()
    header(fig, ax_a, 'a', 'Measured', x_in=x0)
    header(fig, ax_b, 'b', 'Simulated, and post hoc order', x_in=x0 + wa + data_w + gap)
    return fig


def fig_order(S: Path, path: Path):
    """Order selection, one row of three panels: (a, b) post hoc pooled paired difference, cross-validated order minus
    a fixed order of one to five points, for the point models before the extension (read from the same greedy fits),
    in simulation and on the room grid; (c) fraction of problems at each cross-validated order in all three studies."""
    fo = jload(S / 'posthoc' / 'fixed-order' / 'fixed-order.json')['studies']
    s2 = jload(S / 'studies' / 'TB-S2' / 'primary' / 'summary.json')['cells']
    m2 = jload(S / 'studies' / 'TB-M2' / 'primary' / 'summary.json')['cells']
    m3 = jload(S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'summary.json')
    sim_cells = [c for c in s2 if c != FAR_CELL]
    assert len(sim_cells) == 63 and len(m2) == 16
    hs = np.sum([s2[c]['K_cv_hist'] for c in sim_cells], axis=0).astype(float)
    hm = np.sum([m2[c]['K_cv_hist'] for c in m2], axis=0).astype(float)
    h3 = (np.array(m3['A1']['K_cv_hist']) + np.array(m3['R2']['K_cv_hist'])).astype(float)
    assert hs.shape == hm.shape == h3.shape == (6,) and h3.sum() == 1536
    fs, fm, f3 = hs / hs.sum(), hm / hm.sum(), h3 / h3.sum()
    H = 1.95
    fig = plt.figure(figsize=(DW, H))
    bot, hgt, wf, wh = 0.42, 1.25, 1.62, 2.20
    la = 0.50
    lb = la + wf + 0.56
    lc = lb + wf + 0.58
    assert lc + wh <= DW - 0.05, 'order figure wider than the page'
    Ks = np.arange(1, 6)
    axes = {}
    for letter, st, col, left in (('a', 'TB-S2', ORANGE, la), ('b', 'TB-M2', BLUE, lb)):
        assert fo[st]['cv_vs_fixed'].keys() >= {f'K{K}' for K in Ks}
        ax = axes_in(fig, left, bot, wf, hgt)
        axes[letter] = ax
        zero_line(ax)
        vals = [ivl(fo[st]['cv_vs_fixed'][f'K{K}']['pooled']) for K in Ks]
        for K, v in zip(Ks, vals):
            assert defined(v), f'undefined fixed-order comparison {st} K{K}'
            ax.plot([K, K], [v['low'], v['high']], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
            ax.plot([K], [v['mean']], ls='none', marker='s', ms=3.6, mfc=col, mec=col, mew=0.8, zorder=3)
        lo = min(0.0, min(v['low'] for v in vals))
        hi = max(0.0, max(v['high'] for v in vals))
        span = hi - lo
        ax.set_ylim(lo - 0.06 * span, hi + 0.10 * span)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(Ks)
        fixed_ticks(ax, 'y', *ax.get_ylim(), nbins=4)
        ax.set_xlabel('Fixed order $K$')
        ax.set_ylabel('Difference (dB)')
    ax = axes_in(fig, lc, bot, wh, hgt)
    axes['c'] = ax
    K = np.arange(6)
    ax.bar(K - 0.27, fs, width=0.26, color=ORANGE, lw=0, zorder=2, label='Simulation')
    ax.bar(K, fm, width=0.26, color=BLUE, lw=0, zorder=2, label='Room grid')
    ax.bar(K + 0.27, f3, width=0.26, color=INK, lw=0, zorder=2, label='Anechoic grid')
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(0.0, max(fs.max(), fm.max(), f3.max()) * 1.08)
    leg = inside(ax.legend(loc='upper left', borderaxespad=0.2, handlelength=0.9))
    clear_top(fig, ax, [leg])
    ax.set_xticks(K)
    fixed_ticks(ax, 'y', 0.0, ax.get_ylim()[1], nbins=4)
    ax.set_xlabel('Selected order $K$')
    ax.set_ylabel('Fraction of problems')
    fig.canvas.draw()
    r = renderer(fig)
    for letter, title in (('a', 'Simulation'), ('b', 'Room grid'), ('c', 'Selected order')):
        header(fig, axes[letter], letter, title, x_in=axes[letter].yaxis.get_tightbbox(r).x0 / fig.dpi)
    return fig, (hs, hm, h3)


def log_ticks(ax, axis, vals):
    loc = ticker.FixedLocator(vals)
    fmt = ticker.FixedFormatter([f'{v:g}' for v in vals])
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(loc); a.set_major_formatter(fmt); a.set_minor_locator(ticker.NullLocator())


def fig_identifiability(S: Path, path: Path):
    """The Cramer-Rao bound on synthetic ball geometries (post hoc theory curves), one panel."""
    th = jload(S / 'posthoc' / 'crb' / 'theory.json')
    H = 1.78
    fig = plt.figure(figsize=(CW, H))
    w = CW - 0.52 - 0.08
    ax = axes_in(fig, 0.52, 0.42, w, H - 0.42 - 0.24)
    kas = ((0.5, ORANGE), (2.0, BLUE), (8.0, VERMILLION))
    for ka, col in kas:
        c = sorted((r for r in th['curves'] if r['M'] == 48 and r['ka'] == ka), key=lambda r: r['d_over_a'])
        q = [r['d_over_a'] for r in c]
        ax.plot(q, [r['one_bin'] for r in c], color=col, lw=0.9, ls=(0, (3.0, 1.6)), zorder=3)
        ax.plot(q, [r['six_bins'] for r in c], color=col, lw=0.9, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (1.2, 1.6)), zorder=1, gid=BG)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1.05, 10.5); ax.set_ylim(0.02, 40.0)
    log_ticks(ax, 'x', [1.1, 1.5, 2, 3, 5, 10]); log_ticks(ax, 'y', [0.1, 1, 10])
    ax.set_xlabel('Source distance from the centre / radius')
    ax.set_ylabel('Bound / radius')
    style = [Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3.0, 1.6))), Line2D([], [], color=MUTED, lw=0.9)]
    leg = inside(ax.legend(style, ['One bin', 'Six bins, shared'], loc='upper left', borderaxespad=0.2,
                           handlelength=1.8))
    clear_top(fig, ax, [leg])
    header_legend(fig, ax, [Line2D([], [], color=col, lw=0.9) for _, col in kas],
                  [f'$ka={ka:g}$' for ka, _ in kas], ncol=3)
    return fig


# ----------------------------------------------------------------------------------------------------------- checks
def pdf_text_sizes(path: Path) -> list:
    """Font sizes of every text operator (Tf) in the content streams of a PDF."""
    data = path.read_bytes()
    sizes = set()
    for m in re.finditer(rb'stream\r?\n', data):
        try:
            d = zlib.decompressobj().decompress(data[m.end():])
        except zlib.error:
            continue
        if b'BT' in d and b'Tf' in d:
            sizes.update(float(x) for x in re.findall(rb'/F\d+\s+([0-9.]+)\s+Tf', d))
    return sorted(sizes)


def pdf_fonts(path: Path):
    res = subprocess.run(['pdffonts', str(path)], capture_output=True, text=True, check=True)
    lines = res.stdout.splitlines()
    spans = [(m.start(), m.end()) for m in re.finditer(r'-+', lines[1])]
    rows = []
    for ln in lines[2:]:
        if ln.strip():
            f = [ln[s:(spans[i + 1][0] if i + 1 < len(spans) else len(ln))].strip() for i, (s, _) in enumerate(spans)]
            rows.append(dict(name=f[0], type=f[1], encoding=f[2], emb=f[3], sub=f[4], uni=f[5]))
    return rows, res.stderr.strip()


def pdf_page_in(path: Path):
    info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True).stdout
    m = re.search(r'Page size:\s+([0-9.]+) x ([0-9.]+) pts', info)
    return float(m.group(1)) / 72.0, float(m.group(2)) / 72.0


def save(fig, path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*[Gg]lyph.*missing.*')
        fig.savefig(path, format='pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None})


# ----------------------------------------------------------------------------------------------------------- main
def main():
    S = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_STUDY
    if not (S / 'studies' / 'TB-S2' / 'primary' / 'summary.json').exists():
        raise SystemExit(f'study root not found: {S}')
    OUT.mkdir(parents=True, exist_ok=True)
    report = ['Version 2 figure check (written by tools/make_figures_v2.py)', '',
              f'Study root: {S}', 'Text font: Arial (TrueType); mathtext: Arial through the custom font set. '
              'All fonts embedded as Type 42 (CID TrueType).',
              f'Required: every text >= {MIN_PT} pt at printed size; Arial only.', '']
    failures = []
    with tempfile.TemporaryDirectory(prefix='v2-figure-fonts-') as tmp:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*[Gg]lyph.*missing.*')
            used = install_arial(Path(tmp))
            apply_style()
            report.insert(4, 'Font sources: ' + ', '.join(f'{k} <- {v}' for k, v in used.items()))
            builds = (('v2-geometry.pdf', lambda p: fig_geometry(S, p)),
                      ('v2-simulation.pdf', lambda p: fig_simulation(S, p)),
                      ('v2-measured.pdf', lambda p: fig_measured(S, p, ('a', 'd'))),
                      ('v2-measured-variants.pdf', lambda p: fig_measured(S, p, ('b', 'c'))),
                      ('v2-competitors.pdf', lambda p: fig_competitors(S, p)),
                      ('v2-miracle.pdf', lambda p: fig_miracle(S, p)),
                      ('v2-absolute.pdf', lambda p: fig_absolute(S, p)),
                      ('v2-order.pdf', lambda p: fig_order(S, p)[0]),
                      ('v2-identifiability.pdf', lambda p: fig_identifiability(S, p)))
            for name, build in builds:
                INSIDE.clear()
                path = OUT / name
                fig = build(path)
                issues, smallest, sizes = audit(fig)
                save(fig, path)
                plt.close(fig)
                w_in, h_in = pdf_page_in(path)
                tf = pdf_text_sizes(path)
                fonts, warn = pdf_fonts(path)
                bad_fonts = [f['name'] for f in fonts
                             if not re.search(r'(^|\+)Arial', f['name']) or f['emb'] != 'yes']
                type42 = all(f['type'] == 'CID TrueType' for f in fonts)
                ok = (not issues and min(tf) >= MIN_PT and smallest >= MIN_PT and not bad_fonts and type42
                      and not warn)
                if not ok:
                    failures.append(name)
                report += [name,
                           f'  page size: {w_in:.2f} x {h_in:.2f} in',
                           f'  smallest font size: {min(tf):g} pt (text operators in the PDF; sizes used: '
                           + ', '.join(f'{s:g}' for s in tf) + f' pt; figure text objects: {smallest:g} pt)',
                           '  embedded fonts (pdffonts: name, type, encoding, emb, sub, uni):']
                report += [f"    {f['name']:<34s} {f['type']:<14s} {f['encoding']:<11s} {f['emb']:<3s} "
                           f"{f['sub']:<3s} {f['uni']}" for f in fonts]
                if warn:
                    report.append(f'  pdffonts warnings: {warn}')
                report.append('  layout audit (text overlaps, text off the page, text over another plotting area, '
                              'in-panel labels or legends over data): '
                              + ('none' if not issues else f'{len(issues)} issue(s)'))
                report += [f'    {s}' for s in issues]
                report.append('  status: ' + ('pass' if ok else 'FAIL'))
                report.append('')
                print(f'{name}: {w_in:.2f} x {h_in:.2f} in, min {min(tf):g} pt, fonts '
                      + ', '.join(sorted({f["name"].split("+")[-1] for f in fonts}))
                      + (f', {len(issues)} layout issue(s)' if issues else '') + ('' if ok else '  FAIL'))
                for s in issues:
                    print('   ', s)
    (OUT / 'v2-figure-check.txt').write_text('\n'.join(report).rstrip() + '\n')
    if failures:
        raise SystemExit('checks failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
